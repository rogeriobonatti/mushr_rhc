
import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import Parameters

logger = logging.getLogger(__name__)

import numpy as np


class MiniPointNet(nn.Module):
    def __init__(self, nb_features, nb_channels, max_points):
        super().__init__()
        # assume size here is (N, 7, 60, 1200)
        self.conv1 = nn.Conv2d(nb_features, nb_channels, 1)
        # assume size here is (N, 64, 60, 1200)
        self.conv1_bn = nn.BatchNorm2d(nb_channels, eps=1e-3, momentum=0.01)
        self.conv1_relu = nn.ReLU()
        self.conv1_pool = nn.MaxPool2d(kernel_size=(max_points, 1))
        # assume size here is (N, 64, 1, 1200)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.conv1_relu(x)
        x = self.conv1_pool(x)
        return x


class PointPillarsScatter(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=64,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddle2K'):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size):

        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.nchannels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny,
                                         self.nx)
        return batch_canvas


class BackBoneEncoder(nn.Module):
    def __init__(self, nb_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, 1)
        self.conv1_bn = nn.BatchNorm2d(nb_channels, eps=1e-3, momentum=0.01)
        self.conv1_relu = nn.ReLU()
        self.conv1_pool = nn.MaxPool2d(kernel_size=(1, max_points))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.conv1_relu(x)
        x = self.conv1_pool(x)
        return x


class PillarNet(nn.Module):
    def __init__(self, max_pillars, max_points, nb_features, nb_channels, batch_size, image_size):
        self.mpn = MiniPointNet(nb_features, nb_channels, max_points)
        self.scatter = PointPillarsScatter()
        self.encoder = BackBoneEncoder()

    def forward(self, x):
        x = self.mpn(x)
        x = self.scatter(x)
        x = self.encoder(x)
        return x


def build_point_pillar_graph(params: Parameters):
    
    # extract required parameters
    max_pillars = int(params.max_pillars)
    max_points  = int(params.max_points_per_pillar)
    nb_features = int(params.nb_features)
    nb_channels = int(params.nb_channels)
    batch_size  = int(params.batch_size)
    image_size  = tuple([params.Xn, params.Yn])

    input_shape = (max_pillars, max_points, nb_features)

    net = PillarNet(max_pillars, max_points, nb_features, nb_channels, batch_size, image_size)
    # return net

    ############################

    input_pillars = tf.keras.layers.Input(input_shape, batch_size=batch_size, name="pillars/input")
    input_indices = tf.keras.layers.Input((max_pillars, 3), batch_size=batch_size, name="pillars/indices",
                                          dtype=tf.int32)

    def correct_batch_indices(tensor, batch_size):
        array = np.zeros((batch_size, max_pillars, 3), dtype=np.float32)
        for i in range(batch_size):
            array[i, :, 0] = i
        return tensor + tf.constant(array, dtype=tf.int32)

    if batch_size > 1:
            corrected_indices = tf.keras.layers.Lambda(lambda t: correct_batch_indices(t, batch_size))(input_indices)
    else:
        corrected_indices = input_indices

    # pillars
    x = tf.keras.layers.Conv2D(nb_channels, (1, 1), activation='linear', use_bias=False, name="pillars/conv2d")(input_pillars)
    x = tf.keras.layers.BatchNormalization(name="pillars/batchnorm", fused=True, epsilon=1e-3, momentum=0.99)(x)
    x = tf.keras.layers.Activation("relu", name="pillars/relu")(x)
    x = tf.keras.layers.MaxPool2D((1, max_points), name="pillars/maxpooling2d")(x)

    if tf.keras.backend.image_data_format() == "channels_first":
        reshape_shape = (nb_channels, max_pillars)
    else:
        reshape_shape = (max_pillars, nb_channels)

    x = tf.keras.layers.Reshape(reshape_shape, name="pillars/reshape")(x)
    pillars = tf.keras.layers.Lambda(lambda inp: tf.scatter_nd(inp[0], inp[1],
                                                               (batch_size,) + image_size + (nb_channels,)),
                                     name="pillars/scatter_nd")([corrected_indices, x])

    # 2d cnn backbone

    # Block1(S, 4, C)
    x = pillars
    for n in range(4):
        S = (2, 2) if n == 0 else (1, 1)
        x = tf.keras.layers.Conv2D(nb_channels, (3, 3), strides=S, padding="same", activation="relu",
                                   name="cnn/block1/conv2d%i" % n)(x)
        x = tf.keras.layers.BatchNormalization(name="cnn/block1/bn%i" % n, fused=True)(x)
    x1 = x

    # Block2(2S, 6, 2C)
    for n in range(6):
        S = (2, 2) if n == 0 else (1, 1)
        x = tf.keras.layers.Conv2D(2 * nb_channels, (3, 3), strides=S, padding="same", activation="relu",
                                   name="cnn/block2/conv2d%i" % n)(x)
        x = tf.keras.layers.BatchNormalization(name="cnn/block2/bn%i" % n, fused=True)(x)
    x2 = x

    # Block3(4S, 6, 4C)
    for n in range(6):
        S = (2, 2) if n == 0 else (1, 1)
        x = tf.keras.layers.Conv2D(2 * nb_channels, (3, 3), strides=S, padding="same", activation="relu",
                                   name="cnn/block3/conv2d%i" % n)(x)
        x = tf.keras.layers.BatchNormalization(name="cnn/block3/bn%i" % n, fused=True)(x)
    x3 = x

    # Up1 (S, S, 2C)
    up1 = tf.keras.layers.Conv2DTranspose(2 * nb_channels, (3, 3), strides=(1, 1), padding="same", activation="relu",
                                          name="cnn/up1/conv2dt")(x1)
    up1 = tf.keras.layers.BatchNormalization(name="cnn/up1/bn", fused=True)(up1)

    # Up2 (2S, S, 2C)
    up2 = tf.keras.layers.Conv2DTranspose(2 * nb_channels, (3, 3), strides=(2, 2), padding="same", activation="relu",
                                          name="cnn/up2/conv2dt")(x2)
    up2 = tf.keras.layers.BatchNormalization(name="cnn/up2/bn", fused=True)(up2)

    # Up3 (4S, S, 2C)
    up3 = tf.keras.layers.Conv2DTranspose(2 * nb_channels, (3, 3), strides=(4, 4), padding="same", activation="relu",
                                          name="cnn/up3/conv2dt")(x3)
    up3 = tf.keras.layers.BatchNormalization(name="cnn/up3/bn", fused=True)(up3)

    # Concat
    concat = tf.keras.layers.Concatenate(name="cnn/concatenate")([up1, up2, up3])

    # Detection head
    occ = tf.keras.layers.Conv2D(nb_anchors, (1, 1), name="occupancy/conv2d", activation="sigmoid")(concat)

    loc = tf.keras.layers.Conv2D(nb_anchors * 3, (1, 1), name="loc/conv2d", kernel_initializer=tf.keras.initializers.TruncatedNormal(0, 0.001))(concat)
    loc = tf.keras.layers.Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="loc/reshape")(loc)

    size = tf.keras.layers.Conv2D(nb_anchors * 3, (1, 1), name="size/conv2d", kernel_initializer=tf.keras.initializers.TruncatedNormal(0, 0.001))(concat)
    size = tf.keras.layers.Reshape(tuple(i//2 for i in image_size) + (nb_anchors, 3), name="size/reshape")(size)

    angle = tf.keras.layers.Conv2D(nb_anchors, (1, 1), name="angle/conv2d")(concat)

    heading = tf.keras.layers.Conv2D(nb_anchors, (1, 1), name="heading/conv2d", activation="sigmoid")(concat)

    clf = tf.keras.layers.Conv2D(nb_anchors * nb_classes, (1, 1), name="clf/conv2d")(concat)
    clf = tf.keras.layers.Reshape(tuple(i // 2 for i in image_size) + (nb_anchors, nb_classes), name="clf/reshape")(clf)

    pillar_net = tf.keras.models.Model([input_pillars, input_indices], [occ, loc, size, angle, heading, clf])
#     print(pillar_net.summary())

    return pillar_net
