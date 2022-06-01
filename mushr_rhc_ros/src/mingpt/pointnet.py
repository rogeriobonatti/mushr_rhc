import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

class PCL_encoder(nn.Module):
    def __init__(self, feat_size):
        """
        data_len (int)      : 2 for X and Y, 3 to include polarity as well
        latent_size (int)            : Size of latent vector
        tc (bool)           : True if temporal coding is included
        params (list)       : Currently just the image resolution (H, W)
        """
        super(PCL_encoder, self).__init__()

        # input is tensor of size [B, transformer_hist_length, num_points(720 max), 2(XY)]
        # output is of size [B, transformer_hist_length, feat_size]
        self.feat_size = feat_size

        self.featnet = nn.Sequential(
            nn.Conv1d(2, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.encoder = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, feat_size)
        )

        self.weight_init()

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def weight_init(self):
        self.featnet.apply(self.kaiming_init)
        self.encoder.apply(self.kaiming_init)

    def forward(self, x, times=None):
        # ECN computes per-event spatial features
        x = self.featnet(x)

        # Symmetric function to reduce N features to 1 a la PointNet
        x, _ = torch.max(x, 2, keepdim=True)
        x = x.view(-1, 1024)

        # Compress to latent space
        embedding = self.encoder(x)

        return embedding