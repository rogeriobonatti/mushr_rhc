import numpy as np


class GridParameters:
    x_min = -15.0
    x_max = 15.0
    x_step = 0.1

    y_min = -15.0
    y_max = 15.0
    y_step = 0.1

    z_min = -1.0
    z_max = 3.0

    # derived parameters
    Xn_f = float(x_max - x_min) / x_step
    Yn_f = float(y_max - y_min) / y_step
    Xn = int(Xn_f)
    Yn = int(Yn_f)

    def __init__(self):
        super(GridParameters, self).__init__()


class NetworkParameters:

    max_points_per_pillar = 100
    max_pillars = 12000
    nb_features = 7
    nb_channels = 64
    downscaling_factor = 2

    nb_dims = 3

    batch_size = 4
    total_training_epochs = 160
    iters_to_decay = 101040.    # 15 * 4 * ceil(6733. / 4) --> every 15 epochs on 6733 kitti samples, cf. pillar paper
    learning_rate = 2e-4
    decay_rate = 1e-8
    L1 = 0
    L2 = 0
    alpha = 0.25
    gamma = 2.0
                            # original pillars paper values
    focal_weight = 3.0      # 1.0
    loc_weight = 2.0        # 2.0
    size_weight = 2.0       # 2.0
    angle_weight = 1.0      # 2.0
    heading_weight = 0.2    # 0.2
    class_weight = 0.5      # 0.2

    def __init__(self):
        super(NetworkParameters, self).__init__()


class Parameters(GridParameters, NetworkParameters):

    def __init__(self):
        super(Parameters, self).__init__()
