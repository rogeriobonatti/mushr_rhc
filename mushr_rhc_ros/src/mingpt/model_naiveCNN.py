"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
from resnet_custom import resnet18_custom

logger = logging.getLogger(__name__)

import numpy as np

import sys
sys.path.append('../')
from models.compass.select_backbone import select_resnet


class NaiveCNN(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, device):
        super().__init__()

        self.device = device

        self.resnet = resnet18_custom(pretrained=False)
        self.state_encoder = nn.Sequential(nn.Linear(1000, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32,1))

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        criterion = torch.nn.MSELoss(reduction='mean')
        self.criterion = criterion.cuda(device)


    def configure_optimizers(self, train_config):
        optimizer = torch.optim.Adam(self.parameters(), train_config.learning_rate)
        return optimizer

    # state, and action
    def forward(self, states, actions, targets=None, timesteps=None):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # timesteps: (batch, 1, 1) 

        action_preds = self.state_encoder(self.resnet(states))

        loss = None
        if targets is not None:
            loss = self.criterion(actions, action_preds) 
            
        return action_preds, loss
