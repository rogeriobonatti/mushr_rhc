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
from resnet_custom import resnet18_custom, resnet50_custom

logger = logging.getLogger(__name__)

import numpy as np

import sys
sys.path.append('../')
# from models.compass.select_backbone import select_resnet

class ResnetDirect(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, device, clip_len, restype):
        super().__init__()

        self.device = device

        if restype=='resnet18':
            self.resnet = resnet18_custom(pretrained=False, clip_len=clip_len+1)
        elif restype=='resnet50':
            self.resnet = resnet50_custom(pretrained=False, clip_len=clip_len+1)

        self.state_encoder = nn.Sequential(nn.Linear(1000, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32,1))

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        criterion = torch.nn.MSELoss(reduction='mean')
        self.criterion = criterion.cuda(device)


    def configure_optimizers(self, train_config):
        optimizer = torch.optim.Adam(self.parameters(), train_config.learning_rate)
        return optimizer

    # state, and action
    def forward(self, x_imgs, x_act, y_imgs=None, y_act=None):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # timesteps: (batch, 1, 1) 

        # print("batch forward")
        input = torch.cat((x_imgs, y_imgs), dim=1)
        action_preds = self.state_encoder(self.resnet(input))

        loss = None
        if y_act is not None:
            loss = self.criterion(y_act, action_preds)
            
        return action_preds, loss

class ResnetDirectWithActions(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, device, clip_len, restype):
        super().__init__()

        self.device = device

        if restype=='resnet18':
            self.resnet = resnet18_custom(pretrained=False, clip_len=clip_len+1)
        elif restype=='resnet50':
            self.resnet = resnet50_custom(pretrained=False, clip_len=clip_len+1)

        self.actions_encoder = nn.Sequential(nn.Linear(clip_len, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU())

        self.state_encoder = nn.Sequential(nn.Linear(1000+128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32,1))

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        criterion = torch.nn.MSELoss(reduction='mean')
        self.criterion = criterion.cuda(device)


    def configure_optimizers(self, train_config):
        optimizer = torch.optim.Adam(self.parameters(), train_config.learning_rate)
        return optimizer

    # state, and action
    def forward(self, x_imgs, x_act, y_imgs=None, y_act=None):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # timesteps: (batch, 1, 1) 

        # print("batch forward")
        input = torch.cat((x_imgs, y_imgs), dim=1)
        action_preds = self.state_encoder(torch.cat((self.resnet(input), self.actions_encoder(x_act)), dim=1))

        loss = None
        if y_act is not None:
            loss = self.criterion(y_act, action_preds)
            
        return action_preds, loss
