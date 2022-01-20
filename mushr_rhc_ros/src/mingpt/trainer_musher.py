"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

from mingpt.utils import sample
#import atari_py
from collections import deque
import random
import cv2
import torch
from PIL import Image
import os
from utils.meters import AverageMeter
from utils.dataset_utils import de_normalize_v, remove_prefix

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, max_timestep, **kwargs):
        self.max_timestep = max_timestep 
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, device, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = device
 
    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        # torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self, experiment):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        print('==========trianing==========')
        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            loader = self.train_dataset if is_train else self.test_dataset
            #print('len', len(loader))
            
            losses = AverageMeter('Loss', ':.4e')

            for idx, used_data in enumerate(loader):

                input_img = used_data['img'] # B, C, N, H, W
                input_img = input_img.to(self.device)

                if config.flatten_img:
                    B, N, C = input_img.shape
                else:
                    B, C, N, H, W = input_img.shape
                    input_img = input_img.permute(0, 2, 1, 3, 4).contiguous().view(B, N, C*H*W)
                x = input_img 
                x = x.to(self.device)

                #========== process velocity command============
                if config.loss == 'cross_entropy':
                    y = used_data['label'].to(self.device)
                    y = y.view(B, N, 1)
                elif config.loss == 'MSE':
                    y = used_data['act'].to(self.device)
                    y = y.view(B, N , 1)
                

                # ========== generate timesteps t ===========
                t = np.ones((B, 1, 1), dtype=int) * 7
                t = torch.tensor(t)
                t = t.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    action_preds, loss = model(x, y, y, t)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.update(loss.item(), B)
                    #print('--------- loss', loss.item())

                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
    
            if is_train:
                print('ep%d: training loss: %4f' %(epoch_num, losses.avg))            
                experiment.log_metric('train/loss', losses.avg, epoch_num) 


            if not is_train:
                test_loss = losses.avg
                experiment.log_metric('val/loss', losses.avg, epoch_num)
                print('ep%d: testing loss: %4f'%(epoch_num, test_loss))
                return test_loss

        # best_loss = float('inf')
        
        self.tokens = 0 # counter used for learning rate decay

        
        for epoch in range(config.max_epochs):

            run_epoch('train', epoch_num=epoch)
            run_epoch('test', epoch_num=epoch)

            save_dict = {'epoch': epoch,
                         'state_dict': model.state_dict()}
            if epoch % config.save_freq == 0:
                filename = os.path.join(config.model_path, 'epoch%s.pth.tar' %str(epoch))
                torch.save(save_dict, filename)
                print('save checkpoint to', config.model_path)
            
            
            
            

class Args:
    def __init__(self, game, seed):
        self.device = torch.device('cuda')
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4
