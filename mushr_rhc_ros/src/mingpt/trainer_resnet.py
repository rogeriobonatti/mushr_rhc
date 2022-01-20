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
import time

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
import matplotlib.pyplot as plt

class TrainerResnetConfig:
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

def denorm_angle(angle):
    # normalize all actions
    act_max = 0.38
    act_min = -0.38
    return 0.5*(angle*(act_max-act_min)+act_max+act_min)

class TrainerResnet:

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

    def train(self, experiment=None, SW=None):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        print('==========trianing==========')
        def run_epoch(split, epoch_num=0,):
            is_train = split == 'train'
            model.train(is_train)
            loader = self.train_dataset if is_train else self.test_dataset
            #print('len', len(loader))
            
            losses = AverageMeter('Loss', ':.4e')
            start = time.time()

            for idx, used_data in enumerate(loader):

                end = time.time()
                # print(end - start)
                start = time.time()

                imgs = used_data['img'] # B, N, C, H, W
                imgs = imgs.to(self.device)

                if config.flatten_img:
                    B, N, C = imgs.shape
                else:
                    B, N, C, H, W = imgs.shape
                    # input_img = input_img.permute(0, 2, 1, 3, 4).contiguous().view(B, N, C*H*W)
                    x_imgs = imgs[:,:N-1,:,:,:].view(B, C*(N-1), H, W)
                    y_imgs = imgs[:,N-1,:,:,:].view(B, 1, H, W)
                x_imgs = x_imgs.to(self.device)
                y_imgs = y_imgs.to(self.device)

                #========== process velocity command============
                if config.loss == 'cross_entropy':
                    y = used_data['label'].to(self.device) # B, N, C, H, W
                    y = y.view(B, N, 1)
                elif config.loss == 'MSE':
                    acts = used_data['act'].to(self.device)
                    B, N = acts.shape
                    x_act = acts[:,:N-1].view(B, N-1)
                    y_act = acts[:,N-1].view(B, 1)

                # ========== generate timesteps t ===========
                # t = np.ones((B, 1, 1), dtype=int) * 7
                # t = torch.tensor(t)
                # t = t.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    action_preds, loss = model(x_imgs, x_act, y_imgs, y_act)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.update(loss.item(), 1)
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
                SW.add_scalar('loss/train', losses.avg, epoch_num)
                # experiment.log_metric('train/loss', losses.avg, epoch_num)

            if not is_train:
                test_loss = losses.avg
                SW.add_scalar('loss/test', test_loss, epoch_num)
                # experiment.log_metric('val/loss', losses.avg, epoch_num)
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
            

class DeployerResnet:

    def __init__(self, model, device, test_dataset):
        self.model = model
        self.test_dataset = test_dataset
        self.device = device

    def deploy_hist(self):
        idx_save = 0
        model= self.model
        # gt_actions = np.zeros(shape=(0,))
        # pred_actions = np.zeros(shape=(0,))
        gt_actions = []
        pred_actions = []
        count=0
        l = len(self.test_dataset)
        for idx, used_data in enumerate(self.test_dataset):
            imgs = used_data['img'] # B, N, C, H, W
            imgs = imgs.to(self.device)
            B, N, C, H, W = imgs.shape
            count+= B
            print('Completed: {:.2f}%'.format(count/l))
            x_imgs = imgs[:,:N-1,:,:,:].view(B, C*(N-1), H, W)
            y_imgs = imgs[:,N-1,:,:,:].view(B, 1, H, W)
            x_imgs = x_imgs.to(self.device)
            y_imgs = y_imgs.to(self.device)
            acts = used_data['act'].to(self.device)
            B, N = acts.shape
            x_act = acts[:,:N-1].view(B, N-1)
            y_act = acts[:,N-1].view(B, 1)
            with torch.set_grad_enabled(False):
                action_preds, loss = model(x_imgs, x_act, y_imgs, y_act)
                # gt_actions = np.append(gt_actions, y_act.cpu().numpy().flatten())
                # pred_actions = np.append(pred_actions, action_preds.cpu().numpy().flatten())
                gt_actions.extend(y_act.cpu().flatten().tolist())
                pred_actions.extend(action_preds.cpu().flatten().tolist())
        
        # create histogram of errors
        gt_actions = np.array(gt_actions)
        pred_actions = np.array(pred_actions)
        errors = np.absolute(pred_actions-gt_actions)
        fig, axs = plt.subplots(1, 2, tight_layout=True)
        axs[0].hist(gt_actions)
        axs[1].hist(errors)
        axs[0].set_title('GT actions')
        axs[1].set_title('Error distribution')
        plt.show()
        plt.clf()

        unique_angles = np.unique(gt_actions)
        angle_avg_errors = np.zeros(shape=unique_angles.shape)
        for idx_angle, angle in enumerate(unique_angles):
            angle_avg_errors[idx_angle] = np.mean(errors[gt_actions==angle])
        plt.bar(x=unique_angles, height=angle_avg_errors, width=0.02)
        plt.title("Mean error for each action angle")
        plt.show()

 
    def deploy(self):
        out_path = '/home/azureuser/hackathon_data/model_eval'
        out_path = os.path.join(out_path, str(time.time()))
        if not os.path.exists(out_path): 
            os.makedirs(out_path)
        idx_save = 0

        model= self.model
        for idx, used_data in enumerate(self.test_dataset):
                imgs = used_data['img'] # B, N, C, H, W
                imgs = imgs.to(self.device)
                B, N, C, H, W = imgs.shape
                x_imgs = imgs[:,:N-1,:,:,:].view(B, C*(N-1), H, W)
                y_imgs = imgs[:,N-1,:,:,:].view(B, 1, H, W)
                x_imgs = x_imgs.to(self.device)
                y_imgs = y_imgs.to(self.device)

                acts = used_data['act'].to(self.device)
                B, N = acts.shape
                x_act = acts[:,:N-1].view(B, N-1)
                y_act = acts[:,N-1].view(B, 1)

                rows = 5
                columns = N

                with torch.set_grad_enabled(False):
                    action_preds, loss = model(x_imgs, x_act, y_imgs, y_act)
                    
                    # save the first m_frames
                    for k in range(B):
                        plt.imshow(imgs[k,N-1,0,:,:].cpu())
                        img_path = os.path.join(out_path, str(idx_save)+'.png')
                        # plot the GT and predicted directions
                        l = 50
                        angle_orig_gt = y_act[k,0].cpu().item()
                        angle = angle_orig_gt*-1.0-np.pi/2.0
                        x0 = y0 = 100
                        dx = l*np.cos(angle)
                        dy = l*np.sin(angle)
                        plt.arrow(x0,y0,dx,dy, color='r')
                        angle_orig_pred = action_preds[k,0].cpu().item()
                        angle = angle_orig_pred*-1.0-np.pi/2.0
                        x0 = y0 = 100
                        dx = l*np.cos(angle)
                        dy = l*np.sin(angle)
                        plt.arrow(x0,y0,dx,dy, color='g')
                        gt = np.rad2deg(denorm_angle(angle_orig_gt))
                        pred = np.rad2deg(denorm_angle(angle_orig_pred))
                        plt.title("GT(r): {:.2f} | Pred(g): {:.2f}".format(gt, pred))
                        plt.savefig(img_path)
                        plt.clf()
                        idx_save += 1
                        if idx_save%500==0:
                            print("Printed {} images".format(idx_save))


                    # print the sequence of size rows
                    # fig = plt.figure(figsize=(20, 20))
                    # for k in range(rows):
                    #     # for i in range(1, N+1):
                    #     for i in range(k*N+1, k*N+1+N):
                    #         fig.add_subplot(rows, columns, i)
                    #         plt.imshow(imgs[k,i%N-1,0,:,:].cpu())
                    #         plt.axis('off')
                    #         if i%N==0:
                    #             # plot the GT and predicted directions
                    #             l = 50
                    #             angle_orig_gt = y_act[k,0].cpu().item()
                    #             angle = angle_orig_gt*-1.0-np.pi/2.0
                    #             x0 = y0 = 100
                    #             dx = l*np.cos(angle)
                    #             dy = l*np.sin(angle)
                    #             plt.arrow(x0,y0,dx,dy, color='r')
                    #             angle_orig_pred = action_preds[k,0].cpu().item()
                    #             angle = angle_orig_pred*-1.0-np.pi/2.0
                    #             x0 = y0 = 100
                    #             dx = l*np.cos(angle)
                    #             dy = l*np.sin(angle)
                    #             plt.arrow(x0,y0,dx,dy, color='g')
                    #             gt = np.rad2deg(denorm_angle(angle_orig_gt))
                    #             pred = np.rad2deg(denorm_angle(angle_orig_pred))
                    #             plt.title("GT(r): {:.2f} | Pred(g): {:.2f}".format(gt, pred))
                    #     # plt.show()
                    #     # fig.savefig(os.path.join('/home/rb/Pictures', 'reconstruction_results.png'))
                    # plt.show()


        
            

class Args:
    def __init__(self, game, seed):
        self.device = torch.device('cuda')
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4
