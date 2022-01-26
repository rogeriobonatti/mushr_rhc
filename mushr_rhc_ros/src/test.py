# from torchsummary import summary
import sys
import os
import signal
import threading
import random
import numpy as np
from queue import Queue
import time
from collections import OrderedDict

import torch
from mingpt.model_resnetdirect import ResnetDirect, ResnetDirectWithActions
# from mingpt.model_musher import GPT, GPTConfig
from mingpt.model_mushr_rogerio import GPT, GPTConfig

import torch_tensorrt
# import tensorrt as trt

# TRT_LOGGER = trt.Logger()
# builder = trt.Builder(TRT_LOGGER)
# network = builder.create_network()


default_speed = 1.0
# self.default_speed = 1.5
default_angle = 0.0
nx = None
ny = None
        
# network loading
print("Starting to load model")
os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
device = torch.device('cuda')
        
clip_len = 16
# saved_model_path = '/home/rb/downloaded_models/epoch30.pth.tar'
saved_model_path = '/home/robot/weight_files/epoch15.pth.tar'
# saved_model_path = '/home/rb/hackathon_data/aml_outputs/log_output/gpt_resnet18_0/GPTgpt_resnet18_4gpu_2022-01-24_1642987604.6403077_2022-01-24_1642987604.640322/model/epoch15.pth.tar'
# saved_model_path = '/home/rb/hackathon_data/aml_outputs/log_output/gpt_resnet18_8_exp2/GPTgpt_resnet18_8gpu_exp2_2022-01-25_1643076745.003202_2022-01-25_1643076745.0032148/model/epoch12.pth.tar'
vocab_size = 100
block_size = clip_len * 2
max_timestep = 7
# mconf = GPTConfig(vocab_size, block_size, max_timestep,
#               n_layer=6, n_head=8, n_embd=128, model_type='GPT', use_pred_state=True,
#               state_tokenizer='conv2D', train_mode='e2e', pretrained_model_path='')
mconf = GPTConfig(vocab_size, block_size, max_timestep,
              n_layer=6, n_head=8, n_embd=128, model_type='GPT', use_pred_state=True,
              state_tokenizer='resnet18', train_mode='e2e', pretrained_model_path='', pretrained_encoder_path='', loss='MSE')              
model = GPT(mconf, device)
# model=torch.nn.DataParallel(model)

checkpoint = torch.load(saved_model_path)
new_checkpoint = OrderedDict()
for key in checkpoint['state_dict'].keys():
    new_checkpoint[key.split("module.",1)[1]] = checkpoint['state_dict'][key]

model.load_state_dict(new_checkpoint)
model.eval()
model.to(device)
model.half()




input=(torch.zeros(1,clip_len,200*200, dtype=torch.float32).to(device),
       torch.zeros(1,clip_len,1, dtype=torch.float32).to(device),
       torch.zeros(1,clip_len,1, dtype=torch.float32).to(device),
       torch.zeros(1,1,1, dtype=torch.float32).to(device))
input_names=['states', 'actions', 'targets', 'timesteps']
ONNX_FILE_PATH = '/home/robot/weight_files/test.onnx'
# torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=input_names,
#                   output_names=['output'], export_params=True)
torch.onnx.export(model, input, ONNX_FILE_PATH)

inputs = [torch_tensorrt.Input((1, clip_len, 200*200),dtype=torch.half),
          torch_tensorrt.Input((1, clip_len, 1),dtype=torch.half),
          torch_tensorrt.Input((1, clip_len, 1),dtype=torch.half),
          torch_tensorrt.Input((1, 1, 1),dtype=torch.half)
         ]

enabled_precisions = {torch.float, torch.half}
trt_ts_module = trt.compile(model, inputs=inputs, enabled_precisions=enabled_precisions)

model.to(device)
model = model
device = device
print("Finished loading model")