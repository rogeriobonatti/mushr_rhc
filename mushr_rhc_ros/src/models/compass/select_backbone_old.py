"""
Modified from https://github.com/TengdaHan/MemDPC/blob/master/backbone/select_backbone.py
"""
from .resnet_2d3d import * 

def select_resnet(network):
    if network == 'resnet18':
        model_img = resnet18_2d3d_full(track_running_stats=True)
        model_seg = resnet18_2d3d_full_C1(track_running_stats=True)
        model_depth = resnet18_2d3d_full_C1(track_running_stats=True)
        model_flow = resnet18_2d3d_full_C2(track_running_stats=True) 
        param = {'feature_size': 256}
    else: 
        raise NotImplementedError

    return model_img, model_seg, model_depth, model_flow,  param