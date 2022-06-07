#!/usr/bin/env python3

# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

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
import math
import copy

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import ColorRGBA, Empty, String
from std_srvs.srv import Empty as SrvEmpty
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan

import logger
import parameters
import rhcbase
import rhctensor
import utilss
import librhc.utils as utils_other

import torch
from mingpt.model_resnetdirect import ResnetDirect, ResnetDirectWithActions
# from mingpt.model_musher import GPT, GPTConfig
# from mingpt.model_mushr_rogerio import GPT, GPTConfig
# from mingpt.model_mushr_nips import GPT, GPTConfig
# from mingpt.model_mushr_new import GPT, GPTdiff, GPTConfig
from mingpt.model_mushr_new2 import GPT, GPTConfig
import preprocessing_utils as pre
from visualization_msgs.msg import Marker

# import torch_tensorrt


class RHCNode(rhcbase.RHCBase):
    def __init__(self, dtype, params, logger, name):
        rospy.init_node(name, anonymous=True, disable_signals=True)

        super(RHCNode, self).__init__(dtype, params, logger)
        
        self.small_queue_lock = threading.Lock() # for mapping and action prediction
        self.large_queue_lock = threading.Lock() # for localization
        self.pos_lock = threading.Lock() # for storing the vehicle pose

        self.curr_pose = None

        self.reset_lock = threading.Lock()
        self.inferred_pose_lock = threading.Lock()
        self.inferred_pose_lock_prev = threading.Lock()
        self._inferred_pose = None
        self._inferred_pose_prev = None
        self._time_of_inferred_pose = None
        self._time_of_inferred_pose_prev = None

        self.hp_zerocost_ids = None
        self.hp_map = None
        self.hp_world = None
        self.time_started_goal = None
        self.num_trials = 0

        self.act_inference_time_sum = 0.0
        self.act_inference_time_count = 0

        self.cur_rollout = self.cur_rollout_ip = None
        self.traj_pub_lock = threading.Lock()

        self.goal_event = threading.Event()
        self.map_metadata_event = threading.Event()
        self.ready_event = threading.Event()
        self.events = [self.goal_event, self.map_metadata_event, self.ready_event]
        self.run = True

        self.default_speed = 2.5
        # self.default_speed = 1.0
        self.default_angle = 0.0
        self.nx = None
        self.ny = None
        
        self.points_viz_list = None
        self.map_recon = None
        self.loc_counter = 0
        self.time_sent_reset = None
        self.current_frame = None
        
        # network loading
        print("Starting to load model")
        os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
        device = torch.device('cuda')
        # device = "cpu"

        self.device = device
        self.clip_len = 16

        self.is_real_deployment = rospy.get_param("~is_real_deployment", False)

        self.map_type = rospy.get_param("~deployment_map", 'train')

        self.use_map = rospy.get_param("~use_map", False)
        self.use_loc = rospy.get_param("~use_loc", False)

        saved_model_path_action = rospy.get_param("~model_path_act", '')
        self.out_path = rospy.get_param("~out_path", 'default_value')

        self.n_layers = rospy.get_param("~n_layers", 12)

        vocab_size = 100
        block_size = self.clip_len * 2
        max_timestep = self.clip_len

        mconf = GPTConfig(block_size, max_timestep,
                      n_layer=self.n_layers, n_head=8, n_embd=128, model_type='GPT', use_pred_state=True,
                      state_tokenizer='pointnet', pretrained_encoder_path='',
                      loss='MSE', train_mode='e2e', pretrained_model_path='',
                      map_decoder='deconv', map_recon_dim=64, freeze_core=False,
                      state_loss_weight=0.1,
                      loc_x_loss_weight=0.01, loc_y_loss_weight=0.1, loc_angle_loss_weight=10.0,
                      loc_decoder_type='joint')
        model = GPT(mconf, device)
        # model=torch.nn.DataParallel(model)

        if len(saved_model_path_action)>3: # some small number, path must have more
            checkpoint = torch.load(saved_model_path_action, map_location=device)
            # old code for loading model
            model.load_state_dict(checkpoint['state_dict'])
            # new code for loading mode
            # new_checkpoint = OrderedDict()
            # for key in checkpoint['state_dict'].keys():
            #     new_checkpoint[key.split("module.",1)[1]] = checkpoint['state_dict'][key]
            # model.load_state_dict(new_checkpoint)

            # ckpt = torch.load('/home/rb/downloaded_models/epoch30.pth.tar')['state_dict']
            # for key in ckpt:
            #     print('********',key)
            # model.load_state_dict(torch.load('/home/rb/downloaded_models/epoch30.pth.tar')['state_dict'], strict=True)

        model.eval()
        # model.half()
        model.to(device)
        
        # inputs = [torch_tensorrt.Input(
        #     states_shape=[1, self.clip_len, 200*200],
        #     actions_shape=[1, self.clip_len , 1],
        #     targets_shape=[1, self.clip_len , 1],
        #     timesteps_shape=[1, 1, 1],
        #     dtype=torch.half,
        # )]
        # enabled_precisions = {torch.float, torch.half}
        # trt_ts_module = torch_tensorrt.compile(model, inputs=inputs, enabled_precisions=enabled_precisions)

        self.model = model
        print("Finished loading model")

        # mapping model
        if self.use_map:
            
            saved_map_model_path = rospy.get_param("~model_path_map", '')

            mconf_map = GPTConfig(block_size, max_timestep,
                      n_layer=self.n_layers, n_head=8, n_embd=128, model_type='GPT', use_pred_state=True,
                      state_tokenizer='pointnet', pretrained_encoder_path='',
                      loss='MSE', train_mode='map', pretrained_model_path='',
                      map_decoder='deconv', map_recon_dim=64, freeze_core=False,
                      state_loss_weight=0.1,
                      loc_x_loss_weight=0.01, loc_y_loss_weight=0.1, loc_angle_loss_weight=10.0,
                      loc_decoder_type='joint')
            map_model = GPT(mconf_map, device)
            # map_model=torch.nn.DataParallel(map_model)

            checkpoint = torch.load(saved_map_model_path, map_location=device)

            # old code for loading model
            map_model.load_state_dict(checkpoint['state_dict'])
            # new code for loading mode
            # new_checkpoint = OrderedDict()
            # for key in checkpoint['state_dict'].keys():
            #     new_checkpoint[key.split("module.",1)[1]] = checkpoint['state_dict'][key]
            # map_model.load_state_dict(new_checkpoint)
            
            map_model.eval()
            map_model.to(device)
            self.map_model = map_model
            rate_map_display = 1.0
            

        # localization model
        if self.use_loc:
            
            saved_loc_model_path = rospy.get_param("~model_path_loc", '')
            
            mconf_loc = GPTConfig(block_size, max_timestep,
                      n_layer=self.n_layers, n_head=8, n_embd=128, model_type='GPT', use_pred_state=True,
                      state_tokenizer='pointnet', pretrained_encoder_path='',
                      loss='MSE', train_mode='loc', pretrained_model_path='',
                      map_decoder='deconv', map_recon_dim=64, freeze_core=False,
                      state_loss_weight=0.1,
                      loc_x_loss_weight=0.01, loc_y_loss_weight=0.1, loc_angle_loss_weight=10.0,
                      loc_decoder_type='joint')
            loc_model = GPT(mconf_loc, device)
            # map_model=torch.nn.DataParallel(map_model)

            checkpoint = torch.load(saved_loc_model_path, map_location=device)

            # old code for loading model
            loc_model.load_state_dict(checkpoint['state_dict'])
            # new code for loading mode
            # new_checkpoint = OrderedDict()
            # for key in checkpoint['state_dict'].keys():
            #     new_checkpoint[key.split("module.",1)[1]] = checkpoint['state_dict'][key]
            # loc_model.load_state_dict(new_checkpoint)
            
            loc_model.eval()
            loc_model.to(device)
            self.loc_model = loc_model
            rate_loc_display = 20
            


        self.small_queue = Queue(maxsize = self.clip_len) # stores current scan, action, pose. up to 16 elements
        self.large_queue = Queue() # stores current scan, action, pose. no limit of elements

        self.last_action = self.default_angle
        self.new_scan_arrived = False
        self.compute_network_action = False
        self.compute_network_loc = False
        self.has_loc_anchor = False
        self.did_reset = False

        # parameters for model evaluation
        self.reset_counter = 0
        self.last_reset_time = time.time()
        self.distance_so_far = 0.0
        self.time_so_far = 0.0
        self.file_name = os.path.join(self.out_path,'info.csv')

        # define timer callbacks:
        if self.use_map:
            self.map_viz_loc = rospy.Timer(rospy.Duration(1.0 / rate_loc_display), self.loc_viz_cb)
        if self.use_loc:
            self.map_viz_timer = rospy.Timer(rospy.Duration(1.0 / rate_map_display), self.map_viz_cb)

        
        
        


    def start(self):
        self.logger.info("Starting RHController")
        self.setup_pub_sub()
        self.rhctrl = self.load_controller()
        self.find_allowable_pts() # gets the allowed halton points from the map

        self.ready_event.set()

        rate_hz = 50
        rate = rospy.Rate(rate_hz)
        self.logger.info("Initialized")

        # set initial pose for the car in the very first time in an allowable region
        if self.map_type == 'train':
                self.send_initial_pose()
        else:
            self.send_initial_pose_12f()
        
        self.time_started = rospy.Time.now()

        # wait until we actually have a car pose
        rospy.loginfo("Waiting to receive pose")
        while not rospy.is_shutdown() and self.inferred_pose is None:
            pass
        rospy.loginfo("Vehicle pose received")
 
        while not rospy.is_shutdown() and self.run:

            # check if we should reset the vehicle if crashed
            if not self.is_real_deployment:
                if self.check_reset(rate_hz):
                    rospy.loginfo("Resetting the car's position")

            # publish next action
            if self.compute_network_action:
                # don't have to run the network at all times, only when scans change and scans are full
                self.last_action = self.apply_network()
                # rospy.loginfo("Applied network: "+str(self.last_action))
                self.compute_network_action = False
                self.publish_vel_marker()
            
            self.publish_traj(self.default_speed, self.last_action)
            
            # if map is not None:
            rate.sleep()

    def map_viz_cb(self, timer):

        # find the middle pose for plotting reference
        self.small_queue_lock.acquire()
        pos_queue_list = list(self.small_queue.queue)
        self.small_queue_lock.release()
        pos_size = len(pos_queue_list)
        
        if pos_size==16:
            pose_mid = pos_queue_list[int(pos_size/2) -1][2]
            # if not self.is_real_deployment:
            #     pose_mid = pos_queue_list[int(pos_size/2) -1][2]
            # else:
            #     pose_mid = PoseStamped()
            x_imgs, x_act, t = self.prepare_model_inputs(queue_type='small')
            start = time.time()
            # with torch.set_grad_enabled(False):
            with torch.inference_mode():
                self.map_recon, _ = self.map_model(states=x_imgs, actions=x_act, targets=x_act, gt_map=None, timesteps=t, poses=None, compute_loss=False)
            finished_map_network = time.time()
            rospy.loginfo("map network delay: "+str(finished_map_network-start))
            
            # publish the GT pose of the map center
            self.pose_marker_pub.publish(self.create_position_marker(pose_mid))
            # publish the map itself
            self.map_marker_pub.publish(self.create_map_marker(pose_mid))
        
    def loc_viz_cb(self, timer):


        if self.compute_network_loc is False or self.time_sent_reset is None:
            return

        # create anchor pose for localization
        if time.time()-self.time_sent_reset>3.0 and self.did_reset is True:
            self.loc_counter = 0
            self.has_loc_anchor = False
            self.large_queue_lock.acquire()
            self.large_queue = Queue() # reset the queue as well, we can't mix unfinished elements from previous run
            self.large_queue_lock.release()
            self.did_reset = False
            rospy.logwarn("Resetting the loc position")

        self.large_queue_lock.acquire()
        large_queue_list = copy.deepcopy(list(self.large_queue.queue))
        self.large_queue_lock.release()
        large_queue_len = len(large_queue_list)
        
        # set the anchor and equal to the first reference when we count 16 scans after reset
        if large_queue_len>=self.clip_len and self.has_loc_anchor is False:
            rospy.logwarn("Setting the loc anchor position when completed 16 scans")
            self.pose_anchor = copy.deepcopy(self.curr_pose)
            self.current_frame = copy.deepcopy(self.pose_anchor)
            self.has_loc_anchor = True

        if large_queue_len>=self.clip_len and self.has_loc_anchor is True and self.compute_network_loc is True:
            x_imgs, x_act, t = self.prepare_model_inputs(queue_type='large')
            # remove the oldest element from the large queue
            self.large_queue_lock.acquire()
            self.large_queue.get()
            self.large_queue_lock.release()
            
            start = time.time()
            # with torch.set_grad_enabled(False):
            with torch.inference_mode():
                pose_preds, _, _, _, _ = self.loc_model(states=x_imgs, actions=x_act, targets=x_act, gt_map=None, timesteps=t, poses=None, compute_loss=False)
            finished_loc_network = time.time()
            rospy.loginfo("loc network delay: "+str(finished_loc_network-start))
            # publish anchor pose of the map center
            self.loc_anchor_pose_marker_pub.publish(self.create_position_marker(self.pose_anchor, color=[0,0,1,1]))
            # publish the current accumulated pose
            # delta_pose_pred = pose_preds[0,self.clip_len-1,:].cpu().numpy()
            delta_pose_pred = pose_preds[0,-1,:].cpu().numpy()
            # calculate the change in coordinates
            self.current_frame = self.transform_poses(self.current_frame, delta_pose_pred)
            self.loc_current_pose_marker_pub.publish(self.create_position_marker(self.current_frame, color=[1,0,0,1]))


    def transform_poses(self, current_pose, delta_pose_pred):
        # elements of homogeneous matrix expressing point from local frame into world frame coords
        current_angle = utilss.rosquaternion_to_angle(current_pose.pose.orientation)
        R = np.array([[np.cos(current_angle),-np.sin(current_angle)],
                      [np.sin(current_angle),np.cos(current_angle)]])
        t = np.array([[current_pose.pose.position.x],
                      [current_pose.pose.position.y]])
        T = np.array([[R[0,0],R[0,1],t[0,0]],
                      [R[1,0],R[1,1],t[1,0]],
                      [0,0,1]])
        # now transform the position of the next point from local to world frame
        pose_local = np.array([[delta_pose_pred[0]],
                               [delta_pose_pred[1]],
                               [1]])
        pose_world = np.matmul(T, pose_local)
        current_pose.pose.position.x = pose_world[0,0]
        current_pose.pose.position.y = pose_world[1,0]
        current_angle += delta_pose_pred[2]
        current_pose.pose.orientation = utilss.angle_to_rosquaternion(current_angle)
        return current_pose
    
    def calc_stamped_poses(self, pos1, pose_f, pose_semi_f):
        pos1.pose.position.x = pos1.pose.position.x + pose_f[0] - pose_semi_f[0]
        pos1.pose.position.y = pos1.pose.position.y + pose_f[1] - pose_semi_f[1]
        angle_f = math.atan2(pose_f[3], pose_f[2])
        angle_semi_f = math.atan2(pose_semi_f[3], pose_semi_f[2])
        orig_angle = utilss.rosquaternion_to_angle(pos1.pose.orientation)
        pos1.pose.orientation = utilss.angle_to_rosquaternion(orig_angle+angle_f-angle_semi_f)
        return pos1

    def sum_stamped_poses(self, pos1, pose_pred):
        pos1.pose.position.x += pose_pred[0]
        pos1.pose.position.y += pose_pred[1]
        angle = math.atan2(pose_pred[3], pose_pred[2])
        orig_angle = utilss.rosquaternion_to_angle(pos1.pose.orientation)
        pos1.pose.orientation = utilss.angle_to_rosquaternion(orig_angle+angle)
        return pos1
    
    def create_map_marker(self, pose_stamped):

        start = time.time()
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.header.stamp = rospy.Time.now()
        marker.type = 8 # points
        marker.id = 0

        # Set the scale of the marker
        marker.scale.x = 0.1
        marker.scale.y = 0.1

        map_recon = self.map_recon.cpu().numpy()[0,0,:]
        [w,h] = map_recon.shape
        
        m_per_px = 12.0/64.0

        # iterate over all pixels from the image to create the map in points
        # only do this for the very first time. then skip because rel pos is the same
        if self.points_viz_list is None:
            self.points_viz_list = []
            for i in range(w):
                for j in range(h):
                    p = Point()
                    p.x = +6.0 - i*m_per_px
                    p.y = +6.0 - j*m_per_px
                    p.z = 0.0
                    self.points_viz_list.append(p)
        marker.points = self.points_viz_list

        finished_points = time.time()
        rospy.loginfo("points delay: "+str(finished_points-start))
        
        # loop to figure out the colors for each point
        for i in range(w):
            for j in range(h):
                cell_val = map_recon[i,j]
                if cell_val < 0.2:
                    alpha = 0.0
                else:
                    alpha = 0.7
                color = ColorRGBA(r=0.0, g=min(max(cell_val,0.0),1.0), b=0.0, a=alpha)
                marker.colors.append(color)

        finished_colors = time.time()
        rospy.loginfo("color delay: "+str(finished_colors-finished_points))

        # Set the pose of the marker
        marker.pose = pose_stamped.pose
        return marker
        
    def create_position_marker(self, pose_stamped, color=[0,1,0,1]):
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.header.stamp = rospy.Time.now()
        marker.type = 0 # arrow
        marker.id = 0

        # Set the scale of the marker
        marker.scale.x = 1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        # Set the color
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

        # Set the pose of the marker
        marker.pose = pose_stamped.pose
        return marker
        
    
    def publish_vel_marker(self):
        marker = Marker()
        marker.header.frame_id = "/car/base_link"
        marker.header.stamp = rospy.Time.now()
        marker.type = 0 # arrow
        marker.id = 0

        # Set the scale of the marker
        marker.scale.x = 1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        # Set the color
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # set the first point
        # point_start = Point()
        # point_start.x = point_start.y = point_start.z = 0.0
        # marker.points.append(point_start)

        # l = 5.0
        # point_end = Point()
        # point_end.x = l*np.cos(self.last_action)
        # point_end.y = l*np.sin(self.last_action)
        # point_end.z = 0.0
        # marker.points.append(point_end)

        # Set the pose of the marker
        marker.pose.position.x = 0.32
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        marker.pose.orientation = utilss.angle_to_rosquaternion(self.last_action)
        # marker.pose.orientation.x = 0.0
        # marker.pose.orientation.y = 0.0
        # marker.pose.orientation.z = 0.0
        # marker.pose.orientation.w = 1.0

        self.vel_marker_pub.publish(marker)


    def apply_network(self):
        start_zero = time.time()
        x_imgs, x_act, t = self.prepare_model_inputs(queue_type='small')
        start = time.time()
        # with torch.set_grad_enabled(False):
        with torch.inference_mode():
            # action_pred = 0.0
            action_pred, _, _, _ = self.model(states=x_imgs, actions=x_act, targets=x_act, gt_map=None, timesteps=t, poses=None, compute_loss=False)
            finished_action_network = time.time()
            # rospy.loginfo("action network delay: "+str(finished_action_network-start))
            # self.act_inference_time_sum += finished_action_network-start
            # self.act_inference_time_count += 1
            rospy.loginfo_throttle(10, "action network delay: "+str(finished_action_network-start))
            # rospy.loginfo_throttle(10, "AVG action network delay: "+str(self.act_inference_time_sum/self.act_inference_time_count))
            action_pred = action_pred[0,-1,0].cpu().flatten().item()
            # if self.use_map:
            #     map_pred, loss = self.map_model(states=x_imgs, actions=x_act, targets=x_act, gt_map=None, timesteps=t, poses=None)
            #     finished_map_network = time.time()
            #     rospy.loginfo("map network delay: "+str(finished_map_network-finished_action_network))
        finished_network = time.time()
        # rospy.loginfo("network delay total: "+str(finished_network-start_zero))
        # de-normalize
        action_pred = pre.denorm_angle(action_pred)
        return action_pred


    def prepare_model_inputs(self, queue_type):
        # start = time.time()
        # organize the scan input
        x_imgs = torch.zeros(1,self.clip_len,720,2)
        x_act = torch.zeros(1,self.clip_len)
        if queue_type=='small':
            self.small_queue_lock.acquire()
            queue_list = list(self.small_queue.queue)
            self.small_queue_lock.release()
        elif queue_type=='large':
            self.large_queue_lock.acquire()
            queue_list = list(self.small_queue.queue)[:self.clip_len]
            self.large_queue_lock.release()
        for idx, element in enumerate(queue_list):
            x_imgs[0,idx,:] = torch.tensor(element[0])
            x_act[0,idx] = torch.tensor(pre.norm_angle(element[1]))
        # x_imgs = x_imgs.contiguous().view(1, self.clip_len, 200*200)
        x_imgs = x_imgs.to(self.device)
        x_act = x_act.view(1, self.clip_len , 1)
        x_act = x_act.to(self.device)
        t = torch.arange(0, self.clip_len).view(1,-1).to(self.device)
        t = t.repeat(1,1)
        # finish_processing = time.time()
        # rospy.loginfo("processing delay: "+str(finish_processing-start))
        return x_imgs, x_act, t

    def check_reset(self, rate_hz):
        # condition if the car gets stuck
        if self.inferred_pose_prev() is not None and self.time_started is not None and self._time_of_inferred_pose is not None and self._time_of_inferred_pose_prev is not None:
            # calculate distance traveled
            delta_dist = np.linalg.norm(np.asarray(self.inferred_pose())-np.asarray(self.inferred_pose_prev()))
            v = 2.0 # default value
            if delta_dist < 0.5:
                delta_time_poses = (self._time_of_inferred_pose-self._time_of_inferred_pose_prev).to_sec()
                self.distance_so_far += delta_dist
                self.time_so_far += delta_time_poses
                # look at speed and termination condition
                if delta_time_poses > 0.001:
                    v = delta_dist / delta_time_poses
                # print('v = {}'.format(v))
            if v < 0.05 and rospy.Time.now().to_sec() - self.time_started.to_sec() > 1.0:
                # this means that the car was supposed to follow a traj, but velocity is too low bc it's stuck
                # first we reset the car pose
                self.reset_counter +=1
        if self.reset_counter > 5 :
            # save distance data to file and reset distance
            delta_time = time.time() - self.last_reset_time
            print("Distance: {}  | Time: {} | Time so far: {}".format(self.distance_so_far, delta_time, self.time_so_far))
            with open(self.file_name,'a') as fd:
                fd.write(str(self.distance_so_far)+','+str(self.time_so_far)+'\n')
            if self.map_type == 'train':
                self.send_initial_pose()
            else:
                self.send_initial_pose_12f()
            rospy.loginfo("Got stuck, resetting pose of the car to default value")
            msg = String()
            msg.data = "got stuck"
            self.expr_at_goal.publish(msg)
            self.reset_counter = 0
            # new_line = np.array([self.distance_so_far, delta_time])
            # self.out_file = open(self.file_name,'ab')
            # np.savetxt(self.out_file, new_line, delimiter=',')
            # self.out_file.close()
            self.distance_so_far = 0.0
            self.time_so_far = 0.0
            self.last_reset_time = time.time()
            return True
        else:
            return False

    def send_initial_pose(self):
        # sample a initial pose for the car based on the valid samples
        hp_world_valid = self.hp_world[self.hp_zerocost_ids]
        new_pos_idx = np.random.randint(0, hp_world_valid.shape[0])
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        # msg.pose.pose.position.x = hp_world_valid[new_pos_idx][0]
        # msg.pose.pose.position.y = hp_world_valid[new_pos_idx][1]
        # msg.pose.pose.position.z = 0.0
        # quat = utilss.angle_to_rosquaternion(hp_world_valid[new_pos_idx][1])
        msg.pose.pose.position.x = 4.12211 + (np.random.rand()-0.5)*2.0*0.5
        msg.pose.pose.position.y = -7.49623 + (np.random.rand()-0.5)*2.0*0.5
        msg.pose.pose.position.z = 0.0
        quat = utilss.angle_to_rosquaternion(np.radians(68 + (np.random.rand()-0.5)*2.0*360)) # 360 instead of zero at the end
        msg.pose.pose.orientation = quat

        self.did_reset = True
        self.time_sent_reset = time.time()

        # # create anchor pose for localization
        # self.pose_anchor = PoseStamped()
        # self.pose_anchor.header = msg.header
        # self.pose_anchor.pose = msg.pose.pose
        # self.current_pose = copy.deepcopy(self.pose_anchor)
        # self.has_loc_anchor = True

        self.pose_reset.publish(msg)

    def send_initial_pose_12f(self):
        # sample a initial pose for the car based on the valid samples
        hp_world_valid = self.hp_world[self.hp_zerocost_ids]
        new_pos_idx = np.random.randint(0, hp_world_valid.shape[0])
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        # msg.pose.pose.position.x = hp_world_valid[new_pos_idx][0]
        # msg.pose.pose.position.y = hp_world_valid[new_pos_idx][1]
        # msg.pose.pose.position.z = 0.0
        # quat = utilss.angle_to_rosquaternion(hp_world_valid[new_pos_idx][1])
        msg.pose.pose.position.x = -3.3559 + (np.random.rand()-0.5)*2.0*0.5
        msg.pose.pose.position.y = 4.511 + (np.random.rand()-0.5)*2.0*0.5
        msg.pose.pose.position.z = 0.0
        quat = utilss.angle_to_rosquaternion(np.radians(-83.115 + (np.random.rand()-0.5)*2.0*360)) # 360 instead of zero at the end
        msg.pose.pose.orientation = quat

        self.did_reset = True
        self.time_sent_reset = time.time()

        self.pose_reset.publish(msg)

    def shutdown(self, signum, frame):
        rospy.signal_shutdown("SIGINT recieved")
        self.run = False
        for ev in self.events:
            ev.set()
    
    def process_scan(self, msg):
        scan = np.zeros((721), dtype=np.float)
        scan[0] = msg.header.stamp.to_sec()
        scan[1:] = msg.ranges
        original_points, sensor_origins, time_stamps, pc_range, voxel_size, lo_occupied, lo_free = pre.load_params(scan)
        points_to_save = np.zeros(shape=(720,2))
        points_to_save[:original_points.shape[0],:] = original_points[:,:2]
        if original_points.shape[0]>0:
            random_indices = np.random.choice(original_points.shape[0], 720-original_points.shape[0], replace=True)
            points_to_save[original_points.shape[0]:,:] = original_points[random_indices,:2]
        return points_to_save

    def cb_scan(self, msg):
        # new lidar scan arrived
        
        # first process all the information
        processed_scan = self.process_scan(msg)
        current_action = copy.deepcopy(self.last_action)
        self.pos_lock.acquire()
        current_pose_gt = copy.deepcopy(self.curr_pose)
        self.pos_lock.release()
        current_pose_pred = copy.deepcopy(self.current_frame)
        queue_element = [processed_scan, current_action, current_pose_gt, current_pose_pred]
        
        # update the small queue
        self.small_queue_lock.acquire()
        if self.small_queue.full():
            self.small_queue.get()  # remove the oldest element, will be replaced next
        self.small_queue.put(queue_element)
        self.small_queue_lock.release()

        # update the large queue. won't remove any elements, only increment
        self.large_queue_lock.acquire()
        self.large_queue.put(queue_element)
        self.large_queue_lock.release()

        # control flags for other processes are activated now that queues have been updated
        self.new_scan_arrived = True
        self.compute_network_action = True
        self.compute_network_loc = True
        self.loc_counter += 1
        

    def setup_pub_sub(self):
        rospy.Service("~reset/soft", SrvEmpty, self.srv_reset_soft)
        rospy.Service("~reset/hard", SrvEmpty, self.srv_reset_hard)

        car_name = self.params.get_str("car_name", default="car")

        rospy.Subscriber(
            "/" + car_name + "/" + 'scan',
            LaserScan,
            self.cb_scan,
            queue_size=10,
        )

        rospy.Subscriber(
            "/" + car_name + "/" + rospy.get_param("~inferred_pose_t"),
            PoseStamped,
            self.cb_pose,
            queue_size=10,
        )

        self.rp_ctrls = rospy.Publisher(
            "/"
            + car_name
            + "/"
            + self.params.get_str(
                "ctrl_topic", default="mux/ackermann_cmd_mux/input/navigation"
            ),
            AckermannDriveStamped,
            queue_size=2,
        )

        self.vel_marker_pub = rospy.Publisher("/model_action_marker", Marker, queue_size = 1)

        # markers for mapping visualization
        self.pose_marker_pub = rospy.Publisher("/pose_marker", Marker, queue_size = 1)
        self.map_marker_pub = rospy.Publisher("/map_marker", Marker, queue_size = 1)

        # markers for localization visualization
        self.loc_anchor_pose_marker_pub = rospy.Publisher("/loc_anchor_pose_marker", Marker, queue_size = 1)
        self.loc_current_pose_marker_pub = rospy.Publisher("/loc_current_pose_marker", Marker, queue_size = 1)

        self.pose_reset = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)

        traj_chosen_t = self.params.get_str("traj_chosen_topic", default="~traj_chosen")
        self.traj_chosen_pub = rospy.Publisher(traj_chosen_t, Marker, queue_size=10)

        # For the experiment framework, need indicators to listen on
        self.expr_at_goal = rospy.Publisher("experiments/finished", String, queue_size=1)
        
        # to publish the new goal, for visualization
        self.goal_pub = rospy.Publisher("~goal", Marker, queue_size=10)

    def srv_reset_hard(self, msg):
        """
        Hard reset does a complete reload of the controller
        """
        rospy.loginfo("Start hard reset")
        self.reset_lock.acquire()
        self.load_controller()
        self.goal_event.clear()
        self.reset_lock.release()
        rospy.loginfo("End hard reset")
        return []

    def srv_reset_soft(self, msg):
        """
        Soft reset only resets soft state (like tensors). No dependencies or maps
        are reloaded
        """
        rospy.loginfo("Start soft reset")
        self.reset_lock.acquire()
        self.rhctrl.reset()
        self.goal_event.clear()
        self.reset_lock.release()
        rospy.loginfo("End soft reset")
        return []

    def find_allowable_pts(self):
        self.hp_map, self.hp_world = self.rhctrl.cost.value_fn._get_halton_pts()
        self.hp_zerocost_ids = np.zeros(self.hp_map.shape[0], dtype=bool)
        for i, pts in enumerate(self.hp_map):
            pts = pts.astype(np.int)
            if int(pts[0])<self.rhctrl.cost.world_rep.dist_field.shape[1] and int(pts[1])<self.rhctrl.cost.world_rep.dist_field.shape[0]:
                if self.rhctrl.cost.world_rep.dist_field[pts[1],pts[0]] == 0.0:
                    self.hp_zerocost_ids[i] = True             

    def cb_pose(self, msg):

        self.pos_lock.acquire()
        self.curr_pose = msg
        self.pos_lock.release()

        if self.inferred_pose is not None:
            self.set_inferred_pose_prev(self.inferred_pose())
            self._time_of_inferred_pose_prev = self._time_of_inferred_pose
        self.set_inferred_pose(self.dtype(utilss.rospose_to_posetup(msg.pose)))
        self._time_of_inferred_pose = msg.header.stamp

        if self.cur_rollout is not None and self.cur_rollout_ip is not None:
            m = Marker()
            m.header.frame_id = "map"
            m.type = m.LINE_STRIP
            m.action = m.ADD
            with self.traj_pub_lock:
                pts = (
                    self.cur_rollout[:, :2] - self.cur_rollout_ip[:2]
                ) + self.inferred_pose()[:2]

            m.points = list(map(lambda xy: Point(x=xy[0], y=xy[1]), pts))

            r, g, b = 0x36, 0xCD, 0xC4
            m.colors = [ColorRGBA(r=r / 255.0, g=g / 255.0, b=b / 255.0, a=0.7)] * len(
                list(m.points)
            )
            m.scale.x = 0.05
            self.traj_chosen_pub.publish(m)

    def publish_traj(self, speed, angle):
        ctrlmsg = AckermannDriveStamped()
        ctrlmsg.header.stamp = rospy.Time.now()
        ctrlmsg.drive.speed = speed
        ctrlmsg.drive.steering_angle = angle
        self.rp_ctrls.publish(ctrlmsg)

    def set_inferred_pose(self, ip):
        with self.inferred_pose_lock:
            self._inferred_pose = ip

    def inferred_pose(self):
        with self.inferred_pose_lock:
            return self._inferred_pose
    
    def set_inferred_pose_prev(self, ip_prev):
        with self.inferred_pose_lock_prev:
            self._inferred_pose_prev = ip_prev

    def inferred_pose_prev(self):
        with self.inferred_pose_lock_prev:
            return self._inferred_pose_prev


if __name__ == "__main__":
    params = parameters.RosParams()
    logger = logger.RosLog()
    node = RHCNode(rhctensor.float_tensor(), params, logger, "rhcontroller")

    signal.signal(signal.SIGINT, node.shutdown)
    rhc = threading.Thread(target=node.start)
    rhc.start()

    # wait for a signal to shutdown
    while node.run:
        signal.pause()

    rhc.join()
