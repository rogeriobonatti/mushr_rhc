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
from mingpt.model_mushr_nips import GPT, GPTConfig
import preprocessing_utils as pre
from visualization_msgs.msg import Marker

# import torch_tensorrt


class RHCNode(rhcbase.RHCBase):
    def __init__(self, dtype, params, logger, name):
        rospy.init_node(name, anonymous=True, disable_signals=True)

        super(RHCNode, self).__init__(dtype, params, logger)

        self.scan_lock = threading.Lock()
        self.pos_lock = threading.Lock()
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
        self.use_map = True
        self.use_loc = True
        self.points_viz_list = None
        self.map_recon = None
        self.loc_counter = 0
        
        # network loading
        print("Starting to load model")
        os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
        device = torch.device('cuda')
        # device = "cpu"

        self.device = device
        self.clip_len = 16

        # tests for IROS
        saved_model_path = rospy.get_param("~model_path", 'default_value')
        self.out_path = rospy.get_param("~out_path", 'default_value')
        # saved_model_path = '/home/rb/hackathon_data/aml_outputs/log_output/normal-kingfish/GPTiros_e2e_8gpu_2022-02-17_1645120431.7528405_2022-02-17_1645120431.7528613/model/epoch10.pth.tar'

        # saved_model_path = '/home/rb/downloaded_models/epoch30.pth.tar'
        # saved_model_path = '/home/robot/weight_files/epoch15.pth.tar'
        # saved_model_path = '/home/rb/hackathon_data/aml_outputs/log_output/gpt_resnet18_0/GPTgpt_resnet18_4gpu_2022-01-24_1642987604.6403077_2022-01-24_1642987604.640322/model/epoch15.pth.tar'
        # saved_model_path = '/home/rb/hackathon_data/aml_outputs/log_output/gpt_resnet18_8_exp2/GPTgpt_resnet18_8gpu_exp2_2022-01-25_1643076745.003202_2022-01-25_1643076745.0032148/model/epoch12.pth.tar'
        vocab_size = 100
        block_size = self.clip_len * 2
        max_timestep = 7
        # mconf = GPTConfig(vocab_size, block_size, max_timestep,
        #               n_layer=6, n_head=8, n_embd=128, model_type='GPT', use_pred_state=True,
        #               state_tokenizer='conv2D', train_mode='e2e', pretrained_model_path='')
        mconf = GPTConfig(vocab_size, block_size, max_timestep,
                      n_layer=6, n_head=8, n_embd=128, model_type='GPT', use_pred_state=True,
                      state_tokenizer='resnet18', train_mode='e2e', pretrained_model_path='', pretrained_encoder_path='', loss='MSE',
                      map_decoder='deconv', map_recon_dim=128)
        model = GPT(mconf, device)
        # model=torch.nn.DataParallel(model)

        checkpoint = torch.load(saved_model_path, map_location=device)
        # old code for loading model
        # model.load_state_dict(checkpoint['state_dict'])
        # new code for loading mode
        new_checkpoint = OrderedDict()
        for key in checkpoint['state_dict'].keys():
            new_checkpoint[key.split("module.",1)[1]] = checkpoint['state_dict'][key]
        model.load_state_dict(new_checkpoint)

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
            saved_map_model_path = '/home/rb/hackathon_data/aml_outputs/log_output/mushr_nips/GPTnips_8gpu_relu_map_2022-03-31_1648698125.6499674_2022-03-31_1648698125.6499808/model/epoch18.pth.tar'
            map_mconf = GPTConfig(vocab_size, block_size, max_timestep,
                          n_layer=6, n_head=8, n_embd=128, model_type='GPT', use_pred_state=True,
                          state_tokenizer='resnet18', train_mode='map', pretrained_model_path='', pretrained_encoder_path='', loss='MSE',
                          map_decoder='deconv', map_recon_dim=128)
            map_model = GPT(map_mconf, device)
            # map_model=torch.nn.DataParallel(map_model)
            checkpoint = torch.load(saved_map_model_path, map_location=device)

            # old code for loading model
            # map_model.load_state_dict(checkpoint['state_dict'])
            # new code for loading mode
            new_checkpoint = OrderedDict()
            for key in checkpoint['state_dict'].keys():
                new_checkpoint[key.split("module.",1)[1]] = checkpoint['state_dict'][key]
            map_model.load_state_dict(new_checkpoint)
            
            map_model.eval()
            map_model.to(device)
            self.map_model = map_model

        # localization model
        if self.use_loc:
            saved_loc_model_path = '/home/rb/hackathon_data/aml_outputs/log_output/mushr_nips/GPTnips_8gpu_relu_loc_2022-03-31_1648697715.6208682_2022-03-31_1648697715.6208813/model/epoch29.pth.tar'
            loc_mconf = GPTConfig(vocab_size, block_size, max_timestep,
                          n_layer=6, n_head=8, n_embd=128, model_type='GPT', use_pred_state=True,
                          state_tokenizer='resnet18', train_mode='loc', pretrained_model_path='', pretrained_encoder_path='', loss='MSE',
                          map_decoder='deconv', map_recon_dim=128)
            loc_model = GPT(loc_mconf, device)
            # map_model=torch.nn.DataParallel(map_model)
            checkpoint = torch.load(saved_loc_model_path, map_location=device)

            # old code for loading model
            # loc_model.load_state_dict(checkpoint['state_dict'])
            # new code for loading mode
            new_checkpoint = OrderedDict()
            for key in checkpoint['state_dict'].keys():
                new_checkpoint[key.split("module.",1)[1]] = checkpoint['state_dict'][key]
            loc_model.load_state_dict(new_checkpoint)
            
            loc_model.eval()
            loc_model.to(device)
            self.loc_model = loc_model


        self.q_scans = Queue(maxsize = self.clip_len)
        self.q_actions = Queue(maxsize = self.clip_len)
        self.q_pos = Queue(maxsize = self.clip_len)
        for i in range(self.clip_len):
            self.q_actions.put(self.default_angle)
        self.last_action = self.default_angle
        self.compute_network = False
        self.compute_network_loc = False
        self.has_loc_anchor = False
        self.did_reset = False

        # parameters for model evaluation
        self.reset_counter = 0
        self.last_reset_time = time.time()
        self.distance_so_far = 0.0
        self.time_so_far = 0.0
        self.file_name = os.path.join(self.out_path,'info.csv')

        # set timer callbacks for visualization
        rate_map_display = 1.0
        rate_loc_display = 15
        self.map_viz_timer = rospy.Timer(rospy.Duration(1.0 / rate_map_display), self.map_viz_cb)
        self.map_viz_loc = rospy.Timer(rospy.Duration(1.0 / rate_loc_display), self.loc_viz_cb)


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
        self.send_initial_pose()
        # self.send_initial_pose_12f()
        self.time_started = rospy.Time.now()

        # wait until we actually have a car pose
        rospy.loginfo("Waiting to receive pose")
        while not rospy.is_shutdown() and self.inferred_pose is None:
            pass
        rospy.loginfo("Vehicle pose received")
 
        while not rospy.is_shutdown() and self.run:

            # check if we should reset the vehicle if crashed
            if self.check_reset(rate_hz):
                rospy.loginfo("Resetting the car's position")

            # publish next action
            if self.compute_network:
                # don't have to run the network at all times, only when scans change and scans are full
                self.last_action = self.apply_network()
                self.q_actions.get()  # remove the oldest action from the queue
                self.q_actions.put(self.last_action)
                # rospy.loginfo("Applied network: "+str(self.last_action))
                self.compute_network = False
            
            self.publish_vel_marker()
            self.publish_traj(self.default_speed, self.last_action)
            
            # if map is not None:
            rate.sleep()

    def map_viz_cb(self, timer):
        self.pos_lock.acquire()
        pos_queue_list = list(self.q_pos.queue)
        pos_size = len(pos_queue_list)
        self.pos_lock.release()
        if pos_size==16:
            x_imgs, x_act, t = self.prepare_model_inputs()
            start = time.time()
            # with torch.set_grad_enabled(False):
            with torch.inference_mode():
                self.map_recon, loss = self.map_model(states=x_imgs, actions=x_act, targets=x_act, gt_map=None, timesteps=t, poses=None)
            finished_map_network = time.time()
            rospy.loginfo("map network delay: "+str(finished_map_network-start))
            pose_mid = pos_queue_list[int(pos_size/2) -1]
            # publish the GT pose of the map center
            self.pose_marker_pub.publish(self.create_position_marker(pose_mid))
            # publish the map itself
            self.map_marker_pub.publish(self.create_map_marker(pose_mid))
        

        
    def loc_viz_cb(self, timer):
        if self.compute_network_loc is False:
            return
        self.pos_lock.acquire()
        pos_queue_list = list(self.q_pos.queue)
        pos_size = len(pos_queue_list)
        self.pos_lock.release()

        # create anchor pose for localization
        if time.time()-self.time_sent_reset>3.0 and self.did_reset is True:
            self.loc_counter = 0
            self.pose_anchor = copy.deepcopy(self.curr_pose)
            self.current_pose = copy.deepcopy(self.pose_anchor)
            self.has_loc_anchor = True
            self.did_reset = False

        if self.loc_counter>=16 and self.has_loc_anchor is True:
            self.loc_counter = 0
            x_imgs, x_act, t = self.prepare_model_inputs()
            start = time.time()
            # with torch.set_grad_enabled(False):
            with torch.inference_mode():
                pose_preds, loss = self.loc_model(states=x_imgs, actions=x_act, targets=x_act, gt_map=None, timesteps=t, poses=None)
            finished_loc_network = time.time()
            rospy.loginfo("loc network delay: "+str(finished_loc_network-start))
            # publish anchor pose of the map center
            self.loc_anchor_pose_marker_pub.publish(self.create_position_marker(self.pose_anchor, color=[0,0,1,1]))
            # publish the current accumulated pose
            pose_pred = pose_preds[0,self.clip_len-1,:].cpu().numpy()
            self.current_pose = self.sum_stamped_poses(self.current_pose, pose_pred)
            self.loc_current_pose_marker_pub.publish(self.create_position_marker(self.current_pose, color=[1,0,0,1]))

        # if pos_size==16 and self.has_loc_anchor is True:
        #     x_imgs, x_act, t = self.prepare_model_inputs()
        #     start = time.time()
        #     with torch.set_grad_enabled(False):
        #         pose_preds, loss = self.loc_model(states=x_imgs, actions=x_act, targets=x_act, gt_map=None, timesteps=t, poses=None)
        #     finished_loc_network = time.time()
        #     rospy.loginfo("loc network delay: "+str(finished_loc_network-start))
        #     # publish anchor pose of the map center
        #     self.loc_anchor_pose_marker_pub.publish(self.create_position_marker(self.pose_anchor, color=[0,0,1,1]))
        #     # publish the current accumulated pose
        #     pose_f = pose_preds[0,self.clip_len-1,:].cpu().numpy()
        #     pose_semi_f = pose_preds[0,self.clip_len-2,:].cpu().numpy()
        #     self.current_pose = self.calc_stamped_poses(self.current_pose, pose_f, pose_semi_f)
        #     self.loc_current_pose_marker_pub.publish(self.create_position_marker(self.current_pose, color=[1,0,0,1]))

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
        
        m_per_px = 20.0/128.0

        # iterate over all pixels from the image to create the map in points
        # only do this for the very first time. then skip because rel pos is the same
        if self.points_viz_list is None:
            self.points_viz_list = []
            for i in range(w):
                for j in range(h):
                    p = Point()
                    p.x = +10.0 - i*m_per_px
                    p.y = +10.0 - j*m_per_px
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
        x_imgs, x_act, t = self.prepare_model_inputs()
        start = time.time()
        # with torch.set_grad_enabled(False):
        with torch.inference_mode():
            # action_pred = 0.0
            action_pred, loss = self.model(states=x_imgs, actions=x_act, targets=x_act, gt_map=None, timesteps=t, poses=None)
            finished_action_network = time.time()
            rospy.loginfo("action network delay: "+str(finished_action_network-start))
            action_pred = action_pred[0,self.clip_len-1,0].cpu().flatten().item()
            # if self.use_map:
            #     map_pred, loss = self.map_model(states=x_imgs, actions=x_act, targets=x_act, gt_map=None, timesteps=t, poses=None)
            #     finished_map_network = time.time()
            #     rospy.loginfo("map network delay: "+str(finished_map_network-finished_action_network))
        finished_network = time.time()
        # rospy.loginfo("network delay: "+str(finished_network-finish_processing))

        # de-normalize
        action_pred = pre.denorm_angle(action_pred)
        return action_pred
        # if self.use_map:
        #     return action_pred
        # else:
        #     return action_pred

    def prepare_model_inputs(self):
        start = time.time()
        # organize the scan input
        x_imgs = torch.zeros(1,self.clip_len,self.nx,self.ny)
        x_act = torch.zeros(1,self.clip_len)
        
        self.scan_lock.acquire()
        queue_list = list(self.q_scans.queue)
        queue_size = self.q_scans.qsize()
        self.scan_lock.release()
        
        idx = 0
        for img in queue_list:
            x_imgs[0,idx,:] = torch.tensor(img)
            idx+=1
        idx = 0
        for act in self.q_actions.queue:
            x_act[0,idx] = torch.tensor(act)
            idx+=1

        x_imgs = x_imgs.contiguous().view(1, self.clip_len, 200*200)
        x_imgs = x_imgs.to(self.device)

        x_act = x_act.view(1, self.clip_len , 1)
        x_act = x_act.to(self.device)

        t = np.ones((1, 1, 1), dtype=int) * 7
        t = torch.tensor(t)
        t = t.to(self.device)

        finish_processing = time.time()
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
            self.send_initial_pose()
            # self.send_initial_pose_12f()
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
        quat = utilss.angle_to_rosquaternion(np.radians(68 + (np.random.rand()-0.5)*2.0*0)) # 360 instead of zero at the end
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
        quat = utilss.angle_to_rosquaternion(np.radians(-83.115 + (np.random.rand()-0.5)*2.0*0)) # 360 instead of zero at the end
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
        vis_mat, nx, ny = pre.compute_bev_image(original_points, sensor_origins, time_stamps, pc_range, voxel_size)
        if self.nx is None:
            self.nx = nx
            self.ny = ny
        return vis_mat

    def cb_scan(self, msg):

        # remove element from position queue:
        self.pos_lock.acquire()
        if self.q_pos.full():
            self.q_pos.get()  # remove the oldest element, will be replaced next
        self.pos_lock.release()

        # add new vehicle position
        self.pos_lock.acquire()
        if self.curr_pose is None:
            self.pos_lock.release()
            # exist the callback if there is no current pose: will only happen at the very beginning
            return
        else:
            self.q_pos.put(self.curr_pose)
        self.pos_lock.release()

        # remove oldest element if the queue is already full
        self.scan_lock.acquire()
        if self.q_scans.full():
            self.compute_network = True  # start running the network in the main loop from now on
            self.compute_network_loc = True
            self.loc_counter += 1
            self.q_scans.get()  # remove the oldest element, will be replaced next
        self.scan_lock.release()
        
        # add new processed scan
        tmp = self.process_scan(msg)
        self.scan_lock.acquire()
        self.q_scans.put(tmp) # store matrices from 0-1 with the scans
        self.scan_lock.release()
        

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
