#!/usr/bin/env python3

# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

from torchsummary import summary
import sys
import os
import signal
import threading
import random
import numpy as np
from queue import Queue
import time

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
from mingpt.model_mushr_rogerio import GPT, GPTConfig
import preprocessing_utils as pre

class RHCNode(rhcbase.RHCBase):
    def __init__(self, dtype, params, logger, name):
        rospy.init_node(name, anonymous=True, disable_signals=True)

        super(RHCNode, self).__init__(dtype, params, logger)

        self.scan_lock = threading.Lock()

        self.reset_lock = threading.Lock()
        self.inferred_pose_lock = threading.Lock()
        self.inferred_pose_lock_prev = threading.Lock()
        self._inferred_pose = None
        self._inferred_pose_prev = None

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
        self.default_angle = 0.0
        self.nx = None
        self.ny = None
        self.reset_counter = 0
        
        # network loading
        print("Starting to load model")
        os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
        device = torch.device('cuda')
        
        self.clip_len = 16
        # saved_model_path = '/home/rb/downloaded_models/epoch30.pth.tar'
        saved_model_path = '/home/rb/hackathon_data/aml_outputs/log_output/gpt_resnet18_0/GPTgpt_resnet18_4gpu_2022-01-24_1642987604.6403077_2022-01-24_1642987604.640322/model/epoch15.pth.tar'
        vocab_size = 100
        block_size = self.clip_len * 2
        max_timestep = 7
        # mconf = GPTConfig(vocab_size, block_size, max_timestep,
        #               n_layer=6, n_head=8, n_embd=128, model_type='GPT', use_pred_state=True,
        #               state_tokenizer='conv2D', train_mode='e2e', pretrained_model_path='')
        mconf = GPTConfig(vocab_size, block_size, max_timestep,
                      n_layer=6, n_head=8, n_embd=128, model_type='GPT', use_pred_state=True,
                      state_tokenizer='resnet18', train_mode='e2e', pretrained_model_path='', pretrained_encoder_path='', loss='MSE')              
        model = GPT(mconf, device)
        model=torch.nn.DataParallel(model)

        # ckpt = torch.load('/home/rb/downloaded_models/epoch30.pth.tar')['state_dict']
        # for key in ckpt:
        #     print('********',key)
        # model.load_state_dict(torch.load('/home/rb/downloaded_models/epoch30.pth.tar')['state_dict'], strict=True)
        
        checkpoint = torch.load(saved_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model.to(device)
        self.model = model
        self.device = device
        print("Finished loading model")

        self.q_scans = Queue(maxsize = self.clip_len+1)
        self.q_actions = Queue(maxsize = self.clip_len)
        for i in range(self.clip_len):
            self.q_actions.put(self.default_angle)
        self.last_action = self.default_angle
        self.compute_network = False
        self.priming_phase = True
        self.time_last_reset = rospy.Time.now()
        self.priming_array_straight = np.array([[3.0, 0.0, 0.0]])
        self.priming_array_zigzag = np.array([[0.1, 0.0, 0.0],
                                             [0.2, 0.0, -0.2],
                                             [0.4, -0.2, 0.34],
                                             [0.6, 0.34, -0.34],
                                             [0.8, -0.34, 0.34],
                                             [1.2, 0.34, -0.34],
                                             [1.4, -0.34, 0.34],
                                             [1.6, 0.34, -0.34],
                                             [1.8, -0.34, 0.34],
                                             [2.0, 0.34, -0.34],
                                             [2.2, -0.34, 0.34],
                                             [2.4, 0.34, -0.34],
                                             [2.6, -0.34, 0.34],
                                             [2.8, 0.34, -0.34],])
        self.priming_array_curve = np.array([[2.9, 0.0, 0.0],
                                             [3.0, 0.0, -0.34],
                                             [3.65, -0.34, -0.34],
                                             [3.75, -0.34, 0.0],
                                             [6.0, 0.0, 0.0],
                                             [6.1, 0.0, -0.34],
                                             [6.75, -0.34, -0.34],
                                             [6.85, -0.34, 0.0],
                                             [8.0, 0.0, 0.0]])

    def find_row(self, t, table):
        # table contains Tf, amin, amax
        # T init is zero
        t_prev = 0.0
        for i in range(table.shape[0]):
            if t>=t_prev and t<table[i,0]:
                # this is the correct row
                frac = (t-t_prev)/(table[i,0]-t_prev)
                print("Frac: {:.2f}".format(frac))
                return table[i,1] + frac*(table[i,2]-table[i,1]), True
            t_prev = table[i,0]
        return table[-1,2], False # return the last action and stop priming from now on
        

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
        self.time_started = rospy.Time.now()

        # wait until we actually have a car pose
        rospy.loginfo("Waiting to receive pose")
        while not rospy.is_shutdown() and self.inferred_pose is None:
            pass
        rospy.loginfo("Vehicle pose received")
 
        while not rospy.is_shutdown() and self.run:

            # check if we should reset the vehicle if crashed
            if self.check_reset_improved(rate_hz):
                rospy.loginfo("Resetting the car's position")

            # publish next action
            if self.priming_phase:
                time_diff = (rospy.Time.now()-self.time_last_reset).to_sec()
                # compute some action that depends on the model priming logic, and not the network
                self.last_action, self.priming_phase = self.find_row(time_diff, self.priming_array_straight) 
                # self.last_action, self.priming_phase = self.find_row(time_diff, self.priming_array_zigzag)          
                # self.last_action, self.priming_phase = self.find_row(time_diff, self.priming_array_curve)
                self.q_actions.get()  # remove the oldest action from the queue
                self.q_actions.put(self.last_action)
                rospy.loginfo("Priming: "+str(self.last_action))
            elif self.compute_network:
                # don't have to run the network at all times, only when scans change and scans are full
                self.last_action = self.apply_network()
                self.q_actions.get()  # remove the oldest action from the queue
                self.q_actions.put(self.last_action)
                rospy.loginfo("Applied network: "+str(self.last_action))
                self.compute_network = False
            
            self.publish_traj(self.default_speed, self.last_action)

            rate.sleep()

    def apply_network(self):
        start = time.time()
        # organize the scan input
        x_imgs = torch.zeros(1,self.clip_len,self.nx,self.ny)
        y_imgs = torch.zeros(1,1,self.nx,self.ny)
        x_act = torch.zeros(1,self.clip_len)
        y_act = None
        
        
        self.scan_lock.acquire()
        queue_list = list(self.q_scans.queue)
        queue_size = self.q_scans.qsize()
        self.scan_lock.release()

        # while True:
        #     try:
        #         queue_list = self.q_scans.queue
        #         if len(queue_list)==self.clip_len+1:
        #             break
        #     except ValueError:
        #         print("EXCEPTION: diff number of images, or read at the wrong time")
        
        idx = 0
        for img in queue_list:
            if idx==queue_size-1:
                y_imgs[0,0,:] = torch.tensor(img)
            else:
                x_imgs[0,idx,:] = torch.tensor(img)
            idx+=1
        idx = 0
        for act in self.q_actions.queue:
            x_act[0,idx] = torch.tensor(act)
            idx+=1

        x_imgs = x_imgs.contiguous().view(1, self.clip_len, 200*200)
        x_imgs = x_imgs.to(self.device)
        # y_imgs = y_imgs.to(self.device)

        x_act = x_act.view(1, self.clip_len , 1)
        x_act = x_act.to(self.device)
        # y_act = y_act.to(self.device)

        t = np.ones((1, 1, 1), dtype=int) * 7
        t = torch.tensor(t)
        t = t.to(self.device)

        finish_processing = time.time()
        # rospy.loginfo("processing delay: "+str(finish_processing-start))

        # organize the action input
        with torch.set_grad_enabled(False):
            action_pred, loss = self.model(states=x_imgs, actions=x_act, targets=x_act, timesteps=t)
            action_pred = action_pred[0,self.clip_len-1,0].cpu().flatten().item()
        finished_network = time.time()
        # rospy.loginfo("network delay: "+str(finished_network-finish_processing))

        # de-normalize
        action_pred = pre.denorm_angle(action_pred)
        return action_pred

    def check_reset_improved(self, rate_hz):
        # condition if the car gets stuck
        if self.inferred_pose_prev() is not None and self.time_started is not None:
            v = np.linalg.norm(np.asarray(self.inferred_pose())-np.asarray(self.inferred_pose_prev())) * rate_hz
            if v < 0.05 and rospy.Time.now().to_sec() - self.time_started.to_sec() > 1.0:
                # this means that the car was supposed to follow a traj, but velocity is too low bc it's stuck
                # first we reset the car pose
                self.reset_counter +=1
        if self.reset_counter >5:
            self.send_initial_pose()
            rospy.loginfo("Got stuck, resetting pose of the car to default value")
            msg = String()
            msg.data = "got stuck"
            self.expr_at_goal.publish(msg)
            self.reset_counter = 0
            return True
        else:
            return False

    def check_reset(self, rate_hz):
        # condition if the car gets stuck
        if self.inferred_pose_prev() is not None and self.time_started is not None:
            v = np.linalg.norm(np.asarray(self.inferred_pose())-np.asarray(self.inferred_pose_prev())) * rate_hz
            if v < 0.05 and rospy.Time.now().to_sec() - self.time_started.to_sec() > 1.0:
                # this means that the car was supposed to follow a traj, but velocity is too low bc it's stuck
                # first we reset the car pose
                self.send_initial_pose()
                rospy.loginfo("Got stuck, resetting pose of the car to default value")
                msg = String()
                msg.data = "got stuck"
                self.expr_at_goal.publish(msg)
                return True
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
        
        # msg.pose.pose.position.x = 4.12211 + (np.random.rand()-0.5)*2.0*0.5
        # msg.pose.pose.position.y = -7.49623 + (np.random.rand()-0.5)*2.0*0.5
        # msg.pose.pose.position.z = 0.0
        # quat = utilss.angle_to_rosquaternion(np.radians(62.373 + (np.random.rand()-0.5)*2.0*3))
        
        msg.pose.pose.position.x = 0.0767436203959 + (np.random.rand()-0.5)*2.0*0.1
        msg.pose.pose.position.y = -17.7567761914 + (np.random.rand()-0.5)*2.0*0.1
        msg.pose.pose.position.z = 0.0
        quat = utilss.angle_to_rosquaternion(np.radians(67.032 + (np.random.rand()-0.5)*2.0*1))
        
        msg.pose.pose.orientation = quat
        self.pose_reset.publish(msg)
        self.priming_phase = True
        self.time_last_reset = rospy.Time.now()

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
        # remove oldest element if the queue is already full
        self.scan_lock.acquire()
        if self.q_scans.full():
            self.compute_network = True  # start running the network in the main loop from now on
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
        if self.inferred_pose is not None:
            self.set_inferred_pose_prev(self.inferred_pose())
        self.set_inferred_pose(self.dtype(utilss.rospose_to_posetup(msg.pose)))

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
