#!/usr/bin/env python3

# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import sys
print(sys.version)
import cProfile
import os
import signal
import threading
import random
import numpy as np

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import ColorRGBA, Empty, String
from std_srvs.srv import Empty as SrvEmpty
from visualization_msgs.msg import Marker

import logger
import parameters
import rhcbase
import rhctensor
import utils
import librhc.utils as utils_other


class RHCNode(rhcbase.RHCBase):
    def __init__(self, dtype, params, logger, name):
        rospy.init_node(name, anonymous=True, disable_signals=True)

        super(RHCNode, self).__init__(dtype, params, logger)

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

        self.ready_for_network = False
        self.default_speed = 2.5
        self.default_angle = 0.0

        self.do_profile = self.params.get_bool("profile", default=False)

    def start_profile(self):
        if self.do_profile:
            self.logger.warn("Running with profiling")
            self.pr = cProfile.Profile()
            self.pr.enable()

    def end_profile(self):
        if self.do_profile:
            self.pr.disable()
            self.pr.dump_stats(os.path.expanduser("~/mushr_rhc_stats.prof"))

    def start(self):
        self.logger.info("Starting RHController")
        self.start_profile()
        self.setup_pub_sub()
        self.rhctrl = self.load_controller()
        self.T = self.params.get_int("T")
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
            if self.check_reset(rate_hz):
                rospy.loginfo("Resetting the car's position")

            # publish next action
            if self.ready_for_network is True:
                angle = self.apply_network()
                self.publish_traj(self.default_speed, angle)
            else:
                self.publish_traj(self.default_speed, self.default_angle)
                print(sys.version)

            rate.sleep()

        self.end_profile()
    
    def apply_network(self):
        pass

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
        msg.pose.pose.position.x = hp_world_valid[new_pos_idx][0]
        msg.pose.pose.position.y = hp_world_valid[new_pos_idx][1]
        msg.pose.pose.position.z = 0.0
        quat = utils.angle_to_rosquaternion(hp_world_valid[new_pos_idx][1])
        msg.pose.pose.orientation = quat
        self.pose_reset.publish(msg)

    def shutdown(self, signum, frame):
        rospy.signal_shutdown("SIGINT recieved")
        self.run = False
        for ev in self.events:
            ev.set()

    def setup_pub_sub(self):
        rospy.Service("~reset/soft", SrvEmpty, self.srv_reset_soft)
        rospy.Service("~reset/hard", SrvEmpty, self.srv_reset_hard)

        car_name = self.params.get_str("car_name", default="car")

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
        self.set_inferred_pose(self.dtype(utils.rospose_to_posetup(msg.pose)))

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
