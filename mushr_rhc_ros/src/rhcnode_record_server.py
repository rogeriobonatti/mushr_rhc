#!/usr/bin/env python

# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

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
        # self.max_trials = 3

        self.cur_rollout = self.cur_rollout_ip = None
        self.traj_pub_lock = threading.Lock()

        self.goal_event = threading.Event()
        self.map_metadata_event = threading.Event()
        self.ready_event = threading.Event()
        self.events = [self.goal_event, self.map_metadata_event, self.ready_event]
        self.run = True

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

        # wait until we actually have a car pose
        rospy.loginfo("Waiting to receive pose")
        while not rospy.is_shutdown() and self.inferred_pose is None:
            pass
        rospy.loginfo("Vehicle pose received")
 
        while not rospy.is_shutdown() and self.run:
            ip = self.inferred_pose()
            next_traj, rollout = self.run_loop(ip)

            # check if we should send set a new goal location
            if self.check_new_goal(next_traj, rate_hz):
                rospy.loginfo("Going to set next goal: number {}".format(self.num_trials))
                self.num_trials += 1
                # if self.num_trials > self.max_trials:
                #     rospy.loginfo("Shutting down...")
                    # rospy.signal_shutdown("Reach max trials")
                    # continue
                self.set_new_random_goal()

            with self.traj_pub_lock:
                if rollout is not None:
                    self.cur_rollout = rollout.clone()
                    self.cur_rollout_ip = ip

            if next_traj is not None:
                self.publish_traj(next_traj, rollout)
                # For experiments. If the car is at the goal, notify the
                # experiment tool
                # if self.rhctrl.at_goal(self.inferred_pose()):
                #     self.expr_at_goal.publish(Empty())
                #     self.goal_event.clear()

            rate.sleep()

        self.end_profile()

    def check_new_goal(self, next_traj, rate_hz):
        # condition if there is no goal currently set
        if self.goal_event.is_set() is False:
            msg = String()
            msg.data = "first time"
            self.expr_at_goal.publish(msg)
            return True
        # condition if the goal is reached
        if self.rhctrl.at_goal(self.inferred_pose()):
            rospy.loginfo("Reached the goal")
            msg = String()
            msg.data = "reached goal"
            self.expr_at_goal.publish(msg)
            self.goal_event.clear()
            return True
        # condition if the car gets stuck
        if self.inferred_pose_prev() is not None and next_traj is not None and self.time_started_goal is not None:
            v = np.linalg.norm(np.asarray(self.inferred_pose())-np.asarray(self.inferred_pose_prev())) * rate_hz
            if v < 0.05 and rospy.Time.now().to_sec() - self.time_started_goal.to_sec() > 1.0:
                # this means that the car was supposed to follow a traj, but velocity is too low bc it's stuck
                # first we reset the car pose, then we return True to select a new goal
                self.goal_event.clear()
                self.send_initial_pose()
                rospy.loginfo("Got stuck, resetting pose of the car to default value")
                msg = String()
                msg.data = "got stuck"
                self.expr_at_goal.publish(msg)
                return True
        # condition if the car gets inside an infinite loop and never reaches goal
        if self.time_started_goal is not None:
            if rospy.Time.now().to_sec() - self.time_started_goal.to_sec() > 90.0:
                rospy.loginfo("Timeout, couldn't reach goal after a while")
                msg = String()
                msg.data = "timeout"
                self.expr_at_goal.publish(msg)
                self.goal_event.clear()
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
    
    def run_loop(self, ip):
        # self.goal_event.wait()
        if rospy.is_shutdown() or ip is None:
            return None, None
        with self.reset_lock:
            # If a reset is initialed after the goal_event was set, the goal
            # will be cleared. So we have to have another goal check here.
            if not self.goal_event.is_set():
                return None, None
            if ip is not None:
                return self.rhctrl.step(ip)
            self.logger.err("Shouldn't get here: run_loop")

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
            "/move_base_simple/goal", PoseStamped, self.cb_goal, queue_size=1
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

    def cb_goal(self, msg):
        goal = self.dtype(utils.rospose_to_posetup(msg.pose))
        self.ready_event.wait()
        if not self.rhctrl.set_goal(goal):
            self.logger.err("That goal is unreachable, please choose another")
            return
        else:
            self.logger.info("Goal set")
            self.time_started_goal = rospy.Time.now()
        self.goal_event.set()
        print("Setting goal from RVIZ")

    def select_random_goal(self, pts):
        i = random.randint(0,pts.shape[0])
        return pts[i,:][0], pts[i,:][1], 0.0

    def find_allowable_pts(self):
        self.hp_map, self.hp_world = self.rhctrl.cost.value_fn._get_halton_pts()
        self.hp_zerocost_ids = np.zeros(self.hp_map.shape[0], dtype=bool)
        for i, pts in enumerate(self.hp_map):
            pts = pts.astype(np.int)
            if int(pts[0])<self.rhctrl.cost.world_rep.dist_field.shape[1] and int(pts[1])<self.rhctrl.cost.world_rep.dist_field.shape[0]:
                if self.rhctrl.cost.world_rep.dist_field[pts[1],pts[0]] == 0.0:
                    self.hp_zerocost_ids[i] = True

    def set_new_random_goal(self):
        self.logger.info("Setting a new random goal in the map")
        self.display_halton(self.hp_map, self.hp_world)
        hp_world_valid = self.hp_world[self.hp_zerocost_ids]
        new_goal_idx = np.random.randint(0, hp_world_valid.shape[0])
        goal = self.dtype((hp_world_valid[new_goal_idx][0], 
                           hp_world_valid[new_goal_idx][1], 
                           np.random.uniform(-np.pi, np.pi)))
        self.display_goal(goal)
        # sanity check
        self.ready_event.wait()
        if not self.rhctrl.set_goal(goal):
            self.logger.err("That goal is unreachable, please choose another")
            return
        else:
            self.logger.info("Goal set")
        self.goal_event.set()
        self.time_started_goal = rospy.Time.now()
        # # query the set of all possible points
        # hp_map, hp_world = self.rhctrl.cost.value_fn._get_halton_pts()
        # self.display_halton(hp_map, hp_world)
        # goal_ok = False
        # while goal_ok is not True:
        #     print("Sampling new goal...")
            # goal = self.dtype(self.select_random_goal(hp_world))
        #     goal_new = goal.unsqueeze(0)
        #     map_goal = self.dtype(goal_new.size())
        #     utils_other.map.world2map(self.map_data, goal_new, out=map_goal)
        #     m = map_goal.int().numpy()[0]
        #     # NOTE: x and y are flipped when looking at image values
        #     if int(m[0])>=self.rhctrl.cost.world_rep.dist_field.shape[1] or int(m[1])>=self.rhctrl.cost.world_rep.dist_field.shape[0]:
        #         continue
        #     else:
        #         if self.rhctrl.cost.world_rep.dist_field[m[1],m[0]] != 0.0:
        #             continue
        #         # if we got here, then the point has zero value
        #         goal_ok = True
        # # sanity check
        # self.ready_event.wait()
        # if not self.rhctrl.set_goal(goal):
        #     self.logger.err("That goal is unreachable, please choose another")
        #     return
        # else:
        #     self.logger.info("Goal set")
        # self.goal_event.set()
        # self.time_started_goal = rospy.Time.now()

    def display_halton(self, hp_map, hp_world):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.ns = "hp_free"
        m.id = 0
        m.type = m.POINTS
        m.action = m.ADD
        m.pose.position.x = 0
        m.pose.position.y = 0
        m.pose.position.z = 0
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = 0.1
        m.scale.y = 0.1
        m.scale.z = 0.1
        for i, pts in enumerate(self.hp_world[self.hp_zerocost_ids]):
            p = Point()
            c = ColorRGBA()
            c.a = 1
            c.r = 1.0
            c.g = 0
            c.b = 0
            p.x, p.y = pts[0], pts[1]
            m.points.append(p)
            m.colors.append(c)
        pub = rospy.Publisher("~markers_zero_cost", Marker, queue_size=100)
        pub.publish(m)

    def display_goal(self, goal):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.id = 1
        m.type = m.SPHERE
        m.action = m.ADD
        m.pose.position.x = goal[0]
        m.pose.position.y = goal[1]
        m.pose.position.z = 0
        quat = utils.angle_to_rosquaternion(goal[2])
        m.pose.orientation.x = quat.x
        m.pose.orientation.y = quat.y
        m.pose.orientation.z = quat.z
        m.pose.orientation.w = quat.w
        m.color.g = 1.0
        m.scale.x = 0.5
        m.scale.y = 0.5
        m.scale.z = 0.5
        m.color.a = 1.0
        self.goal_pub.publish(m)

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

    def publish_traj(self, traj, rollout):
        assert traj.size() == (self.T, 2)
        assert rollout.size() == (self.T, 3)

        ctrl = traj[0]
        ctrlmsg = AckermannDriveStamped()
        ctrlmsg.header.stamp = rospy.Time.now()
        ctrlmsg.drive.speed = ctrl[0]
        ctrlmsg.drive.steering_angle = ctrl[1]
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
