#!/usr/bin/env python

import rospy
from std_msgs.msg import String

class CtrNode:
    def __init__(self):
        rospy.init_node("control_node")

        self.num_trials = 0
        self.max_trials = 5
        self.rate = rospy.Rate(10)
        
        rospy.Subscriber("experiments/finished", String, self.finished_cb)

    def finished_cb(self, msg):
        self.num_trials = self.num_trials + 1
        rospy.loginfo("[Control node] {} out of {} trials".format(self.num_trials, self.max_trials))

    def loop(self):
        while not rospy.is_shutdown():
            if self.num_trials > self.max_trials:
                rospy.loginfo("Shutting down...")
                rospy.signal_shutdown("Reached max trials")
            self.rate.sleep()

if __name__ == '__main__':
    n = CtrNode()
    n.loop()