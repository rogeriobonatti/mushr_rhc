#!/usr/bin/env python3

import sys
import rospy
from std_msgs.msg import String

class CtrNodeNetwork:
    def __init__(self):
        rospy.init_node("control_node_network")

        self.num_trials = 0
        self.max_trials = 250
        self.rate = rospy.Rate(10)
        
        rospy.Subscriber("experiments/finished", String, self.finished_cb)

    def finished_cb(self, msg):
        self.num_trials = self.num_trials + 1
        rospy.loginfo("[Control node] {} out of {} trials".format(self.num_trials, self.max_trials))

    def loop(self):
        while not rospy.is_shutdown():
            print (sys.version)
            self.rate.sleep()

if __name__ == '__main__':
    n = CtrNodeNetwork()
    n.loop()