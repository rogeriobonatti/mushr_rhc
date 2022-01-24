import rosbag
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def extract_bag_variables(bag_name):
    bag = rosbag.Bag(bag_name)
    # initialize data structures of the correct size
    finished = np.zeros((bag.get_message_count('experiments/finished'),2), dtype=np.float)
    angles = np.zeros((bag.get_message_count('mux/ackermann_cmd_mux/input/navigation'),2), dtype=np.float)
    active = np.zeros((bag.get_message_count('mux/ackermann_cmd_mux/active'),2), dtype=np.float)
    scans = np.zeros((bag.get_message_count('scan'),721), dtype=np.float)
    poses = np.zeros((bag.get_message_count('particle_filter/inferred_pose'),3), dtype=np.float)
    idx_dict = {
        "experiments/finished": 0,
        "mux/ackermann_cmd_mux/input/navigation": 0,
        "mux/ackermann_cmd_mux/active": 0,
        "scan": 0,
        "particle_filter/inferred_pose": 0
    }
    # fill the data structures
    for topic, msg, t in bag.read_messages(topics=['scan', 'particle_filter/inferred_pose']):
        idx = idx_dict[topic]
        if topic == 'experiments/finished':
            finished[idx, 0] = t.to_sec()
            finished[idx, 1] = finished_dict[msg.data]     
        elif topic == 'mux/ackermann_cmd_mux/input/navigation':
            angles[idx, 0] = t.to_sec()
            angles[idx, 1] = msg.drive.steering_angle
        elif topic == 'mux/ackermann_cmd_mux/active':
            active[idx, 0] = t.to_sec()
            active[idx, 1] = active_dict[msg.data]
        elif topic == 'scan':
            scans[idx, 0] = t.to_sec()
            scans[idx, 1:] = msg.ranges
        elif topic == 'particle_filter/inferred_pose':
            poses[idx, 0] = msg.header.stamp.to_sec()
            poses[idx, 1] = msg.pose.position.x
            poses[idx, 2] = msg.pose.position.y
        idx_dict[topic] += 1
        
    bag.close()
    return finished, angles, active, scans, poses


# define script parameters
bag_location = '/home/rb/Downloads/bravern0.bag'
finished, angles, active, scans, poses = extract_bag_variables(bag_location)

v = np.zeros(shape=(poses.shape[0]-1,2))
N=3
t = np.convolve(poses[:,1], np.ones(N)/N, mode='valid')
x = np.convolve(poses[:,1], np.ones(N)/N, mode='valid')
y = np.convolve(poses[:,1], np.ones(N)/N, mode='valid')
for i in range(t-1):
    delta_t = poses[i+1,0]-poses[i,0]
    delta_x = poses[i+1,1]-poses[i,1]
    delta_y = poses[i+1,2]-poses[i,2]
    v[i,0]= poses[i,0]
    v[i,1]= np.sqrt(delta_x*delta_x+delta_y*delta_y)/delta_t



for i in range(poses.shape[0]-1):
    delta_t = poses[i+1,0]-poses[i,0]
    delta_x = poses[i+1,1]-poses[i,1]
    delta_y = poses[i+1,2]-poses[i,2]
    v[i,0]= poses[i,0]
    v[i,1]= np.sqrt(delta_x*delta_x+delta_y*delta_y)/delta_t

plt.plot(v[:,0], v[:,1])
plt.show()

bla=1