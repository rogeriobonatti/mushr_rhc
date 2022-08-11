import rosbag
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def extract_bag_variables(bag_name):
    bag = rosbag.Bag(bag_name)
    # initialize data structures of the correct size

    poses = np.zeros((bag.get_message_count('car_pose'),3), dtype=np.float)
    idx_dict = {
        "experiments/finished": 0,
        "mux/ackermann_cmd_mux/input/navigation": 0,
        "mux/ackermann_cmd_mux/active": 0,
        "scan": 0,
        "car_pose": 0
    }
    # fill the data structures
    for topic, msg, t in bag.read_messages(topics=['car_pose']):
        idx = idx_dict[topic]
        poses[idx, 0] = msg.header.stamp.to_sec()
        poses[idx, 1] = msg.pose.position.x
        poses[idx, 2] = msg.pose.position.y
        idx_dict[topic] += 1
        
    bag.close()
    return poses

def plot_poses(bag_location):
    bag_fname = bag_location.split(".")[0]
    poses = extract_bag_variables(bag_location)

    cmap_list = [
                # "crest", 
                # "flare", 
                # "magma", 
                "viridis", 
                "rocket_r", 
                # "cubehelix"
                ]

    resolution = 0.05 
    num_pixels_per_meter = 1. / resolution
    origin = np.array([-32.925,-37.3])
    origin_pixel = origin * num_pixels_per_meter 

    poses = poses[:,1:] 
    poses *= num_pixels_per_meter
    np.swapaxes(poses, 0,1) # swap x and y
    poses *= np.array([1, -1]) # flip y 
    poses -= origin_pixel # translate 
    img = plt.imread('/home/rb/hackathon_data_premium/hackathon_data_2p5_nonoise3/bravern_floor.png')
    alpha = 0.66
    traj_color = "magenta"

    for cmap in cmap_list:
        plt.figure()
        plt.imshow(img, cmap=plt.cm.gray, alpha=alpha)
        res=sns.kdeplot(poses[:,0], poses[:,1], cmap="viridis", shade=True, fill=True, alpha=alpha)
        plt.axis('off')
        plt.savefig("/home/rb/data/0.jpg", bbox_inches = 'tight', dpi=1200)

        # plt.figure()
        # plt.imshow(img, cmap=plt.cm.gray)
        # res=sns.kdeplot(poses[:,0], poses[:,1], cmap=cmap, fill=False, alpha=alpha, linewidth=0.25)
        # plt.axis('off')
        # plt.savefig("/home/rb/data/1.jpg", bbox_inches = 'tight', dpi=1200)

        # plt.figure()
        # plt.imshow(img, cmap=plt.cm.gray)
        # plt.scatter(poses[:,0], poses[:,1], s=0.25, marker='o', c=traj_color)
        # plt.axis('off')
        # plt.savefig("/home/rb/data/2.jpg", bbox_inches = 'tight', dpi=1200)

        # plt.figure()
        # plt.imshow(img, cmap=plt.cm.gray)
        # plt.scatter(poses[:,0], poses[:,1], s=0.25, marker='o', c=traj_color)
        # res=sns.kdeplot(poses[:,0], poses[:,1], cmap=cmap, fill=True, alpha=alpha,  linewidth=0.25)
        # plt.axis('off')
        # plt.savefig("/home/rb/data/3.jpg", bbox_inches = 'tight', dpi=1200)

        # plt.figure()
        # plt.imshow(img, cmap=plt.cm.gray)
        # plt.scatter(poses[:,0], poses[:,1], s=0.25, marker='o', c=traj_color)
        # res=sns.kdeplot(poses[:,0], poses[:,1], cmap=cmap, fill=False, alpha=alpha, linewidth=0.25)
        # plt.axis('off')
        # plt.savefig("/home/rb/data/4.jpg", bbox_inches = 'tight', dpi=1200)

def main():
    bag_list = ["/home/rb/data/data.bag"]
    # bag_list = ["/home/rb/data/data_bias.bag"]

    for bag in bag_list:
        plot_poses(bag)

if __name__=="__main__":
    main()