import rosbag
import os
import glob
import numpy as np
import utilss

def extract_bag_variables(bag_name):
    started_episode = False
    started_moving = False
    bag = rosbag.Bag(bag_name)
    # initialize data structures of the correct size
    finished = np.zeros((bag.get_message_count('experiments/finished'),2), dtype=np.float)
    angles = np.zeros((bag.get_message_count('mux/ackermann_cmd_mux/input/navigation'),2), dtype=np.float)
    active = np.zeros((bag.get_message_count('mux/ackermann_cmd_mux/active'),2), dtype=np.float)
    scans = np.zeros((bag.get_message_count('scan'),721), dtype=np.float)
    poses = np.zeros((bag.get_message_count('particle_filter/inferred_pose'),4), dtype=np.float)
    goals = np.zeros((bag.get_message_count('rhcontroller/goal'),4), dtype=np.float)
    idx_dict = {
        "experiments/finished": 0,
        "mux/ackermann_cmd_mux/input/navigation": 0,
        "mux/ackermann_cmd_mux/active": 0,
        "scan": 0,
        "particle_filter/inferred_pose": 0,
        "rhcontroller/goal": 0
    }
    finished_dict = {
        "reached goal": 0,
        "first time": 1,
        "got stuck": 2,
        "timeout": 3
    }
    active_dict = {
        "Default": 0,
        "Navigation": 1,
        "idle": 2,
        'Teleoperation': 3
    }
    # fill the data structures
    for topic, msg, t in bag.read_messages(topics=['experiments/finished', 
                                                   'mux/ackermann_cmd_mux/input/navigation',
                                                   'mux/ackermann_cmd_mux/active',
                                                   'scan',
                                                   'particle_filter/inferred_pose',
                                                   'rhcontroller/goal']):
        idx = idx_dict[topic]
        if topic == 'experiments/finished':
            finished[idx, 0] = t.to_sec()
            finished[idx, 1] = finished_dict[msg.data]     
        elif topic == 'mux/ackermann_cmd_mux/input/navigation':
            angles[idx, 0] = msg.header.stamp.to_sec()
            angles[idx, 1] = msg.drive.steering_angle
        elif topic == 'mux/ackermann_cmd_mux/active':
            active[idx, 0] = t.to_sec()
            active[idx, 1] = active_dict[msg.data]
        elif topic == 'scan':
            scans[idx, 0] = msg.header.stamp.to_sec()
            # scans[idx, 0] = t.to_sec()
            scans[idx, 1:] = msg.ranges
        elif topic == 'particle_filter/inferred_pose':
            poses[idx, 0] = msg.header.stamp.to_sec()
            poses[idx, 1] = msg.pose.position.x
            poses[idx, 2] = msg.pose.position.y
            poses[idx, 3] = utilss.rosquaternion_to_angle(msg.pose.orientation)
        elif topic == 'rhcontroller/goal':
            goals[idx, 0] = msg.header.stamp.to_sec()
            goals[idx, 1] = msg.pose.position.x
            goals[idx, 2] = msg.pose.position.y
        idx_dict[topic] += 1
    bag.close()
    return finished, angles, active, scans, poses, goals


def process_folder(folder, processed_dataset_path):
    bag_file_name = glob.glob(os.path.join(folder, '*.bag'))[0]
    finished, angles, active, scans, poses, goals = extract_bag_variables(bag_file_name)
    t_start = None
    t_end = None
    ep_num = 0
    for count, (t, event) in enumerate(finished):
        print("Event = {} out of {}: {}%".format(count, finished.shape[0], int(100.0*count/finished.shape[0])))
        if t_start is None:
            t_start = t
            continue
        else:
            t_end = t
            # collect appropriate data
            if (int(event)==0 or int(event)==3) and (t_end - t_start)>5.0:
                # if it's here, in this episode it reached goal or timed out
                # both are ok, so we process the episode
                # 0) get the goal location
                goal = goals[(goals[:,0]>=t_start) & (goals[:,0]<t_end)]
                if goal.shape[0] == 0:
                    # there's some issue with the goal of this episode, so discard
                    pass
                else:
                    # 1) get the time intervals when we have commands
                    e_angles = angles[(angles[:,0]>=t_start) & (angles[:,0]<t_end)]
                    e_poses = poses[(poses[:,0]>=t_start) & (poses[:,0]<t_end)]
                    # catch condition where it terminates too early because of some error
                    if e_angles.shape[0]>100:
                        t_angles = [e_angles[0,0], e_angles[-1,0]]
                        # 2) get lidar scans within this interval
                        e_scans = scans[(scans[:,0]>=t_angles[0]) & (scans[:,0]<t_angles[1])]
                        # 3) for each scan, find the appropriate angle label (don't iterate over last scan of episode)
                        # time 1 | action 1| lidar 720 | pose 3 | goal 2
                        # episode_data = np.zeros((e_scans.shape[0]-1, 727))
                        episode_ts = np.zeros((e_scans.shape[0]-1, 1))
                        episode_angles = np.zeros((e_scans.shape[0]-1, 1))
                        episode_lidars = np.zeros((e_scans.shape[0]-1, 720))
                        episode_poses = np.zeros((e_scans.shape[0]-1, 3))
                        for idx in range(e_scans.shape[0]-1):
                            t_scan = [e_scans[idx,0], e_scans[idx+1,0]]
                            angle = np.mean(e_angles[(e_angles[:,0]>=t_scan[0]) & (e_angles[:,0]<t_scan[1])][:,1])
                            poses_valid = e_poses[(e_poses[:,0]>=t_scan[0]) & (e_poses[:,0]<t_scan[1])]
                            pose_x = np.mean(poses_valid[:,1])
                            pose_y = np.mean(poses_valid[:,2])
                            mean_s = np.mean(np.sin(poses_valid[:,3]))
                            mean_c = np.mean(np.cos(poses_valid[:,3]))
                            mean_angle = np.arctan2(mean_s, mean_c)
                            # add all the components of the data I'm saving
                            episode_ts[idx,0] = e_scans[idx, 0]
                            episode_angles[idx,0] = angle
                            episode_lidars[idx,:] = e_scans[idx, 1:]
                            episode_poses[idx, :] = np.array([pose_x, pose_y, mean_angle])
                            # episode_data[idx, 0] = e_scans[idx, 0]
                            # episode_data[idx, 1] = angle
                            # episode_data[idx, 2:722] = e_scans[idx, 1:]
                            # episode_data[idx, 723:726] = [pose_x, pose_y, mean_angle]
                        # 4) save the data as a separate episode in a np variable
                        filename = os.path.join(processed_dataset_path, 'ep'+str(ep_num))
                        np.savez(filename, ts=episode_ts, angles=episode_angles, lidars=episode_lidars, poses=episode_poses, goal=goal[0,1:3])
                        # np.save(filename, episode_data)
                        ep_num += 1
            else:
                # don't save this data because there was some issue with this episode
                pass
            # re-set the start time
            t_start = t_end


# define script parameters
base_folder = '/home/azureuser/hackathon_data/real'
output_folder_name = 'processed_withpose2'
folders_list = sorted(glob.glob(os.path.join(base_folder, '*')))
total_n_folders = len(folders_list)
print("Total number of folders to be processed = {}".format(total_n_folders))

for i, folder in enumerate(folders_list, 1):
    print("Beginning folder {} number = {} out of {}: {}%".format(folder, i, total_n_folders, int(100.0*i/total_n_folders)))
    # create processed folder
    processed_dataset_path = os.path.join(folder, output_folder_name)
    if not os.path.isdir(processed_dataset_path):
        os.makedirs(processed_dataset_path)
        # move on to process that bag file inside the folder
        process_folder(folder, processed_dataset_path)
    else:
        print('Warning: path already exists. Skipping this folder...')

