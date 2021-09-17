import rosbag
import os
import glob
import numpy as np

def extract_bag_variables(bag_name, dataset_path):
    started_episode = False
    started_moving = False
    bag = rosbag.Bag(bag_name)
    # initialize data structures of the correct size
    finished = np.zeros((bag.get_message_count('experiments/finished'),2), dtype=np.float)
    angles = np.zeros((bag.get_message_count('mux/ackermann_cmd_mux/input/navigation'),2), dtype=np.float)
    active = np.zeros((bag.get_message_count('mux/ackermann_cmd_mux/active'),2), dtype=np.float)
    scans = np.zeros((bag.get_message_count('scan'),721), dtype=np.float)
    idx_dict = {
        "experiments/finished": 0,
        "mux/ackermann_cmd_mux/input/navigation": 0,
        "mux/ackermann_cmd_mux/active": 0,
        "scan": 0
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
        "idle": 2
    }
    # fill the data structures
    for topic, msg, t in bag.read_messages(topics=['experiments/finished', 
                                                   'mux/ackermann_cmd_mux/input/navigation',
                                                   'mux/ackermann_cmd_mux/active',
                                                   'scan']):
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
        idx_dict[topic] += 1
    bag.close()
    return finished, angles, active, scans


# define script parameters
bags_location = '/home/rb/recordings'
output_folder_name = 'process_0'
dataset_base = '/home/rb/data'

# check if output folders already exists
dataset_path = os.path.join(dataset_base, output_folder_name)
if not os.path.isdir(dataset_path):
    os.makedirs(dataset_path)
else:
    print('Warning: path already exists. May overrite things')

print("Going to process all bags in directory: " + bags_location)
files_list = sorted(glob.glob(os.path.join(bags_location, '*.bag')))
total_size = len(files_list)
print("Total number of files to be processed = {}".format(total_size))

ep_num = 0
for num, file in enumerate(files_list, 1):
        print("Beginning file number = {} out of {}: {}%".format(num, total_size, int(100.0*num/total_size)))
        finished, angles, active, scans = extract_bag_variables(file, dataset_path)
        t_start = None
        t_end = None
        for count, (t, event) in enumerate(finished):
            print("Event = {} out of {}: {}%".format(count, finished.shape[0], int(100.0*count/finished.shape[0])))
            if t_start is None:
                t_start = t
                continue
            else:
                t_end = t
                # collect appropriate data
                if int(event)==0 or int(event)==3:
                    # if it's here, in this episode it reached goal or timed out
                    # both are ok, so we process the episode
                    # 1) get the time intervals when we have commands
                    e_angles = angles[(angles[:,0]>=t_start) & (angles[:,0]<t_end)]
                    # catch condition where it terminates too early because of some error
                    if e_angles.shape[0]>100:
                        t_angles = [e_angles[0,0], e_angles[-1,0]]
                        # 2) get lidar scans within this interval
                        e_scans = scans[(scans[:,0]>=t_angles[0]) & (scans[:,0]<t_angles[1])]
                        # 3) for each scan, find the appropriate angle label (don't iterate over last scan of episode)
                        episode_data = np.zeros((e_scans.shape[0]-1, 722))
                        for idx in range(e_scans.shape[0]-1):
                            t_scan = [e_scans[idx,0], e_scans[idx+1,0]]
                            angle = np.mean(e_angles[(e_angles[:,0]>=t_scan[0]) & (e_angles[:,0]<t_scan[1])][:,1])
                            episode_data[idx, 0] = e_scans[idx, 0]
                            episode_data[idx, 1] = angle
                            episode_data[idx, 2:] = e_scans[idx, 1:]
                        # 4) save the data as a separate episode in a np variable
                        filename = os.path.join(dataset_path, 'ep'+str(ep_num))
                        np.save(filename, episode_data)
                        ep_num += 1
                else:
                    # don't save this data because there was some issue with this episode
                    pass
                # re-set the start time
                t_start = t_end
