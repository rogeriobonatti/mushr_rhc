import numpy as np

def norm_angle(angle):
    # normalize all actions
    act_max = 0.38
    act_min = -0.38
    return 2.0 * (angle - act_min) / (act_max - act_min) -1.0

def denorm_angle(angle):
    # denormalize all actions
    act_max = 0.38
    act_min = -0.38
    return (angle*(act_max-act_min)+act_max+act_min)/2

def load_params(data_col):
    condition = (data_col[1:]<12.0) & (data_col[1:]>0.5) & (~np.isnan(data_col[1:]))
    ok_R = np.extract(condition, data_col[1:])
    num_points = ok_R.shape[0]
    # angles = np.linspace(0, 2*np.pi, 720)*-1.0 + np.pi # aligned in car coordinate frame (because ydlidar points backwards)
    angles = np.linspace(0, 2*np.pi, 720)
    ok_angles = np.extract(condition, angles)
    original_points = np.zeros(shape=(num_points, 3), dtype=float) # leave z as zero always, just change X and Y next
    # car coord points x forward, y right. car front points up in the picture
    # ydlidar has zero deg pointing backwards in the car, and angle grows clock-wise
    # original_points[:,0] = -np.cos(ok_angles)*ok_R # X
    # original_points[:,1] = -np.sin(ok_angles)*ok_R # Y
    original_points[:,0] = -np.cos(ok_angles)*ok_R # X
    original_points[:,1] = np.sin(ok_angles)*ok_R # Y
    voxel_size = 0.1
    range_x = 10.0
    range_y = 10.0
    range_z = voxel_size/2.0
    pc_range = np.array([-range_x, -range_y, -range_z, range_x, range_y, range_z], dtype=float)
    lo_occupied = np.log(0.7 / (1 - 0.7))
    lo_free = np.log(0.4 / (1 - 0.4))
    sensor_origins = np.tile(np.array([range_x, range_y, range_z]) , (num_points, 1)).astype(float)
    # original_points = original_points + sensor_origins # add sensor origin offset in the map 
    time_stamps = np.repeat(data_col[0], num_points).astype(float)
    return original_points, sensor_origins, time_stamps, pc_range, voxel_size, lo_occupied, lo_free

def compute_bev_image(original_points, sensor_origins, time_stamps, pc_range, voxel_size):
    nx = int(np.floor((pc_range[3]-pc_range[0])/voxel_size))
    ny = int(np.floor((pc_range[4]-pc_range[1])/voxel_size))
    vis_mat = np.zeros(shape=(ny, nx))
    original_points_idx = np.floor(original_points / voxel_size).astype(int) # becomes relative indexes instead of meters
    # transform from car-centered pixels towards standard image reference frame on upper left corner
    # Y is rows, down and X is cols, to the right
    points_vis_idx = np.zeros(shape=original_points_idx.shape, dtype=int)
    points_vis_idx[:,0] = int(ny/2)-original_points_idx[:,0] # y dir in image
    points_vis_idx[:,1] = int(nx/2)+original_points_idx[:,1] # x dir in image
    # remove indexes out of bounds for the image
    filtered_points_idx = points_vis_idx[(points_vis_idx[:,0]>0) & \
                                         (points_vis_idx[:,0]<=(ny-1)) & \
                                         (points_vis_idx[:,1]>0) & \
                                         (points_vis_idx[:,1]<=(nx-1))] 
    for p in filtered_points_idx:
        vis_mat[p[0], p[1]] = 1.0
    return vis_mat, nx, ny

def process_episode(input):
    episode_name = input[0]
    episode_folder = input[1]
    print("Processing episode: " + episode_name)
    data = np.load(episode_name)
    times = data['ts']
    actions = data['angles']
    lidars = data['lidars']
    # poses = data['poses']
    # goal = data['goal']
    num_images = times.shape[0]
    data = np.concatenate((times, actions, lidars), axis=1)
    for i in range(num_images):
        original_points, sensor_origins, time_stamps, pc_range, voxel_size, lo_occupied, lo_free = load_params(data[i, :])
        # compute visibility
        vis_mat, nx, ny = compute_bev_image(original_points, sensor_origins, time_stamps, pc_range, voxel_size)
        vis_mat = vis_mat*127+127