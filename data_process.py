import numpy as np
import pandas as pd

def episode_windows(trajectory, win, skip_frame, stride):
    sample_count = len(trajectory) // skip_frame
    points_by_offset = [
        trajectory[offset::skip_frame]
        for offset in range(skip_frame)
    ]

    windows = []
    for start in range(0, sample_count - win + 1, stride):
        for points in points_by_offset:
            windows.append(points[start : start + win])
    return windows

def data_process(save_file, skip_frame, input_history, final_target, stride):
    # Data processing 
    try:
        data = pd.read_csv(save_file)
    except:
        print("Data Not loaded !")

    lw = data[["task", "episode", "time_step","LWristOut_x", "LWristOut_y", "LWristOut_z"]]
    rw = data[["task", "episode", "time_step","RWristOut_x", "RWristOut_y", "RWristOut_z"]]

    glw = lw.groupby(["task", "episode"],sort=False)
    grw = rw.groupby(["task", "episode"],sort=False)

    win =  input_history + final_target
    train_l = []
    ep_id_l = []  
    for ep_idx, (key, pos) in enumerate(glw):
        pos = pos.sort_values("time_step")
        traj_l = pos[["LWristOut_x", "LWristOut_y", "LWristOut_z"]].to_numpy()
        windows = episode_windows(traj_l, win, skip_frame, stride)
        train_l.extend(windows)
        ep_id_l.extend([ep_idx] * len(windows))
    train_lw = np.stack(train_l).astype(np.float32) 
    ep_id_l  = np.array(ep_id_l)  
    
    train_r = []    
    for key, pos in grw:
        pos = pos.sort_values("time_step")
        traj_r = pos[["RWristOut_x", "RWristOut_y", "RWristOut_z"]].to_numpy()
        train_r.extend(episode_windows(traj_r, win, skip_frame, stride))
    train_rw = np.stack(train_r).astype(np.float32) 

    return train_lw, train_rw, ep_id_l

def data_process_robot(save_file, skip_frame, input_history, final_target, stride):
    try:
        data = pd.read_csv(save_file)
    except:
        print("Data Not loaded !")

    robot = data[["task", "mode", "episode", "time_step","Robot_11_x", "Robot_11_y", "Robot_11_z",]]

    glw = robot.groupby(["task", "episode"],sort=False)
    win =  input_history + final_target
    train_l = []
    ep_id_l = []  
    for ep_idx, (key, pos) in enumerate(glw):
        pos = pos.sort_values("time_step")
        traj_l = pos[["Robot_11_x", "Robot_11_y", "Robot_11_z"]].to_numpy()
        windows = episode_windows(traj_l, win, skip_frame, stride)
        train_l.extend(windows)
        ep_id_l.extend([ep_idx] * len(windows))
    train_lw = np.stack(train_l).astype(np.float32) 
    ep_id_l  = np.array(ep_id_l)  
    
    return train_lw, ep_id_l