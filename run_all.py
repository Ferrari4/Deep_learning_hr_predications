import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from data_process import data_process, data_process_robot
from main_transformer import main_transformer
from main_lstm import main_lstm
from test_transformer import main_test_transformer
from test_lstm import main_test_lstm

def main():
    # Data Split
    skip, stride_ = 4, 12
    input_his_l = 40
    target_his_l = 16
   
    # Network
    batch_size = 64
    epoch = 500
    lr = 1e-3
    batch = 64

    # Save trained model
    run_tag = "lr03_ss_4_12"

    # Do not change
    code_dir = Path(__file__).resolve().parent
    models_dir = code_dir / "save_models"
    train_dir = code_dir / "data_train"
    test_dir = code_dir / "data_test"
    checkpoint_transformer = models_dir / f"transformer_wrist_{run_tag}.pt"
    checkpoint_lstm = models_dir / f"lstm_wrist_{run_tag}.pt"
    csv_path_human_train = train_dir / "trajectories_train_HR.csv"
    csv_path_robot_train = train_dir / "trajectories_robot_train_HR.csv"
    csv_path_human_test = test_dir / "trajectories_test_HR.csv"
    csv_path_robot_test = test_dir / "trajectories_robot_test_HR.csv"
    
    train_lw, train_rw, ep_id = data_process(
                                        csv_path_human_train, 
                                        skip_frame=skip, 
                                        input_history=input_his_l, 
                                        final_target=target_his_l, 
                                        stride=stride_)

    train_robot, robot_ep_id  = data_process_robot(
                                        csv_path_robot_train,
                                        skip_frame=skip,
                                        input_history=input_his_l,
                                        final_target=target_his_l,
                                        stride=stride_)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    rng = np.random.default_rng(0)
    assert np.array_equal(ep_id, robot_ep_id)
    all_eps = np.unique(ep_id)
    val_eps = rng.choice(all_eps, size=len(all_eps) // 10, replace=False)
    val_mask = np.isin(ep_id, val_eps)

    val_rw = train_rw[val_mask]
    train_rw = train_rw[~val_mask]

    x = torch.from_numpy(train_rw[:, :input_his_l])
    y = torch.from_numpy(train_rw[:, input_his_l:])
    xv = torch.from_numpy(val_rw[:, :input_his_l])
    yv = torch.from_numpy(val_rw[:, input_his_l:])

    val_robot = train_robot[val_mask]
    train_robot = train_robot[~val_mask]

    xr = torch.from_numpy(train_robot[:, :input_his_l])
    yr = torch.from_numpy(train_robot[:, input_his_l:])
    xvr = torch.from_numpy(val_robot[:, :input_his_l])
    yvr = torch.from_numpy(val_robot[:, input_his_l:])

    loader = DataLoader(
    TensorDataset(x, yr, y),
    batch_size=batch_size,
    shuffle=True,
    
    )
    
    val_loader = DataLoader(
        TensorDataset(xv, yvr, yv),
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"Using {device} | train windows: {len(x)} | val windows: {len(xv)}")

    # Train code
    main_transformer(device, input_his_l, target_his_l, loader, val_loader, checkpoint_transformer,
                     num_epochs=epoch, lr=lr, batch_size=batch, omega=1.0, rho=1.0)

    print("-------------------------------------- END OF TRAIN LOOP Transformer -------------------------------")
    
    main_lstm(device, input_his_l, target_his_l, loader, val_loader, checkpoint_lstm,
                     num_epochs=epoch, lr=lr, batch_size=batch, omega=1.0, rho=1.0)

    print("-------------------------------------- END OF TRAIN LOOP LSTM -------------------------------")

    # Test code 
    main_test_transformer(csv_path_human_test, csv_path_robot_test, checkpoint_transformer, input_his_l, target_his_l, skip, stride_, omega=1.0, rho=1.0)

    print("-------------------------------------- END OF TEST LOOP Transformer -------------------------------")

    main_test_lstm(csv_path_human_test, csv_path_robot_test, checkpoint_lstm, input_his_l, target_his_l, skip, stride_, omega=1.0, rho=1.0)

    print("-------------------------------------- END OF TEST LOOP LSTM -------------------------------")

if __name__ == "__main__":
    main()
