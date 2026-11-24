from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_process import data_process, data_process_robot
from Transformer import WristForecaster

def main_test_transformer(test_csv, robot_test_csv, checkpoint ,history_len, horizon, skip, stride_, omega=1.0, rho=1.0):

    window_args = dict(
        skip_frame=skip,
        input_history=history_len,
        final_target=horizon,
        stride=stride_,
    )

    _, test_rw, human_ep_id = data_process(test_csv, **window_args)
    test_robot, robot_ep_id = data_process_robot(robot_test_csv, **window_args)

    if not np.array_equal(human_ep_id, robot_ep_id):
        raise ValueError("Human and robot test windows are out of alignment.")

    x = torch.from_numpy(test_rw[:, :history_len])
    robot_future = torch.from_numpy(test_robot[:, history_len:])
    y = torch.from_numpy(test_rw[:, history_len:])

    test_loader = DataLoader(
        TensorDataset(x, robot_future, y),
        batch_size=64,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WristForecaster(history_len=history_len, horizon=horizon).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()

    criterion_mse = nn.MSELoss()
    criterion_nll = nn.GaussianNLLLoss()
    mse_sum = nll_sum = baseline_sum = 0.0

    print(f"Using {device} | test windows: {len(x)}")

    with torch.no_grad():
        for xb, robotb, yb in test_loader:
            xb, robotb, yb = xb.to(device), robotb.to(device), yb.to(device)
            mean, variance = model(xb, robotb)
            count = xb.size(0)

            mse_sum += criterion_mse(mean, yb).item() * count
            nll_sum += criterion_nll(mean, yb, variance).item() * count

            baseline = xb[:, -1:].expand_as(yb)
            baseline_sum += criterion_mse(baseline, yb).item() * count

    test_mse = mse_sum / len(x)
    test_nll = nll_sum / len(x)
    baseline_mse = baseline_sum / len(x)
    test_loss = omega * test_mse + rho * test_nll

    print(
        f"test loss {test_loss:.6f} | "
        f"test MSE {test_mse:.6f} | "
        f"test NLL {test_nll:.6f} | "
        f"test RMSE {1000 * test_mse**0.5:.2f} mm | "
        f"baseline RMSE {1000 * baseline_mse**0.5:.2f} mm"
    )


if __name__ == "__main__":
    main_test_transformer()