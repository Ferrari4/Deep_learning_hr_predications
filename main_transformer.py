import os
import csv
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from Transformer import WristForecaster

def main_transformer(device, input_his_l, target_his_l, loader, val_loader, save_file,
                     num_epochs=100, lr=1e-3, batch_size=64, omega=1.0, rho=1.0):

    best_val = float("inf")
    model = WristForecaster(history_len=input_his_l, horizon=target_his_l).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_mse = nn.MSELoss()
    criterion_nll = nn.GaussianNLLLoss()
   
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = os.path.join(
        os.path.dirname(__file__), f"train_logs/transformers_training_losses_{run_id}.csv"
    )
    log_buffer = []
    log_columns = [
        "epoch",
        "train_loss", "train_nll", "train_rmse_mm",
        "val_loss", "val_nll", "val_rmse_mm", "baseline_rmse_mm",
    ]

    for epoch in range(num_epochs):
        model.train()
        train_total = train_mse_total = train_nll_total = 0.0

        for xb, robotb, yb in loader:
            xb, robotb, yb = xb.to(device), robotb.to(device), yb.to(device)
            optimizer.zero_grad()

            mean, variance = model(xb, robotb)
            mse = criterion_mse(mean, yb)
            nll = criterion_nll(mean, yb, variance)
            loss = omega * mse + rho * nll

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = xb.size(0)
            train_total += loss.item() * count
            train_mse_total += mse.item() * count
            train_nll_total += nll.item() * count

        model.eval()
        mse_total = nll_total = baseline_total = 0.0

        with torch.no_grad():
            for xb, robotb, yb in val_loader:
                xb, robotb, yb = xb.to(device), robotb.to(device), yb.to(device)
                mean, variance = model(xb, robotb)
                count = xb.size(0)

                mse_total += criterion_mse(mean, yb).item() * count
                nll_total += criterion_nll(mean, yb, variance).item() * count

                # Predict that the wrist stays at its last observed position.
                baseline = xb[:, -1:].expand_as(yb)
                baseline_total += criterion_mse(baseline, yb).item() * count

        train_count = len(loader.dataset)
        val_count = len(val_loader.dataset)

        train_loss = train_total / train_count
        val_mse = mse_total / val_count
        val_nll = nll_total / val_count
        val_loss = omega * val_mse + rho * val_nll
        base_loss = baseline_total / val_count

        print(
            f"epoch {epoch + 1} | train loss {train_loss:.6f} | "
            f"val loss {val_loss:.6f} | val MSE {val_mse:.6f} | "
            f"val NLL {val_nll:.6f} | "
            f"val RMSE {val_mse**0.5:.4f} m | "
            f"baseline RMSE {base_loss**0.5:.4f} m"
        )

        log_buffer.append([
            epoch + 1,
            train_loss,
            train_nll_total / train_count,
            1000 * (train_mse_total / train_count)**0.5,
            val_loss,
            val_nll,
            1000 * val_mse**0.5,
            1000 * base_loss**0.5,
        ])

        if len(log_buffer) == 50 or epoch + 1 == num_epochs:
            with open(log_path, "a", newline="", encoding="utf-8") as log_file:
                writer = csv.writer(log_file)
                if log_file.tell() == 0:
                    writer.writerow(log_columns)
                writer.writerows(log_buffer)

            log_buffer.clear()

        if val_loss < best_val:
            best_val = val_loss
            checkpoint = save_file
            torch.save(model.state_dict(), checkpoint)
            print(f"Saved file at epoch: {epoch}")
