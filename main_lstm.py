import os
import csv
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from Lstm import wrist_forcaster

def main_lstm(device, input_his_l, target_his_l, loader, val_loader, save_file,
                     num_epochs=100, lr=1e-3, batch_size=64, omega=1.0, rho=1.0):
    
    best_val = float("inf")
    model = wrist_forcaster( device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_epochs,eta_min=1e-6)
    criterion_mse = nn.MSELoss()
    criterion_nll = nn.GaussianNLLLoss()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = os.path.join(
        os.path.dirname(__file__), f"train_logs/lstm_training_losses_{run_id}.csv"
    )
    log_buffer = []
    log_columns = [
        "epoch",
        "train_loss", "train_nll", "train_rmse_mm",
        "val_loss", "val_nll", "val_rmse_mm", "baseline_rmse_mm",
    ]

    for epoch in range(num_epochs):
        running = train_mse_total = train_nll_total = 0.0
        model.train()
        for xb, yr, yb in loader:  
            xb = xb.permute(1, 0, 2).to(device)
            yb = yb.permute(1, 0, 2).to(device)
            yr = yr.permute(1, 0, 2).to(device)
            optimizer.zero_grad()
            pred = model(xb, yr) 
            loss_mse = criterion_mse(pred[:, :, :3], yb)  
            loss_nll = criterion_nll(pred[:, :, :3],yb,torch.exp(pred[:, :,3:]))
            loss = omega*loss_mse + rho*loss_nll
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = xb.shape[1]  # LSTM batches are [time, batch, features]
            running += loss.item() * count
            train_mse_total += loss_mse.item() * count
            train_nll_total += loss_nll.item() * count

        train_loss = running / len(loader.dataset)
        # scheduler.step()

        # validation: no teacher forcing, no gradients
        model.eval()
        vrun, brun = 0.0, 0.0
        mse_run, nll_run = 0.0, 0.0
        with torch.no_grad():
            for xb, robotb, yb in val_loader:
                xb = xb.permute(1, 0, 2).to(device)
                robotb = robotb.permute(1, 0, 2).to(device)
                yb = yb.permute(1, 0, 2).to(device)
                pred = model(xb, robotb)

                mu = pred[:, :, :3]
                var = torch.exp(pred[:, :, 3:])
                val_mse = criterion_mse(mu, yb)
                val_nll = criterion_nll(mu, yb, var)
                val_total = omega * val_mse + rho * val_nll

                batch_size = xb.shape[1]
                mse_run += val_mse.item() * batch_size
                nll_run += val_nll.item() * batch_size
                vrun += val_total.item() * batch_size
                
                baseline = xb[-1].unsqueeze(0).expand_as(yb)
                brun += criterion_mse(baseline, yb).item() * batch_size

        val_loss = vrun / len(val_loader.dataset)
        val_mse  = mse_run / len(val_loader.dataset)
        val_nll  = nll_run / len(val_loader.dataset)
        base_loss = brun / len(val_loader.dataset)

        print(
            f"epoch {epoch + 1} | train loss {train_loss:.6f} | "
            f"val loss {val_loss:.6f} | val MSE {val_mse:.6f} | "
            f"val NLL {val_nll:.6f} | "
            f"val RMSE {val_mse**0.5:.4f} m | "
            f"baseline RMSE {base_loss**0.5:.4f} m"
        )

        train_count = len(loader.dataset)
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

