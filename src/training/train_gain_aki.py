"""
GAIN (Generative Adversarial Imputation Network) Training for AKI Cohort (stable logits version)

- Uses BCEWithLogitsLoss + logits discriminator for numerical stability
- Proper GAIN hint construction: H = M ⊙ B + 0.5*(1-B)
- Sanitizes NaNs/Infs before feeding the discriminator
- TensorBoard logging preserved
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler
import joblib

# Path setup so "src" is importable
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
except NameError:
    sys.path.append(os.path.abspath('..'))

from src.models.gain import Generator, Discriminator


class Config:
    PROCESSED_DATA_DIR = "data/processed/aki"
    MODEL_DIR = "models"
    LOG_DIR = "logs_gain_aki"

    TRAIN_X_FILE = os.path.join(PROCESSED_DATA_DIR, "train_X.csv")
    GENERATOR_SAVE_PATH = os.path.join(MODEL_DIR, "generator_aki.pth")
    GAIN_SCALER_SAVE_PATH = os.path.join(MODEL_DIR, "gain_scaler_aki.joblib") 

    BATCH_SIZE = 128
    EPOCHS = 100
    HINT_RATE = 0.9      # p(B=1)
    MASK_MISS_P = 0.5    # p(M=1) during training corruption
    ALPHA = 10.0
    LEARNING_RATE = 1e-3
    NUM_WORKERS = 0      # dataloader workers (set >0 if you like)


def binary_sampler(p, rows, cols):
    return np.random.binomial(1, p, size=(rows, cols))


def train_gain_aki(cfg: Config):
    print("--- Starting GAIN Training for AKI Cohort (stable logits) ---")
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    # Load data
    try:
        X_train = pd.read_csv(cfg.TRAIN_X_FILE)
    except FileNotFoundError:
        print(f"Error: Training data not found at {cfg.TRAIN_X_FILE}")
        return

    num_features = X_train.shape[1]
    print(f"Using {num_features} features; total rows: {len(X_train)}")

    # Scale to [0,1] (GAIN expects normalized features)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.values.astype(np.float32))
    joblib.dump(scaler, Config.GAIN_SCALER_SAVE_PATH)  # NEW
    print(f"Saved GAIN MinMax scaler to {Config.GAIN_SCALER_SAVE_PATH}")  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=cfg.LOG_DIR)

    data_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    loader = DataLoader(TensorDataset(data_tensor),
                        batch_size=cfg.BATCH_SIZE,
                        shuffle=True,
                        num_workers=cfg.NUM_WORKERS,
                        pin_memory=(device.type == "cuda"))

    G = Generator(input_dim=num_features).to(device)
    D = Discriminator(input_dim=num_features).to(device)

    opt_G = optim.Adam(G.parameters(), lr=cfg.LEARNING_RATE)
    opt_D = optim.Adam(D.parameters(), lr=cfg.LEARNING_RATE)

    # Logits-version losses
    bce_logits = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    print(f"Training for {cfg.EPOCHS} epochs on device: {device}")

    for epoch in range(cfg.EPOCHS):
        G_adv_loss_sum = 0.0
        G_mse_loss_sum = 0.0
        D_loss_sum = 0.0
        nb = 0

        for (x_batch_cpu,) in loader:
            nb += 1
            x_batch = x_batch_cpu.to(device, non_blocking=True)        # in [0,1]

            # --- Build random corruption mask M (1=observed, 0=missing) ---
            M_np = binary_sampler(cfg.MASK_MISS_P, x_batch.size(0), num_features).astype(np.float32)
            M = torch.from_numpy(M_np).to(device)

            # --- Noisy placeholder for missing entries Z ~ U(0, 0.01) ---
            Z = torch.rand_like(x_batch) * 0.01

            # --- Corrupt input ---
            X_tilde = x_batch * M + (1 - M) * Z

            # --- Generator forward ---
            G_sample = G(X_tilde, M)                        # in [0,1]
            X_hat = x_batch * M + (1 - M) * G_sample        # composite

            # sanitize before D
            X_hat = torch.nan_to_num(X_hat, nan=0.5, posinf=1.0, neginf=0.0)
            X_hat = torch.clamp(X_hat, 0.0, 1.0)

            # --- Hint mask B ~ Bernoulli(HINT_RATE); H = M ⊙ B + 0.5*(1-B) ---
            B_np = binary_sampler(cfg.HINT_RATE, x_batch.size(0), num_features).astype(np.float32)
            B = torch.from_numpy(B_np).to(device)
            H = M * B + 0.5 * (1.0 - B)

            # ------------------ Train Discriminator ------------------
            opt_D.zero_grad(set_to_none=True)
            D_logits = D(X_hat.detach(), H)                 # logits, shape = (B, D)
            # Targets are M (0/1). BCEWithLogitsLoss expects logits.
            D_loss = bce_logits(D_logits, M)
            D_loss.backward()
            opt_D.step()

            # ------------------ Train Generator ---------------------
            opt_G.zero_grad(set_to_none=True)

            # Recompute with fresh forward (not strictly necessary)
            G_sample = G(X_tilde, M)
            X_hat = x_batch * M + (1 - M) * G_sample
            X_hat = torch.nan_to_num(X_hat, nan=0.5, posinf=1.0, neginf=0.0)
            X_hat = torch.clamp(X_hat, 0.0, 1.0)

            D_logits = D(X_hat, H)

            # Adversarial: M vs D_logits (we want D to predict "missing" where it is actually missing)
            G_loss_adv = bce_logits(D_logits, 1.0 - M)

            # Reconstruction MSE on *missing* entries only
            G_loss_mse = mse(X_hat * (1 - M), x_batch * (1 - M))

            G_loss = G_loss_adv + cfg.ALPHA * G_loss_mse
            G_loss.backward()
            opt_G.step()

            # Accumulate logs
            D_loss_sum += float(D_loss.item())
            G_adv_loss_sum += float(G_loss_adv.item())
            G_mse_loss_sum += float(G_loss_mse.item())

        # Epoch logs
        D_loss_avg = D_loss_sum / max(1, nb)
        G_adv_avg = G_adv_loss_sum / max(1, nb)
        G_mse_avg = G_mse_loss_sum / max(1, nb)

        writer.add_scalar('Loss/Discriminator', D_loss_avg, epoch)
        writer.add_scalar('Loss/Generator_Adversarial', G_adv_avg, epoch)
        writer.add_scalar('Loss/Generator_MSE', G_mse_avg, epoch)

        print(f"Epoch {epoch+1}/{cfg.EPOCHS} | D: {D_loss_avg:.4f} | G_adv: {G_adv_avg:.4f} | G_mse: {G_mse_avg:.4f}")

    writer.close()
    torch.save(G.state_dict(), cfg.GENERATOR_SAVE_PATH)
    print("\n--- GAIN Training Finished ---")
    print(f"✅ Saved AKI generator to {cfg.GENERATOR_SAVE_PATH}")
    print(f"📈 TensorBoard: tensorboard --logdir={cfg.LOG_DIR}")


if __name__ == "__main__":
    train_gain_aki(Config())
