"""
GAIN (Generative Adversarial Imputation Network) Training for AKI Cohort (v2.1)

v2.1: Adds professional TensorBoard logging to track training performance.
"""
# --- 1. Imports ---
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import os
import sys

# --- Path Setup ---
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
except NameError:
    sys.path.append(os.path.abspath('..'))

from src.models.gain import Generator, Discriminator
from sklearn.preprocessing import MinMaxScaler

# --- 2. Configuration ---
class Config:
    PROCESSED_DATA_DIR = "data/processed/aki"
    MODEL_DIR = "models"
    LOG_DIR = "logs_gain_aki" # Directory to save TensorBoard logs
    TRAIN_X_FILE = os.path.join(PROCESSED_DATA_DIR, "train_X.csv")
    GENERATOR_SAVE_PATH = os.path.join(MODEL_DIR, "generator_aki.pth")
    
    BATCH_SIZE = 128
    EPOCHS = 100
    HINT_RATE = 0.9
    ALPHA = 10.0
    LEARNING_RATE = 0.001

# --- 3. Helper Functions ---
def binary_sampler(p, rows, cols):
    return np.random.binomial(1, p, (rows, cols))

# --- 4. Main Training Function ---
def train_gain_aki(config):
    print("--- Starting GAIN Training for AKI Cohort (with Logging) ---")
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # --- NEW: Initialize TensorBoard Writer ---
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    
    try:
        X_train = pd.read_csv(config.TRAIN_X_FILE)
    except FileNotFoundError:
        print(f"Error: Training data not found at {config.TRAIN_X_FILE}")
        return
        
    num_features = X_train.shape[1]
    print(f"Using {num_features} features; total rows: {len(X_train)}")

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    dataloader = DataLoader(TensorDataset(data_tensor), batch_size=config.BATCH_SIZE, shuffle=True)
    
    generator = Generator(input_dim=num_features).to(device)
    discriminator = Discriminator(input_dim=num_features).to(device)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE)
    
    D_loss_fn = nn.BCELoss()
    G_loss_fn_adv = nn.BCELoss()
    G_loss_fn_mse = nn.MSELoss()
    
    print(f"Starting GAIN training for {config.EPOCHS} epochs on device: {device}")
    
    # --- Training Loop ---
    for epoch in range(config.EPOCHS):
        total_D_loss, total_G_loss_adv, total_G_loss_mse = 0, 0, 0
        
        for batch in dataloader:
            x_batch = batch[0]
            batch_size = x_batch.shape[0]
            
            # Train Discriminator
            optimizer_D.zero_grad()
            m_tensor = torch.tensor(binary_sampler(0.5, batch_size, num_features), dtype=torch.float32).to(device)
            z_tensor = torch.tensor(np.random.uniform(0, 0.01, size=(batch_size, num_features)), dtype=torch.float32).to(device)
            x_tilde = x_batch * m_tensor + (1 - m_tensor) * z_tensor
            g_sample = generator(x_tilde, m_tensor)
            x_hat = x_batch * m_tensor + (1 - m_tensor) * g_sample
            h_tensor = torch.tensor(binary_sampler(config.HINT_RATE, batch_size, num_features), dtype=torch.float32).to(device)
            d_prob = discriminator(x_hat, h_tensor)
            D_loss = D_loss_fn(d_prob, m_tensor)
            D_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            g_sample = generator(x_tilde, m_tensor)
            x_hat = x_batch * m_tensor + (1 - m_tensor) * g_sample
            d_prob = discriminator(x_hat, h_tensor)
            G_loss_adv = G_loss_fn_adv(d_prob, 1 - m_tensor)
            G_loss_mse = G_loss_fn_mse(x_hat * (1 - m_tensor), x_batch * (1 - m_tensor))
            G_loss = G_loss_adv + config.ALPHA * G_loss_mse
            G_loss.backward()
            optimizer_G.step()
            
            total_D_loss += D_loss.item()
            total_G_loss_adv += G_loss_adv.item()
            total_G_loss_mse += G_loss_mse.item()

        # --- NEW: Log metrics to TensorBoard at the end of each epoch ---
        avg_D_loss = total_D_loss / len(dataloader)
        avg_G_adv = total_G_loss_adv / len(dataloader)
        avg_G_mse = total_G_loss_mse / len(dataloader)
        
        writer.add_scalar('Loss/Discriminator', avg_D_loss, epoch)
        writer.add_scalar('Loss/Generator_Adversarial', avg_G_adv, epoch)
        writer.add_scalar('Loss/Generator_MSE', avg_G_mse, epoch)
        
        print(f"Epoch {epoch+1}/{config.EPOCHS} | D: {avg_D_loss:.4f} | G_adv: {avg_G_adv:.4f} | G_mse: {avg_G_mse:.4f}")

    # --- Close the writer and save the model ---
    writer.close()
    torch.save(generator.state_dict(), config.GENERATOR_SAVE_PATH)
    print("\n--- GAIN Training Finished ---")
    print(f"✅ Saved AKI generator to {config.GENERATOR_SAVE_PATH}")
    print(f"📈 To view logs, run: tensorboard --logdir={config.LOG_DIR}")

if __name__ == '__main__':
    train_gain_aki(Config())