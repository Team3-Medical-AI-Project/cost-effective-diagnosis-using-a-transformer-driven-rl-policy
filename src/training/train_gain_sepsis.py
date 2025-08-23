"""
GAIN Training Script for Sepsis Cohort (v3.0 - Final with Validation)

Description:
This definitive version includes a validation loop and early stopping. It
evaluates the imputer's performance (RMSE) on a holdout set at each epoch
and saves only the best-performing model, ensuring optimal accuracy.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import os
import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
except NameError:
    sys.path.append(os.path.abspath('..'))
from src.models.gain import Generator, Discriminator

# --- Configuration ---
CONFIG = {
    "INPUT_FILE": "data/preprocessed/sepsis_feature_matrix.csv",
    "OUTPUT_MODEL_FILE": "models/generator_sepsis.pth",
    "ID_COLUMNS": ['subject_id', 'hadm_id', 'stay_id'],
    "TARGET_COLUMN": "hospital_expire_flag",
    "HINT_RATE": 0.9,
    "ALPHA": 10.0,
    "BATCH_SIZE": 128,
    "EPOCHS": 100 # We still set a max, but will save the best model found within this
}

def binary_sampler(p, rows, cols):
    return np.random.binomial(1, p, (rows, cols))

def train_gain_sepsis(config):
    """Main training function with a validation loop."""
    print("--- Starting GAIN Training Pipeline for Sepsis (with Validation) ---")
    
    # --- 1. Load and Prepare Data ---
    full_df = pd.read_csv(config["INPUT_FILE"])
    feature_cols = [col for col in full_df.columns if col not in config["ID_COLUMNS"] + [config["TARGET_COLUMN"]]]
    complete_df = full_df[feature_cols].dropna()
    print(f"Found {len(complete_df)} complete rows for training and validation.")

    # --- NEW: Split complete data into training and validation sets ---
    train_df, val_df = train_test_split(complete_df, test_size=0.2, random_state=42)
    
    # --- 2. Scale Data ---
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df) # Use the same scaler for validation
    
    # --- 3. Initialize Models and Dataloader ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = TensorDataset(torch.from_numpy(train_scaled).float().to(device))
    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True)
    val_tensor = torch.from_numpy(val_scaled).float().to(device)
    
    input_dim = len(feature_cols)
    generator = Generator(input_dim=input_dim).to(device)
    discriminator = Discriminator(input_dim=input_dim).to(device)
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    
    D_loss_fn = nn.BCELoss()
    G_loss_fn_adv = nn.BCELoss()
    G_loss_fn_mse = nn.MSELoss()

    print(f"Starting GAIN training for up to {config['EPOCHS']} epochs on device: {device}")

    # --- 4. Training Loop with Validation ---
    best_val_rmse = float('inf') # We want to minimize the RMSE

    for epoch in range(config["EPOCHS"]):
        generator.train() # Set models to training mode
        discriminator.train()
        
        for batch in train_loader:
            x_batch = batch[0]
            batch_size = x_batch.shape[0]
            
            mask = torch.tensor(binary_sampler(0.5, batch_size, input_dim), dtype=torch.float32).to(device)
            noise = torch.rand(x_batch.shape, device=device)
            hint = torch.tensor(binary_sampler(config["HINT_RATE"], batch_size, input_dim), dtype=torch.float32).to(device)
            hint = mask * hint
            corrupted_x = x_batch * mask + noise * (1 - mask)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            imputed_data = generator(corrupted_x, mask)
            d_input = imputed_data.detach() # Detach to prevent gradients flowing to generator
            d_prob = discriminator(d_input, hint)
            d_loss = D_loss_fn(d_prob, mask)
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            imputed_data_g = generator(corrupted_x, mask)
            d_prob_g = discriminator(imputed_data_g, hint)
            g_adversarial_loss = G_loss_fn_adv(d_prob_g, mask)
            mse_reconstruction_loss = G_loss_fn_mse(x_batch * mask, imputed_data_g * mask)
            g_loss = g_adversarial_loss + config["ALPHA"] * mse_reconstruction_loss
            g_loss.backward()
            optimizer_G.step()
        
        # --- NEW: Validation Step ---
        generator.eval() # Set generator to evaluation mode
        with torch.no_grad():
            # Artificially mask the validation data
            val_mask_np = binary_sampler(0.5, val_tensor.shape[0], input_dim)
            val_mask = torch.tensor(val_mask_np, dtype=torch.float32).to(device)
            val_noise = torch.rand(val_tensor.shape, device=device)
            corrupted_val = val_tensor * val_mask + val_noise * (1 - val_mask)
            
            # Impute the corrupted validation data
            imputed_val = generator(corrupted_val, val_mask)
            
            # Calculate RMSE only on the values that were imputed
            # This is the true measure of imputation accuracy
            mse_val_loss = G_loss_fn_mse(val_tensor * (1 - val_mask), imputed_val * (1 - val_mask))
            val_rmse = torch.sqrt(mse_val_loss).item()
        
        print(f"Epoch {epoch+1}/{config['EPOCHS']} | Validation RMSE: {val_rmse:.4f}")
        
        # Early stopping: save the model only if validation performance improves
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            output_path = Path(config["OUTPUT_MODEL_FILE"])
            output_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(generator.state_dict(), output_path)
            print(f"  -> New best model found! Saved to '{output_path}'")

    print("\n--- GAIN Training Finished ---")
    print(f"✅ Best Validation RMSE: {best_val_rmse:.4f}")

if __name__ == '__main__':
    train_gain_sepsis(CONFIG)