"""
GAIN Training Script for Sepsis Cohort (v4.0 - Harmonized with StandardScaler)

Description:
This version is corrected to use the project's main StandardScaler (scaler.joblib)
instead of MinMaxScaler. This ensures that the data distribution GAIN is trained on
matches the distribution expected by the downstream classifier, fixing the
pipeline mismatch.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import os
import sys
import joblib # Using joblib to load the project's scaler

from sklearn.model_selection import train_test_split
# Ensure src is in the path to import models
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
except NameError:
    sys.path.append(os.path.abspath('..'))
from src.models.gain import Generator, Discriminator

# --- Configuration ---
CONFIG = {
    "INPUT_FILE": "data/preprocessed/sepsis_feature_matrix.csv",
    "SCALER_FILE": "data/processed/sepsis/scaler.joblib",
    "OUTPUT_MODEL_FILE": "models/generator_sepsis.pth",
    "ID_COLUMNS": ['subject_id', 'hadm_id', 'stay_id'],
    "TARGET_COLUMN": "hospital_expire_flag",
    "HINT_RATE": 0.9,
    "ALPHA": 10.0,
    "BATCH_SIZE": 128,
    "EPOCHS": 100,
    "PATIENCE": 12 # Early stopping patience
}

def binary_sampler(p, rows, cols):
    return np.random.binomial(1, p, (rows, cols))

def train_gain_sepsis_harmonized(config):
    """Main training function corrected to use StandardScaler."""
    print("--- Starting HARMONIZED GAIN Training Pipeline for Sepsis ---")
    
    # --- 1. Load Data ---
    full_df = pd.read_csv(config["INPUT_FILE"])

    # Enforce EXACT 37-feature schema to match scaler + downstream models
    SEPSIS_37 = [
        'age','gender',
        'heart_rate_mean','sbp_mean','dbp_mean','respiratory_rate_mean','spo2_mean','temperature_c_mean',
        'heart_rate_min','sbp_min','dbp_min','respiratory_rate_min','spo2_min','temperature_c_min',
        'heart_rate_max','sbp_max','dbp_max','respiratory_rate_max','spo2_max','temperature_c_max',
        'abg_base_excess','cmp_lactate','abg_o2_saturation','abg_ph',
        'cmp_aniongap','cmp_bicarbonate','cmp_creatinine','cmp_glucose','cmp_potassium','cmp_bun',
        'cbc_hematocrit','cbc_hemoglobin','aptt_inr','cbc_platelet','aptt_ptt','cbc_rbc','cbc_wbc'
    ]
    ID_COLS = ['subject_id','hadm_id','stay_id','intime','endtime','admittime','dischtime','hospital_expire_flag']

    # Drop IDs/extra cols, enforce order
    X_all = full_df.drop(columns=[c for c in ID_COLS if c in full_df.columns], errors='ignore')
    missing = [c for c in SEPSIS_37 if c not in X_all.columns]
    if missing:
        raise ValueError(f"Missing required 37-feature columns: {missing}")
    X_all = X_all.reindex(columns=SEPSIS_37)

    # Use only complete rows for GAIN training
    complete_df = X_all.dropna()
    print(f"Found {len(complete_df)} complete rows for training and validation.")

    # --- 2. Load the Project's StandardScaler ---
    try:
        scaler = joblib.load(config["SCALER_FILE"])
        print(f"Successfully loaded existing StandardScaler from '{config['SCALER_FILE']}'")
    except FileNotFoundError:
        print(f"ERROR: Scaler file not found at '{config['SCALER_FILE']}'. Please ensure it exists.")
        return

    # --- 3. Split and Scale Data ---
    train_df, val_df = train_test_split(complete_df, test_size=0.2, random_state=42)
    
    # Use the loaded scaler. GAIN will now learn on the same data distribution as the classifier.
    train_scaled = scaler.transform(train_df)
    val_scaled = scaler.transform(val_df)
    
    # --- 4. Initialize Models and Dataloader ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = TensorDataset(torch.from_numpy(train_scaled).float().to(device))
    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True)
    val_tensor = torch.from_numpy(val_scaled).float().to(device)
    
    input_dim = len(SEPSIS_37)
    generator = Generator(input_dim=input_dim).to(device)
    discriminator = Discriminator(input_dim=input_dim).to(device)
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    
    D_loss_fn = nn.BCELoss()
    G_loss_fn_adv = nn.BCELoss()
    G_loss_fn_mse = nn.MSELoss()

    print(f"Starting GAIN training for up to {config['EPOCHS']} epochs on device: {device}")

    # --- 5. Training Loop with Validation ---
    best_val_rmse = float('inf')
    bad_epochs = 0

    for epoch in range(config["EPOCHS"]):
        generator.train()
        discriminator.train()
        
        for batch in train_loader:
            x_batch = batch[0]
            batch_size = x_batch.shape[0]
            
            mask = torch.tensor(binary_sampler(0.5, batch_size, input_dim), dtype=torch.float32).to(device)
            noise = torch.randn(x_batch.shape, device=device) # Use standard normal noise
            hint = torch.tensor(binary_sampler(config["HINT_RATE"], batch_size, input_dim), dtype=torch.float32).to(device)
            hint = mask * hint
            corrupted_x = x_batch * mask + noise * (1 - mask)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            imputed_data = generator(corrupted_x, mask)
            d_input = imputed_data.detach()
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
        
        # --- Validation Step ---
        generator.eval()
        with torch.no_grad():
            val_mask_np = binary_sampler(0.5, val_tensor.shape[0], input_dim)
            val_mask = torch.tensor(val_mask_np, dtype=torch.float32).to(device)
            val_noise = torch.randn(val_tensor.shape, device=device)
            corrupted_val = val_tensor * val_mask + val_noise * (1 - val_mask)
            
            imputed_val = generator(corrupted_val, val_mask)
            
            mse_val_loss = G_loss_fn_mse(val_tensor * (1 - val_mask), imputed_val * (1 - val_mask))
            val_rmse = torch.sqrt(mse_val_loss).item()
        
        print(f"Epoch {epoch+1}/{config['EPOCHS']} | Validation RMSE: {val_rmse:.4f}")
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            bad_epochs = 0
            output_path = Path(config["OUTPUT_MODEL_FILE"])
            output_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(generator.state_dict(), output_path)
            print(f"  -> New best model found! Saved to '{output_path}'")
        else:
            bad_epochs += 1
            if bad_epochs >= config["PATIENCE"]:
                print(f"Early stopping after {bad_epochs} epochs with no improvement.")
                break

    print("\n--- GAIN Training Finished ---")
    print(f"✅ Best Validation RMSE: {best_val_rmse:.4f}")
    print(f"✅ New harmonized generator saved to '{config['OUTPUT_MODEL_FILE']}'")

if __name__ == '__main__':
    train_gain_sepsis_harmonized(CONFIG)