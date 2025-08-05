"""
GAIN Training Script for AKI Cohort

Description:
This script pre-trains the GAIN imputer on the complete rows of the 
preprocessed AKI dataset. It imports the model architecture from 
src/models/gain.py.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# Import the GAIN model blueprints
from src.models.gain import Generator, Discriminator

# --- Configuration for AKI ---
# Define the project root to make file paths more robust
PROJECT_ROOT = Path(__file__).resolve().parents[2] 

CONFIG = {
    "INPUT_FILE": PROJECT_ROOT / "data" / "preprocessed" / "aki_feature_matrix.csv",
    "SCALER_FILE": PROJECT_ROOT / "data" / "processed" / "aki" / "scaler_aki.joblib",
    "OUTPUT_MODEL_FILE": PROJECT_ROOT / "models" / "generator_aki.pth",
    "ID_COLUMNS": ['subject_id', 'hadm_id', 'stay_id'],
    "TARGET_COLUMN": "kdigo_aki",
    "MISSING_RATE": 0.2, # Percentage of values to artificially mask
    "HINT_RATE": 0.9,    # Percentage of known values to reveal to the discriminator
    "ALPHA": 10.0,       # Hyperparameter for the reconstruction loss
    "BATCH_SIZE": 128,
    "EPOCHS": 100
}

class GAIN:
    """
    The main GAIN class that encapsulates the Generator, Discriminator,
    and the training loop.
    """
    def __init__(self, input_dim, alpha=10.0):
        self.input_dim = input_dim
        self.alpha = alpha

        # Initialize models from the imported blueprints
        self.generator = Generator(input_dim)
        self.discriminator = Discriminator(input_dim)
        
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0001)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001)

        self.d_loss_fn = nn.BCELoss()
        self.g_loss_fn = nn.BCELoss()
        self.mse_loss_fn = nn.MSELoss()

    def train(self, data_loader, epochs=100):
        """
        The main training loop for the GAIN model.
        """
        print("Starting GAIN training for AKI cohort...")
        for epoch in range(epochs):
            d_loss_total, g_loss_total, mse_loss_total = 0, 0, 0
            for batch in data_loader:
                x_batch = batch[0]
                
                # --- Train Discriminator ---
                self.optimizer_D.zero_grad()
                
                mask = np.random.binomial(1, 1 - CONFIG["MISSING_RATE"], size=x_batch.shape)
                mask = torch.from_numpy(mask).float()
                
                noise = torch.rand(x_batch.shape)
                corrupted_x = x_batch * mask + noise * (1 - mask)
                
                with torch.no_grad():
                    imputed_data = self.generator(corrupted_x, mask)
                
                hint_mask = np.random.binomial(1, CONFIG["HINT_RATE"], size=x_batch.shape)
                hint_mask = torch.from_numpy(hint_mask).float()
                hint = mask * hint_mask
                
                d_input = imputed_data * (1 - mask) + x_batch * mask
                d_prob = self.discriminator(d_input, hint)
                
                d_loss = self.d_loss_fn(d_prob, mask)
                d_loss.backward()
                self.optimizer_D.step()

                # --- Train Generator ---
                self.optimizer_G.zero_grad()
                
                imputed_data_g = self.generator(corrupted_x, mask)
                d_input_g = imputed_data_g * (1 - mask) + x_batch * mask
                d_prob_g = self.discriminator(d_input_g, hint)
                
                g_adversarial_loss = self.g_loss_fn(d_prob_g, mask)
                mse_reconstruction_loss = self.mse_loss_fn(imputed_data_g * mask, x_batch * mask)
                
                g_loss = g_adversarial_loss + self.alpha * mse_reconstruction_loss
                g_loss.backward()
                self.optimizer_G.step()
                
                d_loss_total += d_loss.item()
                g_loss_total += g_adversarial_loss.item()
                mse_loss_total += mse_reconstruction_loss.item()

            print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss_total/len(data_loader):.4f} | G Loss: {g_loss_total/len(data_loader):.4f} | MSE Loss: {mse_loss_total/len(data_loader):.4f}")
        print("GAIN training finished.")

if __name__ == '__main__':
    print("--- Initializing GAIN Training Pipeline for AKI ---")
    
    try:
        full_df = pd.read_csv(CONFIG["INPUT_FILE"])
    except FileNotFoundError:
        print(f"Error: Could not find '{CONFIG['INPUT_FILE']}'. Please run AKI preprocessing first.")
        exit()

    feature_cols = [col for col in full_df.columns if col not in CONFIG["ID_COLUMNS"] + [CONFIG["TARGET_COLUMN"]]]
    complete_df = full_df[feature_cols].dropna()
    print(f"Found {len(complete_df)} complete rows to train GAIN on.")

    try:
        scaler = joblib.load(CONFIG["SCALER_FILE"])
        scaled_data = scaler.transform(complete_df)
    except FileNotFoundError:
        print(f"Error: Could not find scaler at '{CONFIG['SCALER_FILE']}'. Please run Phase 2 for AKI first.")
        exit()

    dataset = TensorDataset(torch.from_numpy(scaled_data).float())
    data_loader = DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    
    input_dim = len(feature_cols)
    gain_model = GAIN(input_dim=input_dim, alpha=CONFIG["ALPHA"])
    gain_model.train(data_loader, epochs=CONFIG["EPOCHS"])
    
    output_path = Path(CONFIG["OUTPUT_MODEL_FILE"])
    output_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(gain_model.generator.state_dict(), output_path)
    print(f"\n--- Trained AKI Generator saved to '{output_path}' ---")
