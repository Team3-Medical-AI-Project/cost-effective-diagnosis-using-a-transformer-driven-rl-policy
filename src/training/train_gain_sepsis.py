# """
# GAIN: Generative Adversarial Imputation Networks
# Implementation in PyTorch.

# This file defines the architecture for the Generator and Discriminator networks
# that form the GAIN model, as well as the main class to handle the training process.
# """

# import torch
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# import joblib
# from torch.utils.data import DataLoader, TensorDataset
# from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parents[2]  # two levels up from gain.py
# # --- Configuration ---
# CONFIG = {
#     #"INPUT_FILE": "data/preprocessed/sepsis_feature_matrix.csv",
#     #"SCALER_FILE": "data/processed/sepsis/scaler.joblib",
#     "INPUT_FILE": PROJECT_ROOT / "data" / "preprocessed" / "sepsis_feature_matrix.csv",
#     "SCALER_FILE": PROJECT_ROOT / "data" / "processed" / "sepsis" / "scaler.joblib",
#     "OUTPUT_MODEL_FILE": "models/generator_sepsis.pth",
#     "ID_COLUMNS": ['subject_id', 'hadm_id', 'stay_id'],
#     "TARGET_COLUMN": "hospital_expire_flag",
#     "MISSING_RATE": 0.2, # Percentage of values to artificially mask
#     "HINT_RATE": 0.9,    # Percentage of known values to reveal to the discriminator
#     "ALPHA": 10.0,       # Hyperparameter for the reconstruction loss
#     "BATCH_SIZE": 128,
#     "EPOCHS": 100
# }

# class Generator(nn.Module):
#     """
#     The Generator network.
#     Takes a data vector + mask vector as input and outputs an imputed data vector.
#     """
#     def __init__(self, input_dim):
#         super(Generator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim * 2, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, input_dim),
#             nn.Sigmoid()
#         )

#     def forward(self, x, m):
#         """
#         x: data vector with missing values (NaNs replaced by 0)
#         m: mask vector (0 for missing, 1 for present)
#         """
#         input_cat = torch.cat([x, m], dim=1)
#         imputed_data = self.model(input_cat)
#         return imputed_data

# class Discriminator(nn.Module):
#     """
#     The Discriminator network.
#     Takes an imputed data vector + hint vector and outputs a probability mask.
#     """
#     def __init__(self, input_dim):
#         super(Discriminator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim * 2, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, input_dim),
#             nn.Sigmoid()
#         )

#     def forward(self, x, h):
#         """
#         x: the imputed data vector from the Generator
#         h: the hint vector
#         """
#         input_cat = torch.cat([x, h], dim=1)
#         probability_mask = self.model(input_cat)
#         return probability_mask

# class GAIN:
#     """
#     The main GAIN class that encapsulates the Generator, Discriminator,
#     and the training loop.
#     """
#     def __init__(self, input_dim, alpha=10.0):
#         self.input_dim = input_dim
#         self.alpha = alpha

#         self.generator = Generator(input_dim)
#         self.discriminator = Discriminator(input_dim)
        
#         self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.001)
#         self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)

#         self.d_loss_fn = nn.BCELoss()
#         self.g_loss_fn = nn.BCELoss()
#         self.mse_loss_fn = nn.MSELoss()

#     def train(self, data_loader, epochs=100):
#         """
#         The main training loop for the GAIN model.
#         """
#         print("Starting GAIN training...")
#         for epoch in range(epochs):
#             d_loss_total, g_loss_total, mse_loss_total = 0, 0, 0
#             for batch in data_loader:
#                 x_batch = batch[0]
#                 batch_size = x_batch.shape[0]

#                 # --- Train Discriminator ---
#                 self.optimizer_D.zero_grad()
                
#                 # 1. Create artificial missingness
#                 mask = np.random.binomial(1, 1 - CONFIG["MISSING_RATE"], size=x_batch.shape)
#                 mask = torch.from_numpy(mask).float()
                
#                 noise = torch.rand(x_batch.shape)
#                 corrupted_x = x_batch * mask + noise * (1 - mask)
                
#                 # 2. Generate imputed data
#                 imputed_data = self.generator(corrupted_x, mask)
                
#                 # 3. Create hint vector
#                 hint_mask = np.random.binomial(1, CONFIG["HINT_RATE"], size=x_batch.shape)
#                 hint_mask = torch.from_numpy(hint_mask).float()
#                 hint = mask * hint_mask
                
#                 # 4. Discriminate
#                 d_input = imputed_data * (1 - mask) + x_batch * mask
#                 d_prob = self.discriminator(d_input, hint)
                
#                 # 5. Calculate Discriminator loss and update
#                 d_loss = self.d_loss_fn(d_prob, mask)
#                 d_loss.backward()
#                 self.optimizer_D.step()

#                 # --- Train Generator ---
#                 self.optimizer_G.zero_grad()
                
#                 # 1. Generate new imputed data
#                 imputed_data = self.generator(corrupted_x, mask)
                
#                 # 2. Discriminate the new data
#                 d_input = imputed_data * (1 - mask) + x_batch * mask
#                 d_prob = self.discriminator(d_input, hint)
                
#                 # 3. Calculate Generator losses
#                 g_adversarial_loss = self.g_loss_fn(d_prob, mask)
#                 mse_reconstruction_loss = self.mse_loss_fn(imputed_data * mask, x_batch * mask)
                
#                 g_loss = g_adversarial_loss + self.alpha * mse_reconstruction_loss
#                 g_loss.backward()
#                 self.optimizer_G.step()
                
#                 d_loss_total += d_loss.item()
#                 g_loss_total += g_adversarial_loss.item()
#                 mse_loss_total += mse_reconstruction_loss.item()

#             print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss_total/len(data_loader):.4f} | G Loss: {g_loss_total/len(data_loader):.4f} | MSE Loss: {mse_loss_total/len(data_loader):.4f}")
#         print("GAIN training finished.")

# if __name__ == '__main__':
#     print("--- Initializing GAIN Training Pipeline ---")
    
#     # 1. Load the original, pre-split feature matrix
#     try:
#         full_df = pd.read_csv(CONFIG["INPUT_FILE"])
#     except FileNotFoundError:
#         print(f"Error: Could not find '{CONFIG['INPUT_FILE']}'. Please run preprocessing first.")
#         exit()

#     # 2. Isolate complete rows to create a "ground truth" dataset for training
#     feature_cols = [col for col in full_df.columns if col not in CONFIG["ID_COLUMNS"] + [CONFIG["TARGET_COLUMN"]]]
#     complete_df = full_df[feature_cols].dropna()
#     print(f"Found {len(complete_df)} complete rows to train GAIN on.")

#     # 3. Load the scaler and apply it to the complete data
#     try:
#         scaler = joblib.load(CONFIG["SCALER_FILE"])
#         scaled_data = scaler.transform(complete_df)
#     except FileNotFoundError:
#         print(f"Error: Could not find scaler at '{CONFIG['SCALER_FILE']}'. Please run Phase 2 first.")
#         exit()

#     # 4. Create PyTorch DataLoader
#     dataset = TensorDataset(torch.from_numpy(scaled_data).float())
#     data_loader = DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    
#     # 5. Initialize and train the GAIN model
#     input_dim = len(feature_cols)
#     gain_model = GAIN(input_dim=input_dim, alpha=CONFIG["ALPHA"])
#     gain_model.train(data_loader, epochs=CONFIG["EPOCHS"])
    
#     # 6. Save the trained generator model
#     output_path = Path(CONFIG["OUTPUT_MODEL_FILE"])
#     output_path.parent.mkdir(exist_ok=True, parents=True)
#     torch.save(gain_model.generator.state_dict(), output_path)
#     print(f"\n--- Trained Generator saved to '{output_path}' ---")


"""
GAIN Training Script for Sepsis Cohort

Description:
This script pre-trains the GAIN imputer on the complete rows of the 
preprocessed Sepsis dataset. It imports the model architecture from 
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

# --- Configuration for Sepsis ---
# Define the project root to make file paths more robust
PROJECT_ROOT = Path(__file__).resolve().parents[2] 

CONFIG = {
    "INPUT_FILE": PROJECT_ROOT / "data" / "preprocessed" / "sepsis_feature_matrix.csv",
    "SCALER_FILE": PROJECT_ROOT / "data" / "processed" / "sepsis" / "scaler.joblib",
    "OUTPUT_MODEL_FILE": PROJECT_ROOT / "models" / "generator_sepsis.pth",
    "ID_COLUMNS": ['subject_id', 'hadm_id', 'stay_id'],
    "TARGET_COLUMN": "hospital_expire_flag",
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
        print("Starting GAIN training for Sepsis cohort...")
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
    print("--- Initializing GAIN Training Pipeline for Sepsis ---")
    
    try:
        full_df = pd.read_csv(CONFIG["INPUT_FILE"])
    except FileNotFoundError:
        print(f"Error: Could not find '{CONFIG['INPUT_FILE']}'. Please run preprocessing first.")
        exit()

    feature_cols = [col for col in full_df.columns if col not in CONFIG["ID_COLUMNS"] + [CONFIG["TARGET_COLUMN"]]]
    complete_df = full_df[feature_cols].dropna()
    print(f"Found {len(complete_df)} complete rows to train GAIN on.")

    try:
        scaler = joblib.load(CONFIG["SCALER_FILE"])
        scaled_data = scaler.transform(complete_df)
    except FileNotFoundError:
        print(f"Error: Could not find scaler at '{CONFIG['SCALER_FILE']}'. Please run Phase 2 first.")
        exit()

    dataset = TensorDataset(torch.from_numpy(scaled_data).float())
    data_loader = DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    
    input_dim = len(feature_cols)
    gain_model = GAIN(input_dim=input_dim, alpha=CONFIG["ALPHA"])
    gain_model.train(data_loader, epochs=CONFIG["EPOCHS"])
    
    output_path = Path(CONFIG["OUTPUT_MODEL_FILE"])
    output_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(gain_model.generator.state_dict(), output_path)
    print(f"\n--- Trained Sepsis Generator saved to '{output_path}' ---")
