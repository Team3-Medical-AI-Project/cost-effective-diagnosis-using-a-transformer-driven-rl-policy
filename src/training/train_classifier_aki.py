"""
Preliminary Classifier Training for AKI Cohort (v2.0 - Final)

This script trains a simple MLP classifier on the processed and augmented
AKI data. The trained model is used within the RL environment to provide an
initial diagnostic probability.
"""
# --- 1. Imports ---
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os
import sys

# --- Path Setup ---
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
except NameError:
    sys.path.append(os.path.abspath('..'))

from src.models.classifier import PreliminaryClassifier

# --- 2. Configuration ---
class Config:
    PROCESSED_DATA_DIR = "data/processed/aki"
    MODEL_DIR = "models"
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "classifier_aki.pth")
    
    # Hyperparameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    EPOCHS = 50

# --- 3. Main Training Function ---
def train_classifier_aki(config):
    """
    Main function to load data, train the classifier, and save the best model.
    """
    print("--- Starting AKI Classifier Training ---")
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # Load data
    try:
        X_train = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "train_X.csv"))
        y_train = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "train_y.csv"))
        X_val = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "val_X.csv"))
        y_val = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "val_y.csv"))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Could not find the necessary processed CSV files.")
        print("Please run the data preparation/augmentation script first.")
        return

    input_dim = X_train.shape[1]
    print(f"Data loaded successfully. Detected {input_dim} features.")
    
    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Initialize Model and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure output_dim=1 for binary classification with BCEWithLogitsLoss
    model = PreliminaryClassifier(input_dim=input_dim, output_dim=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # --- Training Loop ---
    best_val_f1 = 0.0
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                preds = torch.round(torch.sigmoid(outputs))
                val_preds.extend(preds.cpu().numpy().flatten())
                val_true.extend(batch_y.cpu().numpy().flatten())

        val_f1 = f1_score(val_true, val_preds)
        val_auc = roc_auc_score(val_true, val_preds)
        print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss: {avg_train_loss:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Model improved and saved to {config.MODEL_SAVE_PATH}")

    print("\n--- Training Complete ---")
    print(f"Best validation F1-Score: {best_val_f1:.4f}")
    print(f"✅ Saved AKI classifier to {config.MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_classifier_aki(Config())