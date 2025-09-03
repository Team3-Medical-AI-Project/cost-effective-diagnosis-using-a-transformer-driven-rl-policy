"""
Preliminary Classifier Training for AKI Cohort (v2.1)

This trains the simple MLP on the GAIN-imputed AKI splits.
Changes from v2.0:
- AUROC/AUPRC computed on probabilities (sigmoid(logits))
- Prints Acc, F1, AUROC, AUPRC
"""
import os, sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
except NameError:
    sys.path.append(os.path.abspath('..'))

from src.models.classifier import PreliminaryClassifier


class Config:
    PROCESSED_DATA_DIR = "data/processed/aki"
    MODEL_DIR = "models"
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "classifier_aki.pth")

    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 50


def train_classifier_aki(config: Config):
    print("--- Starting AKI Classifier Training ---")
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # Load GAIN-imputed splits
    X_train = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "train_X.csv"))
    y_train = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "train_y.csv"))
    X_val   = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "val_X.csv"))
    y_val   = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "val_y.csv"))

    input_dim = X_train.shape[1]
    print(f"Data loaded. Features: {input_dim} | "
          f"train N={len(X_train)}, val N={len(X_val)}")

    # Tensors
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_t   = torch.tensor(X_val.values,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val.values,   dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),
                              batch_size=config.BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PreliminaryClassifier(input_dim=input_dim, output_dim=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_f1 = 0.0
    for epoch in range(config.EPOCHS):
        # ---- Train
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        train_loss = running / max(1, len(train_loader))

        # ---- Validate
        model.eval()
        val_true, val_probs, val_preds = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits)            # probabilities in [0,1]
                preds = (probs >= 0.5).float()           # hard predictions
                val_true.extend(yb.cpu().numpy().ravel())
                val_probs.extend(probs.cpu().numpy().ravel())
                val_preds.extend(preds.cpu().numpy().ravel())

        acc  = accuracy_score(val_true, val_preds)
        f1   = f1_score(val_true, val_preds)
        auc  = roc_auc_score(val_true, val_probs)
        aupr = average_precision_score(val_true, val_probs)

        print(f"Epoch [{epoch+1}/{config.EPOCHS}] "
              f"| TrainLoss: {train_loss:.4f} "
              f"| Val Acc: {acc:.4f} F1: {f1:.4f} AUROC: {auc:.4f} AUPRC: {aupr:.4f}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"↑ Saved best model to {config.MODEL_SAVE_PATH} (F1={best_val_f1:.4f})")

    print("\n--- Training Complete ---")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"✅ Saved AKI classifier to {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_classifier_aki(Config())
