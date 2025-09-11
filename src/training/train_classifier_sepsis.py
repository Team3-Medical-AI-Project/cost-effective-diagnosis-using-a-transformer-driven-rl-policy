# train_prelim_classifier.py
import os, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_recall_curve, roc_curve, confusion_matrix)
import joblib
import matplotlib.pyplot as plt

from src.models.classifier import PreliminaryClassifier

class Cfg:
    # ----- choose your data source -----
    USE_AUGMENTED = True  # True = use augmented/scaled; False = raw -> apply imputer+scaler
    DATA_DIR = Path("data/processed/sepsis")
    AUG_DIR  = DATA_DIR / "augmented"

    # ----- training -----
    LR = 1e-3
    BATCH = 128
    EPOCHS = 100
    PATIENCE = 12
    MODEL_DIR = Path("models")
    MODEL_SAVE_PATH = MODEL_DIR / "classifier_sepsis.pth"

    # ----- reports -----
    REPORT_DIR = Path("reports/classifier_sepsis")

def _load_data(cfg: Cfg):
    if cfg.USE_AUGMENTED:
        # scaled features ready-to-train
        Xtr = pd.read_csv(cfg.AUG_DIR / "train_X_scaled.csv").values.astype(np.float32)
        def _load_y(path):
            dfy = pd.read_csv(path)
            if "hospital_expire_flag" in dfy.columns:
                col = dfy["hospital_expire_flag"]
            elif "y" in dfy.columns:
                col = dfy["y"]
            else:
                col = dfy.iloc[:, -1]
            return col.values.astype(np.int64).ravel()
        ytr = _load_y(cfg.AUG_DIR / "train_y_aug.csv")
        Xv  = pd.read_csv(cfg.AUG_DIR / "val_X_scaled.csv").values.astype(np.float32)
        yv  = _load_y(cfg.AUG_DIR / "val_y.csv")
        Xt  = pd.read_csv(cfg.AUG_DIR / "test_X_scaled.csv").values.astype(np.float32)
        yt  = _load_y(cfg.AUG_DIR / "test_y.csv")
        imputer = scaler = None
    else:
        # RAW: apply the same imputer+scaler you saved
        imputer = joblib.load(cfg.DATA_DIR / "imputer.joblib")
        scaler  = joblib.load(cfg.DATA_DIR / "scaler.joblib")

        def _tx(path):
            X = pd.read_csv(path)
            X = pd.DataFrame(imputer.transform(X), columns=X.columns)
            X = pd.DataFrame(scaler.transform(X), columns=X.columns)
            return X.values.astype(np.float32)

        Xtr = _tx(cfg.DATA_DIR / "train_X.csv")
        ytr = _load_y(cfg.DATA_DIR / "train_y.csv")
        Xv  = _tx(cfg.DATA_DIR / "val_X.csv")
        yv  = _load_y(cfg.DATA_DIR / "val_y.csv")
        Xt  = _tx(cfg.DATA_DIR / "test_X.csv")
        yt  = _load_y(cfg.DATA_DIR / "test_y.csv")

    return (Xtr, ytr, Xv, yv, Xt, yt)

def train(cfg=Cfg()):
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, Xv, yv, Xt, yt = _load_data(cfg)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PreliminaryClassifier(input_dim=Xtr.shape[1], output_dim=2).to(dev)

    # class weights (inverse frequency)
    uniq, cnt = np.unique(ytr, return_counts=True)
    total = cnt.sum()
    w = np.array([total/(2*dict(zip(uniq, cnt)).get(c,1)) for c in [0,1]], dtype=np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, device=dev))

    opt = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-4)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr).reshape(-1).long()),
                              batch_size=cfg.BATCH, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(Xv),  torch.from_numpy(yv).reshape(-1).long()),
                              batch_size=cfg.BATCH)

    best_ap = 0.0; bad = 0
    for ep in range(1, cfg.EPOCHS+1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); opt.step()

        # validate
        model.eval(); P=[]; Y=[]
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(dev)
                p1 = torch.softmax(model(xb), dim=1)[:,1].cpu().numpy()
                P.append(p1); Y.append(yb.numpy())
        P = np.concatenate(P); Y = np.concatenate(Y)
        auc = roc_auc_score(Y, P); ap = average_precision_score(Y, P)
        f1  = f1_score(Y, (P>=0.5).astype(int))
        print(f"Epoch {ep:03d} | AUC={auc:.3f}  AP={ap:.3f}  F1@0.50={f1:.3f}")

        if ap > best_ap:
            best_ap = ap; bad = 0
            torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
        else:
            bad += 1
            if bad >= cfg.PATIENCE:
                print("Early stopping."); break

    print(f"Best val AP={best_ap:.3f}. Saved → {cfg.MODEL_SAVE_PATH}")

    # ---------- test evaluation + plots ----------
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=dev))
    model.eval()
    with torch.no_grad():
        logits_test = model(torch.from_numpy(Xt).to(dev))
        Ptest = torch.softmax(logits_test, dim=1)[:,1].cpu().numpy()
    AUC  = roc_auc_score(yt, Ptest)
    AP   = average_precision_score(yt, Ptest)
    F1   = f1_score(yt, (Ptest>=0.5).astype(int))

    # PR / ROC
    pr_p, pr_r, _ = precision_recall_curve(yt, Ptest)
    fpr, tpr, _   = roc_curve(yt, Ptest)

    plt.figure(); plt.plot(pr_r, pr_p, label=f"AP={AP:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve"); plt.legend()
    plt.savefig(cfg.REPORT_DIR / "pr.png", dpi=150); plt.close()

    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={AUC:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curve"); plt.legend()
    plt.savefig(cfg.REPORT_DIR / "roc.png", dpi=150); plt.close()

    # Confusion Matrix @0.5
    cm = confusion_matrix(yt, (Ptest>=0.5).astype(int))
    pd.DataFrame(cm, index=["True 0","True 1"], columns=["Pred 0","Pred 1"]).to_csv(cfg.REPORT_DIR / "confusion_matrix.csv")

    # metrics json
    metrics = {"AUC": float(AUC), "AUPRC": float(AP), "F1@0.50": float(F1)}
    with open(cfg.REPORT_DIR / "metrics.json", "w") as f: json.dump(metrics, f, indent=2)

    print("Test metrics:", metrics)
    print(f"Artifacts saved in {cfg.REPORT_DIR}")

if __name__ == "__main__":
    train()
