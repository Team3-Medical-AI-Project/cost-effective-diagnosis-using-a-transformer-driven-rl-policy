# src/inference/eval_gain_aki.py
import os, sys, numpy as np, pandas as pd, torch
from sklearn.preprocessing import MinMaxScaler
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
except NameError:
    sys.path.append(os.path.abspath('..'))
from src.models.gain import Generator

MODEL_PATH = "models/generator_aki.pth"
DATA_DIR   = "data/processed/aki"
TRAIN_X    = f"{DATA_DIR}/train_X.csv"
VAL_X      = f"{DATA_DIR}/val_X.csv"

def binary_mask(p, rows, cols):
    return np.random.binomial(1, 1.0-p, (rows, cols))  # 1=observed, (1-miss)

def main():
    # load data
    Xtr = pd.read_csv(TRAIN_X).values.astype("float32")
    Xva = pd.read_csv(VAL_X).values.astype("float32")
    nfeat = Xtr.shape[1]

    # scale to [0,1] using train stats (same convention as training)
    scaler = MinMaxScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype("float32")
    Xva = scaler.transform(Xva).astype("float32")

    # load generator (CPU)
    G = Generator(input_dim=nfeat)
    G.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    G.eval()

    # self-mask 20% of entries on a val subset
    N = min(5000, Xva.shape[0])
    x = torch.from_numpy(Xva[:N])
    m = torch.from_numpy(binary_mask(p=0.20, rows=N, cols=nfeat).astype("float32"))

    with torch.no_grad():
        z = torch.from_numpy(np.random.uniform(0, 0.01, size=x.shape).astype("float32"))
        x_tilde = x * m + (1-m) * z
        x_hat = x * m + (1-m) * G(x_tilde, m)   # imputed only where masked
        num = torch.sum((x_hat - x).pow(2) * (1-m))
        den = torch.sum(1-m) + 1e-8
        rmse = torch.sqrt(num / den).item()

    print(f"[GAIN quick-check] RMSE on 20% self-masked val subset: {rmse:.4f}")
    # (Optional) save an imputed sample file
    out = x_hat.numpy()
    out = scaler.inverse_transform(out)
    pd.DataFrame(out).to_csv(f"{DATA_DIR}/val_X_gain_selfmask_imputed.csv", index=False)
    print(f"Saved sample imputations to {DATA_DIR}/val_X_gain_selfmask_imputed.csv")

if __name__ == "__main__":
    main()
