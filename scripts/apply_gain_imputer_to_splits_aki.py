# scripts/apply_gain_imputer_to_splits_aki.py
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import joblib

# --- ensure project root is on sys.path (same pattern as your train_gain_aki.py) ---
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- try to import your shared Generator; if missing, use a compatible fallback ---
Generator = None
try:
    from src.models.gain import Generator  # your common blueprint
except Exception:
    # Fallback that matches your original blueprint (no final sigmoid)
    class Generator(nn.Module):
        def __init__(self, input_dim, hidden=256):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim * 2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, input_dim)
            )
        def forward(self, x, m):
            return self.model(torch.cat([x, m], dim=1))


GEN_PATH = Path("models/generator_aki.pth")
SCALER_PATH = Path("models/gain_scaler_aki.joblib")

DATA_DIR = Path("data/processed/aki")
SPLITS = ["train", "val", "test"]

DEVICE = "cpu"     # change to "cuda" if you like
BATCH = 2048

def fill_with_gain(X: pd.DataFrame, G: nn.Module, scaler, device="cpu", batch=2048) -> pd.DataFrame:
    """
    Fill NaNs only (observed values stay exactly as-is).
    Assumes scaler was fit on TRAIN raw data and generator was trained on MinMax-scaled inputs.
    """
    X_np = X.values.astype("float32")
    mask = (~np.isnan(X_np)).astype("float32")  # 1=observed, 0=missing

    # Replace NaNs with 0 before scaling (consistent with your training)
    X_zeros = np.nan_to_num(X_np, nan=0.0)
    X_mm = scaler.transform(X_zeros)

    G.eval().to(device)
    preds_mm = np.zeros_like(X_mm, dtype="float32")

    with torch.no_grad():
        for i in range(0, X_mm.shape[0], batch):
            sl = slice(i, i + batch)
            Xb = torch.from_numpy(X_mm[sl]).to(device)
            Mb = torch.from_numpy(mask[sl]).to(device)
            Gb = G(Xb, Mb)
            Gb = Gb.detach().cpu().numpy()
            # your original generator has no sigmoid; clamp to [0,1] before invert scaling
            Gb = np.clip(Gb, 0.0, 1.0)
            preds_mm[sl] = Gb

    # back to original units
    preds_raw = scaler.inverse_transform(preds_mm)

    # fill only missing entries
    X_out = X_np.copy()
    miss = (mask == 0.0)
    X_out[miss] = preds_raw[miss]
    return pd.DataFrame(X_out, columns=X.columns)

def main():
    assert GEN_PATH.exists(), f"missing {GEN_PATH}"
    assert SCALER_PATH.exists(), f"missing {SCALER_PATH}"

    scaler = joblib.load(SCALER_PATH)
    d = int(scaler.n_features_in_)
    print(f"Loaded scaler with n_features_in_={d}")

    G = Generator(input_dim=d)
    G.load_state_dict(torch.load(GEN_PATH, map_location="cpu"))
    print(f"Loaded generator weights from {GEN_PATH}")

    for split in SPLITS:
        x_path = DATA_DIR / f"{split}_X.csv"
        out_path = DATA_DIR / f"{split}_X_filled.csv"
        if not x_path.exists():
            print(f"⚠️  {x_path} not found; skipping")
            continue

        X = pd.read_csv(x_path)
        assert X.shape[1] == d, f"feature count {X.shape[1]} != scaler dim {d}"
        n_allnan_rows = int(np.isnan(X.values).all(axis=1).sum())
        print(f"{x_path} -> shape={X.shape}, all-NaN rows={n_allnan_rows}")

        X_filled = fill_with_gain(X, G, scaler, device=DEVICE, batch=BATCH)
        # sanity
        nan_rate_before = float(np.isnan(X.values).mean())
        nan_rate_after = float(np.isnan(X_filled.values).mean())
        print(f"NaN rate: before={nan_rate_before:.6f} after={nan_rate_after:.6f}")

        X_filled.to_csv(out_path, index=False)
        print(f"✅ wrote {out_path}")

if __name__ == "__main__":
    main()
