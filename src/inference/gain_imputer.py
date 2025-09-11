# src/inference/gain_imputer.py
import numpy as np, torch, joblib
from src.models.gain import Generator  # <-- your shared blueprint

class GAINImputer:
    """
    Drop-in imputer that uses your existing Generator weights.
    Works with CPU or CUDA; defaults to CPU so it never conflicts with sepsis.
    """
    def __init__(self, generator_path: str, scaler_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.scaler = joblib.load(scaler_path)
        d = self.scaler.n_features_in_
        self.G = Generator(input_dim=d).to(self.device)
        self.G.load_state_dict(torch.load(generator_path, map_location=self.device))
        self.G.eval()

    @torch.no_grad()
    def impute_raw(self, x_raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        x_raw: (D,) raw units with np.nan where missing
        mask : (D,) 1=observed, 0=missing
        returns: (D,) raw units with missings filled only at 0s in mask
        """
        x = x_raw.astype("float32", copy=True)
        m = mask.astype("float32", copy=False)
        if m.ndim == 1:
            x = x[None, :]
            m = m[None, :]

        # nothing observed -> return zeros (env still carries mask)
        if m.sum() == 0:
            return np.nan_to_num(x, nan=0.0)[0]

        # scale to [0,1] using the same scaler as training
        X_mm = self.scaler.transform(np.nan_to_num(x, nan=0.0))
        X = torch.from_numpy(X_mm).to(self.device)
        M = torch.from_numpy(m).to(self.device)

        # Your Generator expects (x, m) concatenated internally
        G_out = self.G(X, M).cpu().numpy()   # already in [0,1] with your Sigmoid

        # fill missings in MinMax space only
        X_hat = X_mm * m + G_out * (1.0 - m)

        # back to raw units
        X_hat_raw = self.scaler.inverse_transform(X_hat)
        filled = x.copy()
        miss = (m == 0)
        filled[miss] = X_hat_raw[miss]
        return filled[0]
