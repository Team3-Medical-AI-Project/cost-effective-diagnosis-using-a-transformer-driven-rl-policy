import os, json, numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score
from src.training.sepsis_env_fast import Config
from src.models.gain import Generator
from src.models.classifier import PreliminaryClassifier
import yaml

CFG = "configs/sepsis_config.yaml"
cfg = Config(yaml.safe_load(open(CFG)))

proc = cfg.get("processed_data_dir")
X = np.loadtxt(os.path.join(proc, "val_X.csv"), delimiter=",", skiprows=1).astype(np.float32)
y = np.loadtxt(os.path.join(proc, "val_y.csv"), delimiter=",", skiprows=1).astype(np.int64).ravel()

dev = torch.device(cfg.get("device","cpu"))
nf  = int(cfg.get("num_features"))

gain = Generator(input_dim=nf).to(dev).eval()
clf  = PreliminaryClassifier(input_dim=nf, output_dim=2).to(dev).eval()

def _load_sd(p):
    sd = torch.load(p, map_location=dev)
    return sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd

gain.load_state_dict(_load_sd(cfg.get("gain_generator_path")))
clf.load_state_dict(_load_sd(cfg.get("prelim_classifier_path")))

X_t = torch.tensor(X, dtype=torch.float32, device=dev)

def eval_with_mask(mask_val: float):
    M_t = torch.full_like(X_t, fill_value=mask_val)  # 1.0 => fully observed; 0.0 => nothing observed
    with torch.no_grad():
        X_imp = gain(X_t, M_t)
        logits = clf(X_imp)
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return {
        "auc_roc": float(roc_auc_score(y, p)),
        "auc_pr": float(average_precision_score(y, p)),
        "p_mean": float(p.mean()),
        "p_std": float(p.std()),
        "p_min": float(p.min()),
        "p_max": float(p.max()),
    }

res_all_obs = eval_with_mask(1.0)  # correct way to sanity-check the classifier
res_no_obs  = eval_with_mask(0.0)  # shows the “flat ~0.5” behavior when nothing is observed

print(json.dumps({
    "n": int(len(y)),
    "all_observed": res_all_obs,
    "no_observed":  res_no_obs
}, indent=2))
