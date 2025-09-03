"""
Recompute thresholds from validation set using the SAME pipeline as the env:
(full row) -> (mask all observed) -> GAIN -> (optional scaler) -> classifier -> softmax -> (optional calibrator)

Writes cfg['threshold_json'] (default models/threshold_sepsis.json) with:
- threshold_f2
- threshold_recall90
- chosen: "threshold_f2" (or as --choose)

Also saves PR/ROC svgs in --outdir (default reports/thr_recalib).
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score

# Use the exact env implementation
from src.training.sepsis_env_fast import SepsisEnvFast, Config as EnvConfig

def load_yaml(path: str):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def probs_from_env(env: SepsisEnvFast):
    """
    Compute p(expired) for every validation row using the env's pipeline.
    We set: curr = X[i], mask = valid[i]  (i.e., all observed features),
    then call env._p_expired() which runs GAIN->(scaler?)->clf->softmax->(calibrator?).
    """
    base = env  # env is already unwrapped here
    X = base.X
    valid = base.valid
    y = base.y.detach().cpu().numpy().astype(int).ravel()
    n = X.shape[0]
    probs = np.zeros(n, dtype=np.float32)

    for i in range(n):
        # reveal all available features for this row
        base.curr.copy_(X[i])
        base.mask.copy_(valid[i])
        # env computes probability with its own pipeline
        p = base._p_expired()
        probs[i] = float(p)

    return y, probs

def choose_thresholds(y: np.ndarray, p: np.ndarray):
    # PR/ROC + metrics
    prec, rec, thr = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = auc(fpr, tpr)

    # F2 sweep (align shapes: len(thr) == len(prec) - 1)
    beta = 2.0
    f2_all = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec + 1e-12)
    f2_valid = f2_all[:-1]  # match thr
    thr_valid = thr
    j_f2 = int(np.nanargmax(f2_valid))
    t_f2 = float(thr_valid[j_f2])

    # smallest threshold s.t. recall >= 0.90 (choose best F2 among those)
    mask = (rec[:-1] >= 0.90)
    if mask.any():
        f2_masked = f2_valid[mask]
        thr_masked = thr_valid[mask]
        j = int(np.nanargmax(f2_masked))
        t_r90 = float(thr_masked[j])
    else:
        t_r90 = t_f2

    return {
        "AUC": float(roc_auc),
        "AP": float(ap),
        "prec": prec, "rec": rec, "thr": thr,
        "threshold_f2": t_f2,
        "threshold_recall90": t_r90,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config used by SepsisEnvFast")
    ap.add_argument("--outdir", default="reports/thr_recalib")
    ap.add_argument("--choose", default="threshold_f2",
                    choices=["threshold_f2", "threshold_recall90"])
    args = ap.parse_args()

    # Build env from the same YAML the trainer/evaluator uses
    cfg_dict = load_yaml(args.config) or {}
    cfg = EnvConfig(cfg_dict)
    env = SepsisEnvFast(cfg)  # loads data, GAIN, classifier, scaler, calibrator from cfg

    # Get probs on the env's own validation set with ALL features revealed
    y, p = probs_from_env(env)

    res = choose_thresholds(y, p)
    t_f2, t_r90 = res["threshold_f2"], res["threshold_recall90"]
    auc_roc, ap_val = res["AUC"], res["AP"]
    prec, rec, thr = res["prec"], res["rec"], res["thr"]

    # Where to write thresholds
    thr_json_path = getattr(cfg, "threshold_json", None) or "models/threshold_sepsis.json"
    ensure_dir(os.path.dirname(thr_json_path))
    out = {
        "threshold_f2": float(t_f2),
        "threshold_recall90": float(t_r90),
        "chosen": args.choose,
        "AUC": float(auc_roc),
        "AP": float(ap_val),
    }
    with open(thr_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[thr] Wrote {thr_json_path}")
    print(json.dumps(out, indent=2))

    # Figures
    ensure_dir(args.outdir)
    import matplotlib.pyplot as plt

    # PR with both operating points
    plt.figure(); plt.title(f"PR (AP={ap_val:.3f})")
    plt.plot(rec, prec, label="PR")
    # mark F2-opt
    preds = (p >= t_f2).astype(int)
    tp = ((y==1)&(preds==1)).sum(); fp = ((y==0)&(preds==1)).sum(); fn = ((y==1)&(preds==0)).sum()
    pr_pt = tp / max(1, tp+fp); rc_pt = tp / max(1, tp+fn)
    plt.scatter([rc_pt],[pr_pt], s=50, label=f"F2-opt@{t_f2:.3f}")
    # mark R>=0.90-opt
    preds = (p >= t_r90).astype(int)
    tp = ((y==1)&(preds==1)).sum(); fp = ((y==0)&(preds==1)).sum(); fn = ((y==1)&(preds==0)).sum()
    pr_pt = tp / max(1, tp+fp); rc_pt = tp / max(1, tp+fn)
    plt.scatter([rc_pt],[pr_pt], s=50, marker="^", label=f"R≥0.90@{t_r90:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "pr.svg")); plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(); plt.title(f"ROC (AUC={auc_roc:.3f})")
    plt.plot(fpr, tpr); plt.plot([0,1],[0,1], "--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "roc.svg")); plt.close()

if __name__ == "__main__":
    main()
