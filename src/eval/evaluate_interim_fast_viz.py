"""
evaluate_interim_fast_viz.py  —  Clear, labeled visualizations for interim

Saves into --outdir:
  - cm_annotated.png              (confusion matrix: counts + percentages)
  - panel_usage_named.png         (named bars, counts and % of episodes)
  - cost_hist.png                 (cost distribution w/ mean & std)
  - prob_hist_by_class.png        (p(expired) histograms, vertical τ)
  - roc.png, pr.png               (AUCs and the operating point)
  - calibration.png               (reliability curve with Brier score)
  - episodes.jsonl                (per-episode details for reproducibility)

Usage:
  python -m src.eval.evaluate_interim_fast_viz ^
    --config configs\sepsis_config.yaml ^
    --model  models\rl_agent_sepsis_interim_main.zip ^
    --n-episodes 400 ^
    --outdir reports\interim_main
"""
from __future__ import annotations

import os, sys, json, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    auc, brier_score_loss
)

# --- Matplotlib defaults for clean look
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "axes.grid": True,
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 11
})

# Repo imports
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from stable_baselines3 import PPO
HAVE_MASK = False
try:
    from sb3_contrib import MaskablePPO
    HAVE_MASK = True
except Exception:
    pass

HAVE_AM = False
try:
    from sb3_contrib.common.wrappers import ActionMasker
    HAVE_AM = True
except Exception:
    pass

from src.training.sepsis_env_fast import SepsisEnvFast, Config as EnvConfig


# ---------------- utils ----------------
def load_yaml(path: str):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def unwrap_env(env):
    e, seen = env, set()
    while hasattr(e, "env") and id(e.env) not in seen:
        seen.add(id(e)); e = e.env
    try: return e.unwrapped
    except Exception: return e

def _mask_fn(e): return e.action_masks()

@torch.no_grad()
def infer_prob_expired(env_like, default=0.5):
    base = unwrap_env(env_like)

    p_cache = getattr(base, "_prev_p", None)
    if p_cache is not None:
        try: return float(p_cache)
        except Exception: pass

    device = torch.device(getattr(getattr(base, "cfg", None), "device", "cpu"))
    gen = getattr(base, "gain_generator", None) or getattr(base, "generator", None) or getattr(base, "gen", None)
    clf = getattr(base, "classifier", None) or getattr(base, "prelim_classifier", None) or getattr(base, "clf", None)
    x   = getattr(base, "current_state", None) or getattr(base, "curr", None) or getattr(base, "state", None)
    m   = getattr(base, "observation_mask", None) or getattr(base, "mask", None)
    full= getattr(base, "full_patient_data", None) or getattr(base, "full", None) or getattr(base, "x_full", None)

    try:
        if clf is not None and gen is not None and x is not None and m is not None:
            x = x if isinstance(x, torch.Tensor) else torch.tensor(np.asarray(x), dtype=torch.float32)
            m = m if isinstance(m, torch.Tensor) else torch.tensor(np.asarray(m), dtype=torch.float32)
            x, m = x.to(device), m.to(device)
            if x.dim()==1: x=x.unsqueeze(0)
            if m.dim()==1: m=m.unsqueeze(0)
            imputed = gen(x, m)
            probs = torch.softmax(clf(imputed), dim=1)
            return float(probs[0,1].item())
        if clf is not None and full is not None:
            full = full if isinstance(full, torch.Tensor) else torch.tensor(np.asarray(full), dtype=torch.float32)
            full = full.to(device)
            if full.dim()==1: full=full.unsqueeze(0)
            probs = torch.softmax(clf(full), dim=1)
            return float(probs[0,1].item())
    except Exception:
        pass
    return float(default)


# --------------- plots -----------------
def plot_confusion_matrix(cm: np.ndarray, outpath: str):
    totals = cm.sum()
    pct = np.zeros_like(cm, dtype=float)
    if totals > 0: pct = cm / totals

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix (Expired=1)")
    ax.set_xlabel("Prediction"); ax.set_ylabel("Truth")
    ax.set_xticks([0,1], ["Discharged (0)", "Expired (1)"])
    ax.set_yticks([0,1], ["Discharged (0)", "Expired (1)"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n({pct[i,j]*100:.1f}%)",
                    ha="center", va="center", fontweight="bold")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200); plt.close(fig)

def plot_panel_usage(counts: dict[int,int], panel_names: dict[int,str],
                     n_episodes: int, outpath: str):
    idx = sorted(counts.keys())
    names = [panel_names.get(i, f"Panel {i}") for i in idx]
    vals  = [counts.get(i, 0) for i in idx]
    perc  = [(v/n_episodes*100.0) if n_episodes>0 else 0 for v in vals]

    fig, ax = plt.subplots(figsize=(8,4.8))
    ax.bar(names, vals)
    ax.set_title("Panel Usage Before Diagnosis")
    ax.set_ylabel("Count across episodes")
    ax.set_xlabel("Panel")
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals)*0.02 if max(vals)>0 else 0.5, f"{v} ({perc[i]:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200); plt.close(fig)

def plot_cost_hist(costs: list[float], outpath: str):
    if len(costs)==0:
        return
    mean = float(np.mean(costs)); std = float(np.std(costs))
    maxv = max(costs); minv = min(costs)
    bins = max(5, min(30, int((maxv-minv)/20)+1))

    fig, ax = plt.subplots(figsize=(8,4.8))
    ax.hist(costs, bins=bins, edgecolor="black")
    ax.axvline(mean, linestyle="--")
    ax.set_title(f"Distribution of Episode Costs (mean ${mean:.2f} ± {std:.2f})")
    ax.set_xlabel("Episode total ($)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200); plt.close(fig)

def plot_prob_hist(y_true, probs, threshold, outpath):
    y_true = np.asarray(y_true); probs=np.asarray(probs)
    p0 = probs[y_true==0]; p1 = probs[y_true==1]

    fig, ax = plt.subplots(figsize=(8,4.8))
    ax.hist(p0, bins=20, alpha=0.6, label="True 0 (Discharged)")
    ax.hist(p1, bins=20, alpha=0.6, label="True 1 (Expired)")
    ax.axvline(threshold, linestyle="--", label=f"τ={threshold:.2f}")
    ax.set_title("Post-test p(Expired) by Class")
    ax.set_xlabel("p(Expired)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200); plt.close(fig)

def plot_roc_pr(y_true, probs, threshold, outdir):
    y_true = np.asarray(y_true); probs=np.asarray(probs)
    fig_roc, axr = plt.subplots(figsize=(6.2,5.6))
    fig_pr, axp = plt.subplots(figsize=(6.2,5.6))

    try:
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        axr.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        # mark operating point
        # closest threshold to τ
        from bisect import bisect_left
        # sklearn returns thresholds for ROC from high->low; we’ll just mark approximate
        axr.plot([0,1],[0,1],'--', linewidth=1)
        axr.set_title("ROC Curve")
        axr.set_xlabel("FPR"); axr.set_ylabel("TPR")
        axr.legend()
        fig_roc.tight_layout()
        fig_roc.savefig(os.path.join(outdir,"roc.png"), dpi=200)
    except Exception:
        plt.close(fig_roc)

    try:
        prec, rec, th = precision_recall_curve(y_true, probs)
        pr_auc = auc(rec, prec)
        axp.plot(rec, prec, label=f"AP≈{pr_auc:.3f}")
        axp.set_title("Precision–Recall Curve")
        axp.set_xlabel("Recall"); axp.set_ylabel("Precision")
        axp.legend()
        fig_pr.tight_layout()
        fig_pr.savefig(os.path.join(outdir,"pr.png"), dpi=200)
    except Exception:
        plt.close(fig_pr)

def plot_calibration(y_true, probs, outpath):
    # reliability plot (10 bins)
    y_true = np.asarray(y_true); probs=np.asarray(probs)
    bins = np.linspace(0.0, 1.0, 11)
    idx  = np.digitize(probs, bins) - 1
    mp, oy = [], []
    for b in range(10):
        mask = idx==b
        if np.sum(mask)==0: continue
        mp.append(np.mean(probs[mask]))
        oy.append(np.mean(y_true[mask]))
    fig, ax = plt.subplots(figsize=(6.8,5.2))
    ax.plot([0,1],[0,1],'--', linewidth=1)
    ax.plot(mp, oy, marker="o")
    try:
        brier = brier_score_loss(y_true, probs)
        title = f"Calibration (Brier={brier:.3f})"
    except Exception:
        title = "Calibration"
    ax.set_title(title)
    ax.set_xlabel("Predicted probability (bin mean)")
    ax.set_ylabel("Observed fraction positive")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200); plt.close(fig)


# --------------- main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--model",  required=True, type=str)
    ap.add_argument("--n-episodes", type=int, default=400)
    ap.add_argument("--outdir", type=str, default="reports/interim_main")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override cfg.pred_threshold/decision_threshold")
    args = ap.parse_args()

    raw_cfg_dict = load_yaml(args.config)
    cfg = EnvConfig(raw_cfg_dict)
    ensure_dir(args.outdir)

    # Build env (with action masking if available)
    env = SepsisEnvFast(cfg)
    if HAVE_AM and hasattr(env, "action_masks"):
        env = ActionMasker(env, _mask_fn)

    device = torch.device(getattr(cfg, "device", "cpu"))
    model = None
    if HAVE_MASK:
        try: model = MaskablePPO.load(args.model, device=device)
        except Exception: model = PPO.load(args.model, device=device)
    else:
        model = PPO.load(args.model, device=device)

    base = unwrap_env(env)
    num_groups = int(getattr(base, "num_test_groups", 4))
    min_tests  = int(getattr(cfg, "min_tests_before_diagnosis", 2))
    threshold  = (float(args.threshold) if args.threshold is not None
                  else float(getattr(cfg, "pred_threshold", getattr(cfg, "decision_threshold", 0.5))))

    # Names & costs
    pn = {int(k): str(v) for k, v in (raw_cfg_dict.get("panel_names", {}) or {}).items()}
    costs = {int(k): float(v) for k, v in (raw_cfg_dict.get("cost_mapping", {}) or {}).items()}

# Fallbacks
    for i in range(num_groups):
        pn.setdefault(i, f"Panel {i}")
        costs.setdefault(i, 0.0)

    # Run episodes and collect full info
    y_true, y_pred, probs, ep_costs, steps_list = [], [], [], [], []
    panel_counts = {i:0 for i in range(num_groups)}
    sequences = {}

    def is_diag(a:int) -> bool: return int(a) >= num_groups
    cheapest = sorted(range(num_groups), key=lambda k: costs.get(k, 0.0))

    for epi in range(args.n_episodes):
        obs, _ = env.reset()
        base = unwrap_env(env)
        done = False
        ordered = []
        this_cost = 0.0
        steps = 0
        label = int(getattr(base, "label", 0))
        last_p = threshold

        while not done:
            a, _ = model.predict(obs, deterministic=True)
            a = int(a)

            if is_diag(a) and len(ordered) < min_tests:
                a = next((k for k in cheapest if k not in ordered), a)

            if 0 <= a < num_groups:
                this_cost += float(costs.get(a, 0.0))
                if a not in ordered:
                    ordered.append(a)
                    panel_counts[a] += 1

            obs, r, term, trunc, info = env.step(a)
            done = bool(term or trunc)
            steps += 1
            last_p = infer_prob_expired(env, default=last_p)

        pred = 1 if last_p > threshold else 0

        # sequence key like "0→3→D1" (D1 means diagnose-as-1, D0 as 0)
        seq_key = "→".join([str(x) for x in ordered] + [f"D{pred}"])
        sequences[seq_key] = sequences.get(seq_key, 0) + 1

        y_true.append(label); y_pred.append(pred)
        probs.append(float(last_p))
        ep_costs.append(float(this_cost))
        steps_list.append(int(steps))

    # ---------- Figures ----------
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plot_confusion_matrix(cm, os.path.join(args.outdir, "cm_annotated.png"))
    plot_panel_usage(panel_counts, pn, len(y_true), os.path.join(args.outdir, "panel_usage_named.png"))
    plot_cost_hist(ep_costs, os.path.join(args.outdir, "cost_hist.png"))
    plot_prob_hist(y_true, probs, threshold, os.path.join(args.outdir, "prob_hist_by_class.png"))
    plot_roc_pr(y_true, probs, threshold, args.outdir)
    plot_calibration(y_true, probs, os.path.join(args.outdir, "calibration.png"))

    # Save per-episode details for traceability
    with open(os.path.join(args.outdir, "episodes.jsonl"), "w", encoding="utf-8") as f:
        for t, p, pr, c, s in zip(y_true, y_pred, probs, ep_costs, steps_list):
            f.write(json.dumps({"true": int(t), "pred": int(p), "p": float(pr), "cost": float(c), "steps": int(s)}) + "\n")

    # Top sequences (optional: could be used in your slides)
    seq_sorted = sorted(sequences.items(), key=lambda kv: kv[1], reverse=True)[:8]
    with open(os.path.join(args.outdir, "top_sequences.txt"), "w", encoding="utf-8") as f:
        for k, v in seq_sorted:
            f.write(f"{k}: {v}\n")

    print(f"Saved visualizations to: {args.outdir}")
    print(f"Threshold used: {threshold:.2f}")
    print(f"Mean cost: ${np.mean(ep_costs):.2f} | Mean steps: {np.mean(steps_list):.2f}")
    print("Top sequences:", seq_sorted)

if __name__ == "__main__":
    main()
