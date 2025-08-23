"""
evaluate_interim_fast.py
Robust evaluator for SepsisEnvFast.

Artifacts:
- metrics.json              (acc, precision, recall, f1, avg_cost, avg_steps, etc.)
- confusion_matrix.png
- panel_usage.png
- case_studies.txt

Key behaviors:
- Unwraps wrappers to access base env internals.
- Enforces min_tests_before_diagnosis during eval.
- Any action >= num_test_groups is treated as a "diagnose" action.
- Final label uses probability threshold: pred = 1 if p_expired > threshold else 0.
- Threshold comes from cfg.pred_threshold (else cfg.decision_threshold, else 0.5).
- Optional --auto_threshold sweeps thresholds and reports best F1.

Usage:
  python -m src.eval.evaluate_interim_fast --config configs\sepsis_config.yaml --model models\rl_agent_sepsis_interim_main.zip --n-episodes 400 --outdir reports\interim_main
  python -m src.eval.evaluate_interim_fast --config ... --model ... --n-episodes 400 --outdir ... --auto_threshold
"""
from __future__ import annotations
import os, sys, argparse, json
from collections import Counter
import numpy as np
import torch
import matplotlib.pyplot as plt

# Make repo importable
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from stable_baselines3 import PPO

HAVE_MASK_ALGO = False
try:
    from sb3_contrib import MaskablePPO
    HAVE_MASK_ALGO = True
except Exception:
    pass

HAVE_ACTION_MASKER = False
try:
    from sb3_contrib.common.wrappers import ActionMasker
    HAVE_ACTION_MASKER = True
except Exception:
    pass

from src.training.sepsis_env_fast import SepsisEnvFast, Config as EnvConfig


# ------------------------- utils -------------------------
def load_yaml(path: str):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def confusion_matrix(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def metrics_from_cm(cm):
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    acc = (tn + tp) / max(1, int(cm.sum()))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return dict(acc=float(acc), precision=float(prec), recall=float(rec), f1=float(f1),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def _np2py(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.ndarray,)): return o.tolist()
    return o

def _to_plain_dict(maybe_cfg):
    if isinstance(maybe_cfg, dict): return maybe_cfg
    if hasattr(maybe_cfg, "_d") and isinstance(maybe_cfg._d, dict): return maybe_cfg._d
    try: return dict(maybe_cfg)
    except Exception: return {}

def _mask_fn(env):
    return env.action_masks()

def unwrap_env(env):
    """Walk through wrappers (e.g., Monitor, ActionMasker) to the base env."""
    e = env
    seen = set()
    while hasattr(e, "env") and id(e.env) not in seen:
        seen.add(id(e))
        e = e.env
    try:
        return e.unwrapped
    except Exception:
        return e


# ------------------- probability extraction -------------------
@torch.no_grad()
def infer_prob_expired(env_like, default=0.5):
    """
    Compute p(expired) from the base env using its GAIN + classifier.
    Robust to attribute names; falls back to default on any error.
    """
    base = unwrap_env(env_like)

    # quick path: env may update it internally
    p_cache = getattr(base, "_prev_p", None)
    if p_cache is not None:
        try:
            return float(p_cache)
        except Exception:
            pass

    device = torch.device(getattr(getattr(base, "cfg", None), "device", "cpu"))

    # likely attribute names used across your codebase
    gen = getattr(base, "gain_generator", None) or getattr(base, "generator", None) or getattr(base, "gen", None)
    clf = getattr(base, "classifier", None) or getattr(base, "prelim_classifier", None) or getattr(base, "clf", None)

    x = getattr(base, "current_state", None) or getattr(base, "curr", None) or getattr(base, "state", None)
    m = getattr(base, "observation_mask", None) or getattr(base, "mask", None)
    full = getattr(base, "full_patient_data", None) or getattr(base, "full", None) or getattr(base, "x_full", None)

    try:
        if clf is not None and gen is not None and x is not None and m is not None:
            # Impute observed state with GAIN, then classify
            x = x if isinstance(x, torch.Tensor) else torch.tensor(np.asarray(x), dtype=torch.float32)
            m = m if isinstance(m, torch.Tensor) else torch.tensor(np.asarray(m), dtype=torch.float32)
            x = x.to(device); m = m.to(device)
            if x.dim() == 1: x = x.unsqueeze(0)
            if m.dim() == 1: m = m.unsqueeze(0)
            imputed = gen(x, m)          # (B, D)
            logits = clf(imputed)        # (B, 2)
            probs = torch.softmax(logits, dim=1)
            return float(probs[0, 1].item())
        if clf is not None and full is not None:
            # Fallback: classify the full row
            full = full if isinstance(full, torch.Tensor) else torch.tensor(np.asarray(full), dtype=torch.float32)
            full = full.to(device)
            if full.dim() == 1: full = full.unsqueeze(0)
            logits = clf(full)
            probs = torch.softmax(logits, dim=1)
            return float(probs[0, 1].item())
    except Exception:
        pass

    return float(default)


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--n-episodes", type=int, default=400)
    ap.add_argument("--outdir", type=str, default="reports/interim_eval")
    ap.add_argument("--auto_threshold", action="store_true",
                    help="Sweep thresholds (0.30..0.60) and report best F1.")
    args = ap.parse_args()

    cfg = EnvConfig(load_yaml(args.config))
    device = torch.device(getattr(cfg, "device", "cpu"))
    ensure_dir(args.outdir)

    # Build env + optional action masking wrapper
    env = SepsisEnvFast(cfg)
    if HAVE_ACTION_MASKER and hasattr(env, "action_masks"):
        env = ActionMasker(env, _mask_fn)

    # Load agent (MaskablePPO if available, else PPO)
    if HAVE_MASK_ALGO:
        try:
            model = MaskablePPO.load(args.model, device=device)
        except Exception:
            model = PPO.load(args.model, device=device)
    else:
        model = PPO.load(args.model, device=device)

    base = unwrap_env(env)
    num_groups = int(getattr(base, "num_test_groups", 4))
    # pick a usable threshold
    pred_threshold = float(getattr(cfg, "pred_threshold",
                           getattr(cfg, "decision_threshold", 0.5)))
    # tie-break strictly: > threshold (not >=)
    min_tests = int(getattr(cfg, "min_tests_before_diagnosis", 2))

    # Costs (string keys in YAML → ints here)
    cost_map = {}
    for k, v in _to_plain_dict(getattr(cfg, "cost_mapping", {})).items():
        try:
            cost_map[int(k)] = float(v)
        except Exception:
            pass
    for k in range(num_groups):
        cost_map.setdefault(k, 0.0)
    cheapest_order = sorted(range(num_groups), key=lambda k: cost_map.get(k, 0.0))

    def is_diag(a: int) -> bool:
        # Robust: any action >= num_groups is a diagnosis action
        return int(a) >= num_groups

    y_true, y_pred, costs, n_steps_list = [], [], [], []
    panels_all, case_lines, probs = [], [], []

    for epi in range(args.n_episodes):
        obs, _ = env.reset()
        base = unwrap_env(env)
        done = False
        total_cost = 0.0
        steps = 0
        ordered_panels = set()
        actions_hist = []
        true_label = int(getattr(base, "label", 0))
        last_p = float(pred_threshold)  # initialize near threshold

        while not done:
            a, _ = model.predict(obs, deterministic=True)
            a = int(a)

            # Enforce min-tests-before-diagnosis
            if is_diag(a) and len(ordered_panels) < min_tests:
                # choose cheapest un-ordered panel
                fallback = next((k for k in cheapest_order if k not in ordered_panels), None)
                if fallback is not None:
                    a = int(fallback)

            if 0 <= a < num_groups:
                total_cost += float(cost_map.get(a, 0.0))
                ordered_panels.add(a)
                panels_all.append(a)

            actions_hist.append(a)
            obs, reward, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
            steps += 1

            # Update probability from base env
            last_p = infer_prob_expired(env_like=env, default=last_p)

        # Final decision: probability threshold
        pred = 1 if last_p > pred_threshold else 0

        y_true.append(true_label)
        y_pred.append(pred)
        costs.append(total_cost)
        n_steps_list.append(steps)
        probs.append(float(last_p))

        if epi < 12:
            lines = []
            lines.append("--------------------------------------------------")
            lines.append(f"--- Case Study for Episode: {epi} ---")
            lines.append(f"True Patient Outcome: {'Expired' if true_label==1 else 'Discharged'}")
            for t, aa in enumerate(actions_hist, 1):
                if 0 <= aa < num_groups:
                    lines.append(f"Step {t}: Ordered panel {aa} (approx cost ${cost_map.get(int(aa),0.0):.2f})")
                else:
                    lines.append(f"Step {t}: Diagnose action {aa}")
            lines.append(f"Final Prediction: {'Expired' if pred==1 else 'Discharged'} (p_expired~{last_p*100:.2f}%)")
            lines.append(f"Total Cost Incurred: ${total_cost:.2f}")
            lines.append(f"Result: {'CORRECT' if pred==true_label else 'INCORRECT'}")
            lines.append("--------------------------------------------------\n")
            case_lines.extend(lines)

    # Base metrics
    cm2 = confusion_matrix(y_true, y_pred)
    m = metrics_from_cm(cm2)
    m["avg_cost"] = float(np.mean(costs)) if costs else 0.0
    m["avg_steps"] = float(np.mean(n_steps_list)) if n_steps_list else 0.0
    m["n_episodes"] = int(len(y_true))
    m["threshold_used"] = float(pred_threshold)

    # Optional: auto threshold sweep for best F1 (report only)
    if args.auto_threshold:
        ths = np.arange(0.30, 0.60 + 1e-9, 0.01)
        best = {"threshold": None, "metrics": None, "f1": -1.0}
        for th in ths:
            preds = [1 if p > th else 0 for p in probs]
            cm = confusion_matrix(y_true, preds)
            ms = metrics_from_cm(cm)
            if ms["f1"] > best["f1"]:
                best = {"threshold": float(th), "metrics": ms, "f1": float(ms["f1"])}
        m["auto_threshold"] = best

    # Save metrics
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, default=_np2py)

    # Confusion matrix plot
    plt.figure()
    plt.title("Confusion Matrix (Expired=1)")
    plt.imshow(cm2, interpolation="nearest")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm2[i, j]), ha="center", va="center")
    plt.xlabel("Prediction"); plt.ylabel("Truth"); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "confusion_matrix.png")); plt.close()

    # Panel usage plot
    counts = Counter(panels_all)
    xs = list(range(num_groups))
    ys = [counts.get(i, 0) for i in xs]
    plt.figure()
    plt.title("Panel Usage (counts over episodes)")
    plt.bar(xs, ys)
    plt.xlabel("Panel index"); plt.ylabel("Count used before diagnosis")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "panel_usage.png")); plt.close()

    # Case studies
    with open(os.path.join(args.outdir, "case_studies.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(case_lines))

    print(f"Saved evaluation artifacts to: {args.outdir}")
    print(json.dumps(m, indent=2, default=_np2py))


if __name__ == "__main__":
    main()

