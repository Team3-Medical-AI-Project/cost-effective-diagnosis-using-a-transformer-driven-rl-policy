"""
evaluate_interim_fast.py — robust evaluator for SepsisEnvFast.

Artifacts saved to --outdir:
- metrics.json
- confusion_matrix.png
- panel_usage.png
- case_studies.txt
- roc.png
- pr.png (with operating point marker)

Usage:
  python -m src.eval.evaluate_interim_fast --config configs\sepsis_config_accfocus.yaml --model models\rl_agent_sepsis_interim_main.zip --n-episodes 400 --outdir reports\interim_accfocus
  # optional sweep report
  python -m src.eval.evaluate_interim_fast --config ... --model ... --n-episodes 400 --outdir ... --auto_threshold
"""
from __future__ import annotations
import os, sys, json, argparse
from collections import Counter


import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, average_precision_score
)

# --- Repo import path ---
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# --- SB3 (Maskable if available) ---
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

# --- Env + config wrapper ---
from src.training.sepsis_env_fast import SepsisEnvFast, Config as EnvConfig
from src.models.custom_policy import TransformerPolicyExtractor
# evaluate_interim_fast.py

# Add this line
from src.tools.ppo_speedups import set_torch_runtime_threads


# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force_threshold", type=float, default=None,
                help="Override decision threshold (0..1).")
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--n-episodes", type=int, default=400)
    ap.add_argument("--outdir", type=str, default="reports/interim_eval")
    ap.add_argument("--auto_threshold", action="store_true",
                    help="Sweep thresholds (0.00..1.00) and report best F1.")
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "cuda"],
                    help="Force policy/env device (default: auto)")
    return ap.parse_args()


# -------------------- utils --------------------
def load_yaml(path: str):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
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
    acc  = (tn + tp) / max(1, int(cm.sum()))
    prec = tp / max(1, (tp + fp))
    rec  = tp / max(1, (tp + fn))
    spec = tn / max(1, (tn + fp))
    f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    denom = 4 * prec + rec  # F2
    f2   = 0.0 if denom == 0 else (5 * prec * rec) / denom
    return dict(acc=float(acc), precision=float(prec), recall=float(rec),
                specificity=float(spec), f1=float(f1), f2=float(f2),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def _np2py(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.ndarray,)): return o.tolist()
    return o

def _mask_fn(env):
    return env.action_masks()

def unwrap_env(env):
    e = env
    seen = set()
    while hasattr(e, "env") and id(e.env) not in seen:
        seen.add(id(e))
        e = e.env
    try:
        return e.unwrapped
    except Exception:
        return e

def _safe_to_dict(x):
    """Read mapping-like objects from either dict or your Config wrapper."""
    if isinstance(x, dict):
        return x
    # your Config has ._d
    if hasattr(x, "_d") and isinstance(getattr(x, "_d"), dict):
        return getattr(x, "_d")
    # or ._store (other wrappers)
    if hasattr(x, "_store") and isinstance(getattr(x, "_store"), dict):
        return getattr(x, "_store")
    return {}


# ---------- get p(expired) from env ----------
@torch.no_grad()
def infer_prob_expired(env_like, default=0.5):
    base = unwrap_env(env_like)

    # this is maintained by SepsisEnvFast
    p_cache = getattr(base, "_prev_p", None)
    if p_cache is not None:
        try:
            return float(p_cache)
        except Exception:
            pass

    # fallback: recompute from current state
    device = torch.device(getattr(getattr(base, "cfg", None), "device", "cpu"))

    gen = getattr(base, "gain", None) or getattr(base, "generator", None)
    clf = getattr(base, "clf", None) or getattr(base, "classifier", None)
    x = getattr(base, "curr", None)
    m = getattr(base, "mask", None)
    try:
        if gen is not None and clf is not None and x is not None and m is not None:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(np.asarray(x), dtype=torch.float32)
            if not isinstance(m, torch.Tensor):
                m = torch.tensor(np.asarray(m), dtype=torch.float32)
            x = x.to(device); m = m.to(device)
            if x.dim() == 1: x = x.unsqueeze(0)
            if m.dim() == 1: m = m.unsqueeze(0)
            imputed = gen(x, m)
            logits  = clf(imputed)
            probs = torch.softmax(logits, dim=1)
            return float(probs[0, 1].item())
    except Exception:
        pass

    return float(default)


# -------------------- main --------------------
def main():
    set_torch_runtime_threads()
    args = parse_args()
    ensure_dir(args.outdir)

    # --- load config / env ---
    cfg_dict = load_yaml(args.config) or {}
    cfg = EnvConfig(cfg_dict)
    # allow overriding device from CLI
    if args.device and args.device != "auto":
        try:
            setattr(cfg, "device", args.device)
            if hasattr(cfg, "_d") and isinstance(cfg._d, dict):
                cfg._d["device"] = args.device
        except Exception:
            pass
    env = SepsisEnvFast(cfg)
    if HAVE_ACTION_MASKER and hasattr(env, "action_masks"):
        env = ActionMasker(env, _mask_fn)

    # Panel names from YAML (fall back to generic labels)
    _raw_pn = _safe_to_dict(getattr(cfg, "panel_names", {}))
    _panel_names = {int(k): str(v) for k, v in _raw_pn.items()} if _raw_pn else {
        0: "CBC", 1: "CMP", 2: "ABG", 3: "aPTT"
    }
    def _pname(i: int) -> str:
        return _panel_names.get(int(i), f"Panel {int(i)}")

    # --- load model on sensible device ---
    # Resolve policy device with CLI override
    if args.device == "cpu":
        policy_device = torch.device("cpu")
    elif args.device == "cuda":
        policy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        policy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if HAVE_MASK_ALGO:
        try:
            model = MaskablePPO.load(args.model, device=policy_device)
        except Exception:
            model = PPO.load(args.model, device=policy_device)
    else:
        model = PPO.load(args.model, device=policy_device)

    # --- derive threshold (JSON > YAML > 0.5) ---
    thr = 0.5
    chosen_name = "manual_or_default"
    thr_json_path = getattr(cfg, "threshold_json", None)
    if thr_json_path and os.path.exists(thr_json_path):
        try:
            with open(thr_json_path, "r", encoding="utf-8") as f:
                j = json.load(f)
            chosen_name = str(j.get("chosen", "threshold_f2"))
            thr = float(j.get(chosen_name, j.get("threshold_f2", 0.5)))
        except Exception:
            pass
    else:
        # YAML fallbacks
        thr = float(getattr(cfg, "pred_threshold", getattr(cfg, "decision_threshold", 0.5)))
    
    # --- optional CLI override ---
    if args.force_threshold is not None:
        thr = float(args.force_threshold)
        chosen_name = "forced"


    # --- env specifics we need for rule enforcement ---
    base = unwrap_env(env)
    print(
        "[eval] mode=", getattr(base, "diagnose_mode", None),
        "| num_groups=", getattr(base, "num_test_groups", None),
        "| n_actions=", getattr(env.action_space, "n", None),
        "| min_tests=", getattr(base, "min_tests_before_diagnosis",
                                getattr(base, "min_tests_before_dx", None)),
        "| DIAG_POS=", getattr(base, "DIAG_POS", None),
        "| DIAG_NEG=", getattr(base, "DIAG_NEG", None),
    )
    num_groups = int(getattr(base, "num_test_groups", 4))
    min_tests = int(getattr(base, "min_tests_before_dx",
                 getattr(base, "min_tests_before_diagnosis", 2)))
    
    # Costs from the live env (so eval matches training economics)
    cost_map = {int(k): float(v) for k, v in getattr(base, "cost_mapping", {}).items()}
    for k in range(num_groups):
        cost_map.setdefault(k, 0.0)

    # Build a cost map (string keys in YAML → ints)
    # raw_cm = getattr(cfg, "cost_mapping", {}) or {}
    # if hasattr(raw_cm, "to_dict"):
    #     raw_cm = raw_cm.to_dict()
    # elif not isinstance(raw_cm, dict):
    #     raw_cm = {}
    # cost_map = {int(k): float(v) for k, v in raw_cm.items()}
    # for k in range(num_groups):
    #     cost_map.setdefault(k, 0.0)
    # cheapest_order = sorted(range(num_groups), key=lambda k: cost_map.get(k, 0.0))

    def is_diag(a: int) -> bool:
        # robust: any action >= num_groups is a diagnosis attempt
        return int(a) >= num_groups

    # -------- evaluation loop --------
    y_true, y_pred, probs, costs, n_steps = [], [], [], [], []
    panels_all, case_lines = [], []

    for epi in range(int(args.n_episodes)):
        obs, _ = env.reset()
        base = unwrap_env(env)
        true_label = int(getattr(base, "label", 0))
        done = False
        total_cost = 0.0
        ordered_panels = set()
        actions_hist = []
        infos_hist = []
        last_p = 0.5

        took_terminal_diag = False
        terminal_pred = None

        while not done:
            a, _ = model.predict(obs, deterministic=True)
            a = int(a)

            # Enforce min-tests rule if the algo ignores masks
            if int(a) >= num_groups and len(ordered_panels) < min_tests:
                # pick the first unseen panel deterministically
                for k in range(num_groups):
                    if k not in ordered_panels:
                        a = k
                        break

            if 0 <= a < num_groups:
                total_cost += float(cost_map.get(a, 0.0))
                ordered_panels.add(a)
                panels_all.append(a)

            actions_hist.append(a)
            obs, reward, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
            infos_hist.append(info)

            # track probability if env provides it
            if info.get("action_type") == "diagnose":
                took_terminal_diag = True
                terminal_pred = int(info.get("pred", 1))
                if "p" in info:
                    last_p = float(info["p"])
                break

        pred = 1 if last_p > thr else 0

        y_true.append(true_label)
        y_pred.append(pred)
        probs.append(float(last_p))
        costs.append(total_cost)
        n_steps.append(len(actions_hist))

        # Per-episode completion log for better responsiveness during long runs
        print(
            f"[eval] episode {epi+1}/{args.n_episodes} finished: "
            f"steps={len(actions_hist)} cost={total_cost:.2f} p={last_p:.3f} pred={pred} true={true_label}",
            flush=True,
        )

        # short case studies
        if epi < 12:
            lines = []
            lines.append("--------------------------------------------------")
            lines.append(f"Episode {epi}")
            lines.append(f"True outcome: {'Expired (1)' if true_label==1 else 'Discharged (0)'}")
            for t, aa in enumerate(actions_hist, 1):
                step_info = infos_hist[t-1] if t-1 < len(infos_hist) else {}
                if 0 <= aa < num_groups:
                    p_b = step_info.get("p_before", None)
                    p_a = step_info.get("p_after", None)
                    if p_b is not None and p_a is not None:
                        lines.append(
                            f"Step {t}: Ordered {_pname(aa)} (#{aa}) "
                            f"(p_before={p_b:.3f} → p_after={p_a:.3f}) "
                            f"(≈${cost_map.get(int(aa),0.0):.2f})"
                        )
                    else:
                        # fallback if env didn't supply probabilities
                        lines.append(
                            f"Step {t}: Ordered {_pname(aa)} (#{aa}) "
                            f"(≈${cost_map.get(int(aa),0.0):.2f})"
                        )
                else:
                    # show terminal decision if env provided it
                    pred_str = None
                    if "pred" in step_info:
                        pred_str = "Expired" if int(step_info["pred"]) == 1 else "Discharged"
                    p_show = step_info.get("p", None)
                    if pred_str and p_show is not None:
                        lines.append(f"  Step {t}: Diagnose action ({aa}) → {pred_str} (p={p_show:.3f})")
                    else:
                        lines.append(f"  Step {t}: Diagnose action ({aa})")

            lines.append(f"Final p(expired)≈{last_p:.3f}  → Pred={pred}  @ thr={thr:.3f}")
            lines.append(f"Total cost=${total_cost:.2f} | steps={len(actions_hist)}")
            lines.append(f"Result: {'CORRECT' if pred==true_label else 'INCORRECT'}")
            case_lines.extend(lines + [""])

    # -------- metrics & curves --------
    cm = confusion_matrix(y_true, y_pred)
    m = metrics_from_cm(cm)
    m["avg_cost"] = float(np.mean(costs)) if costs else 0.0
    m["avg_steps"] = float(np.mean(n_steps)) if n_steps else 0.0
    m["n_episodes"] = int(len(y_true))
    m["threshold_used"] = float(thr)
    m["threshold_source"] = chosen_name

    # ROC / PR on episode-level probs
    fpr = tpr = prec = rec = None
    try:
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        m["AUC"] = float(roc_auc)
        m["AUPRC"] = float(ap)
    except Exception:
        pass

    # optional auto-threshold sweep for best F1
    if args.auto_threshold:
        best = {"threshold": None, "metrics": None, "f1": -1.0}
        ths = np.linspace(0.0, 1.0, 101)
        for t in ths:
            preds = [1 if p > t else 0 for p in probs]
            cm_s = confusion_matrix(y_true, preds)
            ms = metrics_from_cm(cm_s)
            if ms["f1"] > best["f1"]:
                best = {"threshold": float(t), "metrics": ms, "f1": float(ms["f1"])}
        m["auto_threshold"] = best

    # -------- save artifacts --------
    ensure_dir(args.outdir)

    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, default=_np2py)

    # confusion matrix
    plt.figure()
    plt.title("Confusion Matrix (Expired=1)")
    plt.imshow(cm, interpolation="nearest")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.xlabel("Prediction"); plt.ylabel("Truth"); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "confusion_matrix.png")); plt.close()


    # panel usage
    counts = Counter(panels_all)
    xs = list(range(num_groups)); ys = [counts.get(i, 0) for i in xs]
    labels = [_pname(i) for i in xs]
    plt.figure()
    plt.title("Panel Usage (counts over episodes)")
    plt.bar(xs, ys)
    plt.xticks(xs, labels)
    plt.xlabel("Panel"); plt.ylabel("Count used before diagnosis")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "panel_usage.png")); plt.close()

    # ROC curve
    if fpr is not None and tpr is not None:
        plt.figure()
        plt.title(f"ROC (AUC={m.get('AUC', 0.0):.3f})")
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "roc.png")); plt.close()

    # PR curve + operating point
    if prec is not None and rec is not None:
        plt.figure()
        plt.title(f"PR (AP={m.get('AUPRC', 0.0):.3f})")
        plt.plot(rec, prec, label="PR")
        # operating point at the chosen threshold
        preds_at_thr = np.array([1 if p > thr else 0 for p in probs])
        tp = np.sum((np.array(y_true)==1) & (preds_at_thr==1))
        fp = np.sum((np.array(y_true)==0) & (preds_at_thr==1))
        fn = np.sum((np.array(y_true)==1) & (preds_at_thr==0))
        prec_pt = tp / max(1, tp + fp)
        rec_pt  = tp / max(1, tp + fn)
        plt.scatter([rec_pt], [prec_pt], s=60, marker="o", label=f"op@{thr:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "pr.png")); plt.close()

    # case studies
    with open(os.path.join(args.outdir, "case_studies.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(case_lines))

    print(f"Saved evaluation artifacts to: {args.outdir}")
    np.save(os.path.join(args.outdir, "probs.npy"), np.array(probs))
    
    # --- ADD THESE TWO LINES ---
    np.save(os.path.join(args.outdir, "costs_all.npy"), np.array(costs))
    np.save(os.path.join(args.outdir, "panels_all.npy"), np.array(panels_all))
    # ---------------------------

    print(f"[debug] probs: min={float(np.min(probs)):.3f}, max={float(np.max(probs)):.3f}, mean={float(np.mean(probs)):.3f}")

    print(json.dumps(m, indent=2, default=_np2py))


if __name__ == "__main__":
    main()
