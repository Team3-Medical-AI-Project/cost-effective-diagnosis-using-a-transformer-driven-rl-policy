# src/eval/evaluate_rl_agent_aki.py
# YAML-driven evaluation (panels, costs, diagnose_mode), threshold control, case studies, sweep

import os, sys, json, argparse, math
from collections import Counter
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import yaml

# repo import
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.gain import Generator
from src.models.classifier import PreliminaryClassifier
# --- safe, chatty loader for .pth (works across torch versions)
def _safe_load_state_dict(path, device):
    import time, os, torch
    sz = os.path.getsize(path) if os.path.exists(path) else -1
    print(f"[load] {path} (size={sz} bytes) ... ", end="", flush=True)
    t0 = time.time()
    try:
        sd = torch.load(path, map_location=device, weights_only=True)  # torch>=2.4
    except TypeError:
        sd = torch.load(path, map_location=device)  # older torch
    print(f"ok in {time.time()-t0:.2f}s")
    return sd



def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def safe_div(a, b): return (a / b) if b else 0.0


class EvalEnvAKI(gym.Env):
    metadata = {}

    def __init__(self, cfg: dict, split: str, device: str = "cpu", thr: float = None):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.device = torch.device(device)
        self.thr = float(thr) if thr is not None else float(cfg.get("decision_threshold", 0.5))

        data_dir = cfg.get("processed_data_dir", "data/processed/aki")
        Xp = os.path.join(data_dir, f"{split}_X.csv")
        yp = os.path.join(data_dir, f"{split}_y.csv")
        self.X = pd.read_csv(Xp)
        self.y = pd.read_csv(yp).iloc[:, 0].values.astype(int)
        self.N, self.D = self.X.shape

        # --- panels/groups/costs/names from YAML (exactly what training used) ---
        raw_groups = cfg.get("feature_groups", {})
        self.groups = {int(k): list(v) for k, v in raw_groups.items()}
        self.group_ids = sorted(self.groups.keys())
        self.n_panels = len(self.group_ids)

        raw_costs = cfg.get("cost_mapping", {})
        self.panel_costs = {int(k): float(v) for k, v in raw_costs.items()}

        names_map = cfg.get("panel_names", {})
        self.panel_names = []
        for gid in self.group_ids:
            nm = names_map.get(str(gid), names_map.get(gid, f"panel_{gid}"))
            self.panel_names.append(nm)

        # optional: require some panels before diagnosis
        self.required_before_diag = set(int(x) for x in cfg.get("required_before_diag", []))

        # models
        G_path = cfg.get("gain_generator_path", "models/generator_aki.pth")
        C_path = cfg.get("prelim_classifier_path", "models/classifier_aki.pth")
        self.G = Generator(input_dim=self.D).to(self.device).eval()
        self.G.load_state_dict(_safe_load_state_dict(G_path, self.device), strict=False)

        self.C = PreliminaryClassifier(input_dim=self.D, output_dim=1).to(self.device).eval()
        self.C.load_state_dict(_safe_load_state_dict(C_path, self.device), strict=False)


        # diagnose mode
        self.diagnose_mode = str(cfg.get("diagnose_mode", "split")).lower()
        if self.diagnose_mode not in ("split", "single"):
            self.diagnose_mode = "split"

        self.min_before_diag = int(cfg.get("min_tests_before_diagnosis", 3))

        # gym spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.D * 2,), dtype=np.float32)
        if self.diagnose_mode == "split":
            self.DIAG_POS = self.n_panels
            self.DIAG_NEG = self.n_panels + 1
            self.action_space = spaces.Discrete(self.n_panels + 2)
        else:
            self.DIAG = self.n_panels
            self.action_space = spaces.Discrete(self.n_panels + 1)

        # episode ptrs
        self.order = np.arange(self.N)  # deterministic order
        self.ptr = 0
        self._reset_buffers()

    # -------- Mask for MaskablePPO --------
    def action_masks(self):
        if self.diagnose_mode == "split":
            mask = np.ones(self.n_panels + 2, dtype=np.int8)
        else:
            mask = np.ones(self.n_panels + 1, dtype=np.int8)

        # prevent re-ordering same panel
        for i in range(self.n_panels):
            if self.ordered_panels[i]:
                mask[i] = 0

        # gate diagnosis until min panels + required panels (if any)
        allow_diag = self.num_orders >= self.min_before_diag
        if self.required_before_diag:
            allow_diag = allow_diag and all(self.ordered_panels[self.group_ids.index(g)] for g in self.required_before_diag if g in self.group_ids)

        if self.diagnose_mode == "split":
            if not allow_diag:
                mask[self.DIAG_POS] = 0
                mask[self.DIAG_NEG] = 0
        else:
            if not allow_diag:
                mask[self.DIAG] = 0

        return mask

    # -------- buffers / reset / step --------
    def _reset_buffers(self):
        self.idx = None
        self.row = None
        self.label = None
        self.state = None
        self.fmask = None
        self.ordered_panels = None
        self.num_orders = 0
        self.steps = 0
        self.total_cost = 0.0
        self.panels_taken = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_buffers()
        if self.ptr >= self.N:
            self.ptr = 0
        self.idx = int(self.order[self.ptr]); self.ptr += 1

        self.row = torch.tensor(self.X.iloc[self.idx].values, dtype=torch.float32, device=self.device)
        self.label = int(self.y[self.idx])

        self.state = torch.zeros(self.D, dtype=torch.float32, device=self.device)
        self.fmask = torch.zeros(self.D, dtype=torch.float32, device=self.device)
        self.ordered_panels = np.zeros(self.n_panels, dtype=np.int8)

        obs = torch.cat([self.state, self.fmask]).cpu().numpy().astype(np.float32)
        return obs, {}

    def _infer_prob(self):
        with torch.no_grad():
            x_hat = self.G(self.state.unsqueeze(0), self.fmask.unsqueeze(0)).squeeze(0)
            logits = self.C(x_hat.unsqueeze(0))
            return torch.sigmoid(logits).item()

    def step(self, action: int):
        done = False; reward = 0.0; info = {}
        if (self.diagnose_mode == "split" and action in (self.DIAG_POS, self.DIAG_NEG)) or \
           (self.diagnose_mode == "single" and action == self.DIAG):
            prob_1 = self._infer_prob()
            if self.diagnose_mode == "split":
                pred = 1 if action == self.DIAG_POS else 0
            else:
                pred = 1 if prob_1 >= self.thr else 0
            correct = int(pred == self.label)
            reward = 100.0 if correct else -120.0
            done = True
            info = {
                "patient_idx": self.idx,
                "true_label": self.label,
                "pred_label": pred,
                "prob_1": float(prob_1),
                "n_panels": int(self.num_orders),
                "total_cost": float(self.total_cost),
                "reward": float(reward),
                "panels_taken": [self.panel_names[i] for i in self.panels_taken]
            }
        else:
            # order a panel index within self.group_ids
            pid = self.group_ids[int(action)]
            idxs = self.groups[pid]
            for j in idxs:
                if self.fmask[j] == 0:
                    self.state[j] = self.row[j]
                    self.fmask[j] = 1.0
            if self.ordered_panels[int(action)] == 0:
                self.ordered_panels[int(action)] = 1
                self.num_orders += 1
                self.panels_taken.append(int(action))
                self.total_cost += float(self.panel_costs.get(pid, 0.0))

        self.steps += 1
        if self.num_orders >= self.n_panels:
            done = True

        # Safety: if still not done after trying a bunch of steps, force a diagnosis
        if not done and self.steps >= (self.n_panels + 2):
            prob_1 = self._infer_prob()
            if self.diagnose_mode == "split":
                pred = 1 if prob_1 >= 0.5 else 0
            else:
                pred = 1 if prob_1 >= self.thr else 0
            correct = int(pred == self.label)
            reward = 100.0 if correct else -120.0
            done = True
            info = {
                "patient_idx": self.idx,
                "true_label": self.label,
                "pred_label": pred,
                "prob_1": float(prob_1),
                "n_panels": int(self.num_orders),
                "total_cost": float(self.total_cost),
                "reward": float(reward),
                "panels_taken": [self.panel_names[i] for i in self.panels_taken]
            }

        obs = torch.cat([self.state, self.fmask]).cpu().numpy().astype(np.float32)
        return obs, float(reward), bool(done), False, info



def mask_fn(env):  # for ActionMasker
    return env.action_masks()


def summarize_episode_rows(rows, panel_names):
    df = pd.DataFrame(rows)
    y_true = df["true_label"].astype(int).values
    y_pred = df["pred_label"].astype(int).values
    y_prob = df["prob_1"].astype(float).values

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = safe_div(tn, tn + fp)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try: auroc = roc_auc_score(y_true, y_prob)
    except: auroc = float("nan")
    try: auprc = average_precision_score(y_true, y_prob)
    except: auprc = float("nan")

    costs = df["total_cost"].tolist() if len(df) else []
    n_panels = df["n_panels"].tolist() if len(df) else []
    rewards = df["reward"].tolist() if len(df) else []

    usage = Counter()
    for seq in df["panels_taken"]:
        for a in seq:
            # accept either an index or an already-resolved name
            try:
                name = panel_names[int(a)]
            except Exception:
                name = str(a)
            usage[name] += 1
    usage = {k: safe_div(v, len(df)) for k, v in usage.items()}

    out = {
        "n_episodes": int(len(df)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "accuracy": float(acc), "precision": float(prec), "recall_sensitivity": float(rec),
        "specificity": float(spec), "f1": float(f1),
        "auroc": float(auroc), "auprc": float(auprc),
        "avg_num_panels": float(np.mean(n_panels)) if n_panels else 0.0,
        "avg_cost": float(np.mean(costs)) if costs else 0.0,
        "median_cost": float(np.median(costs)) if costs else 0.0,
        "p25_cost": float(np.percentile(costs, 25)) if costs else 0.0,
        "p75_cost": float(np.percentile(costs, 75)) if costs else 0.0,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "panel_usage_rate": usage,
        "panel_names": panel_names,
    }
    return df, out


def pretty_summary(split, s):
    print(f"\n=== AKI RL Evaluation — {split.upper()} ===")
    print(f"Episodes: {s['n_episodes']}")
    print(f"Confusion Matrix TP:{s['TP']} TN:{s['TN']} FP:{s['FP']} FN:{s['FN']}")
    print(f"Metrics: Acc={s['accuracy']:.3f} | Prec={s['precision']:.3f} | "
          f"Rec={s['recall_sensitivity']:.3f} | Spec={s['specificity']:.3f} | "
          f"F1={s['f1']:.3f} | AUROC={s['auroc']:.3f} | AUPRC={s['auprc']:.3f}")
    print(f"Efficiency: #Panels={s['avg_num_panels']:.2f} | "
          f"Cost(avg/med)={s['avg_cost']:.2f}/{s['median_cost']:.2f} "
          f"[p25={s['p25_cost']:.2f}, p75={s['p75_cost']:.2f}] | Reward={s['avg_reward']:.2f}")
    top = sorted(s["panel_usage_rate"].items(), key=lambda kv: kv[1], reverse=True)[:5]
    if top:
        print("Top panels:", ", ".join([f"{k} ({v*100:.1f}%)" for k, v in top]))
# --- NEW: pretty paper-style report ---
def paper_style_report(split, s, thr=None, currency="$"):
    n = int(s["n_episodes"])
    acc = s["accuracy"] * 100.0
    avg_cost, med_cost = s["avg_cost"], s["median_cost"]
    p25, p75 = s["p25_cost"], s["p75_cost"]
    iqr = p75 - p25
    avg_tests = s["avg_num_panels"]
    avg_reward = s["avg_reward"]
    f1, auroc = s["f1"], s["auroc"]
    prec, rec, spec = s["precision"], s["recall_sensitivity"], s["specificity"]
    tp, tn, fp, fn = s["TP"], s["TN"], s["FP"], s["FN"]
    usage = s.get("panel_usage_rate", {})
    names = s.get("panel_names", [])

    # absolute counts from rates
    usage_items = []
    for name, rate in sorted(usage.items(), key=lambda kv: kv[1], reverse=True):
        usage_items.append((name, int(round(rate * n)), rate))

    print("\n==================================================")
    print("--- Final Performance Report ---")
    print("==================================================")
    print(f"📈 Final Diagnostic Accuracy: {acc:.2f}%")
    print(f"💰 Average Financial Cost per Patient: {currency}{avg_cost:.2f}")
    print(f"📉 Average Number of Tests Ordered: {avg_tests:.2f}")
    print(f"📊 Average Reward: {avg_reward:.2f}\n")

    print("==================================================")
    print("--- Agent's Testing Strategy ---")
    print("==================================================")
    for name, count, _rate in usage_items:
        print(f"  - {name}: {count} times")
    print("==================================================\n")

    print("--- Evaluation (Paper-style metrics) ---")
    print("==================================================")
    print(f"F1-score: {f1:.3f}")
    print(f"AUROC: {auroc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall (Sensitivity): {rec:.3f}  Specificity: {spec:.3f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"Cost (mean/median, IQR): {currency}{avg_cost:.2f} / {currency}{med_cost:.2f}, IQR={currency}{iqr:.2f}")
    print(f"IQR= [{currency}{p25:.2f}, {currency}{p75:.2f}]")
    if thr is not None:
        print(f"Calibrated threshold used for guidance: {thr:.3f}")

    print("Panel usage rate (% of patients with panel selected at least once):")
    for name, count, rate in usage_items:
        print(f"- {name}: {rate*100:.2f}%")
    print("==================================================")



def run_split(agent, cfg, split, out_dir, device, thr=None, max_episodes=None, case_studies=0):
    env_core = EvalEnvAKI(cfg, split=split, device=device, thr=thr)
    env = ActionMasker(env_core, mask_fn)
    
    print("-----------------------------------------")
    print(f"--- Evaluating Final AKI Agent ({split.upper()}) ---")
    print(f"Loaded {split.upper()} data with {env_core.N} patients.")
    print("Trained agent loaded successfully.\n")


    os.makedirs(out_dir, exist_ok=True)
    rows = []

    obs, _ = env.reset()
    n_target = env_core.N if not max_episodes else min(max_episodes, env_core.N)
    print(f"[eval] {split}: evaluating {n_target} episodes...", flush=True)
    ep = 0
    while ep < n_target:
        action, _ = agent.predict(obs, deterministic=True, action_masks=env_core.action_masks())
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            if not info:
                # force a diagnosis at episode end using current prob_1 + threshold
                _thr = thr if thr is not None else float(cfg.get("decision_threshold", 0.5))
                prob_1 = float(env_core._infer_prob())
                pred = int(prob_1 >= _thr)
                info = {
                    "patient_idx": int(env_core.idx),
                    "true_label": int(env_core.label),
                    "pred_label": pred,
                    "prob_1": prob_1,
                    "n_panels": int(env_core.num_orders),
                    "total_cost": float(env_core.total_cost),
                    "reward": float(100.0 if pred == env_core.label else -120.0),
                    "panels_taken": [env_core.panel_names[i] for i in env_core.panels_taken],
                }
            rows.append(info)
            ep += 1
            if ep % 500 == 0:
                print(f"[eval:{split}] {ep}/{n_target} episodes done", flush=True)
            obs, _ = env.reset()



    df, s = summarize_episode_rows(rows, env_core.panel_names)
    csv_path = os.path.join(out_dir, f"aki_rl_eval_{split}.csv")
    json_path = os.path.join(out_dir, f"aki_rl_eval_{split}_summary.json")
    df.to_json(json_path.replace("_summary.json", "_episodes.json"), orient="records", indent=2)
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f: json.dump(s, f, indent=2)
    pretty_summary(split, s)
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    paper_style_report(split, s, thr=thr, currency="$")


    # ---- case studies (step-by-step) ----
    if case_studies and case_studies > 0:
        k = int(case_studies)
        step_rows = []
        env_core.ptr = 0  # start from beginning deterministically
        for _ in range(min(k, env_core.N)):
            obs, _ = env.reset()
            done = False
            step_no = 0
            while not done and step_no < env_core.n_panels + 2:
                action, _ = agent.predict(obs, deterministic=True)
                # log before step
                step_rows.append({
                    "split": split, "patient_idx": int(env_core.idx), "step": step_no,
                    "action": int(action),
                    "action_label": (
                        env_core.panel_names[action] if int(action) < env_core.n_panels else
                        ("DIAG_POS" if (env_core.diagnose_mode=="split" and int(action)==env_core.DIAG_POS)
                         else "DIAG_NEG" if (env_core.diagnose_mode=="split" and int(action)==env_core.DIAG_NEG)
                         else "DIAG")
                    ),
                    "num_orders_so_far": int(env_core.num_orders),
                    "cost_so_far": float(env_core.total_cost),
                    "prob_1_now": float(env_core._infer_prob()),
                })
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
                step_no += 1
        step_csv = os.path.join(out_dir, f"aki_case_studies_{split}.csv")
        pd.DataFrame(step_rows).to_csv(step_csv, index=False)
        print(f"Saved case studies: {step_csv}")

    return s


def sweep_threshold(agent, cfg, device, split="val"):
    # Evaluate across thresholds and pick the best F1 (tie-break by balanced accuracy)
    thrs = np.linspace(0.3, 0.85, 24)
    best = None
    for t in thrs:
        s = run_split(agent, cfg, split, out_dir="reports/eval/aki/_sweep_tmp", device=device, thr=t, max_episodes=None, case_studies=0)
        bal_acc = 0.5 * (s["recall_sensitivity"] + s["specificity"])
        score = (s["f1"], bal_acc)
        if best is None or score > best[0]:
            best = (score, t, s)
    # Clean tmp (optional)
    print("\n--- Threshold sweep (VAL) ---")
    print(f"Best thr={best[1]:.3f} | F1={best[2]['f1']:.3f} | Acc={best[2]['accuracy']:.3f} | "
          f"Rec={best[2]['recall_sensitivity']:.3f} | Spec={best[2]['specificity']:.3f} | "
          f"AUROC={best[2]['auroc']:.3f} | AUPRC={best[2]['auprc']:.3f}")
    return best[1]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/aki_config_accfocus.yaml")
    ap.add_argument("--model_path", type=str, default="models/rl_agent_aki_final.zip")
    ap.add_argument("--splits", type=str, nargs="+", default=["val"], choices=["train","val","test"])
    ap.add_argument("--out_dir", type=str, default="reports/eval/aki")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    ap.add_argument("--thr", type=float, default=None, help="Override decision threshold (single mode)")
    ap.add_argument("--max_episodes", type=int, default=None, help="For quick smoke tests")
    ap.add_argument("--case_studies", type=int, default=0, help="Save step-by-step logs for K patients")
    ap.add_argument("--sweep_threshold", action="store_true", help="Find best threshold on VAL and print it")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

# Load agent
    agent = MaskablePPO.load(args.model_path, device=args.device)

    # --- Auto-align diagnose_mode with the trained policy's action space ---
    n_panels = len(cfg.get("feature_groups", {}))
    n_actions_agent = int(agent.action_space.n)
    expected_single = n_panels + 1
    expected_split  = n_panels + 2

    if n_actions_agent == expected_split:
        inferred_mode = "split"
    elif n_actions_agent == expected_single:
        inferred_mode = "single"
    else:
        raise RuntimeError(
            f"Action-space mismatch: agent has {n_actions_agent} actions but config "
            f"has {n_panels} panels (expect {expected_single} for single or {expected_split} for split)."
        )

    if cfg.get("diagnose_mode", "split") != inferred_mode:
        print(f"[auto] overriding diagnose_mode: {cfg.get('diagnose_mode')} -> {inferred_mode} to match agent ({n_actions_agent} actions)")
        cfg["diagnose_mode"] = inferred_mode


    print("========== AKI RL Agent Evaluation ==========")
    print(f"Model:  {args.model_path}")
    print(f"Config: {args.config}")
    print(f"Splits: {', '.join(args.splits)}")
    print(f"Device: {args.device}")
    print(f"Diagnose mode: {cfg.get('diagnose_mode','split')}  | min_tests_before_diagnosis: {cfg.get('min_tests_before_diagnosis',3)}")
    if args.thr is not None:
        print(f"Decision threshold override: {args.thr:.3f}")

    if args.sweep_threshold:
        best_thr = sweep_threshold(agent, cfg, args.device, split="val")
        print(f"\nRecommended --thr {best_thr:.3f} (from validation sweep)")

    summaries = {}
    for split in args.splits:
        s = run_split(agent, cfg, split, args.out_dir, device=args.device,
                      thr=args.thr, max_episodes=args.max_episodes, case_studies=args.case_studies)
        summaries[split] = s

    if len(summaries) > 1:
        print("\n=== Combined Summary (key metrics) ===")
        for split, s in summaries.items():
            print(f"{split:>5}: Acc={s['accuracy']:.3f}, F1={s['f1']:.3f}, "
                  f"AUROC={s['auroc']:.3f}, AUPRC={s['auprc']:.3f}, "
                  f"#Panels={s['avg_num_panels']:.2f}, Cost={s['avg_cost']:.2f}")


if __name__ == "__main__":
    main()
