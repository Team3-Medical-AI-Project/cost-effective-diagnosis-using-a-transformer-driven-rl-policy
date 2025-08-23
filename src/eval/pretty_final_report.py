# src/eval/pretty_final_report.py
import argparse, json, os, sys, math
from collections import Counter

import yaml
import numpy as np

# Optional AUROC (uses sklearn if present)
try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def try_count_test_rows(processed_dir):
    """Best-effort: count rows in test_y.csv to print dataset size."""
    try:
        p = os.path.join(processed_dir or "", "test_y.csv")
        if not p or not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            n = sum(1 for _ in f) - 1  # header row
        return max(n, 0)
    except Exception:
        return None


def rfloat(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def rint(x, default=0):
    try:
        return int(x)
    except Exception:
        return int(default)


def parse_actions_from_seq(seq):
    """
    Fallback parser for strings like: '0→1→D1' or '0->3->D0'.
    Returns a list of integer panel actions only (diagnosis tokens ignored).
    """
    if not isinstance(seq, str):
        return []
    arrows = ["→", "->"]
    for a in arrows:
        if a in seq:
            toks = [t.strip() for t in seq.split(a)]
            break
    else:
        toks = [seq]
    out = []
    for t in toks:
        try:
            out.append(int(t))
        except Exception:
            pass  # ignore 'D0','D1','DIAG', etc.
    return out


def main():
    ap = argparse.ArgumentParser(description="Pretty, paper-style console report for a sepsis RL run.")
    ap.add_argument("--config", required=True, help="configs/sepsis_config.yaml")
    ap.add_argument("--in", dest="indir", required=True, help="evaluation folder (metrics.json, episodes.jsonl)")
    ap.add_argument("--model-name", default="", help="optional: model filename to show in the header")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    indir = args.indir

    metrics_p = os.path.join(indir, "metrics.json")
    episodes_p = os.path.join(indir, "episodes.jsonl")

    if not os.path.exists(metrics_p):
        print("[ERROR] Missing {}".format(metrics_p), file=sys.stderr)
        sys.exit(1)

    # ---------- load metrics ----------
    with open(metrics_p, "r", encoding="utf-8") as f:
        M = json.load(f)

    acc  = rfloat(M.get("acc", 0.0))
    prec = rfloat(M.get("precision", 0.0))
    rec  = rfloat(M.get("recall", 0.0))
    f1   = rfloat(M.get("f1", 0.0))
    tn   = rint(M.get("tn", 0))
    fp   = rint(M.get("fp", 0))
    fn   = rint(M.get("fn", 0))
    tp   = rint(M.get("tp", 0))
    n_ep = rint(M.get("n_episodes", 0))
    thr  = M.get("threshold_used", None)

    # ---------- config-driven bits ----------
    n_groups = rint(cfg.get("num_test_groups", 4))
    processed_dir = cfg.get("processed_data_dir", "")
    diag_mode = str(cfg.get("diagnose_mode", "split")).lower()

    # Panel names
    default_names = {str(i): "Panel {}".format(i) for i in range(n_groups)}
    panel_names_raw = default_names.copy()
    try:
        pn = cfg.get("panel_names", {}) or {}
        for k, v in pn.items():
            panel_names_raw[str(k)] = str(v)
    except Exception:
        pass
    panel_names = {}
    for k, v in panel_names_raw.items():
        try:
            panel_names[int(k)] = str(v)
        except Exception:
            pass

    # ---------- read episodes ----------
    probs, labels, costs, steps = [], [], [], []
    order_counts = Counter()       # total orders per panel (counts repeats)
    patient_used_panel = Counter() # #patients with panel ≥1
    n_patients_for_usage = 0

    if os.path.exists(episodes_p):
        with open(episodes_p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ep = json.loads(line)
                except Exception:
                    continue

                y = rint(ep.get("true", ep.get("label", 0)))
                p = rfloat(ep.get("p", ep.get("prob", 0.5)))
                c = rfloat(ep.get("cost", ep.get("total_cost", 0.0)))
                s = rint(ep.get("steps", ep.get("n_steps", 0)))

                labels.append(y)
                probs.append(p)
                costs.append(c)
                steps.append(s)

                panels_this_patient = set()
                acts = ep.get("actions", None)
                if isinstance(acts, list):
                    for a in acts:
                        try:
                            ai = int(a)
                        except Exception:
                            continue
                        if ai < n_groups:
                            order_counts[ai] += 1
                            panels_this_patient.add(ai)
                else:
                    seq = ep.get("seq") or ep.get("sequence")
                    pans = parse_actions_from_seq(seq) if seq else []
                    for ai in pans:
                        if ai < n_groups:
                            order_counts[ai] += 1
                            panels_this_patient.add(ai)

                for ai in panels_this_patient:
                    patient_used_panel[ai] += 1

                n_patients_for_usage += 1

    # ---------- AUROC ----------
    auc_roc = None
    if probs and labels and roc_auc_score is not None and len(set(labels)) > 1:
        try:
            auc_roc = float(roc_auc_score(labels, probs))
        except Exception:
            auc_roc = None

    # ---------- cost stats ----------
    mean_cost = float(np.mean(costs)) if costs else rfloat(M.get("avg_cost", 0.0))
    median_cost = float(np.median(costs)) if costs else mean_cost
    if costs:
        q25, q75 = np.percentile(costs, [25, 75]).tolist()
    else:
        q25, q75 = 0.0, 0.0

    mean_steps = float(np.mean(steps)) if steps else rfloat(M.get("avg_steps", 0.0))

    # ---------- usage summaries ----------
    panels_sorted = sorted(order_counts.items(), key=lambda kv: (-kv[1], kv[0]))

    usage_lines = []
    if n_patients_for_usage > 0:
        for i in range(n_groups):
            name = panel_names.get(i, "Panel {}".format(i))
            used = patient_used_panel.get(i, 0)
            pct = 100.0 * used / n_patients_for_usage
            usage_lines.append((name, pct))

    # ---------- header ----------
    model_str = args.model_name if args.model_name else "Agent"
    print("--- Evaluating {} ---".format(model_str))

    total_patients = try_count_test_rows(processed_dir)
    if total_patients is not None:
        print("Loaded TEST data with {} patients.".format(total_patients))
    if n_ep:
        print("Evaluated {} episodes.".format(n_ep))
    print("Trained agent loaded successfully.\n")

    # ---------- compact performance block ----------
    print("==================================================")
    print("--- Final Performance Report ---")
    print("==================================================")
    print("📈 Final Diagnostic Accuracy: {:.2f}%".format(acc * 100.0))
    print("💰 Average Financial Cost per Patient: ${:.2f}".format(mean_cost))
    print("🧪 Average Number of Tests Ordered: {:.2f}".format(mean_steps))
    if auc_roc is not None:
        print("📐 AUROC: {:.3f}".format(auc_roc))
    if isinstance(thr, (int, float)):
        print("🔧 Decision threshold used: {:.3f}".format(float(thr)))
    print()

    # ---------- testing strategy ----------
    print("--- Agent's Testing Strategy ---")
    print("==================================================")
    if panels_sorted:
        for pid, cnt in panels_sorted:
            name = panel_names.get(pid, "Panel {}".format(pid))
            print("  - {}: {} times".format(name, cnt))
    else:
        for i in range(n_groups):
            name = panel_names.get(i, "Panel {}".format(i))
            print("  - {}: 0 times".format(name))
    print("==================================================\n")

    # ---------- paper-style metrics ----------
    spec = (float(tn) / float(tn + fp)) if (tn + fp) > 0 else 0.0
    print("--- Evaluation (Paper-style metrics) ---")
    print("==================================================")
    print("F1-score: {:.3f}".format(f1))
    if auc_roc is not None:
        print("AUROC: {:.3f}".format(auc_roc))
    print("Precision: {:.3f}".format(prec))
    print("Recall (Sensitivity): {:.3f}  Specificity: {:.3f}".format(rec, spec))
    print("Confusion Matrix: TP={}, FP={}, TN={}, FN={}".format(tp, fp, tn, fn))
    print("Cost (mean/median, IQR): ${:.2f} / ${:.2f}, IQR=${:.2f}".format(
        mean_cost, median_cost, (q75 - q25)))
    print("IQR= [${:.2f}, ${:.2f}]".format(q25, q75))
    print("Panel usage rate (% of patients with panel selected at least once): -")
    if usage_lines:
        for name, pct in sorted(usage_lines, key=lambda x: -x[1]):
            print("- {}: {:.2f}%".format(name, pct))
    else:
        for i in range(n_groups):
            name = panel_names.get(i, "Panel {}".format(i))
            print("- {}: 0.00%".format(name))
    print("==================================================")


if __name__ == "__main__":
    main()
