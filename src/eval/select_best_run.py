# src/eval/select_best_run.py
"""
Rank multiple evaluation folders and pick the best run.

Inputs: one or more report folders (each must contain metrics.json).
Selection rule (defaults):
  - PRIMARY metric: F2 (higher is better)
  - HARD CONSTRAINT: recall >= --min-recall (default 0.0)
  - TIE-BREAKERS (in order): F1, AUPRC, AUROC, lower avg_cost, higher specificity, higher accuracy

You can change the primary via --primary and add a cost penalty via --cost-penalty.

Outputs in --out:
  - best_run.json (full metrics + path)
  - RANKING.md (sortable table)
  - best_run/ (confusion_matrix.png, roc.png, pr.png, panel_usage.png, case_studies.txt if available)
"""
from __future__ import annotations
import os, json, argparse, shutil, math
from typing import Dict, Any, List, Tuple

IMG_FILES = ["confusion_matrix.png", "roc.png", "pr.png", "panel_usage.png", "case_studies.txt", "probs.npy"]

def load_metrics(run_dir: str) -> Dict[str, Any]:
    mpath = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"metrics.json not found in: {run_dir}")
    with open(mpath, "r", encoding="utf-8") as f:
        m = json.load(f)
    m["_run_dir"] = os.path.abspath(run_dir)
    m["_name"] = os.path.basename(os.path.normpath(run_dir))
    return m

def get(m: Dict[str, Any], key: str, default: float = 0.0) -> float:
    v = m.get(key, default)
    try:
        return float(v)
    except Exception:
        return default

def score_tuple(
    m: Dict[str, Any],
    primary: str = "f2",
    min_recall: float = 0.0,
    cost_penalty: float = 0.0
) -> Tuple:
    """
    Build a tuple for sorting (descending). First element is the primary score,
    potentially penalized by cost. Any run with recall < min_recall is demoted.
    """
    recall = get(m, "recall", 0.0)
    if recall < min_recall:
        # demote below any valid run
        return (-1e9, -1e9, -1e9, -1e9, math.inf, -1e9, -1e9)

    primary_value = {
        "f2": get(m, "f2", 0.0),
        "f1": get(m, "f1", 0.0),
        "auprc": get(m, "AUPRC", 0.0),
        "auc": get(m, "AUC", 0.0),
        "acc": get(m, "acc", 0.0),
        "precision": get(m, "precision", 0.0),
        "recall": recall,
        "specificity": get(m, "specificity", 0.0),
    }.get(primary.lower(), get(m, "f2", 0.0))

    # Optional linear cost penalty in “per-dollar” units.
    # Example: cost_penalty=0.001 subtracts 0.001 * avg_cost from the primary metric
    eff_primary = primary_value - cost_penalty * get(m, "avg_cost", 0.0)

    # Larger-is-better except cost (smaller is better).
    # Order: primary, F1, AUPRC, AUROC, -avg_cost, specificity, accuracy
    return (
        eff_primary,
        get(m, "f1", 0.0),
        get(m, "AUPRC", 0.0),
        get(m, "AUC", 0.0),
        -get(m, "avg_cost", float("inf")),
        get(m, "specificity", 0.0),
        get(m, "acc", 0.0),
    )

def copy_artifacts(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    for fn in IMG_FILES:
        sp = os.path.join(src_dir, fn)
        if os.path.exists(sp):
            shutil.copy2(sp, os.path.join(dst_dir, fn))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="One or more evaluation folders (each must contain metrics.json).")
    ap.add_argument("--out", default="reports/SEPSIS_MASTER",
                    help="Directory to write best_run.json, RANKING.md, and best_run/ assets.")
    ap.add_argument("--primary",
                    choices=["f2", "f1", "auprc", "auc", "acc", "precision", "recall", "specificity"],
                    default="f2",
                    help="Primary metric to maximize (default: f2).")
    ap.add_argument("--min-recall", type=float, default=0.0,
                    help="Hard constraint on recall; runs below this are auto-demoted.")
    ap.add_argument("--cost-penalty", type=float, default=0.0,
                    help="Subtract (cost_penalty * avg_cost) from the primary score (units: per-dollar).")
    args = ap.parse_args()

    runs: List[Dict[str, Any]] = []
    for rd in args.runs:
        try:
            runs.append(load_metrics(rd))
        except Exception as e:
            print(f"[warn] Skipping {rd}: {e}")

    if not runs:
        raise SystemExit("No valid runs with metrics.json were found.")

    # rank
    ranked = sorted(
        runs,
        key=lambda m: score_tuple(m, args.primary, args.min_recall, args.cost_penalty),
        reverse=True
    )
    best = ranked[0]

    # outputs
    os.makedirs(args.out, exist_ok=True)
    # RANKING.md
    lines = []
    lines.append(f"# Ranking (primary={args.primary}, min_recall={args.min_recall}, cost_penalty={args.cost_penalty})\n")
    lines.append("| Rank | Run | acc | f1 | f2 | AUPRC | AUROC | recall | spec | precision | avg_cost | steps | thr | src |")
    lines.append("|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|")
    for i, m in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {m['_name']} | "
            f"{get(m,'acc'):.3f} | {get(m,'f1'):.3f} | {get(m,'f2'):.3f} | "
            f"{get(m,'AUPRC'):.3f} | {get(m,'AUC'):.3f} | {get(m,'recall'):.3f} | {get(m,'specificity'):.3f} | "
            f"{get(m,'precision'):.3f} | ${get(m,'avg_cost'):.2f} | {get(m,'avg_steps'):.2f} | "
            f"{get(m,'threshold_used'):.3f} | {m.get('threshold_source','')} |"
        )
    with open(os.path.join(args.out, "RANKING.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # best_run.json
    out_json = {
        "selected_primary": args.primary,
        "min_recall": args.min_recall,
        "cost_penalty": args.cost_penalty,
        "best_run_dir": best["_run_dir"],
        "best_run_name": best["_name"],
        "best_metrics": best,
        "all_runs_sorted": [m["_name"] for m in ranked]
    }
    with open(os.path.join(args.out, "best_run.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    # copy artifacts
    copy_artifacts(best["_run_dir"], os.path.join(args.out, "best_run"))

    print(f"[ok] Selected best: {best['_name']} (from {best['_run_dir']})")
    print(f"[ok] Wrote: {os.path.join(args.out, 'best_run.json')}")
    print(f"[ok] Wrote: {os.path.join(args.out, 'RANKING.md')}")
    print(f"[ok] Copied artifacts to: {os.path.join(args.out, 'best_run')}")

if __name__ == "__main__":
    main()
