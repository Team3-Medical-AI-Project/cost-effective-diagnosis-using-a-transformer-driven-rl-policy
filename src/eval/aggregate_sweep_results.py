
import json, os, glob, math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

# --------- CONFIG (edit if your names differ) ----------
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports")
MODELS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
PATTERN     = "rl_agent_sepsis_*_results.json"  # matches your sweep files
# Clinical-cost weights for threshold tuning (edit if desired)
FN_COST = 10.0   # cost of missing a death (false negative) -> push recall up
FP_COST = 1.0    # cost of a false alarm (false positive)
# Optional hard cost cap when filtering models for "best at <= cost"
COST_CAPS = [80, 120, 200, 400, 600, 800, 1000]
# ------------------------------------------------------

@dataclass
class Point:
    model_name: str
    unc: float
    cost_w: float
    # Summary metrics as logged at the time of training/eval
    acc: float
    auroc: float
    f1: float
    precision: float
    recall: float
    specificity: float
    avg_cost: float
    tests_avg: float
    # Optionally store probabilities to re-tune threshold if present
    y_true: np.ndarray = None
    y_prob: np.ndarray = None
    # Will be filled after threshold tuning
    tuned: Dict = None

def _safe_get(d: Dict, key: str, default=None):
    return d[key] if key in d and d[key] is not None else default

def load_points() -> List[Point]:
    files = sorted(glob.glob(os.path.join(REPORTS_DIR, PATTERN)))
    points: List[Point] = []
    for fp in files:
        with open(fp, "r") as f:
            R = json.load(f)

        # Try to parse uncertainty/cost from filename: rl_agent_sepsis_unc-0.1_cost-1.5_results.json
        base = os.path.basename(fp)
        unc, cw = None, None
        try:
            parts = base.split("_")
            for p in parts:
                if p.startswith("unc-"):
                    unc = float(p.replace("unc-","").replace(".json",""))
                if p.startswith("cost-"):
                    cw = float(p.replace("cost-","").replace(".json",""))
        except:
            pass

        pts = Point(
            model_name=_safe_get(R, "model_name", os.path.splitext(base)[0]),
            unc=unc if unc is not None else float(_safe_get(R, "uncertainty_factor", 0.0)),
            cost_w=cw if cw is not None else float(_safe_get(R, "cost_weight", 1.0)),
            acc=float(_safe_get(R, "accuracy", np.nan)),
            auroc=float(_safe_get(R, "auroc", np.nan)),
            f1=float(_safe_get(R, "f1", np.nan)),
            precision=float(_safe_get(R, "precision", np.nan)),
            recall=float(_safe_get(R, "recall", np.nan)),
            specificity=float(_safe_get(R, "specificity", np.nan)),
            avg_cost=float(_safe_get(R, "avg_cost", np.nan)),
            tests_avg=float(_safe_get(R, "tests_per_patient", np.nan)),
        )

        # If your evaluator saved raw probs to allow threshold re-tuning, load them:
        # Expect keys like: "y_true": [...], "y_prob": [...]
        if "y_true" in R and "y_prob" in R and len(R["y_true"]) == len(R["y_prob"]):
            pts.y_true = np.asarray(R["y_true"]).astype(int)
            pts.y_prob = np.asarray(R["y_prob"]).astype(float)

        points.append(pts)
    return points

def cost_sensitive_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                             fn_cost: float = FN_COST, fp_cost: float = FP_COST) -> Tuple[float, Dict[str,float]]:
    """
    Choose threshold t to minimize expected misclassification cost:
      E[cost] = fn_cost * P(FN) + fp_cost * P(FP)
    """
    assert y_true.ndim == 1 and y_prob.ndim == 1 and len(y_true) == len(y_prob)
    # Evaluate on candidate thresholds
    cand = np.unique(np.concatenate([np.linspace(0,1,501), y_prob]))
    best_t, best_c = 0.5, math.inf
    best = {}
    for t in cand:
        y_pred = (y_prob >= t).astype(int)
        TP = int(np.sum((y_true==1)&(y_pred==1)))
        TN = int(np.sum((y_true==0)&(y_pred==0)))
        FP = int(np.sum((y_true==0)&(y_pred==1)))
        FN = int(np.sum((y_true==1)&(y_pred==0)))
        N  = len(y_true)
        exp_cost = (fn_cost * FN + fp_cost * FP) / max(N,1)
        # derived metrics
        prec = TP / max((TP+FP),1)
        rec  = TP / max((TP+FN),1)
        spec = TN / max((TN+FP),1)
        f1   = (2*prec*rec)/max((prec+rec),1e-12)
        if exp_cost < best_c:
            best_c = exp_cost
            best_t = t
            best = dict(
                threshold=float(t),
                exp_cost=float(exp_cost),
                precision=float(prec),
                recall=float(rec),
                specificity=float(spec),
                f1=float(f1),
                tp=TP, tn=TN, fp=FP, fn=FN
            )
    return best_t, best

def pareto_front(points: List[Point], maximize_metric: str = "f1") -> List[Point]:
    """
    Standard Pareto on (avg_cost minimize, metric maximize).
    """
    valid = [p for p in points if not (np.isnan(p.avg_cost) or np.isnan(getattr(p, maximize_metric, np.nan)))]
    # sort by cost asc, break ties by metric desc
    valid.sort(key=lambda p: (p.avg_cost, -getattr(p, maximize_metric)))
    front = []
    best_metric = -np.inf
    for p in valid:
        m = getattr(p, maximize_metric)
        if m > best_metric:
            front.append(p)
            best_metric = m
    return front

def main():
    points = load_points()
    if not points:
        print("No report JSONs found. Check REPORTS_DIR/PATTERN.")
        return

    # Threshold tuning where y_prob is available
    tuned_count = 0
    for p in points:
        if p.y_true is not None and p.y_prob is not None:
            t, info = cost_sensitive_threshold(p.y_true, p.y_prob, FN_COST, FP_COST)
            p.tuned = info
            tuned_count += 1
    print(f"Tuned thresholds for {tuned_count} model(s).")

    # Write a consolidated CSV-like report to stdout (you can redirect to file)
    header = [
        "model_name","unc","cost_w","avg_cost","tests_avg",
        "acc","auroc","f1","precision","recall","specificity",
        "tuned_threshold","tuned_f1","tuned_recall","tuned_precision","tuned_specificity","tuned_exp_cost"
    ]
    print(",".join(header))
    for p in points:
        row = [
            p.model_name,
            f"{p.unc:.4g}" if p.unc is not None else "",
            f"{p.cost_w:.4g}" if p.cost_w is not None else "",
            f"{p.avg_cost:.4f}",
            f"{p.tests_avg:.4f}",
            f"{p.acc:.4f}",
            f"{p.auroc:.4f}",
            f"{p.f1:.4f}",
            f"{p.precision:.4f}",
            f"{p.recall:.4f}",
            f"{p.specificity:.4f}",
        ]
        if p.tuned:
            row += [
                f"{p.tuned['threshold']:.4f}",
                f"{p.tuned['f1']:.4f}",
                f"{p.tuned['recall']:.4f}",
                f"{p.tuned['precision']:.4f}",
                f"{p.tuned['specificity']:.4f}",
                f"{p.tuned['exp_cost']:.4f}",
            ]
        else:
            row += [""]*6
        print(",".join(row))

    # Print best models under practical cost caps
    print("\n=== Best @ cost caps (by tuned F1 where available, else original F1) ===")
    for cap in COST_CAPS:
        cand = [p for p in points if not np.isnan(p.avg_cost) and p.avg_cost <= cap]
        if not cand:
            print(f"<= ${cap}: none")
            continue
        def key(p):
            if p.tuned: return (p.tuned["f1"], p.auroc)
            return (p.f1, p.auroc)
        best = sorted(cand, key=key, reverse=True)[0]
        if best.tuned:
            print(f"<= ${cap}: {best.model_name}  | cost=${best.avg_cost:.1f}  tunedF1={best.tuned['f1']:.3f}  tunedR={best.tuned['recall']:.3f}  AUROC={best.auroc:.3f}")
        else:
            print(f"<= ${cap}: {best.model_name}  | cost=${best.avg_cost:.1f}  F1={best.f1:.3f}  R={best.recall:.3f}  AUROC={best.auroc:.3f}")

    # Show Pareto front (using original F1, because avg_cost is threshold-agnostic)
    front = pareto_front(points, "f1")
    print("\n=== Pareto Front (Cost vs. F1 from JSON) ===")
    for p in front:
        print(f"{p.model_name:55s}  cost=${p.avg_cost:7.2f}  F1={p.f1:0.3f}  AUROC={p.auroc:0.3f}  unc={p.unc}  cost_w={p.cost_w}")

if __name__ == "__main__":
    main()
