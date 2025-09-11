# src/analysis/aki_plots.py
# Produce paper-style figures/tables from evaluator outputs.
# Inputs: reports/eval/aki/aki_rl_eval_<split>_episodes.json + ..._summary.json (+ optional case studies CSV)

import os, json, argparse, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_split(report_dir, split):
    ep_path   = os.path.join(report_dir, f"aki_rl_eval_{split}_episodes.json")
    sum_path  = os.path.join(report_dir, f"aki_rl_eval_{split}_summary.json")
    if not os.path.exists(ep_path):
        raise FileNotFoundError(f"Missing episodes JSON: {ep_path}")
    if not os.path.exists(sum_path):
        raise FileNotFoundError(f"Missing summary JSON: {sum_path}")
    df = pd.read_json(ep_path)
    with open(sum_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return df, summary

def metrics_at_threshold(df, thr):
    y_true = df["true_label"].astype(int).values
    y_prob = df["prob_1"].astype(float).values
    y_pred = (y_prob >= float(thr)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    acc  = (tp + tn) / (tp + tn + fp + fn)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    spec = (tn / (tn + fp)) if (tn + fp) else 0.0
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, acc=acc, prec=prec, rec=rec, f1=f1, spec=spec)

def fig_class_balance(df, out, title):
    y_true = df["true_label"].astype(int)
    counts = y_true.value_counts().reindex([0,1]).fillna(0).astype(int)
    total = counts.sum()
    pos_rate = counts.get(1,0) / total if total else 0.0
    fig, ax = plt.subplots(figsize=(4.5,3.2))
    ax.bar(["Negative (no AKI)","Positive (AKI)"], [counts.get(0,0), counts.get(1,0)])
    ax.set_ylabel("Patients")
    ax.set_title(title + " — Class balance")
    ax.text(1, counts.get(1,0), f"Prevalence = {pos_rate:.3f}", ha="center", va="bottom")
    plt.savefig(out); plt.close(fig)

def fig_roc_pr(df, out_roc, out_pr, title):
    y_true = df["true_label"].astype(int).values
    y_prob = df["prob_1"].astype(float).values
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)

    fig1, ax1 = plt.subplots(figsize=(4.5,3.2))
    ax1.plot(fpr, tpr, lw=2)
    ax1.plot([0,1],[0,1],"--", lw=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"{title} — ROC (AUC={roc_auc:.3f})")
    plt.savefig(out_roc); plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(4.5,3.2))
    ax2.plot(rec, prec, lw=2)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"{title} — PR (AUC={pr_auc:.3f})")
    plt.savefig(out_pr); plt.close(fig2)

def fig_confmat(df, thr, out, title):
    y_true = df["true_label"].astype(int).values
    y_prob = df["prob_1"].astype(float).values
    y_pred = (y_prob >= float(thr)).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    fig, ax = plt.subplots(figsize=(3.6,3.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred 0","Pred 1"]); ax.set_yticklabels(["True 0","True 1"])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center", color="black")
    ax.set_title(f"{title} — Confusion (thr={thr:.2f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.savefig(out); plt.close(fig)

def fig_cost_distribution(df, out_hist, out_box, title):
    costs = df["total_cost"].astype(float).values
    fig1, ax1 = plt.subplots(figsize=(4.5,3.2))
    ax1.hist(costs, bins=40)
    ax1.set_xlabel("Cost ($)")
    ax1.set_ylabel("Patients")
    ax1.set_title(title + " — Cost distribution")
    plt.savefig(out_hist); plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(2.8,3.6))
    ax2.boxplot(costs, vert=True, showmeans=True)
    ax2.set_ylabel("Cost ($)")
    ax2.set_title(title + " — Cost (box)")
    plt.savefig(out_box); plt.close(fig2)

def fig_panel_usage(df, out, title):
    # episodes JSON stores list of panel *names* in "panels_taken"
    counts = {}
    for lst in df["panels_taken"]:
        for name in lst:
            counts[name] = counts.get(name, 0) + 1
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k,_ in items]
    vals  = [v for _,v in items]
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.barh(names[::-1], vals[::-1])
    ax.set_xlabel("Times selected across all episodes")
    ax.set_title(title + " — Panel usage")
    plt.tight_layout(); plt.savefig(out); plt.close(fig)

def fig_co_usage(df, out, title):
    # co-occurrence matrix over panel names
    panels = sorted({name for lst in df["panels_taken"] for name in lst})
    idx = {p:i for i,p in enumerate(panels)}
    M = np.zeros((len(panels), len(panels)), dtype=int)
    for lst in df["panels_taken"]:
        unique = sorted(set(lst))
        for i in range(len(unique)):
            for j in range(len(unique)):
                M[idx[unique[i]], idx[unique[j]]] += 1
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(M, cmap="Purples")
    ax.set_xticks(range(len(panels))); ax.set_yticks(range(len(panels)))
    ax.set_xticklabels(panels, rotation=90); ax.set_yticklabels(panels)
    ax.set_title(title + " — Panel co-usage")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.savefig(out); plt.close(fig)

def fig_calibration(df, out, title):
    y_true = df["true_label"].astype(int).values
    y_prob = df["prob_1"].astype(float).values
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(4.5,3.2))
    ax.plot(prob_pred, prob_true, marker="o")
    ax.plot([0,1],[0,1], "--")
    ax.set_xlabel("Predicted probability (mean in bin)")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(title + " — Calibration curve")
    plt.savefig(out); plt.close(fig)

def fig_threshold_sweep(df, out, title):
    y_true = df["true_label"].astype(int).values
    y_prob = df["prob_1"].astype(float).values
    thrs = np.linspace(0.05, 0.95, 37)
    f1s, precs, recs = [], [], []
    for t in thrs:
        y_pred = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))
    fig, ax = plt.subplots(figsize=(5.2,3.4))
    ax.plot(thrs, f1s, label="F1")
    ax.plot(thrs, precs, label="Precision")
    ax.plot(thrs, recs, label="Recall")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0,1)
    ax.legend()
    ax.set_title(title + " — Threshold sweep (from stored probs)")
    plt.savefig(out); plt.close(fig)

def fig_case_trajectories(case_csv, out_dir, title, k=6):
    if not os.path.exists(case_csv):
        return
    cs = pd.read_csv(case_csv)
    # pick up to k unique patients
    unique_ids = cs["patient_idx"].unique()[:k]
    for pid in unique_ids:
        sub = cs[cs["patient_idx"] == pid].sort_values("step")
        fig, ax = plt.subplots(figsize=(5.5,3.2))
        ax.plot(sub["step"], sub["prob_1_now"], marker="o")
        ax.set_xlabel("Step")
        ax.set_ylabel("P(AKI)")
        ax.set_title(f"{title} — patient {int(pid)} trajectory")
        # annotate final action if present
        last = sub.iloc[-1]
        ax.text(last["step"], last["prob_1_now"], last["action_label"], ha="left", va="bottom")
        out = os.path.join(out_dir, f"case_traj_{int(pid)}.png")
        plt.savefig(out); plt.close(fig)

def print_table(summary, thr_metrics, title):
    print("\n" + "="*50)
    print(f"{title} — Paper-style metrics")
    print("="*50)
    print(f"Accuracy: {thr_metrics['acc']*100:.2f}%")
    print(f"F1-score: {thr_metrics['f1']:.3f} | AUROC: {summary.get('auroc', float('nan')):.3f}")
    print(f"Precision: {thr_metrics['prec']:.3f}")
    print(f"Recall (Sensitivity): {thr_metrics['rec']:.3f}   Specificity: {thr_metrics['spec']:.3f}")
    print(f"Confusion Matrix: TP={thr_metrics['tp']}, FP={thr_metrics['fp']}, TN={thr_metrics['tn']}, FN={thr_metrics['fn']}")
    print(f"Avg cost: ${summary.get('avg_cost', float('nan')):.2f} | Median: ${summary.get('median_cost', float('nan')):.2f} | IQR=${summary.get('p75_cost',0)-summary.get('p25_cost',0):.2f}")
    print("="*50)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_dir", type=str, default="reports/eval/aki")
    ap.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    ap.add_argument("--outdir", type=str, default="reports/eval/aki/figs")
    ap.add_argument("--title", type=str, default="AKI RL — Test")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--case_csv", type=str, default=None, help="Path to aki_case_studies_<split>.csv if you generated it")
    args = ap.parse_args()

    _ensure_dir(args.outdir)

    df, summary = load_split(args.report_dir, args.split)
    thrm = metrics_at_threshold(df, args.threshold)

    # Figures
    fig_class_balance(df, os.path.join(args.outdir, "class_balance.png"), args.title)
    fig_roc_pr(df, os.path.join(args.outdir, "roc.png"), os.path.join(args.outdir, "pr.png"), args.title)
    fig_confmat(df, args.threshold, os.path.join(args.outdir, "confusion.png"), args.title)
    fig_cost_distribution(df, os.path.join(args.outdir, "cost_hist.png"), os.path.join(args.outdir, "cost_box.png"), args.title)
    fig_panel_usage(df, os.path.join(args.outdir, "panel_usage.png"), args.title)
    fig_co_usage(df, os.path.join(args.outdir, "panel_co_usage.png"), args.title)
    fig_calibration(df, os.path.join(args.outdir, "calibration.png"), args.title)
    fig_threshold_sweep(df, os.path.join(args.outdir, "threshold_sweep.png"), args.title)

    # Optional case studies (if provided)
    if args.case_csv:
        fig_case_trajectories(args.case_csv, args.outdir, args.title, k=6)

    # Paper-style console table (recomputed at given threshold)
    print_table(summary, thrm, args.title)

    print(f"\nSaved figures to: {args.outdir}")
    if args.case_csv:
        print(f"Case study trajectories also saved in: {args.outdir}")

if __name__ == "__main__":
    main()
