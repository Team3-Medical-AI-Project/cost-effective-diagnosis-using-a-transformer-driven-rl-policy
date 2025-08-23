"""
make_interim_summary_plot.py
----------------------------
Create a compact PNG summary table from metrics.json.
"""
from __future__ import annotations
import os, sys, argparse, json
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Model Summary")
    args = ap.parse_args()

    with open(args.metrics, "r") as f:
        m = json.load(f)

    acc = float(m.get("acc", 0.0)); prec = float(m.get("precision", 0.0))
    rec = float(m.get("recall", 0.0)); f1 = float(m.get("f1", 0.0))
    avg_cost = float(m.get("avg_cost", 0.0)); avg_steps = float(m.get("avg_steps", 0.0))
    tn = int(m.get("tn", 0)); fp = int(m.get("fp", 0)); fn = int(m.get("fn", 0)); tp = int(m.get("tp", 0))
    n = int(m.get("n_episodes", tn+fp+fn+tp))

    rows = [
        ["Accuracy", f"{acc*100:.2f}%"],
        ["Precision (Expired=1)", f"{prec*100:.2f}%"],
        ["Recall (Expired=1)", f"{rec*100:.2f}%"],
        ["F1 (Expired=1)", f"{f1*100:.2f}%"],
        ["Avg Cost ($)", f"{avg_cost:.2f}"],
        ["Avg Steps", f"{avg_steps:.2f}"],
        ["Episodes", f"{n}"],
        ["Confusion (TN, FP, FN, TP)", f"{tn}, {fp}, {fn}, {tp}"],
    ]

    fig = plt.figure(figsize=(7, 3.8))
    plt.axis("off")
    plt.title(args.title, pad=12)

    table_data = [["Metric", "Value"]] + rows
    table = plt.table(cellText=table_data, loc="center", cellLoc="left", colLoc="left")
    table.scale(1.0, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0: cell.set_text_props(weight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
