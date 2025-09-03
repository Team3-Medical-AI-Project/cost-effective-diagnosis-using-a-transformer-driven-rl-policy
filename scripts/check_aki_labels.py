# save as: scripts/check_aki_labels.py
import re, sys, json
from pathlib import Path
import pandas as pd

def find_label_column(df):
    # common candidates
    candidates = [c for c in df.columns if re.search(r'aki.*(label|72|pos|target)', c, re.I)]
    # fallback: plain 'aki' or 'label'
    if not candidates:
        for k in ['aki', 'label', 'target']:
            if k in df.columns: candidates.append(k)
    return candidates

def summarize(path):
    p = Path(path)
    df = pd.read_csv(p)
    label_cols = find_label_column(df)
    out = {"file": str(p), "n": len(df), "label_cols_found": label_cols}
    if not label_cols:
        out["error"] = "No obvious AKI label column found. Please tell me the exact name."
        return out

    # pick the first plausible label column
    ycol = label_cols[0]
    y = df[ycol].astype(int)
    out["label_col_used"] = ycol
    out["positives"] = int(y.sum())
    out["negatives"] = int((1 - y).sum())
    out["prevalence_pct"] = round(100.0 * y.mean(), 2)

    # very rough window hints (if present in columns/attrs)
    possible_24h = any(re.search(r'0[_\-]?24h|first_?24|t0_24', c, re.I) for c in df.columns)
    possible_72h = any(re.search(r'72', c, re.I) for c in df.columns)
    out["has_0_24h_hint"] = possible_24h
    out["has_72h_hint"] = possible_72h
    return out

if __name__ == "__main__":
    # pass one or more CSVs; e.g.:
    #   python scripts/check_aki_labels.py data/preprocessed/aki_feature_matrix.csv
    #   python scripts/check_aki_labels.py data/processed/aki/train_y.csv data/processed/aki/val_y.csv data/processed/aki/test_y.csv
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_aki_labels.py <one-or-more CSV paths>")
        sys.exit(1)

    results = [summarize(arg) for arg in sys.argv[1:]]
    print(json.dumps(results, indent=2))
