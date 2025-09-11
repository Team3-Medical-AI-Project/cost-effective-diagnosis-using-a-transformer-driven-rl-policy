# scripts/audit_aki_augmentation_and_leakage.py
"""
Audit AKI augmentation & leakage (CTGAN+SMOTE) in one pass.

What it checks:
1) Schema & balance
   - Column alignment (train/val/test)
   - Class balance per split
   - NaN presence (should be none in train_X if you saved imputed version)
2) Train/Val/Test overlap (leakage)
   - Exact duplicate rows across splits (hash-based)
   - Intra-train duplicates (possible oversampling duplicates)
3) Distribution shift & authenticity
   - KS tests per feature: train vs val (BH-adjusted p-values)
   - Domain classifier (LogReg) distinguishing train_aug vs val (AUC ~ 0.5 is good)
   - TSTR / TRTS sanity (LogReg): AUC/AUPRC on val/test and reverse

Outputs (under reports/proof_aug/aki_audit/):
  - summary.txt
  - ks_feature_shift.csv
  - domain_auc.txt
  - tstr_trts.csv
  - dup_stats.json
"""

import argparse
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils.extmath import randomized_svd

from pandas.util import hash_pandas_object

warnings.filterwarnings("ignore", category=UserWarning)

# Optional SciPy for KS; fall back to simple heuristic if missing
try:
    from scipy.stats import ks_2samp
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


def load_cfg(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_splits(cfg) -> Bunch:
    proc = Path(cfg["data"]["processed_dir"])
    Xtr = pd.read_csv(proc / "train_X.csv")
    ytr = pd.read_csv(proc / "train_y.csv")["y"].values
    Xva = pd.read_csv(proc / "val_X.csv")
    yva = pd.read_csv(proc / "val_y.csv")["y"].values
    Xte = pd.read_csv(proc / "test_X.csv")
    yte = pd.read_csv(proc / "test_y.csv")["y"].values
    return Bunch(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte)


def ensure_same_columns(Xtr, Xva, Xte):
    cols = Xtr.columns
    assert list(Xva.columns) == list(cols), "val_X columns differ from train_X"
    assert list(Xte.columns) == list(cols), "test_X columns differ from train_X"
    return cols


def class_balance(y):
    p = float(np.mean(y))
    return {"n": int(len(y)), "pos": int(np.sum(y)), "neg": int(len(y) - np.sum(y)), "pos_rate": p}


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def hash_rows(df: pd.DataFrame):
    # Stable 64-bit hash per row
    return hash_pandas_object(df, index=False).values


def duplicate_stats(Xtr, Xva, Xte):
    """
    Check exact duplicates:
      - within train (possible SMOTE/CTGAN low-variance edge, though exact dup from SMOTE is uncommon)
      - train vs val
      - train vs test
    """
    htr = hash_rows(Xtr)
    hva = hash_rows(Xva)
    hte = hash_rows(Xte)

    # within train
    _, counts = np.unique(htr, return_counts=True)
    intra_train_dups = int(np.sum(counts > 1))

    # cross overlaps
    tr_set = set(htr.tolist())
    val_set = set(hva.tolist())
    te_set  = set(hte.tolist())

    tr_val_overlap = len(tr_set & val_set)
    tr_te_overlap  = len(tr_set & te_set)

    return {
        "intra_train_exact_dups": intra_train_dups,
        "train_vs_val_exact_overlap": tr_val_overlap,
        "train_vs_test_exact_overlap": tr_te_overlap
    }


def ks_feature_shift(Xtr, Xva, p_adj_method="bh"):
    """
    KS test per feature: train vs val.
    Returns DataFrame with columns: feature, ks_stat, p_value, p_adj, mean_train, mean_val
    """
    rows = []
    for col in Xtr.columns:
        a = pd.to_numeric(Xtr[col], errors="coerce").dropna().values
        b = pd.to_numeric(Xva[col], errors="coerce").dropna().values
        if len(a) < 20 or len(b) < 20:
            stat, p = np.nan, np.nan
        else:
            if SCIPY_OK:
                stat, p = ks_2samp(a, b, alternative="two-sided", mode="auto")
            else:
                # crude fallback: compare quantiles
                qa = np.nanquantile(a, [0.1,0.5,0.9])
                qb = np.nanquantile(b, [0.1,0.5,0.9])
                stat = float(np.max(np.abs(qa - qb)))
                p = np.nan
        rows.append([col, float(stat), float(p), float(np.nanmean(a)), float(np.nanmean(b))])
    df = pd.DataFrame(rows, columns=["feature","ks_stat","p_value","mean_train","mean_val"])
    # Benjamini-Hochberg (BH) FDR
    if SCIPY_OK and df["p_value"].notna().any():
        p = df["p_value"].copy()
        m = p.notna().sum()
        order = np.argsort(p.fillna(1).values)
        p_sorted = p.fillna(1).values[order]
        adj = np.empty_like(p_sorted, dtype=float)
        prev = 1.0
        for i in range(m, 0, -1):
            rank = i
            val = min(prev, (p_sorted[i-1] * m) / rank)
            adj[i-1] = val
            prev = val
        p_adj = np.ones(len(df))
        p_adj[order] = adj
        df["p_adj_bh"] = p_adj
    else:
        df["p_adj_bh"] = np.nan
    df.sort_values(["p_adj_bh","ks_stat"], ascending=[True, False], inplace=True, na_position="last")
    return df


def domain_classifier_auc(Xtr, Xva, sample_max=20000, seed=42):
    """
    AUC for a classifier trying to tell train_aug (1) vs val (0).
    If augmentation is authentic, AUC should be ~0.5–0.6. Higher => shift.
    """
    rng = np.random.default_rng(seed)
    # Sample to a manageable size
    ntr = min(len(Xtr), sample_max)
    nva = min(len(Xva), sample_max)
    idx_tr = rng.choice(len(Xtr), size=ntr, replace=False)
    idx_va = rng.choice(len(Xva), size=nva, replace=False)

    X = pd.concat([Xtr.iloc[idx_tr], Xva.iloc[idx_va]], axis=0, ignore_index=True)
    d = np.concatenate([np.ones(ntr), np.zeros(nva)])
    # Simple pipeline: impute (median) + scale + logistic
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(with_mean=False), LogisticRegression(max_iter=200))
    Xtr_, Xte_, dtr, dte = train_test_split(X, d, test_size=0.4, random_state=seed, stratify=d)
    pipe.fit(Xtr_, dtr)
    p = pipe.predict_proba(Xte_)[:,1]
    auc = roc_auc_score(dte, p)
    return float(auc)


def tstr_trts_metrics(Xtr, ytr, Xva, yva, Xte, yte, seed=42):
    """
    TSTR/TRTS with a simple logreg pipeline (median-impute + scale).
    - TSTR: train on TRAIN_aug, test on VAL and TEST
    - TRTS: train on VAL, test on TRAIN_aug (rough check)
    """
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(with_mean=False), LogisticRegression(max_iter=200))

    # TSTR
    pipe.fit(Xtr, ytr)
    p_va = pipe.predict_proba(Xva)[:,1]
    p_te = pipe.predict_proba(Xte)[:,1]
    tstr = {
        "TSTR_val_AUROC": roc_auc_score(yva, p_va),
        "TSTR_val_AUPRC": average_precision_score(yva, p_va),
        "TSTR_test_AUROC": roc_auc_score(yte, p_te),
        "TSTR_test_AUPRC": average_precision_score(yte, p_te),
    }

    # TRTS (train on val, test on train_aug)
    pipe.fit(Xva, yva)
    p_tr = pipe.predict_proba(Xtr)[:,1]
    trts = {
        "TRTS_train_AUROC": roc_auc_score(ytr, p_tr),
        "TRTS_train_AUPRC": average_precision_score(ytr, p_tr)
    }
    return {**tstr, **trts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/aki_config_accfocus.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    out_dir = Path(cfg["reports"]["proof_dir"])
    audit_dir = out_dir.parent / (out_dir.name + "_audit")
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    S = load_splits(cfg)
    cols = ensure_same_columns(S.Xtr, S.Xva, S.Xte)

    # === 1) Schema & balance ===
    bal = {
        "train": class_balance(S.ytr),
        "val":   class_balance(S.yva),
        "test":  class_balance(S.yte),
    }
    nan_train = float(np.isnan(S.Xtr.values).mean())
    nan_val   = float(np.isnan(S.Xva.values).mean())
    nan_test  = float(np.isnan(S.Xte.values).mean())

    # === 2) Leakage (duplicates/overlaps) ===
    dups = duplicate_stats(S.Xtr, S.Xva, S.Xte)

    # === 3) Distribution shift & authenticity ===
    ks_df = ks_feature_shift(S.Xtr, S.Xva)
    ks_df.to_csv(audit_dir / "ks_feature_shift.csv", index=False)

    dom_auc = domain_classifier_auc(S.Xtr, S.Xva)
    write(audit_dir / "domain_auc.txt", f"{dom_auc:.6f}")

    tstr = tstr_trts_metrics(S.Xtr, S.ytr, S.Xva, S.yva, S.Xte, S.yte)
    pd.DataFrame([tstr]).to_csv(audit_dir / "tstr_trts.csv", index=False)

    # === Summary report ===
    summary = []
    summary.append("# AKI Augmentation & Leakage Audit\n")
    summary.append("## Shapes\n")
    summary.append(f"- train_X: {S.Xtr.shape}, val_X: {S.Xva.shape}, test_X: {S.Xte.shape}\n")
    summary.append("## Class balance\n")
    for k, v in bal.items():
        summary.append(f"- {k}: n={v['n']} pos={v['pos']} neg={v['neg']} pos_rate={v['pos_rate']:.3f}\n")
    summary.append("\n## NaN rates\n")
    summary.append(f"- train_X NaN rate: {nan_train:.6f}\n- val_X NaN rate: {nan_val:.6f}\n- test_X NaN rate: {nan_test:.6f}\n")
    summary.append("\n## Duplicate / Overlap checks (exact row matches)\n")
    summary.append(json.dumps(dups, indent=2))
    summary.append("\n\n## KS shifts (train vs val)\n")
    if "p_adj_bh" in ks_df.columns:
        n_sig = int((ks_df["p_adj_bh"] < 0.01).sum())
        summary.append(f"- Significant features @ FDR<1%: {n_sig} / {len(ks_df)}\n")
        top = ks_df.head(10)[["feature","ks_stat","p_adj_bh","mean_train","mean_val"]]
    else:
        # If SciPy not available
        top = ks_df.head(10)[["feature","ks_stat","mean_train","mean_val"]]
        summary.append("- SciPy not available; KS p-values not computed.\n")
    summary.append(top.to_string(index=False))
    summary.append("\n\n## Domain classifier (train_aug vs val)\n")
    summary.append(f"- AUC: {dom_auc:.4f}  (≈0.5–0.6 is expected; >0.7 indicates notable shift)\n")
    summary.append("\n## TSTR/TRTS\n")
    for k, v in tstr.items():
        summary.append(f"- {k}: {v:.4f}\n")

    write(audit_dir / "summary.txt", "\n".join(summary))

    # Also save dup stats JSON
    with open(audit_dir / "dup_stats.json", "w", encoding="utf-8") as f:
        json.dump(dups, f, indent=2)

    print(f"[OK] Audit written to {audit_dir}")
    print("  - ks_feature_shift.csv")
    print("  - domain_auc.txt")
    print("  - tstr_trts.csv")
    print("  - dup_stats.json")
    print("  - summary.txt")


if __name__ == "__main__":
    main()
