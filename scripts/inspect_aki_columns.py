# save as: scripts/inspect_aki_columns.py
import re, sys
from pathlib import Path
import pandas as pd

PANEL_RULES = {
    "CBC":      r"(wbc|rbc|hemoglobin|hematocrit|platelet)",
    "CMP":      r"(creatinine|bun|bicarbonate|sodium|potassium|chloride|calcium|magnesium|phosphate|aniongap|glucose|egfr)",
    "aPTT":     r"(ptt|aptt|inr|prothrombin)",
    "ABG":      r"(ph(?!os)|base_excess|abg)"
}
VITAL_RULE  = r"(o2_saturation|spo2|heart_rate|respiratory_rate|temp(erature)?|sbp|dbp)"

def categorize(cols):
    mapping = {"CBC":[], "CMP":[], "aPTT":[], "ABG":[], "Vitals/Static":[], "Unmatched":[]}
    for c in cols:
        c_lower = c.lower()
        matched = False
        for panel, pat in PANEL_RULES.items():
            if re.search(pat, c_lower):
                mapping[panel].append(c); matched = True; break
        if not matched:
            if re.search(VITAL_RULE, c_lower) or c_lower in ("age","gender","sex"):
                mapping["Vitals/Static"].append(c); matched = True
        if not matched:
            mapping["Unmatched"].append(c)
    return mapping

if __name__ == "__main__":
    # point to any AKI X file you currently have (even if it's temporary)
    # e.g., python scripts/inspect_aki_columns.py data/preprocessed/aki_feature_matrix.csv
    if len(sys.argv) != 2:
        print("Usage: python scripts/inspect_aki_columns.py <CSV path>")
        sys.exit(1)
    df = pd.read_csv(Path(sys.argv[1]), nrows=5)  # just need headers
    mapping = categorize(df.columns.tolist())
    for k,v in mapping.items():
        print(f"\n[{k}] ({len(v)}):")
        for name in v:
            print("  -", name)
