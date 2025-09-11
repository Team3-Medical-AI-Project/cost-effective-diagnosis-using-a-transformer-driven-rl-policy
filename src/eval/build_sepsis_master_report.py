# src/eval/build_sepsis_master_report.py
import os, sys, json, glob, shutil, argparse, datetime, html, re, collections
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

IMG_NAMES = ["confusion_matrix.png", "roc.png", "pr.png", "panel_usage.png"]
METRICS_NAME = "metrics.json"
CASE_NAME = "case_studies.txt"

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def safe_read_json(p: str) -> Dict[str, Any]:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def yaml_panel_map_from_configs(config_glob: str = "configs/*.yaml") -> Dict[int, str]:
    """Find first panel_names mapping in any configs/*.yaml."""
    try:
        import yaml
    except Exception:
        return {}
    for yml in glob.glob(config_glob):
        try:
            with open(yml, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f) or {}
            pn = y.get("panel_names") or {}
            # keys might be str; coerce to int
            if isinstance(pn, dict) and pn:
                out = {}
                for k, v in pn.items():
                    try:
                        out[int(k)] = str(v)
                    except Exception:
                        pass
                if out:
                    return out
        except Exception:
            continue
    return {}

ORDER_RE = re.compile(r"Ordered\s+(.+?)\s+\(#(\d+)\)", re.IGNORECASE)

def panel_map_from_case_studies(path: str) -> Dict[int, str]:
    """Extract 'name (#idx)' pairs from case_studies preview file."""
    out = {}
    if not os.path.exists(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                m = ORDER_RE.search(line)
                if m:
                    name = m.group(1).strip()
                    idx = int(m.group(2))
                    # first seen wins
                    out.setdefault(idx, name)
    except Exception:
        pass
    return out

def discover_panel_map(report_dirs: List[str]) -> Dict[int, str]:
    """Priority: explicit JSON next time (not used), configs/*, then case_studies from any run."""
    # 1) configs
    m = yaml_panel_map_from_configs("configs/*.yaml")
    if m:
        return m
    # 2) scan case studies
    agg = {}
    for rd in report_dirs:
        p = os.path.join(rd, CASE_NAME)
        mm = panel_map_from_case_studies(p)
        for k, v in mm.items():
            agg.setdefault(k, v)
    return agg

def find_report_dirs(includes: List[str]) -> List[str]:
    candidates = set()
    if not includes:
        for d in glob.glob(os.path.join("reports", "**"), recursive=True):
            if os.path.isdir(d) and os.path.exists(os.path.join(d, METRICS_NAME)):
                candidates.add(os.path.normpath(d))
    else:
        for pat in includes:
            for d in glob.glob(pat):
                if os.path.isdir(d) and os.path.exists(os.path.join(d, METRICS_NAME)):
                    candidates.add(os.path.normpath(d))
                elif os.path.isfile(d):
                    parent = os.path.dirname(d)
                    if os.path.exists(os.path.join(parent, METRICS_NAME)):
                        candidates.add(os.path.normpath(parent))
    return sorted(candidates)

def copy_images(src_dir: str, dst_dir: str) -> Dict[str, str]:
    ensure_dir(dst_dir)
    out = {}
    for name in IMG_NAMES:
        s = os.path.join(src_dir, name)
        if os.path.exists(s):
            d = os.path.join(dst_dir, name)
            shutil.copy2(s, d)
            out[name] = os.path.relpath(d, os.path.dirname(dst_dir))
    # also copy probs.npy if present
    p = os.path.join(src_dir, "probs.npy")
    if os.path.exists(p):
        d = os.path.join(dst_dir, "probs.npy")
        shutil.copy2(p, d)
        out["probs.npy"] = os.path.relpath(d, os.path.dirname(dst_dir))
    return out

def read_case_preview(src_dir: str, max_lines: int = 8) -> str:
    p = os.path.join(src_dir, CASE_NAME)
    if not os.path.exists(p): return ""
    try:
        with open(p, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f.readlines()]
        preview = "\n".join(lines[:max_lines])
        more = f"\n… (see full file in {p})" if len(lines) > max_lines else ""
        return html.escape(preview + more)
    except Exception:
        return ""

def extract_panel_counts_from_case(src_dir: str) -> Dict[int, int]:
    """Counts orders from case_studies (note: preview subset, not the whole run)."""
    p = os.path.join(src_dir, CASE_NAME)
    counts = collections.Counter()
    if not os.path.exists(p): return dict(counts)
    try:
        with open(p, "r", encoding="utf-8") as f:
            for ln in f:
                m = ORDER_RE.search(ln)
                if m:
                    idx = int(m.group(2))
                    counts[idx] += 1
    except Exception:
        pass
    return dict(counts)

def try_load_panel_counts_json(src_dir: str) -> Optional[Dict[int, int]]:
    """If a future run writes panel_counts.json (orders over episodes), prefer that."""
    pj = os.path.join(src_dir, "panel_counts.json")
    if not os.path.exists(pj): return None
    try:
        j = safe_read_json(pj)
        # coerce keys to int
        return {int(k): int(v) for k, v in j.items()}
    except Exception:
        return None

def build_summary_table(metrics_rows: List[Dict[str, Any]]) -> str:
    if not metrics_rows:
        return "<p><i>No metrics found.</i></p>"
    df = pd.DataFrame(metrics_rows)
    preferred = [
        "name","acc","precision","recall","specificity","f1","f2",
        "AUC","AUPRC","avg_cost","avg_steps","n_episodes",
        "threshold_used","threshold_source","tn","fp","fn","tp"
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]
    for c in ["acc","precision","recall","specificity","f1","f2","AUC","AUPRC"]:
        if c in df.columns: df[c] = (df[c]*1.0).round(3)
    for c in ["avg_cost","avg_steps"]:
        if c in df.columns: df[c] = (df[c]*1.0).round(3)
    return df.to_html(index=False, border=1)

def make_avg_cost_bar(fig_path: str, rows: List[Dict[str, Any]]):
    names = [r["name"] for r in rows]
    costs = [r.get("avg_cost", None) for r in rows]
    names2, costs2 = [], []
    for n, c in zip(names, costs):
        if isinstance(c, (int, float)):
            names2.append(n)
            costs2.append(float(c))
    if not names2:
        return
    plt.figure()
    plt.title("Average Cost by Experiment")
    plt.barh(range(len(names2)), costs2)
    plt.yticks(range(len(names2)), [n[-60:] for n in names2])
    plt.xlabel("Average Cost ($)")
    plt.tight_layout()
    plt.savefig(fig_path); plt.close()

def make_panel_aggregate_charts(fig_dir: str,
                                panel_map: Dict[int, str],
                                per_run_counts: List[Tuple[str, Dict[int, int]]]) -> Tuple[str, str]:
    """Return (bar_path, stacked_path) or ('','') if nothing to plot."""
    # sum across runs
    agg = collections.Counter()
    run_names = []
    all_indices = set()
    for name, cnt in per_run_counts:
        run_names.append(name)
        for k, v in (cnt or {}).items():
            agg[k] += int(v)
            all_indices.add(int(k))
    if not agg:
        return "", ""

    # sorted by index
    idxs = sorted(all_indices)
    labels = [panel_map.get(i, f"Panel {i}") for i in idxs]

    # 1) global bar
    bar_path = os.path.join(fig_dir, "aggregate_panel_usage_bar.png")
    plt.figure()
    plt.title("Global Panel Usage (orders across included runs)")
    plt.bar(range(len(idxs)), [agg[i] for i in idxs])
    plt.xticks(range(len(idxs)), labels, rotation=20, ha="right")
    plt.ylabel("Order count")
    plt.tight_layout()
    plt.savefig(bar_path); plt.close()

    # 2) stacked by run
    stacked_path = os.path.join(fig_dir, "aggregate_panel_usage_stacked.png")
    # build matrix [runs x panels]
    data = []
    rnames = []
    for name, cnt in per_run_counts:
        row = [int(cnt.get(i, 0)) for i in idxs]
        data.append(row)
        rnames.append(name[-60:])
    data = pd.DataFrame(data, columns=labels, index=rnames)
    ax = data.plot(kind="bar", stacked=True, figsize=(10, 6))
    ax.set_title("Panel Usage by Run (stacked orders)")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Order count")
    plt.tight_layout()
    plt.savefig(stacked_path); plt.close()

    return bar_path, stacked_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output folder for master report")
    ap.add_argument("--title", default="Sepsis RL Pipeline — Master Report")
    ap.add_argument("--include", nargs="*", default=[], help="Optional glob(s) for report dirs to include")
    args = ap.parse_args()

    out_dir = os.path.normpath(args.out)
    ensure_dir(out_dir)
    figs_root = os.path.join(out_dir, "figs"); ensure_dir(figs_root)

    # 1) experiments
    report_dirs = find_report_dirs(args.include)
    if not report_dirs:
        print("No report folders with metrics.json found.")
        sys.exit(0)

    # 2) panel name mapping (configs first, then case study hints)
    panel_map = discover_panel_map(report_dirs)

    # 3) collect metrics + copy images + (try) panel counts
    rows = []
    cards_html = []
    per_run_counts = []
    for rd in report_dirs:
        mpath = os.path.join(rd, METRICS_NAME)
        met = safe_read_json(mpath)
        name = os.path.relpath(rd, start=".")

        # figures
        this_figs_dir = os.path.join(figs_root, name.replace("\\", "__").replace("/", "__"))
        copied = copy_images(rd, this_figs_dir)

        # case preview
        case_prev = read_case_preview(rd)

        # panel counts (prefer JSON if present)
        panel_counts = try_load_panel_counts_json(rd)
        if panel_counts is None:
            panel_counts = extract_panel_counts_from_case(rd)  # note: preview-based

        if panel_counts:
            per_run_counts.append((name, panel_counts))

        row = {"name": name}
        row.update({k: met.get(k, None) for k in [
            "acc","precision","recall","specificity","f1","f2",
            "AUC","AUPRC","avg_cost","avg_steps","n_episodes",
            "threshold_used","threshold_source","tn","fp","fn","tp"
        ]})
        rows.append(row)

        # Build card
        card = [f"<section style='border:1px solid #ddd; border-radius:12px; padding:16px; margin:12px 0;'>"]
        card.append(f"<h3 style='margin-top:0;'>{html.escape(name)}</h3>")

        small = pd.DataFrame([row]).T.reset_index()
        small.columns = ["metric", "value"]
        card.append(small.to_html(index=False, border=1))

        # If we know the panel mapping, show a tiny legend table
        if panel_map:
            pm_df = pd.DataFrame(
                [{"index": i, "name": panel_map.get(i, f'Panel {i}')} for i in sorted(panel_map)]
            )
            card.append("<details><summary><b>Panel index → name</b></summary>")
            card.append(pm_df.to_html(index=False, border=1))
            card.append("</details>")

        # figs
        figbits = []
        order = [("confusion_matrix.png","Confusion Matrix"),
                 ("roc.png","ROC"),
                 ("pr.png","Precision–Recall"),
                 ("panel_usage.png","Panel Usage (as rendered during eval)")]
        for fn, label in order:
            if fn in copied:
                rel = os.path.join("figs", os.path.relpath(os.path.join(this_figs_dir, fn), figs_root))
                figbits.append(f"<figure><img src='{html.escape(rel)}' width='420'><figcaption>{label}</figcaption></figure>")
        if figbits:
            card.append("<div style='display:flex; flex-wrap:wrap; gap:16px; align-items:flex-start;'>")
            card.append("\n".join(figbits))
            card.append("</div>")

        if case_prev:
            card.append("<details style='margin-top:10px;'><summary><b>Case study preview</b></summary>")
            card.append(f"<pre style='white-space:pre-wrap; background:#fafafa; padding:12px; border:1px solid #eee; border-radius:8px;'>{case_prev}</pre>")
            card.append("</details>")

        # panel counts (if any for this run) as a small table
        if panel_counts:
            pc_rows = [{"panel": panel_map.get(int(k), f"Panel {int(k)}"), "orders": int(v)}
                       for k, v in sorted(panel_counts.items(), key=lambda x: int(x[0]))]
            pc_df = pd.DataFrame(pc_rows)
            card.append("<details><summary><b>Parsed panel usage (orders)</b></summary>")
            card.append(pc_df.to_html(index=False, border=1))
            if try_load_panel_counts_json(rd) is None:
                card.append("<div style='color:#777;font-size:12px;'>* Parsed from case preview (subset), for full accuracy consider emitting panel_counts.json during eval.</div>")
            card.append("</details>")

        card.append("</section>")
        cards_html.append("\n".join(card))

    # 4) global visuals: average cost and aggregate panel usage
    global_bits = []
    # average cost bar
    avg_cost_png = os.path.join(figs_root, "avg_cost_by_experiment.png")
    make_avg_cost_bar(avg_cost_png, rows)
    if os.path.exists(avg_cost_png):
        global_bits.append(f"<figure><img src='figs/{html.escape(os.path.basename(avg_cost_png))}' width='700'><figcaption>Average cost by experiment</figcaption></figure>")

    # aggregate panel usage
    bar_path, stacked_path = make_panel_aggregate_charts(figs_root, panel_map, per_run_counts)
    if bar_path:
        global_bits.append(f"<figure><img src='figs/{html.escape(os.path.basename(bar_path))}' width='640'><figcaption>Global panel usage (orders, summed across runs)</figcaption></figure>")
    if stacked_path:
        global_bits.append(f"<figure><img src='figs/{html.escape(os.path.basename(stacked_path))}' width='820'><figcaption>Panel usage by run (stacked orders)</figcaption></figure>")

    summary_html = build_summary_table(rows)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{html.escape(args.title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
 body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; line-height:1.4; padding:20px; color:#222; }}
 h1 {{ margin-bottom:0; }}
 .sub {{ color:#666; margin-top:4px; }}
 table {{ border-collapse: collapse; }}
 th, td {{ padding:6px 10px; }}
 figure {{ margin:0; }}
 code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
</style>
</head>
<body>
  <h1>{html.escape(args.title)}</h1>
  <div class="sub">Generated: {now}</div>

  <h2>Overview (All Experiments)</h2>
  {summary_html}

  <h2>Global Cost & Panel Usage</h2>
  {"".join(global_bits) if global_bits else "<p><i>No aggregate visuals could be computed (need avg_costs and some panel usage information).</i></p>"}

  <h2>Per-Experiment Details</h2>
  {"".join(cards_html)}

  <hr/>
  <p style="color:#666">Notes:
    Panel names come from <code>configs/*</code> (<code>panel_names</code>) when available; otherwise we infer from the case study text.
    For precise aggregate panel usage, consider having your eval write <code>panel_counts.json</code> (orders per index).
  </p>
</body>
</html>
"""
    out_html = os.path.join(out_dir, "index.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(page)

    print(f"[master] Wrote {out_html}")
    print(f"[master] Found {len(report_dirs)} experiment(s).")
    for rd in report_dirs:
        print(" -", rd)

if __name__ == "__main__":
    main()
