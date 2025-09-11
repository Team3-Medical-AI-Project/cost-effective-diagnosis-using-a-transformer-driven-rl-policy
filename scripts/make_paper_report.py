# scripts/make_paper_report.py

import json
import numpy as np
import os
import argparse
from collections import Counter

def generate_paper_style_report(report_dir: str):
    """
    Loads data from a specified report directory and prints a detailed,
    paper-style summary of the agent's performance.
    """
    print(f"--- Generating Paper-Style Report for: {report_dir} ---")

    # --- Load required files ---
    try:
        with open(os.path.join(report_dir, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
        
        panels_all = np.load(os.path.join(report_dir, 'panels_all.npy'), allow_pickle=True)
        costs_all = np.load(os.path.join(report_dir, 'costs_all.npy'))
        
        # We need the config to get panel names
        with open(os.path.join('configs\sepsis_config_accfocus.yaml'), 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        panel_names = config.get('panel_names', {})

    except FileNotFoundError as e:
        print(f"\nError: Could not find a required file in the report directory.")
        print(f"Missing file: {e.filename}")
        print("Please ensure you have run the evaluation with the updated script that saves .npy files.")
        return

    # --- Calculate Metrics ---
    n_episodes = metrics.get('n_episodes', 0)
    
    # Panel Usage
    panel_counts = Counter(panels_all)
    total_panels_ordered = len(panels_all)
    
    # To get panel usage rate, we need to count unique patients per panel.
    # This requires parsing case studies, which is complex. A good approximation
    # is the total count of each panel ordered.
    panel_strategy = {
        panel_names.get(str(k), f"Panel {k}"): f"{v} times"
        for k, v in sorted(panel_counts.items())
    }

    # Cost Statistics
    cost_mean = np.mean(costs_all) if len(costs_all) > 0 else 0
    cost_median = np.median(costs_all) if len(costs_all) > 0 else 0
    cost_q1 = np.percentile(costs_all, 25) if len(costs_all) > 0 else 0
    cost_q3 = np.percentile(costs_all, 75) if len(costs_all) > 0 else 0
    cost_iqr = cost_q3 - cost_q1

    # --- Format the Report ---
    report = f"""
==================================================
--- Final Performance Report ---
==================================================
📈 Final Diagnostic Accuracy: {metrics.get('acc', 0) * 100:.2f}%
💰 Average Financial Cost per Patient: ${metrics.get('avg_cost', 0):.2f}
📉 Average Number of Tests Ordered: {metrics.get('avg_steps', 0):.2f}

==================================================
--- Agent's Testing Strategy ---
==================================================
"""
    for name, count_str in panel_strategy.items():
        report += f"  - {name}: {count_str}\n"

    report += f"""
==================================================
--- Evaluation (Paper-style metrics) ---
==================================================
F1-score: {metrics.get('f1', 0):.3f}
AUROC: {metrics.get('AUC', 0):.3f}
AUPRC: {metrics.get('AUPRC', 0):.3f}
Precision: {metrics.get('precision', 0):.3f}
Recall (Sensitivity): {metrics.get('recall', 0):.3f}
Specificity: {metrics.get('specificity', 0):.3f}
Confusion Matrix: TP={metrics.get('tp', 0)}, FP={metrics.get('fp', 0)}, TN={metrics.get('tn', 0)}, FN={metrics.get('fn', 0)}
Cost (mean/median, IQR): ${cost_mean:.2f} / ${cost_median:.2f}, IQR=${cost_iqr:.2f}
IQR= [${cost_q1:.2f}, ${cost_q3:.2f}]
Decision threshold used: {metrics.get('threshold_used', 'N/A'):.3f}
==================================================
"""
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a paper-style report from an evaluation directory.")
    parser.add_argument(
        "report_dir", 
        type=str, 
        help="Path to the report directory (e.g., reports/final_thr28)"
    )
    args = parser.parse_args()
    
    if not os.path.isdir(args.report_dir):
        print(f"Error: Directory not found at '{args.report_dir}'")
    else:
        generate_paper_style_report(args.report_dir)

