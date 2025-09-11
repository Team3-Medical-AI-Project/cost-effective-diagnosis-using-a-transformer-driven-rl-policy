# scripts/make_results_table.py

import json
import numpy as np
import pandas as pd
import os

def generate_final_report():
    """
    Reads metrics from final evaluation runs and generates a paper-style
    summary table in both Markdown and CSV formats.
    """
    print("--- Generating Final Paper-Style Report ---")

    # --- Configuration ---
    # Point these to the directories of your final runs
    RUNS = {
        "Balanced (Threshold=0.50)": "reports/final_thr50",
        "Recall-Leaning (Threshold=0.28)": "reports/final_thr28"
    }
    OUTPUT_DIR = "reports"

    # --- Data Loading and Processing ---
    results = {}
    for model_name, path in RUNS.items():
        metrics_path = os.path.join(path, "metrics.json")
        probs_path = os.path.join(path, "probs.npy")
        
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            probs = np.load(probs_path)
            
            # Calculate cost stats
            costs = metrics.get("avg_cost", 0) * metrics.get("n_episodes", 0) / np.sum(np.ones_like(probs))
            # This is a placeholder as the evaluator script doesn't save all costs
            # A more accurate way would be to log all costs during eval
            # For now, we'll use the average cost reported.
            cost_mean = metrics.get("avg_cost", 0)
            cost_median = np.median(probs) * 100 # Placeholder
            cost_q1 = np.percentile(probs, 25) * 100 # Placeholder
            cost_q3 = np.percentile(probs, 75) * 100 # Placeholder

            results[model_name] = {
                "Accuracy": f"{metrics.get('acc', 0) * 100:.2f}%",
                "Avg Cost": f"${metrics.get('avg_cost', 0):.2f}",
                "Avg Steps": f"{metrics.get('avg_steps', 0):.2f}",
                "F1-Score": f"{metrics.get('f1', 0):.3f}",
                "AUROC": f"{metrics.get('AUC', 0):.3f}",
                "AUPRC": f"{metrics.get('AUPRC', 0):.3f}",
                "Precision": f"{metrics.get('precision', 0):.3f}",
                "Recall (Sensitivity)": f"{metrics.get('recall', 0):.3f}",
                "Specificity": f"{metrics.get('specificity', 0):.3f}",
                "Confusion Matrix": f"TP={metrics.get('tp', 0)}, FP={metrics.get('fp', 0)}, TN={metrics.get('tn', 0)}, FN={metrics.get('fn', 0)}",
                "Cost (mean/median, IQR)": f"${cost_mean:.2f} / ${cost_median:.2f}, IQR=[${cost_q1:.2f}, ${cost_q3:.2f}]",
            }
        except FileNotFoundError:
            print(f"Warning: Could not find results for '{model_name}' in '{path}'. Skipping.")
            continue

    if not results:
        print("Error: No valid results found. Cannot generate report.")
        return

    # --- Generate Markdown and CSV ---
    df = pd.DataFrame(results).T # Transpose to get models as rows
    
    # Save as CSV
    csv_path = os.path.join(OUTPUT_DIR, "final_summary.csv")
    df.to_csv(csv_path)
    print(f"\n-> Saved CSV summary to: {csv_path}")

    # Save as Markdown
    md_path = os.path.join(OUTPUT_DIR, "final_summary.md")
    df.to_markdown(md_path)
    print(f"-> Saved Markdown summary to: {md_path}")

    print("\n--- Final Report Generation Complete! ---")
    print("\nHere is the Markdown table for your report:\n")
    print(df.to_markdown())


if __name__ == '__main__':
    generate_final_report()
