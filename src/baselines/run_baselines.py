"""
Baseline Model Training and Evaluation Script for Sepsis Cohort

This script trains, evaluates, and reports performance for standard machine
learning baselines (Logistic Regression and XGBoost). It assumes a static
scenario where all features are available after imputation.
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import argparse
import sys

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

# --- Path Setup ---
try:
    # This allows the script to be run from the project root (e.g., python src/baselines/...)
    sys.path.append(str(Path(__file__).resolve().parents[2]))
except NameError:
    # Fallback for interactive environments
    sys.path.append('..')

# --- Main Function ---
def run_baseline(model_type: str, config_path: str = "configs/sepsis_config.yaml"):
    """
    Trains and evaluates a specified baseline model.

    Args:
        model_type (str): The model to run ('logreg' or 'xgboost').
        config_path (str): Path to the sepsis config file to get cost info.
    """
    print(f"\n--- Running Baseline: {model_type.upper()} ---")

    # --- 1. Load Data ---
    # For baselines, we use the fully processed, scaled, and imputed data.
    # The training set is already augmented (resampled).
    processed_dir = Path("data/processed/sepsis/")
    try:
        X_train = pd.read_csv(processed_dir / "train_X.csv")
        y_train = pd.read_csv(processed_dir / "train_y.csv").values.ravel()
        X_test = pd.read_csv(processed_dir / "test_X.csv")
        y_test = pd.read_csv(processed_dir / "test_y.csv").values.ravel()
    except FileNotFoundError as e:
        print(f"❌ ERROR: Processed data file not found. {e}")
        print("Please ensure the data preparation pipeline has been run successfully.")
        return

    print(f"Loaded data. Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # --- 2. Initialize and Train Model ---
    if model_type == 'logreg':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'xgboost':
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError("Invalid model_type. Choose 'logreg' or 'xgboost'.")

    print(f"Training {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    print("Training complete.")

    # --- 3. Evaluate Model ---
    print("Evaluating model on the test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class (1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    
    # --- 4. Calculate Cost ---
    # For these static models, the "cost" is fixed, as they use all features.
    # The cost is the sum of all available test panels.
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    total_cost = sum(config['cost_mapping'].values())
    num_tests = len(config['cost_mapping'])

    print("Evaluation complete.")

    # --- 5. Save Results ---
    results = {
        "model_type": model_type,
        "metrics": {
            "accuracy": accuracy,
            "auroc": auroc,
            "auprc": auprc,
        },
        "cost": {
            "average_usd_cost": total_cost,
            "average_num_tests": num_tests,
        }
    }
    
    # Ensure the reports directory exists
    reports_dir = Path("reports/")
    reports_dir.mkdir(exist_ok=True)
    output_path = reports_dir / f"baseline_{model_type}_results.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"✅ Results saved to: {output_path}")
    print(json.dumps(results, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run baseline ML models for the Sepsis task.")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=['logreg', 'xgboost'],
        help="The type of baseline model to run."
    )
    args = parser.parse_args()
    
    run_baseline(model_type=args.model)