import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    confusion_matrix, 
    precision_recall_fscore_support, 
    average_precision_score, 
    roc_auc_score
)

def train_and_evaluate_baseline():
    """
    Trains and evaluates a baseline XGBoost model on the original, 
    un-augmented sepsis data.
    """
    print("--- Phase 1: Training and Evaluating Baseline Model ---")

    # 1. Load Data
    try:
        X_train = pd.read_csv("data/processed/sepsis/train_X.csv")
        y_train = pd.read_csv("data/processed/sepsis/train_y.csv").squeeze()
        X_val = pd.read_csv("data/processed/sepsis/val_X.csv")
        y_val = pd.read_csv("data/processed/sepsis/val_y.csv").squeeze()
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure this script is run from the project's root directory.")
        return

    # 2. Calculate scale_pos_weight for class imbalance
    # This is the most direct way to tell XGBoost to account for the imbalance.
    # It's calculated as: (count of negative class) / (count of positive class)
    neg_count = y_train.value_counts()[0]
    pos_count = y_train.value_counts()[1]
    scale_pos_weight_value = neg_count / pos_count
    
    print(f"Class distribution in training data: Discharged (0): {neg_count}, Expired (1): {pos_count}")
    print(f"Calculated scale_pos_weight: {scale_pos_weight_value:.2f}")

    # 3. Initialize and Train the XGBoost Model
    print("\nTraining XGBoost model...")
    # We use a strong but standard set of parameters.
    # The key is `scale_pos_weight` which handles the class imbalance.
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight_value,
        use_label_encoder=False,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 4. Evaluate on the UNTOUCHED Validation Set
    print("\n--- Baseline Model Evaluation (on Validation Set) ---")
    
    # Get predicted probabilities for the positive class (Expired)
    y_val_probs = model.predict_proba(X_val)[:, 1]
    # Get predictions using the default 0.5 threshold for now.
    # We will tune this threshold in Phase 3.
    y_val_preds = (y_val_probs >= 0.5).astype(int)

    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_val, y_val_preds).ravel()
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_val, y_val_preds, average='binary')
    specificity = tn / (tn + fp)
    auprc = average_precision_score(y_val, y_val_probs)
    auroc = roc_auc_score(y_val, y_val_probs)

    # --- Print Report ---
    print("\nConfusion Matrix:")
    print(f"                 PREDICTED")
    print(f"               Discharged (0) | Expired (1)")
    print(f"ACTUAL  Discharged (0) | {tn: <12} | {fp: <12}")
    print(f"ACTUAL  Expired (1)    | {fn: <12} | {tp: <12}")
    
    print(f"\nKey Performance Metrics (Threshold = 0.5):")
    print("---------------------------------------------")
    print(f"Goal: Minimize both FN and FP.")
    print(f"False Negatives (FN - Missed Expired): {fn}")
    print(f"False Positives (FP - Incorrectly Flagged): {fp}")
    print("---------------------------------------------")
    print(f"Recall (Sensitivity): {recall:.4f}  <- How many of the actual 'Expired' did we catch?")
    print(f"Specificity:          {specificity:.4f}  <- How many of the actual 'Discharged' did we correctly identify?")
    print(f"Precision:            {precision:.4f}  <- When we predict 'Expired', how often are we right?")
    print(f"F1-Score:             {f1_score:.4f}  <- Balanced measure of Precision and Recall.")
    print("---------------------------------------------")
    print(f"AUPRC (PR Curve):     {auprc:.4f}  <- Overall performance metric, good for imbalance.")
    print(f"AUROC (ROC Curve):    {auroc:.4f}")
    print("---------------------------------------------")


if __name__ == '__main__':
    train_and_evaluate_baseline()