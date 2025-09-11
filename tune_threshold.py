import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    precision_recall_curve,
    precision_recall_fscore_support
)

def find_optimal_threshold():
    """
    Loads the trained baseline model and finds the optimal decision threshold
    on the validation set that maximizes the F2-score.
    """
    print("--- Phase 2/3: Finding Optimal Decision Threshold ---")

    # 1. Load Data
    try:
        X_train = pd.read_csv("data/processed/sepsis/train_X.csv")
        y_train = pd.read_csv("data/processed/sepsis/train_y.csv").squeeze()
        X_val = pd.read_csv("data/processed/sepsis/val_X.csv")
        y_val = pd.read_csv("data/processed/sepsis/val_y.csv").squeeze()
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # 2. Re-train the baseline model exactly as before to have it in memory
    print("\nRe-training the baseline model...")
    neg_count = y_train.value_counts()[0]
    pos_count = y_train.value_counts()[1]
    scale_pos_weight_value = neg_count / pos_count
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight_value,
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Model ready.")

    # 3. Get predicted probabilities on the validation set
    y_val_probs = model.predict_proba(X_val)[:, 1]

    # 4. Calculate Precision-Recall Curve and Find Optimal Threshold for F2-Score
    precision, recall, thresholds = precision_recall_curve(y_val, y_val_probs)

    # To avoid division by zero, we operate on slices where precision and recall are not zero
    # The last value of thresholds is 1.0, which corresponds to predicting no positives, so we exclude it.
    pr_slice = (precision > 0) & (recall > 0)
    precision = precision[pr_slice]
    recall = recall[pr_slice]
    thresholds = thresholds[pr_slice[:-1]] # Aligning shape with precision/recall

    # Calculate F2-score for each threshold. F2 gives more weight to recall.
    beta = 2
    f2_scores = ((1 + beta**2) * precision * recall) / ((beta**2 * precision) + recall)
    
    # Find the index and value of the best threshold
    best_f2_idx = np.argmax(f2_scores)
    optimal_threshold = thresholds[best_f2_idx]
    best_f2_score = f2_scores[best_f2_idx]
    
    print(f"\nOptimal threshold found that maximizes F2-Score: {optimal_threshold:.4f}")
    print(f"This threshold achieves an F2-Score of: {best_f2_score:.4f} on the validation set.")

    # 5. Re-evaluate using the NEW optimal threshold
    print("\n--- Evaluation with OPTIMAL Threshold ---")
    y_val_preds_optimal = (y_val_probs >= optimal_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_val, y_val_preds_optimal).ravel()
    p, r, f1, _ = precision_recall_fscore_support(y_val, y_val_preds_optimal, average='binary')
    specificity = tn / (tn + fp)

    print("\nNew Confusion Matrix:")
    print(f"                 PREDICTED")
    print(f"               Discharged (0) | Expired (1)")
    print(f"ACTUAL  Discharged (0) | {tn: <12} | {fp: <12}")
    print(f"ACTUAL  Expired (1)    | {fn: <12} | {tp: <12}")
    
    print(f"\nNew Performance Metrics (Threshold = {optimal_threshold:.4f}):")
    print("---------------------------------------------")
    print(f"False Negatives (FN - Missed Expired): {fn}  <-- COMPARE THIS TO 308!")
    print(f"False Positives (FP - Incorrectly Flagged): {fp}")
    print("---------------------------------------------")
    print(f"Recall (Sensitivity): {r:.4f}")
    print(f"Specificity:          {specificity:.4f}")
    print(f"Precision:            {p:.4f}")
    print(f"F1-Score:             {f1:.4f}")
    print("---------------------------------------------")

    # 6. Plot and save the Precision-Recall Curve
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, label='Precision-Recall Curve', color='navy')
    plt.scatter(
        recall[best_f2_idx], 
        precision[best_f2_idx], 
        marker='o', 
        color='red', 
        s=100,
        zorder=5,
        label=f'Best F2-Score (Threshold={optimal_threshold:.2f})'
    )
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Sepsis Model')
    plt.legend()
    plt.grid(True)
    
    pr_curve_path = "precision_recall_curve.png"
    plt.savefig(pr_curve_path)
    print(f"\nPrecision-Recall curve saved to: {pr_curve_path}")

if __name__ == '__main__':
    find_optimal_threshold()