import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    confusion_matrix, 
    precision_recall_fscore_support, 
    average_precision_score, 
    roc_auc_score
)
import joblib # To save the final model

def final_evaluation():
    """
    Trains the final model on the augmented training data and evaluates it 
    on the hold-out test set using the optimal threshold.
    """
    print("--- Phase 4: Final Model Evaluation (on Test Set) ---")
    
    # This is the optimal threshold we discovered in the previous step.
    # It is now fixed and will be used for the final evaluation.
    OPTIMAL_THRESHOLD = 0.0536

    # 1. Load Data
    try:
        X_train = pd.read_csv("data/processed/sepsis/train_X.csv")
        y_train = pd.read_csv("data/processed/sepsis/train_y.csv").squeeze()
        X_test = pd.read_csv("data/processed/sepsis/test_X.csv")
        y_test = pd.read_csv("data/processed/sepsis/test_y.csv").squeeze()
        print("Training and Test data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # 2. Train the final model on the full augmented training data
    print("\nTraining final model...")
    neg_count = y_train.value_counts()[0]
    pos_count = y_train.value_counts()[1]
    scale_pos_weight_value = neg_count / pos_count
    
    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight_value,
        use_label_encoder=False,
        random_state=42
    )
    
    final_model.fit(X_train, y_train)
    print("Final model training complete.")

    # Save the trained model for future use
    model_filename = "final_sepsis_model.joblib"
    joblib.dump(final_model, model_filename)
    print(f"Final model saved to '{model_filename}'")

    # 3. Evaluate on the UNTOUCHED Test Set using the OPTIMAL threshold
    print(f"\n--- Final Performance Report (Threshold = {OPTIMAL_THRESHOLD}) ---")
    
    y_test_probs = final_model.predict_proba(X_test)[:, 1]
    y_test_preds = (y_test_probs >= OPTIMAL_THRESHOLD).astype(int)

    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_preds).ravel()
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_test_preds, average='binary')
    specificity = tn / (tn + fp)
    auprc = average_precision_score(y_test, y_test_probs)

    # --- Print Final Report ---
    print("\nFinal Confusion Matrix (on Test Set):")
    print(f"                 PREDICTED")
    print(f"               Discharged (0) | Expired (1)")
    print(f"ACTUAL  Discharged (0) | {tn: <12} | {fp: <12}")
    print(f"ACTUAL  Expired (1)    | {fn: <12} | {tp: <12}")
    
    print(f"\nFinal Key Performance Metrics:")
    print("---------------------------------------------")
    print(f"False Negatives (FN - Missed Expired): {fn}")
    print(f"False Positives (FP - Incorrectly Flagged): {fp}")
    print("---------------------------------------------")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity:          {specificity:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"F1-Score:             {f1_score:.4f}")
    print(f"AUPRC (PR Curve):     {auprc:.4f}")
    print("---------------------------------------------")
    print("\nThis is the expected performance of your model on new, unseen data.")

if __name__ == '__main__':
    final_evaluation()