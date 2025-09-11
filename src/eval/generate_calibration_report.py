"""
Calibration Analysis Script (v4.0 - Final)

v4.0: Compares Platt Scaling and Isotonic Regression using cross-validation,
      plots both results, and saves the best-performing calibrator.
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
import joblib

def generate_calibration_report():
    print("--- Generating Final Calibration Report (Comparing Platt vs. Isotonic) ---")
    reports_dir = Path("reports/")
    models_dir = Path("models/")
    models_dir.mkdir(exist_ok=True)
    calibration_data_path = reports_dir / "sepsis_calibration_data.npz"

    if not calibration_data_path.exists():
        print(f"❌ ERROR: Calibration data not found at '{calibration_data_path}'"); return

    data = np.load(calibration_data_path)
    y_true = data['y_true']
    y_prob_uncalibrated = data['y_prob']
    
    print("Loaded uncalibrated probability data.")

    # --- Cross-Validate both calibration models ---
    X_cal = y_prob_uncalibrated.reshape(-1, 1)
    platt_calibrated_probs = np.zeros_like(y_prob_uncalibrated)
    isotonic_calibrated_probs = np.zeros_like(y_prob_uncalibrated)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Performing 5-fold cross-validation for both calibrators...")
    for train_index, test_index in skf.split(X_cal, y_true):
        X_train_cal, X_test_cal = X_cal[train_index], X_cal[test_index]
        y_train_cal = y_true[train_index]
        
        # Train Platt Scaling model
        platt_calibrator = LogisticRegression(class_weight='balanced')
        platt_calibrator.fit(X_train_cal, y_train_cal)
        platt_calibrated_probs[test_index] = platt_calibrator.predict_proba(X_test_cal)[:, 1]
        
        # Train Isotonic Regression model
        isotonic_calibrator = IsotonicRegression(out_of_bounds='clip')
        isotonic_calibrator.fit(X_train_cal.flatten(), y_train_cal)
        isotonic_calibrated_probs[test_index] = isotonic_calibrator.predict(X_test_cal.flatten())
    
    print("Cross-validation complete.")

    # --- Calculate Brier Scores for all three ---
    brier_uncalibrated = brier_score_loss(y_true, y_prob_uncalibrated)
    brier_platt = brier_score_loss(y_true, platt_calibrated_probs)
    brier_isotonic = brier_score_loss(y_true, isotonic_calibrated_probs)
    
    # --- Determine and save the best calibrator ---
    if brier_isotonic < brier_platt:
        print("Isotonic Regression is the best calibrator. Training final version on all data...")
        best_calibrator = IsotonicRegression(out_of_bounds='clip')
        best_calibrator.fit(y_prob_uncalibrated, y_true)
    else:
        print("Platt Scaling is the best calibrator. Training final version on all data...")
        best_calibrator = LogisticRegression(class_weight='balanced')
        best_calibrator.fit(X_cal, y_true)
        
    calibrator_path = models_dir / "sepsis_calibrator_final.joblib"
    joblib.dump(best_calibrator, calibrator_path)
    print(f"✅ Best calibrator model saved to: {calibrator_path}")

    # --- Generate Final Comparison Plot ---
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_true, y_prob_uncalibrated, n_bins=10, strategy='uniform')
    prob_true_platt, prob_pred_platt = calibration_curve(y_true, platt_calibrated_probs, n_bins=10, strategy='uniform')
    prob_true_iso, prob_pred_iso = calibration_curve(y_true, isotonic_calibrated_probs, n_bins=10, strategy='uniform')

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 9))
    
    ax.plot(prob_pred_uncal, prob_true_uncal, marker='o', linewidth=2, label=f'Uncalibrated (Brier: {brier_uncalibrated:.3f})')
    ax.plot(prob_pred_platt, prob_true_platt, marker='s', linewidth=2, label=f'Calibrated - Platt (Brier: {brier_platt:.3f})')
    ax.plot(prob_pred_iso, prob_true_iso, marker='^', linewidth=2, label=f'Calibrated - Isotonic (Brier: {brier_isotonic:.3f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    ax.set_title("Calibration Curve Comparison (Sepsis)", fontsize=16)
    ax.set_xlabel("Mean Predicted Probability (of Expired)", fontsize=12)
    ax.set_ylabel("Fraction of Positives (True Expired Rate)", fontsize=12)
    ax.legend()
    plt.tight_layout()
    
    final_plot_path = reports_dir / "final_calibration_comparison.png"
    plt.savefig(final_plot_path)
    print(f"\n✅ Final calibration comparison plot saved to: {final_plot_path}")
    plt.show()

if __name__ == '__main__':
    generate_calibration_report()