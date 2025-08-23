"""
Definitive Master Evaluation and Report Generation Script for the Sepsis Project (v3.2)
v3.2: Fixes the ImportError for brier_score_loss.
"""
import torch, numpy as np, pandas as pd, gymnasium as gym, os, sys, yaml, json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve

# --- CORRECTED IMPORTS ---
# All metrics functions are correctly imported from sklearn.metrics
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from adjustText import adjust_text

# --- Path Setup ---
try:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
except NameError:
    sys.path.append('..')

from src.training.train_rl_agent_sepsis import SepsisEnv, Config, TransformerPolicy
from src.utils import load_config

def evaluate_single_rl_agent(config, model_path: Path, reports_dir: Path):
    model_name = model_path.stem
    report_path = reports_dir / f"{model_name}_results.json"
    
    if report_path.exists():
        print(f"--- Skipping Evaluation for {model_name} (report already exists) ---")
        if model_name == "rl_agent_sepsis_final_reproducible" and not (reports_dir / "sepsis_calibration_data.npz").exists():
             pass
        else:
            return

    print(f"\n--- Evaluating Model: {model_name} ---")
    
    env = SepsisEnv(config)
    y_test = pd.read_csv(os.path.join(config.processed_data_dir, "test_y.csv")).values.ravel()
    env.X_val = pd.read_csv(os.path.join(config.processed_data_dir, "test_X.csv"))
    env.y_val = pd.DataFrame(y_test, columns=['hospital_expire_flag'])
    env.num_patients = len(env.X_val)
    
    is_maskable = "no_masking" not in model_name
    has_transformer = "no_transformer" not in model_name

    custom_objects = {}
    if has_transformer:
        custom_objects = {"policy_kwargs": dict(features_extractor_class=TransformerPolicy, features_extractor_kwargs=dict(features_dim=config.features_dim))}

    if is_maskable:
        model = MaskablePPO.load(model_path, env=env, custom_objects=custom_objects)
    else:
        model = PPO.load(model_path, env=env, custom_objects=custom_objects)
    
    final_predictions, final_probabilities = [], []
    test_usage_counter = Counter()

    for i in range(env.num_patients):
        obs, info = env.reset()
        done = False
        while not done:
            if is_maskable:
                action_masks = env.action_masks()
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            action = action.item()
            if action != env.DIAGNOSE_ACTION:
                test_usage_counter[action] += 1
            obs, reward, done, _, _ = env.step(action)
        with torch.no_grad():
            imputed_state = env.gain_generator(env.current_state.unsqueeze(0), env.observation_mask.unsqueeze(0)).squeeze(0)
            logits = env.classifier(imputed_state.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1).squeeze()
            final_predictions.append(torch.argmax(probabilities).item())
            final_probabilities.append(probabilities[1].item())
            
    accuracy = np.mean(np.array(final_predictions) == y_test) * 100
    auroc = roc_auc_score(y_test, final_probabilities)
    auprc = average_precision_score(y_test, final_probabilities)
    total_dollar_cost = sum(config.cost_mapping.get(action, 0) for action, count in test_usage_counter.items())
    avg_dollar_cost = total_dollar_cost / env.num_patients if env.num_patients > 0 else 0
    avg_num_tests = sum(test_usage_counter.values()) / env.num_patients if env.num_patients > 0 else 0

    results = {
        "model_name": model_name, "metrics": {"accuracy": accuracy, "auroc": auroc, "auprc": auprc},
        "cost": {"avg_usd_cost": avg_dollar_cost, "avg_num_tests": avg_num_tests}, "test_usage": dict(test_usage_counter)
    }
    with open(report_path, 'w') as f: json.dump(results, f, indent=4)
    print(f"✅ Evaluation report saved to: {report_path}")

    if "final_reproducible" in model_name:
        calibration_data_path = reports_dir / "sepsis_calibration_data.npz"
        np.savez(calibration_data_path, y_true=y_test, y_prob=np.array(final_probabilities))
        print(f"✅ Data for calibration plot saved to: {calibration_data_path}")

def generate_final_plots(reports_dir: Path):
    print("\n" + "="*50 + "\n--- Generating Final Plots and Summary Table ---" + "="*50)
    all_results = []
    for json_path in reports_dir.glob("*_results.json"):
        with open(json_path, 'r') as f:
            data = json.load(f)
            model_name_full = data.get("model_type", data.get("model_name"))
            model_name_short = model_name_full.replace("rl_agent_sepsis_", "").replace("_results", "").replace("baseline_", "")
            
            result_row = {
                "Model": model_name_short,
                "Accuracy (%)": data["metrics"].get("accuracy"), "AUROC": data["metrics"].get("auroc"),
                "AUPRC": data["metrics"].get("auprc"), "Avg Cost ($)": data["cost"].get("avg_usd_cost"),
                "Avg # Tests": data["cost"].get("avg_num_tests", data["cost"].get("average_decision_steps"))
            }
            all_results.append(result_row)
    
    if not all_results: print("❌ No result files found to generate plots."); return

    summary_df = pd.DataFrame(all_results).round(3).sort_values(by="Accuracy (%)", ascending=False).set_index("Model")
    summary_table_path = reports_dir / "summary_table_sepsis.csv"
    summary_df.to_csv(summary_table_path)
    print(f"\n✅ Consolidated summary table saved to: {summary_table_path}")
    print("--- Summary Table ---"); print(summary_df.to_string())

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))
    plot_df = summary_df.dropna(subset=["Avg Cost ($)", "Accuracy (%)"]).copy()
    
    def get_model_type(model_name):
        if "logreg" in model_name or "xgboost" in model_name: return "Baseline (ML)"
        if "rule" in model_name: return "Baseline (Rule)"
        if "ablation" in model_name: return "RL Agent (Ablation)"
        if "unc-" in model_name: return "RL Agent (Sweep)"
        return "RL Agent (Main)"
    plot_df['Model Type'] = plot_df.index.to_series().apply(get_model_type)
    
    sns.scatterplot(data=plot_df, x="Avg Cost ($)", y="Accuracy (%)", hue="Model Type", s=250, ax=ax, palette="deep", style="Model Type")
    texts = [ax.text(row["Avg Cost ($)"], row["Accuracy (%)"], index, fontsize=9) for index, row in plot_df.iterrows()]
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
    ax.set_title("Cost vs. Accuracy Pareto Front (Sepsis)", fontsize=18)
    ax.set_xlabel("Average Financial Cost per Patient ($)", fontsize=14)
    ax.set_ylabel("Diagnostic Accuracy (%)", fontsize=14)
    ax.legend(title="Model Type")
    plt.tight_layout()
    pareto_plot_path = reports_dir / "final_pareto_front_sepsis.png"
    plt.savefig(pareto_plot_path, bbox_inches='tight')
    print(f"\n✅ Pareto front plot saved to: {pareto_plot_path}")
    plt.show()

    calibration_data_path = reports_dir / "sepsis_calibration_data.npz"
    if calibration_data_path.exists():
        data = np.load(calibration_data_path)
        y_true, y_prob = data['y_true'], data['y_prob']
        brier_score = brier_score_loss(y_true, y_prob)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label=f'Sepsis RL Agent (Brier: {brier_score:.3f})')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        ax.set_title("Calibration Curve (Main RL Agent)", fontsize=16)
        ax.set_xlabel("Mean Predicted Probability (of Expired)", fontsize=12)
        ax.set_ylabel("Fraction of Positives (True Expired Rate)", fontsize=12)
        ax.legend()
        plt.tight_layout()
        calibration_plot_path = reports_dir / "final_calibration_sepsis.png"
        plt.savefig(calibration_plot_path)
        print(f"✅ Calibration plot saved to: {calibration_plot_path}")
        plt.show()
    
    main_model_json_path = reports_dir / "rl_agent_sepsis_final_reproducible_results.json"
    if main_model_json_path.exists():
        with open(main_model_json_path, 'r') as f: data = json.load(f).get("test_usage", {})
        PANEL_NAMES = {0: "CBC", 1: "CMP", 2: "ABG", 3: "aPTT"}
        usage = {PANEL_NAMES.get(int(k), "Unknown"): v for k, v in data.items()}
        
        if usage:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=list(usage.keys()), y=list(usage.values()), ax=ax, palette="plasma")
            ax.set_title("Main Agent's Panel Selection Frequency", fontsize=16)
            ax.set_ylabel("Times Ordered")
            usage_path = reports_dir / "final_panel_usage_sepsis.png"
            plt.savefig(usage_path)
            print(f"✅ Panel usage plot saved to: {usage_path}")
            plt.show()

if __name__ == '__main__':
    config = load_config("configs/sepsis_config.yaml")
    reports_dir = Path("reports/")
    models_dir = Path(config.model_dir)
    
    # PHASE 1: EVALUATION
    rl_model_paths = list(models_dir.glob("rl_agent_sepsis*.zip"))
    if rl_model_paths:
        for model_path in rl_model_paths:
            evaluate_single_rl_agent(config, model_path, reports_dir)
    
    # PHASE 2: REPORTING
    generate_final_plots(reports_dir)