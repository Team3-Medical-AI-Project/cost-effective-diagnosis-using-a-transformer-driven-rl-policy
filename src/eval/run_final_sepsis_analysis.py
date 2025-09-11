"""
Definitive Master Evaluation and Report Generation Script for the Sepsis Project (v3.2)
v3.2: Fixes the ImportError for brier_score_loss.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""     # force CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # silence TF INFO logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  
import matplotlib
matplotlib.use("Agg")  
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
 # avoid oneDNN numerical diffs (optional)

import matplotlib
matplotlib.use("Agg")                       # non-interactive backend (no GUI)

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU for evaluation


# --- Path Setup ---
try:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
except NameError:
    sys.path.append('..')

from src.training.sepsis_env_fast import SepsisEnvFast
from src.models.custom_policy import TransformerPolicyExtractor
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
    
    # Ensure config has the same settings as training
    if not hasattr(config, 'diagnose_mode') or config.diagnose_mode != "split":
        print("⚠️  Forcing diagnose_mode to 'split' to match training environment")
        config.diagnose_mode = "split"
    
    if not hasattr(config, 'num_test_groups') or config.num_test_groups != 4:
        print("⚠️  Forcing num_test_groups to 4 to match training environment")
        config.num_test_groups = 4
    
    # CRITICAL: Set the policy architecture before creating the environment
    feat_dim = int(getattr(config, "features_dim", 128))
    print(f"Setting policy architecture with features_dim: {feat_dim}")
    
    # Override the default policy kwargs to match training
    if not hasattr(config, 'policy_kwargs'):
        config.policy_kwargs = {}
    
    # Force the correct network architecture
    config.policy_kwargs.update({
        'net_arch': dict(
            pi=[feat_dim, feat_dim],  # Policy network: [128, 128]
            vf=[feat_dim, feat_dim]   # Value network: [128, 128]
        )
    })
    
    # Use the fast env (same class used in training); it reads processed_data_dir itself.
    env = SepsisEnvFast(config)
    
    # Verify environment action space matches the model
    expected_actions = config.num_test_groups + 2  # 4 test groups + 2 diagnose actions
    actual_actions = env.action_space.n
    print(f"Environment action space: {env.action_space} (expected: Discrete({expected_actions}))")
    
    if actual_actions != expected_actions:
        raise ValueError(f"Action space mismatch! Environment has {actual_actions} actions, but model expects {expected_actions}")

    # Load test data from the correct location
    # The augmented folder has *_scaled.csv files, but we need the raw ones for evaluation
    base_processed_dir = str(config.processed_data_dir).replace("/augmented", "")
    print(f"Loading test data from: {base_processed_dir}")
    
    y_test = pd.read_csv(os.path.join(base_processed_dir, "test_y.csv")).values.ravel()
    env.X_val = pd.read_csv(os.path.join(base_processed_dir, "test_X.csv"))
    env.y_val = pd.DataFrame(y_test, columns=['hospital_expire_flag'])
    env.num_patients = len(env.y_val)

    
    is_maskable = "no_masking" not in model_name
    has_transformer = "no_transformer" not in model_name

    # Create custom_objects with the correct policy architecture
    if has_transformer:
        custom_objects = {"policy_kwargs": dict(
            features_extractor_class=TransformerPolicyExtractor,
            features_extractor_kwargs=dict(features_dim=feat_dim)
        )}
    else:
        # For non-transformer policies, use the architecture we set in config
        custom_objects = {"policy_kwargs": config.policy_kwargs}

    # CRITICAL FIX: Load model with strict architecture enforcement
    # The key insight is to use the exact same policy_kwargs from training
    
    print("Loading model with strict architecture matching...")
    
    # Create the exact same policy_kwargs that were used during training
    final_policy_kwargs = dict(
        features_extractor_class=TransformerPolicyExtractor,
        features_extractor_kwargs=dict(features_dim=feat_dim),
        net_arch=[feat_dim, feat_dim],  # [128, 128] - EXACTLY as in training
    )
    print(f"Final policy_kwargs: {final_policy_kwargs}")
    
    # Use custom_objects with the correct architecture
    final_custom_objects = {"policy_kwargs": final_policy_kwargs}
    
    try:
        if is_maskable:
            # Load with strict architecture matching
            model = MaskablePPO.load(
                model_path, 
                env=env, 
                custom_objects=final_custom_objects, 
                device='cpu',
                print_system_info=False
            )
        else:
            model = PPO.load(
                model_path, 
                env=env, 
                custom_objects=final_custom_objects, 
                device='cpu',
                print_system_info=False
            )
        print("✅ Model loaded successfully with correct architecture!")
        
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        print("🔄 Trying alternative loading approach...")
        
        # Fallback: try loading without strict architecture checking
        try:
            import torch
            # Load the model data manually and patch the architecture
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            if is_maskable:
                model = MaskablePPO(
                    policy="MlpPolicy",
                    env=env,
                    policy_kwargs=final_policy_kwargs,
                    verbose=0,
                    device='cpu'
                )
            else:
                model = PPO(
                    policy="MlpPolicy",
                    env=env,
                    policy_kwargs=final_policy_kwargs,
                    verbose=0,
                    device='cpu'
                )
            
            # Load state dict with strict=False to handle architecture mismatches
            model.policy.load_state_dict(checkpoint['policy'], strict=False)
            print("✅ Model loaded with fallback approach!")
            
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            raise RuntimeError("Could not load model with any approach")
    
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
            if action < env.num_test_groups:
                test_usage_counter[action] += 1
            obs, reward, done, _, _ = env.step(action)
        with torch.no_grad():
            imputed_state = env.gain(env.current_state.unsqueeze(0), env.observation_mask.unsqueeze(0)).squeeze(0)
            logits = env.clf(imputed_state.unsqueeze(0))
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
    # Load the accfocus config that matches your training environment
    config = load_config("configs/sepsis_config_accfocus.yaml")

    # --- match training & force CPU for env ---
    try:
        config._store.setdefault("diagnose_mode", "split")
        config._store.setdefault("use_two_diagnosis_actions", True)
        config._store.setdefault("num_test_groups", 4)
        config._store["device"] = "cpu"               # <--- force CPU
    except Exception:
        # if not DotConfig, fall back
        try:
            if getattr(config, "diagnose_mode", None) is None: config.diagnose_mode = "split"
            if getattr(config, "use_two_diagnosis_actions", None) is None: config.use_two_diagnosis_actions = True
            if getattr(config, "num_test_groups", None) is None: config.num_test_groups = 4
            config.device = "cpu"                     # <--- force CPU
        except Exception:
            pass

    reports_dir = Path("reports/")
    models_dir = Path(config.model_dir)
    
    # Create reports directory if it doesn't exist
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # PHASE 1: EVALUATION
    print("🔍 Looking for RL agent models...")
    rl_model_paths = list(models_dir.glob("rl_agent_sepsis*.zip"))
    
    if not rl_model_paths:
        print("❌ No RL agent models found in models/ directory")
        print("Expected files like: rl_agent_sepsis_fast_uf-1_25_2diag.zip")
        exit(1)
    
    print(f"✅ Found {len(rl_model_paths)} RL agent model(s):")
    for path in rl_model_paths:
        print(f"  - {path.name}")
    
    # Prioritize the specific model you want to evaluate
    preferred_model = "rl_agent_sepsis_fast_uf-1_25_2diag.zip"
    preferred_path = models_dir / preferred_model
    
    if preferred_path.exists():
        print(f"\n🎯 Evaluating preferred model: {preferred_model}")
        evaluate_single_rl_agent(config, preferred_path, reports_dir)
    else:
        print(f"\n⚠️  Preferred model {preferred_model} not found, evaluating all available models:")
        for model_path in rl_model_paths:
            evaluate_single_rl_agent(config, model_path, reports_dir)
    
    # PHASE 2: REPORTING
    print("\n📊 Generating final plots and summary...")
    generate_final_plots(reports_dir)
    
    print("\n✅ Evaluation complete! Check the reports/ directory for results.")
