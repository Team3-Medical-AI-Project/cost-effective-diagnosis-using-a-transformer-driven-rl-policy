"""
Classifier-Only Baseline Evaluation Script for Sepsis Cohort

This script evaluates a heuristic policy that uses the pre-trained
preliminary classifier to make a simple, uncertainty-based decision.
"""
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys
import yaml

# --- Path Setup ---
try:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
except NameError:
    sys.path.append('..')

# Import necessary components
from src.training.train_rl_agent_sepsis import SepsisEnv, Config
from src.models.gain import Generator
from src.models.classifier import PreliminaryClassifier

def run_classifier_heuristic(config_path: str = "configs/sepsis_config.yaml"):
    """
    Simulates a heuristic policy over the entire test set.
    """
    print("\n--- Simulating Policy: Classifier-Only Heuristic ---")

    # --- 1. Load Config and Models ---
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Config(config_dict)

    device = torch.device(config.device)
    
    # Load the pre-trained models this policy relies on
    gen_weights = torch.load(config.gain_generator_path, map_location=device, weights_only=True)
    clf_weights = torch.load(config.prelim_classifier_path, map_location=device, weights_only=True)

    generator = Generator(input_dim=config.num_features).to(device).eval()
    generator.load_state_dict(gen_weights)
    
    classifier = PreliminaryClassifier(input_dim=config.num_features, output_dim=2).to(device).eval()
    classifier.load_state_dict(clf_weights)
    
    # --- 2. Load Test Data ---
    test_data_dir = Path(config.processed_data_dir)
    X_test = pd.read_csv(test_data_dir / "test_X.csv")
    y_test = pd.read_csv(test_data_dir / "test_y.csv").values.ravel()
    num_patients = len(X_test)
    print(f"Loaded TEST data with {num_patients} patients.")

    # --- 3. Simulation Loop ---
    total_costs, num_correct_diagnoses, episode_lengths = [], 0, []
    
    # Define the heuristic's parameters
    UNCERTAINTY_THRESHOLD = 0.3 # Corresponds to the classifier being <70% confident
    CHEAPEST_PANEL_ACTION = 0 # CBC
    SECOND_CHEAPEST_PANEL_ACTION = 3 # aPTT

    for i in range(num_patients):
        full_patient_data = torch.tensor(X_test.iloc[i].values, dtype=torch.float32).to(device)
        true_label = y_test[i]
        
        current_state = torch.zeros(config.num_features, device=device)
        observation_mask = torch.zeros(config.num_features, device=device)
        
        episode_cost = 0
        
        # --- Heuristic Policy Logic ---
        # Step 1: Always order the cheapest panel (CBC)
        features_to_reveal = SepsisEnv(config).feature_groups[CHEAPEST_PANEL_ACTION] # A bit of a hack to get groups
        observation_mask[features_to_reveal] = 1.0
        current_state[features_to_reveal] = full_patient_data[features_to_reveal]
        episode_cost += config.cost_mapping[CHEAPEST_PANEL_ACTION]
        
        # Step 2: Check uncertainty
        with torch.no_grad():
            imputed_state = generator(current_state.unsqueeze(0), observation_mask.unsqueeze(0)).squeeze(0)
            logits = classifier(imputed_state.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1).squeeze()
            uncertainty = 1 - torch.max(probabilities).item()

        # Step 3: If uncertain, order the next cheapest panel (aPTT)
        if uncertainty > UNCERTAINTY_THRESHOLD:
            features_to_reveal = SepsisEnv(config).feature_groups[SECOND_CHEAPEST_PANEL_ACTION]
            observation_mask[features_to_reveal] = 1.0
            current_state[features_to_reveal] = full_patient_data[features_to_reveal]
            episode_cost += config.cost_mapping[SECOND_CHEAPEST_PANEL_ACTION]

        # Step 4: Final Diagnosis
        with torch.no_grad():
            imputed_state = generator(current_state.unsqueeze(0), observation_mask.unsqueeze(0)).squeeze(0)
            logits = classifier(imputed_state.unsqueeze(0))
            prediction = torch.argmax(logits, dim=1).item()
            
        if prediction == true_label:
            num_correct_diagnoses += 1
            
        total_costs.append(episode_cost)
        episode_lengths.append(torch.sum(observation_mask != 0).item() / 10) # Approximate panel count

    # --- 4. Report and Save Results ---
    accuracy = (num_correct_diagnoses / num_patients) * 100
    avg_cost = np.mean(total_costs)
    avg_tests = np.mean(episode_lengths)

    results = {
        "model_type": "heuristic_classifier_policy",
        "metrics": {"accuracy": accuracy},
        "cost": {"average_usd_cost": avg_cost, "average_num_tests": avg_tests}
    }
    
    reports_dir = Path("reports/")
    reports_dir.mkdir(exist_ok=True)
    output_path = reports_dir / "baseline_classifier_policy_results.json"

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"✅ Results saved to: {output_path}")
    print(json.dumps(results, indent=4))


if __name__ == '__main__':
    run_classifier_heuristic()