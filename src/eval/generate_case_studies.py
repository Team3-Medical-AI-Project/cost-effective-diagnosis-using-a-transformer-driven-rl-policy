"""
Patient Trajectory Case Study Generator for the Sepsis RL Agent

This script loads the final trained RL agent and runs it on a few specific
patients from the test set. It logs every action, cost, and the change in
diagnostic probability to create a narrative case study for the final report.
"""
import torch
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import yaml

# --- Path Setup & Imports ---
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
except NameError:
    sys.path.append(os.path.abspath('..'))

from sb3_contrib import MaskablePPO
from src.training.train_rl_agent_sepsis import SepsisEnv, Config, TransformerPolicy
from src.utils import load_config

def generate_case_studies(config_path: str, model_path: str, patient_indices: list, output_dir: Path):
    """
    Runs specific patients through the trained agent and prints a detailed trajectory.
    """
    print("\n" + "="*50 + "\n--- Generating Patient Trajectory Case Studies ---\n" + "="*50)
    
    config = load_config(config_path)
    output_dir.mkdir(exist_ok=True)
    
    # --- 1. Load Environment and Model ---
    env = SepsisEnv(config)
    y_test = pd.read_csv(os.path.join(config.processed_data_dir, "test_y.csv")).values.ravel()
    env.X_val = pd.read_csv(os.path.join(config.processed_data_dir, "test_X.csv"))
    env.y_val = pd.DataFrame(y_test, columns=['hospital_expire_flag'])
    env.num_patients = len(env.X_val)
    
    custom_objects = {"policy_kwargs": dict(features_extractor_class=TransformerPolicy, features_extractor_kwargs=dict(features_dim=config.features_dim))}
    model = MaskablePPO.load(model_path, env=env, custom_objects=custom_objects)
    print("Trained agent loaded successfully.")

    PANEL_NAMES = {0: "CBC", 1: "CMP", 2: "ABG", 3: "aPTT"}

    # --- 2. Loop Through Specified Patients ---
    for patient_index in patient_indices:
        print("\n" + "-"*50)
        print(f"--- Case Study for Test Patient Index: {patient_index} ---")
        
        # Manually reset the environment to a specific patient
        env.patient_idx = patient_index
        obs, info = env.reset(seed=42) # Use a seed for consistency in this call
        
        true_outcome = "Expired" if env.true_label == 1 else "Discharged"
        print(f"True Patient Outcome: {true_outcome}")
        
        done = False
        step = 0
        total_cost = 0.0
        
        while not done:
            step += 1
            print(f"\nStep {step}:")
            
            # Get current probability before taking an action
            with torch.no_grad():
                imputed_state = env.gain_generator(env.current_state.unsqueeze(0), env.observation_mask.unsqueeze(0)).squeeze(0)
                logits = env.classifier(imputed_state.unsqueeze(0))
                probabilities = torch.softmax(logits, dim=1).squeeze()
                current_prob_expired = probabilities[1].item()
            print(f"  - Agent's Current Confidence (Prob. Expired): {current_prob_expired:.2%}")
            
            # Agent makes a decision
            action_masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            action = action.item()
            
            if action == env.DIAGNOSE_ACTION:
                print(f"  - Agent's Action: Make Final Diagnosis")
            else:
                panel_name = PANEL_NAMES.get(action, "Unknown")
                cost = config.cost_mapping.get(action, 0)
                total_cost += cost
                print(f"  - Agent's Action: Order '{panel_name}' (Cost: ${cost:.2f})")
            
            # Environment updates
            obs, reward, done, _, _ = env.step(action)
        
        # --- Final Outcome ---
        final_prediction_prob = current_prob_expired
        final_prediction = 1 if final_prediction_prob > 0.5 else 0
        final_outcome = "Expired" if final_prediction == 1 else "Discharged"
        
        print("\n--- Episode End ---")
        print(f"Final Prediction: {final_outcome} (Confidence: {final_prediction_prob:.2%})")
        print(f"Total Cost Incurred: ${total_cost:.2f}")
        print(f"Result: {'CORRECT' if final_prediction == env.true_label else 'INCORRECT'}")
        print("-"*50)

if __name__ == '__main__':
    # --- Configuration ---
    CONFIG_PATH = "configs/sepsis_config.yaml"
    MODEL_PATH = "models/rl_agent_sepsis_final_reproducible.zip"
    

    PATIENT_INDICES_TO_ANALYZE = [2,3,18,55,100,120] # Example: Analyze the 10th, 25th, and 50th patients
    
    REPORTS_DIR = Path("reports/")
    
    generate_case_studies(CONFIG_PATH, MODEL_PATH, PATIENT_INDICES_TO_ANALYZE, REPORTS_DIR)