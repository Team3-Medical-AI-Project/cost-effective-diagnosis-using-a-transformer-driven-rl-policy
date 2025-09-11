"""
Final Evaluation Script for the Single-Test Action RL Agent (v7.0)

This script evaluates the final, clinically-constrained agent that learns to
select individual tests, providing a detailed breakdown of its performance
and diagnostic strategy.
"""
# --- 1. Imports ---
import torch
import numpy as np
import pandas as pd
import gymnasium as gym
import os
import sys
from collections import Counter

# Use the correct agent for action masking
from sb3_contrib import MaskablePPO

# --- Path Setup ---
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
except NameError:
    sys.path.append(os.path.abspath('..'))

# Import the environment and config from the final AKI training script
from src.training.train_rl_agent_aki import AkiEnv, Config, TransformerPolicy

# --- 2. Evaluation Main Function ---
def evaluate_aki_agent():
    """
    Loads the trained single-test agent, evaluates it on the test set,
    and reports on accuracy, cost, and individual test selection frequency.
    """
    print("--- Evaluating Final Single-Test AKI Agent ---")
    
    config = Config()

    # Load the final agent trained with the single-test methodology
    agent_path = os.path.join(config.MODEL_DIR, "rl_agent_aki_single_test_final.zip")
    if not os.path.exists(agent_path):
        print(f"Error: Model not found at {agent_path}. Please train the agent first.")
        return

    # 1. Set up the Environment with the TEST data
    env = AkiEnv(config=config)
    # The test set features must be loaded to get the correct column names for the report
    test_features = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "test_X.csv"))
    test_labels = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "test_y.csv"))
    
    env.X_val = test_features
    env.y_val = test_labels
    env.num_patients = len(env.X_val)
    print(f"Loaded TEST data with {env.num_patients} patients.")
    
    # 2. Load the Trained Agent
    model = MaskablePPO.load(agent_path)
    print("Trained agent loaded successfully.")

    # 3. Initialize Metric Trackers
    total_rewards = []
    episode_lengths = [] # Will now represent the number of individual tests
    num_correct_diagnoses = 0
    test_usage_counter = Counter()

    # 4. Run Evaluation Loop
    print("\nRunning evaluation across all test patients...")
    for i in range(env.num_patients):
        obs, info = env.reset()
        done = False
        
        while not done:
            action_masks = env.action_masks()
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            action = action.item()
            
            # Track which individual test was chosen by its feature name
            if action != env.DIAGNOSE_ACTION:
                feature_name = env.X_val.columns[action]
                test_usage_counter[feature_name] += 1
            
            obs, reward, done, truncated, info = env.step(action)
        
        # The episode is over, record final metrics
        if reward > 0:  # A positive final reward means a correct diagnosis
            num_correct_diagnoses += 1
        total_rewards.append(reward)
        episode_lengths.append(env.step_count)

    # 5. Calculate and Report Final Metrics
    avg_reward = np.mean(total_rewards)
    avg_tests_ordered = np.mean(episode_lengths)
    accuracy = (num_correct_diagnoses / env.num_patients) * 100

    # --- Print Final Report ---
    print("\n" + "="*50 + "\n--- Final AKI Performance Report ---\n" + "="*50)
    print(f"📈 Final Diagnostic Accuracy: {accuracy:.2f}%")
    print(f"📉 Average Number of Tests Ordered: {avg_tests_ordered:.2f}")
    print(f"📊 Average Reward: {avg_reward:,.2f}")
    print("\n" + "="*50 + "\n--- Agent's Top 15 Most Frequently Ordered Tests ---\n" + "="*50)
    
    # Display the top 15 most frequently chosen individual tests
    for feature, count in test_usage_counter.most_common(15):
        print(f"  - {feature}: {count} times")
    print("="*50)

if __name__ == "__main__":
    evaluate_aki_agent()