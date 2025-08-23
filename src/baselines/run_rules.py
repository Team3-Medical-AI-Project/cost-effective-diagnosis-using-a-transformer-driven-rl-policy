import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys
import yaml

try:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
except NameError:
    sys.path.append('..')

# This import will now work correctly
from src.training.train_rl_agent_sepsis import SepsisEnv, Config

def simulate_policy(policy_type: str, env: SepsisEnv, config: Config):
    print(f"\n--- Simulating Policy: {policy_type.replace('_', ' ').title()} ---")
    num_episodes = env.num_patients
    total_rewards, total_costs, num_correct_diagnoses, episode_lengths = [], [], 0, []

    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        
        if policy_type == 'order_nothing':
            action = env.DIAGNOSE_ACTION
            obs, reward, done, _, _ = env.step(action)
        elif policy_type == 'order_all':
            for test_action in range(config.num_test_groups):
                obs, reward, done, _, _ = env.step(test_action)
            action = env.DIAGNOSE_ACTION
            obs, reward, done, _, _ = env.step(action)
        
        if reward > 0:
            num_correct_diagnoses += 1
        total_rewards.append(reward)
        
        if policy_type == 'order_nothing':
            total_costs.append(0)
            episode_lengths.append(1)
        elif policy_type == 'order_all':
            total_costs.append(sum(config.cost_mapping.values()))
            episode_lengths.append(config.num_test_groups + 1)
            
    accuracy = (num_correct_diagnoses / num_episodes) * 100
    avg_reward = np.mean(total_rewards)
    avg_cost = np.mean(total_costs)
    avg_steps = np.mean(episode_lengths)

    results = {
        "model_type": f"rule_{policy_type}",
        "metrics": {"accuracy": accuracy, "average_reward": avg_reward},
        "cost": {"average_usd_cost": avg_cost, "average_decision_steps": avg_steps}
    }
    
    reports_dir = Path("reports/")
    reports_dir.mkdir(exist_ok=True)
    output_path = reports_dir / f"baseline_{policy_type}_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✅ Results saved to: {output_path}")

if __name__ == '__main__':
    with open("configs/sepsis_config.yaml", 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Config(config_dict)

    env = SepsisEnv(config=config)
    env.X_val = pd.read_csv(Path(config.processed_data_dir) / "test_X.csv")
    env.y_val = pd.read_csv(Path(config.processed_data_dir) / "test_y.csv")
    env.num_patients = len(env.X_val)
    print(f"Loaded TEST data with {env.num_patients} patients.")
    
    simulate_policy('order_nothing', env, config)
    simulate_policy('order_all', env, config)