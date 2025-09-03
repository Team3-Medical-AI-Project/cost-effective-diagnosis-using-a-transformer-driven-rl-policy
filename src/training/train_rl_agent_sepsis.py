"""
Main Training Script for the Sepsis RL Agent (v8.0 - Final Definitive)

v8.0: This definitive version is fully self-contained, reproducible, and
      implements an asymmetric reward structure to train a more clinically
      responsible and accurate agent.
"""
# --- 1. Imports ---
import torch, numpy as np, pandas as pd, gymnasium as gym, os, sys, yaml, argparse, random
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor

# --- Path Setup ---
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
except NameError:
    sys.path.append(os.path.abspath('..'))

from src.models.gain import Generator
from src.models.classifier import PreliminaryClassifier
from src.models.transformer import TransformerSelector

# --- 2. Helper Classes & Functions (Self-Contained) ---
class Config:
    """A robust helper class to access dictionary keys from the YAML file as attributes."""
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict) and all(isinstance(k, str) for k in value.keys()):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

def load_config(config_path: str):
    """Loads a YAML config file and returns it as a Config object."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)

def set_seed(seed):
    """Sets random seeds for reproducibility across all libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# --- 3. The RL Environment with Asymmetric Rewards ---
class SepsisEnv(gym.Env):
    def __init__(self, config):
        super(SepsisEnv, self).__init__()
        self.config = config
        # Force CPU device for evaluation/training in this script
        self.device = torch.device("cpu")
        self.X_val = pd.read_csv(os.path.join(config.processed_data_dir, "val_X.csv"))
        self.y_val = pd.read_csv(os.path.join(config.processed_data_dir, "val_y.csv"))
        self.num_patients = len(self.X_val)
        
        # Force CPU-safe deserialization for weights (works on any device at inference)
        gen_weights = torch.load(config.gain_generator_path, map_location=torch.device("cpu"), weights_only=True)
        clf_weights = torch.load(config.prelim_classifier_path, map_location=torch.device("cpu"), weights_only=True)

        self.gain_generator = Generator(input_dim=self.config.num_features).to(self.device).eval()
        self.gain_generator.load_state_dict(gen_weights)
        self.classifier = PreliminaryClassifier(input_dim=self.config.num_features, output_dim=2).to(self.device).eval()
        self.classifier.load_state_dict(clf_weights)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.num_features * 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.config.num_test_groups + 1)
        self.feature_groups = {i: list(range(10*i, 10*(i+1) if i < 3 else self.config.num_features)) for i in range(self.config.num_test_groups)}
        self.DIAGNOSE_ACTION = self.config.num_test_groups

    def action_masks(self):
        mask = np.ones(self.action_space.n, dtype=np.int8)
        if self.step_count < self.config.min_tests_before_diagnosis:
            mask[self.DIAGNOSE_ACTION] = 0
        for action, features in self.feature_groups.items():
            if self.observation_mask[features[0]].item() == 1:
                mask[action] = 0
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.patient_idx = np.random.randint(self.num_patients)
        self.full_patient_data = torch.tensor(self.X_val.iloc[self.patient_idx].values, dtype=torch.float32).to(self.device)
        self.true_label = self.y_val.iloc[self.patient_idx].values[0]
        self.current_state = torch.zeros(self.config.num_features, device=self.device)
        self.observation_mask = torch.zeros(self.config.num_features, device=self.device)
        self.step_count = 0
        return torch.cat([self.current_state, self.observation_mask]).cpu().numpy(), {}

    def step(self, action):
        done = False; reward = 0.0
        if action == self.DIAGNOSE_ACTION:
            with torch.no_grad():
                imputed_state = self.gain_generator(self.current_state.unsqueeze(0), self.observation_mask.unsqueeze(0)).squeeze(0)
                logits = self.classifier(imputed_state.unsqueeze(0))
                prediction = torch.argmax(logits, dim=1).item()
            
            # --- Asymmetric Reward Logic ---
            if prediction == 1 and self.true_label == 1: # True Positive
                reward = self.config.reward_true_positive
            elif prediction == 0 and self.true_label == 0: # True Negative
                reward = self.config.reward_true_negative
            elif prediction == 1 and self.true_label == 0: # False Positive
                reward = self.config.penalty_false_positive
            elif prediction == 0 and self.true_label == 1: # False Negative (worst case)
                reward = self.config.penalty_false_negative
            
            done = True
        else:
            features_to_reveal = self.feature_groups.get(action, [])
            self.observation_mask[features_to_reveal] = 1.0
            self.current_state[features_to_reveal] = self.full_patient_data[features_to_reveal]
            reward -= self.config.cost_mapping.get(action, 0)
            with torch.no_grad():
                imputed_state = self.gain_generator(self.current_state.unsqueeze(0), self.observation_mask.unsqueeze(0)).squeeze(0)
                logits = self.classifier(imputed_state.unsqueeze(0))
                probabilities = torch.softmax(logits, dim=1)
                shannon_entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-8))
                reward += (1.0 - shannon_entropy.item()) * self.config.uncertainty_factor
        
        self.step_count += 1
        if self.step_count >= self.config.num_test_groups + 1: done = True
        return torch.cat([self.current_state, self.observation_mask]).cpu().numpy(), reward, done, False, {}

# --- 4. Custom Transformer Policy ---
class TransformerPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int):
        super(TransformerPolicy, self).__init__(observation_space, features_dim)
        self.transformer_selector = TransformerSelector(
            input_dim=observation_space.shape[0], embed_dim=features_dim,
            num_heads=4, ff_dim=128, dropout=0.1
        )
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.transformer_selector(observations)

# --- 5. Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sepsis_config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    
    print("--- Starting RL Agent Training (from Config File) ---")
    set_seed(config.seed)
    print(f"Running with random seed: {config.seed}")
    
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    env = SepsisEnv(config=config)
    env = Monitor(env, config.log_dir)
    
    policy_kwargs = dict(features_extractor_class=TransformerPolicy, 
                         features_extractor_kwargs=dict(features_dim=config.features_dim))

    agent = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
                        tensorboard_log=config.log_dir, device=config.device, seed=config.seed)

    print("\nTraining a clinically-constrained agent with asymmetric rewards...")
    agent.learn(total_timesteps=config.total_timesteps, progress_bar=True)
    
    final_model_path = os.path.join(config.model_dir, "rl_agent_sepsis_asymmetric_final.zip")
    agent.save(final_model_path)
    print(f"\n✅ Final agent saved to: {final_model_path}")