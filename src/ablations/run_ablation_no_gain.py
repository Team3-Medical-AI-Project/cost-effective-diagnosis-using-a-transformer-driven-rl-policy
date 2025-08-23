"""
Ablation Study: No GAIN Imputer (v2.0 - Efficient)

v2.0: Fixes a major performance bug by pre-calculating the imputed dataset,
      making the training loop computationally efficient.
"""
# --- 1. Imports ---
import torch, numpy as np, pandas as pd, gymnasium as gym, os, sys, yaml
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from sklearn.impute import SimpleImputer

# --- Path Setup ---
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
except NameError:
    sys.path.append(os.path.abspath('..'))

from src.training.train_rl_agent_sepsis import TransformerPolicy
from src.models.classifier import PreliminaryClassifier
from src.utils import load_config, set_seed

# --- 2. The RL Environment with an Efficient Simple Imputer ---
class SepsisEnvSimpleImpute(gym.Env):
    """
    A modified SepsisEnv for the ablation study. This version pre-calculates
    the fully imputed dataset for computational efficiency.
    """
    def __init__(self, config):
        super(SepsisEnvSimpleImpute, self).__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Load data
        train_data_path = os.path.join(config.processed_data_dir, "train_X.csv")
        X_train = pd.read_csv(train_data_path)
        self.X_val = pd.read_csv(os.path.join(config.processed_data_dir, "val_X.csv"))
        self.y_val = pd.read_csv(os.path.join(config.processed_data_dir, "val_y.csv"))
        self.num_patients = len(self.X_val)
        
        # --- EFFICIENT IMPUTATION ---
        print("Initializing environment with SimpleImputer (mean strategy)...")
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(X_train) # Fit the imputer on the training data
        
        # Pre-calculate the fully imputed validation set ONCE.
        X_val_imputed_np = imputer.transform(self.X_val)
        self.X_val_imputed = torch.tensor(X_val_imputed_np, dtype=torch.float32).to(self.device)
        print("Pre-imputed validation set created.")
        
        # Load only the classifier model
        clf_weights = torch.load(config.prelim_classifier_path, map_location=self.device, weights_only=True)
        self.classifier = PreliminaryClassifier(input_dim=self.config.num_features, output_dim=2).to(self.device).eval()
        self.classifier.load_state_dict(clf_weights)
        
        # Define spaces (unchanged)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.num_features * 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.config.num_test_groups + 1)
        self.feature_groups = {i: list(range(10*i, 10*(i+1) if i < 3 else self.config.num_features)) for i in range(self.config.num_test_groups)}
        self.DIAGNOSE_ACTION = self.config.num_test_groups
    
    def action_masks(self):
        # (This method is unchanged)
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
        # Get the corresponding pre-imputed data for this patient
        self.imputed_patient_data = self.X_val_imputed[self.patient_idx]
        self.true_label = self.y_val.iloc[self.patient_idx].values[0]
        self.current_state = torch.zeros(self.config.num_features, device=self.device)
        self.observation_mask = torch.zeros(self.config.num_features, device=self.device)
        self.step_count = 0
        return torch.cat([self.current_state, self.observation_mask]).cpu().numpy(), {}

    def step(self, action):
        done = False; reward = 0.0
        
        # --- EFFICIENT IMPUTATION LOGIC ---
        def get_imputed_state():
            """Combines true observed values with pre-calculated imputed values."""
            return self.current_state * self.observation_mask + self.imputed_patient_data * (1 - self.observation_mask)

        if action == self.DIAGNOSE_ACTION:
            with torch.no_grad():
                imputed_state = get_imputed_state()
                logits = self.classifier(imputed_state.unsqueeze(0))
                prediction = torch.argmax(logits, dim=1).item()
            reward = self.config.reward_correct if prediction == self.true_label else self.config.reward_wrong
            done = True
        else:
            features_to_reveal = self.feature_groups.get(action, [])
            self.observation_mask[features_to_reveal] = 1.0
            self.current_state[features_to_reveal] = self.full_patient_data[features_to_reveal]
            reward -= self.config.cost_mapping.get(action, 0)
            with torch.no_grad():
                imputed_state = get_imputed_state()
                logits = self.classifier(imputed_state.unsqueeze(0))
                probabilities = torch.softmax(logits, dim=1)
                shannon_entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-8))
                reward += (1.0 - shannon_entropy.item()) * self.config.uncertainty_factor
        
        self.step_count += 1
        if self.step_count >= self.config.num_test_groups + 1: done = True
        return torch.cat([self.current_state, self.observation_mask]).cpu().numpy(), reward, done, False, {}

# --- 3. Main Execution ---
if __name__ == '__main__':
    config = load_config("configs/sepsis_config.yaml")
    
    print("--- Running Ablation Study: NO GAIN ---")
    set_seed(config.seed)
    
    ablation_log_dir = os.path.join(config.log_dir, "ablation_no_gain")
    ablation_model_path = os.path.join(config.model_dir, "rl_agent_sepsis_ablation_no_gain.zip")
    
    os.makedirs(ablation_log_dir, exist_ok=True)

    env = SepsisEnvSimpleImpute(config=config)
    env = Monitor(env, ablation_log_dir)
    
    policy_kwargs = dict(features_extractor_class=TransformerPolicy, 
                         features_extractor_kwargs=dict(features_dim=config.features_dim))

    agent = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
                        tensorboard_log=ablation_log_dir, device=config.device, seed=config.seed)

    print("\nTraining agent with SimpleImputer...")
    agent.learn(total_timesteps=config.total_timesteps, progress_bar=True)
    
    agent.save(ablation_model_path)
    print(f"\n✅ Ablation agent saved to: {ablation_model_path}")