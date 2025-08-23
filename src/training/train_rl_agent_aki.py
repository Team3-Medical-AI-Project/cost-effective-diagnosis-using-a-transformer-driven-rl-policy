"""
Main Training Script for the AKI RL Agent (v7.3 - Final Config)

v7.3: Corrects the feature count to 43 to align with the final data pipeline.
      This is the definitive version.
"""
# --- 1. Imports ---
import torch, numpy as np, pandas as pd, gymnasium as gym, os, sys
from gymnasium import spaces
from sb3_contrib import MaskablePPO
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

# --- 2. Configuration ---
class Config:
    PROCESSED_DATA_DIR = "data/processed/aki"
    MODEL_DIR = "models"
    LOG_DIR = "logs_aki_final"
    GAIN_GENERATOR_PATH = os.path.join(MODEL_DIR, "generator_aki.pth")
    PRELIM_CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier_aki.pth")
    
    # --- MODIFIED: Parameters aligned with the 43 features from your pipeline ---
    NUM_FEATURES = 43
    ACTION_DIM = NUM_FEATURES + 1
    
    MIN_TESTS_BEFORE_DIAGNOSIS = 5
    COST_PER_TEST = -1.0
    REWARD_CORRECT_DIAGNOSIS = 100.0
    REWARD_WRONG_DIAGNOSIS = -100.0
    UNCERTAINTY_REWARD_FACTOR = 0.1
    TOTAL_TIMESTEPS = 100000

# --- 3. The RL Environment (AkiEnv) ---
class AkiEnv(gym.Env):
    def __init__(self, config):
        super(AkiEnv, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X_val = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "val_X.csv"))
        self.y_val = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "val_y.csv"))
        
        # This will now correctly confirm 43 features
        self.config.NUM_FEATURES = self.X_val.shape[1]
        self.num_patients = len(self.X_val)
        
        self.gain_generator = Generator(input_dim=self.config.NUM_FEATURES).to(self.device).eval()
        self.gain_generator.load_state_dict(torch.load(config.GAIN_GENERATOR_PATH, map_location=self.device))
        self.classifier = PreliminaryClassifier(input_dim=self.config.NUM_FEATURES, output_dim=1).to(self.device).eval()
        self.classifier.load_state_dict(torch.load(config.PRELIM_CLASSIFIER_PATH, map_location=self.device))
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.NUM_FEATURES * 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.config.ACTION_DIM)
        self.DIAGNOSE_ACTION = self.config.NUM_FEATURES

    def action_masks(self) -> np.ndarray:
        mask = np.ones(self.config.ACTION_DIM, dtype=np.int8)
        if self.step_count < self.config.MIN_TESTS_BEFORE_DIAGNOSIS:
            mask[self.DIAGNOSE_ACTION] = 0
        mask[:self.config.NUM_FEATURES] = 1 - self.observation_mask.cpu().numpy()
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.patient_idx = np.random.randint(self.num_patients)
        self.full_patient_data = torch.tensor(self.X_val.iloc[self.patient_idx].values, dtype=torch.float32).to(self.device)
        self.true_label = self.y_val.iloc[self.patient_idx].values[0]
        self.current_state = torch.zeros(self.config.NUM_FEATURES, device=self.device)
        self.observation_mask = torch.zeros(self.config.NUM_FEATURES, device=self.device)
        self.step_count = 0
        return torch.cat([self.current_state, self.observation_mask]).cpu().numpy(), {}

    def step(self, action):
        done = False; reward = 0.0
        if action == self.DIAGNOSE_ACTION:
            with torch.no_grad():
                imputed_state = self.gain_generator(self.current_state.unsqueeze(0), self.observation_mask.unsqueeze(0)).squeeze(0)
                logits = self.classifier(imputed_state.unsqueeze(0))
                prediction = torch.round(torch.sigmoid(logits)).item()
            reward = self.config.REWARD_CORRECT_DIAGNOSIS if prediction == self.true_label else self.config.REWARD_WRONG_DIAGNOSIS
            done = True
        else:
            self.observation_mask[action] = 1.0
            self.current_state[action] = self.full_patient_data[action]
            reward += self.config.COST_PER_TEST
            with torch.no_grad():
                imputed_state = self.gain_generator(self.current_state.unsqueeze(0), self.observation_mask.unsqueeze(0)).squeeze(0)
                logits = self.classifier(imputed_state.unsqueeze(0))
                prob_1 = torch.sigmoid(logits); prob_0 = 1 - prob_1
                probabilities = torch.cat([prob_0, prob_1], dim=1)
                shannon_entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-8))
                reward += (1.0 - shannon_entropy.item()) * self.config.UNCERTAINTY_REWARD_FACTOR
        self.step_count += 1
        if self.step_count >= self.config.NUM_FEATURES: done = True
        return torch.cat([self.current_state, self.observation_mask]).cpu().numpy(), reward, done, False, {}

# --- Custom Transformer Policy ---
class TransformerPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int):
        super(TransformerPolicy, self).__init__(observation_space, features_dim)
        self.transformer_selector = TransformerSelector(
            input_dim=observation_space.shape[0], embed_dim=features_dim,
            num_heads=4, ff_dim=128, dropout=0.1
        )
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.transformer_selector(observations)

# --- Main Execution ---
if __name__ == '__main__':
    print("--- Starting RL Agent Training for AKI (v7.3 - Final Config) ---")
    config = Config()
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    env = AkiEnv(config=config)
    env = Monitor(env, config.LOG_DIR)
    
    policy_kwargs = dict(features_extractor_class=TransformerPolicy, features_extractor_kwargs=dict(features_dim=128))

    agent = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
                        tensorboard_log=config.LOG_DIR, device='cpu')

    print("\nTraining a dynamic, single-test agent...")
    agent.learn(total_timesteps=config.TOTAL_TIMESTEPS, progress_bar=True)
    
    final_model_path = os.path.join(config.MODEL_DIR, "rl_agent_aki_single_test_final.zip")
    agent.save(final_model_path)
    print(f"\n✅ Final AKI agent saved to: {final_model_path}")