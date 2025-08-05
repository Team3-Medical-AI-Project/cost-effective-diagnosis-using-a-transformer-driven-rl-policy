"""
Main Training Script for the Reinforcement Learning Agent (v1.2 - Corrected)

Description:
This script assembles all pre-trained components into a custom RL environment.
It defines a custom network architecture using our TransformerSelector and then
trains a PPO agent to make cost-effective diagnostic decisions.

v1.2: Refactored the custom policy to a custom network architecture to fix a
      ValueError from the stable-baselines3 training loop.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

# Import our custom model architectures
from src.models.gain import Generator as GAINImputer
from src.models.transformer import TransformerSelector
from src.models.classifier import PreliminaryClassifier

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = {
    "TASK": "sepsis",
    "DATA_DIR": PROJECT_ROOT / "data" / "processed",
    "MODELS_DIR": PROJECT_ROOT / "models",
    "ID_COLUMNS": ['subject_id', 'hadm_id', 'stay_id'],
    "CLASSIFIER_OUTPUT_DIM": 2,
    "NUM_ACTIONS": 5,
    "TOTAL_TIMESTEPS": 50000,
    "PANEL_COSTS": {0: 44, 1: 48, 2: 473, 3: 26},
    "REWARD_WEIGHTS": {"ACCURACY": 1.0, "COST": 0.001, "ENTROPY": 0.5}
}

class CustomNetwork(nn.Module):
    """
    Custom network for the PPO agent, using a Transformer as the feature extractor.
    """
    def __init__(self, feature_dim: int, num_features: int):
        super(CustomNetwork, self).__init__()
        
        # The shared feature extractor is our Transformer
        self.latent_net = TransformerSelector(input_dim=feature_dim, num_actions=CONFIG["NUM_ACTIONS"])
        
        # The policy head (actor) and value head (critic) are defined by the Transformer's output layer
        # and a new linear layer for the value function.
        self.policy_net = self.latent_net.policy_head
        self.value_net = nn.Linear(self.latent_net.policy_head.in_features, 1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the latent features for the policy and value function.
        """
        # The TransformerSelector acts as the shared network body
        latent_representation = self.latent_net(features)
        # The output of the transformer's main body is used for both heads
        return self.policy_net(latent_representation), self.value_net(latent_representation)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )
        # Disable default weight initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim, CONFIG["NUM_FEATURES"])


class MedicalDiagnosisEnv(gym.Env):
    """A custom Gymnasium environment for the medical diagnosis task."""
    def __init__(self, data, gain_imputer, classifier, panel_mapping, config):
        super(MedicalDiagnosisEnv, self).__init__()
        
        self.feature_columns = [col for col in data[0].columns if col not in config["ID_COLUMNS"]]
        self.num_features = len(self.feature_columns)
        
        self.data_X = data[0][self.feature_columns].values
        self.data_y = data[1].values
        self.gain_imputer = gain_imputer
        self.classifier = classifier
        self.panel_mapping = panel_mapping
        self.config = config
        
        self.action_space = spaces.Discrete(config["NUM_ACTIONS"])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.num_features * 2 + config["CLASSIFIER_OUTPUT_DIM"],), 
            dtype=np.float32
        )
        self.current_patient_idx = -1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_patient_idx = (self.current_patient_idx + 1) % len(self.data_X)
        self.patient_features = self.data_X[self.current_patient_idx]
        self.patient_label = self.data_y[self.current_patient_idx]
        self.missing_mask = np.zeros(self.num_features)
        self.current_entropy = np.log2(self.config["CLASSIFIER_OUTPUT_DIM"])
        
        observation, _ = self._get_observation()
        info = {}
        return observation, info

    def _get_observation(self):
        corrupted_features = self.patient_features * self.missing_mask
        
        with torch.no_grad():
            corrupted_tensor = torch.from_numpy(corrupted_features).float().unsqueeze(0)
            mask_tensor = torch.from_numpy(self.missing_mask).float().unsqueeze(0)
            imputed_features_tensor = self.gain_imputer(corrupted_tensor, mask_tensor)
        
        imputed_features = imputed_features_tensor.squeeze(0).numpy()
        
        with torch.no_grad():
            imputed_for_classifier = torch.from_numpy(imputed_features).float().unsqueeze(0)
            classifier_logits = self.classifier(imputed_for_classifier)
            classifier_probs = torch.softmax(classifier_logits, dim=1).squeeze(0).numpy()
            
        observation = np.concatenate([imputed_features, classifier_probs, self.missing_mask]).astype(np.float32)
        return observation, classifier_probs

    def step(self, action):
        terminated = False
        cost_penalty = self.config["PANEL_COSTS"].get(action, 0)
        
        if action == self.config["NUM_ACTIONS"] - 1: # Diagnose action
            terminated = True
            _, classifier_probs = self._get_observation()
            predicted_class = np.argmax(classifier_probs)
            accuracy_reward = 1.0 if predicted_class == self.patient_label else -1.0
        else: # Order a test panel
            accuracy_reward = 0.0
            feature_indices = self.panel_mapping.get(action, [])
            self.missing_mask[feature_indices] = 1

        observation, classifier_probs = self._get_observation()
        new_entropy = -np.sum(classifier_probs * np.log2(classifier_probs + 1e-8))
        entropy_reward = self.current_entropy - new_entropy
        self.current_entropy = new_entropy
        
        reward = (self.config["REWARD_WEIGHTS"]["ACCURACY"] * accuracy_reward -
                  self.config["REWARD_WEIGHTS"]["COST"] * cost_penalty +
                  self.config["REWARD_WEIGHTS"]["ENTROPY"] * entropy_reward)

        info = {}
        return observation, reward, terminated, False, info

def create_panel_to_feature_mapping(feature_columns):
    """Programmatically creates the mapping from panel action to feature indices."""
    mapping = {0: [], 1: [], 2: [], 3: []}
    for i, col_name in enumerate(feature_columns):
        if 'cbc_' in col_name: mapping[0].append(i)
        elif 'cmp_' in col_name: mapping[1].append(i)
        elif 'abg_' in col_name: mapping[2].append(i)
        elif 'aptt_' in col_name: mapping[3].append(i)
    return mapping

def main():
    print("--- Initializing RL Agent Training Pipeline ---")
    
    task = CONFIG["TASK"]
    data_dir = CONFIG["DATA_DIR"] / task
    
    X_train = pd.read_csv(data_dir / "train_X.csv")
    y_train = pd.read_csv(data_dir / "train_y.csv").squeeze()
    
    feature_cols = [col for col in X_train.columns if col not in CONFIG["ID_COLUMNS"]]
    num_features = len(feature_cols)
    print(f"Detected {num_features} features for task: {task}")
    
    panel_mapping = create_panel_to_feature_mapping(feature_cols)

    gain_path = CONFIG["MODELS_DIR"] / f"generator_{task}.pth"
    gain_imputer = GAINImputer(input_dim=num_features)
    gain_imputer.load_state_dict(torch.load(gain_path))
    gain_imputer.eval()
    
    classifier = PreliminaryClassifier(input_dim=num_features)
    
    env = MedicalDiagnosisEnv(data=(X_train, y_train), gain_imputer=gain_imputer, classifier=classifier, panel_mapping=panel_mapping, config=CONFIG)
    
    # Pass the custom network architecture to the PPO agent
    policy_kwargs = {
        "features_extractor_class": CustomNetwork,
        "features_extractor_kwargs": {"num_features": num_features},
    }
    
    agent = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1)
    
    print("\n--- Starting RL Agent Training ---")
    agent.learn(total_timesteps=CONFIG["TOTAL_TIMESTEPS"])
    
    agent.save(CONFIG["MODELS_DIR"] / f"rl_agent_{task}")
    print(f"\n--- Training complete. Agent saved to '{CONFIG['MODELS_DIR'] / f'rl_agent_{task}'}' ---")

if __name__ == "__main__":
    main()
