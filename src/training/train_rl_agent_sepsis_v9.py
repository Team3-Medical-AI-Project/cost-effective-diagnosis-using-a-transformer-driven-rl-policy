"""
Sepsis RL Agent (v9) — definitive, robust to YAML numeric keys and with explicit decision threshold.
"""

import os, sys, argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# --- project path
try:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
except NameError:
    sys.path.append('..')

from src.utils import load_config, Config
from src.models.gain import Generator
from src.models.classifier import PreliminaryClassifier
from src.models.transformer import TransformerSelector


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# -------------------------
# Env
# -------------------------
class SepsisEnv(gym.Env):
    """
    Panels: 0=CBC, 1=CMP, 2=ABG, 3=aPTT, DIAGNOSE=4
    Observation: concat( current_state, mask )  (size = 2 * num_features)
    """
    metadata = {"render_modes": []}

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(self.cfg.get("device", "cpu"))

        # Data
        proc = self.cfg.get("processed_data_dir", "data/processed/sepsis")
        self.X_val = pd.read_csv(os.path.join(proc, "val_X.csv"))
        self.y_val = pd.read_csv(os.path.join(proc, "val_y.csv"))
        self.num_patients = len(self.X_val)

        # Models (fallback if torch.load doesn't support weights_only in your torch)
        gen_path = self.cfg.get("gain_generator_path", "models/generator_sepsis.pth")
        clf_path = self.cfg.get("prelim_classifier_path", "models/classifier_sepsis.pth")

        try:
            gen_state = torch.load(gen_path, map_location=self.device, weights_only=True)
        except TypeError:
            gen_state = torch.load(gen_path, map_location=self.device)
        try:
            clf_state = torch.load(clf_path, map_location=self.device, weights_only=True)
        except TypeError:
            clf_state = torch.load(clf_path, map_location=self.device)

        self.num_features: int = int(self.cfg.get("num_features", 37))
        self.uncertainty_factor: float = float(self.cfg.get("uncertainty_factor", 0.0))
        self.min_tests_before_diagnosis: int = int(self.cfg.get("min_tests_before_diagnosis", 2))
        self.decision_threshold: float = float(self.cfg.get("decision_threshold", 0.5))  # NEW

        # Asymmetric rewards (strong FN penalty to avoid default "Discharged")
        self.r_tp = float(self.cfg.get("reward_true_positive",  10000.0))
        self.r_tn = float(self.cfg.get("reward_true_negative",  1000.0))
        self.r_fp = float(self.cfg.get("penalty_false_positive", -5000.0))
        self.r_fn = float(self.cfg.get("penalty_false_negative", -20000.0))

        # Costs (dict with int keys preserved)
        self.cost_mapping = self.cfg.get("cost_mapping", {}).to_dict() if isinstance(self.cfg.get("cost_mapping"), Config) else self.cfg.get("cost_mapping", {})
        # Ensure numeric casting
        self.cost_mapping = {int(k): float(v) for k, v in self.cost_mapping.items()}

        # Models
        self.gain = Generator(input_dim=self.num_features).to(self.device).eval()
        self.gain.load_state_dict(gen_state)

        self.clf = PreliminaryClassifier(input_dim=self.num_features, output_dim=2).to(self.device).eval()
        self.clf.load_state_dict(clf_state)

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_features * 2,), dtype=np.float32
        )
        self.num_test_groups = int(self.cfg.get("num_test_groups", 4))
        self.DIAGNOSE_ACTION = self.num_test_groups
        self.action_space = spaces.Discrete(self.num_test_groups + 1)

        # Simple feature-grouping: 0..(n-1) split into 4 contiguous blocks (last takes the remainder)
        # If you already have a precise mapping, replace here.
        base = self.num_features // self.num_test_groups
        self.feature_groups = {}
        start = 0
        for a in range(self.num_test_groups):
            end = start + base
            if a == self.num_test_groups - 1:
                end = self.num_features
            self.feature_groups[a] = list(range(start, end))
            start = end

        self.reset()

    def action_masks(self):
        mask = np.ones(self.action_space.n, dtype=np.int8)
        if self.step_count < self.min_tests_before_diagnosis:
            mask[self.DIAGNOSE_ACTION] = 0
        # disable panels already revealed
        for action, features in self.feature_groups.items():
            if len(features) == 0:
                mask[action] = 0
                continue
            if self.observation_mask[features[0]].item() == 1:
                mask[action] = 0
        return mask

    def _current_probs(self) -> float:
        with torch.no_grad():
            imputed = self.gain(self.current_state.unsqueeze(0), self.observation_mask.unsqueeze(0)).squeeze(0)
            logits = self.clf(imputed.unsqueeze(0))
            probs = torch.softmax(logits, dim=1).squeeze()
        return probs[1].item()  # P(Expired)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.patient_idx = np.random.randint(self.num_patients)
        row = self.X_val.iloc[self.patient_idx].values
        self.full_patient = torch.tensor(row, dtype=torch.float32, device=self.device)
        self.true_label = int(self.y_val.iloc[self.patient_idx].values[0])

        self.current_state = torch.zeros(self.num_features, device=self.device)
        self.observation_mask = torch.zeros(self.num_features, device=self.device)
        self.step_count = 0

        obs = torch.cat([self.current_state, self.observation_mask]).cpu().numpy()
        return obs, {}

    def step(self, action: int):
        done = False
        reward = 0.0

        if action == self.DIAGNOSE_ACTION:
            p_expired = self._current_probs()
            pred = 1 if p_expired >= self.decision_threshold else 0  # explicit threshold
            if   pred == 1 and self.true_label == 1: reward = self.r_tp
            elif pred == 0 and self.true_label == 0: reward = self.r_tn
            elif pred == 1 and self.true_label == 0: reward = self.r_fp
            else:                                    reward = self.r_fn
            done = True
        else:
            # reveal features & pay cost
            feats = self.feature_groups.get(int(action), [])
            if feats:
                self.observation_mask[feats] = 1.0
                self.current_state[feats] = self.full_patient[feats]
            reward -= float(self.cost_mapping.get(int(action), 0.0))

            # uncertainty bonus (higher when entropy lower => more confident after revealing)
            if self.uncertainty_factor != 0.0:
                with torch.no_grad():
                    imputed = self.gain(self.current_state.unsqueeze(0), self.observation_mask.unsqueeze(0)).squeeze(0)
                    logits = self.clf(imputed.unsqueeze(0))
                    probs = torch.softmax(logits, dim=1)
                    # Shannon entropy in bits
                    entropy = -torch.sum(probs * torch.log2(probs + 1e-12)).item()
                reward += (1.0 - entropy) * float(self.uncertainty_factor)

        self.step_count += 1
        if self.step_count >= self.num_test_groups + 1:
            done = True

        obs = torch.cat([self.current_state, self.observation_mask]).cpu().numpy()
        return obs, reward, done, False, {}

# -------------------------
# Transformer policy wrapper
# -------------------------
class TransformerPolicy(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int):
        super().__init__(observation_space, features_dim)
        self.selector = TransformerSelector(
            input_dim=observation_space.shape[0],
            embed_dim=features_dim,
            num_heads=4,
            ff_dim=128,
            dropout=0.1,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.selector(observations)

# -------------------------
# Train
# -------------------------
def train_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sepsis_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    os.makedirs(cfg.get("model_dir", "models"), exist_ok=True)
    os.makedirs(cfg.get("log_dir", "logs_sepsis_final"), exist_ok=True)

    env = SepsisEnv(cfg)
    env = Monitor(env, cfg.get("log_dir"))

    policy_kwargs = dict(
        features_extractor_class=TransformerPolicy,
        features_extractor_kwargs=dict(features_dim=int(cfg.get("features_dim", 128))),
    )

    agent = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=cfg.get("log_dir"),
        device=cfg.get("device", "cpu"),
        seed=int(cfg.get("seed", 42)),
    )

    print("\nTraining with asymmetric rewards and uncertainty bonus…")
    agent.learn(total_timesteps=int(cfg.get("total_timesteps", 500000)), progress_bar=True)

    out_path = os.path.join(cfg.get("model_dir"), "rl_agent_sepsis_v9_final.zip")
    agent.save(out_path)
    print(f"\n✅ Saved: {out_path}")

if __name__ == "__main__":
    train_main()
