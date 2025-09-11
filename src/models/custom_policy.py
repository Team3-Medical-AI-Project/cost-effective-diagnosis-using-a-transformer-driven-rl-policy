# src/models/custom_policy.py

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.models.transformer import TransformerSelector

class TransformerPolicyExtractor(BaseFeaturesExtractor):
    """
    Wraps your TransformerSelector so SB3 can use it as features_extractor.
    Adds hard NaN/Inf guards and magnitude clamping at the input and output,
    so the policy never sees non-finite activations.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        input_dim = int(np.prod(observation_space.shape))
        self.selector = TransformerSelector(
            input_dim=input_dim,
            embed_dim=features_dim,
            num_heads=4,
            ff_dim=128,
            dropout=0.10,
        )
        self.post_norm = nn.LayerNorm(features_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # 1) sanitize incoming obs
        if not torch.isfinite(obs).all():
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        # 2) clamp magnitude; prevents extreme inputs from blowing up attention
        obs = torch.clamp(obs, -10.0, 10.0)

        out = self.selector(obs)

        # 3) sanitize and clamp features too
        if not torch.isfinite(out).all():
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = torch.clamp(out, -10.0, 10.0)

        # 4) light normalization to keep scales stable
        out = self.post_norm(out)
        return out