# src/training/train_rl_agent_sepsis_fast.py
# v1.2 â€” fast, Windows-safe, ablations-ready, fixes "always discharged" via two diagnosis actions

import os
import sys
import json
import yaml
import time
import math
import argparse
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable

import numpy as np
import torch
import gymnasium as gym

# --- Path setup so "src.*" imports work whether launched from project root or elsewhere
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Models (your extensions)
from src.models.gain import Generator as GAINGenerator
from src.models.classifier import PreliminaryClassifier
from src.models.transformer import TransformerSelector

# --- Your fast environment (already updated on your side)
from src.training.sepsis_env_fast import SepsisEnvFast  # <- must exist (you confirmed)

# --- SB3 / wrappers
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed


# =========================================================
# Config loader (dot-access + dict-like .get without recursion)
# =========================================================
@dataclass
class DotConfig:
    _store: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, item):
        try:
            v = self._store[item]
        except KeyError as e:
            raise AttributeError(item) from e
        if isinstance(v, dict):
            return DotConfig(v)
        return v

    def __setattr__(self, key, value):
        if key == "_store":
            super().__setattr__(key, value)
        else:
            self._store[key] = value

    def get(self, key, default=None):
        return self._store.get(key, default)

    def to_dict(self):
        out = {}
        for k, v in self._store.items():
            if isinstance(v, DotConfig):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out


def load_config(path: str) -> DotConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return DotConfig(raw if isinstance(raw, dict) else {})


# =========================================================
# Transformer features extractor for the policy
# =========================================================
class TransformerPolicyExtractor(BaseFeaturesExtractor):
    """
    Wraps your TransformerSelector so SB3 can use it as features_extractor.
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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.selector(obs)


# =========================================================
# Env factory (handles masking + monitor); Windows-safe vec setup
# =========================================================
def _mask_fn(env: gym.Env) -> np.ndarray:
    # SepsisEnvFast should expose .action_masks()
    return env.action_masks()

def make_env(cfg: DotConfig, seed: int, rank: int, config_path_for_subproc: Optional[str] = None) -> Callable[[], gym.Env]:
    """
    For SubprocVecEnv: re-load config in child to avoid pickling issues.
    For DummyVecEnv: closure is fine on Windows.
    """
    def _init():
        local_cfg = cfg
        if config_path_for_subproc is not None:
            local_cfg = load_config(config_path_for_subproc)  # re-load in the child

        # Push ablation toggles into cfg that the env expects/uses
        # (SepsisEnvFast should read them; if not, they are simply ignored)
        # These defaults bias against "always discharged"
        local_cfg._store.setdefault("use_two_diagnosis_actions", True)  # P/N actions, not single "diagnose"
        local_cfg._store.setdefault("min_tests_before_diagnosis", max(1, int(local_cfg.get("min_tests_before_diagnosis", 2))))
        local_cfg._store.setdefault("uncertainty_factor", float(local_cfg.get("uncertainty_factor", 0.5)))
        local_cfg._store.setdefault("reward_true_positive", float(local_cfg.get("reward_true_positive", 10000.0)))
        local_cfg._store.setdefault("reward_true_negative", float(local_cfg.get("reward_true_negative", 1000.0)))
        local_cfg._store.setdefault("penalty_false_positive", float(local_cfg.get("penalty_false_positive", -5000.0)))
        local_cfg._store.setdefault("penalty_false_negative", float(local_cfg.get("penalty_false_negative", -20000.0)))
        local_cfg._store.setdefault("decision_threshold", float(local_cfg.get("decision_threshold", 0.5)))  # if env uses classifier threshold
        local_cfg._store.setdefault("class_weight_expired", float(local_cfg.get("class_weight_expired", 3.0)))  # if env uses cross-entropy weighting

        # Ablations (all default False)
        local_cfg._store.setdefault("ablation_no_gain", False)
        local_cfg._store.setdefault("ablation_no_transformer", False)
        local_cfg._store.setdefault("ablation_no_masking", False)
        local_cfg._store.setdefault("ablation_no_entropy", False)
        local_cfg._store.setdefault("ablation_symmetric_reward", False)
        local_cfg._store.setdefault("baseline_order_all", False)
        local_cfg._store.setdefault("baseline_order_nothing", False)

        env = SepsisEnvFast(local_cfg)
        env = Monitor(env)
        if hasattr(env, "action_masks"):
            env = ActionMasker(env, _mask_fn)

        try:
            env.reset(seed=seed + rank)
        except TypeError:
            pass
        try:
            env.reset(seed=seed + rank)   # Gymnasium-style seeding
        except TypeError:
            pass  # old gym fallback
        return env


    return _init


# =========================================================
# Utilities
# =========================================================
def ensure_dirs(cfg: DotConfig):
    os.makedirs(cfg.get("model_dir", "models"), exist_ok=True)
    os.makedirs(cfg.get("log_dir", "logs_sepsis_final"), exist_ok=True)

def save_run_metadata(cfg: DotConfig, args: argparse.Namespace, out_dir: str):
    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": sys.version,
        "args": vars(args),
        "cfg": cfg.to_dict(),
    }
    with open(os.path.join(out_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

def pick_policy_kwargs(cfg: DotConfig) -> Dict[str, Any]:
    """
    Use Transformer extractor unless ablated.
    """
    if cfg.get("ablation_no_transformer", False):
        # default MLP extractor (SB3) â€” nothing to pass
        return {}
    else:
        # Transformer-based feature extractor
        return dict(
            features_extractor_class=TransformerPolicyExtractor,
            features_extractor_kwargs=dict(features_dim=int(cfg.get("features_dim", 128))),
            net_arch=[128, 128],  # small head after transformer features
        )

def safe_vec_env(cfg: DotConfig, seed: int, n_envs: int, config_path: str):
    """
    Windows -> DummyVecEnv. Linux/Mac -> SubprocVecEnv if n_envs > 1.
    """
    system = platform.system().lower()
    if system == "windows" or n_envs <= 1:
        env_fns = [make_env(cfg, seed, 0, None)]
        vec = DummyVecEnv(env_fns)
    else:
        env_fns = [make_env(cfg, seed, i, config_path_for_subproc=config_path) for i in range(n_envs)]
        vec = SubprocVecEnv(env_fns, start_method="spawn")
    return VecMonitor(vec)


# =========================================================
# CLI / Main
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description="Train Sepsis RL agent (fast, ablations-ready).")
    p.add_argument("--config", type=str, default="configs/sepsis_config.yaml", help="Path to YAML config")
    p.add_argument("--seed", type=int, default=None, help="Override seed from config")
    p.add_argument("--device", type=str, default=None, help='Force device: "cpu" or "cuda"')
    p.add_argument("--total_timesteps", type=int, default=None, help="Override total timesteps")

    # Speed / vec env
    p.add_argument("--n_envs", type=int, default=1, help="Number of parallel envs (Windows uses DummyVecEnv)")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_steps", type=int, default=2048)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--target_kl", type=float, default=0.03)

    # Fix "always discharged" & uncertainty shaping
    p.add_argument("--use_two_diagnosis_actions", action="store_true", default=True)
    p.add_argument("--uncertainty_factor", type=float, default=None)
    p.add_argument("--penalty_false_negative", type=float, default=None)
    p.add_argument("--penalty_false_positive", type=float, default=None)
    p.add_argument("--reward_true_positive", type=float, default=None)
    p.add_argument("--reward_true_negative", type=float, default=None)
    p.add_argument("--decision_threshold", type=float, default=None)

    # Ablations
    p.add_argument("--ablation_no_gain", action="store_true")
    p.add_argument("--ablation_no_transformer", action="store_true")
    p.add_argument("--ablation_no_masking", action="store_true")
    p.add_argument("--ablation_no_entropy", action="store_true")
    p.add_argument("--ablation_symmetric_reward", action="store_true")
    p.add_argument("--baseline_order_all", action="store_true")
    p.add_argument("--baseline_order_nothing", action="store_true")

    # Model I/O
    p.add_argument("--model_name", type=str, default=None, help="Filename for saved .zip")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Apply explicit CLI overrides into cfg (keeps env as single source of truth)
    if args.seed is not None:
        cfg._store["seed"] = int(args.seed)
    seed = int(cfg.get("seed", 42))
    set_random_seed(seed)

    if args.device is not None:
        cfg._store["device"] = args.device
    device = cfg.get("device", "cpu")
    torch.set_num_threads(max(1, os.cpu_count() // 2 if platform.system().lower() == "windows" else os.cpu_count() // 2))

    if args.total_timesteps is not None:
        cfg._store["total_timesteps"] = int(args.total_timesteps)
    total_timesteps = int(cfg.get("total_timesteps", 500_000))

    # Fix bias: two terminal actions + strong FN penalty + (optionally) tuned threshold
    cfg._store["use_two_diagnosis_actions"] = bool(args.use_two_diagnosis_actions)
    if args.uncertainty_factor is not None:
        cfg._store["uncertainty_factor"] = float(args.uncertainty_factor)
    if args.penalty_false_negative is not None:
        cfg._store["penalty_false_negative"] = float(args.penalty_false_negative)
    if args.penalty_false_positive is not None:
        cfg._store["penalty_false_positive"] = float(args.penalty_false_positive)
    if args.reward_true_positive is not None:
        cfg._store["reward_true_positive"] = float(args.reward_true_positive)
    if args.reward_true_negative is not None:
        cfg._store["reward_true_negative"] = float(args.reward_true_negative)
    if args.decision_threshold is not None:
        cfg._store["decision_threshold"] = float(args.decision_threshold)

    # Ablations
    cfg._store["ablation_no_gain"] = bool(args.ablation_no_gain)
    cfg._store["ablation_no_transformer"] = bool(args.ablation_no_transformer)
    cfg._store["ablation_no_masking"] = bool(args.ablation_no_masking)
    cfg._store["ablation_no_entropy"] = bool(args.ablation_no_entropy)
    cfg._store["ablation_symmetric_reward"] = bool(args.ablation_symmetric_reward)
    cfg._store["baseline_order_all"] = bool(args.baseline_order_all)
    cfg._store["baseline_order_nothing"] = bool(args.baseline_order_nothing)

    ensure_dirs(cfg)
    log_dir = cfg.get("log_dir", "logs_sepsis_final")
    model_dir = cfg.get("model_dir", "models")
    save_run_metadata(cfg, args, log_dir)

    # Build vec env (Windows -> DummyVecEnv)
    env = safe_vec_env(cfg, seed, int(args.n_envs), args.config)

    # Policy kwargs (transformer vs ablated)
    policy_kwargs = pick_policy_kwargs(cfg)

    # MaskablePPO (works with ActionMasker)
    agent = MaskablePPO(
        policy="MlpPolicy",  # extractor will replace backbone if transformer enabled
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        device=device,
        seed=seed,
        gamma=float(args.gamma),
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        ent_coef=float(args.ent_coef),
        vf_coef=float(args.vf_coef),
        clip_range=float(args.clip_range),
        target_kl=float(args.target_kl),
        policy_kwargs=policy_kwargs,
    )

    print("\n=== Training Sepsis RL Agent (FAST) ===")
    print(f"Device: {device} | Seed: {seed} | Steps: {total_timesteps:,} | n_envs: {args.n_envs}")
    print(f"Ablations: no_gain={cfg.get('ablation_no_gain', False)}, no_transformer={cfg.get('ablation_no_transformer', False)}, "
          f"no_masking={cfg.get('ablation_no_masking', False)}, no_entropy={cfg.get('ablation_no_entropy', False)}, "
          f"symmetric_reward={cfg.get('ablation_symmetric_reward', False)}")
    print(f"Baselines: order_all={cfg.get('baseline_order_all', False)}, order_nothing={cfg.get('baseline_order_nothing', False)}")
    print(f"Bias fix: two_diagnosis_actions={cfg.get('use_two_diagnosis_actions', True)}, "
          f"penalty_FN={cfg.get('penalty_false_negative', -20000.0)}, thr={cfg.get('decision_threshold', 0.5)}")

    agent.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Save model with descriptive name
    if args.model_name:
        model_name = args.model_name
    else:
        # auto-name using key toggles (for bookkeeping)
        tags = []
        if cfg.get("ablation_no_gain", False): tags.append("no_gain")
        if cfg.get("ablation_no_transformer", False): tags.append("no_tx")
        if cfg.get("ablation_no_masking", False): tags.append("no_mask")
        if cfg.get("ablation_no_entropy", False): tags.append("no_ent")
        if cfg.get("ablation_symmetric_reward", False): tags.append("symR")
        if cfg.get("baseline_order_all", False): tags.append("all")
        if cfg.get("baseline_order_nothing", False): tags.append("none")
        if cfg.get("use_two_diagnosis_actions", True): tags.append("2diag")
        uf = cfg.get("uncertainty_factor", 0.5)
        model_name = f"rl_agent_sepsis_fast_uf-{uf}".replace(".", "_")
        if tags:
            model_name += "_" + "-".join(tags)
        model_name += ".zip"

    save_path = os.path.join(model_dir, model_name)
    agent.save(save_path)
    print(f"\nâœ… Saved agent to: {save_path}")

    # Clean up
    env.close()


if __name__ == "__main__":
    main()

