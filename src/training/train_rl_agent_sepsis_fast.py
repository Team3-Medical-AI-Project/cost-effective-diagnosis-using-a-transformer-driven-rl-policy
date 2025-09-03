# src/training/train_rl_agent_sepsis_fast.py
# v1.4 — fast, Windows-safe, CUDA policy, NaN guards (callback + grad hook), semantics preserved

import os
import sys
import json
import yaml
import time
import argparse
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable

import numpy as np
import gymnasium as gym
import torch

# ---- Safe speedups (no change to learning semantics)
from src.tools.ppo_speedups import (
    set_torch_runtime_threads,
    enable_tf32_if_available,
    pick_policy_device,
    make_vec_env,
)

# --- Path setup so "src.*" imports work whether launched from project root or elsewhere
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Models / env
from src.models.transformer import TransformerSelector
from src.training.sepsis_env_fast import SepsisEnvFast


from src.models.custom_policy import TransformerPolicyExtractor

# --- SB3 / wrappers
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback


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
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return DotConfig(raw if isinstance(raw, dict) else {})



import torch.nn as nn


from stable_baselines3.common.callbacks import BaseCallback

class NaNGuardCallback(BaseCallback):
    """
    Defensive checks for non-finite values during training.
    - Validates rollout buffer (obs/actions/returns/advantages) at rollout end.
    - Implements _on_step() (required by BaseCallback), returns True to continue.
    """

    def __init__(self, check_every: int = 1, verbose: int = 1):
        super().__init__(verbose)
        self.check_every = int(check_every)
        self._step_count = 0

    def _on_training_start(self) -> None:
        if self.verbose:
            print("[NaNGuardCallback] Enabled.")

    def _on_step(self) -> bool:
        # Must return True to keep training; we don’t do per-step checks here for speed.
        self._step_count += 1
        return True

    def _on_rollout_end(self) -> None:
        # Called after each rollout (n_steps * n_envs collected)
        try:
            rb = self.model.rollout_buffer
        except AttributeError:
            # Older/newer SB3 variants might store buffers differently; bail out silently.
            return

        to_check = {
            "observations": getattr(rb, "observations", None),
            "actions": getattr(rb, "actions", None),
            "returns": getattr(rb, "returns", None),
            "advantages": getattr(rb, "advantages", None),
            "values": getattr(rb, "values", None),
            "log_probs": getattr(rb, "log_probs", None),
        }

        for name, arr in to_check.items():
            if arr is None:
                continue
            if isinstance(arr, torch.Tensor):
                if not torch.isfinite(arr).all():
                    bad = (~torch.isfinite(arr)).nonzero(as_tuple=False)
                    where = bad[0].tolist() if bad.numel() > 0 else []
                    print(f"[NaNGuardCallback] Non-finite in {name} (tensor) at index {where}")
                    raise RuntimeError(f"NaN/Inf detected in rollout buffer: {name}")
            else:
                arr_np = np.asarray(arr)
                if not np.isfinite(arr_np).all():
                    where = np.where(~np.isfinite(arr_np))
                    print(f"[NaNGuardCallback] Non-finite in {name} (array) at indices {where}")
                    raise RuntimeError(f"NaN/Inf detected in rollout buffer: {name}")



# =========================================================
# Env factory (handles masking + monitor)
# =========================================================
def _mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()

def make_env(cfg: DotConfig, seed: int, rank: int, config_path_for_subproc: Optional[str] = None) -> Callable[[], gym.Env]:
    """
    For SubprocVecEnv: re-load config in child to avoid pickling issues.
    For DummyVecEnv: closure also works; we still reuse same code path.
    """
    def _init():
        local_cfg = cfg
        if config_path_for_subproc is not None:
            local_cfg = load_config(config_path_for_subproc)

        # Defaults / toggles the env expects (safe if ignored)
        local_cfg._store.setdefault("use_two_diagnosis_actions", True)
        local_cfg._store.setdefault("min_tests_before_diagnosis", max(1, int(local_cfg.get("min_tests_before_diagnosis", 2))))
        local_cfg._store.setdefault("uncertainty_factor", float(local_cfg.get("uncertainty_factor", 0.5)))
        local_cfg._store.setdefault("reward_true_positive", float(local_cfg.get("reward_true_positive", 10000.0)))
        local_cfg._store.setdefault("reward_true_negative", float(local_cfg.get("reward_true_negative", 1000.0)))
        local_cfg._store.setdefault("penalty_false_positive", float(local_cfg.get("penalty_false_positive", -5000.0)))
        local_cfg._store.setdefault("penalty_false_negative", float(local_cfg.get("penalty_false_negative", -20000.0)))
        local_cfg._store.setdefault("decision_threshold", float(local_cfg.get("decision_threshold", 0.5)))
        local_cfg._store.setdefault("class_weight_expired", float(local_cfg.get("class_weight_expired", 3.0)))

        # Ablations defaults
        for key in [
            "ablation_no_gain", "ablation_no_transformer", "ablation_no_masking",
            "ablation_no_entropy", "ablation_symmetric_reward",
            "baseline_order_all", "baseline_order_nothing"
        ]:
            local_cfg._store.setdefault(key, False)

        env = SepsisEnvFast(local_cfg)
        env = Monitor(env)
        if hasattr(env, "action_masks"):
            env = ActionMasker(env, _mask_fn)

        # Seeding
        try:
            env.reset(seed=seed + rank)  # Gymnasium
        except TypeError:
            try:
                env.seed(seed + rank)    # Old gym
            except Exception:
                pass
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
    with open(os.path.join(out_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def pick_policy_kwargs(cfg: DotConfig) -> Dict[str, Any]:
    if cfg.get("ablation_no_transformer", False):
        return {}
    else:
        return dict(
            features_extractor_class=TransformerPolicyExtractor,
            features_extractor_kwargs=dict(features_dim=int(cfg.get("features_dim", 128))),
            net_arch=[128, 128],
        )

def safe_vec_env(cfg: DotConfig, seed: int, n_envs: int, config_path: str):
    """
    Windows: ALWAYS DummyVecEnv to avoid spawn overhead and disk-contention.
    Linux/Mac: SubprocVecEnv for n_envs > 1, else DummyVecEnv.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    env_fns = [make_env(cfg, seed, i, config_path_for_subproc=None) for i in range(max(1, n_envs))]
    system = platform.system().lower()

    # Force DummyVecEnv on Windows
    if system == "windows":
        vec = DummyVecEnv(env_fns)
    else:
        if n_envs > 1:
            vec = SubprocVecEnv([make_env(cfg, seed, i, config_path_for_subproc=config_path) for i in range(n_envs)],
                                start_method="spawn")
        else:
            vec = DummyVecEnv(env_fns)

    return VecMonitor(vec)



# =========================================================
# CLI / Main
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(description="Train Sepsis RL agent (fast, ablations-ready).")
    p.add_argument("--config", type=str, default="configs/sepsis_config.yaml", help="Path to YAML config")
    p.add_argument("--seed", type=int, default=None, help="Override seed from config")
    p.add_argument("--device", type=str, default=None, help='Env device: "cpu" or "cuda" (policy uses CUDA automatically if available)')
    p.add_argument("--total_timesteps", type=int, default=None, help="Override total timesteps")

    # Speed / vec env
    p.add_argument("--n_envs", type=int, default=1, help="Number of parallel envs")
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--n_steps", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--ent_coef", type=float, default=None)
    p.add_argument("--vf_coef", type=float, default=None)
    p.add_argument("--clip_range", type=float, default=None)
    p.add_argument("--target_kl", type=float, default=None)

    # Rewards / shaping / bias fixes
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
    # --- safe, semantics-preserving speedups ---
    set_torch_runtime_threads()
    enable_tf32_if_available()

    args = parse_args()
    cfg = load_config(args.config)

    # helper: only override from CLI if the flag actually appeared on the command line
    def _flag_provided(flag: str) -> bool:
        return any(a == flag or a.startswith(flag + "=") for a in sys.argv)

    def _maybe_cli_set(key: str, flag: str, value):
        if _flag_provided(flag) and value is not None:
            cfg._store[key] = value

    # Seed
    if args.seed is not None:
        cfg._store["seed"] = int(args.seed)
    seed = int(cfg.get("seed", 42))
    set_random_seed(seed)

    # Env device (policy device picked separately)
    if args.device is not None:
        cfg._store["device"] = args.device
    env_device = cfg.get("device", "cpu")

    # Timesteps
    if args.total_timesteps is not None:
        cfg._store["total_timesteps"] = int(args.total_timesteps)
    total_timesteps = int(cfg.get("total_timesteps", 500_000))

    # Rewards / shaping (CLI only if provided)
    _maybe_cli_set("uncertainty_factor", "--uncertainty_factor", args.uncertainty_factor)
    _maybe_cli_set("penalty_false_negative", "--penalty_false_negative", args.penalty_false_negative)
    _maybe_cli_set("penalty_false_positive", "--penalty_false_positive", args.penalty_false_positive)
    _maybe_cli_set("reward_true_positive", "--reward_true_positive", args.reward_true_positive)
    _maybe_cli_set("reward_true_negative", "--reward_true_negative", args.reward_true_negative)
    _maybe_cli_set("decision_threshold", "--decision_threshold", args.decision_threshold)
    cfg._store["use_two_diagnosis_actions"] = bool(args.use_two_diagnosis_actions)

    # PPO core hparams (prefer YAML unless the exact flag is present)
    _maybe_cli_set("gamma", "--gamma", args.gamma)
    _maybe_cli_set("n_steps", "--n_steps", args.n_steps)
    _maybe_cli_set("batch_size", "--batch_size", args.batch_size)
    _maybe_cli_set("learning_rate", "--learning_rate", args.learning_rate)
    _maybe_cli_set("ent_coef", "--ent_coef", args.ent_coef)
    _maybe_cli_set("vf_coef", "--vf_coef", args.vf_coef)
    _maybe_cli_set("clip_range", "--clip_range", args.clip_range)
    _maybe_cli_set("target_kl", "--target_kl", args.target_kl)

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
    if args.model_name:
        model_name = args.model_name
    else:
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
    print(f"\n---> This run will save the final model to: {save_path}\n")

    # Build vec env (respects n_envs on Windows)
    env = safe_vec_env(cfg, seed, int(args.n_envs), args.config)

    # Policy kwargs (transformer vs ablated)
    policy_kwargs = pick_policy_kwargs(cfg)

    # PPO policy device: allow override via config (ppo_device: "cpu"|"auto")
    ppo_dev_cfg = str(cfg.get("ppo_device", "auto")).lower()
    if ppo_dev_cfg == "cpu":
        policy_device = "cpu"
    else:
        # default: auto-pick CUDA if available
        policy_device = pick_policy_device()

    # Effective PPO hparams after YAML+CLI merge
    eff_gamma       = float(cfg.get("gamma", 0.995))
    eff_n_steps     = int(cfg.get("n_steps", 2048))
    eff_batch_size  = int(cfg.get("batch_size", 256))
    eff_lr          = float(cfg.get("learning_rate", 3e-4))
    eff_ent_coef    = float(cfg.get("ent_coef", 0.01))
    eff_vf_coef     = float(cfg.get("vf_coef", 0.5))
    eff_clip        = float(cfg.get("clip_range", 0.2))
    eff_target_kl   = float(cfg.get("target_kl", 0.03))

    agent = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        device=policy_device,
        seed=seed,
        gamma=eff_gamma,
        n_steps=eff_n_steps,
        batch_size=eff_batch_size,
        learning_rate=eff_lr,
        ent_coef=eff_ent_coef,
        vf_coef=eff_vf_coef,
        clip_range=eff_clip,
        target_kl=eff_target_kl,
        policy_kwargs=policy_kwargs,
    )

    # ---- Gradient NaN hook (prints immediately if any grad becomes NaN)
    def _nan_grad_hook(grad):
        if torch.isnan(grad).any():
            print("[NaNGuard] Gradient has NaN!")
        return grad

    for p in agent.policy.parameters():
        if p.requires_grad:
            p.register_hook(_nan_grad_hook)

    # ---- Rollout NaN guard callback
    nan_cb = NaNGuardCallback()

    print("\n=== Training Sepsis RL Agent (FAST) ===")
    print(f"Env device: {env_device} | PPO policy device: {policy_device}")
    print(
        f"Seed: {seed} | Total steps: {total_timesteps:,} | n_envs: {args.n_envs} | "
        f"n_steps: {eff_n_steps} | batch_size: {eff_batch_size}"
    )
    print(
        f"PPO: gamma={eff_gamma} lr={eff_lr} ent_coef={eff_ent_coef} vf_coef={eff_vf_coef} "
        f"clip_range={eff_clip} target_kl={eff_target_kl}"
    )
    print(
        f"Ablations: no_gain={cfg.get('ablation_no_gain', False)}, no_transformer={cfg.get('ablation_no_transformer', False)}, "
        f"no_masking={cfg.get('ablation_no_masking', False)}, no_entropy={cfg.get('ablation_no_entropy', False)}, "
        f"symmetric_reward={cfg.get('ablation_symmetric_reward', False)}"
    )
    print(
        f"Baselines: order_all={cfg.get('baseline_order_all', False)}, order_nothing={cfg.get('baseline_order_nothing', False)}"
    )
    print(
        f"Bias fix: two_diagnosis_actions={cfg.get('use_two_diagnosis_actions', True)}, "
        f"penalty_FN={cfg.get('penalty_false_negative', -20000.0)}, thr={cfg.get('decision_threshold', 0.5)}"
    )

    # ----------------- HERE is .learn(...) -----------------
    try:
        agent.learn(total_timesteps=total_timesteps, progress_bar=True, callback=nan_cb)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. The model will now be saved.")
    # -------------------------------------------------------

    # Save model with descriptive name
    # REPLACE WITH THIS
    agent.save(save_path)
    print(f"\n✅ Saved agent to: {save_path}")

    env.close()



if __name__ == "__main__":
    main()
