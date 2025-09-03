# src/training/train_rl_agent_aki.py
# v2.3 — Windows-safe, faster: TF32, threads, inference_mode, optional torch.compile,
#        GPU env (GAIN+clf), preloaded tensors, checkpoint cleanup, YAML-driven knobs.

import os, sys, time, json, yaml, argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gymnasium as gym

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

# --- project path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.gain import Generator
from src.models.classifier import PreliminaryClassifier
from src.models.transformer import TransformerSelector


# ----------------------------- config loader -----------------------------
@dataclass
class DotCfg:
    _store: Dict[str, Any] = field(default_factory=dict)
    def __getattr__(self, k):
        v = self._store[k]
        return DotCfg(v) if isinstance(v, dict) else v
    def get(self, k, default=None): return self._store.get(k, default)
    def set(self, k, v): self._store[k] = v

def load_cfg(path: str) -> DotCfg:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return DotCfg(raw if isinstance(raw, dict) else {})


# ----------------------------- speedups ----------------------------------
def apply_speedups():
    # Torch thread budget (prevents oversubscription on Windows)
    try:
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    except Exception:
        pass
    # TF32 (no semantic change for PPO; small numeric drift is fine)
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Torch 2.x hint for GEMM precision
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass


# ------------------------ transformer extractor --------------------------
class TransformerPolicyExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        in_dim = int(np.prod(observation_space.shape))
        self.tx = TransformerSelector(input_dim=in_dim, embed_dim=features_dim,
                                      num_heads=4, ff_dim=128, dropout=0.1)
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.tx(obs)


# --------------------------------- Env -----------------------------------
class AkiEnv(gym.Env):
    """
    Grouped actions from YAML + diagnose action.
    Obs = [state, mask] of length 2*num_features (float32).
    Uses GAIN + classifier during interaction.
    """
    metadata = {"render_modes": []}

    def __init__(self, cfg: DotCfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cpu"))  # <- set "cuda" in YAML for speed

        ddir = cfg.get("processed_data_dir", "data/processed/aki")
        # Read as float32 to avoid dtype conversions later
        self.X_train = pd.read_csv(os.path.join(ddir, "train_X.csv"), dtype=np.float32)
        self.y_train = pd.read_csv(os.path.join(ddir, "train_y.csv"))

        self.num_features = self.X_train.shape[1]

        # Preload to tensors on the target device (big speedup at reset)
        self.X_tensor = torch.tensor(self.X_train.values, dtype=torch.float32, device=self.device)
        self.y_tensor = torch.tensor(self.y_train.values.squeeze(), dtype=torch.long, device=self.device)

        # Grouped actions (fallback = per-feature)
        self.feature_groups: Dict[int, List[int]] = {int(k): list(v) for k, v in cfg.get("feature_groups", {}).items()}
        if not self.feature_groups:
            self.feature_groups = {i: [i] for i in range(self.num_features)}
        self.num_groups = len(self.feature_groups)

        # Costs/repeats
        self.cost_mapping: Dict[str, float] = {str(k): float(v) for k, v in cfg.get("cost_mapping", {}).items()}
        self.block_repeats = bool(cfg.get("block_repeats", True))
        self.repeat_penalty = float(cfg.get("repeat_penalty", -200.0))

        # Shaping/economics
        self.info_cost_tradeoff = float(cfg.get("info_cost_tradeoff", 0.25))
        self.info_gain_reward   = float(cfg.get("info_gain_reward", 5.0))
        self.step_penalty       = float(cfg.get("step_penalty", -0.5))
        self.min_tests_before_dx = int(cfg.get("min_tests_before_diagnosis", 3))
        self.entropy_every_k    = int(cfg.get("entropy_every_k", 1))  # 1 = every step (exact)

        # Terminal rewards
        self.R_TP = float(cfg.get("reward_true_positive", 8000.0))
        self.R_TN = float(cfg.get("reward_true_negative", 2000.0))
        self.R_FP = float(cfg.get("penalty_false_positive", -6500.0))
        self.R_FN = float(cfg.get("penalty_false_negative", -16000.0))

        # Action/obs spaces
        self.DIAGNOSE = self.num_groups
        self.action_space = gym.spaces.Discrete(self.num_groups + 1)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_features * 2,), dtype=np.float32
        )

        # Models
        self.G = Generator(input_dim=self.num_features).to(self.device).eval()
        self.G.load_state_dict(torch.load(cfg.get("gain_generator_path", "models/generator_aki.pth"),
                                          map_location=self.device))
        self.clf = PreliminaryClassifier(input_dim=self.num_features, output_dim=1).to(self.device).eval()
        self.clf.load_state_dict(torch.load(cfg.get("prelim_classifier_path", "models/classifier_aki.pth"),
                                            map_location=self.device))

        # Optional compile (PyTorch 2.x)
        if bool(cfg.get("compile_torch", False)) and hasattr(torch, "compile"):
            try:
                self.G = torch.compile(self.G, mode="reduce-overhead")
                self.clf = torch.compile(self.clf, mode="reduce-overhead")
            except Exception:
                pass

        # Episode state
        self._rng = np.random.default_rng(int(cfg.get("seed", 108)))
        self._last_entropy = 1.0
        self._entropy_tick = 0

    # --- helpers
    def _obs(self):
        return torch.cat([self.current_state, self.observation_mask], dim=0).detach().cpu().numpy().astype(np.float32)

    def action_masks(self) -> np.ndarray:
        mask = np.ones(self.num_groups + 1, dtype=np.int8)
        # mask fully revealed groups
        for a, idxs in self.feature_groups.items():
            if self.observation_mask[idxs].sum().item() >= len(idxs):
                mask[a] = 0
        if self.step_count < self.min_tests_before_dx:
            mask[self.DIAGNOSE] = 0
        return mask

    # --- gym API
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        i = int(self._rng.integers(0, self.X_tensor.shape[0]))
        self.full_patient = self.X_tensor[i]   # already on device
        self.true_label = int(self.y_tensor[i].item())

        self.current_state = torch.zeros(self.num_features, device=self.device)
        self.observation_mask = torch.zeros(self.num_features, device=self.device)
        self.step_count = 0
        self._last_entropy = 1.0
        self._entropy_tick = 0
        return self._obs(), {}

    def step(self, action: int):
        done = False
        reward = 0.0

        if action == self.DIAGNOSE:
            with torch.inference_mode():
                xhat = self.G(self.current_state.unsqueeze(0), self.observation_mask.unsqueeze(0)).squeeze(0)
                logit = self.clf(xhat.unsqueeze(0))
                pred = torch.round(torch.sigmoid(logit)).item()
            reward += self.R_TP if (pred == 1 and self.true_label == 1) else 0.0
            reward += self.R_TN if (pred == 0 and self.true_label == 0) else 0.0
            reward += self.R_FN if (pred == 0 and self.true_label == 1) else 0.0
            reward += self.R_FP if (pred == 1 and self.true_label == 0) else 0.0
            done = True
        else:
            idxs = self.feature_groups[int(action)]
            already = self.observation_mask[idxs].sum().item()
            if already >= len(idxs):
                reward += self.repeat_penalty if self.block_repeats else 0.0
            else:
                self.current_state[idxs] = self.full_patient[idxs]
                self.observation_mask[idxs] = 1.0
                # cost
                cost = float(self.cost_mapping.get(str(int(action)), 0.0))
                reward -= self.info_cost_tradeoff * cost

                # entropy drop (optionally sub-sampled)
                self._entropy_tick += 1
                if self._entropy_tick % max(1, int(self.entropy_every_k)) == 0:
                    with torch.inference_mode():
                        xhat = self.G(self.current_state.unsqueeze(0), self.observation_mask.unsqueeze(0)).squeeze(0)
                        p1 = torch.sigmoid(self.clf(xhat.unsqueeze(0))).clamp(1e-8, 1 - 1e-8)
                        p0 = 1 - p1
                        ent = - (p0 * torch.log2(p0) + p1 * torch.log2(p1)).item()
                    entropy_drop = max(0.0, self._last_entropy - ent)
                    self._last_entropy = ent
                    reward += self.info_gain_reward * entropy_drop

            reward += self.step_penalty

        self.step_count += 1
        if self.step_count >= (self.num_groups + 5):
            done = True

        return self._obs(), float(reward), done, False, {}

    def render(self): return None


# ---------------------------- checkpointing ------------------------------
class StepCheckpoint(BaseCallback):
    def __init__(self, save_dir: str, every_steps: int, final_steps: int, verbose: int = 1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.every_steps = int(every_steps)
        self.final_steps = int(final_steps)
        os.makedirs(self.save_dir, exist_ok=True)
        self._saved: List[str] = []

    def _on_step(self) -> bool:
        steps = self.num_timesteps
        if self.every_steps > 0 and steps > 0 and (steps % self.every_steps == 0):
            path = os.path.join(self.save_dir, f"rl_agent_aki_ckpt_{steps}.zip")
            self.model.save(path)
            # keep only newest
            for old in self._saved[:-1]:
                try: os.remove(old)
                except Exception: pass
            self._saved = [path]
            if self.verbose:
                print(f"[ckpt] saved {path}")
        return True

    def _on_training_end(self) -> None:
        if self.num_timesteps >= self.final_steps:
            for p in self._saved:
                try: os.remove(p)
                except Exception: pass
            if self.verbose and self._saved:
                print("[ckpt] final reached; deleted intermediate checkpoints.")


# --------------------------------- main ----------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/aki_config_accfocus.yaml")
    p.add_argument("--total_timesteps", type=int, default=None)
    p.add_argument("--n_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--clip_range", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--ent_coef", type=float, default=None)
    p.add_argument("--vf_coef", type=float, default=None)
    p.add_argument("--target_kl", type=float, default=None)
    p.add_argument("--checkpoint_every", type=int, default=None)
    p.add_argument("--device", type=str, default=None)  # "cpu" or "cuda" for the ENV
    p.add_argument("--compile_torch", action="store_true")
    return p.parse_args()


def main():
    apply_speedups()

    args = parse_args()
    cfg = load_cfg(args.config)

    # Merge CLI overrides when provided
    def maybe(k, v):
        if v is not None:
            cfg.set(k, v)
    for k in ["total_timesteps","n_steps","batch_size","learning_rate","clip_range",
              "gamma","ent_coef","vf_coef","target_kl","checkpoint_every","device"]:
        maybe(k, getattr(args, k))
    if args.compile_torch:
        cfg.set("compile_torch", True)

    model_dir = cfg.get("model_dir", "models"); os.makedirs(model_dir, exist_ok=True)
    log_dir   = cfg.get("log_dir",   "logs_aki_final"); os.makedirs(log_dir, exist_ok=True)

    seed = int(cfg.get("seed", 108))
    set_random_seed(seed)

    # Single-env DummyVecEnv (Windows-safe)
    def make():
        return Monitor(AkiEnv(cfg))
    vec = DummyVecEnv([make])
    vec = VecMonitor(vec)

    # Optional observation/reward normalization for stability
    norm_obs     = bool(cfg.get("normalize_obs", True))
    norm_reward  = bool(cfg.get("normalize_reward", True))
    clip_obs     = float(cfg.get("clip_obs", 10.0))
    clip_reward  = float(cfg.get("clip_reward", 10.0))
    norm_gamma   = float(cfg.get("norm_gamma", 0.99))

    if norm_obs or norm_reward:
        vec = VecNormalize(
            vec,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=clip_obs,
            clip_reward=clip_reward,
            gamma=norm_gamma,
        )

    # Policy extractor
    policy_kwargs = dict(
        features_extractor_class=TransformerPolicyExtractor,
        features_extractor_kwargs=dict(features_dim=int(cfg.get("features_dim", 128))),
        net_arch=[128, 128],
    )

    # PPO hparams
    gamma      = float(cfg.get("gamma", 0.99))
    n_steps    = int(cfg.get("n_steps", 2048))
    batch_size = int(cfg.get("batch_size", 256))
    lr         = float(cfg.get("learning_rate", 2.5e-4))
    ent_coef   = float(cfg.get("ent_coef", 0.01))
    vf_coef    = float(cfg.get("vf_coef", 0.5))
    clip       = float(cfg.get("clip_range", 0.2))
    target_kl  = float(cfg.get("target_kl", 0.03))
    total_ts   = int(cfg.get("total_timesteps", 300_000))
    ckpt_every = int(cfg.get("checkpoint_every", 200_000))

    print(f"\n[AKI-RL] policy_device=cpu "
          f"| env_device={cfg.get('device','cpu')} | steps={total_ts:,}")
    print(f"PPO: gamma={gamma} n_steps={n_steps} batch={batch_size} lr={lr} clip={clip} target_kl={target_kl}")

    agent = MaskablePPO(
        policy="MlpPolicy",
        env=vec,
        tensorboard_log=log_dir,
        verbose=1,
        seed=seed,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=lr,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        clip_range=clip,
        target_kl=target_kl,
        device="cpu",
        policy_kwargs=policy_kwargs,
    )

    ckpt_cb = StepCheckpoint(save_dir=model_dir, every_steps=ckpt_every, final_steps=total_ts, verbose=1)

    try:
        agent.learn(total_timesteps=total_ts, progress_bar=True, callback=ckpt_cb)
    except KeyboardInterrupt:
        print("\n[AKI-RL] Interrupted. Latest checkpoint kept for resume.")

    final_path = os.path.join(model_dir, "rl_agent_aki_final.zip")
    agent.save(final_path)
    # Save VecNormalize statistics if used
    try:
        if isinstance(vec, VecNormalize):
            vec.save(os.path.join(model_dir, "vecnormalize_aki.pkl"))
    except Exception:
        pass
    print(f"\n✅ Final AKI agent saved to: {final_path}")

    vec.close()


if __name__ == "__main__":
    main()
