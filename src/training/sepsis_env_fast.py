"""
SepsisEnvFast — full-feature, optimized environment
---------------------------------------------------
- Reads all hyperparameters from YAML (no hard-coded knobs).
- Asymmetric final rewards + shaping (uncertainty + progress-toward-true).
- Action masking: blocks re-ordering same panel; blocks diagnosis until min tests.
- Two diagnosis modes:
    * "split"  : DIAG_POS and DIAG_NEG as separate terminal actions
    * "single" : one DIAG action using an explicit decision_threshold
- Speed optimizations:
    * Data loaded once, kept on device
    * No pandas in step()
    * Optional TorchScript JIT for GAIN + classifier

YAML keys used (examples):
  processed_data_dir: "data/processed/sepsis"
  model_dir: "models"
  log_dir: "logs_sepsis_final"
  gain_generator_path: "models/generator_sepsis.pth"
  prelim_classifier_path: "models/classifier_sepsis.pth"

  num_features: 37
  num_test_groups: 4
  min_tests_before_diagnosis: 2
  diagnose_mode: "split"   # or "single"
  decision_threshold: 0.45 # only for single mode
  device: "cuda"
  seed: 42

  feature_groups:          # optional; else fallback to 10/10/10/rest or even split
    0: [0,1,2,3,4,5,6,7,8,9]
    1: [10,11,12,13,14,15,16,17,18,19]
    2: [20,21,22,23,24,25]
    3: [26,27,28,29,30,31,32,33,34,35]

  cost_mapping:            # string or int keys are fine
    "0": 30.86
    "1": 67.73
    "2": 509.0
    "3": 49.0

  # Rewards
  reward_true_positive: 10000.0
  reward_true_negative: 1000.0
  penalty_false_positive: -5000.0
  penalty_false_negative: -20000.0

  # Shaping
  uncertainty_factor: 0.5
  progress_toward_true_factor: 1.0

  # Practical nudges (optional; default 0 → off)
  first_k_cost_discount: {k: 0, factor: 1.0}   # e.g., {k: 2, factor: 0.25}
  info_cost_tradeoff: 0.0                       # e.g., 0.25
  jit: false
"""

from __future__ import annotations
import os, math
from typing import Any, Dict, List

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces


# -------------------------
# Robust Config wrapper
# -------------------------
class Config:
    """
    Hybrid attribute/dict access with .get() and .to_dict().
    Only string keys are set as attributes to avoid issues with numeric keys.
    """
    def __init__(self, data: Dict[Any, Any]):
        self._d: Dict[Any, Any] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                v = Config(v)
            self._d[k] = v
            if isinstance(k, str):
                try:
                    setattr(self, k, v)
                except Exception:
                    pass

    def get(self, key: Any, default: Any = None) -> Any:
        if isinstance(key, str) and hasattr(self, key):
            return getattr(self, key)
        return self._d.get(key, default)

    def to_dict(self) -> Dict[Any, Any]:
        out = {}
        for k, v in self._d.items():
            out[k] = v.to_dict() if isinstance(v, Config) else v
        return out


# -------------------------
# Environment
# -------------------------
class SepsisEnvFast(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(self.cfg.get("device", "cpu"))
        self.rng = np.random.default_rng(int(self.cfg.get("seed", 42)))

        # === Data (load ONCE, keep on device) ===
        proc = self.cfg.get("processed_data_dir")
        if not proc:
            raise ValueError("processed_data_dir missing in config")

        def _load_csv(path: str) -> np.ndarray:
            # try numpy fast path (skip header row)
            try:
                return np.loadtxt(path, delimiter=",", skiprows=1)
            except Exception:
                import pandas as pd
                return pd.read_csv(path).values

        Xv = _load_csv(os.path.join(proc, "val_X.csv")).astype(np.float32)
        yv = _load_csv(os.path.join(proc, "val_y.csv")).astype(np.int64).ravel()

        self.num_features: int = int(self.cfg.get("num_features"))
        if Xv.shape[1] != self.num_features:
            raise ValueError(f"val_X.csv has {Xv.shape[1]} cols; expected num_features={self.num_features}")

        self.X = torch.tensor(Xv, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(yv, dtype=torch.long, device=self.device)
        self.n = self.X.shape[0]

        # === Models (GAIN + classifier) ===
        from src.models.gain import Generator
        from src.models.classifier import PreliminaryClassifier

        g_path = self.cfg.get("gain_generator_path")
        c_path = self.cfg.get("prelim_classifier_path")
        if not g_path or not os.path.exists(g_path):
            raise FileNotFoundError(f"Missing gain_generator_path: {g_path}")
        if not c_path or not os.path.exists(c_path):
            raise FileNotFoundError(f"Missing prelim_classifier_path: {c_path}")

        self.gain = Generator(input_dim=self.num_features).to(self.device).eval()
        self.clf  = PreliminaryClassifier(input_dim=self.num_features, output_dim=2).to(self.device).eval()

        def _load_sd(path):
            sd = torch.load(path, map_location=self.device)
            # allow {'state_dict': ...} or plain state_dict
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            return sd

        self.gain.load_state_dict(_load_sd(g_path))
        self.clf.load_state_dict(_load_sd(c_path))

        # Optional TorchScript JIT
        if bool(self.cfg.get("jit", False)):
            with torch.no_grad():
                eg_state = torch.zeros(1, self.num_features, device=self.device)
                eg_mask  = torch.zeros(1, self.num_features, device=self.device)
                try:
                    self.gain = torch.jit.trace(self.gain, (eg_state, eg_mask))
                    imputed   = self.gain(eg_state, eg_mask)
                    self.clf  = torch.jit.trace(self.clf, imputed)
                except Exception:
                    pass  # fall back silently

        # === Rewards & shaping weights ===
        self.R_TP = float(self.cfg.get("reward_true_positive", 10000.0))
        self.R_TN = float(self.cfg.get("reward_true_negative", 1000.0))
        self.C_FP = float(self.cfg.get("penalty_false_positive", -5000.0))
        self.C_FN = float(self.cfg.get("penalty_false_negative", -20000.0))

        self.w_entropy  = float(self.cfg.get("uncertainty_factor", 0.5))
        self.w_progress = float(self.cfg.get("progress_toward_true_factor", 1.0))

        # Practical nudges (optional – encourage trying expensive informative panels)
        self.info_cost_tradeoff = float(self.cfg.get("info_cost_tradeoff", 0.0))
        _fk = self.cfg.get("first_k_cost_discount", {}) or {}
        if isinstance(_fk, Config):
            _fk = _fk.to_dict()
        self.discount_k      = int(_fk.get("k", 0))
        self.discount_factor = float(_fk.get("factor", 1.0))

        # === Costs (load BEFORE computing mean cost!) ===
        cm = self.cfg.get("cost_mapping", {}) or {}
        if isinstance(cm, Config):
            cm = cm.to_dict()
        self.cost_mapping: Dict[int, float] = {int(k): float(v) for k, v in cm.items()}

        # mean panel cost (for normalizing info/cost coupling)
        self.mean_panel_cost = float(np.mean(list(self.cost_mapping.values()) or [1.0]))

        # === Diagnosis mode / threshold ===
        self.diagnose_mode = str(self.cfg.get("diagnose_mode", "split")).lower()
        if self.diagnose_mode not in ("split", "single"):
            self.diagnose_mode = "split"
        self.decision_threshold = float(self.cfg.get("decision_threshold", 0.5))

        # === Feature groups ===
        self.num_test_groups     = int(self.cfg.get("num_test_groups", 4))
        self.min_tests_before_dx = int(self.cfg.get("min_tests_before_diagnosis", 2))

        fg = self.cfg.get("feature_groups", None)
        if isinstance(fg, Config):
            fg = fg.to_dict()
        if isinstance(fg, dict) and len(fg) >= self.num_test_groups:
            self.feature_groups = {int(k): list(map(int, v)) for k, v in fg.items()}
        else:
            # fallback: 10/10/10/rest if feasible, else even split
            if self.num_features >= 30 and self.num_test_groups == 4:
                self.feature_groups = {
                    0: list(range(0, 10)),
                    1: list(range(10, 20)),
                    2: list(range(20, 30)),
                    3: list(range(30, self.num_features)),
                }
            else:
                idx = np.arange(self.num_features)
                chunks = np.array_split(idx, self.num_test_groups)
                self.feature_groups = {i: chunks[i].astype(int).tolist() for i in range(self.num_test_groups)}

        # === Spaces ===
        obs_dim = self.num_features * 2  # values + mask
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        if self.diagnose_mode == "split":
            # 0..(num_test_groups-1) panels, num_test_groups=DIAG_POS, num_test_groups+1=DIAG_NEG
            self.DIAG_POS = self.num_test_groups
            self.DIAG_NEG = self.num_test_groups + 1
            n_actions = self.num_test_groups + 2
        else:
            # 0..(num_test_groups-1) panels, num_test_groups=DIAG
            self.DIAG = self.num_test_groups
            n_actions = self.num_test_groups + 1
        self.action_space = spaces.Discrete(n_actions)

        # === Episode state (prealloc tensors) ===
        self.curr = torch.zeros(self.num_features, device=self.device)
        self.mask = torch.zeros(self.num_features, device=self.device)
        self.full = torch.zeros(self.num_features, device=self.device)

        # obs buffer to avoid realloc
        self._obs_buf = np.zeros(obs_dim, dtype=np.float32)

        # trackers for shaping
        self._prev_p = 0.5
        self._prev_H = math.log(2.0)  # nats, Bernoulli(0.5)

        self.label = 0
        self.patient_idx = 0
        self.step_count = 0

    # -------------------------
    # MaskablePPO hook
    # -------------------------
    def action_masks(self) -> np.ndarray:
        mask = np.ones(self.action_space.n, dtype=np.int8)
        # block re-ordering panels
        for a, feats in self.feature_groups.items():
            if feats and self.mask[feats[0]].item() == 1.0:
                mask[a] = 0
        # block diagnose until min tests
        if self.step_count < self.min_tests_before_dx:
            if self.diagnose_mode == "split":
                mask[self.DIAG_POS] = 0
                mask[self.DIAG_NEG] = 0
            else:
                mask[self.DIAG] = 0
        return mask

    # -------------------------
    # Helpers
    # -------------------------
    @torch.no_grad()
    def _p_expired(self) -> float:
        imputed = self.gain(self.curr.unsqueeze(0), self.mask.unsqueeze(0)).squeeze(0)
        logits  = self.clf(imputed.unsqueeze(0))
        p = torch.softmax(logits, dim=1)[0, 1].item()
        return float(np.clip(p, 1e-6, 1 - 1e-6))

    @staticmethod
    def _entropy_nat(p: float) -> float:
        q = 1.0 - p
        return - (p * math.log(p + 1e-12) + q * math.log(q + 1e-12))

    @staticmethod
    def _logit(p: float) -> float:
        return math.log(p / (1.0 - p))

    def _pack_obs(self) -> np.ndarray:
        c = self.curr.detach().cpu().numpy()
        m = self.mask.detach().cpu().numpy()
        self._obs_buf[: self.num_features] = c
        self._obs_buf[self.num_features :] = m
        return self._obs_buf

    # -------------------------
    # Gymnasium API
    # -------------------------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        self.patient_idx = int(self.rng.integers(0, self.n))
        self.label = int(self.y[self.patient_idx].item())

        # copy row into device tensor
        self.full.copy_(self.X[self.patient_idx])
        self.curr.zero_()
        self.mask.zero_()
        self.step_count = 0

        # initialize trackers with current prior belief
        p0 = self._p_expired()
        self._prev_p = p0
        self._prev_H = self._entropy_nat(p0)

        return self._pack_obs(), {}

    def step(self, action: int):
        done = False
        reward = 0.0
        info: Dict[str, Any] = {}

        # --- Diagnose (terminal) ---
        if self.diagnose_mode == "split" and (action == self.DIAG_POS or action == self.DIAG_NEG):
            pred = 1 if action == self.DIAG_POS else 0
            if   pred == 1 and self.label == 1: reward = self.R_TP
            elif pred == 0 and self.label == 0: reward = self.R_TN
            elif pred == 1 and self.label == 0: reward = self.C_FP
            else:                                reward = self.C_FN
            info.update({"action_type": "diagnose", "pred": int(pred), "p": float(self._prev_p)})
            done = True

        elif self.diagnose_mode == "single" and action == self.DIAG:
            p = self._p_expired()
            pred = 1 if p >= self.decision_threshold else 0
            if   pred == 1 and self.label == 1: reward = self.R_TP
            elif pred == 0 and self.label == 0: reward = self.R_TN
            elif pred == 1 and self.label == 0: reward = self.C_FP
            else:                                reward = self.C_FN
            info.update({"action_type": "diagnose", "pred": int(pred), "p": float(p)})
            done = True

        # --- Order a panel ---
        else:
            feats = self.feature_groups.get(int(action), [])
            if feats:
                self.mask[feats] = 1.0
                self.curr[feats] = self.full[feats]

            # cost (optional early discount for the first k tests)
            c = float(self.cost_mapping.get(int(action), 0.0))
            if self.step_count < self.discount_k:
                c *= self.discount_factor
            reward -= c

            # shaping: ΔEntropy + ΔLogitTowardTrue
            p_prev, H_prev = self._prev_p, self._prev_H
            p_new = self._p_expired()
            H_new = self._entropy_nat(p_new)

            info_bonus = self.w_entropy * max(0.0, (H_prev - H_new))
            y_sign = 1.0 if self.label == 1 else -1.0
            progress_bonus = self.w_progress * y_sign * (self._logit(p_new) - self._logit(p_prev))

            # couple information with panel cost to let expensive informative tests be chosen
            subsidy = self.info_cost_tradeoff * info_bonus * (c / (self.mean_panel_cost + 1e-8))

            reward += info_bonus + progress_bonus + subsidy

            # update trackers
            self._prev_p = p_new
            self._prev_H = H_new

            info.update({
                "action_type": "order",
                "panel": int(action),
                "p": float(p_new),
                "action_cost": float(c),
                "info_bonus": float(info_bonus),
                "progress_bonus": float(progress_bonus),
                "subsidy": float(subsidy),
            })

        self.step_count += 1
        if self.step_count >= (self.num_test_groups + 2):  # safety cap
            done = True

        return self._pack_obs(), float(reward), bool(done), False, info


# Factory
def make_env(cfg: Config) -> SepsisEnvFast:
    return SepsisEnvFast(cfg)
