"""
SepsisEnvFast — full-feature, optimized environment
---------------------------------------------------
[... unchanged header/comment ...]
"""

from __future__ import annotations
import os, math, time, json
from typing import Any, Dict
import numpy as np
import pandas as pd
import joblib
import torch
import gymnasium as gym
from gymnasium import spaces



# ---- Per-process model cache (avoid re-loading weights N times) ----
_GAIN_CACHE = None
_CLF_CACHE = None

def _load_models_cached(num_features: int, device: torch.device, g_path: str, c_path: str):
    """
    Load GAIN + classifier only once per process and reuse for all env instances.
    """
    global _GAIN_CACHE, _CLF_CACHE
    if _GAIN_CACHE is not None and _CLF_CACHE is not None:
        return _GAIN_CACHE.to(device), _CLF_CACHE.to(device)

    from src.models.gain import Generator
    from src.models.classifier import PreliminaryClassifier

    gain = Generator(input_dim=num_features).to(device).eval()
    clf  = PreliminaryClassifier(input_dim=num_features, output_dim=2).to(device).eval()

    def _load_sd(path):
        # Prefer weights_only when available (PyTorch 2.1+)
        try:
            sd = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(path, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        return sd

    gain.load_state_dict(_load_sd(g_path))
    clf.load_state_dict(_load_sd(c_path))

    _GAIN_CACHE, _CLF_CACHE = gain, clf
    return _GAIN_CACHE, _CLF_CACHE


# ---- Device resolver (safe "auto" handling) -------------------------------
def _resolve_device(requested: str):
    try:
        req = (requested or "").strip().lower()
    except Exception:
        req = ""

    if req in ("", "auto", "auto:cuda"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        return torch.device(requested)
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Robust Config wrapper
# -------------------------
class Config:
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
        self.device = _resolve_device(self.cfg.get("device", "auto"))
        self.rng = np.random.default_rng(int(self.cfg.get("seed", 42)))

        # === Data (load ONCE, keep on device) ===
        proc = self.cfg.get("processed_data_dir")
        if not proc:
            raise ValueError("processed_data_dir missing in config")

        def _load_csv(path: str) -> np.ndarray:
            try:
                return np.loadtxt(path, delimiter=",", skiprows=1)
            except Exception:
                import pandas as pd
                return pd.read_csv(path).values

        # Allow augmented directory layouts: prefer val_X.csv, otherwise fall back to val_X_scaled.csv
        vx_path = os.path.join(proc, "val_X.csv")
        if not os.path.exists(vx_path):
            alt = os.path.join(proc, "val_X_scaled.csv")
            vx_path = alt if os.path.exists(alt) else vx_path
        Xv_raw = _load_csv(vx_path).astype(np.float32)

        # Labels filename is consistent in both layouts
        yv = _load_csv(os.path.join(proc, "val_y.csv")).astype(np.int64).ravel()

        self.num_features: int = int(self.cfg.get("num_features"))
        if Xv_raw.shape[1] != self.num_features:
            raise ValueError(f"val_X.csv has {Xv_raw.shape[1]} cols; expected num_features={self.num_features}")

        # Sanitize non-finite BEFORE tensors
        valid_mask_np = np.isfinite(Xv_raw).astype(np.float32)
        Xv = np.nan_to_num(Xv_raw, nan=0.0, posinf=0.0, neginf=0.0)

        self.X = torch.tensor(Xv, dtype=torch.float32, device=self.device)
        self.valid = torch.tensor(valid_mask_np, dtype=torch.float32, device=self.device)
        self.y = torch.tensor(yv, dtype=torch.long, device=self.device)
        self.n = self.X.shape[0]

        # === Models (GAIN + classifier) ===
        g_path = self.cfg.get("gain_generator_path")
        c_path = self.cfg.get("prelim_classifier_path")
        if not g_path or not os.path.exists(g_path):
            raise FileNotFoundError(f"Missing gain_generator_path: {g_path}")
        if not c_path or not os.path.exists(c_path):
            raise FileNotFoundError(f"Missing prelim_classifier_path: {c_path}")

        t0 = time.time()
        self.gain, self.clf = _load_models_cached(self.num_features, self.device, g_path, c_path)

# --- NEW: optional StandardScaler (fit during augmentation) ---
        self.scaler = None
        sp = str(self.cfg.get("scaler_path", "data/processed/sepsis/scaler.joblib"))
        if sp and os.path.exists(sp):
            try:
                self.scaler = joblib.load(sp)
                print(f"[SepsisEnvFast] Loaded scaler: {sp}")
            except Exception as e:
                print(f"[SepsisEnvFast] WARNING: could not load scaler at {sp}: {e}")

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
                    pass

        # === Calibrator (optional) — NEW ===
        self.calibrator = None
        cal_path = self.cfg.get("calibrator_path", None)
        if cal_path and os.path.exists(cal_path):
            try:
                
                self.calibrator = joblib.load(cal_path)
                print(f"[SepsisEnvFast] Loaded calibrator from {cal_path}")
            except Exception as e:
                print(f"[SepsisEnvFast] Warning: could not load calibrator ({cal_path}): {e}")
                self.calibrator = None

        # === Rewards & shaping weights ===
        self.R_TP = float(self.cfg.get("reward_true_positive", 10000.0))
        self.R_TN = float(self.cfg.get("reward_true_negative", 1000.0))
        self.C_FP = float(self.cfg.get("penalty_false_positive", -5000.0))
        self.C_FN = float(self.cfg.get("penalty_false_negative", -20000.0))

        self.w_entropy  = float(self.cfg.get("uncertainty_factor", 0.5))
        self.w_progress = float(self.cfg.get("progress_toward_true_factor", 1.0))

        # NEW: uniform reward scale
        self.reward_scale = float(self.cfg.get("reward_scale", 0.01))

        # Practical nudges
        self.info_cost_tradeoff = float(self.cfg.get("info_cost_tradeoff", 0.0))
        _fk = self.cfg.get("first_k_cost_discount", {}) or {}
        if isinstance(_fk, Config):
            _fk = _fk.to_dict()
        self.discount_k      = int(_fk.get("k", 0))
        self.discount_factor = float(_fk.get("factor", 1.0))

        # === Costs ===
        cm = self.cfg.get("cost_mapping", {}) or {}
        if isinstance(cm, Config):
            cm = cm.to_dict()
        self.cost_mapping: Dict[int, float] = {int(k): float(v) for k, v in cm.items()}
        self.mean_panel_cost = float(np.mean(list(self.cost_mapping.values()) or [1.0]))

        # --- Policy guardrails / toggles ---
        self.block_repeats = bool(self.cfg.get("block_repeats", True))
        self.repeat_penalty = float(self.cfg.get("repeat_penalty", -200.0))

        # === Diagnosis mode / threshold ===
        self.diagnose_mode = str(self.cfg.get("diagnose_mode", "split")).lower()
        if self.diagnose_mode not in ("split", "single"):
            self.diagnose_mode = "split"
        self.decision_threshold = float(self.cfg.get("decision_threshold", 0.5))

        # Load threshold from JSON if provided — NEW
        thr_json_path = self.cfg.get("threshold_json", None)
        if thr_json_path and os.path.exists(thr_json_path):
            try:
                with open(thr_json_path, "r") as f:
                    th = json.load(f)
                chosen = th.get("chosen", "threshold_f2")
                self.decision_threshold = float(
                    th["threshold_recall90" if chosen == "threshold_recall90" else "threshold_f2"]
                )
                print(f"[SepsisEnvFast] Loaded decision_threshold={self.decision_threshold:.6f} from {thr_json_path} ({chosen})")
            except Exception as e:
                print(f"[SepsisEnvFast] Warning: could not load threshold_json ({thr_json_path}): {e}")

        # === Feature groups ===
        self.num_test_groups     = int(self.cfg.get("num_test_groups", 4))
        self.min_tests_before_dx = int(self.cfg.get("min_tests_before_diagnosis", 2))
        self.ordered_panels = set()
        fg = self.cfg.get("feature_groups", None)
        if isinstance(fg, Config):
            fg = fg.to_dict()
        if isinstance(fg, dict) and len(fg) >= self.num_test_groups:
            self.feature_groups = {int(k): list(map(int, v)) for k, v in fg.items()}
        else:
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
        obs_dim = self.num_features * 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        if self.diagnose_mode == "split":
            self.DIAG_POS = self.num_test_groups
            self.DIAG_NEG = self.num_test_groups + 1
            n_actions = self.num_test_groups + 2
        else:
            self.DIAG = self.num_test_groups
            n_actions = self.num_test_groups + 1
        self.action_space = spaces.Discrete(n_actions)

        # === Episode state (prealloc tensors) ===
        self.curr = torch.zeros(self.num_features, device=self.device)
        self.mask = torch.zeros(self.num_features, device=self.device)
        self.full = torch.zeros(self.num_features, device=self.device)
        self.full_valid = torch.zeros(self.num_features, device=self.device)
        self._obs_buf = np.zeros(obs_dim, dtype=np.float32)

        self._prev_p = 0.5
        self._prev_H = math.log(2.0)

        self.label = 0
        self.patient_idx = 0
        self.step_count = 0

    # -------------------------
    # MaskablePPO hook
    # -------------------------
    def action_masks(self) -> np.ndarray:
        n = self.action_space.n
        mask = np.ones(n, dtype=np.int8)

        # block already-ordered panels
        for a in self.ordered_panels:
            if 0 <= a < self.num_test_groups:
                mask[a] = 0

        # block diagnose until enough tests are seen
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
    @torch.no_grad()
    def _p_expired(self) -> float:
        # Create a batch dimension for the current state. Shape becomes [1, 37].
        state_batch = self.curr.unsqueeze(0)
        mask_batch = self.mask.unsqueeze(0)

        # Pass the 2D tensor to GAIN. The output 'imputed' will have shape [1, 37].
        imputed = self.gain(state_batch, mask_batch)
        if not torch.isfinite(imputed).all():
            imputed = torch.nan_to_num(imputed, nan=0.0, posinf=0.0, neginf=0.0)

        # Pass the 2D imputed tensor directly to the classifier.
        # The output 'logits' will have the correct shape [1, 2].
        logits = self.clf(imputed)

        # This softmax and indexing will now work correctly on the [1, 2] logits tensor.
        p = float(torch.softmax(logits, dim=1)[0, 1].item())

        # clamp WITHOUT numpy (avoid np scoping issues)
        p = float(max(1e-6, min(p, 1.0 - 1e-6)))

        # optional calibrator (prob -> prob). Use a LOCAL alias to numpy.
        if getattr(self, "calibrator", None) is not None:
            try:
                import numpy as _np
                try:
                    # many calibrators accept a single-prob feature
                    p = float(self.calibrator.predict_proba(_np.array([[p]], dtype=float))[:, 1][0])
                except Exception:
                    # fallback to 2-feature input [1-p, p]
                    p = float(self.calibrator.predict_proba(_np.array([[1.0 - p, p]], dtype=float))[:, 1][0])
            except Exception:
                # if anything goes wrong, keep the unclibrated p
                pass

            # clamp again post-calibration
            p = float(max(1e-6, min(p, 1.0 - 1e-6)))

        # cache for evaluator / terminal info
        self._prev_p = p
        return p




    @staticmethod
    def _entropy_nat(p: float) -> float:
        q = 1.0 - p
        return - (p * math.log(p + 1e-12) + q * math.log(q + 1e-12))

    @staticmethod
    def _logit(p: float) -> float:
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        return math.log(p / (1.0 - p))

    def _pack_obs(self) -> np.ndarray:
        c = self.curr.detach().cpu().numpy()
        m = self.mask.detach().cpu().numpy()
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        self._obs_buf[: self.num_features] = c
        self._obs_buf[self.num_features :] = m
        return self._obs_buf

    def _check_obs(self, obs: np.ndarray, stage="step"):
        if not np.all(np.isfinite(obs)):
            bad = np.where(~np.isfinite(obs))
            print(f"[NaNGuard] Non-finite obs at {stage}: idx={bad}")
        return obs

    # -------------------------
    # Gymnasium API
    # -------------------------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        self.patient_idx = int(self.rng.integers(0, self.n))
        self.label = int(self.y[self.patient_idx].item())

        self.full.copy_(self.X[self.patient_idx])
        self.full_valid.copy_(self.valid[self.patient_idx])

        self.curr.zero_()
        self.mask.zero_()
        self.step_count = 0
        self.ordered_panels = set()

        p0 = self._p_expired()
        self._prev_p = p0
        self._prev_H = self._entropy_nat(p0)

        return self._check_obs(self._pack_obs(), "reset"), {}

    def step(self, action: int):
        done = False
        reward = 0.0
        info: Dict[str, Any] = {}

        # -------- Diagnose (terminal) --------
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

        # -------- Order a panel --------
        else:
            # Case 1: The action is a repeat of an already ordered panel.
            if 0 <= int(action) < self.num_test_groups and int(action) in self.ordered_panels:
                # Apply configured repeat penalty. If block_repeats=True, terminate.
                reward += float(self.repeat_penalty)
                info.update({
                    "action_type": "repeat_action_failure" if self.block_repeats else "repeat_blocked",
                    "panel": int(action),
                    "repeat_penalty": float(self.repeat_penalty),
                })
                if self.block_repeats:
                    done = True
            # Case 2: The action is a valid, new panel order (or an invalid one).
            else:
                feats = self.feature_groups.get(int(action), [])
                if feats:
                    p_before = float(self._prev_p)

                    # reveal features
                    self.mask[feats] = self.full_valid[feats]
                    self.curr[feats] = self.full[feats]
                    self.curr = torch.nan_to_num(self.curr, nan=0.0, posinf=0.0, neginf=0.0)
                    self.mask = torch.nan_to_num(self.mask, nan=0.0, posinf=0.0, neginf=0.0)
                    self.ordered_panels.add(int(action))

                    # cost
                    c = float(self.cost_mapping.get(int(action), 0.0))
                    if self.step_count < self.discount_k:
                        c *= self.discount_factor
                    reward -= c

                    # after
                    p_after = float(self._p_expired())

                    # info-gain shaping
                    gain = max(0.0, abs(p_after - 0.5) - abs(p_before - 0.5))
                    ig_w = float(self.cfg.get("info_gain_reward", 5.0))
                    reward += ig_w * gain

                    # tiny per-step penalty
                    step_pen = float(self.cfg.get("step_penalty", -0.2))
                    reward += step_pen

                    self._prev_H = self._entropy_nat(p_after)

                    info.update({
                        "action_type": "order", "panel": int(action), "action_cost": float(c),
                        "p_before": p_before, "p_after": p_after, "info_gain": float(gain),
                        "info_gain_reward": float(ig_w * gain), "step_penalty": float(step_pen),
                    })
                else:
                    # invalid/no-op
                    info.update({"action_type": "noop", "panel": int(action)})

        self.step_count += 1
        # safety cap (unchanged)
        if self.step_count >= (self.num_test_groups + 2):
            done = True

        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)) * self.reward_scale
        return self._check_obs(self._pack_obs(), "step"), reward, bool(done), False, info


# Factory
def make_env(cfg: Config) -> SepsisEnvFast:
    return SepsisEnvFast(cfg)
