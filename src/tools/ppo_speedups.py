# src/tools/ppo_speedups.py
import os
import torch

# ---- Safe speed knobs that do NOT change learning semantics ----

def set_torch_runtime_threads():
    """
    Let PyTorch/OneDNN use CPU cores efficiently without fighting Python.
    Safe for Windows. Has no effect on gradients/updates.
    """
    try:
        n = max(1, (os.cpu_count() or 8) - 1)
        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

def enable_tf32_if_available():
    """
    TF32 speeds up matmuls on Ampere+ (RTX 3070) with negligible numeric impact for PPO MLPs.
    Safe and widely used.
    """
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            # PyTorch >= 2.0
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    except Exception:
        pass

def pick_policy_device():
    """
    Put the PPO policy on CUDA if available; else CPU.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

def make_vec_env(env_fns):
    """
    Prefer SubprocVecEnv for true parallel stepping; fall back to DummyVecEnv gracefully.
    Keeps behavior identical, only faster wall-clock.
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    try:
        return SubprocVecEnv(env_fns, start_method="spawn")
    except Exception as e:
        print(f"[ppo_speedups] SubprocVecEnv failed ({e}); using DummyVecEnv")
        return DummyVecEnv(env_fns)
