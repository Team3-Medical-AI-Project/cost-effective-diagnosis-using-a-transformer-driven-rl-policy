# -*- coding: utf-8 -*-
"""
MaskablePPO loader that is friendly to SB3 2.x when the saved .zip contains
legacy kwargs (e.g., 'use_sde', 'sde_sample_freq') from SB3 1.x.

Strategy:
  1) Try native MaskablePPO.load().
  2) If that fails on unknown kwargs, fall back to manual loading:
     - read tensors with load_from_zip_file()
     - instantiate a fresh MaskablePPO with minimal/default kwargs
     - load weights with set_parameters(..., exact_match=False)

This avoids passing any legacy constructor args and has been reliable
with archives trained using default MLP policy layouts.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.save_util import load_from_zip_file


def load_maskableppo_compat(path: str,
                            env: Any,
                            device: str = "auto",
                            verbose: int = 0) -> MaskablePPO:
    # 1) Try native loading first
    try:
        return MaskablePPO.load(path, env=env, device=device, print_system_info=False)
    except TypeError as e:
        # most likely "__init__() got an unexpected keyword argument 'use_sde'"
        last_err = e  # noqa: F841
    except Exception as e:
        last_err = e  # noqa: F841

    # 2) Manual fallback
    data, params, _pyt = load_from_zip_file(path, device=device, verbose=verbose)
    # keep policy_kwargs if present, but drop known-legacy entries
    policy_kwargs: Dict[str, Any] = dict(data.get("policy_kwargs") or {})
    for bad in ("use_sde", "sde_sample_freq"):
        policy_kwargs.pop(bad, None)

    # Instantiate a fresh model with minimal/default args; do NOT pass legacy kwargs
    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs if policy_kwargs else None,
        device=device,
        verbose=verbose,
    )

    # Load parameters; exact_match=False tolerates minor name/layout diffs
    if params is not None:
        model.set_parameters(params, device=device, exact_match=False)

    return model
