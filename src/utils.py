# src/utils.py
from __future__ import annotations
import yaml
from typing import Any, Dict

class Config:
    """
    Dict <-> Attribute bridge that:
      - leaves non-string keys (e.g., {0: 30.86}) intact inside dict fields
      - exposes string keys as attributes (recursively)
      - provides dict-like methods: get, __getitem__, to_dict, etc.
    """
    def __init__(self, data: Dict[str, Any]):
        self._data: Dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                # Wrap nested dicts, but DO NOT try to setattr for non-string keys inside them
                v_wrapped = Config(v)
                self._data[k] = v_wrapped
                if isinstance(k, str):
                    setattr(self, k, v_wrapped)
            else:
                self._data[k] = v
                if isinstance(k, str):
                    setattr(self, k, v)

    # --- dict-like helpers ---
    def get(self, key: Any, default: Any = None) -> Any:
        v = self._data.get(key, default)
        return v

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __contains__(self, key: Any) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def to_dict(self) -> Dict[str, Any]:
        out = {}
        for k, v in self._data.items():
            if isinstance(v, Config):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(data)
