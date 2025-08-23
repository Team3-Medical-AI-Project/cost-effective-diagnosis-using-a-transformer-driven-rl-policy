# tools/dump_costs_constraints.py
# I wrote this to export the cost table and constraints my Sepsis env uses.
# It can (A) dump from an already-created env via dump_from_env(env), or
# (B) run standalone (it will try to import + construct the env by itself).

from pathlib import Path
import json, csv
import sys
import inspect

def dump_from_env(env, out_dir: Path | None = None) -> dict:
    # I make the output folder if not given
    out = Path(out_dir) if out_dir else Path.cwd() / "reports"
    out.mkdir(parents=True, exist_ok=True)

    payload = {"panels": {}, "constraints": {}, "meta": {}}

    # I grab obvious attributes; adjust names if my env uses different ones
    for name in dir(env):
        if name.isupper() and any(k in name for k in ["COST", "COSTS", "PRICE", "TEST", "PANEL", "GROUP", "MASK"]):
            val = getattr(env, name)
            if isinstance(val, (dict, list, tuple, int, float, str)):
                payload["panels" if "COST" in name or "TEST" in name or "PANEL" in name or "GROUP" in name else "constraints"][name] = val

    # meta (dims etc.)
    for cand in ["NUM_TEST_GROUPS","ACTION_DIM","NUM_FEATURES","DIAGNOSE_ACTION"]:
        if hasattr(env, cand):
            payload["meta"][cand] = getattr(env, cand)

    # write JSON
    json_path = out / "sepsis_costs_constraints.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print("Wrote:", json_path)

    # try to find a dict that looks like costs and also write CSV
    cost_dict = None
    for k, v in payload["panels"].items():
        if isinstance(v, dict) and "COST" in k:
            cost_dict = v; break
    if isinstance(cost_dict, dict):
        csv_path = out / "sepsis_panel_costs.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["panel", "cost_unit"])
            for k, v in cost_dict.items():
                w.writerow([k, v])
        print("Wrote:", csv_path)

    return payload

def _standalone():
    # I wrote this to work even if I just run: python tools/dump_costs_constraints.py
    sys.path.insert(0, str(Path.cwd()))
    try:
        mod = __import__("src.training.train_rl_agent_sepsis", fromlist=["*"])
    except Exception as e:
        print("import_failed:", e); return
    SepsisEnv = getattr(mod, "SepsisEnv", None)
    if SepsisEnv is None or not inspect.isclass(SepsisEnv):
        print("SepsisEnv_not_found"); return

    # find a config dict to instantiate with
    cfg = None
    for name in dir(mod):
        if name.upper() in ("CONFIG","SEPSIS_CONFIG","DEFAULT_CONFIG"):
            val = getattr(mod, name)
            if isinstance(val, dict):
                cfg = val; break
    if cfg is None:
        print("CONFIG_not_found"); return

    try:
        env = SepsisEnv(cfg)
    except Exception as e:
        print("env_init_failed:", e); return

    dump_from_env(env)

if __name__ == "__main__":
    _standalone()
