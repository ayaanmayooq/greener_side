# utils/config.py
from pathlib import Path
import yaml

def _project_root() -> Path:
    # utils/ is at <root>/utils → parent of this file’s parent = repo root
    return Path(__file__).resolve().parents[1]

def load_config(path: str | Path | None = None):
    cfg_path = _project_root() / "config.yaml" if path is None else Path(path)
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

CFG = load_config()
DATA_ROOT = (_project_root() / CFG.get("data_root", "./data")).resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)