from pathlib import Path
import yaml

def load_config(path: str = "config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

CFG = load_config()
DATA_ROOT = Path(CFG.get("data_root", "./data")).resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)