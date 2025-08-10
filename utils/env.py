import os
from dotenv import load_dotenv
load_dotenv()

def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

POLYGON_API_KEY = require_env("POLYGON_API_KEY")