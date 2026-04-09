from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = DATA_DIR / "raw" / "vip_seed.csv"
CHECKPOINT_DATA = DATA_DIR / "checkpoints" / "vip_seed.csv"
SYNTHETIC_DIR = DATA_DIR / "synthetic"


def ensure_project_root_on_path() -> Path:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT
