import os, json, pickle
from typing import Any, Dict, Optional
from omegaconf import OmegaConf
import pandas as pd
from importlib import resources
import pickle
from pathlib import Path
import glob
import os

CONFIG_ROOT = Path(__file__).parent.parent / "configs"

def load_config(path: str) -> Dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)

def load_meta_dataset(dataset_path: str) -> list[dict]:
    """
    Loads *_dataset.pkl from a given directory.
    """
    matches = glob.glob(os.path.join(dataset_path, "*dataset.pkl"))
    if not matches:
        raise FileNotFoundError(f"No file matching *dataset.pkl found in: {dataset_path}")
    if len(matches) > 1:
        raise ValueError(f"Multiple files match *dataset.pkl: {matches}. Expected exactly one.")
    pkl_path = matches[0]
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def load_quickstart_cfg(config_path: Optional[str]) -> dict:
    """
    Load quickstart config:
      - If config_path is given -> load from this path.
      - Else -> load package resource 'configs/inference/quickstart.yaml'.
    Returns a plain dict (OmegaConf resolved).
    """
    if config_path:
        cfg = OmegaConf.load(config_path)
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    
    cfg_path = CONFIG_ROOT / "inference" / "quickstart.yaml"
    cfg = OmegaConf.load(str(cfg_path))
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)

def save_json(path: str, obj: Any, indent: int = 2) -> None:
    _ensure_dir(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)

def save_pickle(path: str, obj: Any, protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
    _ensure_dir(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)

def save_csv(path: str, df: pd.DataFrame, index: bool = False) -> None:
    _ensure_dir(path)
    df.to_csv(path, index=index)
