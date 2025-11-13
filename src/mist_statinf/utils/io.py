import os, json, pickle
from typing import Any, Dict
from omegaconf import OmegaConf
import pandas as pd

def load_config(path: str) -> Dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)

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
