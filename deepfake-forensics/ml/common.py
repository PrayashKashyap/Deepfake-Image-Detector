"""Shared constants and small utilities used by training, evaluation and the backend.

Class convention (used everywhere in this project):
    0 = REAL, 1 = FAKE
The displayed "fake score" is softmax(logits)[FAKE]. It is a model score, not a
calibrated probability of truth unless calibration has been evaluated separately.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REAL = 0
FAKE = 1
CLASS_NAMES: tuple[str, str] = ("REAL", "FAKE")

TRACK_AI_GENERATED = "ai_generated"
TRACK_FACE_DEEPFAKE = "face_deepfake"
TRACKS: tuple[str, str] = (TRACK_AI_GENERATED, TRACK_FACE_DEEPFAKE)
TRACK_LABELS = {
    TRACK_AI_GENERATED: "AI-generated image detection",
    TRACK_FACE_DEEPFAKE: "Facial deepfake detection",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_SCHEMA_VERSION = "1.0"


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def timestamp_id() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def set_seed(seed: int, deterministic: bool = False) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(preference: str = "auto"):
    """Return a torch.device. 'auto' uses CUDA when available and falls back to CPU."""
    import torch

    preference = (preference or "auto").lower()
    if preference == "cpu":
        return torch.device("cpu")
    if preference.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(preference)
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_mb": int(props.total_memory // (1024 * 1024)),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    return info


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file {path} must contain a mapping at the top level")
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(data: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default, allow_nan=False)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_repo_path(p: str | Path) -> Path:
    """Resolve a path relative to the repository root (absolute paths are kept)."""
    p = Path(os.path.expandvars(str(p))).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def artifacts_dir() -> Path:
    return resolve_repo_path(os.environ.get("DFF_ARTIFACTS_DIR", "artifacts"))


def reports_dir() -> Path:
    return resolve_repo_path(os.environ.get("DFF_REPORTS_DIR", "reports"))
