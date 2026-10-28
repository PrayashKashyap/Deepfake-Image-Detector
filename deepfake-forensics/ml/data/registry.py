"""Resolve named datasets to local paths without hardcoding machine-specific locations.

Lookup order for dataset `<name>`:
    1. Environment variables  DFF_DATASET_<NAME>_ROOT  and  DFF_DATASET_<NAME>_MANIFEST
    2. The YAML file pointed to by DFF_DATASETS_CONFIG (default: ml/configs/datasets.local.yaml)

See ml/configs/datasets.example.yaml for the file format.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ml.common import load_yaml, resolve_repo_path

DEFAULT_DATASETS_CONFIG = "ml/configs/datasets.local.yaml"


@dataclass
class DatasetRef:
    name: str
    root: Path
    manifest: Path
    track: str | None = None
    description: str = ""


def datasets_config_path() -> Path:
    return resolve_repo_path(os.environ.get("DFF_DATASETS_CONFIG", DEFAULT_DATASETS_CONFIG))


def list_datasets() -> dict[str, dict]:
    path = datasets_config_path()
    if not path.exists():
        return {}
    return dict(load_yaml(path).get("datasets", {}) or {})


def resolve_dataset(name: str) -> DatasetRef:
    env_key = name.upper().replace("-", "_")
    env_root = os.environ.get(f"DFF_DATASET_{env_key}_ROOT")
    env_manifest = os.environ.get(f"DFF_DATASET_{env_key}_MANIFEST")
    entry = list_datasets().get(name, {})
    root = env_root or entry.get("root")
    manifest = env_manifest or entry.get("manifest")
    if not root or not manifest:
        raise KeyError(
            f"Dataset {name!r} is not configured. Add it to {datasets_config_path()} "
            f"(see ml/configs/datasets.example.yaml) or set DFF_DATASET_{env_key}_ROOT / _MANIFEST."
        )
    return DatasetRef(
        name=name,
        root=resolve_repo_path(root),
        manifest=resolve_repo_path(manifest),
        track=entry.get("track"),
        description=entry.get("description", ""),
    )
