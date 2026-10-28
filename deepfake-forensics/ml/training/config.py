"""Training configuration (YAML) parsing, validation and dotted overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from ml.common import TRACKS, load_yaml

DEFAULTS: dict[str, Any] = {
    "experiment_id": None,
    "track": None,
    "description": "",
    "dataset": {
        "name": None,  # resolved through ml.data.registry when root/manifest are not given
        "root": None,
        "manifest": None,
        "train_split": "train",
        "val_split": "val",
        "include_sources": [],
        "exclude_sources": [],
        "max_train_per_class": None,
        "max_val_per_class": None,
    },
    "model": {"arch": "resnet50", "pretrained": True, "frequency_branch": False, "dropout": 0.2},
    "preprocess": {"img_size": 224, "resize_mode": "resize_center_crop"},
    "augment": {"hflip": True, "random_crop_scale": None, "jpeg_prob": 0.0, "blur_prob": 0.0},
    "train": {
        "epochs": 20,
        "batch_size": 32,
        "lr": 1.0e-4,
        "weight_decay": 1.0e-4,
        "seed": 42,
        "num_workers": 4,
        "amp": True,
        "device": "auto",
        "scheduler": "cosine",  # cosine | plateau | none
        "warmup_epochs": 1,
        "early_stopping_patience": 5,
        "monitor": "val_auc",  # val_auc | val_loss | val_f1_macro | val_balanced_accuracy
        "grad_clip": 1.0,
        "threshold_method": "youden",
        "max_steps_per_epoch": None,
        "deterministic": False,
    },
    "output_dir": None,  # default: $DFF_ARTIFACTS_DIR/experiments (artifacts/experiments)
}

MONITORS = {"val_auc": "max", "val_loss": "min", "val_f1_macro": "max", "val_balanced_accuracy": "max"}


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply `a.b.c=value` overrides; values are parsed as YAML scalars."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override {item!r} must look like key.sub=value")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_train_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = _deep_update(copy.deepcopy(DEFAULTS), load_yaml(path))
    cfg = apply_overrides(cfg, overrides)
    validate_train_config(cfg)
    return cfg


def validate_train_config(cfg: dict[str, Any]) -> None:
    errors = []
    if not cfg.get("experiment_id"):
        errors.append("experiment_id is required")
    if cfg.get("track") not in TRACKS:
        errors.append(f"track must be one of {TRACKS}")
    ds = cfg["dataset"]
    if not ds.get("name") and not (ds.get("root") and ds.get("manifest")):
        errors.append("dataset.name or both dataset.root and dataset.manifest are required")
    overlap = set(ds.get("include_sources") or []) & set(ds.get("exclude_sources") or [])
    if overlap:
        errors.append(f"sources both included and excluded: {sorted(overlap)}")
    t = cfg["train"]
    if int(t["epochs"]) < 1:
        errors.append("train.epochs must be >= 1")
    if int(t["batch_size"]) < 1:
        errors.append("train.batch_size must be >= 1")
    if float(t["lr"]) <= 0:
        errors.append("train.lr must be > 0")
    if t["scheduler"] not in ("cosine", "plateau", "none"):
        errors.append("train.scheduler must be cosine | plateau | none")
    if t["monitor"] not in MONITORS:
        errors.append(f"train.monitor must be one of {list(MONITORS)}")
    if errors:
        raise ValueError("Invalid training config:\n  - " + "\n  - ".join(errors))
