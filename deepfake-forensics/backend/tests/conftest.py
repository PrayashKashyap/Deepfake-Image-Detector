"""Test fixtures.

Tests use temporary storage/artifact directories and a tiny randomly initialised
EfficientNet-B0 checkpoint (flagged as a synthetic fixture). This exercises the real
checkpoint-loading and inference code path; the numbers it produces are meaningless and
are only checked for type/range/consistency, never for accuracy.
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

# Isolate the module-level app (app.main.app) from the real storage before any app import.
_SESSION_TMP = Path(tempfile.mkdtemp(prefix="dff-tests-"))
os.environ.update({
    "DFF_STORAGE_DIR": str(_SESSION_TMP / "storage"),
    "DFF_ARTIFACTS_DIR": str(_SESSION_TMP / "artifacts"),
    "DFF_REPORTS_DIR": str(_SESSION_TMP / "reports"),
    "DFF_DATABASE_PATH": str(_SESSION_TMP / "storage" / "app.db"),
    "DFF_DEVICE": "cpu",
    "DFF_DATASETS_CONFIG": str(_SESSION_TMP / "datasets.yaml"),
})

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from ml.common import CLASS_NAMES, utc_now_iso  # noqa: E402
from ml.models.checkpoint import save_checkpoint  # noqa: E402
from ml.models.factory import ModelConfig, build_model  # noqa: E402
from ml.preprocessing.transforms import PreprocessConfig  # noqa: E402


def make_settings(tmp: Path, **overrides):
    from app.core.config import Settings

    return Settings(
        storage_dir=tmp / "storage",
        artifacts_dir=tmp / "artifacts",
        reports_dir=tmp / "reports",
        database_path=tmp / "storage" / "app.db",
        device="cpu",
        data_roots=str(tmp / "data"),
        **overrides,
    )


def write_test_checkpoint(artifacts: Path, track: str = "ai_generated", model_id: str = "TEST_fixture_model", img_size: int = 64) -> Path:
    import torch

    torch.manual_seed(0)
    mcfg = ModelConfig(arch="efficientnet_b0", pretrained=False)
    model = build_model(mcfg)
    meta = {
        "model_id": model_id,
        "display_name": "TEST efficientnet_b0",
        "track": track,
        "arch": "efficientnet_b0",
        "frequency_branch": False,
        "model_config": mcfg.to_dict(),
        "pretrained_init": None,
        "preprocess": PreprocessConfig(img_size=img_size, resize_mode="resize").to_dict(),
        "class_names": list(CLASS_NAMES),
        "threshold": {"value": 0.5, "selected_on": "test", "method": "fixed"},
        "dataset": {"dataset_ids": ["SYNTHETIC_FIXTURE_test"], "is_synthetic_fixture": True, "train_sources": []},
        "training": {"experiment_id": "TEST", "run_id": "r0", "seed": 0},
        "version": "test@e0",
        "created_at": utc_now_iso(),
    }
    return save_checkpoint(artifacts / "checkpoints" / track / f"{model_id}.pt", model, meta)


def image_bytes(fmt: str = "PNG", size=(96, 80), seed: int = 0, exif: bool = False) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    kw = {}
    if exif:
        ex = Image.Exif()
        ex[0x0131] = "UnitTestSoftware 1.0"  # Software
        ex[0x013B] = "Secret Person"  # Artist (sensitive -> redacted)
        kw["exif"] = ex
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def client_no_model(tmp_root: Path):
    from app.main import create_app

    app = create_app(make_settings(tmp_root))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client(tmp_root: Path):
    from app.main import create_app

    settings = make_settings(tmp_root)
    write_test_checkpoint(settings.artifacts_dir)
    app = create_app(settings)
    with TestClient(app) as c:
        c.app_settings = settings  # type: ignore[attr-defined]
        yield c
