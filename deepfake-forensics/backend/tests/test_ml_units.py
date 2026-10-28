"""Unit tests for the shared ML package: preprocessing consistency, metrics, splits, perturbations."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from ml.data.dataset import ManifestImageDataset
from ml.data.manifest import check_group_leakage, filter_manifest, group_split
from ml.evaluation.metrics import compute_binary_metrics, grouped_bootstrap_ci, select_threshold
from ml.models.checkpoint import load_model_for_inference, read_checkpoint_meta
from ml.models.factory import DetectorNet, ModelConfig, build_model
from ml.preprocessing.transforms import PreprocessConfig, build_eval_transform, load_image_rgb, resize_for_model
from ml.robustness.perturbations import Perturbation, apply_perturbation, downscale, jpeg_compress
from tests.conftest import image_bytes, write_test_checkpoint


def test_preprocessing_consistent_between_inference_and_evaluation(tmp_path):
    """The API path (bytes -> load_image_rgb -> eval transform) must equal the dataset path."""
    data = image_bytes("PNG", size=(120, 90), seed=7)
    (tmp_path / "img.png").write_bytes(data)
    cfg = PreprocessConfig(img_size=64, resize_mode="resize_center_crop")
    tf = build_eval_transform(cfg)
    api_tensor = tf(load_image_rgb(data))
    ds = ManifestImageDataset(pd.DataFrame([{"path": "img.png", "label": 1}]), tmp_path, tf)
    ds_tensor, label, _ = ds[0]
    assert torch.equal(api_tensor, ds_tensor) and label == 1
    assert api_tensor.shape == (3, 64, 64)


def test_backend_registry_uses_checkpoint_preprocessing(tmp_path):
    from app.services.model_registry import ModelRegistry
    from app.database.db import Database
    from app.database.repositories import SettingsRepository
    from tests.conftest import make_settings

    s = make_settings(tmp_path)
    s.ensure_dirs()
    write_test_checkpoint(s.artifacts_dir, img_size=72)
    reg = ModelRegistry(s, SettingsRepository(Database(s.database_path)))
    lm = reg.require_active("ai_generated")
    assert lm.preprocess.img_size == 72 and lm.preprocess.resize_mode == "resize"
    img = load_image_rgb(image_bytes(size=(100, 50)))
    assert torch.equal(lm.transform(img), build_eval_transform(PreprocessConfig.from_dict(lm.meta["preprocess"]))(img))
    assert reg.load(lm.entry.model_id) is lm  # cached, not reloaded


def test_center_crop_and_resize_modes():
    img = Image.new("RGB", (200, 100))
    assert resize_for_model(img, PreprocessConfig(img_size=64)).size == (64, 64)
    assert resize_for_model(img, PreprocessConfig(img_size=64, resize_mode="resize")).size == (64, 64)
    with pytest.raises(ValueError):
        PreprocessConfig(resize_mode="stretch")


def test_rgba_and_grayscale_converted_to_rgb():
    for mode in ("RGBA", "L", "P"):
        buf = io.BytesIO()
        Image.new(mode, (40, 40)).save(buf, format="PNG")
        assert load_image_rgb(buf.getvalue()).mode == "RGB"
    with pytest.raises(ValueError):
        load_image_rgb(b"garbage")


@pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
@pytest.mark.parametrize("freq", [False, True])
def test_models_return_two_logits(arch, freq):
    m = build_model(ModelConfig(arch=arch, pretrained=False, frequency_branch=freq)).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 2)
    assert isinstance(m.gradcam_target_layer, torch.nn.Module)


def test_gradcam_shape_and_range():
    from ml.explainability.gradcam import GradCAM

    m: DetectorNet = build_model(ModelConfig(arch="resnet50", pretrained=False)).eval()
    cam, logits = GradCAM(m, m.gradcam_target_layer)(torch.randn(1, 3, 64, 64), class_idx=1)
    assert cam.shape == (64, 64) and cam.min() >= 0 and cam.max() <= 1
    assert logits.shape == (1, 2)


def test_checkpoint_roundtrip_identical_outputs(tmp_path):
    p = write_test_checkpoint(tmp_path)
    assert read_checkpoint_meta(p)["model_id"] == "TEST_fixture_model"
    m1, _ = load_model_for_inference(p, torch.device("cpu"))
    m2, _ = load_model_for_inference(p, torch.device("cpu"))
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(m1(x), m2(x))


def test_metrics_known_values():
    m = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], threshold=0.5)
    assert m["accuracy"] == 0.5
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert m["roc_auc"] == 0.75
    perfect = compute_binary_metrics([0, 1], [0.2, 0.8])
    assert perfect["f1_macro"] == 1.0
    assert compute_binary_metrics([1, 1], [0.7, 0.9])["roc_auc"] is None  # undefined, not invented


def test_threshold_selection_separates_classes():
    t = select_threshold([0, 0, 0, 1, 1, 1], [0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    assert 0.3 < t <= 0.6
    assert select_threshold([0, 0], [0.1, 0.2]) == 0.5


def test_grouped_bootstrap_ci():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 50)
    s = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 100), 0, 1)
    groups = [f"g{i // 5}" for i in range(100)]
    ci = grouped_bootstrap_ci(y, s, groups, 0.5, n_boot=200)
    a = ci["metrics"]["accuracy"]
    assert ci["n_groups"] == 20 and a["low"] <= compute_binary_metrics(y, s, 0.5)["accuracy"] <= a["high"]


def test_group_split_never_leaks():
    df = pd.DataFrame({"group": [f"v{i // 4}" for i in range(200)], "label": [(i // 4) % 2 for i in range(200)]})
    df["split"] = group_split(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    assert check_group_leakage(df) == []
    frac = df["split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.7) < 0.1
    again = group_split(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    assert (again == df["split"]).all()  # reproducible


def test_source_filters():
    df = pd.DataFrame({"path": list("abcd"), "label": [0, 1, 0, 1], "split": ["test"] * 4, "source": ["real", "sd", "real", "mj"]})
    assert set(filter_manifest(df, ["test"], exclude_sources=["mj"])["source"]) == {"real", "sd"}
    assert set(filter_manifest(df, ["test"], include_sources=["mj"])["path"]) == {"d"}
    assert len(filter_manifest(df, ["test"], max_per_class=1)) == 2


def test_perturbations():
    img = load_image_rgb(image_bytes(size=(100, 80)))
    assert jpeg_compress(img, 50).size == img.size
    assert downscale(img, factor=0.5).size == img.size
    assert downscale(img, max_side=40, restore=False).size == (40, 32)
    a = apply_perturbation(img, {"type": "noise", "std": 10}, seed=3)
    b = apply_perturbation(img, {"type": "noise", "std": 10}, seed=3)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert Perturbation.from_dict({"type": "jpeg", "quality": 70}).name == "jpeg(quality=70)"
    with pytest.raises(ValueError):
        Perturbation.from_dict({"type": "rotate"})
    with pytest.raises(ValueError):
        jpeg_compress(img, 0)


def test_dhash_near_duplicate_detection():
    from ml.data.validate import dhash, near_duplicate_pairs

    img = load_image_rgb(image_bytes(size=(64, 64), seed=1))
    h1, h2 = dhash(img), dhash(jpeg_compress(img, 90))
    h3 = dhash(load_image_rgb(image_bytes(size=(64, 64), seed=2)))
    pairs = near_duplicate_pairs({0: h1, 1: h2, 2: h3}, max_dist=6)
    assert (0, 1) in [(a, b) for a, b, _ in pairs]
    assert all(2 not in (a, b) for a, b, _ in pairs)


def test_training_config_validation(tmp_path):
    from ml.training.config import load_train_config

    p = tmp_path / "c.yaml"
    p.write_text("experiment_id: x\ntrack: ai_generated\ndataset: {root: a, manifest: b.csv, include_sources: [s], exclude_sources: [s]}\n")
    with pytest.raises(ValueError, match="both included and excluded"):
        load_train_config(p)
    p.write_text("experiment_id: x\ntrack: video\ndataset: {name: d}\n")
    with pytest.raises(ValueError, match="track"):
        load_train_config(p)
    p.write_text("experiment_id: x\ntrack: ai_generated\ndataset: {name: d}\n")
    cfg = load_train_config(p, ["train.epochs=3", "model.arch=efficientnet_b0"])
    assert cfg["train"]["epochs"] == 3 and cfg["model"]["arch"] == "efficientnet_b0"
