"""Real model inference + Grad-CAM. Uses the exact preprocessing stored in the checkpoint."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from ml.common import CLASS_NAMES, FAKE
from ml.explainability.gradcam import GradCAM, heatmap_rgb, overlay
from ml.preprocessing.transforms import denormalize_to_uint8

from app.services.model_registry import LoadedModel, ModelRegistry

DISCLAIMER = (
    "This prediction is a probabilistic model output, not proof of authenticity or manipulation. "
    "The detector can produce false positives and false negatives, particularly for image sources, "
    "generators, manipulation methods or post-processing not represented in its training data. "
    "It must not be used as legal or forensic evidence."
)
SCORE_NOTE = (
    "The fake score is the model's softmax output for the FAKE class. It is not a calibrated probability "
    "that the image is fake unless calibration has been separately evaluated."
)


@dataclass
class Prediction:
    fake_score: float
    label: str
    threshold: float
    threshold_source: str
    inference_ms: float
    logits: list[float]


def _amp_ctx(lm: LoadedModel, enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled and lm.device.type == "cuda")


def predict_image(registry: ModelRegistry, lm: LoadedModel, img: Image.Image, amp: bool) -> Prediction:
    x = lm.transform(img).unsqueeze(0)
    with lm.lock, torch.no_grad():
        if lm.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        xd = x.to(lm.device)
        with _amp_ctx(lm, amp):
            logits = lm.model(xd).float()
        if lm.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    thr, src = registry.effective_threshold(lm.entry.track, lm.meta)
    score = float(probs[FAKE])
    return Prediction(score, CLASS_NAMES[FAKE] if score >= thr else CLASS_NAMES[1 - FAKE], thr, src, ms, logits[0].cpu().tolist())


def gradcam_images(lm: LoadedModel, img: Image.Image, class_idx: int) -> dict[str, Any]:
    """Return model-input view, heatmap and overlay (uint8 RGB arrays) for the given class."""
    x = lm.transform(img).unsqueeze(0).to(lm.device)
    t0 = time.perf_counter()
    with lm.lock:
        cam, _ = GradCAM(lm.model, lm.model.gradcam_target_layer)(x, class_idx)  # full precision for stable gradients
    ms = (time.perf_counter() - t0) * 1000
    view = denormalize_to_uint8(x[0], lm.preprocess)
    return {"input": view, "heatmap": heatmap_rgb(cam), "overlay": overlay(view, cam), "cam": cam, "ms": ms}


def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")
