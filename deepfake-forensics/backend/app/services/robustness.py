"""Robustness laboratory: run the real model on one image and its perturbed variants.

This is a single-image demonstration, NOT a benchmark. Benchmark robustness results come
from `python -m ml.robustness.evaluate_robustness` over a full test split.
"""
from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image

from ml.common import TRACK_LABELS
from ml.preprocessing.transforms import load_image_rgb
from ml.robustness.perturbations import downscale, gaussian_blur, gaussian_noise, jpeg_compress

from app.core.errors import InvalidImageError
from app.services.analysis import model_ref, model_warnings
from app.services.inference import predict_image
from app.services.model_registry import ModelRegistry

PREVIEW_MAX_SIDE = 512


def _data_url(img: Image.Image) -> str:
    im = img.copy()
    im.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def run_robustness(registry: ModelRegistry, data: bytes, track: str, params: dict[str, Any], amp: bool) -> dict[str, Any]:
    lm = registry.require_active(track)
    try:
        img = load_image_rgb(data)
    except ValueError as exc:
        raise InvalidImageError(str(exc)) from exc

    steps: list[tuple[str, dict[str, Any], Any]] = []
    if params.get("jpeg_quality") is not None:
        q = int(params["jpeg_quality"])
        steps.append((f"JPEG quality {q}", {"type": "jpeg", "quality": q}, lambda im, q=q: jpeg_compress(im, q)))
    if params.get("downscale_max_side") is not None:
        s = int(params["downscale_max_side"])
        steps.append((f"Downscale to {s}px (longer side), restored", {"type": "downscale", "max_side": s}, lambda im, s=s: downscale(im, max_side=s, restore=True)))
    if params.get("blur_sigma") is not None:
        b = float(params["blur_sigma"])
        steps.append((f"Gaussian blur sigma {b:g}", {"type": "blur", "sigma": b}, lambda im, b=b: gaussian_blur(im, b)))
    if params.get("noise_std") is not None:
        n = float(params["noise_std"])
        steps.append((f"Gaussian noise std {n:g}", {"type": "noise", "std": n, "seed": 0}, lambda im, n=n: gaussian_noise(im, n, seed=0)))

    base = predict_image(registry, lm, img, amp)

    def _entry(name: str, spec: list[dict[str, Any]], im: Image.Image) -> dict[str, Any]:
        p = predict_image(registry, lm, im, amp)
        return {
            "name": name,
            "perturbations": spec,
            "prediction": p.label,
            "fake_score": p.fake_score,
            "delta_fake_score": p.fake_score - base.fake_score,
            "prediction_changed": p.label != base.label,
            "inference_ms": p.inference_ms,
            "image_data_url": _data_url(im),
        }

    variants = [_entry(name, [spec], fn(img)) for name, spec, fn in steps]
    if len(steps) > 1:
        combined = img
        for _, _, fn in steps:
            combined = fn(combined)
        variants.append(_entry("Combined (applied in the order listed)", [s for _, s, _ in steps], combined))

    return {
        "track": track,
        "track_label": TRACK_LABELS[track],
        "model": model_ref(lm.entry, lm.meta),
        "threshold": base.threshold,
        "threshold_source": base.threshold_source,
        "original": {"name": "Original", "perturbations": [], "prediction": base.label, "fake_score": base.fake_score, "delta_fake_score": 0.0, "prediction_changed": False, "inference_ms": base.inference_ms, "image_data_url": _data_url(img)},
        "variants": variants,
        "warnings": model_warnings(lm.meta),
        "note": "Single-image demonstration with the real model. It is not a benchmark evaluation; see the Model Evaluation page for dataset-level robustness results.",
    }
