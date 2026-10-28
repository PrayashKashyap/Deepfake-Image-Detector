"""Image perturbations for robustness experiments and the robustness laboratory.

All functions take and return RGB PIL images. They are deterministic given their
parameters (noise uses an explicit seed).
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    if not 1 <= int(quality) <= 100:
        raise ValueError("JPEG quality must be in [1, 100]")
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality), subsampling=2 if quality < 90 else 0)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert("RGB")


def downscale(img: Image.Image, max_side: int | None = None, factor: float | None = None, restore: bool = True) -> Image.Image:
    """Reduce resolution. Either `max_side` (pixels of the longer side) or `factor` (0-1].

    With `restore=True` the image is resized back to its original size (information is
    lost but the geometry is unchanged), which isolates the effect of resolution loss.
    """
    w, h = img.size
    if factor is None and max_side is None:
        raise ValueError("Provide max_side or factor")
    if factor is not None:
        if not 0 < factor <= 1:
            raise ValueError("factor must be in (0, 1]")
        nw, nh = max(1, round(w * factor)), max(1, round(h * factor))
    else:
        if int(max_side) < 16:
            raise ValueError("max_side must be >= 16")
        scale = min(1.0, int(max_side) / max(w, h))
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    small = img.resize((nw, nh), Image.BICUBIC)
    if restore and (nw, nh) != (w, h):
        return small.resize((w, h), Image.BICUBIC)
    return small


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if sigma == 0:
        return img.copy()
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def gaussian_noise(img: Image.Image, std: float, seed: int = 0) -> Image.Image:
    """Add zero-mean Gaussian noise; `std` is on the 0-255 intensity scale."""
    if std < 0:
        raise ValueError("std must be >= 0")
    if std == 0:
        return img.copy()
    rng = np.random.default_rng(seed)
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr = arr + rng.normal(0.0, float(std), size=arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).round().astype(np.uint8), mode="RGB")


@dataclass
class Perturbation:
    """A named perturbation spec, e.g. {"type": "jpeg", "quality": 70}."""

    type: str
    params: dict[str, Any]

    @property
    def name(self) -> str:
        if self.type == "clean":
            return "clean"
        inner = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.type}({inner})"

    def apply(self, img: Image.Image, seed: int = 0) -> Image.Image:
        return apply_perturbation(img, {"type": self.type, **self.params}, seed=seed)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Perturbation":
        d = dict(d)
        t = d.pop("type")
        if t not in PERTURBATION_TYPES:
            raise ValueError(f"Unknown perturbation type {t!r}; expected one of {PERTURBATION_TYPES}")
        return cls(type=t, params=d)


PERTURBATION_TYPES = ("clean", "jpeg", "downscale", "blur", "noise")


def apply_perturbation(img: Image.Image, spec: dict[str, Any], seed: int = 0) -> Image.Image:
    t = spec.get("type")
    if t == "clean":
        return img
    if t == "jpeg":
        return jpeg_compress(img, int(spec["quality"]))
    if t == "downscale":
        return downscale(img, max_side=spec.get("max_side"), factor=spec.get("factor"), restore=bool(spec.get("restore", True)))
    if t == "blur":
        return gaussian_blur(img, float(spec["sigma"]))
    if t == "noise":
        return gaussian_noise(img, float(spec["std"]), seed=int(spec.get("seed", seed)))
    raise ValueError(f"Unknown perturbation type {t!r}")


DEFAULT_ROBUSTNESS_SUITE: list[dict[str, Any]] = [
    {"type": "clean"},
    {"type": "jpeg", "quality": 90},
    {"type": "jpeg", "quality": 70},
    {"type": "jpeg", "quality": 50},
    {"type": "downscale", "factor": 0.5},
    {"type": "downscale", "factor": 0.25},
    {"type": "blur", "sigma": 1.0},
    {"type": "blur", "sigma": 2.0},
    {"type": "noise", "std": 5.0},
    {"type": "noise", "std": 15.0},
]
