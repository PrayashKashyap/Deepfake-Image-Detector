"""Image decoding and preprocessing shared by training, evaluation and API inference.

The exact preprocessing configuration is stored inside every checkpoint so that the
backend reproduces the evaluation pipeline bit-for-bit (see `build_eval_transform`).
"""
from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageFile

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Hard cap against decompression bombs. The backend applies a stricter configurable limit.
Image.MAX_IMAGE_PIXELS = 100_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False

RESIZE_MODES = ("resize_center_crop", "resize")


@dataclass
class PreprocessConfig:
    img_size: int = 224
    resize_mode: str = "resize_center_crop"  # or "resize" (aspect ratio not preserved)
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    interpolation: str = "bilinear"
    # Optional bias-equalising source normalisation, applied identically to every image (both
    # classes; training, evaluation and API inference) BEFORE the model resize: resize the short
    # side to `normalize_short_side`, centre-crop square, then re-encode as JPEG. Used as a control
    # against dataset shortcuts (e.g. all fakes 512x512 PNG/JPEG vs. variable-size real JPEGs).
    normalize_short_side: int | None = None
    normalize_jpeg_quality: int | None = None

    def __post_init__(self) -> None:
        if self.resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {RESIZE_MODES}, got {self.resize_mode!r}")
        if not (32 <= int(self.img_size) <= 1024):
            raise ValueError("img_size must be between 32 and 1024")
        self.img_size = int(self.img_size)
        if self.normalize_short_side is not None and not (32 <= int(self.normalize_short_side) <= 2048):
            raise ValueError("normalize_short_side must be in [32, 2048]")
        if self.normalize_jpeg_quality is not None and not (1 <= int(self.normalize_jpeg_quality) <= 100):
            raise ValueError("normalize_jpeg_quality must be in [1, 100]")
        self.mean = tuple(float(v) for v in self.mean)  # type: ignore[assignment]
        self.std = tuple(float(v) for v in self.std)  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mean"] = list(self.mean)
        d["std"] = list(self.std)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "PreprocessConfig":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


_INTERP = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "nearest": Image.NEAREST}


def load_image_rgb(source: str | Path | bytes) -> Image.Image:
    """Decode an image safely and convert to RGB. Raises ValueError on invalid data."""
    try:
        if isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        else:
            img = Image.open(source)
        img.load()
    except (OSError, Image.DecompressionBombError, SyntaxError) as exc:  # PIL raises several types
        raise ValueError(f"Could not decode image: {exc}") from exc
    if img.mode != "RGB":
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            # Composite transparency onto black for determinism.
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (0, 0, 0))
            bg.paste(rgba, mask=rgba.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
    return img


def resize_for_model(img: Image.Image, cfg: PreprocessConfig) -> Image.Image:
    interp = _INTERP.get(cfg.interpolation, Image.BILINEAR)
    s = cfg.img_size
    if cfg.resize_mode == "resize":
        return img.resize((s, s), interp)
    w, h = img.size
    scale = s / min(w, h)
    nw, nh = max(s, round(w * scale)), max(s, round(h * scale))
    img = img.resize((nw, nh), interp)
    left = (nw - s) // 2
    top = (nh - s) // 2
    return img.crop((left, top, left + s, top + s))


def normalize_source(img: Image.Image, cfg: PreprocessConfig) -> Image.Image:
    """Bias-equalising normalisation (no-op unless configured). See PreprocessConfig."""
    if cfg.normalize_short_side:
        s = int(cfg.normalize_short_side)
        w, h = img.size
        scale = s / min(w, h)
        nw, nh = max(s, round(w * scale)), max(s, round(h * scale))
        img = img.resize((nw, nh), Image.BICUBIC)
        left, top = (nw - s) // 2, (nh - s) // 2
        img = img.crop((left, top, left + s, top + s))
    if cfg.normalize_jpeg_quality:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(cfg.normalize_jpeg_quality))
        buf.seek(0)
        img = Image.open(buf)
        img.load()
        img = img.convert("RGB")
    return img


def to_normalized_tensor(img: Image.Image, cfg: PreprocessConfig) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0  # H, W, 3
    arr = (arr - np.asarray(cfg.mean, dtype=np.float32)) / np.asarray(cfg.std, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


class EvalTransform:
    """Deterministic preprocessing used for validation, evaluation AND API inference.
    (A class rather than a closure so it can be pickled into DataLoader workers.)"""

    def __init__(self, cfg: PreprocessConfig) -> None:
        self.cfg = cfg

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return to_normalized_tensor(resize_for_model(normalize_source(img, self.cfg), self.cfg), self.cfg)


def build_eval_transform(cfg: PreprocessConfig) -> Callable[[Image.Image], torch.Tensor]:
    return EvalTransform(cfg)


@dataclass
class AugmentConfig:
    hflip: bool = True
    random_crop_scale: tuple[float, float] | None = None  # e.g. (0.8, 1.0) -> RandomResizedCrop-like
    jpeg_prob: float = 0.0  # probability of random JPEG re-compression (robustness augmentation)
    jpeg_quality: tuple[int, int] = (60, 95)
    blur_prob: float = 0.0
    blur_sigma: tuple[float, float] = (0.1, 1.5)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "AugmentConfig":
        d = dict(d or {})
        out = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if out.random_crop_scale is not None:
            out.random_crop_scale = tuple(out.random_crop_scale)  # type: ignore[assignment]
        out.jpeg_quality = tuple(out.jpeg_quality)  # type: ignore[assignment]
        out.blur_sigma = tuple(out.blur_sigma)  # type: ignore[assignment]
        return out


class _ProcessLocalRng:
    """Lazily seeded RNG. DataLoader workers each get a distinct, reproducible seed
    (torch sets initial_seed = base_seed + worker_id), avoiding identical augmentations."""

    def __init__(self) -> None:
        self._pid: int | None = None
        self._rng: np.random.Generator | None = None

    def get(self) -> np.random.Generator:
        import os

        if self._rng is None or self._pid != os.getpid():
            info = torch.utils.data.get_worker_info()
            seed = (info.seed if info is not None else torch.initial_seed()) % (2**32)
            self._rng = np.random.default_rng(seed)
            self._pid = os.getpid()
        return self._rng


class TrainTransform:
    """Picklable training transform (required for Windows DataLoader workers)."""

    def __init__(self, cfg: PreprocessConfig, aug: AugmentConfig) -> None:
        self.cfg = cfg
        self.aug = aug
        self._rng = _ProcessLocalRng()

    def __call__(self, img: Image.Image) -> torch.Tensor:
        from ml.robustness.perturbations import gaussian_blur, jpeg_compress

        aug, cfg, rng = self.aug, self.cfg, self._rng.get()
        img = normalize_source(img, cfg)
        if aug.random_crop_scale is not None:
            lo, hi = aug.random_crop_scale
            w, h = img.size
            frac = float(rng.uniform(lo, hi))
            cw, ch = max(1, int(w * np.sqrt(frac))), max(1, int(h * np.sqrt(frac)))
            x0 = int(rng.integers(0, w - cw + 1))
            y0 = int(rng.integers(0, h - ch + 1))
            img = img.crop((x0, y0, x0 + cw, y0 + ch))
        if aug.blur_prob > 0 and rng.random() < aug.blur_prob:
            img = gaussian_blur(img, float(rng.uniform(*aug.blur_sigma)))
        if aug.jpeg_prob > 0 and rng.random() < aug.jpeg_prob:
            img = jpeg_compress(img, int(rng.integers(aug.jpeg_quality[0], aug.jpeg_quality[1] + 1)))
        img = resize_for_model(img, cfg)
        if aug.hflip and rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return to_normalized_tensor(img, cfg)


def build_train_transform(cfg: PreprocessConfig, aug: AugmentConfig) -> TrainTransform:
    """Training transform. Augmentations are applied *before* the shared eval resize."""
    return TrainTransform(cfg, aug)


def denormalize_to_uint8(tensor: torch.Tensor, cfg: PreprocessConfig) -> np.ndarray:
    """Convert a normalized CHW tensor back to an HWC uint8 RGB array (for overlays)."""
    arr = tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    arr = arr * np.asarray(cfg.std, dtype=np.float32) + np.asarray(cfg.mean, dtype=np.float32)
    return (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
