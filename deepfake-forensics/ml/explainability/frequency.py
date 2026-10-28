"""Frequency-domain visualizations.

* Centered 2D Fourier log-magnitude spectrum of the luminance channel:
      S(u, v) = log(1 + |F{Y}(u, v)|), with F shifted so the DC component is at the centre.
* Azimuthally averaged power spectrum (1D profile over radial frequency).

Limitations: spectra depend strongly on resizing, compression, content and camera
pipeline. Periodic peaks can come from upsampling layers of generators but also from
JPEG 8x8 blocks, resampling, halftoning or textures. A spectrum pattern does not, on its
own, identify an image as fake.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _luma(img: Image.Image, max_side: int = 512) -> np.ndarray:
    g = img.convert("L")
    if max(g.size) > max_side:
        s = max_side / max(g.size)
        g = g.resize((max(1, round(g.width * s)), max(1, round(g.height * s))), Image.BICUBIC)
    return np.asarray(g, dtype=np.float32) / 255.0


def fourier_log_magnitude(img: Image.Image, max_side: int = 512) -> np.ndarray:
    y = _luma(img, max_side)
    y = y - y.mean()
    # Hann window reduces boundary discontinuity leakage (bright cross artifact).
    win = np.outer(np.hanning(y.shape[0]), np.hanning(y.shape[1])).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(y * win))
    return np.log1p(np.abs(f)).astype(np.float32)


def spectrum_image(img: Image.Image, max_side: int = 512) -> Image.Image:
    s = fourier_log_magnitude(img, max_side)
    lo, hi = np.percentile(s, [1, 99.8])
    s = np.clip((s - lo) / max(hi - lo, 1e-9), 0, 1)
    import cv2

    rgb = cv2.cvtColor(cv2.applyColorMap((s * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def radial_profile(img: Image.Image, n_bins: int = 64, max_side: int = 512) -> dict[str, list[float]]:
    s = fourier_log_magnitude(img, max_side)
    h, w = s.shape
    yy, xx = np.indices(s.shape)
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    r = np.clip(r / np.sqrt(2), 0, 1)
    bins = np.minimum((r * n_bins).astype(int), n_bins - 1)
    sums = np.bincount(bins.ravel(), weights=s.ravel(), minlength=n_bins)
    counts = np.maximum(np.bincount(bins.ravel(), minlength=n_bins), 1)
    prof = sums / counts
    return {"radius": [round(float(v), 4) for v in (np.arange(n_bins) + 0.5) / n_bins], "log_magnitude": [round(float(v), 5) for v in prof]}
