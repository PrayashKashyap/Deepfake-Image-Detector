"""Grad-CAM (Selvaraju et al., ICCV 2017), implemented with forward/backward hooks.

Implemented in-house (instead of a third-party package) to keep dependencies minimal
and compatible with the pinned PyTorch version. Heatmaps are qualitative: they show
which regions most influenced the selected class logit at the target layer's
resolution; they do NOT prove that the model relies on meaningful forensic evidence.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None

    def _fwd_hook(self, _m, _inp, out) -> None:
        self._acts = out
        out.register_hook(self._save_grad)

    def _save_grad(self, grad: torch.Tensor) -> None:
        self._grads = grad

    def __call__(self, x: torch.Tensor, class_idx: int) -> tuple[np.ndarray, torch.Tensor]:
        """Return (cam [H, W] in [0, 1] at input resolution, logits [1, 2])."""
        if x.dim() != 4 or x.size(0) != 1:
            raise ValueError("GradCAM expects a single image batch [1, C, H, W]")
        handle = self.target_layer.register_forward_hook(self._fwd_hook)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.enable_grad():
                x = x.clone().requires_grad_(True)
                logits = self.model(x).float()
                self.model.zero_grad(set_to_none=True)
                logits[0, class_idx].backward()
            acts, grads = self._acts, self._grads
            if acts is None or grads is None:
                raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")
            weights = grads.float().mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * acts.float()).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            cam = cam.detach().cpu().numpy()
            rng = cam.max() - cam.min()
            cam = (cam - cam.min()) / rng if rng > 1e-12 else np.zeros_like(cam)
            return cam.astype(np.float32), logits.detach()
        finally:
            handle.remove()
            self._acts = self._grads = None
            self.model.train(was_training)


def heatmap_rgb(cam: np.ndarray) -> np.ndarray:
    hm = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap over an RGB uint8 image (cam is resized to the image size)."""
    h, w = rgb.shape[:2]
    if cam.shape != (h, w):
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    hm = heatmap_rgb(cam).astype(np.float32)
    out = (1 - alpha) * rgb.astype(np.float32) + alpha * hm
    return np.clip(out, 0, 255).astype(np.uint8)
