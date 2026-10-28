"""PyTorch dataset backed by a manifest DataFrame."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ml.preprocessing.transforms import load_image_rgb
from ml.robustness.perturbations import apply_perturbation


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root: str | Path,
        transform: Callable[[Image.Image], torch.Tensor],
        perturbation: dict[str, Any] | None = None,
        perturbation_seed: int = 0,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.perturbation = perturbation
        self.perturbation_seed = perturbation_seed
        self.paths = self.df["path"].tolist()
        self.labels = self.df["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        img = load_image_rgb(self.root / self.paths[idx])
        if self.perturbation and self.perturbation.get("type") != "clean":
            # Per-sample seed keeps noise reproducible across runs and workers.
            img = apply_perturbation(img, self.perturbation, seed=self.perturbation_seed + idx)
        return self.transform(img), self.labels[idx], idx
