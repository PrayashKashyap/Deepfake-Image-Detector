"""Evaluation figures (PNG)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_confusion_matrix(cm: Sequence[Sequence[int]], path: Path, title: str = "Confusion matrix") -> Path:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=130)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["REAL", "FAKE"])
    ax.set_yticks([0, 1], ["REAL", "FAKE"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_roc(curves: dict[str, dict[str, Any]], path: Path, title: str = "ROC curve") -> Path:
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=130)
    for name, c in curves.items():
        if not c:
            continue
        auc = c.get("auc")
        ax.plot(c["fpr"], c["tpr"], label=f"{name} (AUC={auc:.3f})" if auc is not None else name)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate (REAL flagged as FAKE)")
    ax.set_ylabel("True positive rate (FAKE detected)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_bar(labels: list[str], series: dict[str, list[float | None]], path: Path, title: str, ylabel: str, errors: dict[str, list[float | None]] | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(labels) + 2), 4), dpi=130)
    x = np.arange(len(labels))
    w = 0.8 / max(1, len(series))
    for k, (name, vals) in enumerate(series.items()):
        v = np.array([np.nan if a is None else a for a in vals], dtype=float)
        err = None
        if errors and name in errors:
            err = np.array([0 if a is None else a for a in errors[name]], dtype=float)
        ax.bar(x + k * w - 0.4 + w / 2, v, w, label=name, yerr=err, capsize=3 if err is not None else 0)
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
