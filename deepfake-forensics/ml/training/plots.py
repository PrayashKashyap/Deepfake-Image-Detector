"""Training curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_history(history: list[dict], out_dir: Path) -> list[Path]:
    if not history:
        return []
    df = pd.DataFrame(history)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.plot(df["epoch"], df["train_loss"], marker="o", label="train loss")
    ax.plot(df["epoch"], df["val_loss"], marker="o", label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training and validation loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = out_dir / "loss_curve.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    for col, lab in (("val_auc", "val ROC-AUC"), ("val_f1_macro", "val macro-F1"), ("val_accuracy", "val accuracy")):
        if col in df and df[col].notna().any():
            ax.plot(df["epoch"], df[col], marker="o", label=lab)
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1.02)
    ax.set_title("Validation metrics (threshold 0.5 during training)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = out_dir / "val_metrics_curve.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)
    return paths
