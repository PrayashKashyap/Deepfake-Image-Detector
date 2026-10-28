"""Binary classification metrics (scikit-learn based) with FAKE (=1) as the positive class."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

THRESHOLD_METHODS = ("youden", "balanced_accuracy", "fixed")


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def compute_binary_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)
    p, r, f, sup = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "n": int(len(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)) if len(y) else None,
        "balanced_accuracy": float(np.mean([r[c] for c in (0, 1) if sup[c] > 0])) if len(y) else None,
        "precision_real": float(p[0]),
        "recall_real": float(r[0]),
        "f1_real": float(f[0]),
        "precision_fake": float(p[1]),
        "recall_fake": float(r[1]),
        "f1_fake": float(f[1]),
        "f1_macro": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        "roc_auc": _safe_auc(y, s),
        "support_real": int(sup[0]),
        "support_fake": int(sup[1]),
        "confusion_matrix": cm.astype(int).tolist(),  # rows = true [REAL, FAKE], cols = predicted
    }


def per_class_table(m: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"class": "REAL", "precision": m["precision_real"], "recall": m["recall_real"], "f1": m["f1_real"], "support": m["support_real"]},
        {"class": "FAKE", "precision": m["precision_fake"], "recall": m["recall_fake"], "f1": m["f1_fake"], "support": m["support_fake"]},
    ]


def roc_points(labels: Sequence[int], scores: Sequence[float], max_points: int = 200) -> dict[str, list[float]] | None:
    y = np.asarray(labels, dtype=int)
    if len(np.unique(y)) < 2:
        return None
    fpr, tpr, thr = roc_curve(y, np.asarray(scores, dtype=float))
    if len(fpr) > max_points:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).round().astype(int))
        fpr, tpr, thr = fpr[idx], tpr[idx], thr[idx]
    thr = np.where(np.isfinite(thr), thr, 1.0)
    return {"fpr": fpr.round(6).tolist(), "tpr": tpr.round(6).tolist(), "thresholds": np.clip(thr, 0, 1).round(6).tolist()}


def select_threshold(labels: Sequence[int], scores: Sequence[float], method: str = "youden") -> float:
    """Choose a decision threshold. MUST be called on validation data only."""
    if method not in THRESHOLD_METHODS:
        raise ValueError(f"method must be one of {THRESHOLD_METHODS}")
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if method == "fixed" or len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, s)
    finite = np.isfinite(thr)
    fpr, tpr, thr = fpr[finite], tpr[finite], thr[finite]
    # Youden's J = TPR - FPR, which equals 2*balanced_accuracy - 1: both criteria coincide.
    j = tpr - fpr
    best = int(np.argmax(j))
    return float(np.clip(thr[best], 1e-6, 1 - 1e-6))


def grouped_bootstrap_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    groups: Sequence[str],
    threshold: float,
    metrics: Sequence[str] = ("accuracy", "f1_macro", "roc_auc"),
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict[str, float | None]]:
    """Percentile confidence intervals from a cluster bootstrap over leakage groups.

    Resampling whole groups (e.g. all frames of a video) respects within-group correlation,
    which an i.i.d. bootstrap over frames would ignore and thereby understate uncertainty.
    """
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    g = np.asarray(groups)
    uniq, inv = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])
        res = compute_binary_metrics(y[idx], s[idx], threshold)
        for m in metrics:
            if res.get(m) is not None:
                samples[m].append(res[m])
    out = {}
    for m, vals in samples.items():
        if len(vals) < max(10, n_boot // 10):
            out[m] = {"low": None, "high": None, "n_boot": len(vals)}
        else:
            lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            out[m] = {"low": float(lo), "high": float(hi), "n_boot": len(vals)}
    return {"method": "grouped_percentile_bootstrap", "alpha": alpha, "n_groups": int(len(uniq)), "metrics": out}
