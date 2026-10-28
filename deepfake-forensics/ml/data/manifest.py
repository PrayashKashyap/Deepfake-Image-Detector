"""Dataset manifests: one CSV row per image.

Columns
    path        image path relative to the dataset root (POSIX separators)
    label       0 = REAL, 1 = FAKE
    split       train | val | test
    source      generator name (AI-generated track) or manipulation method / "original" (face track)
    group       leakage group; all rows sharing a group are always placed in the same split
                (e.g. a source video and every manipulated video derived from it)
    identity    subject identity / video id when known, else empty
    video       source video id for frames, else empty
    dataset_id  short dataset identifier, e.g. "genimage" or "ffpp_c23"

Experiments that need cross-source evaluation select rows with `include_sources` /
`exclude_sources` filters instead of creating ad-hoc splits, so the group-level split
remains the single source of truth and test sources can never leak into training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ml.common import sha256_file

MANIFEST_COLUMNS = ["path", "label", "split", "source", "group", "identity", "video", "dataset_id"]
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"path": str, "split": str, "source": str, "group": str, "identity": str, "video": str, "dataset_id": str}, keep_default_na=False)
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {missing}")
    df["label"] = df["label"].astype(int)
    return df[MANIFEST_COLUMNS].copy()


def write_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for c in MANIFEST_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[MANIFEST_COLUMNS].sort_values(["split", "source", "label", "path"]).reset_index(drop=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return path


def manifest_fingerprint(path: str | Path) -> str:
    return sha256_file(path)[:16]


def filter_manifest(
    df: pd.DataFrame,
    splits: Sequence[str] | None = None,
    include_sources: Sequence[str] | None = None,
    exclude_sources: Sequence[str] | None = None,
    max_per_class: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    out = df
    if splits:
        out = out[out["split"].isin(list(splits))]
    if include_sources:
        out = out[out["source"].isin(list(include_sources))]
    if exclude_sources:
        out = out[~out["source"].isin(list(exclude_sources))]
    if max_per_class:
        out = out.sample(frac=1.0, random_state=seed).groupby("label", sort=False).head(int(max_per_class))
        out = out.sort_index()
    return out.reset_index(drop=True)


def group_split(
    df: pd.DataFrame,
    fractions: dict[str, float],
    seed: int = 42,
    group_col: str = "group",
) -> pd.Series:
    """Assign every group to a split so that the per-split sample counts approach `fractions`.

    Groups are shuffled with a fixed seed, then assigned greedily to the split with the
    largest remaining deficit. Rows in one group never end up in different splits.
    Stratification is approximate: it is performed separately for each label of a group's
    majority class, which keeps class balance close to the global ratio.
    """
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {fractions}")
    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    gstats = df.groupby(group_col).agg(n=("label", "size"), fake=("label", "mean"))
    gstats["stratum"] = (gstats["fake"] >= 0.5).astype(int)
    names = list(fractions)
    for _, sub in gstats.groupby("stratum"):
        groups = sub.index.to_numpy().copy()
        rng.shuffle(groups)
        total = float(sub["n"].sum())
        filled = {s: 0.0 for s in names}
        for g in groups:
            deficits = {s: fractions[s] * total - filled[s] for s in names}
            best = max(names, key=lambda s: (deficits[s], -names.index(s)))
            assignment[g] = best
            filled[best] += float(sub.at[g, "n"])
    return df[group_col].map(assignment)


def balanced_subset(df: pd.DataFrame, per_key: int, keys: Iterable[str], seed: int = 42) -> pd.DataFrame:
    """Keep at most `per_key` rows per combination of `keys` (sampled with a fixed seed)."""
    keys = list(keys)
    return df.sample(frac=1.0, random_state=seed).groupby(keys, sort=False).head(int(per_key)).sort_index().reset_index(drop=True)


def summarize_manifest(df: pd.DataFrame) -> dict[str, Any]:
    by_split = {}
    for split, sub in df.groupby("split"):
        by_split[split] = {
            "n": int(len(sub)),
            "real": int((sub["label"] == 0).sum()),
            "fake": int((sub["label"] == 1).sum()),
            "groups": int(sub["group"].nunique()),
            "sources": {str(k): int(v) for k, v in sub["source"].value_counts().sort_index().items()},
        }
    return {
        "n": int(len(df)),
        "real": int((df["label"] == 0).sum()),
        "fake": int((df["label"] == 1).sum()),
        "sources": sorted(df["source"].unique().tolist()),
        "dataset_ids": sorted(df["dataset_id"].unique().tolist()),
        "splits": by_split,
    }


def check_group_leakage(df: pd.DataFrame, group_col: str = "group") -> list[str]:
    """Return groups that appear in more than one split (should be empty)."""
    counts = df.groupby(group_col)["split"].nunique()
    return counts[counts > 1].index.tolist()


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
