"""Merge leakage groups that contain (near-)duplicate images, before splitting.

If two images with different `group` keys are perceptually near-identical (dHash Hamming
distance <= threshold; exact byte duplicates have distance 0), their groups are merged with
union-find, so duplicates can never be placed in different splits.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ml.data.validate import dhash, near_duplicate_scan


def _hash(path: Path) -> int | None:
    try:
        with Image.open(path) as im:
            im.draft("L", (64, 64))  # fast JPEG decode at reduced scale; dHash uses 9x8 anyway
            return dhash(im)
    except Exception:  # noqa: BLE001 - unreadable files are reported by ml.data.validate
        return None


def merge_duplicate_groups(df: pd.DataFrame, root: Path, max_dist: int = 2, workers: int = 12) -> tuple[pd.DataFrame, dict]:
    with ThreadPoolExecutor(workers) as ex:
        hashes = list(ex.map(_hash, (root / p for p in df["path"])))
    ok = np.array([h is not None for h in hashes])
    idx = np.flatnonzero(ok)
    arr = np.array([hashes[i] for i in idx], dtype=np.uint64)
    res = near_duplicate_scan(arr, max_dist, collect=True)

    parent: dict[str, str] = {}

    def find(a: str) -> str:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    groups = df["group"].to_numpy()
    merges = 0
    for a, b, _ in res["pairs"]:
        ga, gb = find(groups[idx[a]]), find(groups[idx[b]])
        if ga != gb:
            parent[max(ga, gb)] = min(ga, gb)
            merges += 1
    out = df.copy()
    out["group"] = [find(g) for g in groups]
    sizes = out["group"].value_counts()
    stats = {
        "dhash_threshold": max_dist,
        "near_duplicate_pairs": res["n_pairs"],
        "group_merges": merges,
        "groups_before": int(df["group"].nunique()),
        "groups_after": int(out["group"].nunique()),
        "largest_group_rows": int(sizes.iloc[0]),
        "unhashable_files": int((~ok).sum()),
    }
    return out, stats
