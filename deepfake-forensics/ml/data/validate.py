"""Validate a manifest: labels, missing/corrupted files, duplicates, near-duplicates, balance.

Near-duplicates use a 64-bit difference hash (dHash). Candidate pairs are found with
8-band locality-sensitive bucketing: two hashes within Hamming distance <= 7 must share
at least one identical byte (pigeonhole), so all such pairs are found exactly.
Cross-split duplicates are reported separately because they indicate train/test leakage.

Example (PowerShell):
    python -m ml.data.validate --manifest data/manifests/genimage_v1.csv --root D:\\datasets\\GenImage `
        --out reports/data_validation/genimage_v1.json --near-duplicates --hamming 4
"""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from ml.common import save_json, utc_now_iso
from ml.data.manifest import check_group_leakage, read_manifest, summarize_manifest


def dhash(img: Image.Image, size: int = 8) -> int:
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _check_file(path: Path) -> tuple[str | None, str | None, int | None]:
    """Return (error, sha256, dhash)."""
    if not path.is_file():
        return "missing", None, None
    try:
        data = path.read_bytes()
        with Image.open(path) as im:
            im.verify()  # structural check
        with Image.open(path) as im:
            im.load()  # full decode (verify() does not decode pixel data)
            h = dhash(im)
        return None, hashlib.sha256(data).hexdigest(), h
    except Exception as exc:  # noqa: BLE001 - any decode failure is a corruption finding
        return f"corrupted: {type(exc).__name__}: {exc}", None, None


def near_duplicate_scan(
    hashes: np.ndarray,
    max_dist: int,
    splits: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    collect: bool = False,
    max_examples: int = 200,
    block: int = 1024,
) -> dict[str, Any]:
    """Exact search for all pairs with Hamming distance <= max_dist among 64-bit hashes.

    8-band bucketing (pigeonhole: <= 7 differing bits implies >= 1 identical byte). A pair is
    counted only in the FIRST band where its bytes agree, so no global "seen" set is needed;
    distances are computed blockwise with NumPy, keeping memory bounded for dense buckets.
    Pairs inside the same leakage group are counted separately (expected, e.g. a real face
    and its manipulations) from cross-group and cross-split pairs (potential leakage).
    """
    if max_dist > 7:
        raise ValueError("max_dist must be <= 7 for exact 8-band LSH")
    h = np.ascontiguousarray(hashes, dtype=np.uint64)
    total = cross_split = same_group = 0
    examples: list[tuple[int, int, int]] = []
    pairs: list[tuple[int, int, int]] = []
    for band in range(8):
        keys = ((h >> np.uint64(8 * band)) & np.uint64(0xFF)).astype(np.int64)
        order = np.argsort(keys, kind="stable")
        bounds = np.flatnonzero(np.diff(keys[order])) + 1
        for members in np.split(order, bounds):
            if len(members) < 2:
                continue
            hm = h[members]
            for r0 in range(0, len(members), block):
                rows = np.arange(r0, min(r0 + block, len(members)))
                x = hm[rows][:, None] ^ hm[None, :]
                d = np.bitwise_count(x)
                ok = (d <= max_dist) & (np.arange(len(members))[None, :] > rows[:, None])
                if band > 0 and ok.any():
                    # drop pairs already counted in an earlier band (an earlier byte is identical)
                    for bb in range(band):
                        ok &= ((x >> np.uint64(8 * bb)) & np.uint64(0xFF)) != 0
                ri, ci = np.nonzero(ok)
                if len(ri) == 0:
                    continue
                a, b, dd = members[rows[ri]], members[ci], d[ri, ci]
                total += len(a)
                sg = groups[a] == groups[b] if groups is not None else np.zeros(len(a), bool)
                same_group += int(sg.sum())
                if splits is not None:
                    cs = (splits[a] != splits[b]) & ~sg
                    cross_split += int(cs.sum())
                    for k in np.flatnonzero(cs)[: max(0, max_examples - len(examples))]:
                        examples.append((int(a[k]), int(b[k]), int(dd[k])))
                if collect:
                    pairs.extend((int(min(i, j)), int(max(i, j)), int(k)) for i, j, k in zip(a, b, dd))
    return {"n_pairs": total, "n_same_group": same_group, "n_cross_split": cross_split, "examples": examples, "pairs": pairs}


def near_duplicate_pairs(hashes: dict[int, int], max_dist: int) -> list[tuple[int, int, int]]:
    """Small-input convenience wrapper returning all (i, j, distance) pairs."""
    ids = np.array(sorted(hashes), dtype=np.int64)
    arr = np.array([hashes[i] for i in ids], dtype=np.uint64)
    res = near_duplicate_scan(arr, max_dist, collect=True)
    return [(int(ids[a]), int(ids[b]), d) for a, b, d in res["pairs"]]


def validate(df: pd.DataFrame, root: Path, near_dups: bool = True, hamming: int = 4, max_report: int = 200) -> dict[str, Any]:
    report: dict[str, Any] = {"created_at": utc_now_iso(), "n_rows": int(len(df))}
    bad_labels = df[~df["label"].isin([0, 1])]
    bad_splits = df[~df["split"].isin(["train", "val", "test"])]
    report["invalid_labels"] = bad_labels["path"].head(max_report).tolist()
    report["invalid_splits"] = bad_splits["path"].head(max_report).tolist()
    report["duplicate_manifest_rows"] = df[df["path"].duplicated()]["path"].head(max_report).tolist()

    errors, shas, hashes = {}, {}, {}
    for i, rel in enumerate(df["path"]):
        err, sha, h = _check_file(root / rel)
        if err:
            errors[rel] = err
        else:
            shas[i], hashes[i] = sha, h
        if (i + 1) % 2000 == 0:
            print(f"  checked {i + 1}/{len(df)}")
    report["file_errors"] = dict(list(errors.items())[:max_report])
    report["n_missing"] = sum(1 for e in errors.values() if e == "missing")
    report["n_corrupted"] = sum(1 for e in errors.values() if e != "missing")

    by_sha: dict[str, list[int]] = defaultdict(list)
    for i, s in shas.items():
        by_sha[s].append(i)
    exact = [ids for ids in by_sha.values() if len(ids) > 1]
    label_conflicts = [ids for ids in exact if df.loc[ids, "label"].nunique() > 1]
    cross_split_exact = [ids for ids in exact if df.loc[ids, "split"].nunique() > 1]
    report["exact_duplicate_sets"] = len(exact)
    report["exact_duplicates_label_conflict"] = [df.loc[ids, "path"].tolist() for ids in label_conflicts[:max_report]]
    report["exact_duplicates_cross_split"] = [df.loc[ids, ["path", "split"]].to_dict("records") for ids in cross_split_exact[:max_report]]

    if near_dups:
        idx = np.array(sorted(hashes), dtype=np.int64)
        res = near_duplicate_scan(
            np.array([hashes[i] for i in idx], dtype=np.uint64), hamming,
            splits=df["split"].to_numpy()[idx], groups=df["group"].to_numpy()[idx], max_examples=max_report,
        )
        report["near_duplicate_pairs"] = res["n_pairs"]
        report["near_duplicate_same_group_pairs"] = res["n_same_group"]
        report["near_duplicate_cross_split_pairs"] = res["n_cross_split"]
        report["near_duplicate_cross_split_examples"] = [
            {"a": df.at[int(idx[a]), "path"], "a_split": df.at[int(idx[a]), "split"], "b": df.at[int(idx[b]), "path"], "b_split": df.at[int(idx[b]), "split"], "hamming": d}
            for a, b, d in res["examples"]
        ]
        report["near_duplicate_hamming_threshold"] = hamming

    report["group_leakage"] = check_group_leakage(df)[:max_report]
    report["balance"] = summarize_manifest(df)
    report["ok"] = not (
        report["invalid_labels"] or report["invalid_splits"] or errors or label_conflicts or report["group_leakage"] or cross_split_exact
    )
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--near-duplicates", action="store_true")
    ap.add_argument("--hamming", type=int, default=4)
    args = ap.parse_args(argv)
    df = read_manifest(args.manifest)
    rep = validate(df, args.root.resolve(), near_dups=args.near_duplicates, hamming=args.hamming)
    rep["manifest"] = args.manifest.as_posix()
    out = args.out or args.manifest.with_suffix(".validation.json")
    save_json(rep, out)
    print(f"Rows: {rep['n_rows']}  missing: {rep['n_missing']}  corrupted: {rep['n_corrupted']}")
    print(f"Exact duplicate sets: {rep['exact_duplicate_sets']} (cross-split: {len(rep['exact_duplicates_cross_split'])})")
    if args.near_duplicates:
        print(f"Near-duplicate pairs (<= {args.hamming} bits): {rep['near_duplicate_pairs']} (cross-split: {rep['near_duplicate_cross_split_pairs']})")
    for split, s in rep["balance"]["splits"].items():
        print(f"  {split:5s} real={s['real']:7d} fake={s['fake']:7d}")
    print(f"Overall OK: {rep['ok']}  -> report: {out}")


if __name__ == "__main__":
    main()
