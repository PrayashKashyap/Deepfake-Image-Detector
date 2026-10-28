"""Build a manifest for the AI-generated-image track.

Supported layouts
-----------------
genimage   GenImage (Zhu et al., NeurIPS 2023 D&B). Expected structure after extraction:
               <root>/<GeneratorDir>/**/{train,val}/{ai,nature}/<images>
           `source` = the top-level generator directory (e.g. "Midjourney", "stable_diffusion_v_1_4").
           GenImage "val" images become the `test` split; GenImage "train" images are split
           into `train`/`val` with a fixed seed.
folder     Generic real/fake folder:
               <root>/{real,fake}/[<source>/]**/<images>
           `source` = first sub-directory below real/ or fake/ (or "real"/"unknown").
           train/val/test are created with a seeded group split.

Cross-generator experiments do not need a special split: train with
`exclude_sources: [X]` and evaluate with `include_sources: [X]` (see ml/configs).

Example (PowerShell):
    python -m ml.data.genimage --root D:\\datasets\\GenImage --out data/manifests/genimage_v1.csv `
        --layout genimage --max-per-class-per-source 2000 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml.common import save_json
from ml.data.manifest import (
    balanced_subset,
    check_group_leakage,
    group_split,
    list_images,
    summarize_manifest,
    write_manifest,
)

REAL_DIRS = {"nature", "real", "0_real"}
FAKE_DIRS = {"ai", "fake", "1_fake"}


def scan_genimage(root: Path) -> pd.DataFrame:
    rows = []
    for p in list_images(root):
        rel = p.relative_to(root)
        parts = [s.lower() for s in rel.parts[:-1]]
        if len(rel.parts) < 3:
            continue
        label = None
        for part in reversed(parts):
            if part in REAL_DIRS:
                label = 0
                break
            if part in FAKE_DIRS:
                label = 1
                break
        if label is None:
            continue
        orig_split = "val" if "val" in parts else ("train" if "train" in parts else "")
        # Generator = first directory that is not a split name (handles <gen>/... and test/<gen>/...).
        source = next((q for q in rel.parts[:-1] if q.lower() not in ("train", "val", "test")), rel.parts[0])
        rows.append({"path": rel.as_posix(), "label": label, "source": source, "orig_split": orig_split})
    return pd.DataFrame(rows)


def scan_folder(root: Path) -> pd.DataFrame:
    rows = []
    for top, label in (("real", 0), ("fake", 1)):
        base = root / top
        if not base.is_dir():
            continue
        for p in list_images(base):
            rel_inner = p.relative_to(base)
            source = rel_inner.parts[0] if len(rel_inner.parts) > 1 else ("real" if label == 0 else "unknown")
            rows.append({"path": p.relative_to(root).as_posix(), "label": label, "source": source, "orig_split": ""})
    return pd.DataFrame(rows)


def build_manifest(
    root: Path,
    layout: str,
    dataset_id: str,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.15,
    max_per_class_per_source: int | None = None,
    merge_near_duplicates: int | None = None,
) -> pd.DataFrame:
    df = scan_genimage(root) if layout == "genimage" else scan_folder(root)
    if df.empty:
        raise SystemExit(f"No labelled images found under {root} for layout={layout!r}")
    if max_per_class_per_source:
        df = balanced_subset(df, max_per_class_per_source, keys=["source", "label", "orig_split"], seed=seed)
    # Each image is its own leakage group here; exact/near duplicates are detected by ml.data.validate.
    df["group"] = df["path"]
    df["identity"] = ""
    df["video"] = ""
    df["dataset_id"] = dataset_id
    if merge_near_duplicates is not None:
        from ml.data.dedup import merge_duplicate_groups

        df, stats = merge_duplicate_groups(df, root, merge_near_duplicates)
        df.attrs["dedup"] = stats
        print(f"  dedup: {stats}")
    if layout == "genimage" and (df["orig_split"] == "val").any():
        df["split"] = "test"
        tr = df["orig_split"] != "val"
        df.loc[tr, "split"] = group_split(df[tr], {"train": 1 - val_frac, "val": val_frac}, seed=seed)
    else:
        train_frac = 1.0 - val_frac - test_frac
        df["split"] = group_split(df, {"train": train_frac, "val": val_frac, "test": test_frac}, seed=seed)
    return df.drop(columns=["orig_split"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--layout", choices=["genimage", "folder"], default="genimage")
    ap.add_argument("--dataset-id", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.15, help="Only used for the folder layout")
    ap.add_argument("--max-per-class-per-source", type=int, default=None, help="Balanced subset size for rapid experiments")
    ap.add_argument("--merge-near-duplicates", type=int, default=None, metavar="HAMMING",
                    help="Merge groups containing images within this dHash distance (e.g. 2) before splitting")
    args = ap.parse_args(argv)

    root = args.root.resolve()
    df = build_manifest(
        root,
        args.layout,
        args.dataset_id or ("genimage" if args.layout == "genimage" else root.name),
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        max_per_class_per_source=args.max_per_class_per_source,
        merge_near_duplicates=args.merge_near_duplicates,
    )
    leaks = check_group_leakage(df)
    if leaks:
        raise SystemExit(f"Internal error: {len(leaks)} groups span multiple splits")
    write_manifest(df, args.out)
    summary = summarize_manifest(df)
    summary["dedup"] = df.attrs.get("dedup")
    summary["build_args"] = {k: (v.as_posix() if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "root"}
    save_json(summary, args.out.with_suffix(".summary.json"))
    print(f"Wrote {len(df)} rows to {args.out}")
    for split, s in summary["splits"].items():
        print(f"  {split:5s} n={s['n']:7d} real={s['real']:7d} fake={s['fake']:7d} sources={len(s['sources'])}")


if __name__ == "__main__":
    main()
