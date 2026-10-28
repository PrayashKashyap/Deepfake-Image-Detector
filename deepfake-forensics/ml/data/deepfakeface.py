"""Manifest builder for the DeepFakeFace (DFF) dataset — facial-deepfake track.

DFF (Song et al., 2023, arXiv:2309.02218; https://huggingface.co/datasets/OpenRL/DeepFakeFace)
contains 30,000 real celebrity images from IMDB-WIKI and 30,000 fakes for each of three
diffusion-era methods, stored under the SAME relative path as the real image they derive from:

    <root>/wiki/<dd>/<id>_<dob>_<year>.jpg        REAL      (source = "wiki")
    <root>/inpainting/<dd>/<same name>.jpg         FAKE      Stable Diffusion inpainting
    <root>/insight/<dd>/<same name>.jpg            FAKE      InsightFace face swap
    <root>/text2img/<dd>/<same name>.jpg           FAKE      Stable Diffusion v1.5 text-to-image

Leakage control: the leading numeric IMDB-WIKI id is used as the leakage group, so a real
image and every fake derived from it always fall into the same split. No person is
identified; the id is only used as an opaque grouping key.

    python -m ml.data.deepfakeface --root C:\\Users\\me\\datasets\\DeepFakeFace --out data\\manifests\\dff_v1.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml.common import save_json
from ml.data.manifest import check_group_leakage, group_split, list_images, summarize_manifest, write_manifest

METHODS = {"wiki": 0, "inpainting": 1, "insight": 1, "text2img": 1}


def build(root: Path, dataset_id: str, seed: int, val_frac: float, test_frac: float, merge_near_duplicates: int | None = None) -> pd.DataFrame:
    rows = []
    for method, label in METHODS.items():
        base = root / method
        if not base.is_dir():
            print(f"  (missing {base}; skipped)")
            continue
        for p in list_images(base):
            rel = p.relative_to(root).as_posix()
            key = p.stem.split("_")[0]
            rows.append({"path": rel, "label": label, "source": method, "group": f"id{key}", "identity": key, "video": ""})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No DFF images found under {root}")
    df["dataset_id"] = dataset_id
    if merge_near_duplicates is not None:
        from ml.data.dedup import merge_duplicate_groups

        df, stats = merge_duplicate_groups(df, root, merge_near_duplicates)
        df.attrs["dedup"] = stats
        print(f"  dedup: {stats}")
    df["split"] = group_split(df, {"train": 1 - val_frac - test_frac, "val": val_frac, "test": test_frac}, seed=seed)
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset-id", default="deepfakeface")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--merge-near-duplicates", type=int, default=None, metavar="HAMMING",
                    help="Merge groups containing images within this dHash distance (e.g. 2) before splitting")
    args = ap.parse_args(argv)
    df = build(args.root.resolve(), args.dataset_id, args.seed, args.val_frac, args.test_frac, args.merge_near_duplicates)
    if check_group_leakage(df):
        raise SystemExit("Leakage check failed")
    write_manifest(df, args.out)
    summary = summarize_manifest(df)
    summary["dedup"] = df.attrs.get("dedup")
    save_json(summary, args.out.with_suffix(".summary.json"))
    print(f"Wrote {len(df)} rows to {args.out}")
    for split, s in summary["splits"].items():
        print(f"  {split:5s} n={s['n']:6d} real={s['real']:6d} fake={s['fake']:6d} groups={s['groups']} sources={s['sources']}")


if __name__ == "__main__":
    main()
