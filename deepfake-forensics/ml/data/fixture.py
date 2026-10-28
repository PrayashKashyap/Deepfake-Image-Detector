"""SYNTHETIC MINIATURE TEST FIXTURE — FOR SOFTWARE TESTING ONLY.

Generates tiny procedurally-drawn images for both tracks so that the data pipeline,
training loop, evaluation, API and UI can be exercised without the real datasets.
The "fake" class is given an artificial periodic (checkerboard) pattern, which is
trivially learnable. Any metric obtained on this fixture says NOTHING about deepfake
detection and must never be reported as a result. All produced manifests use
dataset_id values starting with "SYNTHETIC_FIXTURE".

    python -m ml.data.fixture --out data/fixtures/synthetic --n-per-class 24
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from ml.common import save_json
from ml.data.manifest import group_split, summarize_manifest, write_manifest

FIXTURE_PREFIX = "SYNTHETIC_FIXTURE"


def _base_image(rng: np.random.Generator, size: int) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    c0, c1 = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3)
    t = (xx * rng.uniform(0.2, 1.0) + yy * rng.uniform(0.2, 1.0))[..., None] / 2
    arr = c0 * (1 - t) + c1 * t + rng.normal(0, 6, (size, size, 3))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(2, 6))):
        x0, y0 = rng.integers(0, size - 8, 2)
        x1, y1 = x0 + rng.integers(8, size // 2), y0 + rng.integers(8, size // 2)
        col = tuple(int(v) for v in rng.integers(0, 255, 3))
        (d.ellipse if rng.random() < 0.5 else d.rectangle)([int(x0), int(y0), int(x1), int(y1)], fill=col)
    return img


def _add_fake_artifact(img: Image.Image, rng: np.random.Generator, period: int) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    yy, xx = np.mgrid[0 : arr.shape[0], 0 : arr.shape[1]]
    pattern = (((xx // period) + (yy // period)) % 2).astype(np.float32) * 2 - 1
    arr = arr + pattern[..., None] * rng.uniform(10, 18)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def make_fixture(out: Path, n_per_class: int = 24, size: int = 96, seed: int = 0) -> dict[str, Path]:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    manifests = {}

    # AI-generated track: two pseudo-"generators" with different artifact periods.
    rows = []
    for gen, period in (("fixture_gen_a", 2), ("fixture_gen_b", 4)):
        for i in range(n_per_class):
            for label in (0, 1):
                img = _base_image(rng, size)
                if label == 1:
                    img = _add_fake_artifact(img, rng, period)
                rel = Path("ai_generated") / ("fake" if label else "real") / gen / f"{gen}_{label}_{i:04d}.png"
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                img.save(out / rel)
                rows.append({"path": rel.as_posix(), "label": label, "source": gen if label else "real_" + gen, "group": rel.as_posix(), "identity": "", "video": ""})
    df = pd.DataFrame(rows)
    df["dataset_id"] = f"{FIXTURE_PREFIX}_ai_generated"
    df["split"] = group_split(df, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=seed)
    manifests["ai_generated"] = write_manifest(df, out / "ai_generated_manifest.csv")
    save_json(summarize_manifest(df), out / "ai_generated_manifest.summary.json")

    # Face track: pseudo-"videos" with several frames each; two pseudo manipulation methods.
    rows = []
    n_videos = max(4, n_per_class // 3)
    for v in range(n_videos):
        vid = f"{v:03d}"
        partner = f"{v ^ 1:03d}" if (v ^ 1) < n_videos else vid  # symmetric pairs, like FF++ (000_003 / 003_000)
        for f in range(3):
            img = _base_image(rng, size)
            rel = Path("face_deepfake") / "real" / "original" / vid / f"frame_{f:05d}.png"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            img.save(out / rel)
            rows.append({"path": rel.as_posix(), "label": 0, "source": "original", "video": vid, "identity": vid})
            method, period = ("fixture_swap", 2) if v % 2 == 0 else ("fixture_reenact", 3)
            fimg = _add_fake_artifact(_base_image(rng, size), rng, period)
            fvid = f"{vid}_{partner}"
            rel = Path("face_deepfake") / "fake" / method / fvid / f"frame_{f:05d}.png"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            fimg.save(out / rel)
            rows.append({"path": rel.as_posix(), "label": 1, "source": method, "video": fvid, "identity": vid})
    df = pd.DataFrame(rows)
    df["group"] = "g" + df["identity"]
    df["dataset_id"] = f"{FIXTURE_PREFIX}_face_deepfake"
    df["split"] = group_split(df, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=seed)
    manifests["face_deepfake"] = write_manifest(df, out / "face_deepfake_manifest.csv")
    save_json(summarize_manifest(df), out / "face_deepfake_manifest.summary.json")
    (out / "README_SYNTHETIC_FIXTURE.txt").write_text(__doc__ or "", encoding="utf-8")
    return manifests


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/fixtures/synthetic"))
    ap.add_argument("--n-per-class", type=int, default=24)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    m = make_fixture(args.out, args.n_per_class, args.size, args.seed)
    for k, v in m.items():
        print(f"{k}: {v}")
    print("NOTE: synthetic fixture for software testing only — not evidence of model performance.")


if __name__ == "__main__":
    main()
