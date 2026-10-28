"""Prepare the facial-deepfake track: frame extraction + leakage-aware manifest.

FaceForensics++ (Rössler et al., ICCV 2019) must be requested from its authors and
downloaded with their official script under their terms of use. This module never
downloads data. Expected structure (official download script):

    <root>/original_sequences/youtube/<compression>/videos/000.mp4
    <root>/manipulated_sequences/<Method>/<compression>/videos/000_003.mp4

A manipulated video `000_003` uses target video 000 and source video 003. Both
identities are linked with a union-find structure, and every connected component
becomes one leakage `group`, so frames of a person never appear in two splits.
If the official FF++ split files (train.json / val.json / test.json, lists of
[id_a, id_b] pairs) are supplied with --official-splits, they are used instead.

A generic pre-extracted layout is also supported:
    <root>/{real,fake}/<method>/<video_id>/<frames>
    (`real` frames use method directory name as-is, typically "original")

Examples (PowerShell):
    python -m ml.data.faceforensics extract --root D:\\datasets\\FFpp --out D:\\datasets\\ffpp_frames `
        --compression c23 --frames-per-video 16 --face-crop
    python -m ml.data.faceforensics manifest --frames-root D:\\datasets\\ffpp_frames `
        --out data/manifests/ffpp_c23.csv --official-splits D:\\datasets\\FFpp\\splits
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.common import save_json
from ml.data.manifest import check_group_leakage, group_split, list_images, summarize_manifest, write_manifest

FFPP_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


# ----------------------------------------------------------------------------- extraction
def _face_detector():
    import cv2

    path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    det = cv2.CascadeClassifier(str(path))
    if det.empty():
        raise RuntimeError(f"Could not load OpenCV Haar cascade from {path}")
    return det


def _crop_face(frame_bgr: np.ndarray, detector, margin: float = 1.3) -> np.ndarray | None:
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face; no identification is performed
    cx, cy, side = x + w / 2, y + h / 2, max(w, h) * margin
    H, W = frame_bgr.shape[:2]
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    return frame_bgr[y0:y1, x0:x1]


def extract_frames(video: Path, out_dir: Path, n_frames: int, face_crop: bool, detector=None) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  ! cannot open {video}")
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0
    idxs = np.unique(np.linspace(0, total - 1, num=min(n_frames, total)).round().astype(int))
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        if face_crop:
            frame = _crop_face(frame, detector)
            if frame is None or frame.size == 0:
                continue
        # PNG: lossless, so extraction does not add compression artifacts of its own.
        cv2.imwrite(str(out_dir / f"frame_{i:05d}.png"), frame)
        saved += 1
    cap.release()
    return saved


def cmd_extract(args: argparse.Namespace) -> None:
    root, out = args.root.resolve(), args.out.resolve()
    detector = _face_detector() if args.face_crop else None
    jobs: list[tuple[Path, Path]] = []
    orig = root / "original_sequences" / "youtube" / args.compression / "videos"
    if orig.is_dir():
        jobs += [(v, out / "real" / "original" / v.stem) for v in sorted(orig.iterdir()) if v.suffix.lower() in VIDEO_EXTS]
    for m in args.methods:
        mdir = root / "manipulated_sequences" / m / args.compression / "videos"
        if not mdir.is_dir():
            print(f"  (skipping {m}: {mdir} not found)")
            continue
        jobs += [(v, out / "fake" / m / v.stem) for v in sorted(mdir.iterdir()) if v.suffix.lower() in VIDEO_EXTS]
    if not jobs:
        raise SystemExit(f"No videos found under {root} (compression={args.compression})")
    if args.max_videos:
        jobs = jobs[: args.max_videos]
    total = 0
    for k, (video, dest) in enumerate(jobs, 1):
        if dest.is_dir() and any(dest.iterdir()) and not args.overwrite:
            continue
        n = extract_frames(video, dest, args.frames_per_video, args.face_crop, detector)
        total += n
        if k % 25 == 0 or k == len(jobs):
            print(f"[{k}/{len(jobs)}] frames saved so far: {total}")
    print(f"Done. Frames written under {out}")


# ----------------------------------------------------------------------------- manifest
class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, a: str) -> str:
        self.parent.setdefault(a, a)
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _video_ids(video: str) -> list[str]:
    return [p for p in video.split("_") if p]


def _official_split_map(split_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in ("train", "val", "test"):
        f = split_dir / f"{name}.json"
        if not f.exists():
            raise SystemExit(f"Official split file missing: {f}")
        for pair in json.loads(f.read_text(encoding="utf-8")):
            for vid in pair:
                mapping[str(vid)] = name
    return mapping


def build_face_manifest(frames_root: Path, dataset_id: str, seed: int, val_frac: float, test_frac: float, official_splits: Path | None) -> pd.DataFrame:
    rows = []
    for top, label in (("real", 0), ("fake", 1)):
        base = frames_root / top
        if not base.is_dir():
            continue
        for p in list_images(base):
            rel = p.relative_to(base).parts
            if len(rel) < 3:
                continue  # need <method>/<video>/<frame>
            method, video = rel[0], rel[1]
            rows.append({"path": p.relative_to(frames_root).as_posix(), "label": label, "source": method, "video": video})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No frames found under {frames_root}; expected {{real,fake}}/<method>/<video>/<frame>")

    uf = _UnionFind()
    for v in df["video"].unique():
        ids = _video_ids(v)
        for other in ids[1:]:
            uf.union(ids[0], other)
        uf.find(ids[0])
    df["identity"] = df["video"].map(lambda v: _video_ids(v)[0])  # target identity (video id), no person identification
    df["group"] = df["video"].map(lambda v: "g" + uf.find(_video_ids(v)[0]))
    df["dataset_id"] = dataset_id
    biggest = df["group"].value_counts(normalize=True).iloc[0]
    if biggest > 0.2:
        print(f"  ! the largest leakage group holds {biggest:.0%} of all frames (video ids are chained by "
              "manipulation pairs); a group split cannot be balanced - consider --official-splits")

    if official_splits:
        smap = _official_split_map(official_splits)
        df["split"] = df["identity"].map(smap)
        unknown = df["split"].isna()
        if unknown.any():
            print(f"  ! {int(unknown.sum())} frames have no official split entry; dropping them")
            df = df[~unknown].copy()
        # Official FF++ splits are pair-consistent, but verify our group invariant anyway.
        leaks = check_group_leakage(df)
        if leaks:
            print(f"  ! {len(leaks)} union-find groups span splits under the official split; "
                  "those groups are reassigned to the split of their lowest id")
            first = df.sort_values("identity").groupby("group")["split"].first()
            df["split"] = df["group"].map(first)
    else:
        df["split"] = group_split(df, {"train": 1 - val_frac - test_frac, "val": val_frac, "test": test_frac}, seed=seed)
    return df


def cmd_manifest(args: argparse.Namespace) -> None:
    df = build_face_manifest(args.frames_root.resolve(), args.dataset_id, args.seed, args.val_frac, args.test_frac, args.official_splits)
    leaks = check_group_leakage(df)
    if leaks:
        raise SystemExit(f"Leakage check failed: {len(leaks)} groups span multiple splits")
    write_manifest(df, args.out)
    summary = summarize_manifest(df)
    save_json(summary, args.out.with_suffix(".summary.json"))
    print(f"Wrote {len(df)} rows to {args.out}")
    for split, s in summary["splits"].items():
        print(f"  {split:5s} n={s['n']:7d} real={s['real']:7d} fake={s['fake']:7d} groups={s['groups']}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="Extract frames from FF++ videos")
    e.add_argument("--root", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--compression", default="c23", choices=["raw", "c23", "c40"])
    e.add_argument("--methods", nargs="+", default=list(FFPP_METHODS))
    e.add_argument("--frames-per-video", type=int, default=16)
    e.add_argument("--face-crop", action="store_true", help="Crop the largest detected face (OpenCV Haar cascade)")
    e.add_argument("--max-videos", type=int, default=None)
    e.add_argument("--overwrite", action="store_true")
    e.set_defaults(func=cmd_extract)
    m = sub.add_parser("manifest", help="Build a leakage-aware manifest from extracted frames")
    m.add_argument("--frames-root", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--dataset-id", default="ffpp")
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--val-frac", type=float, default=0.15)
    m.add_argument("--test-frac", type=float, default=0.15)
    m.add_argument("--official-splits", type=Path, default=None)
    m.set_defaults(func=cmd_manifest)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
