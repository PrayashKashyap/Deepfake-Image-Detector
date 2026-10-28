"""Extract a random, balanced subset of a very large remote ZIP via HTTP range requests.

Used to obtain a GenImage subset from its public mirror without downloading the full
archive. Only the central directory and the selected byte ranges are transferred.

    python -m ml.data.remote_zip --url https://huggingface.co/datasets/jzousz/GenImage/resolve/main/genimage_test.zip `
        --out C:\\Users\\me\\datasets\\GenImage_subset --list-only
    python -m ml.data.remote_zip --url ... --out ... --per-group 1000 --chunk 50 --seed 42

Grouping: files are grouped by their parent directory (e.g. <generator>/.../{ai,nature}).
From each group, `per-group` files are taken as random contiguous runs of `chunk` files
(contiguous in archive order, so each run costs one HTTP request). Runs start at random
positions, which spreads the sample over the group instead of taking only the first files.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests, with explicit prefetch segments."""

    def __init__(self, url: str, block: int = 1 << 20) -> None:
        self.origin = url
        self.block = block
        self.pos = 0
        self._resolve()
        self._segments: list[tuple[int, bytes]] = []
        self.bytes_fetched = 0

    def _resolve(self) -> None:
        """Follow redirects to the (time-limited, signed) CDN URL; re-run when it expires."""
        req = urllib.request.Request(self.origin, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            self.url = r.geturl()

    def _get(self, start: int, end: int) -> bytes:
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) != end - start:
                    raise IOError(f"short read {len(data)} != {end - start}")
                self.bytes_fetched += len(data)
                return data
            except Exception as exc:  # noqa: BLE001 - network retry
                if attempt == 5:
                    raise
                if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403):
                    print("    signed URL expired; re-resolving")
                    self._resolve()
                    continue
                time.sleep(2 * (attempt + 1))
                print(f"    retry range {start}-{end}: {exc}")
        raise RuntimeError("unreachable")

    def prefetch(self, start: int, end: int) -> None:
        self._segments = [(start, self._get(start, min(end, self.size)))]

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def readinto(self, b) -> int:
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        for seg_start, data in self._segments:
            if seg_start <= self.pos < seg_start + len(data):
                # Serve from the cached segment; a short read is legal for raw streams and
                # avoids a network round-trip when the caller's buffer extends past the segment.
                off = self.pos - seg_start
                n = min(n, len(data) - off)
                b[:n] = data[off : off + n]
                self.pos += n
                return n
        # Not prefetched (e.g. central directory reads): fetch at least one block.
        start = self.pos
        end = min(self.size, max(start + n, start + self.block))
        data = self._get(start, end)
        self._segments.append((start, data))
        self._segments = self._segments[-8:]
        b[:n] = data[:n]
        self.pos += n
        return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-group", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include", nargs="*", default=None, help="Only groups whose path contains one of these substrings")
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args(argv)

    rf = HTTPRangeFile(args.url)
    print(f"archive size: {rf.size / 1e9:.2f} GB")
    zf = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 20))
    infos = sorted((i for i in zf.infolist() if not i.is_dir() and PurePosixPath(i.filename).suffix.lower() in IMAGE_EXT), key=lambda i: i.header_offset)
    print(f"central directory read ({rf.bytes_fetched / 1e6:.1f} MB transferred): {len(infos)} images")
    order = {i.filename: k for k, i in enumerate(infos)}
    groups: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
    for i in infos:
        g = str(PurePosixPath(i.filename).parent)
        if args.include and not any(s in g for s in args.include):
            continue
        groups[g].append(i)
    for g, items in sorted(groups.items()):
        print(f"  {g}: {len(items)} files, {sum(i.compress_size for i in items) / 1e9:.2f} GB")
    if args.list_only:
        return

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    t0 = time.time()
    for g, items in sorted(groups.items()):
        n_runs = max(1, args.per_group // args.chunk)
        starts = sorted(rng.sample(range(0, max(1, len(items) - args.chunk)), k=min(n_runs, max(1, len(items) - args.chunk))))
        chosen: list[list[zipfile.ZipInfo]] = []
        last_end = -1
        for s in starts:  # non-overlapping runs
            s = max(s, last_end)
            run = items[s : s + args.chunk]
            if run:
                chosen.append(run)
                last_end = s + args.chunk
        got = 0
        for run in chosen:
            first, last = run[0], run[-1]
            nxt = order[last.filename] + 1
            end = infos[nxt].header_offset if nxt < len(infos) else rf.size
            rf.prefetch(first.header_offset, end)
            for info in run:
                dest = args.out / info.filename
                if dest.exists():
                    got += 1
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(info))
                manifest.append({"path": info.filename, "bytes": info.file_size})
                got += 1
        print(f"  {g}: extracted {got} | total transferred {rf.bytes_fetched / 1e9:.2f} GB | {time.time() - t0:.0f}s")
    meta = {"url": args.url, "seed": args.seed, "per_group": args.per_group, "chunk": args.chunk, "groups": {g: len(v) for g, v in groups.items()}}
    (args.out / "subset_source.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"done: {len(manifest)} new files, {rf.bytes_fetched / 1e9:.2f} GB transferred")


if __name__ == "__main__":
    main()
