"""Dataset manifest builders for the real datasets (run on tiny synthetic directory trees)."""
from __future__ import annotations

import io
import zipfile

from PIL import Image

from ml.data.deepfakeface import build as build_dff
from ml.data.genimage import build_manifest as build_genimage
from ml.data.manifest import check_group_leakage


def _img(path, color=(10, 20, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 40), color).save(path)


def test_dff_groups_real_with_its_fakes(tmp_path):
    for i in range(30):
        name = f"{1000 + i}_1980-01-01_2010.jpg"
        for m in ("wiki", "inpainting", "insight", "text2img"):
            _img(tmp_path / m / f"{i % 10:02d}" / name)
    df = build_dff(tmp_path, "dff_test", seed=1, val_frac=0.2, test_frac=0.2)
    assert len(df) == 120
    assert set(df["source"]) == {"wiki", "inpainting", "insight", "text2img"}
    assert (df[df.source == "wiki"]["label"] == 0).all() and (df[df.source != "wiki"]["label"] == 1).all()
    assert check_group_leakage(df) == []
    # every id has its real image and its 3 fakes in the same split
    assert (df.groupby("group")["split"].nunique() == 1).all()
    assert (df.groupby("group").size() == 4).all()
    assert set(df["split"]) == {"train", "val", "test"}


def test_genimage_test_prefix_source_parsing(tmp_path):
    for g in ("biggan_imagenet", "glide_imagenet"):
        for cls, color in (("ai", (200, 0, 0)), ("nature", (0, 200, 0))):
            for i in range(10):
                _img(tmp_path / "test" / g / cls / f"{i}.png", color)
    df = build_genimage(tmp_path, "genimage", "g", seed=0)
    assert set(df["source"]) == {"biggan_imagenet", "glide_imagenet"}
    assert set(df["split"]) == {"train", "val", "test"}  # official 'test' images are re-split
    assert df.groupby("label").size().to_dict() == {0: 20, 1: 20}


def test_remote_zip_reader_on_local_server(tmp_path):
    """HTTPRangeFile + selective extraction against a local HTTP server with Range support."""
    import threading
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    from ml.data.remote_zip import main as rz_main

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for g in ("gen_a/ai", "gen_a/nature"):
            for i in range(20):
                b = io.BytesIO()
                Image.new("RGB", (8, 8), (i, i, i)).save(b, format="PNG")
                zf.writestr(f"test/{g}/{i:03d}.png", b.getvalue())
    (tmp_path / "srv").mkdir()
    (tmp_path / "srv" / "a.zip").write_bytes(buf.getvalue())

    class RangeHandler(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send_head(self):
            rng = self.headers.get("Range")
            path = self.translate_path(self.path)
            data = open(path, "rb").read()
            if rng:
                s, e = rng.split("=")[1].split("-")
                chunk = data[int(s): int(e) + 1]
                self.send_response(206)
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                return io.BytesIO(chunk)
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return io.BytesIO(data if self.command == "GET" else b"")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler, directory=str(tmp_path / "srv")))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        out = tmp_path / "out"
        rz_main(["--url", f"http://127.0.0.1:{srv.server_port}/a.zip", "--out", str(out), "--per-group", "10", "--chunk", "5", "--seed", "0"])
        files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.png"))
        assert len(files) == 20
        assert sum(f.startswith("test/gen_a/ai/") for f in files) == 10
        for f in files:  # extracted bytes decode correctly
            Image.open(out / f).load()
    finally:
        srv.shutdown()
