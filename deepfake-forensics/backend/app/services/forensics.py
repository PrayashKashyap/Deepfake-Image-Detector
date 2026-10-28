"""Directly measured image properties and metadata (no model involved).

Everything here is an *observation* about the file. None of it is, by itself, evidence
that an image is authentic or manipulated: metadata is easily removed, added or edited,
and most platforms strip EXIF on upload.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import ExifTags, Image

from ml.explainability.frequency import radial_profile

# Tags that can identify people/devices/locations. Values are redacted; presence is reported.
SENSITIVE_EXIF = {
    "GPSInfo", "BodySerialNumber", "CameraOwnerName", "LensSerialNumber", "SerialNumber", "ImageUniqueID",
    "Artist", "Copyright", "XPAuthor", "HostComputer", "OwnerName", "UserComment", "MakerNote",
}
MAX_VALUE_LEN = 200

# Standard IJG (libjpeg) luminance quantization table (natural order, quality 50 base).
_IJG_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)

# PIL returns quantization tables in zigzag order; map zigzag index -> natural index.
_ZIGZAG = np.array([
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
])


def _ijg_table(q: int) -> np.ndarray:
    scale = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip(np.floor((_IJG_LUMA * scale + 50) / 100), 1, 255)


def estimate_jpeg_quality(luma_table: list[int]) -> dict[str, Any]:
    """Compare the file's luminance table against standard IJG tables at qualities 1-100."""
    t = np.asarray(luma_table, dtype=np.float64)
    if t.size != 64:
        return {"estimated_quality": None, "method": "unavailable"}
    natural = np.empty(64)
    natural[_ZIGZAG] = t
    # Pillow >= 9 already returns natural order for some builds; test both orders.
    best = None
    for order_name, cand in (("natural", t), ("zigzag", natural)):
        for q in range(1, 101):
            err = float(np.abs(_ijg_table(q) - cand).mean())
            if best is None or err < best[0]:
                best = (err, q, order_name)
    err, q, _ = best  # type: ignore[misc]
    return {
        "estimated_quality": q,
        "mean_abs_table_difference": round(err, 3),
        "matches_standard_table": err == 0.0,
        "method": "nearest standard IJG luminance table (assumes libjpeg-style scaling; custom encoders may differ)",
    }


def _clean_value(v: Any) -> Any:
    if isinstance(v, bytes):
        return f"<{len(v)} bytes>"
    if isinstance(v, (tuple, list)):
        return [_clean_value(x) for x in v[:16]]
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    try:
        f = float(v)  # IFDRational
        return round(f, 6)
    except (TypeError, ValueError):
        pass
    s = str(v).replace("\x00", "").strip()
    return s[:MAX_VALUE_LEN] + ("…" if len(s) > MAX_VALUE_LEN else "")


def extract_exif(img: Image.Image) -> dict[str, Any]:
    try:
        exif = img.getexif()
    except Exception:  # noqa: BLE001
        return {"present": False, "tags": {}, "redacted": []}
    tags: dict[str, Any] = {}
    redacted: list[str] = []
    items = dict(exif.items())
    try:
        items.update(exif.get_ifd(ExifTags.IFD.Exif))
    except Exception:  # noqa: BLE001
        pass
    for tag_id, value in items.items():
        name = ExifTags.TAGS.get(tag_id, f"Tag_{tag_id}")
        if name == "ExifOffset":
            continue
        if name in SENSITIVE_EXIF:
            redacted.append(name)
            continue
        tags[name] = _clean_value(value)
    tags.pop("ExifOffset", None)
    return {"present": bool(tags or redacted), "tags": dict(sorted(tags.items())), "redacted": sorted(set(redacted))}


def analyze_properties(data: bytes, file_name: str) -> dict[str, Any]:
    img = Image.open(io.BytesIO(data))
    img.load()
    w, h = img.size
    props: dict[str, Any] = {
        "file_name": file_name,
        "format": img.format,
        "mime_type": Image.MIME.get(img.format or "", None),
        "file_size_bytes": len(data),
        "width": w,
        "height": h,
        "megapixels": round(w * h / 1e6, 3),
        "aspect_ratio": round(w / h, 4),
        "color_mode": img.mode,
        "bands": list(img.getbands()),
        "has_alpha": img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info,
        "icc_profile_present": bool(img.info.get("icc_profile")),
        "dpi": [_clean_value(x) for x in img.info.get("dpi", ())] or None,
        "bits_per_pixel_compressed": round(8 * len(data) / (w * h), 4),
    }
    jpeg: dict[str, Any] | None = None
    if img.format == "JPEG":
        q = getattr(img, "quantization", None) or {}
        jpeg = {
            "progressive": bool(img.info.get("progressive") or img.info.get("progression")),
            "quantization_tables": len(q),
            "chroma_subsampling": _subsampling(img),
            "quality_estimate": estimate_jpeg_quality(list(q.get(0, []))) if q else None,
            "jfif": "jfif" in img.info,
            "adobe": "adobe" in img.info,
        }
    png: dict[str, Any] | None = None
    if img.format == "PNG":
        text = {k: _clean_value(v) for k, v in (getattr(img, "text", None) or {}).items()}
        png = {"text_chunks": text, "interlaced": bool(img.info.get("interlace")), "gamma": _clean_value(img.info.get("gamma"))}
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    stats = {
        "mean_rgb": [round(float(x), 2) for x in rgb.reshape(-1, 3).mean(0)],
        "std_rgb": [round(float(x), 2) for x in rgb.reshape(-1, 3).std(0)],
        "unique_colors_sampled": int(len(np.unique(rgb[:: max(1, h // 256), :: max(1, w // 256)].reshape(-1, 3), axis=0))),
    }
    exif = extract_exif(img)
    observations = _observations(props, exif, jpeg, png)
    return {"properties": props, "jpeg": jpeg, "png": png, "pixel_statistics": stats, "exif": exif, "frequency": {"radial_profile": radial_profile(img.convert("RGB"))}, "observations": observations}


def _subsampling(img: Image.Image) -> str | None:
    try:
        from PIL import JpegImagePlugin

        return {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}.get(JpegImagePlugin.get_sampling(img), "unknown")
    except Exception:  # noqa: BLE001
        return None


def _observations(props: dict, exif: dict, jpeg: dict | None, png: dict | None) -> list[dict[str, str]]:
    """Plain factual statements, each with an explicit caveat. Never a verdict."""
    obs = []
    if not exif["present"]:
        obs.append({"kind": "observed", "text": "No EXIF metadata is present.", "caveat": "Many platforms and editors remove EXIF; missing metadata does not indicate that an image is fake."})
    else:
        sw = exif["tags"].get("Software")
        if sw:
            obs.append({"kind": "observed", "text": f"EXIF 'Software' tag: {sw}", "caveat": "Metadata can be edited or forged and is not proof of origin."})
        make, model = exif["tags"].get("Make"), exif["tags"].get("Model")
        if make or model:
            obs.append({"kind": "observed", "text": f"EXIF camera fields: {make or '?'} {model or ''}".strip(), "caveat": "Camera fields can be copied or written by any tool."})
        if exif["redacted"]:
            obs.append({"kind": "observed", "text": f"Potentially identifying fields present and redacted: {', '.join(exif['redacted'])}", "caveat": "Values are not displayed or stored in reports to protect privacy."})
    if jpeg and jpeg.get("quality_estimate") and jpeg["quality_estimate"].get("estimated_quality"):
        qe = jpeg["quality_estimate"]
        exact = "exactly matches" if qe["matches_standard_table"] else "is closest to"
        obs.append({"kind": "measured", "text": f"JPEG luminance quantization table {exact} the standard IJG table at quality ≈{qe['estimated_quality']}.", "caveat": "Re-saving changes this value; it reflects the last JPEG encoding only."})
    if png and png.get("text_chunks"):
        obs.append({"kind": "observed", "text": f"PNG text chunks present: {', '.join(list(png['text_chunks'])[:8])}", "caveat": "Some generation tools write parameters into PNG text chunks, but such chunks can be added or removed freely."})
    return obs
