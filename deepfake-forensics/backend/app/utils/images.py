"""Upload validation. File contents are validated; extensions and client MIME types are not trusted."""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from app.core.config import Settings
from app.core.errors import InvalidImageError, PayloadTooLargeError, UnsupportedMediaError

# Magic numbers of the accepted formats.
_SIGNATURES = {
    "JPEG": (b"\xff\xd8\xff",),
    "PNG": (b"\x89PNG\r\n\x1a\n",),
}
ALLOWED_FORMATS = {"JPEG": ("image/jpeg", ".jpg"), "PNG": ("image/png", ".png")}


@dataclass
class ValidatedImage:
    data: bytes
    format: str  # "JPEG" | "PNG"
    mime: str
    extension: str
    width: int
    height: int
    mode: str
    sha256: str
    original_name: str


def sniff_format(data: bytes) -> str | None:
    for fmt, sigs in _SIGNATURES.items():
        if any(data.startswith(s) for s in sigs):
            return fmt
    return None


def sanitize_filename(name: str | None) -> str:
    """Keep only a short, display-safe base name (never used to build filesystem paths)."""
    if not name:
        return "upload"
    base = name.replace("\\", "/").split("/")[-1]
    cleaned = "".join(c for c in base if c.isalnum() or c in "._- ")[:80].strip()
    return cleaned or "upload"


def validate_image_bytes(data: bytes, filename: str | None, settings: Settings) -> ValidatedImage:
    if not data:
        raise InvalidImageError("The uploaded file is empty")
    if len(data) > settings.max_upload_bytes:
        raise PayloadTooLargeError(f"File exceeds the {settings.max_upload_mb:g} MB upload limit")
    fmt = sniff_format(data)
    if fmt is None:
        raise UnsupportedMediaError("Unsupported file type. Only JPEG and PNG images are accepted (checked by file content, not extension).")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            if probe.format != fmt:
                raise InvalidImageError("File content does not match a valid JPEG/PNG image")
            w, h = probe.size
            if w * h > settings.max_image_pixels:
                raise PayloadTooLargeError(f"Image has {w * h:,} pixels; the limit is {settings.max_image_pixels:,}")
            if min(w, h) < settings.min_image_side:
                raise InvalidImageError(f"Image is too small ({w}x{h}); minimum side is {settings.min_image_side}px")
            probe.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()  # full decode catches truncated / malformed pixel data
            mode = im.mode
    except (InvalidImageError, PayloadTooLargeError):
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f"Malformed or corrupted image: {type(exc).__name__}") from exc
    mime, ext = ALLOWED_FORMATS[fmt]
    return ValidatedImage(data=data, format=fmt, mime=mime, extension=ext, width=w, height=h, mode=mode, sha256=hashlib.sha256(data).hexdigest(), original_name=sanitize_filename(filename))
