"""Optional API-key protection and path-safety helpers."""
from __future__ import annotations

import hmac
import re
from pathlib import Path

from fastapi import Depends, Header

from app.core.config import Settings, get_settings
from app.core.errors import APIError, NotFoundError

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,199}$")


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """If DFF_API_KEY is configured, require a matching X-API-Key header (constant-time compare)."""
    if not settings.api_key:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), settings.api_key.encode()):
        raise APIError("Missing or invalid API key", status_code=401, code="unauthorized")


def safe_id(value: str, what: str = "identifier") -> str:
    if not SAFE_ID.match(value or "") or ".." in value:
        raise NotFoundError(f"Unknown {what}")
    return value


def safe_child(base: Path, *parts: str) -> Path:
    """Join parts under base and guarantee the result stays inside base (no traversal)."""
    base = base.resolve()
    target = base.joinpath(*parts).resolve()
    if not target.is_relative_to(base):
        raise NotFoundError("Resource not found")
    return target
