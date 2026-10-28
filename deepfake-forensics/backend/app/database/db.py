"""SQLite persistence (stdlib sqlite3, one short-lived connection per operation)."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    track TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_version TEXT,
    arch TEXT,
    prediction TEXT NOT NULL CHECK (prediction IN ('REAL', 'FAKE')),
    fake_score REAL NOT NULL,
    threshold REAL NOT NULL,
    threshold_source TEXT,
    inference_ms REAL,
    gradcam_ms REAL,
    file_name TEXT,
    file_format TEXT,
    file_size INTEGER,
    width INTEGER,
    height INTEGER,
    sha256 TEXT,
    is_fixture_model INTEGER NOT NULL DEFAULT 0,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_analyses_created ON analyses(created_at);
CREATE INDEX IF NOT EXISTS ix_analyses_track ON analyses(track);
CREATE INDEX IF NOT EXISTS ix_analyses_prediction ON analyses(prediction);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    message TEXT,
    params TEXT,
    result TEXT,
    error TEXT,
    logs TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        with self.connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
        finally:
            con.close()

    def execute(self, sql: str, params: tuple | dict = ()) -> int:
        with self._write_lock, self.connect() as con:
            cur = con.execute(sql, params)
            return cur.rowcount

    def query(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        with self.connect() as con:
            return [dict(r) for r in con.execute(sql, params).fetchall()]

    def query_one(self, sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def ping(self) -> bool:
        try:
            self.query("SELECT 1")
            return True
        except sqlite3.Error:
            return False


def dumps(v: Any) -> str:
    return json.dumps(v, default=str)


def loads(v: str | None, default: Any = None) -> Any:
    if v is None:
        return default
    try:
        return json.loads(v)
    except (TypeError, json.JSONDecodeError):
        return default
