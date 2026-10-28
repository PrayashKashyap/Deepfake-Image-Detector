"""Data-access helpers for analyses, settings and jobs."""
from __future__ import annotations

from typing import Any

from ml.common import utc_now_iso

from app.database.db import Database, dumps, loads

ANALYSIS_COLUMNS = (
    "id", "created_at", "track", "model_id", "model_version", "arch", "prediction", "fake_score", "threshold",
    "threshold_source", "inference_ms", "gradcam_ms", "file_name", "file_format", "file_size", "width", "height",
    "sha256", "is_fixture_model",
)


class AnalysisRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert(self, row: dict[str, Any], payload: dict[str, Any]) -> None:
        cols = list(ANALYSIS_COLUMNS) + ["payload"]
        values = [row.get(c) for c in ANALYSIS_COLUMNS] + [dumps(payload)]
        self.db.execute(f"INSERT INTO analyses ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})", tuple(values))

    def get(self, analysis_id: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
        if row:
            row["payload"] = loads(row["payload"], {})
        return row

    def delete(self, analysis_id: str) -> bool:
        return self.db.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,)) > 0

    def _where(self, prediction: str | None, track: str | None, date_from: str | None, date_to: str | None, q: str | None) -> tuple[str, list[Any]]:
        clauses, params = [], []
        if prediction:
            clauses.append("prediction = ?")
            params.append(prediction)
        if track:
            clauses.append("track = ?")
            params.append(track)
        if date_from:
            clauses.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("created_at <= ?")
            params.append(date_to)
        if q:
            clauses.append("(id LIKE ? OR file_name LIKE ? OR model_id LIKE ?)")
            like = f"%{q}%"
            params += [like, like, like]
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def list(self, page: int, page_size: int, prediction: str | None = None, track: str | None = None, date_from: str | None = None, date_to: str | None = None, q: str | None = None) -> tuple[list[dict[str, Any]], int]:
        where, params = self._where(prediction, track, date_from, date_to, q)
        total = self.db.query_one(f"SELECT COUNT(*) AS n FROM analyses{where}", tuple(params))["n"]  # type: ignore[index]
        rows = self.db.query(
            f"SELECT {', '.join(ANALYSIS_COLUMNS)} FROM analyses{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            tuple(params + [page_size, (page - 1) * page_size]),
        )
        return rows, int(total)

    def stats(self) -> dict[str, Any]:
        total = self.db.query_one("SELECT COUNT(*) AS n FROM analyses")["n"]  # type: ignore[index]
        by_pred = {r["prediction"]: r["n"] for r in self.db.query("SELECT prediction, COUNT(*) AS n FROM analyses GROUP BY prediction")}
        by_track = self.db.query("SELECT track, prediction, COUNT(*) AS n FROM analyses GROUP BY track, prediction")
        fixture = self.db.query_one("SELECT COUNT(*) AS n FROM analyses WHERE is_fixture_model = 1")["n"]  # type: ignore[index]
        per_day = self.db.query("SELECT substr(created_at, 1, 10) AS day, prediction, COUNT(*) AS n FROM analyses GROUP BY day, prediction ORDER BY day DESC LIMIT 60")
        return {"total": int(total), "by_prediction": by_pred, "by_track": by_track, "fixture_model_analyses": int(fixture), "per_day": per_day}


class SettingsRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get(self, key: str, default: Any = None) -> Any:
        row = self.db.query_one("SELECT value FROM settings WHERE key = ?", (key,))
        return loads(row["value"], default) if row else default

    def set(self, key: str, value: Any) -> None:
        if value is None:
            self.db.execute("DELETE FROM settings WHERE key = ?", (key,))
            return
        self.db.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, dumps(value), utc_now_iso()),
        )


class JobRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, job_id: str, kind: str, params: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO jobs (id, kind, status, progress, message, params, logs, created_at) VALUES (?, ?, 'queued', 0, 'queued', ?, '[]', ?)",
            (job_id, kind, dumps(params), utc_now_iso()),
        )

    def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        for k in ("params", "result", "logs"):
            if k in fields and not isinstance(fields[k], str):
                fields[k] = dumps(fields[k])
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.db.execute(f"UPDATE jobs SET {sets} WHERE id = ?", tuple(fields.values()) + (job_id,))

    def get(self, job_id: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if row:
            row["params"] = loads(row["params"], {})
            row["result"] = loads(row["result"])
            row["logs"] = loads(row["logs"], [])
        return row

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT id, kind, status, progress, message, created_at, started_at, finished_at, error FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
        return rows

    def mark_interrupted(self) -> int:
        return self.db.execute(
            "UPDATE jobs SET status = 'failed', error = 'Interrupted: the server restarted while the job was running', finished_at = ? WHERE status IN ('queued', 'running')",
            (utc_now_iso(),),
        )
