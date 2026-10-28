"""A simple managed job queue (thread pool + SQLite status) for long-running evaluations."""
from __future__ import annotations

import logging
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from ml.common import utc_now_iso

from app.core.errors import NotFoundError
from app.core.security import safe_id
from app.database.repositories import JobRepository

log = logging.getLogger("dff.jobs")
MAX_LOG_LINES = 200

JobFn = Callable[[Callable[[float, str], None]], dict[str, Any]]


class JobManager:
    def __init__(self, repo: JobRepository, workers: int = 1) -> None:
        self.repo = repo
        self.pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="dff-job")
        self._logs: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        n = repo.mark_interrupted()
        if n:
            log.warning("Marked %d stale job(s) as interrupted", n)

    def submit(self, kind: str, params: dict[str, Any], fn: JobFn) -> str:
        job_id = uuid.uuid4().hex
        self.repo.create(job_id, kind, params)
        self.pool.submit(self._run, job_id, fn)
        return job_id

    def _progress(self, job_id: str) -> Callable[[float, str], None]:
        last = {"pct": -1.0}

        def cb(frac: float, msg: str) -> None:
            pct = round(max(0.0, min(1.0, frac)) * 100, 1)
            with self._lock:
                lines = self._logs.setdefault(job_id, [])
                lines.append(f"{utc_now_iso()} [{pct:5.1f}%] {msg}")
                del lines[:-MAX_LOG_LINES]
                logs = list(lines)
            if pct - last["pct"] >= 1.0 or frac >= 1.0:  # throttle DB writes
                last["pct"] = pct
                self.repo.update(job_id, progress=pct / 100, message=msg[:300], logs=logs[-50:])

        return cb

    def _run(self, job_id: str, fn: JobFn) -> None:
        self.repo.update(job_id, status="running", started_at=utc_now_iso(), message="running")
        cb = self._progress(job_id)
        try:
            result = fn(cb)
            with self._lock:
                logs = self._logs.pop(job_id, [])
            self.repo.update(job_id, status="succeeded", progress=1.0, message="completed", result=result, finished_at=utc_now_iso(), logs=logs[-50:])
        except Exception as exc:  # noqa: BLE001
            log.error("Job %s failed: %s", job_id, exc)
            with self._lock:
                logs = self._logs.pop(job_id, [])
            logs.append(traceback.format_exc(limit=3).replace(str(Path.home()), "~"))
            self.repo.update(job_id, status="failed", message="failed", error=f"{type(exc).__name__}: {exc}"[:1000], finished_at=utc_now_iso(), logs=logs[-50:])

    def get(self, job_id: str) -> dict[str, Any]:
        job = self.repo.get(safe_id(job_id, "job"))
        if job is None:
            raise NotFoundError("Job not found")
        return job

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.repo.list(limit)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
