"""Read experiment outputs written by the ml package (never computed or invented here).

Experiment ids:
    eval--<eval_id>                 artifacts/evaluations/<eval_id>/result.json
    train--<experiment_id>--<run_id> artifacts/experiments/<experiment_id>/<run_id>/run.json
"""
from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.errors import NotFoundError
from app.core.security import safe_child, safe_id

log = logging.getLogger("dff.evaluation")

ARTIFACT_EXTS = {".png", ".csv", ".json", ".md", ".tex"}
HEADLINE = ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "precision_fake", "recall_fake", "precision_real", "recall_real")


def _read_json(p: Path) -> dict[str, Any] | None:
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Skipping unreadable result file %s: %s", p.name, exc)
        return None


class EvaluationResults:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _summ(self, exp_id: str, r: dict[str, Any]) -> dict[str, Any]:
        kind = r.get("kind", "evaluation")
        metrics = r.get("metrics") or r.get("validation_metrics") or {}
        return {
            "id": exp_id,
            "kind": kind,
            "experiment_id": r.get("experiment_id"),
            "track": r.get("track"),
            "created_at": r.get("created_at"),
            "description": r.get("description", ""),
            "model_id": (r.get("model") or {}).get("model_id"),
            "arch": (r.get("model") or {}).get("arch"),
            "frequency_branch": (r.get("model") or {}).get("frequency_branch"),
            "splits": (r.get("dataset") or {}).get("splits") or ([(r.get("dataset") or {}).get("val_split")] if kind == "training" else None),
            "dataset_ids": (r.get("dataset") or {}).get("dataset_ids"),
            "n_samples": (r.get("dataset") or {}).get("n_samples"),
            "unseen_sources": (r.get("dataset") or {}).get("unseen_sources"),
            "is_synthetic_fixture": bool((r.get("dataset") or {}).get("is_synthetic_fixture")),
            "metrics": {k: metrics.get(k) for k in HEADLINE if k in metrics},
            "metrics_scope": "validation split (training run)" if kind == "training" else ("comparison of runs" if kind == "comparison" else "evaluation split"),
            "comparison_type": r.get("comparison_type"),
            "latency_mean_ms": (r.get("latency") or {}).get("mean_ms"),
        }

    def _iter(self):
        ev = self.settings.evaluations_dir
        if ev.is_dir():
            for d in sorted(ev.iterdir()):
                f = d / "result.json"
                if d.is_dir() and f.is_file():
                    r = _read_json(f)
                    if r is not None:
                        yield f"eval--{d.name}", d, r
        ex = self.settings.experiments_dir
        if ex.is_dir():
            for exp in sorted(ex.iterdir()):
                if not exp.is_dir():
                    continue
                for run in sorted(exp.iterdir()):
                    f = run / "run.json"
                    if run.is_dir() and f.is_file():
                        r = _read_json(f)
                        if r is not None:
                            yield f"train--{exp.name}--{run.name}", run, r

    def list(self, kind: str | None = None, track: str | None = None) -> list[dict[str, Any]]:
        out = [self._summ(i, r) for i, _, r in self._iter()]
        if kind:
            out = [e for e in out if e["kind"] == kind]
        if track:
            out = [e for e in out if e["track"] == track]
        return sorted(out, key=lambda e: e.get("created_at") or "", reverse=True)

    def _locate(self, exp_id: str) -> tuple[Path, Path]:
        safe_id(exp_id, "experiment")
        if exp_id.startswith("eval--"):
            d = safe_child(self.settings.evaluations_dir, exp_id[len("eval--"):])
            return d, d / "result.json"
        if exp_id.startswith("train--"):
            parts = exp_id[len("train--"):].split("--")
            if len(parts) != 2:
                raise NotFoundError("Unknown experiment")
            d = safe_child(self.settings.experiments_dir, parts[0], parts[1])
            return d, d / "run.json"
        raise NotFoundError("Unknown experiment")

    def get(self, exp_id: str) -> dict[str, Any]:
        d, f = self._locate(exp_id)
        r = _read_json(f) if f.is_file() else None
        if r is None:
            raise NotFoundError("Experiment not found")
        files = sorted(p.name for p in d.iterdir() if p.is_file() and p.suffix.lower() in ARTIFACT_EXTS)
        return {
            "summary": self._summ(exp_id, r),
            "result": r,
            "artifacts": [{"name": n, "url": f"/api/evaluation/experiments/{exp_id}/artifacts/{n}"} for n in files],
        }

    def artifact(self, exp_id: str, name: str) -> Path:
        d, _ = self._locate(exp_id)
        safe_id(name, "artifact")
        p = safe_child(d, name)
        if not p.is_file() or p.suffix.lower() not in ARTIFACT_EXTS:
            raise NotFoundError("Artifact not found")
        return p

    def summary(self) -> dict[str, Any]:
        items = self.list()
        evals = [e for e in items if e["kind"] in ("evaluation", "robustness")]
        latest: dict[str, Any] = {}
        for e in evals:
            if e["track"] and e["track"] not in latest:
                latest[e["track"]] = e
        return {
            "available": bool(items),
            "counts": {k: sum(1 for e in items if e["kind"] == k) for k in ("training", "evaluation", "robustness", "comparison")},
            "latest_evaluation_by_track": latest,
            "experiments": items,
            "message": None if items else "No evaluation results available. Run training/evaluation with the documented CLI or start an evaluation job.",
        }

    def metrics_csv(self, exp_ids: list[str] | None = None) -> str:
        items = self.list()
        if exp_ids:
            items = [e for e in items if e["id"] in set(exp_ids)]
        buf = io.StringIO()
        cols = ["id", "kind", "experiment_id", "track", "created_at", "model_id", "arch", "frequency_branch", "metrics_scope", "n_samples", "is_synthetic_fixture", *HEADLINE, "latency_mean_ms"]
        w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for e in items:
            w.writerow({**{k: e.get(k) for k in cols}, **e.get("metrics", {})})
        return buf.getvalue()
