"""Evaluation results, evaluation jobs and job status endpoints."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse, Response

from ml.common import resolve_repo_path
from ml.data.registry import resolve_dataset

from app.core.container import Container, get_container
from app.core.errors import APIError, ModelUnavailableError
from app.schemas.api import (
    EvaluationRunRequest,
    EvaluationSummary,
    ExperimentDetail,
    ExperimentSummary,
    JobCreated,
    JobStatus,
)

router = APIRouter(tags=["evaluation"])


@router.get("/evaluation/summary", response_model=EvaluationSummary, summary="Available evaluation metrics and experiment summaries")
def evaluation_summary(c: Container = Depends(get_container)) -> EvaluationSummary:
    return EvaluationSummary(**c.evaluations.summary())


@router.get("/evaluation/experiments", response_model=list[ExperimentSummary], summary="List completed experiment runs (training, evaluation, robustness, comparison)")
def list_experiments(kind: str | None = Query(None, pattern="^(training|evaluation|robustness|comparison)$"), track: str | None = None, c: Container = Depends(get_container)):
    return [ExperimentSummary(**e) for e in c.evaluations.list(kind, track)]


@router.get("/evaluation/metrics.csv", summary="Export experiment metrics as CSV", response_class=Response, responses={200: {"content": {"text/csv": {}}}})
def metrics_csv(ids: list[str] | None = Query(None), c: Container = Depends(get_container)) -> Response:
    return Response(c.evaluations.metrics_csv(ids), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="experiment-metrics.csv"'})


@router.get("/evaluation/experiments/{experiment_id}", response_model=ExperimentDetail, summary="Metrics and artifact references for one experiment")
def get_experiment(experiment_id: str, c: Container = Depends(get_container)) -> ExperimentDetail:
    return ExperimentDetail(**c.evaluations.get(experiment_id))


@router.get("/evaluation/experiments/{experiment_id}/artifacts/{name}", summary="Download an experiment artifact (plot / CSV / JSON)", response_class=FileResponse)
def experiment_artifact(experiment_id: str, name: str, c: Container = Depends(get_container)):
    p = c.evaluations.artifact(experiment_id, name)
    return FileResponse(p, filename=name, headers={"X-Content-Type-Options": "nosniff"})


def _inside(path: Path, roots: list[Path]) -> bool:
    rp = path.resolve()
    return any(rp.is_relative_to(r.resolve()) for r in roots)


@router.post("/evaluation/run", response_model=JobCreated, status_code=202, summary="Start an evaluation job (background)")
def run_evaluation_job(body: EvaluationRunRequest, c: Container = Depends(get_container)) -> JobCreated:
    entry = c.registry.get_entry(body.model_id)
    if not entry.valid:
        raise ModelUnavailableError(f"Model '{body.model_id}' is invalid: {entry.error}")
    if body.dataset_name:
        try:
            ref = resolve_dataset(body.dataset_name)
        except KeyError as exc:
            raise APIError(str(exc.args[0]), status_code=422, code="dataset_not_configured") from exc
        root, manifest = ref.root, ref.manifest
    else:
        root, manifest = resolve_repo_path(body.dataset_root), resolve_repo_path(body.manifest)  # type: ignore[arg-type]
        roots = c.settings.allowed_data_roots
        if not (_inside(root, roots) and _inside(manifest, roots)):
            raise APIError("Dataset paths must be inside a directory listed in DFF_DATA_ROOTS", status_code=422, code="path_not_allowed")
    if not manifest.is_file():
        raise APIError("Manifest file not found", status_code=422, code="dataset_not_found")
    if not root.is_dir():
        raise APIError("Dataset root directory not found", status_code=422, code="dataset_not_found")
    if manifest.suffix.lower() != ".csv":
        raise APIError("Manifest must be a .csv file", status_code=422, code="validation_error")

    from ml.robustness.perturbations import DEFAULT_ROBUSTNESS_SUITE

    cfg = {
        "checkpoint": str(entry.path),
        "dataset": {"root": str(root), "manifest": str(manifest), "splits": list(body.splits), "include_sources": body.include_sources, "exclude_sources": body.exclude_sources, "max_per_class": body.max_per_class},
        "batch_size": body.batch_size,
        "num_workers": body.num_workers,
        "device": c.settings.device,
        "bootstrap": {"enabled": body.bootstrap, "n_boot": 1000},
        "perturbations": DEFAULT_ROBUSTNESS_SUITE if body.robustness else None,
        "output_dir": str(c.settings.evaluations_dir),
        "description": f"Evaluation job via API ({'robustness' if body.robustness else 'standard'})",
    }
    public_params = body.model_dump(exclude={"dataset_root", "manifest"})

    def _job(progress):
        from ml.evaluation.evaluate import run_evaluation

        res_path = run_evaluation(cfg, progress)
        eval_id = res_path.parent.name
        return {"experiment_id": f"eval--{eval_id}", "result_url": f"/api/evaluation/experiments/eval--{eval_id}"}

    job_id = c.jobs.submit("evaluation", public_params, _job)
    return JobCreated(job_id=job_id, status="queued")


@router.get("/jobs", response_model=list[dict], summary="Recent jobs")
def list_jobs(c: Container = Depends(get_container)):
    return c.jobs.list()


@router.get("/jobs/{job_id}", response_model=JobStatus, summary="Job status, progress, log summary and output artifacts")
def job_status(job_id: str, c: Container = Depends(get_container)) -> JobStatus:
    return JobStatus(**c.jobs.get(job_id))
