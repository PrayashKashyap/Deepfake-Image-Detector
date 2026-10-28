"""Analysis, history, forensics, report and robustness endpoints."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from pydantic import ValidationError

from app.core.container import Container, get_container
from app.core.errors import APIError, PayloadTooLargeError
from app.schemas.api import (
    AnalysisList,
    AnalysisResult,
    AnalysisSummary,
    ErrorResponse,
    ForensicsResponse,
    RobustnessParams,
    RobustnessResponse,
    StatsResponse,
    Track,
)
from app.services.reports import build_analysis_report
from app.services.robustness import run_robustness
from app.utils.images import validate_image_bytes

router = APIRouter(tags=["analysis"])

_ERRORS = {413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse, "description": "No trained model available for the track"}}


async def _read_limited(file: UploadFile, limit: int) -> bytes:
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise PayloadTooLargeError(f"File exceeds the {limit / (1024 * 1024):g} MB upload limit")
    return data


@router.post("/analyze", response_model=AnalysisResult, responses=_ERRORS, summary="Analyze an image with the active model of the selected track")
async def analyze(
    file: UploadFile = File(..., description="JPEG or PNG image"),
    track: Track = Form(..., description="ai_generated | face_deepfake"),
    c: Container = Depends(get_container),
) -> AnalysisResult:
    data = await _read_limited(file, c.settings.max_upload_bytes)
    img = validate_image_bytes(data, file.filename, c.settings)
    result = await run_in_threadpool(c.analyses.analyze, img, track)
    return AnalysisResult(**result)


@router.get("/analyses", response_model=AnalysisList, summary="Paginated analysis history")
def list_analyses(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    prediction: Literal["REAL", "FAKE"] | None = None,
    track: Track | None = None,
    date_from: str | None = Query(None, description="ISO date/time lower bound (UTC)"),
    date_to: str | None = Query(None, description="ISO date/time upper bound (UTC)"),
    q: str | None = Query(None, max_length=100, description="Search id, file name or model id"),
    c: Container = Depends(get_container),
) -> AnalysisList:
    if date_to and len(date_to) == 10:
        date_to = date_to + "T23:59:59+99:99"  # inclusive end-of-day for plain dates
    rows, total = c.analyses_repo.list(page, page_size, prediction, track, date_from, date_to, q)
    return AnalysisList(items=[AnalysisSummary(**{**r, "is_fixture_model": bool(r["is_fixture_model"])}) for r in rows], total=total, page=page, page_size=page_size)


@router.get("/stats", response_model=StatsResponse, summary="Aggregate counts of recorded analyses")
def stats(c: Container = Depends(get_container)) -> StatsResponse:
    s = c.analyses_repo.stats()
    return StatsResponse(total=s["total"], real=s["by_prediction"].get("REAL", 0), fake=s["by_prediction"].get("FAKE", 0), by_track=s["by_track"], per_day=s["per_day"], fixture_model_analyses=s["fixture_model_analyses"])


@router.get("/analyses/{analysis_id}", response_model=AnalysisResult, summary="Details of one analysis")
def get_analysis(analysis_id: str, c: Container = Depends(get_container)) -> AnalysisResult:
    return AnalysisResult(**c.analyses.get(analysis_id))


@router.delete("/analyses/{analysis_id}", status_code=204, summary="Delete an analysis and its stored files")
def delete_analysis(analysis_id: str, c: Container = Depends(get_container)) -> Response:
    c.analyses.delete(analysis_id)
    return Response(status_code=204)


@router.get("/analyses/{analysis_id}/artifacts/{name}", summary="Stored image artifact (original, input, gradcam, heatmap, spectrum)", response_class=FileResponse)
def get_artifact(analysis_id: str, name: str, c: Container = Depends(get_container)):
    p = c.analyses.artifact_path(analysis_id, name)
    return FileResponse(p, headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"})


@router.get("/analyses/{analysis_id}/report", summary="Download a PDF report", response_class=Response, responses={200: {"content": {"application/pdf": {}}}})
def report(analysis_id: str, c: Container = Depends(get_container)) -> Response:
    payload = c.analyses.raw_payload(analysis_id)
    files = {}
    for name in ("original", "gradcam", "spectrum"):
        try:
            files[name] = c.analyses.artifact_path(analysis_id, name)
        except APIError:
            files[name] = None
    pdf = build_analysis_report(payload, files)
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="analysis-{analysis_id[:12]}.pdf"'})


@router.get("/forensics/{analysis_id}", response_model=ForensicsResponse, summary="Measured image properties and available metadata")
def forensics(analysis_id: str, c: Container = Depends(get_container)) -> ForensicsResponse:
    p = c.analyses.get(analysis_id)
    f = p["forensics"]
    return ForensicsResponse(
        analysis_id=analysis_id,
        image=p["image"],
        properties=f["properties"],
        jpeg=f.get("jpeg"),
        png=f.get("png"),
        pixel_statistics=f.get("pixel_statistics", {}),
        exif=f.get("exif", {}),
        frequency=f.get("frequency", {}),
        observations=f.get("observations", []),
        spectrum_url=p["artifacts"].get("spectrum"),
        note="These are direct measurements of the file. They are not model inferences, and missing or present metadata does not by itself indicate whether an image is authentic.",
    )


@router.post("/robustness/analyze", response_model=RobustnessResponse, responses=_ERRORS, summary="Run the model on an image and perturbed variants (single-image demonstration)")
async def robustness(
    file: UploadFile = File(...),
    track: Track = Form(...),
    jpeg_quality: int | None = Form(None),
    downscale_max_side: int | None = Form(None),
    blur_sigma: float | None = Form(None),
    noise_std: float | None = Form(None),
    c: Container = Depends(get_container),
) -> RobustnessResponse:
    try:
        params = RobustnessParams(jpeg_quality=jpeg_quality, downscale_max_side=downscale_max_side, blur_sigma=blur_sigma, noise_std=noise_std)
    except ValidationError as exc:
        raise APIError("Invalid robustness parameters", status_code=422, code="validation_error", details=[{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]) from exc
    data = await _read_limited(file, c.settings.max_upload_bytes)
    validate_image_bytes(data, file.filename, c.settings)
    result = await run_in_threadpool(run_robustness, c.registry, data, track, params.model_dump(), c.settings.inference_amp)
    return RobustnessResponse(**result)
