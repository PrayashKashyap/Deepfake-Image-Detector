"""Orchestrates an analysis: validate -> infer -> explain -> measure -> persist."""
from __future__ import annotations

import io
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from ml.common import CLASS_NAMES, FAKE, TRACK_LABELS, utc_now_iso
from ml.explainability.frequency import spectrum_image
from ml.preprocessing.transforms import load_image_rgb

from app.core.config import Settings
from app.core.errors import InvalidImageError, NotFoundError
from app.core.security import safe_child, safe_id
from app.database.repositories import AnalysisRepository
from app.services import forensics as forensics_service
from app.services.inference import DISCLAIMER, SCORE_NOTE, gradcam_images, predict_image, to_pil
from app.services.model_registry import ModelRegistry
from app.utils.images import ValidatedImage

log = logging.getLogger("dff.analysis")

ARTIFACT_NAMES = ("original", "input", "gradcam", "heatmap", "spectrum")


def model_ref(lm_entry, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": lm_entry.model_id,
        "display_name": meta.get("display_name", lm_entry.model_id),
        "arch": meta.get("arch"),
        "version": meta.get("version"),
        "frequency_branch": bool(meta.get("frequency_branch")),
        "is_synthetic_fixture": bool(meta.get("dataset", {}).get("is_synthetic_fixture")),
        "trained_on": meta.get("dataset", {}).get("dataset_ids", []),
    }


def model_warnings(meta: dict[str, Any]) -> list[str]:
    w = []
    if meta.get("dataset", {}).get("is_synthetic_fixture"):
        w.append("This model was trained on the SYNTHETIC SOFTWARE-TEST FIXTURE. Its predictions are meaningless and exist only to verify the software pipeline.")
    return w


class AnalysisService:
    def __init__(self, settings: Settings, registry: ModelRegistry, repo: AnalysisRepository) -> None:
        self.settings = settings
        self.registry = registry
        self.repo = repo

    def _dir(self, analysis_id: str) -> Path:
        return safe_child(self.settings.analyses_dir, safe_id(analysis_id, "analysis"))

    def artifact_path(self, analysis_id: str, name: str) -> Path:
        row = self.repo.get(safe_id(analysis_id, "analysis"))
        if row is None:
            raise NotFoundError("Analysis not found")
        files = row["payload"].get("_files", {})
        if name not in ARTIFACT_NAMES or name not in files:
            raise NotFoundError(f"Artifact '{name}' is not available for this analysis")
        p = safe_child(self._dir(analysis_id), files[name])
        if not p.is_file():
            raise NotFoundError("Artifact file is missing")
        return p

    def analyze(self, image: ValidatedImage, track: str) -> dict[str, Any]:
        lm = self.registry.require_active(track)  # raises 503 with explanation if no model
        try:
            img = load_image_rgb(image.data)
        except ValueError as exc:
            raise InvalidImageError(str(exc)) from exc

        pred = predict_image(self.registry, lm, img, self.settings.inference_amp)
        analysis_id = uuid.uuid4().hex
        adir = self._dir(analysis_id)
        adir.mkdir(parents=True, exist_ok=False)
        files: dict[str, str] = {}
        try:
            # Generated, safe file names only — the client-supplied name is never used for paths.
            files["original"] = f"original{image.extension}"
            (adir / files["original"]).write_bytes(image.data)

            gradcam_info: dict[str, Any] = {"available": False, "reason": "disabled in settings"}
            gradcam_ms = None
            if self.settings.enable_gradcam:
                cls = FAKE if pred.label == "FAKE" else 1 - FAKE
                g = gradcam_images(lm, img, cls)
                gradcam_ms = g["ms"]
                for k in ("input", "gradcam", "heatmap"):
                    files[k] = f"{k}.png"
                to_pil(g["input"]).save(adir / files["input"])
                to_pil(g["overlay"]).save(adir / files["gradcam"])
                to_pil(g["heatmap"]).save(adir / files["heatmap"])
                gradcam_info = {
                    "available": True,
                    "explained_class": CLASS_NAMES[cls],
                    "target_layer": "layer4[-1]" if lm.meta.get("arch") == "resnet50" else "features[-1]",
                    "resolution_note": f"Computed on the {lm.preprocess.img_size}x{lm.preprocess.img_size} model input ({lm.preprocess.resize_mode}).",
                    "note": "Grad-CAM highlights regions that most increased the explained class logit. It is a qualitative explanation and does not prove that the model used meaningful forensic evidence.",
                }
            files["spectrum"] = "spectrum.png"
            spectrum_image(img).save(adir / files["spectrum"])
            forensic = forensics_service.analyze_properties(image.data, image.original_name)
        except Exception:
            shutil.rmtree(adir, ignore_errors=True)
            raise

        created = utc_now_iso()
        payload = {
            "id": analysis_id,
            "created_at": created,
            "track": track,
            "track_label": TRACK_LABELS[track],
            "prediction": pred.label,
            "fake_score": pred.fake_score,
            "real_score": 1.0 - pred.fake_score,
            "threshold": pred.threshold,
            "threshold_source": pred.threshold_source,
            "inference_ms": pred.inference_ms,
            "gradcam_ms": gradcam_ms,
            "device": str(lm.device),
            "model": model_ref(lm.entry, lm.meta),
            "image": {"file_name": image.original_name, "format": image.format, "file_size_bytes": len(image.data), "width": image.width, "height": image.height, "color_mode": image.mode, "sha256": image.sha256},
            "explanation": gradcam_info,
            "forensics": forensic,
            "warnings": model_warnings(lm.meta),
            "score_note": SCORE_NOTE,
            "disclaimer": DISCLAIMER,
            "_files": files,
        }
        self.repo.insert(
            {
                "id": analysis_id, "created_at": created, "track": track, "model_id": lm.entry.model_id, "model_version": lm.meta.get("version"),
                "arch": lm.meta.get("arch"), "prediction": pred.label, "fake_score": pred.fake_score, "threshold": pred.threshold,
                "threshold_source": pred.threshold_source, "inference_ms": pred.inference_ms, "gradcam_ms": gradcam_ms, "file_name": image.original_name,
                "file_format": image.format, "file_size": len(image.data), "width": image.width, "height": image.height, "sha256": image.sha256,
                "is_fixture_model": int(bool(lm.meta.get("dataset", {}).get("is_synthetic_fixture"))),
            },
            payload,
        )
        log.info("analysis %s track=%s model=%s prediction=%s", analysis_id, track, lm.entry.model_id, pred.label)
        return self.public_payload(payload)

    @staticmethod
    def public_payload(payload: dict[str, Any]) -> dict[str, Any]:
        files = payload.get("_files", {})
        out = {k: v for k, v in payload.items() if not k.startswith("_")}
        out["artifacts"] = {name: f"/api/analyses/{payload['id']}/artifacts/{name}" for name in ARTIFACT_NAMES if name in files}
        return out

    def get(self, analysis_id: str) -> dict[str, Any]:
        row = self.repo.get(safe_id(analysis_id, "analysis"))
        if row is None:
            raise NotFoundError("Analysis not found")
        return self.public_payload(row["payload"])

    def delete(self, analysis_id: str) -> None:
        safe_id(analysis_id, "analysis")
        if self.repo.get(analysis_id) is None:
            raise NotFoundError("Analysis not found")
        shutil.rmtree(self._dir(analysis_id), ignore_errors=True)
        self.repo.delete(analysis_id)

    def raw_payload(self, analysis_id: str) -> dict[str, Any]:
        row = self.repo.get(safe_id(analysis_id, "analysis"))
        if row is None:
            raise NotFoundError("Analysis not found")
        return row["payload"]


def image_bytes_to_png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
