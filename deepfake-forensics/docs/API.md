# REST API

Base URL: `http://127.0.0.1:8000`. Interactive OpenAPI docs are at `/docs` (Swagger) and `/redoc`; the schema is at `/openapi.json`. If `DFF_API_KEY` is set, send `X-API-Key: <key>` on every `/api` request.

All errors use one shape: `{"error": {"code": "...", "message": "...", "details": ...}}`.

| Method | Path | Purpose | Notable responses |
|---|---|---|---|
| GET | `/api/health` | Status, version, device/CUDA info, service status | 200 |
| GET | `/api/models` | Installed checkpoints, track status, architecture status (trained / pretrained / unavailable) | 200 |
| PUT | `/api/models/{track}/active` | `{"model_id": "..." \| null}` selects a checkpoint | 404 unknown model/track, 503 invalid checkpoint |
| GET / PUT | `/api/settings` | Read settings / set per-track threshold overrides `{"thresholds": {"ai_generated": 0.6}}` (`null` resets) | 422 outside (0, 1) |
| GET | `/api/datasets` | Named datasets (names only; paths are never returned) | 200 |
| POST | `/api/analyze` | multipart: `file` (JPEG/PNG), `track` (`ai_generated`\|`face_deepfake`) | 200 result · 413 too large · 415 not JPEG/PNG by content · 422 malformed/too small · 503 `model_unavailable` |
| GET | `/api/analyses` | History. Query: `page`, `page_size` (≤100), `prediction`, `track`, `date_from`, `date_to`, `q` | 200 |
| GET | `/api/analyses/{id}` | One analysis | 404 |
| DELETE | `/api/analyses/{id}` | Delete the record **and** its stored files | 204, 404 |
| GET | `/api/analyses/{id}/artifacts/{name}` | `original`, `input`, `gradcam`, `heatmap`, `spectrum` | 404 |
| GET | `/api/analyses/{id}/report` | PDF report (attachment) | 200 `application/pdf` |
| GET | `/api/forensics/{id}` | Measured properties, JPEG/PNG details, EXIF (sensitive fields redacted), spectrum profile, observations | 404 |
| GET | `/api/stats` | Aggregate counts of recorded analyses | 200 |
| POST | `/api/robustness/analyze` | multipart: `file`, `track`, and at least one of `jpeg_quality` (5–100), `downscale_max_side` (16–4096), `blur_sigma` (0–10), `noise_std` (0–100) | 422 invalid/no parameters · 503 |
| GET | `/api/evaluation/summary` | All results plus the latest evaluation per track; `available=false` when there are none | 200 |
| GET | `/api/evaluation/experiments` | List (filter `kind`, `track`) | 200 |
| GET | `/api/evaluation/experiments/{id}` | Full result JSON plus artifact list (`eval--<id>` or `train--<exp>--<run>`) | 404 |
| GET | `/api/evaluation/experiments/{id}/artifacts/{name}` | PNG/CSV/JSON/MD/TEX artifact | 404 |
| GET | `/api/evaluation/metrics.csv` | Export metrics of all (or `ids=`) experiments | 200 `text/csv` |
| POST | `/api/evaluation/run` | Start a background evaluation job (body below) | 202 `{job_id}` · 404 model · 422 dataset/path |
| GET | `/api/jobs` / `/api/jobs/{id}` | Job list / status, progress, last log lines, result link | 404 |

`POST /api/evaluation/run` body:

```json
{"model_id": "aigen_efficientnet_b0__20261101-101500_s42", "dataset_name": "genimage_v1",
 "splits": ["test"], "include_sources": [], "exclude_sources": [], "max_per_class": 2000,
 "batch_size": 32, "num_workers": 0, "robustness": false, "bootstrap": true}
```

Instead of `dataset_name` you may pass `dataset_root` and `manifest`, which must lie inside `DFF_DATA_ROOTS`.

Example (PowerShell 7+ `curl.exe`):

```powershell
curl.exe -F "file=@C:\images\photo.jpg;type=image/jpeg" -F "track=ai_generated" http://127.0.0.1:8000/api/analyze
```

Key fields of the analysis response: `prediction`, `fake_score` (softmax for FAKE), `real_score`, `threshold`, `threshold_source`, `inference_ms`, `gradcam_ms`, `device`, `model{model_id, version, arch, is_synthetic_fixture, trained_on}`, `image{...}`, `explanation{available, explained_class, target_layer, note}`, `forensics{...}`, `artifacts{name: url}`, `warnings[]`, `score_note`, `disclaimer`.

Security behaviour:
- File contents are validated (magic bytes plus a full decode). Extensions are ignored.
- Stored files get generated names inside `storage/analyses/<uuid4>/`.
- Ids are validated and joined paths are checked to stay inside their base directory.
- Uploaded files are never executed. Image content and metadata are not logged.
