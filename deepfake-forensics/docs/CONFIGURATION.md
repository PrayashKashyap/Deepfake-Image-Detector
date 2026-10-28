# Configuration reference

The backend reads environment variables with the prefix `DFF_`, from the process environment or from `.env` in the repository root (also `backend/.env`). Relative paths resolve against the repository root. Template: [`.env.example`](../.env.example).

| Variable | Default | Description |
|---|---|---|
| `DFF_ENVIRONMENT` | `development` | Free-form label |
| `DFF_LOG_LEVEL` | `INFO` | Python log level. Image contents and metadata values are never logged |
| `DFF_STORAGE_DIR` | `storage` | Uploads and generated analysis images (`storage/analyses/<uuid>/`) |
| `DFF_DATABASE_PATH` | `storage/app.db` | SQLite database (analyses, settings, jobs) |
| `DFF_ARTIFACTS_DIR` | `artifacts` | `checkpoints/<track>/`, `evaluations/`, `experiments/`. Also used by the `ml` CLIs |
| `DFF_REPORTS_DIR` | `reports` | Comparison tables and figures written by `ml.experiments.compare` |
| `DFF_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated allowed origins |
| `DFF_API_KEY` | _(empty)_ | If set, all `/api/*` requests require header `X-API-Key`. Stored images are served only through these authenticated endpoints |
| `DFF_MAX_UPLOAD_MB` | `15` | Upload size limit (HTTP 413 above it) |
| `DFF_MAX_IMAGE_PIXELS` | `40000000` | Decoded pixel limit (decompression-bomb protection) |
| `DFF_MIN_IMAGE_SIDE` | `32` | Minimum width/height |
| `DFF_DEVICE` | `auto` | `auto` (CUDA if available, else CPU), `cpu`, `cuda`, `cuda:N` |
| `DFF_INFERENCE_AMP` | `true` | float16 autocast on CUDA (same setting as the evaluation default) |
| `DFF_MODEL_CACHE_SIZE` | `2` | Number of loaded models kept in memory (LRU) |
| `DFF_ENABLE_GRADCAM` | `true` | Generate Grad-CAM overlays for analyses |
| `DFF_DATA_ROOTS` | `data` | Directories that may contain dataset roots/manifests passed **directly** to `POST /api/evaluation/run` |
| `DFF_JOB_WORKERS` | `1` | Concurrent background jobs |
| `DFF_DATASETS_CONFIG` | `ml/configs/datasets.local.yaml` | Named-dataset registry |
| `DFF_DATASET_<NAME>_ROOT` / `_MANIFEST` | — | Per-dataset override of the registry, e.g. `DFF_DATASET_GENIMAGE_V1_ROOT` |

Frontend (`frontend/.env.local`):

| Variable | Default | Description |
|---|---|---|
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Backend base URL |

Runtime settings changed in the UI (per-track threshold overrides and active checkpoints) are stored in the SQLite `settings` table.
