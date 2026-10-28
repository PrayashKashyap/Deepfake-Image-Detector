# Troubleshooting

| Problem | Cause / fix |
|---|---|
| UI: "Cannot reach the backend at http://127.0.0.1:8000" | Start the backend (`scripts\run-backend.ps1`). Check `VITE_API_BASE_URL` in `frontend\.env.local` |
| Browser console shows a CORS error | Add the UI origin to `DFF_CORS_ORIGINS` and restart the backend |
| Analyze returns 503 "No trained model is available" | Expected until a checkpoint is installed in `artifacts\checkpoints\<track>\`. Train with `--install` (docs/TRAINING.md) |
| Model listed as `invalid` | The `.pt`/`.json` pair is unreadable or inconsistent (wrong track directory, class order or format). The error text is shown in Settings |
| 415 Unsupported file type | Only real JPEG/PNG content is accepted; a renamed GIF/WEBP is rejected |
| 413 / "too many pixels" | Raise `DFF_MAX_UPLOAD_MB` or `DFF_MAX_IMAGE_PIXELS` |
| 401 Unauthorized | `DFF_API_KEY` is set: enter the key under Settings → API key |
| Evaluation job: "Dataset is not configured" | Add the dataset to `ml\configs\datasets.local.yaml` (or `DFF_DATASET_<NAME>_ROOT/_MANIFEST`) |
| Evaluation job: "path_not_allowed" | Direct paths must lie inside `DFF_DATA_ROOTS`. Use a named dataset instead |
| Job shows "Interrupted" | The server restarted during the job. Start it again |
| `ModuleNotFoundError: ml` | Run `pip install -e .` in the repo root, inside the venv |
| `uvicorn: app.main not found` | Run from the repo root with `--app-dir backend` |
| PowerShell blocks scripts | `powershell -ExecutionPolicy Bypass -File scripts\...` or `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Garbled characters in console output | `$env:PYTHONIOENCODING="utf-8"` |
| CUDA problems | See docs/INSTALL_WINDOWS.md §6 |
| `KeyError` about a source in experiment configs | Source names must match the `source` column; check `<manifest>.summary.json` |
| Per-source ROC-AUC is "—" | Undefined for single-class subsets; use detection rate or specificity |
| Training dies with `MemoryError` in a DataLoader worker, or `Couldn't open shared file mapping ... error code 1455` | Windows page-file (commit) exhaustion: each worker process loads its own PyTorch/CUDA DLLs. Use `train.num_workers=3` or fewer (the default configs already do), close other memory-heavy programs, or enlarge the page file. Resume with `--resume <run>\checkpoints\last.pt` |
| Range download of GenImage stops with `HTTP Error 403` | The signed CDN URL expired; `ml.data.remote_zip` re-resolves it automatically. Re-run the same command: existing files are skipped |
