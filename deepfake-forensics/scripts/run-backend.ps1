# Start the FastAPI backend (http://127.0.0.1:8000, docs at /docs)
Set-Location (Split-Path $PSScriptRoot -Parent)
.\.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
