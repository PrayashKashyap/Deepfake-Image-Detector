# One-time setup on Windows 11 (PowerShell). Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1            # NVIDIA GPU (CUDA 13.0 wheels)
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Cpu       # CPU only
param([switch]$Cpu)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python -m pip install --upgrade pip
if ($Cpu) { .\.venv\Scripts\python -m pip install -r backend\requirements-torch-cpu.txt }
else      { .\.venv\Scripts\python -m pip install -r backend\requirements-torch-cuda.txt }
.\.venv\Scripts\python -m pip install -r backend\requirements-dev.txt
.\.venv\Scripts\python -m pip install -e .

if (-not (Test-Path .env)) { Copy-Item .env.example .env }
if (-not (Test-Path ml\configs\datasets.local.yaml)) { Copy-Item ml\configs\datasets.example.yaml ml\configs\datasets.local.yaml }
if (-not (Test-Path frontend\.env.local)) { Copy-Item frontend\.env.example frontend\.env.local }

Push-Location frontend
npm ci
Pop-Location

.\.venv\Scripts\python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
Write-Host "Setup complete. Start the API with scripts\run-backend.ps1 and the UI with scripts\run-frontend.ps1"
