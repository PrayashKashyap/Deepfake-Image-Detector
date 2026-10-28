# Installation on Windows 11

Verified environment: Windows 11, Python 3.14.5, Node.js 24.20 / npm 11.19, NVIDIA RTX 4060 Laptop GPU (8 GB), driver 591.86, PyTorch 2.14.1 + CUDA 13.0 wheels.

## 1. Prerequisites

| Tool | Check | Notes |
|---|---|---|
| Python ≥ 3.11 | `python --version` | Install from python.org; tick "Add to PATH" |
| Node.js ≥ 20 | `node --version` | LTS from nodejs.org |
| Git | `git --version` | |
| NVIDIA driver (optional) | `nvidia-smi` | The CUDA 13.0 wheels need driver ≥ 580. A separate CUDA Toolkit is **not** required |

## 2. Automated setup

```powershell
cd C:\path\to\deepfake-forensics
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1          # GPU
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Cpu     # CPU only
```

## 3. Manual backend setup

```powershell
cd C:\path\to\deepfake-forensics
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# PyTorch first: pick ONE
pip install -r backend\requirements-torch-cuda.txt    # NVIDIA GPU
pip install -r backend\requirements-torch-cpu.txt     # CPU
pip install -r backend\requirements-dev.txt           # runtime + test dependencies (pinned)
pip install -e .                                      # installs the shared `ml` package (editable)
Copy-Item .env.example .env
Copy-Item ml\configs\datasets.example.yaml ml\configs\datasets.local.yaml
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs for the interactive Swagger UI. If `Activate.ps1` is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.\.venv\Scripts\python` directly.

## 4. Manual frontend setup

```powershell
cd frontend
npm ci                                 # exact versions from package-lock.json
Copy-Item .env.example .env.local      # VITE_API_BASE_URL=http://127.0.0.1:8000
npm run dev                            # http://127.0.0.1:5173
npm run build                          # production build in frontend\dist
```

If you change the frontend port or host, add it to `DFF_CORS_ORIGINS` in `.env`.

## 5. Tests

```powershell
.\.venv\Scripts\python -m pytest            # backend (uses temp dirs, CPU, a tiny random checkpoint)
cd frontend; npm run typecheck; npm test    # frontend
```

## 6. CUDA and GPU troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is `False` | `pip show torch`: the version must end in `+cu130`. If it shows `+cpu`, reinstall with `requirements-torch-cuda.txt`. Run `nvidia-smi`: the "CUDA Version" must be ≥ 13.0, otherwise update the driver, or use the cu128 index (`--index-url https://download.pytorch.org/whl/cu128`; Python 3.14 is supported from torch 2.9) |
| `CUDA out of memory` during training | Lower `train.batch_size` (e.g. `--set train.batch_size=24`), keep `train.amp: true`, close other GPU apps. 8 GB fits ResNet-50 at 224 px with batch ≈ 48 under AMP |
| Laptop runs on the integrated GPU | NVIDIA Control Panel → Manage 3D settings → Program settings → set `python.exe` (in `.venv`) to "High-performance NVIDIA processor" |
| Slow first request | The model loads lazily and cuDNN autotunes on the first call. Later requests reuse the cached model |
| `DataLoader` worker errors on Windows | Set `train.num_workers=0` to diagnose. Workers use spawn; run the scripts as `python -m ...` |
| Force CPU | Set `DFF_DEVICE=cpu` in `.env` (API) or `--set train.device=cpu` (training) |

Check the GPU from Python:

```powershell
.\.venv\Scripts\python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
```
