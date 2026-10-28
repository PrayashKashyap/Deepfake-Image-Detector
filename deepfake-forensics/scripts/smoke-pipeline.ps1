# SOFTWARE SMOKE TEST of the full ML pipeline on the SYNTHETIC fixture.
# Writes into a temporary artifacts directory so nothing fixture-derived reaches artifacts\.
# The resulting numbers are meaningless and must never be reported.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$tmp = Join-Path $env:TEMP "dff-smoke"
$env:DFF_ARTIFACTS_DIR = "$tmp\artifacts"
$env:DFF_REPORTS_DIR = "$tmp\reports"
.\.venv\Scripts\python -m ml.data.fixture --out data\fixtures\synthetic
.\.venv\Scripts\python -m ml.data.validate --manifest data\fixtures\synthetic\ai_generated_manifest.csv --root data\fixtures\synthetic --near-duplicates
.\.venv\Scripts\python -m ml.experiments.run --experiment ml\configs\experiments\fixture_smoke.yaml
Write-Host "Smoke outputs in $tmp (synthetic, not results)."
