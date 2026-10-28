# Run all automated tests (backend pytest + frontend vitest + TypeScript check)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
.\.venv\Scripts\python -m pytest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Push-Location frontend
npm run typecheck
if ($LASTEXITCODE -ne 0) { Pop-Location; exit $LASTEXITCODE }
npm test
$code = $LASTEXITCODE
Pop-Location
exit $code
