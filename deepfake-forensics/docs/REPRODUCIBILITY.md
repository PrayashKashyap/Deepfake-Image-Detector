# Reproducibility guide

## What is fixed automatically
- **Environment:** pinned Python requirements (`backend/requirements*.txt`) and an exact npm lockfile (`frontend/package-lock.json`).
- **Splits:** manifests are deterministic for a given seed, and their SHA-256 prefix is recorded in every checkpoint and result.
- **Training:** `set_seed` covers Python, NumPy and torch (CPU and CUDA). The DataLoader shuffling generator is seeded. Augmentation RNGs derive from torch's per-worker seeds. `train.deterministic: true` additionally enables deterministic cuDNN, which is slower.
- **Configuration:** `config.resolved.yaml` and a config hash are stored for every run. Checkpoints embed the preprocessing configuration, and the API uses exactly that configuration.
- **Evaluation:** noise perturbations use per-image seeds; bootstrap resampling uses the evaluation `seed`.

Bit-exact GPU results are not guaranteed: cuDNN autotuning and atomic operations introduce small nondeterminism. Report means over seeds.

## Recipe
```powershell
python -m ml.data.genimage --root <GenImage> --out data\manifests\genimage_v1.csv --max-per-class-per-source 2000 --seed 42
python -m ml.data.validate --manifest data\manifests\genimage_v1.csv --root <GenImage> --near-duplicates
python -m ml.experiments.run --experiment ml\configs\experiments\exp1_baseline_aigen.yaml
```

## What to archive with a paper
1. Git commit hash and `pip freeze` output.
2. Manifests (`data/manifests/*.csv`) plus their `.summary.json` and validation reports. These contain relative paths only.
3. `artifacts/experiments/**/{config.resolved.yaml,run.json,history.csv}` and `experiment_runs.json`.
4. `artifacts/evaluations/**/result.json`, `predictions.csv` and `reports/**`.
5. Checkpoints, if the licenses allow sharing derived models.

## Software smoke test
`scripts\smoke-pipeline.ps1` runs the whole pipeline on the synthetic fixture in a temporary directory. It verifies the software, not the science.
