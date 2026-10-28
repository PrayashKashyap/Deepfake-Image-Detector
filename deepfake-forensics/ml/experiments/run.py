"""Run a declarative research experiment (train -> evaluate -> robustness -> compare).

An experiment YAML (ml/configs/experiments/*.yaml) lists variants. Each variant points
to a training config, optional overrides and the seeds to train. After training, each
best checkpoint is evaluated with the experiment's evaluation spec(s). Finally the
variants are aggregated with ml.experiments.compare.

    python -m ml.experiments.run --experiment ml/configs/experiments/exp1_baseline_aigen.yaml
    python -m ml.experiments.run --experiment ... --dry-run          # print the plan only
    python -m ml.experiments.run --experiment ... --skip-training --runs-from artifacts/experiments

A run manifest (artifacts/experiments/<experiment_id>/experiment_runs.json) records
every produced run/evaluation so the experiment can be resumed or audited.
"""
from __future__ import annotations

import argparse
import copy
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

from ml.common import artifacts_dir, load_yaml, resolve_repo_path, save_json, utc_now_iso
from ml.evaluation.evaluate import run_evaluation
from ml.experiments.compare import compare
from ml.training.config import load_train_config
from ml.training.train import run_training


def _latest_run(exp_id: str, seed: int) -> Path | None:
    base = artifacts_dir() / "experiments" / exp_id
    if not base.is_dir():
        return None
    cands = sorted((p for p in base.iterdir() if p.is_dir() and p.name.endswith(f"_s{seed}") and (p / "run.json").exists()), reverse=True)
    return cands[0] if cands else None


def run_experiment(spec_path: Path, dry_run: bool = False, skip_training: bool = False, reuse_completed: bool = False) -> dict[str, Any]:
    spec = load_yaml(spec_path)
    exp_id = spec["experiment_id"]
    seeds = spec.get("seeds", [42])
    evals: list[dict[str, Any]] = spec.get("evaluations") or [{"name": "test", "dataset": {"splits": ["test"]}}]
    plan, record = [], {"experiment_id": exp_id, "spec": spec_path.as_posix(), "started_at": utc_now_iso(), "variants": {}}
    for v in spec["variants"]:
        for seed in v.get("seeds", seeds):
            plan.append((v, seed))
    print(f"Experiment {exp_id}: {len(plan)} training run(s) x {len(evals)} evaluation(s)")
    for v, seed in plan:
        print(f"  - {v['label']}: config={v['train_config']} seed={seed} overrides={v.get('overrides', [])}")
    if dry_run:
        return record

    groups: dict[str, OrderedDict[str, list[Path]]] = {e["name"]: OrderedDict() for e in evals}
    for v, seed in plan:
        overrides = list(v.get("overrides", [])) + [f"train.seed={seed}", f"experiment_id={exp_id}__{v['key']}"]
        cfg = load_train_config(resolve_repo_path(v["train_config"]), overrides)
        existing = _latest_run(cfg["experiment_id"], seed) if (skip_training or reuse_completed) else None
        if existing is not None:
            run_dir = existing
            print(f"  reusing completed run {run_dir.name} for {v['label']} seed {seed}")
        elif skip_training:
            raise FileNotFoundError(f"No completed run for {cfg['experiment_id']} seed {seed}")
        else:
            summary = run_training(cfg)
            run_dir = Path(summary["run_dir"])
        ckpt = run_dir / "checkpoints" / "best.pt"
        vrec = record["variants"].setdefault(v["label"], [])
        entry = {"seed": seed, "run_dir": run_dir.as_posix(), "evaluations": {}}
        for e in evals:
            if v.get("evaluations") and e["name"] not in v["evaluations"]:
                continue  # variant-specific evaluation list (e.g. only its own held-out source)
            ecfg = copy.deepcopy(e)
            name = ecfg.pop("name")
            ecfg["checkpoint"] = ckpt.as_posix()
            ds = ecfg.setdefault("dataset", {})
            for k in ("name", "root", "manifest"):
                if not ds.get(k) and cfg["dataset"].get(k):
                    ds[k] = cfg["dataset"][k]
            ecfg["eval_id"] = f"{exp_id}__{v['key']}__s{seed}__{name}"
            ecfg["experiment_id"] = exp_id
            res_path = run_evaluation(ecfg, lambda f, m: print(f"    [{f * 100:5.1f}%] {m}", flush=True) if f in (0.0, 1.0) else None)
            groups[name].setdefault(v["label"], []).append(res_path)
            entry["evaluations"][name] = res_path.as_posix()
        vrec.append(entry)
        save_json(record, artifacts_dir() / "experiments" / exp_id / "experiment_runs.json")

    comp_type = spec.get("comparison_type", "comparison")
    baseline = spec.get("baseline")
    record["comparisons"] = {}
    for name, variants in groups.items():
        if not variants or any(e.get("perturbations") for e in evals if e["name"] == name):
            continue  # robustness results are compared within their own result files
        p = compare(f"{exp_id}__{name}", variants, comp_type, spec.get("description", ""), baseline)
        record["comparisons"][name] = p.as_posix()
    record["finished_at"] = utc_now_iso()
    save_json(record, artifacts_dir() / "experiments" / exp_id / "experiment_runs.json")
    print(f"Experiment {exp_id} finished.")
    return record


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-training", action="store_true", help="Reuse the latest completed run for each variant/seed")
    ap.add_argument("--reuse-completed", action="store_true", help="Reuse completed runs where they exist, train the rest")
    args = ap.parse_args(argv)
    try:
        run_experiment(resolve_repo_path(args.experiment), args.dry_run, args.skip_training, args.reuse_completed)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
