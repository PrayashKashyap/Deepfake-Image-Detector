"""Evaluate a checkpoint on a manifest split, optionally under perturbations.

    python -m ml.evaluation.evaluate --checkpoint artifacts/checkpoints/ai_generated/<model>.pt `
        --dataset genimage_v1 --split test
    python -m ml.evaluation.evaluate --config ml/configs/eval/exp2_cross_generator_eval.yaml

The threshold used is the one stored in the checkpoint (selected on validation data
during training). The test set is never used to choose thresholds or hyperparameters.

Output: artifacts/evaluations/<eval_id>/result.json (+ predictions.csv, roc.png,
confusion_matrix.png, per-perturbation files for robustness runs).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ml.common import (
    RESULT_SCHEMA_VERSION,
    artifacts_dir,
    device_info,
    load_yaml,
    resolve_device,
    resolve_repo_path,
    save_json,
    timestamp_id,
    utc_now_iso,
)
from ml.data.dataset import ManifestImageDataset
from ml.data.manifest import filter_manifest, manifest_fingerprint, read_manifest
from ml.data.registry import resolve_dataset
from ml.evaluation.metrics import compute_binary_metrics, grouped_bootstrap_ci, per_class_table, roc_points
from ml.evaluation.plots import plot_bar, plot_confusion_matrix, plot_roc
from ml.models.checkpoint import load_model_for_inference
from ml.preprocessing.transforms import PreprocessConfig, build_eval_transform
from ml.robustness.perturbations import Perturbation
from ml.training.engine import predict

ProgressFn = Callable[[float, str], None]

EVAL_DEFAULTS: dict[str, Any] = {
    "eval_id": None,
    "experiment_id": None,
    "description": "",
    "checkpoint": None,
    "dataset": {"name": None, "root": None, "manifest": None, "splits": ["test"], "include_sources": [], "exclude_sources": [], "max_per_class": None},
    "batch_size": 64,
    "num_workers": 4,
    "amp": True,
    "device": "auto",
    "seed": 0,
    "bootstrap": {"enabled": True, "n_boot": 1000},
    "perturbations": None,  # list of perturbation dicts -> robustness evaluation
    "latency": {"enabled": True, "n_images": 64},
    "output_dir": None,  # default artifacts/evaluations
}


def _merge(base: dict, upd: dict) -> dict:
    out = dict(base)
    for k, v in (upd or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def validate_eval_config(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = _merge(EVAL_DEFAULTS, cfg)
    if not cfg.get("checkpoint"):
        raise ValueError("checkpoint is required")
    ckpt = resolve_repo_path(cfg["checkpoint"])
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    ds = cfg["dataset"]
    if not ds.get("name") and not (ds.get("root") and ds.get("manifest")):
        raise ValueError("dataset.name or dataset.root + dataset.manifest required")
    if not ds.get("splits"):
        raise ValueError("dataset.splits must be non-empty")
    bad = [s for s in ds["splits"] if s not in ("train", "val", "test")]
    if bad:
        raise ValueError(f"Unknown split(s): {bad}")
    if int(cfg["batch_size"]) < 1 or int(cfg["batch_size"]) > 1024:
        raise ValueError("batch_size must be in [1, 1024]")
    if cfg.get("perturbations"):
        cfg["perturbations"] = [Perturbation.from_dict(p) for p in cfg["perturbations"]]
    return cfg


def _resolve_data(ds: dict[str, Any]) -> tuple[Path, Path]:
    if ds.get("root") and ds.get("manifest"):
        root, manifest = resolve_repo_path(ds["root"]), resolve_repo_path(ds["manifest"])
    else:
        ref = resolve_dataset(ds["name"])
        root, manifest = ref.root, ref.manifest
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    return root, manifest


def measure_latency(model, preprocess: PreprocessConfig, device: torch.device, amp: bool, n: int = 64) -> dict[str, Any]:
    """Single-image end-to-end model latency (forward pass incl. host->device copy), batch size 1."""
    x = torch.randn(1, 3, preprocess.img_size, preprocess.img_size)
    times = []
    with torch.no_grad():
        for i in range(n + 5):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            xd = x.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
                model(xd)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= 5:  # warm-up excluded
                times.append((time.perf_counter() - t0) * 1000)
    t = np.asarray(times)
    return {"batch_size": 1, "n": int(len(t)), "mean_ms": float(t.mean()), "p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)), "device": str(device), "amp": bool(amp and device.type == "cuda")}


def per_source_metrics(df: pd.DataFrame, scores: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    out = []
    for src, sub in df.assign(_s=scores).groupby("source"):
        m = compute_binary_metrics(sub["label"].to_numpy(), sub["_s"].to_numpy(), threshold)
        labels = sorted(sub["label"].unique().tolist())
        out.append({
            "source": src,
            "n": m["n"],
            "labels_present": ["REAL" if l == 0 else "FAKE" for l in labels],
            "accuracy": m["accuracy"],
            # For single-class sources accuracy == detection rate (FAKE) or specificity (REAL).
            "detection_rate": m["recall_fake"] if 1 in labels else None,
            "specificity": m["recall_real"] if 0 in labels else None,
            "roc_auc": m["roc_auc"],
            "mean_fake_score": float(sub["_s"].mean()),
        })
    return out


def _evaluate_condition(model, meta, df, root, cfg, device, amp, out_dir: Path, pert: Perturbation | None, progress: ProgressFn | None, frac: tuple[float, float]):
    pcfg = PreprocessConfig.from_dict(meta["preprocess"])
    nw = int(cfg["num_workers"])
    dset = ManifestImageDataset(df, root, build_eval_transform(pcfg), perturbation=None if pert is None else {"type": pert.type, **pert.params}, perturbation_seed=int(cfg["seed"]))
    loader = DataLoader(dset, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=nw, pin_memory=device.type == "cuda", persistent_workers=False)
    lo, hi = frac

    def _p(done: int, total: int) -> None:
        if progress:
            progress(lo + (hi - lo) * done / max(total, 1), f"{pert.name if pert else 'clean'}: batch {done}/{total}")

    res = predict(model, loader, device, amp, progress=_p)
    order = np.argsort(res["indices"])
    scores, labels = res["scores"][order], res["labels"][order]
    thr = float(meta["threshold"]["value"])
    m = compute_binary_metrics(labels, scores, thr)
    m_05 = compute_binary_metrics(labels, scores, 0.5)
    roc = roc_points(labels, scores)
    return scores, labels, m, m_05, roc, res


def run_evaluation(cfg: dict[str, Any], progress: ProgressFn | None = None) -> Path:
    cfg = validate_eval_config(cfg)
    device = resolve_device(cfg["device"])
    amp = bool(cfg["amp"]) and device.type == "cuda"
    ckpt_path = resolve_repo_path(cfg["checkpoint"])
    root, manifest_path = _resolve_data(cfg["dataset"])
    ds = cfg["dataset"]
    df_all = read_manifest(manifest_path)
    df = filter_manifest(df_all, ds["splits"], ds["include_sources"], ds["exclude_sources"], ds["max_per_class"], int(cfg["seed"]))
    if len(df) == 0:
        raise ValueError("No samples left after filtering the manifest")
    if progress:
        progress(0.02, "loading model")
    model, meta = load_model_for_inference(ckpt_path, device)

    # Leakage guard: refuse to evaluate on sources that were explicitly part of training when
    # the experiment declares them as held-out.
    train_sources = set(meta.get("dataset", {}).get("train_sources", []))
    eval_id = cfg.get("eval_id") or f"{timestamp_id()}_{meta.get('model_id', ckpt_path.stem)}"[:120]
    out_dir = resolve_repo_path(cfg["output_dir"]) / eval_id if cfg.get("output_dir") else artifacts_dir() / "evaluations" / eval_id
    out_dir.mkdir(parents=True, exist_ok=True)

    perturbations: list[Perturbation] = cfg["perturbations"] or []
    is_robustness = bool(perturbations)
    if is_robustness and perturbations[0].type != "clean":
        perturbations = [Perturbation("clean", {})] + perturbations
    conditions = perturbations or [None]

    results: list[dict[str, Any]] = []
    clean_payload = None
    for k, pert in enumerate(conditions):
        frac = (0.05 + 0.85 * k / len(conditions), 0.05 + 0.85 * (k + 1) / len(conditions))
        scores, labels, m, m_05, roc, res = _evaluate_condition(model, meta, df, root, cfg, device, amp, out_dir, pert, progress, frac)
        name = pert.name if pert else "clean"
        entry = {"name": name, "perturbation": None if pert is None else {"type": pert.type, **pert.params}, "metrics": m, "metrics_at_0_5": m_05, "roc_curve": roc, "throughput_images_per_s": res["n"] / max(res["model_seconds"], 1e-9)}
        results.append(entry)
        if pert is None or pert.type == "clean":
            clean_payload = (scores, labels, m, roc)

    scores, labels, m, roc = clean_payload  # type: ignore[misc]
    pred_df = df[["path", "label", "source", "group", "split"]].copy()
    pred_df["fake_score"] = scores
    pred_df["predicted"] = (scores >= float(meta["threshold"]["value"])).astype(int)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    plot_confusion_matrix(m["confusion_matrix"], out_dir / "confusion_matrix.png")
    plot_roc({"clean": {**(roc or {}), "auc": m["roc_auc"]}} if roc else {}, out_dir / "roc.png")

    ci = None
    if cfg["bootstrap"]["enabled"] and len(df) >= 20:
        if progress:
            progress(0.92, "bootstrap confidence intervals")
        ci = grouped_bootstrap_ci(labels, scores, df["group"].to_numpy(), float(meta["threshold"]["value"]), n_boot=int(cfg["bootstrap"]["n_boot"]), seed=int(cfg["seed"]))

    latency = measure_latency(model, PreprocessConfig.from_dict(meta["preprocess"]), device, amp, int(cfg["latency"]["n_images"])) if cfg["latency"]["enabled"] else None

    eval_sources = sorted(df["source"].unique().tolist())
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": "robustness" if is_robustness else "evaluation",
        "eval_id": eval_id,
        "experiment_id": cfg.get("experiment_id") or meta.get("training", {}).get("experiment_id"),
        "description": cfg.get("description", ""),
        "created_at": utc_now_iso(),
        "track": meta["track"],
        "model": {
            "model_id": meta.get("model_id"),
            "version": meta.get("version"),
            "arch": meta.get("arch"),
            "frequency_branch": meta.get("frequency_branch"),
            "checkpoint_file": ckpt_path.name,
            "training_run": meta.get("training", {}).get("run_id"),
            "training_seed": meta.get("training", {}).get("seed"),
        },
        "dataset": {
            "dataset_ids": sorted(df["dataset_id"].unique().tolist()),
            "manifest": manifest_path.name,
            "manifest_sha256_16": manifest_fingerprint(manifest_path),
            "splits": ds["splits"],
            "include_sources": ds["include_sources"],
            "exclude_sources": ds["exclude_sources"],
            "eval_sources": eval_sources,
            "train_sources": sorted(train_sources),
            "unseen_sources": sorted(set(eval_sources) - train_sources) if train_sources else None,
            "n_samples": int(len(df)),
            "class_counts": {"real": int((df.label == 0).sum()), "fake": int((df.label == 1).sum())},
            "is_synthetic_fixture": bool(meta.get("dataset", {}).get("is_synthetic_fixture")) or any(str(d).startswith("SYNTHETIC_FIXTURE") for d in df["dataset_id"].unique()),
        },
        "seed": int(cfg["seed"]),
        "threshold": meta["threshold"],
        "metrics": m,
        "per_class": per_class_table(m),
        "confusion_matrix": m["confusion_matrix"],
        "roc_curve": roc,
        "per_source": per_source_metrics(df, scores, float(meta["threshold"]["value"])),
        "confidence_intervals": ci,
        "latency": latency,
        "environment": {k: v for k, v in device_info().items() if k != "devices"},
        "artifacts": {"predictions_csv": "predictions.csv", "confusion_matrix_png": "confusion_matrix.png", "roc_png": "roc.png"},
    }

    if is_robustness:
        clean = results[0]["metrics"]
        rows = []
        for r in results:
            mm = r["metrics"]
            rows.append({
                "name": r["name"],
                "perturbation": r["perturbation"],
                "accuracy": mm["accuracy"],
                "f1_macro": mm["f1_macro"],
                "roc_auc": mm["roc_auc"],
                "recall_fake": mm["recall_fake"],
                "recall_real": mm["recall_real"],
                "delta_accuracy": None if mm["accuracy"] is None else mm["accuracy"] - clean["accuracy"],
                "delta_f1_macro": mm["f1_macro"] - clean["f1_macro"],
                "delta_roc_auc": None if (mm["roc_auc"] is None or clean["roc_auc"] is None) else mm["roc_auc"] - clean["roc_auc"],
            })
        result["robustness"] = {"reference": "clean", "conditions": rows, "detail": results}
        pd.DataFrame(rows).drop(columns=["perturbation"]).to_csv(out_dir / "robustness.csv", index=False)
        plot_bar([r["name"] for r in rows], {"ROC-AUC": [r["roc_auc"] for r in rows], "macro-F1": [r["f1_macro"] for r in rows]}, out_dir / "robustness.png", "Robustness to perturbations", "score")
        plot_roc({r["name"]: {**(r["roc_curve"] or {}), "auc": r["metrics"]["roc_auc"]} for r in results if r["roc_curve"]}, out_dir / "roc_robustness.png", "ROC under perturbations")
        result["artifacts"].update({"robustness_csv": "robustness.csv", "robustness_png": "robustness.png", "roc_robustness_png": "roc_robustness.png"})

    save_json(result, out_dir / "result.json")
    pd.DataFrame([{"eval_id": eval_id, **{k: v for k, v in m.items() if k != "confusion_matrix"}}]).to_csv(out_dir / "metrics.csv", index=False)
    result["artifacts"]["metrics_csv"] = "metrics.csv"
    save_json(result, out_dir / "result.json")
    if progress:
        progress(1.0, "done")
    return out_dir / "result.json"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="Evaluation YAML (see ml/configs/eval)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--dataset", default=None, help="Dataset name from datasets.local.yaml")
    ap.add_argument("--root", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--split", nargs="+", default=None)
    ap.add_argument("--include-sources", nargs="*", default=None)
    ap.add_argument("--exclude-sources", nargs="*", default=None)
    ap.add_argument("--robustness", action="store_true", help="Run the default robustness suite (JPEG 90/70/50, downscale, blur, noise)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--eval-id", default=None)
    args = ap.parse_args(argv)

    cfg: dict[str, Any] = load_yaml(args.config) if args.config else {}
    cfg.setdefault("dataset", {})
    if args.checkpoint:
        cfg["checkpoint"] = args.checkpoint
    if args.dataset:
        cfg["dataset"]["name"] = args.dataset
    if args.root:
        cfg["dataset"]["root"] = args.root
    if args.manifest:
        cfg["dataset"]["manifest"] = args.manifest
    if args.split:
        cfg["dataset"]["splits"] = args.split
    if args.include_sources is not None:
        cfg["dataset"]["include_sources"] = args.include_sources
    if args.exclude_sources is not None:
        cfg["dataset"]["exclude_sources"] = args.exclude_sources
    if args.robustness and not cfg.get("perturbations"):
        from ml.robustness.perturbations import DEFAULT_ROBUSTNESS_SUITE

        cfg["perturbations"] = DEFAULT_ROBUSTNESS_SUITE
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.eval_id:
        cfg["eval_id"] = args.eval_id

    def _progress(f: float, msg: str) -> None:
        print(f"[{f * 100:5.1f}%] {msg}", flush=True)

    try:
        path = run_evaluation(cfg, _progress)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"Result written to {path}")


if __name__ == "__main__":
    main()
