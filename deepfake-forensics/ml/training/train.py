"""Train a detector from a YAML config.

    python -m ml.training.train --config ml/configs/train/aigen_resnet50.yaml
    python -m ml.training.train --config ml/configs/train/aigen_resnet50.yaml --set train.epochs=5 train.batch_size=16
    python -m ml.training.train --config ... --resume artifacts/experiments/<exp>/<run>/checkpoints/last.pt
    python -m ml.training.train --config ... --install      # copy best checkpoint into artifacts/checkpoints/<track>/

Outputs (artifacts/experiments/<experiment_id>/<run_id>/):
    config.resolved.yaml, history.csv, run.json, loss_curve.png, val_metrics_curve.png,
    checkpoints/best.pt (+ .json), checkpoints/last.pt (+ .json), train.log

The decision threshold stored in best.pt is selected on the VALIDATION split only.
Training is never started automatically by the application.
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ml.common import (
    CLASS_NAMES,
    FAKE,
    REAL,
    RESULT_SCHEMA_VERSION,
    artifacts_dir,
    device_info,
    resolve_device,
    resolve_repo_path,
    save_json,
    save_yaml,
    set_seed,
    sha256_text,
    timestamp_id,
    utc_now_iso,
)
from ml.data.dataset import ManifestImageDataset
from ml.data.manifest import filter_manifest, manifest_fingerprint, read_manifest, summarize_manifest
from ml.data.registry import resolve_dataset
from ml.evaluation.metrics import compute_binary_metrics, select_threshold
from ml.models.checkpoint import load_checkpoint_payload, save_checkpoint
from ml.models.factory import PRETRAINED_SOURCES, ModelConfig, build_model, count_parameters
from ml.preprocessing.transforms import AugmentConfig, PreprocessConfig, build_eval_transform, build_train_transform
from ml.training.config import MONITORS, load_train_config
from ml.training.engine import predict, train_one_epoch
from ml.training.plots import plot_history


class _Logger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        self.fh.write(msg + "\n")
        self.fh.flush()


def resolve_data(cfg: dict[str, Any]) -> tuple[Path, Path]:
    ds = cfg["dataset"]
    if ds.get("root") and ds.get("manifest"):
        return resolve_repo_path(ds["root"]), resolve_repo_path(ds["manifest"])
    ref = resolve_dataset(ds["name"])
    return ref.root, ref.manifest


def build_scheduler(optimizer, tcfg: dict[str, Any], steps_per_epoch: int):
    epochs = int(tcfg["epochs"])
    kind = tcfg["scheduler"]
    if kind == "plateau":
        mode = MONITORS[tcfg["monitor"]]
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=0.5, patience=1)
    if kind == "none":
        return None
    warm = max(0, int(tcfg.get("warmup_epochs") or 0))
    total = max(1, epochs * steps_per_epoch)
    warm_steps = min(warm * steps_per_epoch, total - 1)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total - warm_steps), eta_min=float(tcfg["lr"]) * 0.01)
    if warm_steps <= 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warm_steps)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warm_steps])


class _StepSchedulerOptimizer:
    """Wrap an optimizer so a per-step LR scheduler advances after every optimizer.step()."""

    def __init__(self, opt, sched) -> None:
        self.opt, self.sched = opt, sched

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.opt.step()
        if self.sched is not None:
            self.sched.step()

    def __getattr__(self, name):
        return getattr(self.opt, name)


def run_training(cfg: dict[str, Any], resume: str | None = None, install: bool = False) -> dict[str, Any]:
    tcfg = cfg["train"]
    seed = int(tcfg["seed"])
    set_seed(seed, deterministic=bool(tcfg.get("deterministic")))
    device = resolve_device(tcfg.get("device", "auto"))
    amp = bool(tcfg["amp"]) and device.type == "cuda"

    if resume:
        resume_path = resolve_repo_path(resume)
        run_dir = resume_path.parent.parent
        run_id = run_dir.name
    else:
        run_id = f"{timestamp_id()}_s{seed}"
        out_base = resolve_repo_path(cfg["output_dir"]) if cfg.get("output_dir") else artifacts_dir() / "experiments"
        run_dir = out_base / cfg["experiment_id"] / run_id
    ckpt_dir = run_dir / "checkpoints"
    log = _Logger(run_dir / "train.log")
    save_yaml(cfg, run_dir / "config.resolved.yaml")
    config_sha = sha256_text(repr(sorted(cfg.items(), key=lambda kv: kv[0])))[:16]

    root, manifest_path = resolve_data(cfg)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    df = read_manifest(manifest_path)
    ds = cfg["dataset"]
    train_df = filter_manifest(df, [ds["train_split"]], ds["include_sources"], ds["exclude_sources"], ds["max_train_per_class"], seed)
    val_df = filter_manifest(df, [ds["val_split"]], ds["include_sources"], ds["exclude_sources"], ds["max_val_per_class"], seed)
    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(f"Empty split after filtering: train={len(train_df)} val={len(val_df)}")
    if train_df["label"].nunique() < 2:
        raise ValueError("Training split must contain both REAL and FAKE samples")
    is_fixture = any(str(d).startswith("SYNTHETIC_FIXTURE") for d in df["dataset_id"].unique())

    log(f"== {cfg['experiment_id']} / {run_id}")
    log(f"device={device} amp={amp} torch={torch.__version__}")
    log(f"train n={len(train_df)} (fake={int(train_df.label.sum())})  val n={len(val_df)} (fake={int(val_df.label.sum())})")
    if is_fixture:
        log("!! WARNING: training on the SYNTHETIC TEST FIXTURE — results are not meaningful.")

    pcfg = PreprocessConfig.from_dict(cfg["preprocess"])
    acfg = AugmentConfig.from_dict(cfg["augment"])
    mcfg = ModelConfig.from_dict(cfg["model"])
    if resume:
        mcfg.pretrained = False  # weights come from the checkpoint

    nw = int(tcfg["num_workers"])
    loader_kw = dict(num_workers=nw, pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        ManifestImageDataset(train_df, root, build_train_transform(pcfg, acfg)),
        batch_size=int(tcfg["batch_size"]), shuffle=True, drop_last=len(train_df) > int(tcfg["batch_size"]), generator=g, **loader_kw,
    )
    val_loader = DataLoader(
        ManifestImageDataset(val_df, root, build_eval_transform(pcfg)), batch_size=int(tcfg["batch_size"]), shuffle=False,
        # Fewer, non-persistent workers for validation: on Windows every worker process holds its own
        # PyTorch/CUDA DLL commit charge, so two persistent pools can exhaust the page file.
        num_workers=min(nw, 2), pin_memory=device.type == "cuda", persistent_workers=False,
    )

    model = build_model(mcfg).to(device)
    log(f"model={mcfg.arch} freq_branch={mcfg.frequency_branch} params={count_parameters(model):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    steps = len(train_loader) if not tcfg.get("max_steps_per_epoch") else min(len(train_loader), int(tcfg["max_steps_per_epoch"]))
    scheduler = build_scheduler(optimizer, tcfg, steps)
    per_step = scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    monitor, mode = tcfg["monitor"], MONITORS[tcfg["monitor"]]
    start_epoch, history = 1, []
    best_value = -math.inf if mode == "max" else math.inf
    best_epoch, bad_epochs = 0, 0
    if resume:
        payload = load_checkpoint_payload(resolve_repo_path(resume))
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and payload.get("scheduler_state"):
            scheduler.load_state_dict(payload["scheduler_state"])
        if payload.get("scaler_state"):
            scaler.load_state_dict(payload["scaler_state"])
        start_epoch = int(payload["epoch"]) + 1
        history = list(payload.get("history", []))
        best_value = float(payload.get("best_value", best_value))
        best_epoch = int(payload.get("best_epoch", 0))
        bad_epochs = int(payload.get("bad_epochs", 0))
        log(f"resumed from {resume} at epoch {start_epoch}")

    base_meta = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "model_id": f"{cfg['experiment_id']}__{run_id}",
        "display_name": f"{mcfg.arch}{' + FFT' if mcfg.frequency_branch else ''} ({cfg['track']})",
        "track": cfg["track"],
        "arch": mcfg.arch,
        "frequency_branch": mcfg.frequency_branch,
        "model_config": {**mcfg.to_dict(), "pretrained": bool(cfg["model"].get("pretrained", True))},
        "pretrained_init": PRETRAINED_SOURCES[mcfg.arch] if cfg["model"].get("pretrained", True) else None,
        "preprocess": pcfg.to_dict(),
        "class_names": list(CLASS_NAMES),
        "class_index": {"REAL": REAL, "FAKE": FAKE},
        "threshold": {"value": 0.5, "selected_on": "default (not yet selected)", "method": "fixed"},
        "dataset": {
            "dataset_ids": sorted(df["dataset_id"].unique().tolist()),
            "manifest": manifest_path.name,
            "manifest_sha256_16": manifest_fingerprint(manifest_path),
            "train_split": ds["train_split"],
            "val_split": ds["val_split"],
            "include_sources": ds["include_sources"],
            "exclude_sources": ds["exclude_sources"],
            "train_sources": sorted(train_df["source"].unique().tolist()),
            "train_counts": {"real": int((train_df.label == 0).sum()), "fake": int((train_df.label == 1).sum())},
            "is_synthetic_fixture": is_fixture,
        },
        "training": {
            "experiment_id": cfg["experiment_id"],
            "run_id": run_id,
            "seed": seed,
            "config_sha256_16": config_sha,
            "hyperparameters": {k: tcfg[k] for k in ("epochs", "batch_size", "lr", "weight_decay", "scheduler", "amp")},
            "device": str(device),
        },
    }

    for epoch in range(start_epoch, int(tcfg["epochs"]) + 1):
        log(f"-- epoch {epoch}/{tcfg['epochs']}  lr={optimizer.param_groups[0]['lr']:.2e}")
        opt_like = _StepSchedulerOptimizer(optimizer, scheduler if per_step else None)
        tr = train_one_epoch(model, train_loader, opt_like, scaler, device, amp, tcfg.get("grad_clip"), tcfg.get("max_steps_per_epoch"), log)
        pv = predict(model, val_loader, device, amp)
        vm = compute_binary_metrics(pv["labels"], pv["scores"], 0.5)
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_accuracy": tr["accuracy"],
            "val_loss": pv["loss"],
            "val_accuracy": vm["accuracy"],
            "val_balanced_accuracy": vm["balanced_accuracy"],
            "val_f1_macro": vm["f1_macro"],
            "val_auc": vm["roc_auc"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": tr["seconds"],
        }
        history.append(row)
        log(
            f"   train_loss={tr['loss']:.4f} val_loss={pv['loss']:.4f} val_acc={vm['accuracy']:.4f} "
            f"val_f1={vm['f1_macro']:.4f} val_auc={vm['roc_auc'] if vm['roc_auc'] is not None else float('nan'):.4f}"
        )
        current = row[monitor]
        if current is None:
            current = -row["val_loss"] if mode == "max" else row["val_loss"]
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(current)
        improved = current > best_value if mode == "max" else current < best_value
        if improved:
            best_value, best_epoch, bad_epochs = float(current), epoch, 0
            meta = {**base_meta, "training": {**base_meta["training"], "best_epoch": epoch, "epochs_completed": epoch, "monitor": monitor, "best_value": best_value}}
            meta["version"] = f"{run_id}@e{epoch}"
            save_checkpoint(ckpt_dir / "best.pt", model, meta, epoch=epoch)
            log(f"   * new best {monitor}={best_value:.4f} -> best.pt")
        else:
            bad_epochs += 1
        last_meta = {**base_meta, "version": f"{run_id}@e{epoch}", "training": {**base_meta["training"], "epochs_completed": epoch}}
        save_checkpoint(
            ckpt_dir / "last.pt", model, last_meta, epoch=epoch, history=history,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict() if scheduler is not None else None,
            scaler_state=scaler.state_dict(), best_value=best_value, best_epoch=best_epoch, bad_epochs=bad_epochs,
        )
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        plot_history(history, run_dir)
        if bad_epochs >= int(tcfg["early_stopping_patience"]):
            log(f"early stopping: no improvement in {bad_epochs} epochs")
            break

    # ---- threshold selection on the validation split using the best checkpoint
    best_payload = load_checkpoint_payload(ckpt_dir / "best.pt")
    model.load_state_dict(best_payload["model_state"])
    pv = predict(model, val_loader, device, amp)
    thr = select_threshold(pv["labels"], pv["scores"], tcfg["threshold_method"])
    val_at_thr = compute_binary_metrics(pv["labels"], pv["scores"], thr)
    meta = best_payload["meta"]
    meta["threshold"] = {"value": thr, "selected_on": f"validation split '{ds['val_split']}'", "method": tcfg["threshold_method"]}
    meta["validation_metrics"] = {k: val_at_thr[k] for k in ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "n")}
    meta["training"]["epochs_completed"] = history[-1]["epoch"] if history else 0
    meta["created_at"] = utc_now_iso()
    save_checkpoint(ckpt_dir / "best.pt", model, meta, epoch=best_payload.get("epoch"))
    log(f"selected threshold {thr:.4f} on validation ({tcfg['threshold_method']}); val metrics: {meta['validation_metrics']}")

    run_summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": "training",
        "experiment_id": cfg["experiment_id"],
        "run_id": run_id,
        "track": cfg["track"],
        "description": cfg.get("description", ""),
        "created_at": utc_now_iso(),
        "model": {k: meta[k] for k in ("model_id", "arch", "frequency_branch", "model_config", "pretrained_init", "version")},
        "dataset": {**meta["dataset"], "train_summary": summarize_manifest(train_df), "val_summary": summarize_manifest(val_df)},
        "seed": seed,
        "threshold": meta["threshold"],
        "best_epoch": best_epoch,
        "monitor": monitor,
        "best_value": best_value,
        "validation_metrics": val_at_thr,
        "history": history,
        "environment": {"device": str(device), "amp": amp, **{k: v for k, v in device_info().items() if k != "devices"}},
        "artifacts": {"best_checkpoint": "checkpoints/best.pt", "loss_curve": "loss_curve.png", "val_metrics_curve": "val_metrics_curve.png", "history_csv": "history.csv"},
    }
    save_json(run_summary, run_dir / "run.json")
    if install:
        dest = install_checkpoint(ckpt_dir / "best.pt", cfg["track"], meta["model_id"])
        log(f"installed checkpoint -> {dest}")
    log(f"done. run directory: {run_dir}")
    return {**run_summary, "run_dir": run_dir.as_posix()}


def install_checkpoint(ckpt: Path, track: str, model_id: str) -> Path:
    dest_dir = artifacts_dir() / "checkpoints" / track
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{model_id}.pt"
    shutil.copy2(ckpt, dest)
    shutil.copy2(ckpt.with_suffix(".json"), dest.with_suffix(".json"))
    return dest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override config values, e.g. train.epochs=3")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", default=None, help="Path to a last.pt checkpoint to resume from")
    ap.add_argument("--install", action="store_true", help="Install the best checkpoint for use by the API")
    args = ap.parse_args(argv)
    overrides = list(args.set)
    if args.seed is not None:
        overrides.append(f"train.seed={args.seed}")
    cfg = load_train_config(args.config, overrides)
    try:
        run_training(cfg, resume=args.resume, install=args.install)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
