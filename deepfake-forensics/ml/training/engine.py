"""Training / inference loops."""
from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def autocast_ctx(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
    grad_clip: float | None = None,
    max_steps: int | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    t0 = time.perf_counter()
    for step, (x, y, _) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, amp):
            logits = model(x)
            loss = criterion(logits.float(), y)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        bs = y.size(0)
        loss_sum += float(loss.detach()) * bs
        correct += int((logits.argmax(1) == y).sum())
        total += bs
        if step % 50 == 0:
            log(f"    step {step:5d}  loss {float(loss.detach()):.4f}")
    return {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1), "seconds": time.perf_counter() - t0}


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray | float]:
    """Return fake-class scores, labels, dataset indices, mean loss and timing."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    scores, labels, indices = [], [], []
    loss_sum, n = 0.0, 0
    model_seconds = 0.0
    n_batches = len(loader)
    for b, (x, y, idx) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with autocast_ctx(device, amp):
            logits = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        model_seconds += time.perf_counter() - t0
        logits = logits.float()
        loss_sum += float(criterion(logits, y_dev))
        scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        labels.append(y.numpy())
        indices.append(idx.numpy())
        n += y.size(0)
        if progress:
            progress(b + 1, n_batches)
    return {
        "scores": np.concatenate(scores) if scores else np.zeros(0),
        "labels": np.concatenate(labels) if labels else np.zeros(0, dtype=int),
        "indices": np.concatenate(indices) if indices else np.zeros(0, dtype=int),
        "loss": loss_sum / max(n, 1),
        "model_seconds": model_seconds,
        "n": n,
    }
