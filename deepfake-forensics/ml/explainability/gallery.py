"""Experiment 6: Grad-CAM galleries for correctly and incorrectly classified test images.

Uses the predictions.csv of an existing evaluation (so no threshold is re-tuned) and
renders, for each of TP / TN / FP / FN, up to k examples with their Grad-CAM overlays.

    python -m ml.explainability.gallery --eval-dir artifacts/evaluations/<eval_id> `
        --checkpoint artifacts/checkpoints/ai_generated/<model>.pt --root D:\\datasets\\GenImage --k 6

Heatmaps are qualitative explanations, not proof of what evidence the model uses.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ml.common import resolve_device, resolve_repo_path, save_json, utc_now_iso  # noqa: E402
from ml.explainability.gradcam import GradCAM, overlay  # noqa: E402
from ml.models.checkpoint import load_model_for_inference  # noqa: E402
from ml.preprocessing.transforms import PreprocessConfig, build_eval_transform, denormalize_to_uint8, load_image_rgb  # noqa: E402

CASES = {"TP": (1, 1), "TN": (0, 0), "FP": (0, 1), "FN": (1, 0)}


def build_gallery(eval_dir: Path, checkpoint: Path, root: Path, k: int = 6, seed: int = 0, device: str = "auto") -> Path:
    preds = pd.read_csv(eval_dir / "predictions.csv", keep_default_na=False)
    dev = resolve_device(device)
    model, meta = load_model_for_inference(checkpoint, dev)
    pcfg = PreprocessConfig.from_dict(meta["preprocess"])
    tf = build_eval_transform(pcfg)
    cam = GradCAM(model, model.gradcam_target_layer)
    out_dir = eval_dir / "gradcam_gallery"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"created_at": utc_now_iso(), "checkpoint": checkpoint.name, "explained_class": "predicted class", "cases": {}}
    for case, (true, pred) in CASES.items():
        sub = preds[(preds["label"] == true) & (preds["predicted"] == pred)]
        sub = sub.sample(n=min(k, len(sub)), random_state=seed) if len(sub) else sub
        summary["cases"][case] = {"available": int(((preds["label"] == true) & (preds["predicted"] == pred)).sum()), "shown": int(len(sub))}
        if len(sub) == 0:
            continue
        fig, axes = plt.subplots(2, len(sub), figsize=(2.4 * len(sub), 5), dpi=110, squeeze=False)
        for j, (_, row) in enumerate(sub.iterrows()):
            x = tf(load_image_rgb(root / row["path"])).unsqueeze(0).to(dev)
            heat, _ = cam(x, class_idx=pred)
            rgb = denormalize_to_uint8(x[0], pcfg)
            axes[0, j].imshow(rgb)
            axes[0, j].set_title(f"{row['source'][:16]}\nscore={row['fake_score']:.2f}", fontsize=7)
            axes[1, j].imshow(overlay(rgb, heat))
            for a in axes[:, j]:
                a.axis("off")
        fig.suptitle(f"{case}: true={'FAKE' if true else 'REAL'}, predicted={'FAKE' if pred else 'REAL'} (Grad-CAM for predicted class)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"{case}.png")
        plt.close(fig)
    save_json(summary, out_dir / "gallery.json")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    out = build_gallery(resolve_repo_path(args.eval_dir), resolve_repo_path(args.checkpoint), resolve_repo_path(args.root), args.k, args.seed, args.device)
    print(f"Gallery written to {out}")


if __name__ == "__main__":
    main()
