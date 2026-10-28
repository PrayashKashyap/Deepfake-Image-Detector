"""Experiment 4: robustness evaluation (JPEG 90/70/50, downscaling, Gaussian blur, Gaussian noise).

Runs the same checkpoint on the same split under each perturbation and reports metrics
plus the change relative to clean images. Thresholds are NOT re-tuned per condition.

    python -m ml.robustness.evaluate_robustness --checkpoint artifacts/checkpoints/ai_generated/<model>.pt `
        --dataset genimage_v1 --split test
    python -m ml.robustness.evaluate_robustness --config ml/configs/eval/exp4_robustness.yaml --checkpoint <ckpt>
"""
from __future__ import annotations

import sys

from ml.evaluation.evaluate import main as _eval_main


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--robustness" not in argv:
        argv.append("--robustness")
    _eval_main(argv)


if __name__ == "__main__":
    main()
