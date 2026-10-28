"""Aggregate evaluation results into comparison / ablation tables and figures.

Each "variant" is a label plus one or more evaluation result.json files (one per seed).
For each variant the mean and sample standard deviation across seeds are reported.

    python -m ml.experiments.compare --name exp1_baseline_aigen `
        --variant "ResNet-50=artifacts/evaluations/<id_a>/result.json" `
        --variant "EfficientNet-B0=artifacts/evaluations/<id_b>/result.json"

Outputs:
    artifacts/evaluations/<name>/result.json   (kind="comparison"; shown in the dashboard)
    reports/<name>/comparison.csv, comparison.md, comparison.tex, comparison.png
Numbers come exclusively from the referenced result files; nothing is estimated.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.common import RESULT_SCHEMA_VERSION, artifacts_dir, load_json, reports_dir, resolve_repo_path, save_json, utc_now_iso
from ml.evaluation.plots import plot_bar

METRICS = ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "recall_fake", "recall_real")


def _agg(values: list[float | None]) -> dict[str, Any]:
    v = [x for x in values if x is not None]
    if not v:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else None, "n": len(v)}


def compare(name: str, variants: "OrderedDict[str, list[Path]]", kind_label: str = "comparison", description: str = "", baseline: str | None = None) -> Path:
    rows, per_variant = [], {}
    fixture = False
    track = None
    for label, paths in variants.items():
        results = [load_json(p) for p in paths]
        for r in results:
            fixture = fixture or bool(r.get("dataset", {}).get("is_synthetic_fixture"))
            track = track or r.get("track")
        agg = {m: _agg([r["metrics"].get(m) for r in results]) for m in METRICS}
        lat = _agg([(r.get("latency") or {}).get("mean_ms") for r in results])
        per_variant[label] = {
            "n_runs": len(results),
            "eval_ids": [r.get("eval_id") for r in results],
            "models": [r.get("model", {}).get("model_id") for r in results],
            "seeds": [r.get("model", {}).get("training_seed") for r in results],
            "dataset": results[0].get("dataset", {}),
            "metrics": agg,
            "latency_mean_ms": lat,
        }
        rows.append({"variant": label, "n_runs": len(results), **{f"{m}_mean": agg[m]["mean"] for m in METRICS}, **{f"{m}_std": agg[m]["std"] for m in METRICS}, "latency_ms_mean": lat["mean"]})
    df = pd.DataFrame(rows)
    if baseline and baseline in per_variant:
        for m in METRICS:
            b = per_variant[baseline]["metrics"][m]["mean"]
            df[f"{m}_delta_vs_{baseline}"] = [None if (b is None or v is None) else v - b for v in df[f"{m}_mean"]]

    rep = reports_dir() / name
    rep.mkdir(parents=True, exist_ok=True)
    df.to_csv(rep / "comparison.csv", index=False)
    fmt = lambda v, s: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.4f}" + ("" if s is None or (isinstance(s, float) and np.isnan(s)) else f" ± {s:.4f}"))  # noqa: E731
    md = ["| Variant | Runs | Accuracy | Macro-F1 | ROC-AUC | Recall FAKE | Recall REAL |", "|---|---|---|---|---|---|---|"]
    tex = ["\\begin{tabular}{lcccccc}", "\\hline", "Variant & Runs & Acc. & Macro-F1 & ROC-AUC & Rec. FAKE & Rec. REAL \\\\", "\\hline"]
    for label, v in per_variant.items():
        cells = [fmt(v["metrics"][m]["mean"], v["metrics"][m]["std"]) for m in ("accuracy", "f1_macro", "roc_auc", "recall_fake", "recall_real")]
        md.append(f"| {label} | {v['n_runs']} | " + " | ".join(cells) + " |")
        tex.append(f"{label} & {v['n_runs']} & " + " & ".join(c.replace('±', '$\\pm$') for c in cells) + " \\\\")
    tex += ["\\hline", "\\end{tabular}"]
    note = "\n\n> NOTE: includes results on the SYNTHETIC TEST FIXTURE — not meaningful.\n" if fixture else ""
    (rep / "comparison.md").write_text("\n".join(md) + note + "\n", encoding="utf-8")
    (rep / "comparison.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    labels = list(per_variant)
    plot_bar(
        labels,
        {m: [per_variant[l]["metrics"][m]["mean"] for l in labels] for m in ("accuracy", "f1_macro", "roc_auc")},
        rep / "comparison.png",
        f"{name}: mean over runs (error bars = std)",
        "score",
        errors={m: [per_variant[l]["metrics"][m]["std"] for l in labels] for m in ("accuracy", "f1_macro", "roc_auc")},
    )

    out_dir = artifacts_dir() / "evaluations" / name
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": "comparison",
        "comparison_type": kind_label,
        "eval_id": name,
        "experiment_id": name,
        "description": description,
        "created_at": utc_now_iso(),
        "track": track,
        "baseline": baseline,
        "variants": per_variant,
        "table": df.replace({np.nan: None}).to_dict("records"),
        "dataset": {"is_synthetic_fixture": fixture},
        "artifacts": {"comparison_csv": f"reports/{name}/comparison.csv", "comparison_md": f"reports/{name}/comparison.md", "comparison_tex": f"reports/{name}/comparison.tex"},
    }
    save_json(result, out_dir / "result.json")
    df.to_csv(out_dir / "comparison.csv", index=False)
    import shutil

    shutil.copy2(rep / "comparison.png", out_dir / "comparison.png")
    return out_dir / "result.json"


def parse_variants(items: list[str]) -> "OrderedDict[str, list[Path]]":
    out: OrderedDict[str, list[Path]] = OrderedDict()
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--variant must look like LABEL=path[,path...]; got {it!r}")
        label, paths = it.split("=", 1)
        out.setdefault(label.strip(), []).extend(resolve_repo_path(p.strip()) for p in paths.split(",") if p.strip())
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--type", default="comparison", choices=["comparison", "ablation", "cross_source"])
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--description", default="")
    args = ap.parse_args(argv)
    p = compare(args.name, parse_variants(args.variant), args.type, args.description, args.baseline)
    print(f"Comparison written to {p}")


if __name__ == "__main__":
    main()
