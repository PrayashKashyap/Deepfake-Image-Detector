"""Generate a Markdown results report from evaluation result files (numbers are copied, never computed by hand).

    python -m ml.experiments.report --prefix real_ --out reports/RESULTS_real.md

Includes every artifacts/evaluations/<id>/result.json whose eval_id starts with --prefix
(synthetic-fixture results are always excluded).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ml.common import artifacts_dir, load_json, resolve_repo_path, utc_now_iso


def f(v, pct=False, d=4):
    if v is None:
        return "n/a"
    return f"{v * 100:.2f}%" if pct else f"{v:.{d}f}"


def ci(r, k):
    c = ((r.get("confidence_intervals") or {}).get("metrics") or {}).get(k) or {}
    return f" [{c['low']:.3f}, {c['high']:.3f}]" if c.get("low") is not None else ""


def build(prefix: str) -> str:
    ev = artifacts_dir() / "evaluations"
    res = []
    for d in sorted(ev.iterdir()) if ev.is_dir() else []:
        p = d / "result.json"
        if d.name.startswith(prefix) and p.is_file():
            r = load_json(p)
            if not (r.get("dataset") or {}).get("is_synthetic_fixture"):
                res.append(r)
    out = [f"# Measured results (generated {utc_now_iso()})", "", "All numbers below are copied from `artifacts/evaluations/*/result.json` by `ml.experiments.report`.", ""]

    evals = [r for r in res if r["kind"] == "evaluation"]
    if evals:
        out += ["## Evaluations (clean test data)", "", "| Eval id | Model | Train seed | n (real/fake) | Threshold | Accuracy | Macro-F1 [95% CI] | ROC-AUC [95% CI] | Prec. FAKE | Rec. FAKE | Prec. REAL | Rec. REAL | Latency bs=1 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in evals:
            m, ds = r["metrics"], r["dataset"]
            out.append(
                f"| {r['eval_id']} | {r['model']['arch']}{' +FFT' if r['model'].get('frequency_branch') else ''} | {r['model'].get('training_seed')} | "
                f"{ds['n_samples']} ({ds['class_counts']['real']}/{ds['class_counts']['fake']}) | {f(r['threshold']['value'], d=3)} | {f(m['accuracy'], True)} | "
                f"{f(m['f1_macro'])}{ci(r, 'f1_macro')} | {f(m['roc_auc'])}{ci(r, 'roc_auc')} | {f(m['precision_fake'], d=3)} | {f(m['recall_fake'], d=3)} | "
                f"{f(m['precision_real'], d=3)} | {f(m['recall_real'], d=3)} | {f((r.get('latency') or {}).get('mean_ms'), d=1)} ms |"
            )
        out.append("")
        out += ["### Confusion matrices (rows true REAL/FAKE, columns predicted REAL/FAKE)", ""]
        for r in evals:
            cm = r["confusion_matrix"]
            out.append(f"- `{r['eval_id']}`: [[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]")
        out.append("")
        out += ["### Per-source results", ""]
        for r in evals:
            unseen = set(r["dataset"].get("unseen_sources") or [])
            out += [f"**{r['eval_id']}**", "", "| Source | Seen in training | n | Accuracy | Detection rate (FAKE) | Specificity (REAL) | Mean fake score |", "|---|---|---|---|---|---|---|"]
            for s in r["per_source"]:
                seen = "no (unseen)" if s["source"] in unseen else "yes"
                out.append(f"| {s['source']} | {seen} | {s['n']} | {f(s['accuracy'], True)} | {f(s['detection_rate'], True)} | {f(s['specificity'], True)} | {f(s['mean_fake_score'], d=3)} |")
            out.append("")

    rob = [r for r in res if r["kind"] == "robustness"]
    if rob:
        out += ["## Robustness (same checkpoint and threshold; Δ relative to clean)", ""]
        for r in rob:
            out += [f"**{r['eval_id']}** ({r['model']['arch']}, n={r['dataset']['n_samples']})", "", "| Condition | Accuracy | Δ acc | Macro-F1 | ROC-AUC | Δ AUC | Rec. FAKE | Rec. REAL |", "|---|---|---|---|---|---|---|---|"]
            for c in r["robustness"]["conditions"]:
                out.append(f"| {c['name']} | {f(c['accuracy'], True)} | {f(c['delta_accuracy'], d=3)} | {f(c['f1_macro'])} | {f(c['roc_auc'])} | {f(c['delta_roc_auc'], d=3)} | {f(c['recall_fake'], d=3)} | {f(c['recall_real'], d=3)} |")
            out.append("")

    comps = [r for r in res if r["kind"] == "comparison"]
    if comps:
        out += ["## Comparisons / ablations (mean ± sample std over seeds)", ""]
        for r in comps:
            out += [f"**{r['eval_id']}** ({r.get('comparison_type')})", "", "| Variant | Runs | Accuracy | Macro-F1 | ROC-AUC | Rec. FAKE | Rec. REAL |", "|---|---|---|---|---|---|---|"]
            for label, v in r["variants"].items():
                cell = lambda k: "n/a" if v["metrics"][k]["mean"] is None else f"{v['metrics'][k]['mean']:.4f}" + (f" ± {v['metrics'][k]['std']:.4f}" if v["metrics"][k]["std"] is not None else "")  # noqa: E731
                out.append(f"| {label} | {v['n_runs']} | {cell('accuracy')} | {cell('f1_macro')} | {cell('roc_auc')} | {cell('recall_fake')} | {cell('recall_real')} |")
            out.append("")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--out", default="reports/RESULTS.md")
    args = ap.parse_args(argv)
    out = resolve_repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(args.prefix), encoding="utf-8")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
