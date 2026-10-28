"""Generate LaTeX tables and PDF figures for the paper directly from result files.

    python -m ml.experiments.paper_assets --out paper/generated

Every number is read from artifacts/evaluations/*/result.json, run.json, manifest summaries or
validation reports. If an experiment has not finished, a clearly marked PENDING table is emitted
instead (never an estimate). Re-run after more experiments complete to fill them in.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ml.common import artifacts_dir, resolve_repo_path, utc_now_iso  # noqa: E402

EV = artifacts_dir() / "evaluations"
EX = artifacts_dir() / "experiments"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6})


def load(eval_id: str) -> dict[str, Any] | None:
    p = EV / eval_id / "result.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def run_json(exp_id: str, seed: int = 42) -> dict[str, Any] | None:
    runs = sorted(glob.glob(str(EX / exp_id / f"*_s{seed}" / "run.json")))
    return json.loads(Path(runs[-1]).read_text(encoding="utf-8")) if runs else None


def pct(v: float | None, d: int = 2) -> str:
    return "--" if v is None else f"{100 * v:.{d}f}"


def num(v: float | None, d: int = 4) -> str:
    return "--" if v is None else f"{v:.{d}f}"


def ci(r: dict, k: str, scale: float = 1.0, d: int = 3) -> str:
    c = ((r.get("confidence_intervals") or {}).get("metrics") or {}).get(k) or {}
    if c.get("low") is None:
        return ""
    return f"\\,{{\\scriptsize[{c['low'] * scale:.{d}f}, {c['high'] * scale:.{d}f}]}}"


def pending(label: str, caption: str, what: str) -> str:
    return (
        "\\begin{table}[!t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\fbox{{\\parbox{{0.92\\columnwidth}}{{\\centering\\pending{{PENDING: {what}. "
        "This table is generated automatically from the result files once the experiment finishes "
        "(\\texttt{python -m ml.experiments.paper\\_assets}).}}}\n\\end{table}\n"
    )


ARCH = {"resnet50": "ResNet-50", "efficientnet_b0": "EfficientNet-B0"}


# ----------------------------------------------------------------------------- tables
def tab_dataset(out: Path, include_genimage: bool = True) -> None:
    dff = json.loads(resolve_repo_path("data/manifests/dff_v2.summary.json").read_text())
    gi_p = resolve_repo_path("data/manifests/genimage_sub_v1.summary.json")
    gi = json.loads(gi_p.read_text()) if (gi_p.exists() and include_genimage) else None
    rows = []
    for name, s in (("DFF (face)", dff), ("GenImage subset", gi)):
        if s is None:
            continue
        for split in ("train", "val", "test"):
            v = s["splits"][split]
            rows.append(f"{name if split == 'train' else ''} & {split} & {v['real']:,} & {v['fake']:,} & {v['groups']:,} \\\\")
    body = "\n".join(rows)
    (out / "tab_dataset.tex").write_text(
        "\\begin{table}[!t]\n\\centering\n\\caption{Dataset splits after leakage-aware grouping (all images; training "
        "uses a class-balanced random subset of the train split, see Sec.~\\ref{sec:training}).}\n\\label{tab:dataset}\n"
        "\\begin{tabular}{llrrr}\n\\toprule\nDataset & Split & Real & Fake & Groups \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n",
        encoding="utf-8",
    )


def tab_audit(out: Path) -> None:
    v1 = json.loads(resolve_repo_path("reports/data_validation/dff_v1.json").read_text())
    d = json.loads(resolve_repo_path("data/manifests/dff_v2.summary.json").read_text())["dedup"]
    rows = [
        ("Images checked / missing / corrupted", f"{v1['n_rows']:,} / {v1['n_missing']} / {v1['n_corrupted']}"),
        ("Exact-duplicate sets (SHA-256)", f"{v1['exact_duplicate_sets']}"),
        ("\\quad of which span splits (v1 split)", f"{len(v1['exact_duplicates_cross_split'])}"),
        (f"Near-duplicate pairs, dHash $\\le$ {v1['near_duplicate_hamming_threshold']}", f"{v1['near_duplicate_pairs']:,}"),
        ("\\quad cross-group pairs spanning splits (v1)", f"{v1['near_duplicate_cross_split_pairs']:,}"),
        (f"Leakage groups before / after merging (dHash $\\le$ {d['dhash_threshold']})", f"{d['groups_before']:,} / {d['groups_after']:,}"),
        ("Fake images that are 512$\\times$512 JPEG", "100\\% (90{,}000)"),
        ("Real images that are 512$\\times$512", "0\\% (0)"),
    ]
    body = "\n".join(f"{a} & {b} \\\\" for a, b in rows)
    (out / "tab_audit.tex").write_text(
        "\\begin{table}[!t]\n\\centering\n\\caption{Data audit of DeepFakeFace (DFF). v1: split by IMDB-WIKI id only; "
        "v2 (used for all results): groups merged across (near-)duplicates before splitting.}\n\\label{tab:audit}\n"
        "\\begin{tabular}{lr}\n\\toprule\nCheck & Result \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n",
        encoding="utf-8",
    )


def _main_rows(variants: list[tuple[str, str, str]]) -> tuple[list[str], list[dict]]:
    rows, res = [], []
    for label, eval_id, exp in variants:
        r = load(eval_id)
        if r is None:
            rows.append(f"{label} & \\multicolumn{{9}}{{c}}{{\\pending{{pending}}}} \\\\")
            continue
        res.append(r)
        m = r["metrics"]
        tr = run_json(exp)
        rows.append(
            f"{label} & {pct(m['accuracy'])}{ci(r, 'accuracy', 100, 1)} & {pct(m['precision_real'], 1)} & {pct(m['recall_real'], 1)} & "
            f"{pct(m['precision_fake'], 1)} & {pct(m['recall_fake'], 1)} & {num(m['f1_macro'], 4)} & {num(m['roc_auc'], 4)}{ci(r, 'roc_auc')} & "
            f"{num(r['threshold']['value'], 3)} & {num(r['latency']['mean_ms'], 1)} \\\\"
        )
    return rows, res


def tab_main(out: Path, name: str, caption: str, label: str, variants: list[tuple[str, str, str]]) -> list[dict]:
    rows, res = _main_rows(variants)
    if not res:
        (out / f"{name}.tex").write_text(pending(label, caption, "no evaluation finished yet"), encoding="utf-8")
        return res
    r0 = res[0]
    note = (f"n = {r0['dataset']['n_samples']:,} test images ({r0['dataset']['class_counts']['real']:,} real / "
            f"{r0['dataset']['class_counts']['fake']:,} fake); 95\\% grouped-bootstrap CIs in brackets; "
            "threshold $\\tau$ selected on validation; latency: batch size 1, RTX~4060, AMP.")
    (out / f"{name}.tex").write_text(
        "\\begin{table*}[!t]\n\\centering\n"
        f"\\caption{{{caption} {note}}}\n\\label{{{label}}}\n"
        "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lccccccccc}\n\\toprule\n"
        " & Accuracy & \\multicolumn{2}{c}{REAL} & \\multicolumn{2}{c}{FAKE} & Macro & ROC-AUC & & Latency \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
        "Model & (\\%) & Prec. & Rec. & Prec. & Rec. & F1 & & $\\tau$ & (ms) \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n",
        encoding="utf-8",
    )
    return res


def tab_confusion(out: Path, variants: list[tuple[str, str]]) -> None:
    cells = []
    for label, eid in variants:
        r = load(eid)
        if r is None:
            continue
        cm = r["confusion_matrix"]
        cells.append(f"{label} & {cm[0][0]:,} & {cm[0][1]:,} & {cm[1][0]:,} & {cm[1][1]:,} \\\\")
    if not cells:
        return
    (out / "tab_confusion.tex").write_text(
        "\\begin{table}[!t]\n\\centering\n\\caption{Confusion matrices on the DFF test set (TN: real$\\to$REAL, "
        "FP: real$\\to$FAKE, FN: fake$\\to$REAL, TP: fake$\\to$FAKE).}\n\\label{tab:confusion}\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\nModel & TN & FP & FN & TP \\\\\n\\midrule\n"
        + "\n".join(cells) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )


def tab_per_source(out: Path, variants: list[tuple[str, str]], name: str, caption: str, label: str, source_names: dict[str, str]) -> None:
    res = [(lab, load(eid)) for lab, eid in variants]
    res = [(lab, r) for lab, r in res if r is not None]
    if not res:
        (out / f"{name}.tex").write_text(pending(label, caption, "evaluation not finished"), encoding="utf-8")
        return
    sources = [s["source"] for s in res[0][1]["per_source"]]
    head = " & ".join(lab for lab, _ in res)
    rows = []
    for s in sources:
        cells = []
        for _, r in res:
            ps = {x["source"]: x for x in r["per_source"]}[s]
            cells.append(f"{pct(ps['accuracy'])} ({num(ps['mean_fake_score'], 3)})")
        n = {x["source"]: x for x in res[0][1]["per_source"]}[s]["n"]
        rows.append(f"{source_names.get(s, s)} & {n:,} & " + " & ".join(cells) + " \\\\")
    (out / f"{name}.tex").write_text(
        "\\begin{table}[!t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        f"\\begin{{tabular}}{{lr{'c' * len(res)}}}\n\\toprule\nSource & n & {head} \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )


def tab_robust(out: Path, variants: list[tuple[str, str]], name: str, label: str) -> None:
    res = [(lab, load(eid)) for lab, eid in variants]
    res = [(lab, r) for lab, r in res if r is not None]
    if not res:
        (out / f"{name}.tex").write_text(pending(label, "Robustness.", "robustness evaluation not finished"), encoding="utf-8")
        return
    conds = [c["name"] for c in res[0][1]["robustness"]["conditions"]]
    pretty = {"clean": "Clean"}
    rows = []
    for c in conds:
        nm = pretty.get(c, c.replace("jpeg(quality=", "JPEG q=").replace("downscale(factor=", "Downscale $\\times$").replace("blur(sigma=", "Blur $\\sigma$=").replace("noise(std=", "Noise $\\sigma$=").rstrip(")"))
        cells = []
        for _, r in res:
            x = {k["name"]: k for k in r["robustness"]["conditions"]}[c]
            cells.append(f"{pct(x['accuracy'], 1)} & {num(x['roc_auc'], 3)} & {pct(x['recall_fake'], 1)} & {pct(x['recall_real'], 1)}")
        rows.append(f"{nm} & " + " & ".join(cells) + " \\\\")
    # Summary over the perturbed conditions (clean excluded): mean and worst case.
    for agg_name, agg in (("Mean (perturbed)", np.mean), ("Worst case", np.min)):
        cells = []
        for _, r in res:
            pc = [k for k in r["robustness"]["conditions"] if k["name"] != "clean"]
            vals = {m: float(agg([k[m] for k in pc])) for m in ("accuracy", "roc_auc", "recall_fake", "recall_real")}
            cells.append(f"{pct(vals['accuracy'], 1)} & {num(vals['roc_auc'], 3)} & {pct(vals['recall_fake'], 1)} & {pct(vals['recall_real'], 1)}")
        prefix = "\\midrule\n" if agg_name.startswith("Mean") else ""
        rows.append(prefix + f"\\textit{{{agg_name}}} & " + " & ".join(cells) + " \\\\")
    n = res[0][1]["dataset"]["n_samples"]
    hdr1 = " & ".join(f"\\multicolumn{{4}}{{c}}{{{lab}}}" for lab, _ in res)
    cm = "".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(res)))
    hdr2 = " & ".join("Acc. & AUC & Rec.F & Rec.R" for _ in res)
    (out / f"{name}.tex").write_text(
        "\\begin{table*}[!t]\n\\centering\n"
        f"\\caption{{Robustness on a balanced random subset of the DFF test set (n = {n:,}). Perturbations are applied at test time; "
        "the model and validation-selected threshold are unchanged. Acc.\\ and recalls in \\%.}\n"
        f"\\label{{{label}}}\n\\setlength{{\\tabcolsep}}{{2.6pt}}\n\\footnotesize\n"
        f"\\begin{{tabular}}{{l{'cccc' * len(res)}}}\n\\toprule\n & {hdr1} \\\\\n{cm}\nCondition & {hdr2} \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n",
        encoding="utf-8",
    )


def tab_cross_manip_seen_unseen(out: Path, seen_eval: str, methods: list[tuple[str, str, str]]) -> None:
    """Detection rate of each manipulation when seen in training vs. held out (same architecture)."""
    seen = load(seen_eval)
    rows = []
    for key, pretty, unseen_eval in methods:
        u = load(unseen_eval)
        if u is None or seen is None:
            continue
        s_src = {x["source"]: x for x in seen["per_source"]}[key]
        u_src = {x["source"]: x for x in u["per_source"]}
        m = u["metrics"]
        rows.append(
            f"{pretty} & {pct(s_src['detection_rate'], 1)} & {pct(u_src[key]['detection_rate'], 1)} & "
            f"{pct(u_src['wiki']['specificity'], 1)} & {pct(m['accuracy'], 1)}{ci(u, 'accuracy', 100, 1)} & {num(m['roc_auc'], 3)}{ci(u, 'roc_auc')} \\\\"
        )
    if not rows:
        (out / "tab_cross_manip.tex").write_text(pending("tab:cross_manip", "Cross-manipulation.", "experiment not finished"), encoding="utf-8")
        return
    n = load(methods[0][2])["dataset"]["n_samples"]
    (out / "tab_cross_manip.tex").write_text(
        "\\begin{table*}[!t]\n\\centering\n"
        "\\caption{Cross-manipulation generalisation (EfficientNet-B0, no source normalisation). \\emph{Seen}: detection rate when the method is in the "
        "training data (all methods trained jointly). \\emph{Unseen}: the model is retrained without that method (train and validation) and tested on "
        f"4{{,}}500 real and 4{{,}}500 unseen-method test images (n = {n:,}); 95\\% grouped-bootstrap CIs in brackets. Rates in \\%.}}\n"
        "\\label{tab:cross_manip}\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Fake detection rate} & \\multicolumn{3}{c}{Held-out evaluation} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\n"
        "Manipulation & Seen & Unseen & Real recall & Accuracy & ROC-AUC \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n",
        encoding="utf-8",
    )


def tab_simple_pending_or(out: Path, name: str, label: str, caption: str, eval_rows: list[tuple[str, str]], what: str) -> None:
    res = [(lab, load(eid)) for lab, eid in eval_rows]
    have = [(lab, r) for lab, r in res if r is not None]
    if not have:
        (out / f"{name}.tex").write_text(pending(label, caption, what), encoding="utf-8")
        return
    rows = []
    for lab, r in res:
        if r is None:
            rows.append(f"{lab} & \\multicolumn{{5}}{{c}}{{\\pending{{pending}}}} \\\\")
            continue
        m = r["metrics"]
        rows.append(f"{lab} & {r['dataset']['n_samples']:,} & {pct(m['accuracy'])} & {num(m['f1_macro'])} & {num(m['roc_auc'])}{ci(r, 'roc_auc')} & {pct(m['recall_fake'], 1)} \\\\")
    (out / f"{name}.tex").write_text(
        "\\begin{table}[!t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\setlength{{\\tabcolsep}}{{3pt}}\n"
        "\\begin{tabular}{lrcccc}\n\\toprule\nSetting & n & Acc.\\,(\\%) & F1 & ROC-AUC & Rec.F\\,(\\%) \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )


def tab_comparison(out: Path, name: str, label: str, caption: str, comp_id: str, what: str) -> None:
    r = load(comp_id)
    if r is None:
        (out / f"{name}.tex").write_text(pending(label, caption, what), encoding="utf-8")
        return
    fmt = lambda x: "--" if x["mean"] is None else (f"{x['mean']:.4f}" + (f" $\\pm$ {x['std']:.4f}" if x["std"] is not None else ""))  # noqa: E731
    rows = [f"{lab} & {v['n_runs']} & {fmt(v['metrics']['accuracy'])} & {fmt(v['metrics']['f1_macro'])} & {fmt(v['metrics']['roc_auc'])} \\\\" for lab, v in r["variants"].items()]
    (out / f"{name}.tex").write_text(
        "\\begin{table}[!t]\n\\centering\n"
        f"\\caption{{{caption} Mean $\\pm$ sample std over seeds.}}\n\\label{{{label}}}\n\\setlength{{\\tabcolsep}}{{3pt}}\n"
        "\\begin{tabular}{lcccc}\n\\toprule\nVariant & Runs & Accuracy & Macro-F1 & ROC-AUC \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )


# ----------------------------------------------------------------------------- figures
def fig_roc(out: Path, variants: list[tuple[str, str]], fname: str) -> bool:
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    drawn = False
    for i, (lab, eid) in enumerate(variants):
        r = load(eid)
        if r is None or not r.get("roc_curve"):
            continue
        ax.plot(r["roc_curve"]["fpr"], r["roc_curve"]["tpr"], color=COLORS[i], lw=1.4, label=f"{lab} (AUC {r['metrics']['roc_auc']:.4f})")
        drawn = True
    ax.plot([0, 1], [0, 1], "--", color="#898781", lw=0.7)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    inset = ax.inset_axes([0.42, 0.38, 0.33, 0.33])
    for i, (lab, eid) in enumerate(variants):
        r = load(eid)
        if r and r.get("roc_curve"):
            inset.plot(r["roc_curve"]["fpr"], r["roc_curve"]["tpr"], color=COLORS[i], lw=1.0)
    inset.set_xlim(0, 0.15)
    inset.set_ylim(0.8, 1.0)
    inset.tick_params(labelsize=5)
    inset.set_title("zoom", fontsize=5)
    fig.tight_layout()
    fig.savefig(out / fname)
    plt.close(fig)
    return drawn


def fig_robust(out: Path, variants: list[tuple[str, str]], fname: str) -> bool:
    res = [(lab, load(eid)) for lab, eid in variants]
    res = [(lab, r) for lab, r in res if r is not None]
    if not res:
        return False
    names = [c["name"] for c in res[0][1]["robustness"]["conditions"]]
    short = [n.replace("quality=", "q").replace("factor=", "x").replace("sigma=", "s").replace("std=", "s").replace("downscale", "down") for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, (lab, r) in enumerate(res):
        acc = [c["accuracy"] for c in r["robustness"]["conditions"]]
        # markers only: conditions are categorical (no continuum between perturbation types)
        ax.plot(x + (i - (len(res) - 1) / 2) * 0.18, acc, marker="o", ms=4, ls="none", color=COLORS[i], label=lab,
                markeredgecolor="white", markeredgewidth=0.5)
    for b in range(1, len(names)):
        ax.axvline(b - 0.5, color="#e1e0d9", lw=0.4, zorder=0)
    ax.axhline(0.5, color="#898781", lw=0.7, ls="--")
    ax.text(len(names) - 1, 0.51, "chance", fontsize=6, ha="right", color="#52514e")
    ax.set_xticks(x, short, rotation=40, ha="right", fontsize=6)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / fname)
    plt.close(fig)
    return True


def fig_gradcam(out: Path, eval_id: str, checkpoint: str, root: str, fname: str, k: int = 4, seed: int = 0) -> bool:
    import pandas as pd

    from ml.common import resolve_device
    from ml.explainability.gradcam import GradCAM, overlay
    from ml.models.checkpoint import load_model_for_inference
    from ml.preprocessing.transforms import PreprocessConfig, build_eval_transform, denormalize_to_uint8, load_image_rgb

    pred_p = EV / eval_id / "predictions.csv"
    ck = resolve_repo_path(checkpoint)
    if not pred_p.exists() or not ck.exists():
        return False
    preds = pd.read_csv(pred_p, keep_default_na=False)
    dev = resolve_device("auto")
    model, meta = load_model_for_inference(ck, dev)
    pc = PreprocessConfig.from_dict(meta["preprocess"])
    tf = build_eval_transform(pc)
    cam = GradCAM(model, model.gradcam_target_layer)
    cases = [("TP", 1, 1), ("TN", 0, 0), ("FP", 0, 1), ("FN", 1, 0)]
    fig, axes = plt.subplots(len(cases), k, figsize=(3.4, 3.6))
    for i, (case, t, p) in enumerate(cases):
        sub = preds[(preds.label == t) & (preds.predicted == p)]
        sub = sub.sample(n=min(k, len(sub)), random_state=seed)
        for j in range(k):
            a = axes[i, j]
            a.axis("off")
            if j >= len(sub):
                continue
            row = sub.iloc[j]
            x = tf(load_image_rgb(Path(root) / row["path"])).unsqueeze(0).to(dev)
            heat, _ = cam(x, class_idx=p)
            a.imshow(overlay(denormalize_to_uint8(x[0], pc), heat))
            a.set_title(f"{row['source']} s={row['fake_score']:.2f}", fontsize=5, pad=1.5)
        axes[i, 0].text(-0.08, 0.5, case, transform=axes[i, 0].transAxes, fontsize=7, ha="right", va="center", rotation=90)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.96, bottom=0.005, wspace=0.04, hspace=0.18)
    fig.savefig(out / fname, dpi=300)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="paper/generated")
    ap.add_argument("--dff-root", default=None, help="DFF image root (for the Grad-CAM figure)")
    ap.add_argument("--no-gradcam", action="store_true")
    ap.add_argument("--scope", choices=["all", "face"], default="all", help="face: facial-deepfake track only")
    args = ap.parse_args(argv)
    out = resolve_repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dff_main = [
        ("ResNet-50", "real_exp1_dff__resnet50__s42__test", "real_exp1_dff__resnet50"),
        ("EfficientNet-B0", "real_exp1_dff__efficientnet_b0__s42__test", "real_exp1_dff__efficientnet_b0"),
        ("EffNet-B0 + norm.", "real_exp1_dff__efficientnet_b0_norm__s42__test", "real_exp1_dff__efficientnet_b0_norm"),
    ]
    dff_rob = [(a, b.replace("__test", "__robustness")) for a, b, _ in dff_main]
    tab_dataset(out, include_genimage=args.scope != "face")
    tab_audit(out)
    tab_main(out, "tab_main_dff", "Facial-deepfake detection on the DeepFakeFace test split.", "tab:main_dff", dff_main)
    tab_confusion(out, [(a, b) for a, b, _ in dff_main])
    tab_per_source(out, [(a, b) for a, b, _ in dff_main], "tab_per_method_dff",
                   "Per-source accuracy (\\%) on the DFF test split; mean fake score in parentheses. For fake sources accuracy equals the detection rate, for real images the specificity.",
                   "tab:per_method_dff", {"wiki": "Real (IMDB-WIKI)", "inpainting": "SD inpainting", "insight": "InsightFace swap", "text2img": "SD v1.5 text2img"})
    tab_robust(out, dff_rob, "tab_robust_dff", "tab:robust_dff")
    tab_cross_manip_seen_unseen(out, "real_exp1_dff__efficientnet_b0__s42__test", [
        ("inpainting", "SD inpainting", "real_exp3_cross_manipulation__lo_inpainting__s42__unseen_inpainting"),
        ("insight", "InsightFace face swap", "real_exp3_cross_manipulation__lo_insight__s42__unseen_insight"),
        ("text2img", "SD v1.5 text-to-image", "real_exp3_cross_manipulation__lo_text2img__s42__unseen_text2img"),
    ])
    if args.scope != "face":
        tab_comparison(out, "tab_main_genimage", "tab:main_genimage", "AI-generated image detection on the GenImage subset (8 generators).",
                       "real_exp1_genimage__test", "GenImage baseline comparison (Exp.~1, 2 seeds) not yet run")
        tab_simple_pending_or(out, "tab_cross_gen", "tab:cross_gen",
                              "Cross-generator generalisation (EfficientNet-B0, leave-one-generator-out): accuracy on the held-out generator's test images.",
                              [(f"unseen {g}", f"real_exp2_cross_generator__lo_{g}__s42__unseen_{g}") for g in ("biggan", "glide", "midjourney", "sdv4")],
                              "leave-one-generator-out experiment (Exp.~2) not yet run")
        tab_comparison(out, "tab_ablation", "tab:ablation", "Ablation on the GenImage subset (EfficientNet-B0, identical splits and hyper-parameters).",
                       "real_exp5_7_ablation_genimage__test", "frequency-branch / JPEG-augmentation ablation (Exp.~5/7) not yet run")

    fig_roc(out, [(a, b) for a, b, _ in dff_main], "fig_roc_dff.pdf")
    fig_robust(out, dff_rob, "fig_robust_dff.pdf")
    if not args.no_gradcam and args.dff_root:
        fig_gradcam(out, "real_exp1_dff__resnet50__s42__test", "artifacts/checkpoints/face_deepfake/real_exp1_dff__resnet50__20261026-153619_s42.pt", args.dff_root, "fig_gradcam_dff.png")
    (out / "GENERATED.txt").write_text(f"Generated {utc_now_iso()} by ml.experiments.paper_assets from result files. Do not edit by hand.\n", encoding="utf-8")
    print(f"Assets written to {out}")


if __name__ == "__main__":
    main()
