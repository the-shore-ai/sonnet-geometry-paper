"""Analyze existing all-layer reports; never loads a model or uses cloud APIs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from sonnet_geometry.depth_discovery import analyze_depths, json_safe, load_model_rdms


def fingerprint(path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def write_json(path, value):
    with path.open("x") as f:
        json.dump(json_safe(value), f, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        f.write("\n")


def plots(result, out):
    aliases = {item["model_id"]: item["model_id"].split("/")[-1] for item in result["model_layers"]}
    for cohort in result["cohorts"]:
        rows = [r for r in result["maps"] if r["cohort"] == cohort]
        for metric, label in [("spearman", "Spearman RSA"), ("nuisance_residual_pearson", "In-sample residual rank correlation")]:
            fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
            for ax, row in zip(axes.T.flat, rows):
                dy, dx = (.5 / (n - 1) for n in row[metric].shape)
                im = ax.imshow(row[metric], origin="lower", aspect="auto", vmin=-1, vmax=1, cmap="coolwarm",
                               extent=(-dx, 1 + dx, -dy, 1 + dy))
                ax.set_xlim(-dx, 1 + dx)
                ax.set_ylim(-dy, 1 + dy)
                ax.set_title(f"{row['pooling']}\n{aliases[row['model_a']]} / {aliases[row['model_b']]}", fontsize=10)
                ax.set_xlabel("Model B normalized residual depth")
                ax.set_ylabel("Model A normalized residual depth")
            fig.colorbar(im, ax=axes, label=label, shrink=.8)
            fig.suptitle(f"{cohort}: all searched native depth pairs — descriptive, no peak p-values")
            for extension in ["png", "pdf"]:
                fig.savefig(out / f"depth-map-{cohort}-{metric}.{extension}", dpi=150)
            plt.close(fig)
    for cohort in ["all154", "period_without154"]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for col, model in enumerate(result["model_layers"]):
            depth = [layer["normalized_depth"] for layer in model["layers"]]
            for row, pool in enumerate(["last_token", "cumulative_mean"]):
                entry = next(e for e in result["evolution"] if (e["cohort"], e["model_id"], e["pooling"]) == (cohort, model["model_id"], pool))
                ax = axes[row, col]
                for name in ["embedding_rsa", "last_raw_block_rsa", "final_normalized_rsa"]:
                    ax.plot(depth, entry[name], label=name.replace("_rsa", "").replace("_", " "))
                ax.plot(depth[1:], entry["successive_depth_rsa"], label="previous native depth", linestyle="--")
                ax.set_ylim(-1, 1.04)
                ax.set_title(f"{aliases[model['model_id']]} / {pool}")
                ax.set_xlabel("Normalized raw residual depth")
                ax.set_ylabel("Within-model Spearman RSA")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{cohort}: native depth evolution; final normalization is a separate reference")
        for extension in ["png", "pdf"]:
            fig.savefig(out / f"depth-evolution-{cohort}.{extension}", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row, pool in enumerate(["last_token", "cumulative_mean"]):
        entries = [s for s in result["heldout_summary"] if s["cohort"] == "period_without154" and s["pooling"] == pool]
        for col, summary in enumerate(entries):
            ax = axes[row, col]
            conditions = [r for r in result["heldout"] if all(r[k] == summary[k] for k in ["cohort", "model_a", "model_b", "pooling"])]
            points = [[r[objective].get(metric, np.nan) for r in conditions] for objective, metric in [
                ("raw_selected", "heldout_spearman"), ("adjusted_selected", "heldout_spearman"),
                ("adjusted_selected", "heldout_training_cdf_residual_pearson")]]
            for i, values in enumerate(points):
                ax.scatter(i + np.linspace(-.12, .12, len(values)), values, s=24, alpha=.8)
                if np.isfinite(values).any():
                    ax.plot([i-.2, i+.2], [np.nanmedian(values)]*2, color="black", linewidth=2)
            ax.axhline(0, color="grey", linewidth=.5)
            ax.set_ylim(-1, 1.04)
            ax.set_xticks([0, 1, 2], ["Raw-selected\nheldout RSA", "Adjusted-selected\nheldout RSA", "Adjusted-selected\nheldout residual r"], fontsize=8)
            ax.set_title(f"{aliases[summary['model_a']]} / {aliases[summary['model_b']]}\n{pool}", fontsize=10)
        axes[row, 0].set_ylabel("Held-out poem-pair correlation")
    fig.suptitle("Period endpoints, remove Sonnet 154: 15 dependent internal splits; dots are not confidence intervals")
    for extension in ["png", "pdf"]:
        fig.savefig(out / f"depth-heldout-period-without154.{extension}", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("experiments/deep-study-v2"))
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("depth-*")):
        raise FileExistsError("Depth artifacts already exist; choose a new output directory")
    start = time.monotonic()
    models, terminal, sources = [], {}, []
    for slug in ["qwen", "gemma", "muse"]:
        model = load_model_rdms(root / f"outputs/full-{slug}-v1/report.json", models[0].sonnet_ids if models else None)
        models.append(model)
        sources.append(model.source)
        path = root / f"experiments/full-{slug}-v1/descriptive-observations.json"
        sources.append(fingerprint(path))
        audit = json.loads(path.read_text())["terminal_token_audit"]["per_sonnet"]
        if len({x["sonnet_id"] for x in audit}) != len(audit):
            raise ValueError("Duplicate terminal-token audit IDs")
        terminal[model.model_id] = {x["sonnet_id"]: x for x in audit}
    baseline_path, corpus_path = root / "experiments/baselines.json", root / "datasets/normalized/sonnets.jsonl"
    sources.extend([fingerprint(baseline_path), fingerprint(corpus_path)])
    baseline = json.loads(baseline_path.read_text())
    records = [json.loads(line) for line in corpus_path.read_text().splitlines()]
    print("Loaded compact Q12 RDMs for three complete models", flush=True)
    result = analyze_depths(models, baseline, records, terminal)
    print("Completed complete maps, 360 held-out pair/pooling/cohort splits and consensus", flush=True)
    files = {
        "depth-maps.json": {k: result[k] for k in ["maps", "cohorts", "model_layers", "control_names"]},
        "depth-evolution.json": {k: result[k] for k in ["evolution", "cohorts", "model_layers"]},
        "depth-heldout.json": {k: result[k] for k in ["heldout", "heldout_summary", "splits", "cohorts", "model_layers"]},
        "depth-consensus.json": result["consensus"],
        "depth-robustness.json": result["robustness"],
    }
    for name, content in files.items():
        write_json(out / name, {"schema_version": "depth-discovery-v2", "inference": "Post hoc descriptive/internal validation; no p-values, independent confirmation or literary labels", **content})
    with (out / "depth-heldout-summary.csv").open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cohort", "model_a", "model_b", "pooling", "selection", "heldout_rsa_mean", "heldout_rsa_min", "heldout_rsa_max", "heldout_residual_mean", "heldout_residual_min", "heldout_residual_max", "defined_residual_splits"])
        for row in result["heldout_summary"]:
            for objective in ["raw_selected", "adjusted_selected"]:
                raw, adjusted = row[objective]["heldout_spearman"], row[objective]["heldout_training_cdf_residual_pearson"]
                writer.writerow([row[k] for k in ["cohort", "model_a", "model_b", "pooling"]] + [objective] +
                    [raw[k] for k in ["mean", "minimum", "maximum"]] + [adjusted[k] for k in ["mean", "minimum", "maximum", "defined"]])
    plots(result, out)
    artifacts = [fingerprint(path) for path in sorted(out.glob("depth-*")) if path.is_file()]
    write_json(out / "depth-run.json", {"schema_version": "depth-discovery-run-v2", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_cpu_wall_seconds": time.monotonic() - start, "sources": sources,
        "source_code": [fingerprint(root / "src/sonnet_geometry/depth_discovery.py"), fingerprint(Path(__file__))],
        "artifacts": artifacts, "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "distance_policy": "Round every selected cosine RDM to12 decimal places before rankings; preserve original artifacts",
        "scope": "All raw residual depths, both poolings, 4 declared cohorts; final-normalized states separate",
        "selection": "Maximum training-training-edge raw RSA and independently maximum training residual correlation; ties within1e-12 choose shallowerA thenB",
        "heldout": "5 grouped folds x3 rotations seeds17,41,73;153/154 together; evaluate heldout-heldout edges only. Folds/rotations dependent; ranges not confidence intervals.",
        "nuisance": "Train-only empirical mid-CDF transforms and OLS fits, applied to heldout edges; residual Pearson. Baseline TFIDF and stylometric inputs were already fitted on full corpus; this is internal sensitivity, not fully inductive baseline validation.",
        "inference": "No map maxima p-values or significance claims; searched maps and every split are retained. Posthoc internal validation, no literary labels or external confirmation."})
    print(f"Saved {len(artifacts)+1} depth artifacts in {out}", flush=True)


if __name__ == "__main__":
    main()
