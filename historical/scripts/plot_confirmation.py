#!/usr/bin/env python3
"""Render frozen v4 scientific results only from complete, provenance-checked analysis."""
from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from sonnet_geometry.confirmation_analysis import (
    ANALYSIS_PLAN, FAMILIES, SEEDS, STATES, condition_name, validate_protocol,
)
from sonnet_geometry.storage import canonical_json, sha256_file, write_exclusive
from sonnet_geometry.text_control_analysis import file_receipt

ROOT = next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").is_file())
PAIRS = (("qwen", "gemma"), ("qwen", "muse"), ("gemma", "muse"))
LABELS = ["Input", "Early*", "Depth\n0.75", "Final\nraw", "Final\nnorm"]
NAMES = {"qwen": "Qwen3–8B", "gemma": "Gemma4–31B", "muse": "Muse Glimmer–30B"}
SHORT = {"qwen": "Qwen", "gemma": "Gemma", "muse": "Muse"}
COLORS = {"qwen": "#236a9e", "gemma": "#9b6b21", "muse": "#79529e"}
TEXT, MUTED, GREEN = "#203641", "#647580", "#247d6d"


def exactly_one(rows, **criteria):
    hits = [r for r in rows if all(r.get(k) == v for k, v in criteria.items())]
    if len(hits) != 1:
        raise ValueError(f"Expected exactly one genuine analysis record: {criteria}, found {len(hits)}")
    return hits[0]


def verify_receipt(receipt):
    path = Path(receipt["path"])
    if not path.is_absolute():
        path = ROOT/path
    if (not path.is_file() or path.stat().st_size != receipt["bytes"]
            or sha256_file(path) != receipt["sha256"]):
        raise ValueError(f"Actual source receipt no longer matches: {path}")


def load_actual(analysis_path, protocol_path, descriptive_recovery=False):
    if not analysis_path.is_file():
        raise FileNotFoundError("No actual v4 analysis exists; no scientific figure created")
    analysis = json.loads(analysis_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol)
    schema = ("depth-confirmation-descriptive-recovery-v4" if descriptive_recovery
              else "depth-confirmation-analysis-v4")
    if (analysis.get("schema_version") != schema
            or analysis.get("criteria") != ANALYSIS_PLAN
            or analysis.get("poems") != protocol["sonnets"]
            or analysis.get("poem_count") != 24 or analysis.get("rdm_pairs") != 276):
        raise ValueError("Analysis does not match frozen v4 study identity")
    if descriptive_recovery and (
            analysis.get("status") != "strict_confirmation_incomplete_execution_provenance"
            or analysis.get("original_strict_confirmation_eligible") is not False):
        raise ValueError("Recovered figures must explicitly preserve incomplete strict confirmation")
    if analysis["protocol"]["sha256"] != sha256_file(protocol_path):
        raise ValueError("Analysis refers to a different frozen protocol")
    if set(analysis.get("models", {})) != set(FAMILIES):
        raise ValueError("All three actual model results are required")
    for family, model in analysis["models"].items():
        recovered = descriptive_recovery and family == "muse"
        if recovered:
            if (model.get("execution_complete") is not False
                    or model.get("primary_case_coverage_complete") is not True
                    or len(model.get("source_files", [])) < 3):
                raise ValueError("Recovered Muse provenance must distinguish case coverage from finalization")
        elif model.get("execution_complete") is not True or len(model.get("source_files", [])) != 3:
            raise ValueError("Complete actual execution receipts required")
        for receipt in model["source_files"]:
            verify_receipt(receipt)
    for receipt in analysis["source_files"]:
        verify_receipt(receipt)
    counts = {"within_model": 1440, "cross_model": 1440,
              "arithmetic_comparisons": 720, "arithmetic_gates": 675, "seed_summaries": 540}
    if any(len(analysis.get(name, [])) != count for name, count in counts.items()):
        raise ValueError("Incomplete actual comparison grid")
    if len(analysis.get("claims", [])) != 6:
        raise ValueError("All six frozen primary/secondary claims required")
    return analysis, protocol


def recovery_note(fig, analysis, y):
    if analysis.get("original_strict_confirmation_eligible") is False:
        fig.text(.075, y, "DESCRIPTIVE RECOVERY: Muse model-wide precision/runtime records are missing. Strict confirmation remains incomplete.",
                 fontsize=9, color="#9b4239", fontweight="bold")


def style(ax, ylim=(-1.04, 1.04)):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#bbc8cf")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#e3e9ec", linewidth=.7)
    ax.set_axisbelow(True)
    ax.set_xlim(-.15, 4.15)
    ax.set_ylim(*ylim)
    ax.set_xticks(range(5), LABELS)
    ax.axhline(0, color="#b7c4cb", linewidth=.7, linestyle=":")


def seed_curves(ax, array, color):
    """All five trajectories; band only where all five values are defined."""
    values = np.asarray([[np.nan if x is None else x for x in row] for row in array], dtype=float)
    if values.shape != (5, 5) or np.any(np.isfinite(values) & ((values < -1) | (values > 1))):
        raise ValueError("Expected five seed trajectories across five states")
    for row in values:
        ax.plot(range(5), row, color=color, linewidth=.8, alpha=.45, marker=".", markersize=3)
    complete = np.isfinite(values).all(axis=0)
    median, low, high = (np.full(5, np.nan) for _ in range(3))
    for i in np.flatnonzero(complete):
        median[i], low[i], high[i] = np.median(values[:, i]), min(values[:, i]), max(values[:, i])
    ax.fill_between(range(5), low, high, color=color, alpha=.13, linewidth=0)
    ax.plot(range(5), median, color=color, linewidth=2.3, marker="o", markersize=4.5, zorder=4)
    undefined = int((~np.isfinite(values)).sum())
    if undefined:
        ax.text(.02, .03, f"{undefined}/25 undefined seed/state values; gaps retained",
                transform=ax.transAxes, fontsize=8, color=MUTED)
    return {"seeds": list(SEEDS), "states": list(STATES), "values": array,
            "undefined_values": undefined, "median": [None if np.isnan(v) else float(v) for v in median]}


def common_text(fig, title, subtitle, protocol, footnote):
    fig.text(.075, .944, title, fontsize=19, fontweight="bold", color=TEXT)
    fig.text(.075, .897, subtitle, fontsize=10.5, color=MUTED)
    population = protocol["population"]
    year = population["edition"].split()[0]
    fig.text(.075, .852, f"24 {population['author']} sonnets · {population['work']} · {year} edition · five frozen order randomizations",
             fontsize=10, color=MUTED)
    fig.text(.075, .16, footnote, fontsize=9, color=MUTED, linespacing=1.6)
    fig.text(.075, .065, "* Early = zero-based block 1 for Qwen, block 4 for Gemma/Muse. Sampled states are categorical; connecting lines do not locate a transition.\n"
             "Seeds, omissions and poem pairs are dependent. No confidence intervals or literary-usefulness claim. Private research.",
             fontsize=8.3, color=MUTED, linespacing=1.55)


def format_value(value):
    return "undefined" if value is None else f"{value:.3f}"


def within_figure(analysis, protocol):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 7.5))
    fig.subplots_adjust(left=.075, right=.975, bottom=.35, top=.76, wspace=.23)
    values = []
    for ax, model in zip(axes, FAMILIES):
        style(ax)
        ax.set_title(NAMES[model], fontweight="bold", color=TEXT, fontsize=12, pad=14)
        curves = [[exactly_one(analysis["within_model"], model=model, name=condition_name("token_shuffle", seed, "fp32"),
                               state=state, pooling="mean")["rdm_stability"]["rho"] for state in STATES] for seed in SEEDS]
        values.append({"model": model, **seed_curves(ax, curves, COLORS[model])})
        claim = exactly_one(analysis["claims"], model=model)
        early, final = claim["early_stability"]["median"], claim["final_raw_stability"]["median"]
        ax.text(.5, -.24, f"Median early {format_value(early)} → final raw {format_value(final)}\n"
                f"Frozen {claim['role'].replace('_', ' ')} criterion: {claim['status'].replace('_', ' ')}",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=TEXT, linespacing=1.55)
    axes[0].set_ylabel("Original ↔ token-shuffled mean RDM (Spearman ρ)", fontsize=10, color=TEXT)
    common_text(fig, "Does early stability give way to order sensitivity?",
                "Original and perturbed inputs both use FP32 arithmetic; every registered model and seed is shown.", protocol,
                "Thin lines: each of the five seeds. Thick line: median. Shading: observed minimum–maximum across seeds, not a confidence interval.\n"
                "Exact token inventories are preserved within each model; perturbed token strings differ across tokenizers.")
    recovery_note(fig, analysis, .115)
    return fig, values


def cross_figure(analysis, protocol):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 7.5))
    fig.subplots_adjust(left=.075, right=.975, bottom=.38, top=.76, wspace=.23)
    plotted = []
    for ax, pair in zip(axes, PAIRS):
        style(ax)
        ax.set_title(" / ".join(SHORT[f] for f in pair), fontweight="bold", color=TEXT, fontsize=12, pad=14)
        original = [exactly_one(analysis["cross_model"], models=list(pair), name=condition_name("original", None, "fp32"),
                                state=state, pooling="mean")["rho"] for state in STATES]
        curves = [[exactly_one(analysis["cross_model"], models=list(pair), name=condition_name("token_shuffle", seed, "fp32"),
                               state=state, pooling="mean")["rho"] for state in STATES] for seed in SEEDS]
        ax.plot(range(5), [np.nan if v is None else v for v in original], color=TEXT, linewidth=2,
                marker="s", markersize=4, linestyle="--", zorder=5)
        plotted.append({"models": list(pair), "original": original, **seed_curves(ax, curves, GREEN)})
        claim = exactly_one(analysis["claims"], models=list(pair))
        delta = claim["shuffled_minus_original"]["median"]
        ax.text(.5, -.24, f"Median final-raw shuffled − original: {format_value(delta)}\n"
                f"Frozen {claim['role'].replace('_', ' ')} criterion: {claim['status'].replace('_', ' ')}",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=TEXT, linespacing=1.55)
    axes[0].set_ylabel("Cross-model mean RDM agreement (Spearman ρ)", fontsize=10, color=TEXT)
    fig.legend(handles=[Line2D([], [], color=TEXT, lw=2, ls="--", marker="s", label="Original"),
                        Line2D([], [], color=GREEN, lw=2.3, marker="o", label="Token shuffle median"),
                        Patch(facecolor=GREEN, alpha=.13, label="Five-seed min–max (not CI)")],
               loc="lower center", bbox_to_anchor=(.53, .205), ncol=3, frameon=False, fontsize=9)
    common_text(fig, "How does disruption change shared geometry?",
                "Matched FP32 mean representations; all three model pairs remain visible, including failures to transfer.", protocol,
                "Each thin green line is one frozen seed. Exact token shuffles are tokenizer-specific, so seeds do not imply identical perturbed strings.\n"
                "Greater agreement after disruption does not by itself identify a shared literary or semantic mechanism.")
    recovery_note(fig, analysis, .115)
    return fig, plotted


def precision_figure(analysis, protocol):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 10.0), gridspec_kw={"height_ratios": [1.5, 1]})
    fig.subplots_adjust(left=.075, right=.975, bottom=.18, top=.78, hspace=.39, wspace=.23)
    plotted = []
    for column, model in enumerate(FAMILIES):
        ax = axes[0, column]
        style(ax)
        ax.set_title(NAMES[model], fontweight="bold", color=TEXT, fontsize=12, pad=14)
        original = [exactly_one(analysis["arithmetic_comparisons"], model=model, intervention="original", seed=None,
                                state=state, pooling="mean")["rdm_stability"]["rho"] for state in STATES]
        perturbed = [[exactly_one(analysis["arithmetic_comparisons"], model=model, intervention="token_shuffle", seed=seed,
                                  state=state, pooling="mean")["rdm_stability"]["rho"] for state in STATES] for seed in SEEDS]
        ax.axhline(.95, color="#c28a48", linewidth=1.3, ls=":")
        ax.plot(range(5), [np.nan if v is None else v for v in original], color=TEXT, linewidth=2,
                marker="s", markersize=4, ls="--", zorder=5)
        seed_data = seed_curves(ax, perturbed, COLORS[model])
        lower = axes[1, column]
        style(lower, ylim=(0, 5.5))
        lower.set_yticks(range(6))
        counts = []
        for state in STATES:
            rows = [exactly_one(analysis["arithmetic_gates"], model=model, intervention="token_shuffle", seed=seed,
                                state=state, pooling="mean") for seed in SEEDS]
            count = Counter("pass" if r["passes"] is True else "fail" if r["passes"] is False else "unresolved" for r in rows)
            counts.append({status: count[status] for status in ("pass", "fail", "unresolved")})
        bottom = np.zeros(5)
        for status, color in (("pass", "#378572"), ("fail", "#c0654d"), ("unresolved", "#bcc5ca")):
            heights = np.array([r[status] for r in counts])
            lower.bar(range(5), heights, bottom=bottom, color=color, width=.55)
            for i, height in enumerate(heights):
                if height:
                    lower.text(i, bottom[i]+height/2, str(height), ha="center", va="center", fontsize=8,
                               color="white" if status != "unresolved" else TEXT)
            bottom += heights
        plotted.append({"model": model, "original": original, "perturbed": seed_data, "paired_gate_counts": counts})
    axes[0, 0].set_ylabel("BF16 ↔ FP32 RDM agreement (Spearman ρ)", fontsize=10, color=TEXT)
    axes[1, 0].set_ylabel("Paired precision gates (five seeds)", fontsize=10, color=TEXT)
    fig.text(.075, .946, "Check arithmetic on originals and perturbations", fontsize=19, fontweight="bold", color=TEXT)
    fig.text(.075, .903, "Mean representations · original inputs and all five token-shuffle seeds · same loaded checkpoint values", fontsize=10.5, color=MUTED)
    fig.legend(handles=[Line2D([], [], color=TEXT, lw=2, ls="--", marker="s", label="Original input"),
                        Line2D([], [], color=GREEN, lw=2, label="Perturbed: five seeds + median/min–max"),
                        Line2D([], [], color="#c28a48", lw=1.3, ls=":", label="ρ = 0.95 requirement")],
               loc="upper left", bbox_to_anchor=(.069, .883), ncol=3, frameon=False, fontsize=9)
    fig.legend(handles=[Patch(facecolor=c, label=l) for l, c in
                        (("Pass", "#378572"), ("Fail", "#c0654d"), ("Unresolved", "#bcc5ca"))],
               loc="lower center", bbox_to_anchor=(.52, .105), ncol=3, frameon=False, fontsize=9)
    fig.text(.075, .065, "Paired gate: original AND perturbed BF16/FP32 ρ ≥ 0.95; maximum arithmetic RDM difference ≤ 10% of the FP32 intervention difference.\n"
             "Zero or undefined intervention differences remain unresolved. These gates assess BF16 as a proxy; primary claims are tested directly in FP32.\n"
             f"All bands are five-seed min–max, not confidence intervals. Sparse states; 24 {protocol.get('population', {}).get('author', 'Sidney')} poems; private research.",
             fontsize=8.7, color=MUTED, linespacing=1.65)
    recovery_note(fig, analysis, .025)
    return fig, plotted


def save_figure(fig, stem, values, sources, descriptive_recovery=False):
    paths = [stem.with_suffix(suffix) for suffix in (".png", ".pdf", ".receipt.json")]
    for path in paths:
        if path.exists():
            raise FileExistsError(path)
    for suffix in (".png", ".pdf"):
        buffer = BytesIO()
        fig.savefig(buffer, format=suffix[1:], dpi=180, facecolor="white", metadata={"Title": stem.name} if suffix == ".pdf" else None)
        write_exclusive(stem.with_suffix(suffix), buffer.getvalue())
    plt.close(fig)
    receipt = {"schema_version": "confirmation-figure-v4", "source_files": sources,
               "outputs": [file_receipt(path) for path in paths[:2]], "plotted_values": values,
               "original_strict_confirmation_eligible": not descriptive_recovery,
               "interpretation": "Descriptive frozen comparisons; all seeds shown; min–max is not a confidence interval."
               + (" Muse model-wide precision/runtime records are missing; strict confirmation is incomplete."
                  if descriptive_recovery else "")}
    write_exclusive(paths[2], canonical_json(receipt))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ROOT/"outputs/confirmation-v4/analysis.json")
    parser.add_argument("--protocol", type=Path, default=ROOT/"experiments/confirmation-v4/protocol.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/confirmation-v4/figures")
    parser.add_argument("--descriptive-recovery", action="store_true",
                        help="Require the explicit recovered-data schema and label incomplete strict confirmation")
    args = parser.parse_args()
    analysis, protocol = load_actual(args.analysis, args.protocol, args.descriptive_recovery)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "savefig.facecolor": "white"})
    sources = [file_receipt(args.analysis), file_receipt(args.protocol), file_receipt(__file__)]
    generated = []
    for name, function in (("depth-order-fp32", within_figure), ("cross-model-order-fp32", cross_figure),
                           ("paired-precision-diagnostics", precision_figure)):
        fig, values = function(analysis, protocol)
        generated.extend(str(p) for p in save_figure(fig, args.output_dir/name, values, sources,
                                                    args.descriptive_recovery))
    print(json.dumps({"generated": generated, "scientific_source": str(args.analysis)}))


if __name__ == "__main__":
    main()
