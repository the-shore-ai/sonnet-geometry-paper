#!/usr/bin/env python3
"""Plot the frozen v3 controls only after genuine GPU analysis exists.

This is a presentation of recorded results, not a new statistical analysis.
Missing models, partial precision coverage and undefined correlations remain
visible. Existing figure outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("qwen", "gemma", "muse")
NAMES = {"qwen": "Qwen3–8B", "gemma": "Gemma4–31B", "muse": "Muse Glimmer–30B"}
STATES = ("early", "depth75", "final_raw")
CONDITIONS = (
    ("word_shuffle_padded_bf16", "Word shuffle", "#b86c24"),
    ("line_shuffle_padded_bf16", "Line shuffle", "#326b91"),
    ("token_shuffle_padded_bf16", "Token shuffle", "#288675"),
)
TEXT = "#233844"
MUTED = "#61717b"


def exactly_one(rows, **criteria):
    hits = [row for row in rows if all(row.get(k) == v for k, v in criteria.items())]
    if len(hits) != 1:
        raise ValueError(f"Expected exactly one record for {criteria}; found {len(hits)}")
    return hits[0]


def load_required(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required actual result is absent; no scientific figure created: {path}")
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def style_axis(ax, grid="x"):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["bottom", "left"]:
        ax.spines[side].set_color("#c2cbd0")
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    ax.grid(axis=grid, color="#e3e8eb", linewidth=.7)
    ax.set_axisbelow(True)


def panel_title(ax, label, title, subtitle):
    ax.text(0, 1.17, f"{label}  {title}", transform=ax.transAxes,
            fontsize=12, fontweight="bold", color=TEXT, va="bottom")
    ax.text(0, 1.07, subtitle, transform=ax.transAxes,
            fontsize=9, color=MUTED, va="bottom")


def endpoint_panel(ax, endpoint):
    cohort = endpoint["cohorts"]["period_without154"]
    if len(cohort) != 142 or 154 in cohort:
        raise ValueError("Endpoint panel requires the registered 142-poem cohort")
    specifications = (
        ("last_original", "last_lexical", "Original → lexical endpoint", "#795387", "o", -.12),
        ("mean_all", "mean_lexical", "All-token → lexical mean", "#288675", "s", .12),
    )
    plotted = []
    for original, variant, label, color, marker, offset in specifications:
        for i, family in enumerate(FAMILIES):
            row = exactly_one(endpoint["whole_poem_stability"], cohort="period_without154",
                              family=family, state="depth_0_75", original=original, variant=variant)
            value = row["rdm_spearman"]
            if value is None or not np.isfinite(value):
                raise ValueError("Unexpected undefined endpoint value")
            y = 2-i+offset
            ax.scatter(value, y, color=color, marker=marker, s=48, zorder=3)
            ax.annotate(f"{value:.3f}", (value, y), xytext=(5 if value < .94 else -6, 0),
                        textcoords="offset points", ha="left" if value < .94 else "right",
                        va="center", fontsize=8, color=color)
            plotted.append({"family": family, "original": original, "variant": variant, "rho": value})
        ax.scatter([], [], color=color, marker=marker, s=42, label=label)
    style_axis(ax)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-.8, 2.5)
    ax.set_yticks([2, 1, 0], ["Qwen", "Gemma", "Muse"])
    ax.set_xlabel("Within-model RDM Spearman ρ  ·  higher = more stable", fontsize=9, color=TEXT)
    ax.legend(loc="lower left", bbox_to_anchor=(.005, .015), ncol=2, columnspacing=1.0,
              frameon=False, fontsize=8.2, borderaxespad=0)
    panel_title(ax, "A", "Readout choice changes geometry", "142 period-ending poems, excluding Sonnet 154 · depth 0.75")
    return plotted


def historical_panel(ax, editorial):
    run = exactly_one(editorial["runs"], exclude154=True)
    if run["cohort_rows"] != 145:
        raise ValueError("Historical panel requires original 145-row OOF cohort")
    systems = [("conventional", "Text controls"), ("plus_both_final", "Final states"),
               ("single_depth", "Selected single depth"), ("final_plus_fixed_depths", "Multiple depths")]
    sources = [("beeching1904_reported_lee", "Beeching", "#9b7331", -.12),
               ("reed1923", "Reed", "#326b91", .12)]
    plotted, favorable = [], []
    for source_id, short, color, offset in sources:
        source = exactly_one(run["sources"], source_id=source_id)
        population = source["populations"]["new_historical_links"]
        metrics = population["metrics"]
        for i, (system, label) in enumerate(systems):
            value = metrics[system]["mean_rank_percentile"]
            ax.scatter(value, 3-i+offset, color=color, marker="o", s=44, zorder=3)
            ax.annotate(f"{value:.3f}", (value, 3-i+offset), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8, color=color)
            plotted.append({"source": source_id, "system": system, "mean_rank_percentile": value,
                            "new_positive_edges": population["count"]})
        units = metrics["single_depth"]["source_units_with_observations"]
        ax.scatter([], [], color=color, s=42, label=f"{short}: {population['count']} new links / {units} units")
        favorable.append(metrics["final_plus_fixed_depths"]["mean_rank_percentile"] < metrics["single_depth"]["mean_rank_percentile"])
    style_axis(ax)
    ax.set_xlim(.40, .585)
    ax.set_ylim(-.45, 3.45)
    ax.set_yticks([3, 2, 1, 0], [label for _, label in systems])
    ax.set_xticks([.40, .45, .50, .55])
    ax.set_xlabel("Within-band new-link rank percentile  ·  lower = better", fontsize=9, color=TEXT)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.28), frameon=False, fontsize=8.5, borderaxespad=0)
    subtitle = "No consistent gain across sources · unchanged Folger-trained scores" if not all(favorable) else "Descriptive source transfer · unchanged Folger-trained scores"
    panel_title(ax, "B", "Historical editorial transfer", subtitle)
    return plotted


def gate_marker(gate):
    """Filled = full gate pass; open = subset pass; x = fail; triangle = unavailable."""
    if gate.get("passes_on_all_planned_poems") is True:
        return "o", True
    if gate.get("passes_on_tested_subset") is True:
        return "o", False
    if gate.get("passes_on_tested_subset") is False:
        return "x", False
    return "^", False


def controls_panels(axes, controls, protocol):
    planned = len(protocol["sonnets"])
    if controls["schema_version"] != "text-control-analysis-v3":
        raise ValueError("Unsupported actual control-analysis schema")
    plotted, finite_values = [], []
    for row in controls["within_model"]:
        if row["state"] in STATES and row["pooling"] == "mean" and row["condition"] in [c[0] for c in CONDITIONS]:
            rho = row["rdm_stability"]["rho"]
            if rho is not None:
                finite_values.append(rho)
    lower = -.05 if all(x >= 0 for x in finite_values) else -1.05
    for ax, family in zip(axes, FAMILIES):
        style_axis(ax, grid="y")
        ax.axhline(1, color="#b9c7cd", linestyle=":", linewidth=.8)
        ax.set_ylim(lower, 1.08)
        ax.set_xlim(-.18, 2.18)
        ax.set_xticks([0, 1, 2], ["Early", "Depth 0.75", "Final raw"])
        model = controls["models"].get(family, {"status": "missing_model_report"})
        ax.set_title(NAMES[family], fontsize=11, color=TEXT, fontweight="bold", pad=20)
        if model["status"] != "loaded":
            ax.text(.5, .52, "No execution report\nNo result substituted", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED, fontsize=10)
            ax.text(.5, 1.04, f"0 / {planned} poems available", transform=ax.transAxes,
                    ha="center", fontsize=8.5, color=MUTED)
            continue
        counts = model["case_counts"]
        bf16 = sum(count for name, count in counts.items() if name.endswith("bf16"))
        fp32 = counts.get("original_padded_fp32", 0)
        ax.text(.5, 1.04, f"BF16 cases {bf16}/{planned*5} · FP32 {fp32}/{planned}",
                transform=ax.transAxes, ha="center", fontsize=8.5, color=MUTED)
        coverage_lines = []
        for condition, label, color in CONDITIONS:
            points, coverages, precision_counts = [], [], []
            for state in STATES:
                row = exactly_one(controls["within_model"], model=family, state=state,
                                  pooling="mean", condition=condition)
                points.append(row["rdm_stability"]["rho"])
                coverages.append(row["poem_count"])
                precision_counts.append(row["precision_gate"].get("poem_count", 0))
                plotted.append({"family": family, "state": state, "condition": condition,
                                "rho": row["rdm_stability"]["rho"], "poem_count": row["poem_count"],
                                "sonnets": row["sonnets"], "precision_gate": row["precision_gate"]})
            values = [np.nan if value is None else value for value in points]
            ax.plot(range(3), values, color=color, linewidth=1.5, alpha=.85, zorder=2)
            for i, state in enumerate(STATES):
                row = exactly_one(controls["within_model"], model=family, state=state,
                                  pooling="mean", condition=condition)
                value = points[i]
                if value is None:
                    continue
                marker, filled = gate_marker(row["precision_gate"])
                if marker == "x":
                    ax.scatter(i, value, marker="x", color=color, s=48, linewidths=1.5, zorder=4)
                else:
                    ax.scatter(i, value, marker=marker, facecolors=color if filled else "white",
                               edgecolors=color, s=48, linewidths=1.3, zorder=4)
            n = str(coverages[0]) if len(set(coverages)) == 1 else f"{min(coverages)}–{max(coverages)}"
            p = str(precision_counts[0]) if len(set(precision_counts)) == 1 else f"{min(precision_counts)}–{max(precision_counts)}"
            undefined = sum(value is None for value in points)
            coverage_lines.append(f"{label.replace(' shuffle','')}: n={n}/{planned}; precision n={p}/{planned}" + (f"; {undefined} undefined" if undefined else ""))
        ax.text(0, -.19, "\n".join(coverage_lines), transform=ax.transAxes,
                va="top", fontsize=8, color=MUTED, linespacing=1.5)
    axes[0].set_ylabel("Original → shuffled mean-RDM stability (Spearman ρ)", fontsize=9, color=TEXT)
    return plotted


def depth_agreement_figure(controls, protocol):
    """Sparse fixed-state cross-family RSA; no interpolation to an exact layer."""
    states = ("input", "early", "depth75", "final_raw", "final_normalized")
    conditions = (("original_padded_bf16", "Original", TEXT),) + CONDITIONS
    pairs = (("qwen", "gemma"), ("qwen", "muse"), ("gemma", "muse"))
    planned = len(protocol["sonnets"])
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 8.8))
    fig.subplots_adjust(left=.075, right=.965, bottom=.34, top=.70, wspace=.24)
    plotted = []
    for ax, pair in zip(axes, pairs):
        style_axis(ax, grid="y")
        ax.axhline(0, color="#abb9c1", linewidth=.8, linestyle=":")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(-.18, 4.18)
        ax.set_xticks(range(5), ["Input", "Early*", "Depth\n0.75", "Final\nraw", "Final\nnorm"])
        ax.set_title(" / ".join(NAMES[name] for name in pair), fontsize=11, color=TEXT,
                     fontweight="bold", pad=13)
        coverage_lines = []
        for condition, label, color in conditions:
            rows = [exactly_one(controls["cross_model"], models=list(pair), state=state,
                                pooling="mean", condition=condition) for state in states]
            values = [np.nan if row["rho"] is None else row["rho"] for row in rows]
            ax.plot(range(5), values, color=color, linewidth=2 if label == "Original" else 1.45,
                    linestyle="-" if label == "Original" else "--",
                    alpha=1 if label == "Original" else .88, zorder=3 if label == "Original" else 2)
            failed_gates = []
            for i, state in enumerate(states):
                failures = []
                if label != "Original":
                    for family in pair:
                        if controls["models"].get(family, {}).get("status") == "loaded":
                            associated = exactly_one(controls["within_model"], model=family, state=state,
                                                     pooling="mean", condition=condition)
                            if associated["precision_gate"].get("passes_on_tested_subset") is False:
                                failures.append(family)
                failed_gates.append(failures)
                if rows[i]["rho"] is not None:
                    ax.scatter(i, rows[i]["rho"], marker="x" if failures else "o",
                               color=color, s=45 if failures else 21, linewidths=1.4, zorder=5)
            counts = [row["poem_count"] for row in rows]
            # Keep exact counts in state order if any comparison uses fewer poems.
            count_label = str(counts[0]) if len(set(counts)) == 1 else "/".join(map(str, counts))
            undefined = [state for state, row in zip(states, rows) if row["rho"] is None]
            line = f"{label.replace(' shuffle', '')}: n={count_label} of {planned}"
            if undefined:
                line += "; undefined " + ", ".join({"input": "input", "early": "early", "depth75": "0.75", "final_raw": "raw", "final_normalized": "norm"}[s] for s in undefined)
            coverage_lines.append(line)
            for state, row, failures in zip(states, rows, failed_gates):
                plotted.append({"models": list(pair), "state": state, "condition": condition,
                                "associated_within_model_mean_gate_failures": failures,
                                "rho": row["rho"], "status": row["status"],
                                "poem_count": row["poem_count"], "sonnets": row["sonnets"],
                                "complete_planned_cohort": row["complete_planned_cohort"]})
        ax.text(0, -.21, "\n".join(coverage_lines), transform=ax.transAxes,
                va="top", fontsize=8.5, color=MUTED, linespacing=1.5)
        if all(row["rho"] is None for row in plotted if row["models"] == list(pair)):
            ax.text(.5, .5, "No defined comparisons\nNo values substituted", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color=MUTED)
    axes[0].set_ylabel("Cross-family mean-RDM Spearman ρ", fontsize=10, color=TEXT)
    fig.text(.075, .935, "Where does cross-family agreement persist?", fontsize=20,
             fontweight="bold", color=TEXT)
    fig.text(.075, .89, f"Five fixed sampled states · {planned} postdiscovery sonnets · original and three order interventions",
             fontsize=11, color=MUTED)
    fig.text(.075, .845, "Same-condition BF16 comparisons of mean-pooled states; higher values mean more similar poem-to-poem geometry.",
             fontsize=10, color=MUTED)
    pair_values = []
    for pair in pairs:
        rows = [exactly_one(controls["cross_model"], models=list(pair), state="final_raw",
                            pooling="mean", condition=condition)
                for condition in ["original_padded_bf16", "token_shuffle_padded_bf16"]]
        pair_values.append([row["rho"] for row in rows])
    if all(value is not None for pair in pair_values for value in pair):
        names = ["Qwen/Gemma", "Qwen/Muse", "Gemma/Muse"]
        pair_note = "Final-raw token shuffle: " + "; ".join(f"{name} {old:.3f} → {new:.3f}" for name, (old, new) in zip(names, pair_values)) + "."
        fig.text(.075, .795, pair_note, fontsize=10, color=TEXT, fontweight="bold")
    failed_counts = []
    for family in FAMILIES:
        gates = [row["precision_gate"] for row in controls["within_model"]
                 if row["model"] == family and row["pooling"] == "mean" and row["state"] != "input"
                 and row["condition"] in [condition for condition, _, _ in CONDITIONS]]
        failures = sum(gate.get("passes_on_tested_subset") is False for gate in gates)
        failed_counts.append(f"{family.capitalize()} {failures}/{len(gates)}")
    fig.text(.075, .76, "Arithmetic sensitivity: failed post-input mean-control gates — " + "; ".join(failed_counts) + ". Crosses flag affected points.",
             fontsize=9, color="#8b4234", bbox={"facecolor": "#fff2eb", "edgecolor": "none", "pad": 5})
    handles = [Line2D([], [], color=color, linewidth=2 if label == "Original" else 1.5,
                      marker="o", markersize=4, linestyle="-" if label == "Original" else "--", label=label)
               for _, label, color in conditions]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.07, .15), ncol=4,
               frameon=False, fontsize=10, handlelength=2.5, columnspacing=2)
    fig.text(.075, .125, "* Early = block indices Qwen 1, Gemma 4, Muse 4 (zero-based). States are categorical; connections do not locate an exact transition layer.",
             fontsize=8.5, color=MUTED)
    fig.text(.075, .095, "Each point uses the recorded common poems for that pair and condition. Gaps are undefined values; differing n values follow the five-state order.",
             fontsize=8.5, color=MUTED)
    precision = []
    for family in FAMILIES:
        model = controls["models"].get(family, {})
        count = model.get("case_counts", {}).get("original_padded_fp32", 0)
        precision.append(f"{family.capitalize()} {count}/{planned}")
    fig.text(.075, .065, "FP32 original coverage: " + "; ".join(precision) + ". Crosses inherit failed within-model gates; they do not denote a new cross-family significance test.",
             fontsize=8.5, color=MUTED)
    fig.text(.075, .035, "Descriptive selected-sample evidence only. High agreement can survive disrupted order without implying literary understanding or an identified causal mechanism.",
             fontsize=8.5, color=MUTED)
    return fig, plotted


def write_figure(fig, paths, sources, values, kind):
    fig.savefig(paths[0], dpi=180, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight", metadata={"Title": kind, "Subject": "Descriptive frozen-protocol scientific comparisons"})
    plt.close(fig)
    receipt = {"schema_version": "followup-scientific-figure-v3", "figure_kind": kind,
               "source_files": sources,
               "plotting_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "plotted_values": values,
               "outputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                            "bytes": p.stat().st_size} for p in paths[:2]],
               "interpretation": "Descriptive visualization only; no new fitting, inference or significance test."}
    with paths[2].open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ROOT/"experiments/followup-v3/text-control-analysis.json")
    parser.add_argument("--endpoint", type=Path, default=ROOT/"experiments/followup-v3/endpoint-results.json")
    parser.add_argument("--editorial", type=Path, default=ROOT/"experiments/followup-v3/editorial-transfer.json")
    parser.add_argument("--protocol", type=Path, default=ROOT/"experiments/followup-v3/text-control-protocol.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/followup-v3")
    args = parser.parse_args()
    # Check actual inference-derived analysis first, before constructing any figure.
    controls, a = load_required(args.analysis)
    endpoint, b = load_required(args.endpoint)
    editorial, c = load_required(args.editorial)
    protocol, d = load_required(args.protocol)
    if controls["protocol"]["sha256"] != d["sha256"]:
        raise ValueError("Analysis and plotted protocol hashes differ")
    paths = [args.output_dir/f"followup-summary.{extension}" for extension in ["png", "pdf", "receipt.json"]]
    depth_paths = [args.output_dir/f"depth-agreement-and-order.{extension}" for extension in ["png", "pdf", "receipt.json"]]
    if any(path.exists() for path in paths + depth_paths):
        raise FileExistsError("Use a fresh output directory; prior figure evidence is never overwritten")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(14.5, 10.6))
    top = fig.add_gridspec(1, 2, left=.075, right=.965, bottom=.665, top=.855, wspace=.58)
    bottom = fig.add_gridspec(1, 3, left=.075, right=.965, bottom=.225, top=.495, wspace=.23)
    ax_a, ax_b = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])
    axes = [fig.add_subplot(bottom[0, i]) for i in range(3)]
    plotted = {"endpoint": endpoint_panel(ax_a, endpoint), "historical": historical_panel(ax_b, editorial),
               "gpu": controls_panels(axes, controls, protocol)}
    fig.text(.075, .96, "Shakespeare: three checks on the geometric signal", fontsize=19,
             fontweight="bold", color=TEXT, ha="left")
    fig.text(.075, .925, "Private exploratory research · saved-state readouts, historical-source transfer and locked text interventions",
             fontsize=10, color=MUTED)
    fig.text(.075, .575, "C  Text-order controls", fontsize=12, fontweight="bold", color=TEXT)
    fig.text(.075, .553, f"{len(protocol['sonnets'])} planned postdiscovery poems · original padded BF16 reference · nonspecial-token mean pooling",
             fontsize=9, color=MUTED)
    failed = []
    for family in FAMILIES:
        gates = [row["precision_gate"] for row in controls["within_model"]
                 if row["model"] == family and row["pooling"] == "mean" and row["state"] != "input"
                 and row["condition"] in [condition for condition, _, _ in CONDITIONS]]
        failed.append(f"{family.capitalize()} {sum(g.get('passes_on_tested_subset') is False for g in gates)}/{len(gates)}")
    fig.text(.075, .602, "Failed mean-control precision checks (including final norm): " + "; ".join(failed) + ".", fontsize=9,
             color="#8b4234", bbox={"facecolor": "#fff2eb", "edgecolor": "none", "pad": 5})
    condition_handles = [Line2D([], [], color=color, linewidth=1.8, label=label) for _, label, color in CONDITIONS]
    fig.legend(handles=condition_handles, loc="lower left", bbox_to_anchor=(.07, .104), ncol=3,
               frameon=False, fontsize=9, handlelength=2.5)
    gates = [Line2D([], [], marker="o", color=TEXT, linestyle="none", markersize=6, label="Precision gate: full-cohort pass"),
             Line2D([], [], marker="o", markerfacecolor="white", color=TEXT, linestyle="none", markersize=6, label="Subset pass"),
             Line2D([], [], marker="x", color=TEXT, linestyle="none", markersize=6, label="Failed"),
             Line2D([], [], marker="^", markerfacecolor="white", color=TEXT, linestyle="none", markersize=6, label="Not established")]
    fig.legend(handles=gates, loc="lower left", bbox_to_anchor=(.07, .073), ncol=4,
               frameon=False, fontsize=8.5, columnspacing=1.7)
    fig.text(.075, .047, "Precision gate requires original BF16/FP32 RDM ρ ≥ .95 and precision change ≤ 10% of control change on identical poems.",
             fontsize=8.5, color=MUTED)
    fig.text(.075, .025, "Coverage is per comparison; partial gate passes do not establish full-cohort robustness. High stability is not literary utility. No significance tests or causal-mechanism claim.",
             fontsize=8.5, color=MUTED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_figure(fig, paths, [a, b, c, d], plotted, "Shakespeare follow-up controls")
    depth_fig, depth_values = depth_agreement_figure(controls, protocol)
    write_figure(depth_fig, depth_paths, [a, d], depth_values, "Cross-family depth agreement and order interventions")
    print(json.dumps({"summary": [str(p) for p in paths], "depth_agreement": [str(p) for p in depth_paths]}, indent=2))



if __name__ == "__main__":
    main()
