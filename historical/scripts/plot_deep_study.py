"""Static, source-derived figures for the second-pass research report."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("experiments/deep-study-v2")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 180, "font.family": "DejaVu Sans"})


def save(fig, name):
    for suffix in ("png", "pdf"):
        path = ROOT / f"{name}.{suffix}"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    editorial = json.loads((ROOT / "editorial-final-state-sensitivity.json").read_text())
    single = json.loads((ROOT / "editorial-single-depth.json").read_text())
    names = ["conventional", "plus_both_final", "selected_single_depth", "final_plus_fixed_depths", "final_plus_all_blocks", "plus_early_blocks"]
    labels = ["Text baselines", "Text + final states", "Train-selected single depth", "+ fixed depths", "+ every block", "Text + early blocks"]
    colors = ["#677580", "#4C78A8", "#D39839", "#008574", "#7AA89C", "#9E77A8"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for ax, version in zip(axes, editorial["versions"]):
        single_version, = [v for v in single["versions"] if v["exclude154"] == version["exclude154"]]
        version["metrics"]["selected_single_depth"] = single_version["metrics"]
        values = [version["metrics"][name]["log_loss"] for name in names]
        ax.scatter(values, np.arange(len(names)), color=colors, s=70, zorder=3)
        ax.grid(axis="x", alpha=.18)
        ax.set_yticks(np.arange(len(names)), labels); ax.invert_yaxis()
        ax.set_xlim(0.48, 0.61)
        for i, value in enumerate(values):
            ax.text(value+.0015, i, f"{value:.3f}", va="center", fontsize=9)
        ax.set_xlabel("Out-of-fold log loss · lower is better")
        ax.set_title("Without Sonnet 154 · 145 pairs" if version["exclude154"] else "Full diagnostic · 146 pairs")
        ax.axvline(version["metrics"]["prevalence"]["log_loss"], color="#999999", ls=":", lw=1)
    fig.suptitle("Editorially linked neighboring sonnets: a modest intermediate-depth lead", fontsize=14)
    save(fig, "editorial-heldout-comparison")

    reports = [json.loads((ROOT / file).read_text()) for file in ("volta-diagnostic.json", "volta-edition-sensitivity.json")]
    names = ["conventional", "plus_input", "plus_final", "plus_all_depths", "text_vectors", "text_vectors_plus_all_depths"]
    labels = ["Compact text baseline", "+ model inputs", "+ final depth", "+ five depths", "Full BERT text vectors", "BERT vectors + five depths"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")
    for ax, report, title in zip(axes, reports, ("All 96 labeled poems", "91 after word-variant exclusion")):
        values = [report["metrics"][name]["log_loss"] for name in names]
        ax.scatter(values, np.arange(len(names)), color=["#008574", "#9DAAB6", "#4C78A8", "#7AA89C", "#B3A3BC", "#9E77A8"], s=70, zorder=3)
        ax.grid(axis="x", alpha=.18)
        ax.set_yticks(np.arange(len(names)), labels); ax.invert_yaxis()
        ax.set_xlim(.50, .68)
        for i, value in enumerate(values):
            ax.text(value+.002, i, f"{value:.3f}", va="center", fontsize=9)
        ax.set_xlabel("Out-of-fold log loss · lower is better"); ax.set_title(title)
        ax.axvline(report["metrics"]["prevalence"]["log_loss"], color="#999999", ls=":", lw=1)
    fig.suptitle("Implicit line-9 turns: activation features do not beat the compact text baseline", fontsize=13)
    save(fig, "volta-heldout-comparison")

    line = json.loads((ROOT / "line-discovery.json").read_text())
    pairs = sorted({(t["model_a"], t["model_b"]) for t in line["tests"]})
    short = {"Qwen/Qwen3-8B": "Qwen", "google/gemma-4-31B-it": "Gemma", "meta-models/Muse-Glimmer-30B": "Muse"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, layout="constrained")
    for ax, pair in zip(axes, pairs):
        for metric, view, color, label in [("cumulative_cosine_change", "raw", "#9CA7B0", "Original cumulative change"),
                                          ("line_context_cosine_distance", "raw", "#4C78A8", "New-line contrast"),
                                          ("line_context_cosine_distance", "cross_fitted_controls", "#008574", "New-line contrast + controls")]:
            rows = sorted([t for t in line["tests"] if (t["model_a"],t["model_b"]) == pair and
                           t["metric"] == metric and t["view"] == view], key=lambda t:t["requested_depth"])
            if len(rows) != 5:
                raise ValueError("Expected every declared depth in each curve")
            values = [t["mean_within_poem_spearman"] if t["status"] == "computed" else np.nan for t in rows]
            ax.plot([t["requested_depth"] for t in rows], values, marker="o", ms=4, color=color, label=label)
        ax.axhline(0, color="#999999", lw=.7)
        ax.set_ylim(-.1, 1.02); ax.set_xticks([0,.25,.5,.75,1]); ax.set_xlabel("Normalized residual depth")
        ax.set_title("–".join(short[x] for x in pair))
    axes[0].set_ylabel("Mean within-poem rank agreement")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncols=3, frameon=False)
    fig.suptitle("Removing cumulative-mean dilution changes the apparent cross-model agreement", fontsize=13)
    save(fig, "line-contribution-sensitivity")


if __name__ == "__main__":
    main()
