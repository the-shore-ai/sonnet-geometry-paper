"""Describe transition-position patterns from completed saved execution reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sonnet_geometry.line_positions import analyze_line_positions
from sonnet_geometry.storage import sha256_file


def plot_positions(report: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    models = report["models"]
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        for kind, title, filename in [
            ("sonnet_130", "Sonnet 130: adjacent cumulative-prefix cosine changes", "sonnet-130-line-changes"),
            ("standard_form_position_profile", "152 fourteen-line poems: mean change at each transition position", "mean-transition-position-profile"),
        ]:
            fig, axes = plt.subplots(len(models), 2, figsize=(12, 3.0 * len(models)), squeeze=False,
                                     sharex=True, constrained_layout=True)
            for model_index, model in enumerate(models):
                for pool_index, pool in enumerate(["last_token", "cumulative_mean"]):
                    ax = axes[model_index, pool_index]
                    settings = [setting for setting in model["settings"] if setting["pooling"] == pool]
                    for index, setting in enumerate(settings):
                        profile = setting[kind]
                        values = profile["cosine_changes"] if kind == "sonnet_130" else profile["mean_cosine_change"]
                        if values is not None:
                            ax.plot(profile["transition_to_lines"], [np.nan if value is None else value for value in values],
                                    "o-", markersize=3, linewidth=1.2, color=colors[index % len(colors)],
                                    label=f"depth {setting['requested_depth']:g}")
                    for entry in [5, 9, 13]:
                        ax.axvline(entry, color="0.75", linestyle=":", linewidth=.8, zorder=0)
                    ax.set_title(f"{model['model_id'].split('/')[-1]} · {pool.replace('_', ' ')}")
                    ax.set_xticks(range(2, 15))
                    ax.set_ylabel("Cosine change")
                    if model_index == len(models) - 1:
                        ax.set_xlabel("Transition entering verse line")
            axes[0, 0].legend(fontsize=8, ncol=3)
            fig.suptitle(title + "\nDotted lines mark conventional entries 5/9/13; scales vary by panel; descriptive only")
            fig.savefig(output / f"{filename}.png", dpi=180)
            fig.savefig(output / f"{filename}.pdf")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True,
                        help="Full completed execution reports containing results[].summary")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, help="New directory for Sonnet 130 and mean-profile PNG/PDF figures")
    args = parser.parse_args()
    if args.output.exists() or (args.figures is not None and args.figures.exists()):
        raise FileExistsError("Output JSON and optional figure directory must be new")
    reports = [json.loads(path.read_text()) for path in args.reports]
    result = analyze_line_positions(reports)
    result["source_files"] = [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in args.reports]
    encoded = json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    if args.figures is not None:
        plot_positions(result, args.figures)
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size,
                      "models": len(reports), "standard_form_sonnets": 152, "inferential_tests": 0,
                      "figures": str(args.figures) if args.figures is not None else None}))


if __name__ == "__main__":
    main()
