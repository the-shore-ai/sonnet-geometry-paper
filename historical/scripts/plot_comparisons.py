"""Scientific summary figures; source numerical reports remain authoritative."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--rsa", type=Path, required=True)
parser.add_argument("--structure", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
rsa = json.loads(args.rsa.read_text())
structure = json.loads(args.structure.read_text())
for name, source in (("RSA", rsa), ("structure", structure)):
    if len(source["sonnet_ids"]) != len(set(source["sonnet_ids"])):
        raise ValueError(f"{name} report has duplicate sonnet identities")
if set(rsa["sonnet_ids"]) != set(structure["sonnet_ids"]):
    raise ValueError("RSA and structure reports must describe the same sonnets")
decimals = rsa.get("distance_policy", {}).get("decimals")
if decimals != structure.get("distance_policy", {}).get("decimals"):
    raise ValueError("RSA and structure reports must use the same distance precision policy")
precision_label = "" if decimals is None else f"\nNumerical sensitivity: cosine distances rounded to {decimals} decimal places"
args.output.mkdir(parents=True, exist_ok=False)
pairings = list(dict.fromkeys((x["model_a"], x["model_b"]) for x in rsa["cross_model_tests"]))
colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)
    for column, pool in enumerate(("last_token", "cumulative_mean")):
        for index, pair in enumerate(pairings):
            label = " / ".join(x.split("/")[-1] for x in pair)
            values = sorted((x for x in rsa["cross_model_tests"] if (x["model_a"], x["model_b"]) == pair and x["pooling"] == pool), key=lambda x: x["requested_depth"])
            adjusted = sorted((x for x in structure["comparisons"] if (x["model_a"], x["model_b"]) == pair and x["pooling"] == pool), key=lambda x: x["requested_depth"])
            axes[0, column].plot([x["requested_depth"] for x in values], [np.nan if x["rsa_spearman"] is None else x["rsa_spearman"] for x in values], "o-", color=colors[index % len(colors)], label=label)
            axes[1, column].plot([x["requested_depth"] for x in adjusted], [np.nan if x["nuisance_adjusted_rsa"]["partial_rank_correlation"] is None else x["nuisance_adjusted_rsa"]["partial_rank_correlation"] for x in adjusted], "o-", color=colors[index % len(colors)], label=label)
        axes[0, column].set_title(pool.replace("_", " "))
        axes[1, column].set_xlabel("Normalized residual depth (embedding = 0)")
        for ax in axes[:, column]:
            ax.set_ylim(-1, 1.04)
            ax.axhline(0, color="0.6", linewidth=0.6)
            ax.set_xticks([0, .25, .5, .75, 1])
    axes[0, 0].set_ylabel("Whole-sonnet RSA (Spearman)")
    axes[1, 0].set_ylabel("Nuisance-adjusted rank correlation\n(descriptive, fitted on same corpus)")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(f"{len(rsa['sonnet_ids'])} sonnets: cross-model geometry and baseline controls" + precision_label)
    fig.savefig(args.output / "cross-model-agreement.png", dpi=180)
    fig.savefig(args.output / "cross-model-agreement.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(structure["models"]), figsize=(5 * len(structure["models"]), 4.8), constrained_layout=True, squeeze=False)
    for ax, model in zip(axes[0], structure["models"]):
        setting = next(x for x in model["settings"] if x["pooling"] == "cumulative_mean" and x["requested_depth"] == 1)
        display = setting["two_dimensional_display"]
        coordinates = np.asarray(display["coordinates"])
        ax.scatter(*coordinates.T, c="#0072B2", s=16, alpha=.65)
        scores = np.asarray(setting["anomaly_scores"])
        for i in np.argsort(-scores, kind="stable")[:5]:
            ax.annotate(str(int(structure["sonnet_ids"][i].split("-")[-1])), coordinates[i], xytext=(3, 3), textcoords="offset points", fontsize=8)
        fraction = display["two_axis_positive_inertia_fraction"]
        fraction_label = "undefined" if fraction is None else f"{fraction:.1%}"
        ax.set_title(f"{model['model_id'].split('/')[-1]}\nTwo-axis positive inertia: {fraction_label}")
        ax.set_xlabel("Display axis 1")
        ax.set_ylabel("Display axis 2")
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Final residual · cumulative mean · classical chord-distance display\nLabels: five highest mean-distance anomaly scores; axes are model-specific" + precision_label)
    fig.savefig(args.output / "sonnet-projections.png", dpi=180)
    fig.savefig(args.output / "sonnet-projections.pdf")
    plt.close(fig)
print(json.dumps({"output": str(args.output), "figures": ["cross-model-agreement", "sonnet-projections"]}))
