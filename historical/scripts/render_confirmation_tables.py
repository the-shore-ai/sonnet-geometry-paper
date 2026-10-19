"""Render every registered comparison summary, including failed/undefined cells."""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

from plot_confirmation import load_actual
from sonnet_geometry.confirmation_analysis import FAMILIES, INTERVENTIONS, finite_summary
from sonnet_geometry.storage import write_exclusive
from sonnet_geometry.text_control_analysis import POOLS, STATES, file_receipt


def number(value):
    return "undefined" if value is None else f"{value:.3f}"


def interval(summary):
    if summary["computed_count"] != summary["expected_count"]:
        return f"undefined ({summary['computed_count']}/{summary['expected_count']} defined)"
    return f"{number(summary['median'])} [{number(summary['minimum'])}, {number(summary['maximum'])}]"


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows), ""]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/confirmation-v4/protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/confirmation-v4-comparison-grid.md"))
    parser.add_argument("--descriptive-recovery", action="store_true")
    args = parser.parse_args()
    analysis, protocol = load_actual(args.analysis, args.protocol, args.descriptive_recovery)
    namespace = protocol.get("study_namespace", "confirmation-v4")
    title = "Complete v4 comparison grid" if namespace == "confirmation-v4" else "Complete Spenser comparison grid"
    lines = ["# " + title, "",
             "Every registered model, pair, sampled state, readout, intervention and arithmetic condition is retained here. "
             "Values are median [minimum, maximum] across the same five frozen seeds. Ranges are not confidence intervals. "
             "Undefined cells remain explicit; a partial set is not presented as a successful five-seed summary.", ""]
    if args.descriptive_recovery:
        lines += ["**Strict three-model confirmation is incomplete.** Muse's captures were fully read back and pooled locally, "
                  "but its final model-wide runtime/precision record is missing. Muse-dependent numerical results are conditional "
                  "on those recovered captures. Qwen and Gemma retain their completed worker records.", ""]
    lines += ["The primary representation is the nonspecial-token mean under token-inventory shuffling in the FP32-labeled condition. "
              "Last-token and lexical-endpoint readouts and word/line interventions remain separate controls. FP32 promotes the loaded "
              "BF16 checkpoint values. Exact token shuffles differ across tokenizers; word shuffles change token inventories. "
              "Sparse sampled states do not identify a transition layer.", "",
              f"Analysis SHA256: `{file_receipt(args.analysis)['sha256']}`. "
              f"See [findings]({namespace}-findings.md), [machine-readable summary](../experiments/{namespace}/results-summary.json), "
              f"and [private artifact receipts](../experiments/{namespace}/derived-storage-receipt.json).", ""]
    indexed = {}
    for row in analysis["seed_summaries"]:
        subject = row.get("model", tuple(row.get("models", ())))
        identity = row["precision"], row["pooling"], row["intervention"], row["metric"], subject, row["state"]
        if identity in indexed:
            raise ValueError("Duplicate complete-grid summary")
        indexed[identity] = row
    if len(indexed) != 540:
        raise ValueError("Incomplete registered summary grid")
    rendered = 0
    for precision in ("fp32", "bf16"):
        for pool in POOLS:
            lines += [f"## {precision.upper()} · {pool}", ""]
            for control in INTERVENTIONS:
                lines += [f"### {control}: original versus perturbed RDM stability", ""]
                rows = []
                for family in FAMILIES:
                    rows.append([family, *(interval(indexed[precision, pool, control, "within_model_stability", family, state])
                                            for state in STATES)])
                    rendered += len(STATES)
                lines += table(["Model", *STATES], rows)
                lines += [f"### {control}: cross-model agreement change (perturbed − original)", ""]
                rows = []
                for pair in combinations(FAMILIES, 2):
                    rows.append([" / ".join(pair), *(interval(indexed[precision, pool, control, "cross_model_change", pair, state])
                                                     for state in STATES)])
                    rendered += len(STATES)
                lines += table(["Pair", *STATES], rows)
            lines += ["### Original cross-model agreement", ""]
            rows = []
            for pair in combinations(FAMILIES, 2):
                selected = {r["state"]: r for r in analysis["cross_model"] if r["models"] == list(pair)
                            and r["intervention"] == "original" and r["precision"] == precision and r["pooling"] == pool}
                if set(selected) != set(STATES):
                    raise ValueError("Incomplete original cross-model reference grid")
                rows.append([" / ".join(pair), *(number(selected[state]["rho"]) for state in STATES)])
            lines += table(["Pair", *STATES], rows)
    if rendered != 540:
        raise ValueError("Not every registered seed summary was rendered")
    lines += ["## Paired arithmetic diagnostics", "",
              "P/F/U gives pass/fail/unresolved seed counts. Oρ is the original BF16/FP32 correlation; Pρ gives the "
              "perturbed correlations. Ratio is the maximum of original/perturbed median arithmetic distance differences divided "
              "by the FP32 original-versus-perturbed median difference. A pass requires Oρ and Pρ ≥ .95 and ratio ≤ .10. "
              "Zero or unresolved intervention differences cannot pass. Δ stability is FP32 minus BF16 original/perturbed correlation. "
              "These are representation diagnostics, not proof of model-wide precision or mathematical exactness.", ""]
    rendered_gates = 0
    for family in FAMILIES:
        for pool in POOLS:
            lines += [f"### {family} · {pool}", ""]
            rows = []
            for control in INTERVENTIONS:
                for state in STATES:
                    selected = [r for r in analysis["arithmetic_gates"] if
                                (r["model"], r["pooling"], r["intervention"], r["state"]) ==
                                (family, pool, control, state)]
                    if len(selected) != 5:
                        raise ValueError("Incomplete paired arithmetic gate")
                    counts = Counter("P" if r["passes"] is True else "F" if r["passes"] is False else "U" for r in selected)
                    rows.append([control, state, "/".join(str(counts[k]) for k in ("P", "F", "U")),
                                 number(selected[0]["original_precision_rho"]),
                                 *(interval(finite_summary([r[field] for r in selected])) for field in
                                   ("perturbed_precision_rho", "precision_to_fp32_intervention_delta_ratio",
                                    "fp32_minus_bf16_stability"))])
                    rendered_gates += len(selected)
            lines += table(["Intervention", "State", "P/F/U", "Oρ", "Pρ", "Ratio", "Δ stability"], rows)
    if rendered_gates != 675:
        raise ValueError("Not every paired arithmetic gate was rendered")
    lines += ["## Planned single-poem omissions", "",
              "Each row summarizes all 24 omissions × five seeds. These are dependent sensitivity checks, not independent replications "
              "or confidence intervals. Positive direction is required in every one of the 120 values for the corresponding sensitivity label.", ""]
    rows = []
    for claim in analysis["claims"]:
        sensitivity = claim["omission_direction_sensitivity"]
        subject = claim.get("model", " / ".join(claim.get("models", ())))
        selected = [r for r in analysis["planned_omission_sensitivity"]["rows"] if
                    (r.get("model") == claim["model"] if "model" in claim else r.get("models") == claim["models"])]
        if len(selected) != 120:
            raise ValueError("Incomplete omission contrast")
        rows.append([subject, interval(sensitivity), sum(r["delta"] is not None and r["delta"] <= 0 for r in selected),
                     sum(r["delta"] is None for r in selected), claim["sensitivity_qualified_status"]])
    lines += table(["Model or pair", "Delta", "Nonpositive", "Undefined", "Numerical sensitivity decision"], rows)
    lines += ["The complete source analysis retains all 1,440 within-model comparisons, 1,440 cross-model comparisons, "
              "720 arithmetic comparisons, 675 gates, 540 seed summaries, 1,440 distance vectors and 720 omission rows. "
              "No human literary ratings are supplied by these tables.", ""]
    write_exclusive(args.output, "\n".join(lines).encode())
    print({"output": str(args.output), "seed_summaries_rendered": rendered, "arithmetic_gates_rendered": rendered_gates})


if __name__ == "__main__":
    main()
