"""Post-hoc leave-one-poem-out sensitivity, separate from the frozen primary output.

Requested after the preliminary Qwen result on2026-10-18. Every one of the12
planned omissions is retained. No inference, refitting, p-values or subset choice.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path

from sonnet_geometry.storage import canonical_json, write_exclusive
from sonnet_geometry.text_control_analysis import (
    BASELINE, FAMILIES, POOLS, STATES, available, comparison, file_receipt,
    load_run, precision_gate, cosine_rdm, matrix, upper, q12_spearman,
)

POLICY = {
    "status": "posthoc_after_qwen_preliminary",
    "request_date": "2026-10-18",
    "timing": "Requested after inspecting the complete Qwen preliminary result, before final three-model primary analysis. Not part of the frozen primary protocol.",
    "omissions": "Every one of the12originally planned sonnets, once each; no replacement and no selected subset.",
    "comparisons": "Every planned model, condition, state and pooling, plus every model pair with matched conditions/states/poolings. Cross-model condition-minus-original deltas use identical remaining poem identities.",
    "cross_model_extension_timing": "Requested after the Qwen/Muse preliminary showed greater cross-model agreement after disruption, before final three-model primary analysis. Every pair and omission is retained, not only the initially favorable channels.",
    "metrics": "Q12 RDM rho, raw median absolute RDM delta, median paired-vector cosine distance, and unchanged arithmetic-gate criteria on the same remaining-poem intersection.",
    "coverage": "A partial execution stays partial; omitting an already unavailable poem is flagged as no actual removal. A gate pass refers only to the omission subset.",
    "summary": "All per-omission values plus ranges/counts across all omissions, without selecting favorable omissions.",
    "inference": "Descriptive stability audit in a postdiscovery sample; no p-values, confidence interval, independent confirmation, or new semantic interpretation.",
}


def validate_primary(primary, runs, protocol_sha256):
    if primary["schema_version"] != "text-control-analysis-v3":
        raise ValueError("Finalized primary analysis required")
    if primary["protocol"]["sha256"] != protocol_sha256:
        raise ValueError("Primary protocol differs")
    for family in FAMILIES:
        saved = primary["models"][family]
        if family not in runs:
            if saved["status"] != "missing_model_report":
                raise ValueError("A primary model is missing from sensitivity inputs")
            continue
        if saved["status"] != "loaded":
            raise ValueError("Sensitivity would add a model absent from finalized primary")
        old = {Path(r["path"]).name: (r["sha256"], r["bytes"]) for r in saved["source_files"]}
        current = {Path(r["path"]).name: (r["sha256"], r["bytes"]) for r in runs[family].receipts}
        if old != current:
            raise ValueError("Sensitivity source artifacts differ from finalized primary")


def finite_range(values):
    values = [v for v in values if v is not None]
    return {"computed_omissions": len(values), "minimum": min(values) if values else None,
            "maximum": max(values) if values else None}


def build_sensitivity(runs, protocol):
    planned = protocol["sonnets"]
    if len(planned) != 12 or planned != sorted(set(planned)):
        raise ValueError("Exactly the12unique frozen poem IDs are required")
    rows, ranges = [], []
    for family in FAMILIES:
        for condition in protocol["conditions"]:
            for state in STATES:
                for pooling in POOLS:
                    group = []
                    for omitted in planned:
                        row = {"model": family, "condition": condition, "state": state,
                               "pooling": pooling, "omitted_sonnet": omitted,
                               "rho": None, "median_absolute_rdm_change": None,
                               "median_paired_cosine_distance": None,
                               "gate_precision_to_control_delta_ratio": None,
                               "gate_precision_rho": None,
                               "passes_gate_on_omission_subset": None}
                        remaining = [n for n in planned if n != omitted]
                        if family not in runs:
                            row.update(status="missing_model_report", available_poems=0,
                                missing_remaining_sonnets=remaining, omission_was_observed=False,
                                gate_status="missing_model_report", gate_available_poems=0,
                                gate_complete_omission_cohort=False)
                        else:
                            run = runs[family]
                            observed = available(run, condition, state, pooling, planned) & available(run, BASELINE, state, pooling, planned)
                            numbers = [n for n in remaining if n in observed]
                            metrics = comparison(run, numbers, condition, state, pooling)
                            gate = precision_gate(run, condition, state, pooling, remaining)
                            gate_status = gate["status"]
                            if gate_status == "passed_full_cohort":
                                gate_status = "passed_complete_omission_subset"
                            elif gate_status == "passed_partial_cohort":
                                gate_status = "passed_partial_omission_subset"
                            row.update(status=metrics["rdm_stability"]["status"],
                                available_poems=len(numbers), missing_remaining_sonnets=[n for n in remaining if n not in observed],
                                omission_was_observed=omitted in observed,
                                rho=metrics["rdm_stability"]["rho"],
                                median_absolute_rdm_change=metrics["median_absolute_rdm_change"],
                                median_paired_cosine_distance=metrics["median_paired_cosine_distance"],
                                gate_status=gate_status,
                                gate_available_poems=gate.get("poem_count"),
                                gate_complete_omission_cohort=gate.get("complete_planned_cohort", False),
                                gate_precision_to_control_delta_ratio=gate.get("precision_to_control_delta_ratio"),
                                gate_precision_rho=gate.get("precision_rdm_stability", {}).get("rho"),
                                passes_gate_on_omission_subset=gate.get("passes_on_tested_subset"))
                        group.append(row)
                    rows.extend(group)
                    ranges.append({"model": family, "condition": condition, "state": state,
                        "pooling": pooling, "omission_count": len(group),
                        "omitted_sonnets": [r["omitted_sonnet"] for r in group],
                        "available_poems_range": finite_range(r["available_poems"] for r in group),
                        "rho_range": finite_range(r["rho"] for r in group),
                        "median_absolute_rdm_change_range": finite_range(r["median_absolute_rdm_change"] for r in group),
                        "median_paired_cosine_distance_range": finite_range(r["median_paired_cosine_distance"] for r in group),
                        "gate_status_counts": dict(Counter(r["gate_status"] for r in group)),
                        "rho_status_counts": dict(Counter(r["status"] for r in group)),
                        "omission_observed_count": sum(r["omission_was_observed"] for r in group)})
    cross_rows, cross_ranges = cross_model_sensitivity(runs, protocol)
    return {"omission_rows": rows, "ranges": ranges,
            "cross_model_omission_rows": cross_rows, "cross_model_ranges": cross_ranges,
            "expected_omission_rows": len(FAMILIES)*len(protocol["conditions"])*len(STATES)*len(POOLS)*12,
            "expected_cross_model_omission_rows": 3*len(protocol["conditions"])*len(STATES)*len(POOLS)*12}


def cross_model_sensitivity(runs, protocol):
    """Compute each cross-family contrast on exactly the same remaining poems."""
    planned = protocol["sonnets"]
    rows, ranges = [], []
    for left, right in combinations(FAMILIES, 2):
        for condition in protocol["conditions"]:
            for state in STATES:
                for pooling in POOLS:
                    group = []
                    for omitted in planned:
                        remaining = [n for n in planned if n != omitted]
                        row = {"models": [left, right], "condition": condition, "state": state,
                               "pooling": pooling, "omitted_sonnet": omitted,
                               "rho_condition": None, "rho_original_padded_bf16": None,
                               "delta_rho_condition_minus_original": None}
                        if left not in runs or right not in runs:
                            row.update(status="missing_model_report", available_poems=0,
                                missing_remaining_sonnets=remaining, omission_was_observed=False,
                                condition_status="missing_model_report", original_status="missing_model_report")
                        else:
                            observed = set(planned)
                            for family in [left, right]:
                                for compared_condition in [BASELINE, condition]:
                                    observed &= available(runs[family], compared_condition, state, pooling, planned)
                            numbers = [n for n in remaining if n in observed]
                            correlations = []
                            for compared_condition in [condition, BASELINE]:
                                if numbers:
                                    distances = [upper(cosine_rdm(matrix(runs[f], numbers, compared_condition, state, pooling))) for f in [left, right]]
                                    correlations.append(q12_spearman(*distances))
                                else:
                                    correlations.append({"rho": None, "status": "no_common_poems"})
                            changed, original = correlations
                            defined = changed["rho"] is not None and original["rho"] is not None
                            row.update(status="computed" if defined else "undefined_paired_contrast",
                                available_poems=len(numbers), missing_remaining_sonnets=[n for n in remaining if n not in observed],
                                omission_was_observed=omitted in observed,
                                condition_status=changed["status"], original_status=original["status"],
                                rho_condition=changed["rho"], rho_original_padded_bf16=original["rho"],
                                delta_rho_condition_minus_original=changed["rho"]-original["rho"] if defined else None)
                        group.append(row)
                    rows.extend(group)
                    deltas = [r["delta_rho_condition_minus_original"] for r in group]
                    ranges.append({"models": [left, right], "condition": condition, "state": state,
                        "pooling": pooling, "omission_count": len(group),
                        "omitted_sonnets": [r["omitted_sonnet"] for r in group],
                        "available_poems_range": finite_range(r["available_poems"] for r in group),
                        "rho_condition_range": finite_range(r["rho_condition"] for r in group),
                        "rho_original_range": finite_range(r["rho_original_padded_bf16"] for r in group),
                        "paired_delta_rho_range": finite_range(deltas),
                        "positive_delta_omissions": sum(v is not None and v > 0 for v in deltas),
                        "negative_delta_omissions": sum(v is not None and v < 0 for v in deltas),
                        "status_counts": dict(Counter(r["status"] for r in group)),
                        "omission_observed_count": sum(r["omission_was_observed"] for r in group)})
    return rows, ranges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/followup-v3/text-controls"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/followup-v3/text-control-protocol.json"))
    parser.add_argument("--primary", type=Path, default=Path("experiments/followup-v3/text-control-analysis.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/followup-v3/text-control-loo.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    primary = json.loads(args.primary.read_text())  # Do not execute before the primary output exists.
    protocol = json.loads(args.protocol.read_text())
    receipt = file_receipt(args.protocol)
    runs = {}
    for family in FAMILIES:
        report, pooled = args.input_root/family/"report.json", args.input_root/family/"pooled.safetensors"
        if not report.exists() and not pooled.exists():
            continue
        if not report.exists() or not pooled.exists():
            raise FileNotFoundError(f"Incomplete local input pair for {family}")
        runs[family] = load_run(report, pooled, family, protocol, receipt["sha256"])
    if not runs:
        raise FileNotFoundError("No actual control reports")
    validate_primary(primary, runs, receipt["sha256"])
    result = {"schema_version": "text-control-loo-posthoc-v3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "policy": POLICY,
        "protocol": receipt, "primary": file_receipt(args.primary),
        "source_files": [file_receipt(__file__), file_receipt("src/sonnet_geometry/text_control_analysis.py")],
        "models": {family: {"experiment_id": run.report["experiment_id"], "source_files": run.receipts} for family, run in runs.items()},
        **build_sensitivity(runs, protocol)}
    write_exclusive(args.output, canonical_json(result))
    print(json.dumps({"output": str(args.output), "sha256": file_receipt(args.output)["sha256"],
                      "omission_rows": len(result["omission_rows"]), "range_rows": len(result["ranges"]),
                      "cross_model_omission_rows": len(result["cross_model_omission_rows"])}))


if __name__ == "__main__":
    main()
