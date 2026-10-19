"""Descriptive analysis of frozen paired text controls, with explicit missingness."""
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

FAMILIES = ("qwen", "gemma", "muse")
STATES = ("input", "early", "depth75", "final_raw", "final_normalized")
POOLS = ("mean", "last", "last_lexical")
BASELINE = "original_padded_bf16"
PRECISION = "original_padded_fp32"
CONTROLS = ("word_shuffle_padded_bf16", "line_shuffle_padded_bf16", "token_shuffle_padded_bf16")


def file_receipt(path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


@dataclass
class ModelRun:
    family: str
    report: dict
    vectors: dict
    receipts: list


def key(number, condition, state, pooling):
    return f"{number:03d}.{condition}.{state}.{pooling}"


def load_run(report_path, pooled_path, family, protocol, protocol_sha256):
    """Fail closed on altered bytes, mislabeled runs, orphan vectors or shape errors."""
    from safetensors.numpy import load_file
    wrapper = json.loads(Path(report_path).read_text())
    report = wrapper.get("report", wrapper)
    if report["schema_version"] != "text-controls-execution-v3" or report["model"] != family:
        raise ValueError("Execution report identity mismatch")
    if report["config"]["protocol_sha256"] != protocol_sha256:
        raise ValueError("Frozen protocol hash differs from executed protocol")
    pooled_receipt = file_receipt(pooled_path)
    if any(pooled_receipt[k] != report["pooled"][k] for k in ["sha256", "bytes"]):
        raise ValueError("Pooled artifact checksum or size mismatch")
    vectors = load_file(str(pooled_path))
    expected_keys, seen = set(), set()
    for case in report["cases"]:
        number, condition = case["sonnet"], case["condition"]
        if number not in protocol["sonnets"] or condition not in protocol["conditions"]:
            raise ValueError("Unregistered poem or condition")
        if (number, condition) in seen:
            raise ValueError("Duplicate execution case")
        seen.add((number, condition))
        ids = case["tokenization"]["input_ids"]
        length = case["forward_length"]
        if not ids or length < len(ids):
            raise ValueError("Invalid real-token or forward length")
        if case["attention_mask"] != [1] * len(ids) + [0] * (length-len(ids)):
            raise ValueError("Execution attention mask differs from registered right padding")
        if case["position_ids"] != list(range(length)):
            raise ValueError("Execution positions differ from registered positions")
        expected_length = len(ids) if condition == "original_native_bf16" else protocol["pad_length"]
        if length != expected_length:
            raise ValueError("Execution forward shape differs from frozen protocol")
        expected_keys.update(key(number, condition, state, pool) for state in STATES for pool in POOLS)
    if set(vectors) != expected_keys:
        raise ValueError("Missing or orphan pooled vectors")
    widths = set()
    for value in vectors.values():
        if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
            raise ValueError("Invalid pooled vector shape or finite values")
        if np.linalg.norm(value.astype(np.float64)) == 0:
            raise ValueError("Zero-norm pooled vector has undefined cosine")
        widths.add(len(value))
    if len(widths) > 1:
        raise ValueError("Inconsistent vector widths within one checkpoint")
    actual_complete = len(seen) == len(protocol["sonnets"]) * len(protocol["conditions"])
    if bool(report["complete"]) != actual_complete:
        raise ValueError("Execution completion flag contradicts cases")
    return ModelRun(family, report, vectors, [file_receipt(report_path), pooled_receipt])


def cosine_rdm(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError("A finite nonempty vector matrix is required")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError("Zero-norm vectors have undefined cosine")
    normalized = vectors / norms[:, None]
    distances = np.clip(1 - normalized @ normalized.T, 0, 2)
    np.fill_diagonal(distances, 0.)
    return distances


def upper(matrix):
    return matrix[np.triu_indices(len(matrix), 1)]


def q12_spearman(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Matched one-dimensional RDM vectors required")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Nonfinite RDM vector")
    if len(left) < 2:
        return {"rho": None, "status": "insufficient_pairs", "rdm_pairs": len(left)}
    a, b = rankdata(np.round(left, 12)), rankdata(np.round(right, 12))
    constant_a, constant_b = bool(np.ptp(a) == 0), bool(np.ptp(b) == 0)
    if constant_a or constant_b:
        return {"rho": None, "status": "constant_after_q12", "rdm_pairs": len(left),
                "left_constant": constant_a, "right_constant": constant_b}
    return {"rho": float(np.clip(np.corrcoef(a, b)[0, 1], -1, 1)), "status": "computed", "rdm_pairs": len(left)}


def available(run, condition, state, pooling, planned):
    return {n for n in planned if key(n, condition, state, pooling) in run.vectors}


def matrix(run, numbers, condition, state, pooling):
    return np.stack([run.vectors[key(n, condition, state, pooling)] for n in numbers]).astype(np.float64)


def coverage(numbers, planned):
    return {"sonnets": list(numbers), "poem_count": len(numbers),
            "missing_sonnets": [n for n in planned if n not in numbers], "complete_planned_cohort": list(numbers) == list(planned)}


def comparison(run, numbers, condition, state, pooling):
    if not numbers:
        return {"status": "no_common_poems", "rdm_stability": {"rho": None, "status": "no_common_poems"},
                "median_absolute_rdm_change": None, "q12_changed_pair_count": None,
                "paired_cosine_distance": [], "mean_paired_cosine_distance": None,
                "median_paired_cosine_distance": None, "max_paired_cosine_distance": None}
    reference = matrix(run, numbers, BASELINE, state, pooling)
    changed = matrix(run, numbers, condition, state, pooling)
    a, b = upper(cosine_rdm(reference)), upper(cosine_rdm(changed))
    products = np.einsum("ij,ij->i", reference, changed)
    paired = np.clip(1-products/(np.linalg.norm(reference, axis=1)*np.linalg.norm(changed, axis=1)), 0, 2)
    return {"status": "computed", "rdm_stability": q12_spearman(a, b),
            "median_absolute_rdm_change": float(np.median(np.abs(a-b))) if len(a) else None,
            "max_absolute_rdm_change": float(np.max(np.abs(a-b))) if len(a) else None,
            "q12_changed_pair_count": int(np.sum(np.round(a, 12) != np.round(b, 12))),
            "paired_cosine_distance": [{"sonnet": n, "cosine_distance": float(v)} for n, v in zip(numbers, paired)],
            "mean_paired_cosine_distance": float(paired.mean()),
            "median_paired_cosine_distance": float(np.median(paired)),
            "max_paired_cosine_distance": float(paired.max())}


def precision_gate(run, condition, state, pooling, planned):
    if condition not in CONTROLS:
        return {"status": "not_applicable", "reason": "Precision and native/padded comparisons are diagnostics, not text interventions."}
    numbers = [n for n in planned if all(n in available(run, c, state, pooling, planned) for c in [BASELINE, PRECISION, condition])]
    result = coverage(numbers, planned)
    result.update({"passes_on_tested_subset": None, "passes_on_all_planned_poems": False,
                   "minimum_precision_rho": .95, "maximum_precision_to_control_delta_ratio": .1})
    if len(numbers) < 3:
        result.update(status="insufficient_precision_coverage")
        return result
    precision = comparison(run, numbers, PRECISION, state, pooling)
    control = comparison(run, numbers, condition, state, pooling)
    result.update(precision_rdm_stability=precision["rdm_stability"],
                  precision_median_absolute_rdm_change=precision["median_absolute_rdm_change"],
                  control_median_absolute_rdm_change=control["median_absolute_rdm_change"],
                  control_rdm_stability=control["rdm_stability"],
                  control_q12_changed_pair_count=control["q12_changed_pair_count"])
    rho = precision["rdm_stability"]["rho"]
    if rho is None or control["rdm_stability"]["rho"] is None:
        result.update(status="degenerate_representation")
        return result
    if control["q12_changed_pair_count"] == 0:
        result.update(status="no_q12_resolved_control_change", precision_to_control_delta_ratio=None)
        return result
    delta = control["median_absolute_rdm_change"]
    if delta == 0:
        result.update(status="zero_median_control_change", precision_to_control_delta_ratio=None)
        return result
    ratio = precision["median_absolute_rdm_change"]/delta
    passed = bool(rho >= .95 and ratio <= .1)
    result.update(precision_to_control_delta_ratio=float(ratio), passes_on_tested_subset=passed,
                  passes_on_all_planned_poems=passed and result["complete_planned_cohort"],
                  status=("passed_full_cohort" if result["complete_planned_cohort"] else "passed_partial_cohort") if passed else "failed")
    return result


def analyze_runs(runs, protocol):
    planned, conditions = protocol["sonnets"], protocol["conditions"]
    if planned != sorted(set(planned)):
        raise ValueError("Expected unique ordered planned poem identities")
    within, between, rdms, models = [], [], [], {}
    for family in FAMILIES:
        if family not in runs:
            models[family] = {"status": "missing_model_report"}
            continue
        run = runs[family]
        complete_numbers = [n for n in planned if all(key(n, c, s, p) in run.vectors for c in conditions for s in STATES for p in POOLS)]
        models[family] = {"status": "loaded", "experiment_id": run.report.get("experiment_id"),
            "execution_complete": run.report["complete"], "fp32_status": run.report.get("fp32_status"),
            "stop_reason": run.report.get("stop_reason"), "source_files": run.receipts,
            "all_conditions_complete_coverage": coverage(complete_numbers, planned),
            "case_counts": {c: sum(x["condition"] == c for x in run.report["cases"]) for c in conditions}}
        for state in STATES:
            for pooling in POOLS:
                for condition in conditions:
                    present = [n for n in planned if n in available(run, condition, state, pooling, planned)]
                    rdms.append({"model": family, "state": state, "pooling": pooling, "condition": condition,
                        **coverage(present, planned), "cosine_distance_matrix": cosine_rdm(matrix(run, present, condition, state, pooling)).tolist() if present else None})
                    numbers = [n for n in present if n in available(run, BASELINE, state, pooling, planned)]
                    within.append({"model": family, "state": state, "pooling": pooling, "condition": condition,
                        "reference": BASELINE, **coverage(numbers, planned),
                        **comparison(run, numbers, condition, state, pooling),
                        "all_conditions_complete_comparison": {**coverage(complete_numbers, planned),
                            **comparison(run, complete_numbers, condition, state, pooling)},
                        "precision_gate": precision_gate(run, condition, state, pooling, planned)})
    for left, right in combinations(FAMILIES, 2):
        for state in STATES:
            for pooling in POOLS:
                for condition in conditions:
                    row = {"models": [left, right], "state": state, "pooling": pooling, "condition": condition}
                    if left not in runs or right not in runs:
                        row.update(status="missing_model_report", rho=None, **coverage([], planned))
                    else:
                        numbers = [n for n in planned if all(n in available(runs[f], condition, state, pooling, planned) for f in [left, right])]
                        row.update(coverage(numbers, planned))
                        if numbers:
                            vectors = [upper(cosine_rdm(matrix(runs[f], numbers, condition, state, pooling))) for f in [left, right]]
                            row.update(q12_spearman(*vectors))
                        else:
                            row.update(status="no_common_poems", rho=None)
                    between.append(row)
    return {"models": models, "within_model": within, "cross_model": between, "rdms": rdms,
        "metric_semantics": {"rdm": "Float64 cosine distances of saved pooled vectors; full raw distances retained.",
            "rsa": "Spearman rank correlation after rounding each upper-triangle distance to12decimal places; constant ranks are undefined.",
            "delta": "Absolute differences of unrounded float64 cosine distances, paired on identical poem IDs.",
            "coverage": "Per-comparison intersections are explicit; a second within-model table uses the identical all-condition complete cohort.",
            "gate": "Original BF16/FP32 precision and each intervention are compared on the same three-condition intersection. Passing a partial subset is never marked a full-cohort pass. An intervention unchanged at Q12 or with zero median change is not a robustness claim.",
            "no_significance": "No p-values or independence claim; postdiscovery sample and descriptive counterfactuals only."}}
