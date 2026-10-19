"""Frozen v4 descriptive analysis; importing this module never performs inference."""
from itertools import combinations
import hashlib
import json
from pathlib import Path

import numpy as np

from sonnet_geometry.storage import canonical_json
from sonnet_geometry.text_control_analysis import (
    FAMILIES, POOLS, STATES, ModelRun, cosine_rdm, file_receipt, key, q12_spearman, upper,
)

INTERVENTIONS = ("word_shuffle", "line_shuffle", "token_shuffle")
PRECISIONS = ("bf16", "fp32")
SEEDS = tuple(range(2026101801, 2026101806))
STATE_BLOCKS = {
    "early": {"qwen": 1, "gemma": 4, "muse": 4},
    "depth75": {"qwen": 26, "gemma": 44, "muse": 38},
    "final_raw": {"qwen": 35, "gemma": 59, "muse": 51},
}
ANALYSIS_PLAN = {
    "schema_version": "confirmation-analysis-plan-v4",
    "expected_poems": 24,
    "expected_seeds": list(SEEDS),
    "primary_precision": "fp32",
    "primary_pooling": "mean",
    "primary_intervention": "token_shuffle",
    "minimum_median_early_stability": .80,
    "minimum_median_early_minus_final_raw": .30,
    "minimum_median_cross_model_increase": .20,
    "require_all_seed_directions_positive": True,
    "loo_require_all_directions_positive": True,
    "arithmetic_minimum_rho": .95,
    "arithmetic_maximum_delta_ratio": .10,
}


def condition_name(intervention, seed, precision):
    stem = "original" if intervention == "original" else f"{intervention}_seed{seed}"
    return f"{stem}_padded_{precision}"


def condition_specs():
    return [{"name": condition_name(control, seed, precision), "intervention": control,
             "seed": seed, "precision": precision}
            for control, seed in [("original", None),
                                  *((c, s) for c in INTERVENTIONS for s in SEEDS)]
            for precision in PRECISIONS]


def validate_protocol(protocol):
    if protocol.get("schema_version") != "frozen-depth-confirmation-v4":
        raise ValueError("Expected frozen-depth-confirmation-v4 protocol")
    numbers = protocol.get("sonnets", [])
    if (len(numbers) != 24 or numbers != sorted(set(numbers))
            or any(type(n) is not int or n <= 0 for n in numbers)):
        raise ValueError("Exactly 24 unique, ordered positive poem identities are required")
    if protocol.get("models") != list(FAMILIES):
        raise ValueError("All three registered model families are required")
    if set(protocol.get("states", {})) != set(STATES) or protocol.get("poolings") != list(POOLS):
        raise ValueError("All five registered states and three poolings are required")
    if any(protocol["states"][state] != blocks for state, blocks in STATE_BLOCKS.items()):
        raise ValueError("Registered state blocks differ from the frozen v3 depth locations")
    specs = protocol.get("conditions", [])
    expected = {x["name"]: x for x in condition_specs()}
    if len(specs) != 32 or {x["name"]: x for x in specs} != expected:
        raise ValueError("Exactly the 32 registered precision/seed conditions are required")
    if protocol.get("analysis") != ANALYSIS_PLAN:
        raise ValueError("Analysis plan differs from the frozen v4 criteria")
    if type(protocol.get("pad_length")) is not int or protocol["pad_length"] <= 0:
        raise ValueError("A positive fixed padding length is required")
    return expected


def forward_hash(case):
    ids = case["tokenization"]["input_ids"]
    padded = ids + [case["pad_token_id"]] * (case["forward_length"] - len(ids))
    value = {"input_ids": padded, "attention_mask": case["attention_mask"],
             "position_ids": case["position_ids"]}
    return hashlib.sha256(canonical_json(value)).hexdigest()


def validated_cases(cases, protocol):
    """Require complete input coverage and compare all precision pairs exactly."""
    specs = validate_protocol(protocol)
    expected = {(n, c) for n in protocol["sonnets"] for c in specs}
    indexed = {}
    for case in cases:
        identity = case["sonnet"], case["condition"]
        if identity not in expected:
            raise ValueError("Unregistered execution poem or condition")
        if identity in indexed:
            raise ValueError("Duplicate execution case")
        ids = case["tokenization"]["input_ids"]
        length = case["forward_length"]
        if (length != protocol["pad_length"] or not ids or len(ids) > length
                or any(type(x) is not int or x < 0 for x in ids)
                or type(case["pad_token_id"]) is not int or case["pad_token_id"] < 0):
            raise ValueError("Invalid token IDs or frozen forward length")
        if case["attention_mask"] != [1] * len(ids) + [0] * (length - len(ids)):
            raise ValueError("Invalid registered right-padding attention mask")
        if case["position_ids"] != list(range(length)):
            raise ValueError("Invalid registered position IDs")
        if case["forward_sha256"] != forward_hash(case):
            raise ValueError("Forward input checksum mismatch")
        indexed[identity] = case
    if set(indexed) != expected:
        raise ValueError("Incomplete execution coverage: all 24 poems and 32 conditions are required")
    for number in protocol["sonnets"]:
        for spec in specs.values():
            if spec["precision"] != "bf16":
                continue
            a = indexed[number, spec["name"]]
            b = indexed[number, condition_name(spec["intervention"], spec["seed"], "fp32")]
            fields = ("tokenization", "forward_length", "pad_token_id", "attention_mask",
                      "position_ids", "forward_sha256")
            if any(a[field] != b[field] for field in fields):
                raise ValueError("BF16/FP32 precision pair has different inputs or tokenization")
    return indexed


def validate_vectors(run, protocol):
    specs = validate_protocol(protocol)
    expected = {key(n, c, s, p) for n in protocol["sonnets"] for c in specs
                for s in STATES for p in POOLS}
    if set(run.vectors) != expected:
        raise ValueError("Missing or orphan pooled vectors: full registered coverage is required")
    width = protocol["model_locks"][run.family]["hidden_size"]
    for vector in run.vectors.values():
        if (vector.ndim != 1 or len(vector) != width or not np.isfinite(vector).all()
                or np.linalg.norm(vector.astype(np.float64)) == 0):
            raise ValueError("Invalid pooled vector width, finite values, or zero norm")


def validate_execution_precision(report, specs):
    expected_settings = {"fp32_checkpoint_source": "bf16_promoted", "tf32": False, "autocast": False}
    if report.get("precision_settings") != expected_settings:
        raise ValueError("Missing or inconsistent actual precision settings")
    bf16, fp32 = report.get("environment_bf16", {}), report.get("environment_fp32", {})
    bf16_types = set(bf16.get("parameter_dtypes", []))
    if not bf16_types or "torch.bfloat16" not in bf16_types or not bf16_types <= {"torch.bfloat16", "torch.float32"}:
        raise ValueError("BF16 model environment has inconsistent parameter dtypes")
    if fp32.get("parameter_dtypes") != ["torch.float32"]:
        raise ValueError("FP32 model environment does not demonstrate float32 parameters")
    if any(env.get("tf32_matmul_allowed") is not False for env in (bf16, fp32)):
        raise ValueError("TF32 must be disabled in both recorded model environments")
    validate_case_precision(report["cases"], report["model"], specs)


def validate_case_precision(cases, family, specs):
    """Validate surviving case evidence without asserting model-wide parameter dtypes."""
    for case in cases:
        expected_dtype = "torch.float32" if specs[case["condition"]]["precision"] == "fp32" else "torch.bfloat16"
        if case.get("dtypes") != {state: expected_dtype for state in STATES}:
            raise ValueError("Actual captured state dtypes differ from the registered arithmetic condition")
        if case.get("blocks") != {state: values[family] for state, values in STATE_BLOCKS.items()}:
            raise ValueError("Actual captured state blocks differ from frozen depth locations")
        if "source_dtype" in case and case["source_dtype"] != expected_dtype:
            raise ValueError("Actual source dtype differs from the registered arithmetic condition")
        for field in ("tf32_matmul_allowed", "tf32_cudnn_allowed", "autocast_enabled"):
            if case.get(field) is not False:
                raise ValueError("Actual execution must disable TF32 and autocast for every case")


def load_run(report_path, pooled_path, preflight_path, family, protocol, protocol_sha256):
    """Bind every result to frozen inputs, checkpoint, byte receipts and complete cases."""
    from safetensors.numpy import load_file

    specs = validate_protocol(protocol)
    wrapper = json.loads(Path(report_path).read_text())
    report = wrapper.get("report", wrapper)
    preflight = json.loads(Path(preflight_path).read_text())
    if (report.get("schema_version") != "confirmation-execution-v4"
            or report.get("model") != family or report.get("complete") is not True):
        raise ValueError("Execution identity or completion flag is invalid")
    lock = protocol["model_locks"][family]
    for item in (report, preflight):
        if (item.get("model") != family or item.get("model_id") != lock["model_id"]
                or item.get("revision") != lock["revision"]
                or item.get("hidden_size") != lock["hidden_size"]):
            raise ValueError("Checkpoint identity differs from the frozen model lock")
    preflight_receipt = file_receipt(preflight_path)
    if (report["config"]["protocol_sha256"] != protocol_sha256
            or preflight["protocol_sha256"] != protocol_sha256):
        raise ValueError("Frozen protocol hash differs from the executed protocol")
    if report["config"]["preflight_sha256"] != preflight_receipt["sha256"]:
        raise ValueError("Frozen preflight checksum differs from the execution config")
    tokenizer_hash = preflight.get("tokenizer_sha256")
    if (not isinstance(tokenizer_hash, str) or len(tokenizer_hash) != 64
            or any(c not in "0123456789abcdef" for c in tokenizer_hash)
            or report.get("tokenizer", {}).get("settings", {}).get("backend_sha256") != tokenizer_hash):
        raise ValueError("Actual tokenizer backend differs from frozen preflight")
    actual = validated_cases(report["cases"], protocol)
    planned = validated_cases(preflight["cases"], protocol)
    validate_execution_precision(report, specs)
    fields = ("tokenization", "forward_length", "pad_token_id", "attention_mask",
              "position_ids", "forward_sha256")
    for identity in actual:
        if any(actual[identity][field] != planned[identity][field] for field in fields):
            raise ValueError("Execution inputs differ from the frozen preflight")
    pooled_receipt = file_receipt(pooled_path)
    if any(pooled_receipt[k] != report["pooled"][k] for k in ("sha256", "bytes")):
        raise ValueError("Pooled artifact checksum or size mismatch")
    vectors = load_file(str(pooled_path))
    run = ModelRun(family, report, vectors,
                   [file_receipt(report_path), pooled_receipt, preflight_receipt])
    validate_vectors(run, protocol)
    return run


def finite_summary(values):
    present = [float(v) for v in values if v is not None and np.isfinite(v)]
    return {"expected_count": len(values), "computed_count": len(present),
            "status": "complete" if len(present) == len(values) else "undefined_values_retained",
            "minimum": min(present) if present else None,
            "median": float(np.median(present)) if present else None,
            "maximum": max(present) if present else None,
            "all_strictly_positive": all(v > 0 for v in present)
            if len(present) == len(values) and present else None}


def paired_rdm(left, right):
    a, b = upper(left), upper(right)
    change = np.abs(a - b)
    return {"rdm_stability": q12_spearman(a, b),
            "median_absolute_rdm_change": float(np.median(change)),
            "maximum_absolute_rdm_change": float(np.max(change)),
            "q12_changed_pair_count": int(np.sum(np.round(a, 12) != np.round(b, 12)))}


def rho(left, right):
    return q12_spearman(upper(left), upper(right))


def difference(left, right):
    return float(left - right) if left is not None and right is not None else None


def arithmetic_gate(original, perturbed, fp32_effect):
    """Both sides of each intervention must pass; original-only checks are insufficient."""
    rhos = [item["rdm_stability"]["rho"] for item in (original, perturbed)]
    numerator = max(item["median_absolute_rdm_change"] for item in (original, perturbed))
    denominator = fp32_effect["median_absolute_rdm_change"]
    result = {"original_precision_rho": rhos[0], "perturbed_precision_rho": rhos[1],
              "maximum_precision_median_rdm_delta": numerator,
              "fp32_intervention_median_rdm_delta": denominator,
              "precision_to_fp32_intervention_delta_ratio": None, "passes": None}
    if any(v is None for v in rhos) or fp32_effect["rdm_stability"]["rho"] is None:
        return {**result, "status": "undefined_rdm_correlation"}
    if not fp32_effect["q12_changed_pair_count"] or denominator == 0:
        return {**result, "status": "no_resolved_fp32_intervention_delta"}
    ratio = numerator / denominator
    passed = (min(rhos) >= ANALYSIS_PLAN["arithmetic_minimum_rho"]
              and ratio <= ANALYSIS_PLAN["arithmetic_maximum_delta_ratio"])
    return {**result, "precision_to_fp32_intervention_delta_ratio": float(ratio),
            "passes": bool(passed), "status": "passed" if passed else "failed"}


def depth_criterion(early_values, final_values):
    if len(early_values) != len(SEEDS) or len(final_values) != len(SEEDS):
        raise ValueError("Every one of the five registered seed values is required")
    deltas = [difference(a, b) for a, b in zip(early_values, final_values, strict=True)]
    early, late, delta = map(finite_summary, (early_values, final_values, deltas))
    complete = all(x["status"] == "complete" for x in (early, late, delta))
    passed = (early["median"] >= ANALYSIS_PLAN["minimum_median_early_stability"]
              and delta["median"] >= ANALYSIS_PLAN["minimum_median_early_minus_final_raw"]
              and delta["all_strictly_positive"]) if complete else None
    return {"early_stability": early, "final_raw_stability": late,
            "early_minus_final_raw": delta, "passes": passed,
            "status": "undefined" if passed is None else "criterion_met" if passed else "criterion_not_met"}


def cross_criterion(values):
    if len(values) != len(SEEDS):
        raise ValueError("Every one of the five registered seed values is required")
    summary = finite_summary(values)
    passed = (summary["median"] >= ANALYSIS_PLAN["minimum_median_cross_model_increase"]
              and summary["all_strictly_positive"]) if summary["status"] == "complete" else None
    return {"shuffled_minus_original": summary, "passes": passed,
            "status": "undefined" if passed is None else "criterion_met" if passed else "criterion_not_met"}


def planned_omissions(rdms, numbers):
    """Retain every omission for the six registered FP32 mean/token contrasts."""
    rows = []
    original = condition_name("original", None, "fp32")
    for omitted in numbers:
        kept = [i for i, n in enumerate(numbers) if n != omitted]

        def sub(family, condition, state):
            return rdms[family, condition, state, "mean"][np.ix_(kept, kept)]

        for seed in SEEDS:
            condition = condition_name("token_shuffle", seed, "fp32")
            for family in FAMILIES:
                early = rho(sub(family, original, "early"), sub(family, condition, "early"))
                late = rho(sub(family, original, "final_raw"), sub(family, condition, "final_raw"))
                rows.append({"contrast": "early_minus_final_raw_stability", "model": family,
                             "omitted_sonnet": omitted, "remaining_poems": len(kept), "seed": seed,
                             "early": early, "final_raw": late,
                             "delta": difference(early["rho"], late["rho"])})
            for left, right in combinations(FAMILIES, 2):
                baseline = rho(sub(left, original, "final_raw"), sub(right, original, "final_raw"))
                shuffled = rho(sub(left, condition, "final_raw"), sub(right, condition, "final_raw"))
                rows.append({"contrast": "shuffled_minus_original_cross_model", "models": [left, right],
                             "omitted_sonnet": omitted, "remaining_poems": len(kept), "seed": seed,
                             "original": baseline, "shuffled": shuffled,
                             "delta": difference(shuffled["rho"], baseline["rho"])})
    summaries = []
    for family in FAMILIES:
        values = [x["delta"] for x in rows if x.get("model") == family]
        summaries.append({"contrast": "early_minus_final_raw_stability", "model": family,
                          **finite_summary(values)})
    for pair in combinations(FAMILIES, 2):
        values = [x["delta"] for x in rows if x.get("models") == list(pair)]
        summaries.append({"contrast": "shuffled_minus_original_cross_model", "models": list(pair),
                          **finite_summary(values)})
    return {"planned_rows": 720, "actual_rows": len(rows), "rows": rows, "summaries": summaries,
            "interpretation": "Planned sensitivity only. Ranges are not confidence intervals. Omissions and seeds are not independent populations."}


def analyze_runs(runs, protocol):
    specs = validate_protocol(protocol)
    if set(runs) != set(FAMILIES):
        raise ValueError("All three complete model runs are required; no partial confirmation output")
    for family in FAMILIES:
        run = runs[family]
        if run.family != family:
            raise ValueError("Run family identity mismatch")
        if run.report.get("complete") is not True:
            raise ValueError("Every run must explicitly report complete execution")
        validated_cases(run.report["cases"], protocol)
        validate_execution_precision(run.report, specs)
        validate_vectors(run, protocol)
    result = _calculate_runs(runs, protocol)
    for model in result["models"].values():
        model["execution_complete"] = True
    return result


def _calculate_runs(runs, protocol):
    """Pure frozen geometry, called only after the entry point's evidence validation."""
    specs = validate_protocol(protocol)
    numbers = protocol["sonnets"]
    rdms, saved_rdms = {}, []
    for family in FAMILIES:
        run = runs[family]
        for condition in specs:
            for state in STATES:
                for pool in POOLS:
                    vectors = np.stack([run.vectors[key(n, condition, state, pool)] for n in numbers])
                    value = cosine_rdm(vectors)
                    rdms[family, condition, state, pool] = value
                    saved_rdms.append({"model": family, "condition": condition, "state": state,
                                       "pooling": pool, "upper_triangle": upper(value).tolist()})
    within, between, arithmetic, gates = [], [], [], []
    within_index, cross_index, arithmetic_index = {}, {}, {}
    for family in FAMILIES:
        for condition, spec in specs.items():
            baseline = condition_name("original", None, spec["precision"])
            for state in STATES:
                for pool in POOLS:
                    comparison = paired_rdm(rdms[family, baseline, state, pool], rdms[family, condition, state, pool])
                    row = {"model": family, **spec, "state": state, "pooling": pool,
                           "reference": baseline, **comparison}
                    within.append(row)
                    within_index[family, condition, state, pool] = comparison
                    if spec["precision"] == "bf16":
                        promoted = condition_name(spec["intervention"], spec["seed"], "fp32")
                        numeric = paired_rdm(rdms[family, condition, state, pool], rdms[family, promoted, state, pool])
                        arithmetic.append({"model": family, "intervention": spec["intervention"],
                                           "seed": spec["seed"], "state": state, "pooling": pool,
                                           "bf16_condition": condition, "fp32_condition": promoted, **numeric})
                        arithmetic_index[family, spec["intervention"], spec["seed"], state, pool] = numeric
    for left, right in combinations(FAMILIES, 2):
        for condition, spec in specs.items():
            baseline = condition_name("original", None, spec["precision"])
            for state in STATES:
                for pool in POOLS:
                    score = rho(rdms[left, condition, state, pool], rdms[right, condition, state, pool])
                    reference = rho(rdms[left, baseline, state, pool], rdms[right, baseline, state, pool])
                    row = {"models": [left, right], **spec, "state": state, "pooling": pool,
                           **score, "original_rho": reference["rho"], "original_status": reference["status"],
                           "delta_rho_condition_minus_original": difference(score["rho"], reference["rho"])}
                    between.append(row)
                    cross_index[left, right, condition, state, pool] = row
    for family in FAMILIES:
        for control in INTERVENTIONS:
            for seed in SEEDS:
                for state in STATES:
                    for pool in POOLS:
                        bf16 = condition_name(control, seed, "bf16")
                        fp32 = condition_name(control, seed, "fp32")
                        effect = within_index[family, fp32, state, pool]
                        gate = arithmetic_gate(arithmetic_index[family, "original", None, state, pool],
                                               arithmetic_index[family, control, seed, state, pool], effect)
                        gates.append({"model": family, "intervention": control, "seed": seed,
                                      "state": state, "pooling": pool, **gate,
                                      "fp32_minus_bf16_stability": difference(
                                          effect["rdm_stability"]["rho"],
                                          within_index[family, bf16, state, pool]["rdm_stability"]["rho"])})
    seed_summaries = []
    for precision in PRECISIONS:
        for control in INTERVENTIONS:
            for state in STATES:
                for pool in POOLS:
                    conditions = [condition_name(control, seed, precision) for seed in SEEDS]
                    fields = {"precision": precision, "intervention": control, "state": state, "pooling": pool,
                              "seeds": list(SEEDS)}
                    for family in FAMILIES:
                        values = [within_index[family, c, state, pool]["rdm_stability"]["rho"] for c in conditions]
                        seed_summaries.append({**fields, "model": family, "metric": "within_model_stability",
                                               **finite_summary(values)})
                    for left, right in combinations(FAMILIES, 2):
                        values = [cross_index[left, right, c, state, pool]["delta_rho_condition_minus_original"]
                                  for c in conditions]
                        seed_summaries.append({**fields, "models": [left, right], "metric": "cross_model_change",
                                               **finite_summary(values)})
    claims = []
    for family in FAMILIES:
        values = [[within_index[family, condition_name("token_shuffle", seed, "fp32"), state, "mean"]
                   ["rdm_stability"]["rho"] for seed in SEEDS] for state in ("early", "final_raw")]
        claims.append({"claim": "early_stability_and_late_order_sensitivity", "model": family,
                       "role": "primary" if family in ("qwen", "muse") else "prespecified_secondary",
                       "seeds": list(SEEDS), **depth_criterion(*values)})
    for left, right in combinations(FAMILIES, 2):
        values = [cross_index[left, right, condition_name("token_shuffle", seed, "fp32"), "final_raw", "mean"]
                  ["delta_rho_condition_minus_original"] for seed in SEEDS]
        claims.append({"claim": "late_cross_model_agreement_increases", "models": [left, right],
                       "role": "primary" if (left, right) == ("qwen", "muse") else "prespecified_secondary",
                       "seeds": list(SEEDS), **cross_criterion(values)})
    omissions = planned_omissions(rdms, numbers)
    for claim in claims:
        summary = next(x for x in omissions["summaries"]
                       if (("model" in claim and x.get("model") == claim["model"])
                           or ("models" in claim and x.get("models") == claim["models"])))
        claim["omission_direction_sensitivity"] = summary
        if claim["passes"] is None:
            qualified = "undefined_primary_or_secondary_criterion"
        elif not claim["passes"]:
            qualified = "criterion_not_met"
        elif summary["all_strictly_positive"] is None:
            qualified = "criterion_met_omission_sensitivity_undefined"
        elif summary["all_strictly_positive"]:
            qualified = "criterion_met_and_all_omission_directions_positive"
        else:
            qualified = "criterion_met_but_omission_direction_changes"
        claim["sensitivity_qualified_status"] = qualified
    primary = [x["passes"] for x in claims if x["role"] == "primary"]
    outcome = ("undefined_primary_values" if any(v is None for v in primary)
               else "all_frozen_primary_criteria_met" if all(primary) else "frozen_primary_criteria_not_all_met")
    return {"schema_version": "depth-confirmation-analysis-v4", "status": outcome,
            "poems": numbers, "poem_count": len(numbers), "rdm_pairs": len(numbers) * (len(numbers)-1)//2,
            "models": {f: {"source_files": runs[f].receipts} for f in FAMILIES},
            "criteria": ANALYSIS_PLAN.copy(), "claims": claims, "within_model": within,
            "cross_model": between, "arithmetic_comparisons": arithmetic, "arithmetic_gates": gates,
            "seed_summaries": seed_summaries, "rdms": saved_rdms,
            "planned_omission_sensitivity": omissions,
            "metric_semantics": {
                "rdm": "Float64 cosine distance of saved pooled vectors; upper triangles follow ordered poem IDs, row-major i<j.",
                "rsa": "Spearman after Q12 rounding distances, average ranks for ties; constants remain null with explicit status.",
                "precision": "Original and every shuffled input run in both precisions with identical inputs. FP32 promotes the same checkpoint values, not higher-precision source weights.",
                "arithmetic_gate": "Require original AND perturbed BF16/FP32 rho>=.95 and maximum of their median absolute RDM differences <=.10 times the FP32 intervention difference. Undefined/zero differences do not pass.",
                "primary": "Three pragmatic effect-size criteria on FP32 mean token-shuffle RDMs; all five seed directions required. Not a significance test or semantic validation.",
                "inference": "Seeds repeat interventions on the same 24 poems; distance pairs and omissions are dependent. No p-values, confidence intervals, population-level power, or literary-usefulness claim.",
                "coverage": "Every registered model, poem, condition, state and pooling is required. Missing execution prevents scientific output; undefined correlations remain explicit."}}
