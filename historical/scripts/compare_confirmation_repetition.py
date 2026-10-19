"""Compare every saved numerical result and Muse vector across two distinct runs.

This is a deterministic CPU reconciliation, not another inference or statistical
replication. Old incomplete provenance remains incomplete even if all values agree.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations, product
import json
import math
from pathlib import Path
import re

import numpy as np
from safetensors import safe_open

from sonnet_geometry.storage import canonical_json, sha256_file, write_exclusive

FAMILIES = ("qwen", "gemma", "muse")
STATES = ("input", "early", "depth75", "final_raw", "final_normalized")
POOLS = ("mean", "last", "last_lexical")
CONTROLS = ("word_shuffle", "line_shuffle", "token_shuffle")
SEEDS = tuple(range(2026101801, 2026101806))
CLAIM_PROVENANCE = ("original_strict_confirmation_eligible", "depends_on_recovered_model",
                    "execution_provenance_qualification")
GRID_COUNTS = {"within_model": 1440, "cross_model": 1440, "arithmetic_comparisons": 720,
               "arithmetic_gates": 675, "seed_summaries": 540, "rdms": 1440,
               "claims": 6, "omission_rows": 720, "omission_summaries": 6}


def receipt(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return json.loads(Path(path).read_text(), object_pairs_hook=unique, parse_constant=nonfinite)


def same_receipt(left, right):
    return all(left[field] == right[field] for field in ("sha256", "bytes"))


def family_key(row):
    if "model" in row and "models" not in row:
        return row["model"]
    if "models" in row and "model" not in row:
        return "+".join(row["models"])
    raise ValueError("A numerical row must identify exactly one model or model pair")


def row_identity(grid, row):
    family = family_key(row)
    if grid in ("within_model", "cross_model"):
        return family, row["name"], row["state"], row["pooling"]
    if grid == "rdms":
        return family, row["condition"], row["state"], row["pooling"]
    if grid in ("arithmetic_comparisons", "arithmetic_gates"):
        return family, row["intervention"], row["seed"], row["state"], row["pooling"]
    if grid == "seed_summaries":
        return family, row["precision"], row["intervention"], row["state"], row["pooling"], row["metric"]
    if grid == "claims":
        return family, row["claim"]
    if grid == "omission_rows":
        return family, row["contrast"], row["omitted_sonnet"], row["seed"]
    if grid == "omission_summaries":
        return family, row["contrast"]
    raise ValueError(f"Unregistered numerical grid: {grid}")


def expected_identities(protocol):
    poems = protocol["sonnets"]
    if (len(poems) != 24 or poems != sorted(set(poems)) or protocol["models"] != list(FAMILIES)
            or set(protocol["states"]) != set(STATES) or protocol["poolings"] != list(POOLS)):
        raise ValueError("Unexpected frozen comparison population/states/poolings")
    conditions = [condition["name"] for condition in protocol["conditions"]]
    inputs = [("original", None), *product(CONTROLS, SEEDS)]
    expected_conditions = {f"{control}{'_seed' + str(seed) if seed is not None else ''}_padded_{precision}"
                           for control, seed in inputs for precision in ("bf16", "fp32")}
    if len(conditions) != 32 or set(conditions) != expected_conditions:
        raise ValueError("Incomplete or changed registered conditions")
    pairs = ["+".join(pair) for pair in combinations(FAMILIES, 2)]
    families = [*FAMILIES, *pairs]
    contrast = lambda family: ("early_minus_final_raw_stability" if "+" not in family
                               else "shuffled_minus_original_cross_model")
    claim = lambda family: ("early_stability_and_late_order_sensitivity" if "+" not in family
                            else "late_cross_model_agreement_increases")
    return {
        "within_model": set(product(FAMILIES, conditions, STATES, POOLS)),
        "cross_model": set(product(pairs, conditions, STATES, POOLS)),
        "rdms": set(product(FAMILIES, conditions, STATES, POOLS)),
        "arithmetic_comparisons": {(f, c, seed, s, p) for f, (c, seed), s, p
                                   in product(FAMILIES, inputs, STATES, POOLS)},
        "arithmetic_gates": set(product(FAMILIES, CONTROLS, SEEDS, STATES, POOLS)),
        "seed_summaries": {(f, precision, c, s, p, "cross_model_change" if "+" in f else "within_model_stability")
                           for f, precision, c, s, p in product(families, ("bf16", "fp32"), CONTROLS, STATES, POOLS)},
        "claims": {(f, claim(f)) for f in families},
        "omission_rows": {(f, contrast(f), n, seed) for f, n, seed in product(families, poems, SEEDS)},
        "omission_summaries": {(f, contrast(f)) for f in families},
    }


def indexed_grid(analysis, grid, expected):
    rows = (analysis["planned_omission_sensitivity"]["rows" if grid == "omission_rows" else "summaries"]
            if grid.startswith("omission_") else analysis[grid])
    if len(rows) != GRID_COUNTS[grid]:
        raise ValueError(f"Missing or extraneous rows in {grid}")
    indexed = {}
    for row in rows:
        identity = row_identity(grid, row)
        if identity in indexed:
            raise ValueError(f"Duplicate numerical row in {grid}")
        if grid == "rdms" and len(row["upper_triangle"]) != 276:
            raise ValueError("Each RDM must preserve every one of its 276 pairs")
        indexed[identity] = {k: v for k, v in row.items() if grid != "claims" or k not in CLAIM_PROVENANCE}
    if set(indexed) != expected:
        raise ValueError(f"Numerical identities differ from the full registered {grid}")
    return indexed


def compare_grid(left, right):
    fields, categorical, changed_rows = {}, [], []
    numerical_leaves = changed_numerical_leaves = 0
    for identity in sorted(left, key=lambda value: repr(value)):
        row_changed = 0
        def scan(a, b, path):
            nonlocal numerical_leaves, changed_numerical_leaves, row_changed
            if isinstance(a, dict) and isinstance(b, dict):
                if set(a) != set(b):
                    raise ValueError(f"Numerical object keys changed at {identity}: {path}")
                for key in sorted(a):
                    scan(a[key], b[key], f"{path}.{key}" if path else key)
                return
            if isinstance(a, list) and isinstance(b, list):
                if len(a) != len(b):
                    raise ValueError(f"Numerical array shape changed at {identity}: {path}")
                for index, (x, y) in enumerate(zip(a, b, strict=True)):
                    scan(x, y, f"{path}[{index}]")
                return
            numeric = type(a) in (int, float) and type(b) in (int, float)
            if numeric:
                if not math.isfinite(a) or not math.isfinite(b):
                    raise ValueError("Nonfinite numerical leaf")
                numerical_leaves += 1
                field = re.sub(r"\[\d+\]", "[*]", path)
                item = fields.setdefault(field, {"compared": 0, "changed": 0, "maximum_absolute_delta": 0.0,
                                                  "maximum_delta_location": None})
                item["compared"] += 1
                if a != b:
                    changed_numerical_leaves += 1
                    row_changed += 1
                    item["changed"] += 1
                    delta = abs(float(b) - float(a))
                    if delta > item["maximum_absolute_delta"]:
                        item.update(maximum_absolute_delta=delta,
                                    maximum_delta_location={"identity": list(identity), "field": path,
                                                            "old": a, "new": b})
                return
            if type(a) != type(b) or a != b:
                row_changed += 1
                categorical.append({"identity": list(identity), "field": path, "old": a, "new": b})
        scan(left[identity], right[identity], "")
        if row_changed:
            changed_rows.append({"identity": list(identity), "changed_leaves": row_changed})
    return {"rows": len(left), "exactly_equal_rows": len(left) - len(changed_rows),
            "changed_rows": changed_rows, "numerical_leaves_compared": numerical_leaves,
            "changed_numerical_leaves": changed_numerical_leaves, "numeric_fields": fields,
            "categorical_or_null_changes": categorical, "exactly_equal": not changed_rows}


def validate_analyses(old, new, protocol, protocol_receipt):
    if (old.get("schema_version") != "depth-confirmation-descriptive-recovery-v4"
            or old.get("status") != "strict_confirmation_incomplete_execution_provenance"
            or old.get("original_strict_confirmation_eligible") is not False
            or new.get("schema_version") != "depth-confirmation-analysis-v4"
            or new.get("status") not in {"undefined_primary_values", "all_frozen_primary_criteria_met",
                                         "frozen_primary_criteria_not_all_met"}):
        raise ValueError("Comparison requires preserved incomplete recovery and a distinct strict analysis")
    for analysis in (old, new):
        if (not same_receipt(analysis["protocol"], protocol_receipt)
                or analysis["poems"] != protocol["sonnets"] or analysis["poem_count"] != 24
                or analysis["rdm_pairs"] != 276 or analysis["criteria"] != protocol["analysis"]
                or analysis["planned_omission_sensitivity"]["planned_rows"] != 720
                or analysis["planned_omission_sensitivity"]["actual_rows"] != 720):
            raise ValueError("Analysis protocol, criteria or full coverage changed")
    if any(new["models"][family].get("execution_complete") is not True for family in FAMILIES):
        raise ValueError("Replacement analysis must use three completed worker reports")
    if old["models"]["muse"].get("execution_complete") is not False:
        raise ValueError("Original Muse provenance must remain incomplete")
    for field in set(old["metric_semantics"]) | set(new["metric_semantics"]):
        if field not in {"precision", "coverage"} and old["metric_semantics"].get(field) != new["metric_semantics"].get(field):
            raise ValueError("Scientific metric interpretation changed beyond execution qualification")
    for family in FAMILIES:
        a, b = old["models"][family]["source_files"], new["models"][family]["source_files"]
        if not same_receipt(a[2], b[2]):
            raise ValueError("Frozen model preflight changed between analyses")
        if family != "muse" and any(not same_receipt(x, y) for x, y in zip(a[:3], b[:3], strict=True)):
            raise ValueError("Qwen/Gemma source data changed during Muse replacement")
    for basename in ("confirmation_analysis.py", "text_control_analysis.py"):
        refs = [[item for item in analysis["source_files"] if Path(item["path"]).name == basename]
                for analysis in (old, new)]
        if any(len(items) != 1 for items in refs) or not same_receipt(refs[0][0], refs[1][0]):
            raise ValueError("Shared numerical analysis implementation changed")


def compare_vectors(old_path, new_path, protocol):
    expected = {f"{number:03d}.{condition['name']}.{state}.{pool}"
                for number, condition, state, pool in product(protocol["sonnets"], protocol["conditions"], STATES, POOLS)}
    width = protocol["model_locks"]["muse"]["hidden_size"]
    changed, numeric_equal, bitwise_equal, elements = [], 0, 0, 0
    max_delta = 0.0
    with safe_open(old_path, framework="numpy") as old, safe_open(new_path, framework="numpy") as new:
        if set(old.keys()) != expected or set(new.keys()) != expected or len(expected) != 11520:
            raise ValueError("Both Muse artifacts must retain all 11,520 exact vector identities")
        for name in sorted(expected):
            a, b = old.get_tensor(name), new.get_tensor(name)
            if (a.shape != (width,) or b.shape != (width,) or a.dtype != np.float64 or b.dtype != np.float64
                    or not np.isfinite(a).all() or not np.isfinite(b).all()):
                raise ValueError("Muse vector shape, F64 dtype or finite values changed")
            same = np.array_equal(a, b)
            same_bits = np.array_equal(a.view(np.uint8), b.view(np.uint8))
            numeric_equal += int(same)
            bitwise_equal += int(same_bits)
            elements += a.size
            if not same_bits:
                delta = np.abs(a - b)
                maximum = float(delta.max())
                max_delta = max(max_delta, maximum)
                changed.append({"vector": name, "numerically_equal": bool(same),
                                "changed_numeric_elements": int(np.count_nonzero(a != b)),
                                "changed_bitwise_elements": int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64))),
                                "maximum_absolute_delta": maximum, "mean_absolute_delta": float(delta.mean())})
    return {"vectors_compared": len(expected), "elements_compared": elements,
            "numerically_equal_vectors": numeric_equal, "bitwise_equal_vectors": bitwise_equal,
            "maximum_absolute_delta": max_delta, "changed_vectors": changed,
            "all_vectors_numerically_equal": numeric_equal == len(expected),
            "all_vectors_bitwise_equal": bitwise_equal == len(expected),
            "whole_file_bytes_equal": sha256_file(old_path) == sha256_file(new_path),
            "comparison_semantics": "Every F64 element is checked with exact equality, without a tolerance. Bitwise equality separately distinguishes signed zero. File/header equality is separate from tensor payload equality."}


def compare(old_path, new_path, old_pooled, new_pooled, protocol_path):
    old, new, protocol = map(read_json, (old_path, new_path, protocol_path))
    protocol_receipt = receipt(protocol_path)
    validate_analyses(old, new, protocol, protocol_receipt)
    for analysis, path in ((old, old_pooled), (new, new_pooled)):
        if not same_receipt(receipt(path), analysis["models"]["muse"]["source_files"][1]):
            raise ValueError("Muse pooled artifact bytes differ from its analysis source receipt")
    grids, claims = {}, []
    for grid, identities in expected_identities(protocol).items():
        a, b = (indexed_grid(analysis, grid, identities) for analysis in (old, new))
        grids[grid] = compare_grid(a, b)
        if grid == "claims":
            claims = [{"identity": list(identity), "old": a[identity], "new": b[identity]}
                      for identity in sorted(a, key=repr)]
    vectors = compare_vectors(old_pooled, new_pooled, protocol)
    exact = all(grid["exactly_equal"] for grid in grids.values())
    return {"schema_version": "confirmation-muse-repetition-comparison-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_files": [receipt(path) for path in (old_path, new_path, old_pooled, new_pooled, protocol_path, __file__)],
            "all_numerical_grids_exactly_equal": exact, "grids": grids, "muse_vectors": vectors,
            "claims": claims, "original_numerical_criteria_status": old["numerical_criteria_status"],
            "replacement_numerical_criteria_status": new["status"],
            "numerical_criteria_status_unchanged": old["numerical_criteria_status"] == new["status"],
            "provenance": {"original_schema": old["schema_version"], "original_status": old["status"],
                           "original_strict_confirmation_eligible": False,
                           "original_models": old["models"], "replacement_schema": new["schema_version"],
                           "replacement_models": new["models"],
                           "claim_fields_excluded_only_from_numerical_comparison": list(CLAIM_PROVENANCE)},
            "limitations": ["This replacement was requested after the original numerical outcomes were inspected.",
                            "Agreement does not retroactively restore the old run's missing runtime evidence.",
                            "Exact comparison is descriptive repetition, not a significance test or an independent population.",
                            "Full archive verification and an independent numerical audit remain separate requirements."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-analysis", type=Path, default=Path("outputs/confirmation-v4/descriptive-recovery-analysis.json"))
    parser.add_argument("--new-analysis", type=Path, required=True)
    parser.add_argument("--old-pooled", type=Path, default=Path("outputs/confirmation-v4/recovered-muse/pooled.safetensors"))
    parser.add_argument("--new-pooled", type=Path, default=Path("outputs/confirmation-v4/execution/muse/pooled.safetensors"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/confirmation-v4/protocol.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.old_analysis, args.new_analysis, args.old_pooled, args.new_pooled, args.protocol)
    write_exclusive(args.output, canonical_json(result))
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output),
                      "all_numerical_grids_exactly_equal": result["all_numerical_grids_exactly_equal"],
                      "all_muse_vectors_numerically_equal": result["muse_vectors"]["all_vectors_numerically_equal"],
                      "all_muse_vectors_bitwise_equal": result["muse_vectors"]["all_vectors_bitwise_equal"]}))


if __name__ == "__main__":
    main()
