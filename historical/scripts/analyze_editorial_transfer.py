"""Evaluate frozen Folger OOF scores against frozen historical editorial labels.

No model is fitted, no historical omission is labeled semantically unrelated,
and no representation or regularization parameter is chosen here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


def digest(path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def checked_source(receipt):
    current = digest(receipt["path"])
    if current["sha256"] != receipt["sha256"]:
        raise ValueError(f"Frozen source changed: {receipt['path']}")
    return json.loads(Path(receipt["path"]).read_text())


def same_band_ranks(probability, bands, candidate_mask):
    """Descending score midranks within each pre-existing test band."""
    ranks = np.full(len(probability), np.nan)
    for band in sorted(set(bands)):
        rows = np.flatnonzero((bands == band) & candidate_mask)
        if len(rows):
            ranks[rows] = (rankdata(-probability[rows], method="average") - .5) / len(rows)
    return ranks


def positive_metrics(index, probability, ranks):
    index = np.asarray(index, dtype=int)
    if not len(index):
        return {"count": 0, "mean_rank_percentile": None, "median_rank_percentile": None,
                "top_20_percent_recovery": None, "mean_folger_link_probability": None,
                "positive_only_negative_log_score": None}
    p, r = probability[index], ranks[index]
    if not np.isfinite(r).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError("Invalid frozen score or rank")
    return {"count": int(len(index)), "mean_rank_percentile": float(r.mean()),
            "median_rank_percentile": float(np.median(r)),
            "top_20_percent_recovery": float(np.mean(r <= .2)),
            "mean_folger_link_probability": float(p.mean()),
            "positive_only_negative_log_score": float(-np.log(p).mean())}


def validate_old_splits(extended, single, folger):
    if extended["rows"] != single["rows"]:
        raise ValueError("Frozen row identities differ")
    components = {n: g["component_id"] for g in folger["fold_components"] for n in g["sonnet_numbers"]}
    components[153] = components[154] = "positive-control-153-154"
    rows = extended["rows"]
    coverage = []
    audits = []
    for fold, single_fold in zip(extended["folds"], single["folds"], strict=True):
        for key in ["band", "train_rows", "test_rows", "purged_rows", "inner_splits"]:
            if fold[key] != single_fold[key]:
                raise ValueError(f"Frozen split mismatch: {key}")
        coverage.extend(fold["test_rows"])
        for name, train, test in [("outer", fold["train_rows"], fold["test_rows"])] + [
            (f"inner-{k}", s["train_rows"], s["validation_rows"])
            for k, s in enumerate(fold["inner_splits"])
        ]:
            training_components = {components[n] for i in train for n in rows[i]["sonnets"]}
            test_components = {components[n] for i in test for n in rows[i]["sonnets"]}
            if training_components & test_components:
                raise ValueError("Saved split violates Folger component purge")
            audits.append({"band": fold["band"], "split": name,
                           "train_rows": len(train), "test_rows": len(test),
                           "shared_folger_components": 0})
    if sorted(coverage) != list(range(len(rows))):
        raise ValueError("Incomplete or duplicate frozen OOF coverage")
    for name in ["final_plus_fixed_depths", "plus_both_final"]:
        if extended["predictions"][name] != single["comparison_predictions"][name]:
            raise ValueError("Frozen comparator scores differ between saved runs")
    return audits


def source_evaluation(source, rows, scores, include_control):
    pairs = [tuple(r["sonnets"]) for r in rows]
    bands = np.array([r["band"] for r in rows])
    folger_y = np.array([bool(r["editorially_linked"]) for r in rows])
    edge_units = {tuple(e["sonnets"]): e["unit_ids"] for e in source["positive_edges"]}
    units = {u["unit_id"]: u for u in source["units"]}
    control = source.get("positive_control")
    if include_control and control:
        control_unit = f"{source['source_id']}-positive-control"
        edge_units[tuple(control["sonnets"])] = [control_unit]
        units[control_unit] = {"unit_id": control_unit, "sonnets": control["sonnets"],
                               "kind": "separate_positive_control"}
    hist = np.array([p in edge_units for p in pairs])
    # Follow the literal frozen lists. The Beeching list leaves78–86 in;
    # do not silently expand it into an all-groups-of-size>=8 rule.
    broad_units = {
        "beeching1904_reported_lee": {"beeching1904-001-017", "beeching1904-040-062"},
        "reed1923": {"reed1923-001-017", "reed1923-018-025", "reed1923-078-086"},
    }[source["source_id"]]
    narrow = np.array([p in edge_units and any(u not in broad_units for u in edge_units[p]) for p in pairs])
    candidate_masks = {"unmentioned": ~folger_y, "all": np.ones(len(rows), dtype=bool)}
    rank_arrays = {name: {key: same_band_ranks(p, bands, mask) for key, mask in candidate_masks.items()}
                   for name, p in scores.items()}
    policies = {
        "new_historical_links": (hist & ~folger_y, "unmentioned"),
        "new_historical_links_without_broad_groups": (hist & ~folger_y & narrow, "unmentioned"),
        "all_historical_links": (hist, "all"),
        "historical_and_folger_links": (hist & folger_y, "all"),
        "all_historical_links_without_broad_groups": (hist & narrow, "all"),
    }
    populations = {}
    for key, (mask, rank_key) in policies.items():
        index = np.flatnonzero(mask)
        per_model, per_unit = {}, []
        for unit_id, unit in units.items():
            unit_rows = [int(i) for i in index if unit_id in edge_units[pairs[i]]]
            if not unit_rows:
                continue
            per_unit.append({"unit_id": unit_id, "sonnets": unit["sonnets"], "rows": unit_rows,
                "metrics": {name: positive_metrics(unit_rows, p, rank_arrays[name][rank_key])
                            for name, p in scores.items()}})
        for name, p in scores.items():
            result = positive_metrics(index, p, rank_arrays[name][rank_key])
            result["source_unit_macro_mean_rank_percentile"] = (
                float(np.mean([u["metrics"][name]["mean_rank_percentile"] for u in per_unit]))
                if per_unit else None)
            result["source_units_with_observations"] = len(per_unit)
            per_model[name] = result
        comparisons = []
        for name in scores:
            if name == "single_depth":
                continue
            comparisons.append({"model": name, "reference": "single_depth",
                "delta_mean_rank_percentile": (per_model[name]["mean_rank_percentile"] - per_model["single_depth"]["mean_rank_percentile"] if len(index) else None),
                "delta_source_unit_macro_mean_rank_percentile": (
                    per_model[name]["source_unit_macro_mean_rank_percentile"] - per_model["single_depth"]["source_unit_macro_mean_rank_percentile"] if per_unit else None),
                "source_units_improved_mean_rank": sum(u["metrics"][name]["mean_rank_percentile"] < u["metrics"]["single_depth"]["mean_rank_percentile"] for u in per_unit),
                "source_units_worsened_mean_rank": sum(u["metrics"][name]["mean_rank_percentile"] > u["metrics"]["single_depth"]["mean_rank_percentile"] for u in per_unit),
                "edge_rank_improved_count": int(np.sum(rank_arrays[name][rank_key][index] < rank_arrays["single_depth"][rank_key][index])),
                "edge_rank_worsened_count": int(np.sum(rank_arrays[name][rank_key][index] > rank_arrays["single_depth"][rank_key][index]))})
        populations[key] = {"rows": index.tolist(), "count": len(index),
            "bands_with_observations": sorted(set(int(b) for b in bands[index])),
            "candidate_policy": rank_key, "metrics": per_model, "comparisons": comparisons,
            "source_unit_results": per_unit}
    membership = {name: {"auroc": float(roc_auc_score(hist, p)),
                         "average_precision": float(average_precision_score(hist, p))}
                  for name, p in scores.items()} if len(set(hist)) == 2 else None
    availability = []
    pair_set = set(pairs)
    for pair in sorted(edge_units):
        reason = "available" if pair in pair_set else (
            "not_an_adjacent_pair" if pair[1] - pair[0] != 1 else
            "excluded_154" if pair[1] == 154 else
            "crosses_fixed_band_boundary" if (pair[0]-1)//22 != (pair[1]-1)//22 else
            "excluded_folger_ambiguity")
        availability.append({"sonnets": list(pair), "unit_ids": edge_units[pair], "status": reason})
    historical_cross_fold_units = []
    for u in source["units"]:
        unit_bands = sorted({(n-1)//22 for n in u["sonnets"]})
        if len(unit_bands) > 1:
            historical_cross_fold_units.append({"unit_id": u["unit_id"], "bands": unit_bands})
    return {"source_id": source["source_id"], "include_positive_control": include_control,
            "available_historical_edges": int(hist.sum()), "predeclared_excluded_units": sorted(broad_units),
            "positive_unmentioned_overlap": {"historical_and_folger": int(np.sum(hist & folger_y)),
                "historical_only": int(np.sum(hist & ~folger_y)),
                "folger_only": int(np.sum(~hist & folger_y)), "neither_explicitly_labeled": int(np.sum(~hist & ~folger_y))},
            "populations": populations, "finite_source_ledger_recovery_only": membership,
            "availability": availability, "historical_units_crossing_old_folds": historical_cross_fold_units,
            "rows": [{**r, "historical_positive": bool(hist[i]),
                "historical_unit_ids": edge_units.get(pairs[i], []),
                "fixed_probability": {name: float(p[i]) for name, p in scores.items()},
                "all_same_band_rank_percentiles": {name: float(rank_arrays[name]["all"][i]) for name in scores},
                "unmentioned_same_band_rank_percentiles": {name: float(rank_arrays[name]["unmentioned"][i]) if not folger_y[i] else None for name in scores}}
                for i, r in enumerate(rows)]}


def self_checks():
    p = np.array([.9, .9, .1, .8, .2])
    b = np.array([0, 0, 0, 1, 1])
    r = same_band_ranks(p, b, np.array([True, True, False, True, True]))
    np.testing.assert_allclose(r[[0, 1, 3, 4]], [.5, .5, .25, .75])
    assert np.isnan(r[2])
    assert positive_metrics([], p, r)["mean_rank_percentile"] is None
    assert positive_metrics([3], p, r)["top_20_percent_recovery"] == 0
    return {"within_band_ties": "passed", "excluded_candidates_remain_missing": "passed",
            "empty_population_not_fabricated": "passed", "fixed_threshold": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/followup-v3/editorial-transfer.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    start = time.monotonic()
    plan_path = Path("experiments/followup-v3/editorial-transfer-plan.json")
    plan = json.loads(plan_path.read_text())
    labels = checked_source(plan["labels"])
    saved = [checked_source(r) for r in plan["fixed_prediction_files"]]
    historical_receipt = checked_source(labels["source_receipt"])
    for receipt in historical_receipt["sources"]:
        if digest(receipt["path"])["sha256"] != receipt["sha256"]:
            raise ValueError("Historical source changed")
    folger_path = Path("experiments/deep-study-v2/editorial-links.json")
    folger = json.loads(folger_path.read_text())
    runs = []
    for exclude154 in [True, False]:
        reference = next(v for v in saved[0]["versions"] if v["exclude154"] is exclude154)
        single = next(v for v in saved[1]["versions"] if v["exclude154"] is exclude154)
        audits = validate_old_splits(reference, single, folger)
        scores = {name: np.asarray(single["predictions"] if name == "single_depth" else reference["predictions"][name], dtype=float)
                  for name in plan["models"]}
        result = {"exclude154": exclude154, "cohort_rows": len(reference["rows"]), "original_split_audit": audits,
                  "sources": [source_evaluation(s, reference["rows"], scores, not exclude154) for s in labels["sources"]]}
        runs.append(result)
        print(json.dumps({"exclude154": exclude154, "sources": [{"source": s["source_id"],
              "new_historical_links": s["populations"]["new_historical_links"]["metrics"],
              "broad_group_exclusion": s["populations"]["new_historical_links_without_broad_groups"]["metrics"]}
              for s in result["sources"]]}), flush=True)
    output = {"schema_version": "historical-editorial-transfer-v3", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_cpu_wall_seconds": time.monotonic()-start, "self_checks": self_checks(),
        "source_files": [digest(p) for p in [plan_path, plan["labels"]["path"], folger_path, __file__]] + plan["fixed_prediction_files"] + historical_receipt["sources"],
        "plan": plan, "runs": runs,
        "interpretation": "Annotation-source transfer of fixed scores; not new-corpus confirmation, blinded human evaluation or semantic-negative classification.",
        "limitations": ["Both literary corpus and Folger-derived fitted scores were explored before the new labels.",
            "Unmentioned relations remain unlabeled; source-ledger AUROC/AP assess documentation recovery only.",
            "Historical editors share critical tradition; Beeching reports Lee and is not an independent third rater.",
            "Historical broad units cross some original folds; only original Folger components were purged.",
            "Positive-only score summaries cannot establish calibration or utility against false positives.",
            "No p-values, confidence intervals or claim of statistical confirmation are made.",
            "Historical edition text variation and model pretraining exposure remain unresolved."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(output, f, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output)["sha256"]}))


if __name__ == "__main__":
    main()
