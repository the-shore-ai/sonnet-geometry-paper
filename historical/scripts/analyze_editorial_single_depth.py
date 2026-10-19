"""Nested single-state challenge to the existing editorial-link diagnostic."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from sonnet_geometry.corpus import load_corpus
from sonnet_geometry.depth_discovery import load_model_rdms, nearest_depth
from sonnet_geometry.editorial_diagnostic import (
    CS, Features, fitted_prediction, metrics, paired_stability, purged_training_indices,
)

STATES = ("0.25", "0.5", "0.75", "1.0", "final_normalized")
POLICY = {
    "states_in_tie_order": STATES, "C_in_tie_order": CS,
    "selection": "Joint state/C minimum validation-size-weighted inner log loss; exact ties resolve by listed state then C order",
    "feature_count": 20, "state_shared_across_models_and_poolings": True,
    "features": "8 conventional + 6 model/pooling input distances + 6 model/pooling selected-state distances",
    "folds": "Same 7 chronological outer bands and 3 grouped inner folds as saved final-state sensitivity; purge exact poems and linked components",
    "stability": "2000 draws of 7 chronological bands with replacement, seed 20261018; fixed OOF predictions; no p-values",
    "scope": "Post hoc internal challenge; full adjacent-pair cohort and exclusion of Sonnet 154; no new target or independent confirmation",
}


def digest(path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def candidate_features(features, train, state):
    if state not in STATES:
        raise ValueError("Unknown candidate state")
    suffix = "/final_normalized" if state == "final_normalized" else f"/fixed/{state}"
    inputs = [i for i, name in enumerate(features.names) if name.endswith("/input")]
    chosen = [i for i, name in enumerate(features.names) if name.endswith(suffix)]
    if len(inputs) != 6 or len(chosen) != 6:
        raise ValueError("Exactly six input and six selected-state distances are required")
    conventional = features.get(train, "conventional")
    if conventional.shape[1] != 8:
        raise ValueError("Expected the existing eight-feature conventional baseline")
    return np.column_stack([conventional, features.fixed[:, inputs], features.fixed[:, chosen]])


def select_candidate(features, labels, inner_splits):
    """Outer test indices/labels are absent from selection; every candidate saved."""
    candidates = []
    for state in STATES:
        matrices = [candidate_features(features, train, state) for train, _ in inner_splits]
        for c in CS:
            losses, counts = [], []
            for (train, validation), x in zip(inner_splits, matrices):
                probability = fitted_prediction(x, labels, train, validation, c)
                losses.append(float(log_loss(labels[validation], probability, labels=[0, 1])))
                counts.append(len(validation))
            candidates.append({"state": state, "C": c, "inner_fold_log_losses": losses,
                               "inner_validation_counts": counts,
                               "weighted_inner_log_loss": float(np.average(losses, weights=counts))})
    best = min(range(len(candidates)), key=lambda i: (candidates[i]["weighted_inner_log_loss"], i))
    return candidates[best], candidates


def nested_single_depth(features, labels, bands, components):
    labels, bands = np.asarray(labels), np.asarray(bands)
    predictions = np.full(len(labels), np.nan)
    folds = []
    for band in sorted(set(bands)):
        test = np.flatnonzero(bands == band)
        initial = np.flatnonzero(bands != band)
        train = purged_training_indices(features.pairs, initial, test, components)
        if len(np.unique(labels[train])) != 2:
            raise ValueError("Purged outer training set lacks both classes")
        inner = []
        for tr, va in GroupKFold(n_splits=3).split(train, labels[train], bands[train]):
            validation = train[va]
            training = purged_training_indices(features.pairs, train[tr], validation, components)
            if len(np.unique(labels[training])) != 2:
                raise ValueError("Purged inner training set lacks both classes")
            inner.append((training, validation))
        chosen, candidates = select_candidate(features, labels, inner)
        x = candidate_features(features, train, chosen["state"])
        probability = fitted_prediction(x, labels, train, test, chosen["C"])
        predictions[test] = probability
        folds.append({"band": int(band), "train_rows": train.tolist(), "test_rows": test.tolist(),
                      "purged_rows": sorted(set(initial.tolist()) - set(train.tolist())),
                      "inner_splits": [{"train_rows": tr.tolist(), "validation_rows": va.tolist()} for tr, va in inner],
                      "selected_state": chosen["state"], "selected_C": chosen["C"],
                      "selected_inner_log_loss": chosen["weighted_inner_log_loss"],
                      "feature_count": int(x.shape[1]), "candidates": candidates,
                      "test_metrics": metrics(labels[test], probability)})
    if not np.isfinite(predictions).all():
        raise ValueError("Incomplete out-of-fold predictions")
    return {"folds": folds, "metrics": metrics(labels, predictions), "predictions": predictions.tolist()}


def verify_reference_splits(result, reference):
    if len(result["folds"]) != len(reference["folds"]):
        raise ValueError("Reference outer fold count differs")
    for actual, expected in zip(result["folds"], reference["folds"]):
        for key in ["band", "train_rows", "test_rows", "purged_rows", "inner_splits"]:
            if actual[key] != expected[key]:
                raise ValueError(f"Reference split mismatch: {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/deep-study-v2/editorial-single-depth.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    start = time.monotonic()
    corpus_path = Path("datasets/normalized/sonnets.jsonl")
    labels_path = Path("experiments/deep-study-v2/editorial-links.json")
    baselines_path = Path("experiments/baselines.json")
    config_path = Path("configs/baselines.lock.json")
    reference_path = Path("experiments/deep-study-v2/editorial-final-state-sensitivity.json")
    records = load_corpus(corpus_path)
    ids, texts = [r["sonnet_id"] for r in records], [r["text"] for r in records]
    annotations, baselines, config, reference = [json.loads(p.read_text()) for p in [labels_path, baselines_path, config_path, reference_path]]
    if ids != baselines["sonnet_ids"]:
        raise ValueError("Baseline/model identity mismatch")
    components = {n - 1: group["component_id"] for group in annotations["fold_components"] for n in group["sonnet_numbers"]}
    components[152] = components[153] = "positive-control-153-154"
    if set(components) != set(range(154)):
        raise ValueError("Incomplete component assignments")
    positive = {tuple(e["sonnet_numbers"]) for e in annotations["positive_edges"]}
    positive.add(tuple(annotations["positive_control"]["sonnet_numbers"]))
    ambiguous = {tuple(e["sonnet_numbers"]) for e in annotations["excluded_relations"]}
    matrices, source_models = {}, []
    for family in ["qwen", "gemma", "muse"]:
        model = load_model_rdms(f"outputs/full-{family}-v1/report.json", ids)
        source_models.append(model.source)
        for pooling, values in model.distances.items():
            matrices[f"{family}/{pooling}/input"] = values[0]
            for state in STATES:
                if state == "final_normalized":
                    matrices[f"{family}/{pooling}/final_normalized"] = model.final_normalized[pooling]
                else:
                    index = nearest_depth(model.layers, float(state))
                    matrices[f"{family}/{pooling}/fixed/{state}"] = values[index]
    minilm = np.asarray(baselines["baselines"]["minilm_sentence_embedding"]["cosine_distance"])
    versions = []
    for exclude154 in [False, True]:
        saved = next(v for v in reference["versions"] if v["exclude154"] is exclude154)
        pairs = np.array([(i, i+1) for i in range(153) if i // 22 == (i+1) // 22 and (i+1, i+2) not in ambiguous and not (exclude154 and i == 152)])
        labels = np.array([int((i+1, j+1) in positive) for i, j in pairs])
        bands = pairs[:, 0] // 22
        rows = [{"row": r, "sonnets": [int(i+1), int(j+1)], "band": int(bands[r]),
                 "editorially_linked": int(labels[r]), "positive_control": (i+1, j+1) == tuple(annotations["positive_control"]["sonnet_numbers"])} for r, (i, j) in enumerate(pairs)]
        if rows != saved["rows"] or len(set(bands)) != 7:
            raise ValueError("Rows/targets differ from frozen seven-band reference")
        values = np.column_stack([matrix[pairs[:, 0], pairs[:, 1]] for matrix in matrices.values()])
        features = Features(texts, pairs, minilm, values, list(matrices), config["tfidf"])
        result = nested_single_depth(features, labels, bands, components)
        verify_reference_splits(result, saved)
        comparison_predictions = {"single_depth": result["predictions"],
            **{name: saved["predictions"][name] for name in ["final_plus_fixed_depths", "plus_both_final"]}}
        result["exclude154"] = exclude154
        result["rows"] = rows
        result["comparison_metrics"] = {name: metrics(labels, p) for name, p in comparison_predictions.items()}
        result["comparison_predictions"] = comparison_predictions
        result["conditional_block_stability"] = paired_stability(labels, bands, comparison_predictions, reference="single_depth")
        result["paired_band_differences"] = [{"band": int(band), "comparator": name,
            "delta_log_loss_comparator_minus_single": metrics(labels[bands == band], np.asarray(prediction)[bands == band])["log_loss"] - metrics(labels[bands == band], np.asarray(result["predictions"])[bands == band])["log_loss"],
            "delta_brier_comparator_minus_single": metrics(labels[bands == band], np.asarray(prediction)[bands == band])["brier"] - metrics(labels[bands == band], np.asarray(result["predictions"])[bands == band])["brier"]}
            for band in sorted(set(bands)) for name, prediction in comparison_predictions.items() if name != "single_depth"]
        result["reference_splits_exactly_matched"] = True
        versions.append(result)
        print(json.dumps({"exclude154": exclude154, "metrics": result["comparison_metrics"],
                          "choices": [(f["selected_state"], f["selected_C"]) for f in result["folds"]]}), flush=True)
    payload = {"schema_version": "editorial-single-depth-v2", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_cpu_wall_seconds": time.monotonic()-start, "predeclared_policy": POLICY,
        "policy_sha256": hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest(),
        "source_files": [digest(p) for p in [corpus_path, labels_path, baselines_path, config_path, reference_path,
            Path(__file__), Path("src/sonnet_geometry/editorial_diagnostic.py"), Path("src/sonnet_geometry/depth_discovery.py")]] + source_models,
        "feature_names": list(matrices), "versions": versions,
        "limitations": ["All outer-test results are reported once; no post-result choice of state or C.",
            "Target and entire corpus were previously observed; this is posthoc internal validation.",
            "Unmentioned adjacent pairs are not verified semantic negatives.",
            "Seven outer chronological units; overlapping fits and fixed-prediction resampling do not establish independent confirmation.",
            "The shared-depth constraint is one bounded single-state baseline, not exhaustive independent per-model state selection.",
            "Frozen model training exposure and BF16 cross-length numerical sensitivity remain unresolved."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output)["sha256"], "seconds": payload["elapsed_cpu_wall_seconds"]}))


if __name__ == "__main__":
    main()
