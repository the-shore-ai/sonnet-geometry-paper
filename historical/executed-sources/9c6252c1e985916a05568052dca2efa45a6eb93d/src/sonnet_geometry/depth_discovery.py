"""Post hoc all-depth geometry and grouped internal validation of saved RDMs.

No model execution, statistical p-values, or claim of independent confirmation.
Every distance is Q12 before ranking. Final normalization is a separate state,
never another transformer depth. Undefined constant/residual geometry stays NaN
internally and becomes JSON null at the artifact boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import rankdata

from .comparisons import _alignment, _unique_ids, residual_depths
from .structure import _cosine_rdm

POOLINGS = ("last_token", "cumulative_mean")
DEPTHS = (0.0, .25, .5, .75, 1.0)
SEEDS = (17, 41, 73)


@dataclass
class ModelRDMs:
    model_id: str
    revision: str
    sonnet_ids: list[str]
    layers: list[dict]
    distances: dict[str, np.ndarray]
    final_normalized: dict[str, np.ndarray]
    source: dict


def load_model_rdms(path: str | Path, target_ids: Sequence[str] | None = None) -> ModelRDMs:
    """Read one existing batch report, validate IDs/RDMs, retain compact Q12 arrays."""
    path = Path(path)
    data = path.read_bytes()
    report = json.loads(data)
    source = _unique_ids(report["sonnet_ids"], str(path))
    target = _unique_ids(list(target_ids) if target_ids is not None else source, "target")
    order = _alignment(source, target, allow_subset=False, label=str(path))
    layers = residual_depths(report["pooling"][POOLINGS[0]])
    matrices, final = {}, {}
    for pool in POOLINGS:
        if residual_depths(report["pooling"][pool]) != layers:
            raise ValueError("Pooling methods must contain identical raw residual layers")
        def convert(name):
            d = _cosine_rdm(report["pooling"][pool][name]["cosine_distance"], len(source), order,
                            label=f"{report['model_id']}/{pool}/{name}")
            return np.round(d, 12)
        matrices[pool] = np.stack([convert(layer["tensor"]) for layer in layers])
        final[pool] = convert("final_normalized")
    return ModelRDMs(report["model_id"], report["revision"], target, layers, matrices, final,
                     {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)})


def normalized_rows(values: np.ndarray, *, relative_to: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    threshold = np.full(len(norms), 1e-12)
    if relative_to is not None:
        reference = relative_to - relative_to.mean(axis=1, keepdims=True)
        reference_norms = np.linalg.norm(reference, axis=1)
        threshold = np.maximum(threshold, reference_norms * 1e-10)
    valid = norms > threshold
    if relative_to is not None:
        # Projection roundoff cannot create information from a constant outcome.
        valid &= reference_norms > 1e-12
    out = np.zeros_like(centered)
    out[valid] = centered[valid] / norms[valid, None]
    return out, valid


def rank_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return normalized_rows(rankdata(values, axis=1, method="average"))


def correlation_map(a: np.ndarray, b: np.ndarray, *, ranked: bool = True) -> np.ndarray:
    aa, av = rank_rows(a) if ranked else normalized_rows(a)
    bb, bv = rank_rows(b) if ranked else normalized_rows(b)
    result = np.clip(aa @ bb.T, -1, 1)
    result[~av, :] = np.nan
    result[:, ~bv] = np.nan
    return result


def edge_values(matrices: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """One vector per layer, with both endpoints wholly inside the subset."""
    indices = np.asarray(indices, dtype=int)
    triangle = np.triu_indices(len(indices), 1)
    return matrices[:, indices[triangle[0]], indices[triangle[1]]]


def adjusted_map(a: np.ndarray, b: np.ndarray, controls: np.ndarray) -> tuple[np.ndarray, int]:
    design = np.column_stack([np.ones(a.shape[1]), rankdata(controls, axis=1).T])
    y = rankdata(np.concatenate([a, b]), axis=1).T
    residual = (y - design @ np.linalg.lstsq(design, y, rcond=None)[0]).T
    normalized, valid = normalized_rows(residual, relative_to=y.T)
    result = np.clip(normalized[:len(a)] @ normalized[len(a):].T, -1, 1)
    result[~valid[:len(a)], :] = np.nan
    result[:, ~valid[len(a):]] = np.nan
    return result, int(np.linalg.matrix_rank(design))


def empirical_midcdf(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Learn a monotone mid-CDF per feature solely on training edge values."""
    train, test = np.asarray(train, float), np.asarray(test, float)
    if train.ndim != 2 or test.ndim != 2 or train.shape[0] != test.shape[0] or train.shape[1] < 2:
        raise ValueError("CDF transforms need aligned feature rows and at least two training observations")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("CDF transforms require finite observations")
    transformed_train = (rankdata(train, axis=1) - .5) / train.shape[1]
    transformed_test = np.empty_like(test)
    for row in range(len(train)):
        reference = np.sort(train[row])
        transformed_test[row] = (np.searchsorted(reference, test[row], side="left") +
                                 np.searchsorted(reference, test[row], side="right")) / (2 * len(reference))
    return transformed_train, transformed_test


def crossfit_adjusted_maps(a_train, b_train, controls_train, a_test, b_test, controls_test):
    """Fit CDFs and OLS on train edges; evaluate centered residual Pearson on test."""
    ctrain, ctest = empirical_midcdf(controls_train, controls_test)
    ytrain, ytest = empirical_midcdf(np.concatenate([a_train, b_train]), np.concatenate([a_test, b_test]))
    design = np.column_stack([np.ones(ctrain.shape[1]), ctrain.T])
    test_design = np.column_stack([np.ones(ctest.shape[1]), ctest.T])
    coefficients = np.linalg.lstsq(design, ytrain.T, rcond=None)[0]
    train_residual = (ytrain.T - design @ coefficients).T
    test_residual = (ytest.T - test_design @ coefficients).T
    def combine(residual, reference):
        normalized, valid = normalized_rows(residual, relative_to=reference)
        result = np.clip(normalized[:len(a_train)] @ normalized[len(a_train):].T, -1, 1)
        # Training-degenerate outcomes are never reinterpreted as a test signal.
        return result, valid
    train_map, train_valid = combine(train_residual, ytrain)
    test_map, test_valid = combine(test_residual, ytest)
    train_map[~train_valid[:len(a_train)], :] = np.nan
    train_map[:, ~train_valid[len(a_train):]] = np.nan
    test_valid &= train_valid
    test_map[~test_valid[:len(a_train)], :] = np.nan
    test_map[:, ~test_valid[len(a_train):]] = np.nan
    return train_map, test_map, int(np.linalg.matrix_rank(design))


def grouped_splits(ids: Sequence[str], *, folds: int = 5, seeds: Sequence[int] = SEEDS) -> list[dict]:
    """Repeated folds, with the known 153/154 pair always treated as one group."""
    ids = _unique_ids(list(ids), "split")
    groups: dict[str, list[int]] = {}
    for i, sid in enumerate(ids):
        group = "sonnet-153+154" if sid in {"sonnet-153", "sonnet-154"} else sid
        groups.setdefault(group, []).append(i)
    if folds < 2 or len(groups) < folds * 3:
        raise ValueError("Require at least three independent groups per held-out fold")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Rotation seeds must be unique")
    output = []
    for seed in seeds:
        keys = sorted(groups)
        shuffled = np.random.default_rng(seed).permutation(len(keys))
        buckets = [[] for _ in range(folds)]
        for k in shuffled:
            bucket = min(range(folds), key=lambda i: (len(buckets[i]), i))
            buckets[bucket].extend(groups[keys[k]])
        for fold, test in enumerate(buckets):
            test = sorted(test)
            train = sorted(set(range(len(ids))) - set(test))
            output.append({"seed": int(seed), "fold": fold, "train_indices": train, "test_indices": test,
                           "train_ids": [ids[i] for i in train], "test_ids": [ids[i] for i in test],
                           "training_edges": len(train) * (len(train) - 1) // 2,
                           "heldout_edges": len(test) * (len(test) - 1) // 2})
    return output


def maximum_cell(matrix: np.ndarray) -> tuple[int, int] | None:
    if not np.isfinite(matrix).any():
        return None
    best = np.nanmax(matrix)
    # Numerical ties resolve to the shallower A depth, then shallower B depth.
    return tuple(int(x) for x in np.argwhere(np.isfinite(matrix) & (matrix >= best - 1e-12))[0])


def nearest_depth(layers, depth):
    distances = [abs(layer["normalized_depth"] - depth) for layer in layers]
    minimum = min(distances)
    return next(i for i, distance in enumerate(distances) if distance <= minimum + 1e-12)


def selected_summary(train_map, test_raw, test_adjusted, layers_a, layers_b):
    cell = maximum_cell(train_map)
    if cell is None:
        return {"status": "undefined", "reason": "No nondegenerate training geometry"}
    i, j = cell
    return {"status": "computed", "layer_a_index": i, "layer_b_index": j,
            "layer_a": layers_a[i], "layer_b": layers_b[j], "training_objective": float(train_map[i, j]),
            "heldout_spearman": float(test_raw[i, j]),
            "heldout_training_cdf_residual_pearson": float(test_adjusted[i, j])}


def descriptive_peak(objective, raw, adjusted, layers_a, layers_b):
    result = selected_summary(objective, raw, adjusted, layers_a, layers_b)
    if result["status"] == "computed":
        result["searched_objective"] = result.pop("training_objective")
        result["in_sample_spearman"] = result.pop("heldout_spearman")
        result["in_sample_residual_pearson"] = result.pop("heldout_training_cdf_residual_pearson")
    return result


def neighbor_membership(matrix: np.ndarray, k: int = 5) -> np.ndarray:
    """Directed k-neighbor membership includes every tie at the kth distance."""
    n = len(matrix)
    if matrix.shape != (n, n) or not np.isfinite(matrix).all() or not 1 <= k < n:
        raise ValueError("Finite square RDM and 1 <= k < n required")
    distances = matrix.copy()
    np.fill_diagonal(distances, np.inf)
    thresholds = np.partition(distances, k - 1, axis=1)[:, k - 1]
    result = distances <= thresholds[:, None]
    np.fill_diagonal(result, False)
    return result


def _summary(values):
    a = np.asarray(values, float)
    finite = a[np.isfinite(a)]
    return {"defined": len(finite), "total": len(a), "mean": float(finite.mean()) if len(finite) else None,
            "median": float(np.median(finite)) if len(finite) else None,
            "minimum": float(finite.min()) if len(finite) else None,
            "maximum": float(finite.max()) if len(finite) else None,
            "fraction_positive": float(np.mean(finite > 0)) if len(finite) else None}


def consensus(models: list[ModelRDMs], indices: list[int], ids: list[str], k: int = 5) -> dict:
    channels, neighbor_arrays, anomaly_arrays = [], [], []
    for model in models:
        for pool in POOLINGS:
            # Embedding and post-finalnorm are excluded; native raw blocks count equally within a model.
            distances = model.distances[pool][1:][:, indices][:, :, indices]
            members = np.stack([neighbor_membership(d, k) for d in distances])
            persistence = members.mean(axis=0)
            anomaly = distances.sum(axis=2) / (len(indices) - 1)
            # Ranks remain undefined if every poem has the same anomaly score.
            informative = np.ptp(anomaly, axis=1) > 1e-12
            percentiles = (rankdata(anomaly[informative], axis=1) - 1) / (len(indices) - 1)
            median_ranks = np.median(percentiles, axis=0) if informative.any() else np.full(len(ids), np.nan)
            channels.append({"model_id": model.model_id, "pooling": pool, "raw_block_count": len(distances),
                             "informative_anomaly_layers": int(informative.sum()),
                             "median_anomaly_percentiles": median_ranks,
                             "anomaly_percentile_q10": np.quantile(percentiles, .1, axis=0) if informative.any() else np.full(len(ids), np.nan),
                             "mean_neighbor_set_size": float(members.sum(axis=2).mean()),
                             "neighbor_persistence": persistence})
            neighbor_arrays.append(persistence)
            anomaly_arrays.append(median_ranks)
    neighbors, anomaly = np.stack(neighbor_arrays), np.stack(anomaly_arrays)
    # Equal model/pooling channel weights; different native layer counts do not add model weight.
    mean_neighbors, minimum_neighbors = neighbors.mean(axis=0), neighbors.min(axis=0)
    median_anomaly = np.nanmedian(anomaly, axis=0)
    minimum_anomaly = np.nanmin(anomaly, axis=0)
    anomaly_rows = [{"sonnet_id": sid, "median_channel_anomaly_percentile": float(median_anomaly[i]),
                     "minimum_channel_anomaly_percentile": float(minimum_anomaly[i]),
                     "channel_percentiles": anomaly[:, i].tolist()} for i, sid in enumerate(ids)]
    anomaly_rows.sort(key=lambda row: (-row["minimum_channel_anomaly_percentile"], -row["median_channel_anomaly_percentile"], row["sonnet_id"]))
    pairs = []
    for i, j in itertools.combinations(range(len(ids)), 2):
        reciprocal = (neighbors[:, i, j] + neighbors[:, j, i]) / 2
        pairs.append({"sonnet_a": ids[i], "sonnet_b": ids[j], "mean_reciprocal_persistence": float(reciprocal.mean()),
                      "minimum_channel_reciprocal_persistence": float(reciprocal.min()),
                      "channel_reciprocal_persistence": reciprocal.tolist()})
    pairs.sort(key=lambda row: (-row["minimum_channel_reciprocal_persistence"], -row["mean_reciprocal_persistence"], row["sonnet_a"], row["sonnet_b"]))
    return {"sonnet_ids": ids, "neighbor_k": k, "tie_policy": "Include all ties at kth Q12 distance; sizes may exceed k",
            "channels": channels, "mean_neighbor_persistence": mean_neighbors, "minimum_neighbor_persistence": minimum_neighbors,
            "anomaly_consensus": anomaly_rows, "all_neighbor_pairs": pairs,
            "candidate_rule": "Descriptive candidates: minimum channel reciprocal persistence >= .25 and mean >= .5; no p-value or literary label",
            "neighbor_candidates": [p for p in pairs if p["minimum_channel_reciprocal_persistence"] >= .25 and p["mean_reciprocal_persistence"] >= .5]}


def analyze_depths(models: list[ModelRDMs], baselines: dict, records: list[dict], terminal_tokens: dict[str, dict],
                   *, seeds: Sequence[int] = SEEDS, folds: int = 5) -> dict:
    """Bounded exhaustive maps, all searched selections, and grouped held-out evaluation."""
    if len(models) != 3 or len({m.model_id for m in models}) != 3:
        raise ValueError("This three-family audit requires three distinct models")
    ids = models[0].sonnet_ids
    if any(model.sonnet_ids != ids for model in models):
        raise ValueError("Model identities must already be aligned")
    by_id = {r["sonnet_id"]: r for r in records}
    if len(by_id) != len(records) or set(by_id) != set(ids):
        raise ValueError("Unique corpus IDs must exactly match models")
    baseline_ids = _unique_ids(baselines["sonnet_ids"], "baseline")
    order = _alignment(baseline_ids, ids, allow_subset=False, label="baseline")
    baseline_names = list(baselines["baselines"])
    baseline_matrices = np.stack([np.round(_cosine_rdm(baselines["baselines"][name]["cosine_distance"],
        len(ids), order, label=name), 12) for name in baseline_names])
    if set(terminal_tokens) != {m.model_id for m in models}:
        raise ValueError("Each model needs its verified terminal-token metadata")
    labels = []
    for model in models:
        mapping = terminal_tokens[model.model_id]
        if set(mapping) != set(ids):
            raise ValueError("Terminal-token identity mismatch")
        labels.append([mapping[sid]["last_included_token_decoded"] for sid in ids])
    if any(label != labels[0] for label in labels):
        raise ValueError("Cross-model terminal groups differ; define a new explicitly matched cohort")
    punctuation = labels[0]
    control_names = list(baseline_names)
    controls = list(baseline_matrices)
    for name, values in [("absolute_word_count_difference", [len(by_id[s]["text"].split()) for s in ids]),
                         ("absolute_character_count_difference", [len(by_id[s]["text"]) for s in ids])]:
        values = np.asarray(values)
        control_names.append(name)
        controls.append(np.abs(values[:, None] - values[None, :]))
    # Full unordered terminal-pair categories distinguish within-period vs other same-token pairs.
    for pair in itertools.combinations_with_replacement(sorted(set(punctuation)), 2):
        control_names.append("terminal_pair:" + repr(pair))
        controls.append(np.asarray([[float(tuple(sorted((a, b))) == pair) for b in punctuation] for a in punctuation]))
    controls = np.stack(controls)
    cohorts = {"all154": list(range(len(ids))),
               "period143": [i for i, label in enumerate(punctuation) if label == "."],
               "all_without154": [i for i, sid in enumerate(ids) if sid != "sonnet-154"],
               "period_without154": [i for i, sid in enumerate(ids) if punctuation[i] == "." and sid != "sonnet-154"]}
    outputs = {"maps": [], "evolution": [], "heldout": [], "splits": {}, "consensus": {},
               "cohorts": {name: [ids[i] for i in indices] for name, indices in cohorts.items()},
               "model_layers": [{"model_id": m.model_id, "revision": m.revision, "layers": m.layers} for m in models],
               "control_names": control_names}
    for cohort_name, indices in cohorts.items():
        cohort_ids = [ids[i] for i in indices]
        common_controls = edge_values(controls, indices)
        baseline_edges = edge_values(baseline_matrices, indices)
        edges = {(i, pool): edge_values(m.distances[pool], indices) for i, m in enumerate(models) for pool in POOLINGS}
        for i, model in enumerate(models):
            for pool in POOLINGS:
                values = edges[(i, pool)]
                self_map = correlation_map(values, values)
                final = edge_values(model.final_normalized[pool][None, :, :], indices)
                raw_matrix = model.distances[pool][-1][np.ix_(indices, indices)]
                normalized_matrix = model.final_normalized[pool][np.ix_(indices, indices)]
                raw_neighbors, normalized_neighbors = neighbor_membership(raw_matrix), neighbor_membership(normalized_matrix)
                neighborhood_jaccard = (raw_neighbors & normalized_neighbors).sum(axis=1) / (raw_neighbors | normalized_neighbors).sum(axis=1)
                outputs["evolution"].append({"cohort": cohort_name, "model_id": model.model_id, "pooling": pool,
                    "all_depth_within_model_rsa": self_map,
                    "successive_depth_rsa": np.diag(self_map, 1), "embedding_rsa": self_map[:, 0],
                    "last_raw_block_rsa": self_map[:, -1], "final_normalized_rsa": correlation_map(values, final)[:, 0],
                    "baseline_names": baseline_names, "all_depth_baseline_rsa": correlation_map(values, baseline_edges),
                    "final_normalized_baseline_rsa": correlation_map(final, baseline_edges)[0],
                    "last_raw_vs_final_normalized_anomaly_rank_rsa": correlation_map(raw_matrix.sum(axis=1)[None], normalized_matrix.sum(axis=1)[None])[0, 0],
                    "last_raw_vs_final_normalized_neighbor_jaccard": neighborhood_jaccard,
                    "last_raw_vs_final_normalized_mean_neighbor_jaccard": float(neighborhood_jaccard.mean())})
        for a, b in itertools.combinations(range(len(models)), 2):
            for pool in POOLINGS:
                x, y = edges[(a, pool)], edges[(b, pool)]
                nuisance = np.concatenate([common_controls, x[:1], y[:1]])
                raw = correlation_map(x, y)
                adjusted, design_rank = adjusted_map(x, y, nuisance)
                f_a = edge_values(models[a].final_normalized[pool][None], indices)
                f_b = edge_values(models[b].final_normalized[pool][None], indices)
                outputs["maps"].append({"cohort": cohort_name, "model_a": models[a].model_id, "model_b": models[b].model_id,
                    "pooling": pool, "spearman": raw, "nuisance_residual_pearson": adjusted,
                    "nuisance_design_rank": design_rank, "searched_layer_pairs": int(raw.size),
                    "descriptive_raw_maximum": descriptive_peak(raw, raw, adjusted, models[a].layers, models[b].layers),
                    "descriptive_adjusted_maximum": descriptive_peak(adjusted, raw, adjusted, models[a].layers, models[b].layers),
                    "final_normalized_cross_model_rsa": correlation_map(f_a, f_b)[0, 0]})
        splits = grouped_splits(cohort_ids, folds=folds, seeds=seeds)
        outputs["splits"][cohort_name] = splits
        for split in splits:
            train = [indices[i] for i in split["train_indices"]]
            test = [indices[i] for i in split["test_indices"]]
            train_control, test_control = edge_values(controls, train), edge_values(controls, test)
            train_edges = {(i, pool): edge_values(m.distances[pool], train) for i, m in enumerate(models) for pool in POOLINGS}
            test_edges = {(i, pool): edge_values(m.distances[pool], test) for i, m in enumerate(models) for pool in POOLINGS}
            for a, b in itertools.combinations(range(len(models)), 2):
                for pool in POOLINGS:
                    x, y = train_edges[(a, pool)], train_edges[(b, pool)]
                    tx, ty = test_edges[(a, pool)], test_edges[(b, pool)]
                    nuisance_train = np.concatenate([train_control, x[:1], y[:1]])
                    nuisance_test = np.concatenate([test_control, tx[:1], ty[:1]])
                    train_raw, test_raw = correlation_map(x, y), correlation_map(tx, ty)
                    train_adj, test_adj, rank = crossfit_adjusted_maps(x, y, nuisance_train, tx, ty, nuisance_test)
                    fixed = []
                    for depth in DEPTHS:
                        ia, ib = nearest_depth(models[a].layers, depth), nearest_depth(models[b].layers, depth)
                        fixed.append({"normalized_depth": depth, "layer_a_index": ia, "layer_b_index": ib,
                                      "heldout_spearman": test_raw[ia, ib],
                                      "heldout_training_cdf_residual_pearson": test_adj[ia, ib]})
                    outputs["heldout"].append({"cohort": cohort_name, "seed": split["seed"], "fold": split["fold"],
                        "model_a": models[a].model_id, "model_b": models[b].model_id, "pooling": pool,
                        "training_edges": split["training_edges"], "heldout_edges": split["heldout_edges"],
                        "nuisance_design_rank": rank,
                        "raw_selected": selected_summary(train_raw, test_raw, test_adj, models[a].layers, models[b].layers),
                        "adjusted_selected": selected_summary(train_adj, test_raw, test_adj, models[a].layers, models[b].layers),
                        "fixed_depth_comparators": fixed})
        outputs["consensus"][cohort_name] = consensus(models, indices, cohort_ids)
    summaries = []
    for cohort_name in cohorts:
        for a, b in itertools.combinations(models, 2):
            for pool in POOLINGS:
                rows = [r for r in outputs["heldout"] if (r["cohort"], r["model_a"], r["model_b"], r["pooling"]) == (cohort_name, a.model_id, b.model_id, pool)]
                summary = {"cohort": cohort_name, "model_a": a.model_id, "model_b": b.model_id, "pooling": pool}
                for objective in ["raw_selected", "adjusted_selected"]:
                    selections = [r[objective] for r in rows]
                    summary[objective] = {metric: _summary([s.get(metric, np.nan) for s in selections]) for metric in
                        ["training_objective", "heldout_spearman", "heldout_training_cdf_residual_pearson"]}
                    summary[objective]["selected_layer_pairs"] = [{"layer_a_index": s.get("layer_a_index"),
                        "layer_b_index": s.get("layer_b_index"), "count": sum(t.get("layer_a_index") == s.get("layer_a_index") and
                        t.get("layer_b_index") == s.get("layer_b_index") for t in selections)} for s in
                        {str((s.get("layer_a_index"), s.get("layer_b_index"))): s for s in selections}.values()]
                summary["fixed_depth_comparators"] = [{"normalized_depth": depth,
                    **{metric: _summary([r["fixed_depth_comparators"][di][metric] for r in rows]) for metric in
                    ["heldout_spearman", "heldout_training_cdf_residual_pearson"]}} for di, depth in enumerate(DEPTHS)]
                summaries.append(summary)
    outputs["heldout_summary"] = summaries
    outputs["robustness"] = summarize_robustness(outputs)
    return outputs


def summarize_robustness(outputs):
    """Descriptive cross-cohort stability without new hypothesis tests."""
    cohorts = list(outputs["cohorts"])
    common = sorted(set.intersection(*(set(outputs["cohorts"][c]) for c in cohorts)))
    anomalies, pairs = {}, {}
    for cohort in cohorts:
        anomalies[cohort] = {r["sonnet_id"]: r for r in outputs["consensus"][cohort]["anomaly_consensus"]}
        pairs[cohort] = {tuple(sorted((r["sonnet_a"], r["sonnet_b"]))): r for r in outputs["consensus"][cohort]["all_neighbor_pairs"]}
    stable_anomalies = []
    for sid in common:
        values = [anomalies[c][sid]["minimum_channel_anomaly_percentile"] for c in cohorts]
        stable_anomalies.append({"sonnet_id": sid, "minimum_across_cohort_and_channel_percentile": min(values),
                                "per_cohort_minimum_channel_percentile": dict(zip(cohorts, values))})
    stable_anomalies.sort(key=lambda r: (-r["minimum_across_cohort_and_channel_percentile"], r["sonnet_id"]))
    stable_neighbors = []
    for key in itertools.combinations(common, 2):
        values = [pairs[c][key] for c in cohorts]
        stable_neighbors.append({"sonnet_a": key[0], "sonnet_b": key[1],
            "minimum_across_cohort_and_channel_persistence": min(r["minimum_channel_reciprocal_persistence"] for r in values),
            "mean_across_cohort_and_channel_persistence": float(np.mean([r["mean_reciprocal_persistence"] for r in values])),
            "per_cohort_minimum_channel_persistence": {c: r["minimum_channel_reciprocal_persistence"] for c, r in zip(cohorts, values)}})
    stable_neighbors.sort(key=lambda r: (-r["minimum_across_cohort_and_channel_persistence"], -r["mean_across_cohort_and_channel_persistence"], r["sonnet_a"], r["sonnet_b"]))
    map_changes = []
    for first, second in [("all154", "all_without154"), ("period143", "period_without154"), ("all154", "period143")]:
        for row in outputs["maps"]:
            if row["cohort"] != first:
                continue
            other = next(r for r in outputs["maps"] if r["cohort"] == second and all(r[k] == row[k] for k in ["model_a", "model_b", "pooling"]))
            for metric in ["spearman", "nuisance_residual_pearson"]:
                a, b = row[metric].ravel(), other[metric].ravel()
                valid = np.isfinite(a) & np.isfinite(b)
                map_changes.append({"cohort_a": first, "cohort_b": second,
                    **{k: row[k] for k in ["model_a", "model_b", "pooling"]}, "metric": metric,
                    "comparable_cells": int(valid.sum()), "all_cells": len(a),
                    "mean_absolute_cell_change": float(np.mean(np.abs(a[valid] - b[valid]))),
                    "maximum_absolute_cell_change": float(np.max(np.abs(a[valid] - b[valid]))),
                    "map_cell_rank_correlation": correlation_map(a[valid][None], b[valid][None])[0, 0]})
    duplicate = {}
    for c in cohorts:
        if ("sonnet-153", "sonnet-154") in pairs[c]:
            duplicate[c] = pairs[c][("sonnet-153", "sonnet-154")]
    return {"common_sonnet_ids": common, "stable_anomaly_order": stable_anomalies,
            "all_stable_neighbor_pairs": stable_neighbors,
            "stable_neighbor_candidates": [r for r in stable_neighbors if r["minimum_across_cohort_and_channel_persistence"] >= .25 and r["mean_across_cohort_and_channel_persistence"] >= .5],
            "known_153_154_pair": duplicate, "map_cohort_changes": map_changes,
            "inference": "Descriptive consistency of the searched maps and six-channel consensus across declared cohorts; no confidence intervals or significance tests"}


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
