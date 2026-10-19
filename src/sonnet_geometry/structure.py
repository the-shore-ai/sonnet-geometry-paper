"""Descriptive clustering, anomaly agreement, and nuisance-adjusted RSA.

These exploratory summaries supply no p-values and no literary labels. A
correlation after nuisance projection is not evidence of incremental utility.
"""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score

from .comparisons import DEFAULT_DEPTHS, DEFAULT_POOLINGS, _alignment, _rdm, _unique_ids, select_depth
from .distance_policy import apply_distance_policy, distance_policy


def _cosine_rdm(value: Any, n: int, order: list[int], *, label: str) -> np.ndarray:
    matrix = _rdm(value, n, order, label=label)
    if np.any(matrix < -1e-7) or np.any(matrix > 2 + 1e-7):
        raise ValueError(f"{label}: cosine distances must lie within [0, 2]")
    matrix = np.clip((matrix + matrix.T) / 2, 0, 2)
    np.fill_diagonal(matrix, 0)
    return matrix


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = rankdata(a), rankdata(b)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(a @ b / denominator, -1, 1)) if denominator > 1e-12 else None


def nuisance_adjusted_rsa(a: np.ndarray, b: np.ndarray, controls: np.ndarray) -> dict[str, Any]:
    """Residual Pearson correlation after OLS projection of ranked distances."""
    a, b, controls = np.asarray(a), np.asarray(b), np.asarray(controls)
    if a.ndim != 1 or b.shape != a.shape or controls.ndim != 2 or controls.shape[0] != len(a):
        raise ValueError("Nuisance controls must align with the same pairwise observations")
    if not all(np.isfinite(x).all() for x in (a, b, controls)):
        raise ValueError("Nuisance analysis requires finite data")
    design = np.column_stack([np.ones(len(a)), *[rankdata(column) for column in controls.T]])
    ranks = np.column_stack([rankdata(a), rankdata(b)])
    residuals = ranks - design @ np.linalg.lstsq(design, ranks, rcond=None)[0]
    norms = np.linalg.norm(residuals, axis=0)
    rank_norms = np.linalg.norm(ranks - ranks.mean(axis=0), axis=0)
    threshold = np.maximum(1e-9, 1e-10 * rank_norms)
    correlation = None if np.any(norms <= threshold) else float(np.clip(
        residuals[:, 0] @ residuals[:, 1] / np.prod(norms), -1, 1))
    return {"partial_rank_correlation": correlation, "design_rank": int(np.linalg.matrix_rank(design)),
            "residual_norms": norms.tolist(), "undefined_residual_thresholds": threshold.tolist(), "pair_count": len(a),
            "inference": "Descriptive residual correlation; pairwise observations are dependent, no p-value or incremental-value claim"}


def classical_coordinates(cosine_distance: np.ndarray) -> dict[str, Any]:
    """Classical scaling of unit-vector chord distance, retaining two axes."""
    d = np.asarray(cosine_distance, dtype=float)
    if d.ndim != 2 or len(d) < 2:
        raise ValueError("Classical scaling requires a square distance matrix with at least two samples")
    n = len(d)
    d = _cosine_rdm(d, n, list(range(n)), label="classical scaling")
    center = np.eye(n) - np.ones((n, n)) / n
    gram = -center @ d @ center  # Squared chord distance = 2 * cosine distance.
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    coordinates = vectors[:, :2] * np.sqrt(np.maximum(values[:2], 0))
    # Deterministic sign only; repeated eigenvalues still permit arbitrary axes.
    for i in range(coordinates.shape[1]):
        if coordinates[np.argmax(np.abs(coordinates[:, i])), i] < 0:
            coordinates[:, i] *= -1
    positive = np.maximum(values, 0)
    return {"coordinates": coordinates.tolist(), "eigenvalues": values.tolist(),
            "two_axis_positive_inertia_fraction": float(positive[:2].sum() / positive.sum()) if positive.sum() else None,
            "method": "Classical scaling of sqrt(2*cosine_distance); display only, axes are not literary categories"}


def summarize_structure(reports: list[dict], baselines: dict, records: list[dict], *, distance_decimals: int | None = None) -> dict:
    if not reports:
        raise ValueError("At least one model is required")
    numeric_policy, distance_audits = distance_policy(distance_decimals), []
    ids = _unique_ids(reports[0]["sonnet_ids"], "analysis")
    if len(ids) < 7:
        raise ValueError("At least seven distinct sonnets are required")
    if len({(r["model_id"], r["revision"]) for r in reports}) != len(reports):
        raise ValueError("Repeated model revisions are not independent models")
    record_ids = _unique_ids([record["sonnet_id"] for record in records], "corpus")
    baseline_ids = _unique_ids(baselines["sonnet_ids"], "baselines")
    by_id = dict(zip(record_ids, records))
    if set(ids) != set(baseline_ids) or not set(ids) <= set(by_id):
        raise ValueError("Baseline and corpus identities must match model identities")
    triangle = np.triu_indices(len(ids), 1)
    order = _alignment(baseline_ids, ids, allow_subset=False, label="baselines")
    baseline_alignment = {"source_sonnet_ids": baseline_ids, "alignment_indices": order}
    names, controls = [], []
    for name, baseline in baselines["baselines"].items():
        names.append(name)
        matrix = _cosine_rdm(baseline["cosine_distance"], len(baseline_ids), order, label=f"baseline/{name}")
        matrix, audit = apply_distance_policy(matrix, distance_decimals, label=f"baseline/{name}")
        distance_audits.append(audit)
        controls.append(matrix[triangle])
    for name, values in (("absolute_word_count_difference", [len(by_id[s]["text"].split()) for s in ids]),
                         ("absolute_character_count_difference", [len(by_id[s]["text"]) for s in ids])):
        values = np.asarray(values)
        names.append(name)
        controls.append(np.abs(values[:, None] - values[None, :])[triangle])
    control_matrix = np.column_stack(controls)
    models = []
    aligned = []
    for report in reports:
        source = _unique_ids(report["sonnet_ids"], report["model_id"])
        order = _alignment(source, ids, allow_subset=False, label=report["model_id"])
        settings, matrices = [], {}
        for pool in DEFAULT_POOLINGS:
            for depth in DEFAULT_DEPTHS:
                choice = select_depth(report["pooling"][pool], depth)
                distance = _cosine_rdm(report["pooling"][pool][choice["tensor"]]["cosine_distance"],
                                       len(source), order, label=f"{report['model_id']}/{pool}/{choice['tensor']}")
                distance, audit = apply_distance_policy(distance, distance_decimals,
                    label=f"{(report['model_id'], report['revision'])}/{pool}/depth={depth}/{choice['tensor']}")
                distance_audits.append(audit)
                hierarchy = linkage(squareform(distance), method="average", optimal_ordering=True)
                clusters = {str(k): fcluster(hierarchy, k, criterion="maxclust").tolist() for k in range(2, 7)}
                anomaly = distance.sum(axis=1) / (len(ids) - 1)
                settings.append({"pooling": pool, "requested_depth": depth, "layer": choice,
                                 "clusters": clusters, "actual_cluster_counts": {k: len(set(v)) for k, v in clusters.items()},
                                 "anomaly_scores": anomaly.tolist(),
                                 "two_dimensional_display": classical_coordinates(distance) if depth == 1 else None})
                matrices[(pool, depth)] = distance
        models.append({"model_id": report["model_id"], "revision": report["revision"], "settings": settings,
                       "source_sonnet_ids": source, "alignment_indices": order})
        aligned.append(matrices)
    comparisons = []
    for i, j in itertools.combinations(range(len(models)), 2):
        for a, b in zip(models[i]["settings"], models[j]["settings"]):
            key = a["pooling"], a["requested_depth"]
            comparisons.append({"model_a": models[i]["model_id"], "model_b": models[j]["model_id"],
                                "pooling": key[0], "requested_depth": key[1],
                                "anomaly_rank_spearman": rank_correlation(a["anomaly_scores"], b["anomaly_scores"]),
                                "clustering_adjusted_rand": {k: float(adjusted_rand_score(a["clusters"][k], b["clusters"][k])) for k in a["clusters"]},
                                "nuisance_controls": [*names, "model_a_embedding_distance", "model_b_embedding_distance"],
                                "nuisance_adjusted_rsa": nuisance_adjusted_rsa(
                                    aligned[i][key][triangle], aligned[j][key][triangle],
                                    np.column_stack([control_matrix, aligned[i][(key[0], 0.0)][triangle],
                                                     aligned[j][(key[0], 0.0)][triangle]]))})
    return {"schema_version": "descriptive-structure-v1", "sonnet_ids": ids, "models": models,
            "distance_policy": numeric_policy, "distance_audits": distance_audits,
            "comparisons": comparisons, "nuisance_controls": names, "baseline_alignment": baseline_alignment,
            "clustering_policy": "Average linkage on cosine distance; all k=2..6 reported, no data-selected optimal k; exact ties may yield fewer clusters",
            "inference": "Exploratory descriptive analyses. Nuisance projection is in-sample and does not remove all lexical, positional, tokenizer or training-overlap confounds. No claim of added literary information or authorship."}
