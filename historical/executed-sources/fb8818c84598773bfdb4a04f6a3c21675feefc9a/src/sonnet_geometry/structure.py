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

from .comparisons import DEFAULT_DEPTHS, DEFAULT_POOLINGS, select_depth


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
    correlation = None if np.any(norms < 1e-9) else float(np.clip(
        residuals[:, 0] @ residuals[:, 1] / np.prod(norms), -1, 1))
    return {"partial_rank_correlation": correlation, "design_rank": int(np.linalg.matrix_rank(design)),
            "residual_norms": norms.tolist(), "pair_count": len(a),
            "inference": "Descriptive residual correlation; pairwise observations are dependent, no p-value or incremental-value claim"}


def classical_coordinates(cosine_distance: np.ndarray) -> dict[str, Any]:
    """Classical scaling of unit-vector chord distance, retaining two axes."""
    d = np.asarray(cosine_distance, dtype=float)
    n = len(d)
    if d.shape != (n, n) or not np.isfinite(d).all() or not np.allclose(d, d.T):
        raise ValueError("Expected a finite symmetric cosine-distance matrix")
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


def summarize_structure(reports: list[dict], baselines: dict, records: list[dict]) -> dict:
    if not reports:
        raise ValueError("At least one model is required")
    ids = reports[0]["sonnet_ids"]
    if len(ids) < 7 or len(set(ids)) != len(ids):
        raise ValueError("At least seven distinct sonnets are required")
    if len({(r["model_id"], r["revision"]) for r in reports}) != len(reports):
        raise ValueError("Repeated model revisions are not independent models")
    by_id = {record["sonnet_id"]: record for record in records}
    if set(ids) != set(baselines["sonnet_ids"]) or not set(ids) <= set(by_id):
        raise ValueError("Baseline and corpus identities must match model identities")
    triangle = np.triu_indices(len(ids), 1)
    positions = {value: i for i, value in enumerate(baselines["sonnet_ids"])}
    order = [positions[value] for value in ids]
    names, controls = [], []
    for name, baseline in baselines["baselines"].items():
        names.append(name)
        controls.append(np.asarray(baseline["cosine_distance"])[np.ix_(order, order)][triangle])
    for name, values in (("absolute_word_count_difference", [len(by_id[s]["text"].split()) for s in ids]),
                         ("absolute_character_count_difference", [len(by_id[s]["text"]) for s in ids])):
        values = np.asarray(values)
        names.append(name)
        controls.append(np.abs(values[:, None] - values[None, :])[triangle])
    control_matrix = np.column_stack(controls)
    models = []
    aligned = []
    for report in reports:
        source = report["sonnet_ids"]
        if len(set(source)) != len(source) or set(source) != set(ids):
            raise ValueError("Every model must contain the same unique sonnet identities")
        order = [source.index(value) for value in ids]
        settings, matrices = [], {}
        for pool in DEFAULT_POOLINGS:
            for depth in DEFAULT_DEPTHS:
                choice = select_depth(report["pooling"][pool], depth)
                distance = np.asarray(report["pooling"][pool][choice["tensor"]]["cosine_distance"])[np.ix_(order, order)]
                if not np.isfinite(distance).all() or not np.allclose(distance, distance.T) or not np.allclose(np.diag(distance), 0):
                    raise ValueError("Invalid cosine distances")
                distance = np.maximum((distance + distance.T) / 2, 0)
                np.fill_diagonal(distance, 0)
                hierarchy = linkage(squareform(distance), method="average", optimal_ordering=True)
                clusters = {str(k): fcluster(hierarchy, k, criterion="maxclust").tolist() for k in range(2, 7)}
                anomaly = distance.sum(axis=1) / (len(ids) - 1)
                settings.append({"pooling": pool, "requested_depth": depth, "layer": choice,
                                 "clusters": clusters, "actual_cluster_counts": {k: len(set(v)) for k, v in clusters.items()},
                                 "anomaly_scores": anomaly.tolist(),
                                 "two_dimensional_display": classical_coordinates(distance) if depth == 1 else None})
                matrices[(pool, depth)] = distance
        models.append({"model_id": report["model_id"], "revision": report["revision"], "settings": settings})
        aligned.append(matrices)
    comparisons = []
    for i, j in itertools.combinations(range(len(models)), 2):
        for a, b in zip(models[i]["settings"], models[j]["settings"]):
            key = a["pooling"], a["requested_depth"]
            comparisons.append({"model_a": models[i]["model_id"], "model_b": models[j]["model_id"],
                                "pooling": key[0], "requested_depth": key[1],
                                "anomaly_rank_spearman": rank_correlation(a["anomaly_scores"], b["anomaly_scores"]),
                                "clustering_adjusted_rand": {k: float(adjusted_rand_score(a["clusters"][k], b["clusters"][k])) for k in a["clusters"]},
                                "nuisance_adjusted_rsa": nuisance_adjusted_rsa(aligned[i][key][triangle], aligned[j][key][triangle], control_matrix)})
    return {"schema_version": "descriptive-structure-v1", "sonnet_ids": ids, "models": models,
            "comparisons": comparisons, "nuisance_controls": names,
            "clustering_policy": "Average linkage on cosine distance; all k=2..6 reported, no data-selected optimal k; exact ties may yield fewer clusters",
            "inference": "Exploratory descriptive analyses. Nuisance projection is in-sample and does not remove all lexical, positional, tokenizer or training-overlap confounds. No claim of added literary information or authorship."}
