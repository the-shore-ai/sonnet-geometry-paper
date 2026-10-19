"""Exploratory geometry diagnostics, independent of model execution.

Poems are the units. Pairwise edges and shuffle seeds are dependent; these
functions return descriptive measurements, never inferential p-values.
"""
from itertools import combinations

import numpy as np
from scipy.stats import rankdata


def unit_rows(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Finite matrix required")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0):
        raise ValueError("Nonzero rows required")
    return values / norms[:, None]


def upper(matrix):
    return matrix[np.triu_indices(len(matrix), 1)]


def rdm(values):
    u = unit_rows(values)
    distances = np.clip(1 - u @ u.T, 0, 2)
    np.fill_diagonal(distances, 0)
    return distances


def rank_correlation(left, right):
    a, b = [np.asarray(x, dtype=np.float64) for x in (left, right)]
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Matched finite vectors required")
    a, b = [rankdata(np.round(x, 12)) for x in (a, b)]
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return None if denominator <= 1e-20 else float(np.clip(a @ b / denominator, -1, 1))


def partial_rank(left, right, controls):
    """In-sample rank/OLS sensitivity; not partial-Spearman inference."""
    arrays = [np.asarray(x, dtype=np.float64) for x in (left, right, *controls)]
    if any(x.shape != arrays[0].shape or not np.isfinite(x).all() for x in arrays):
        raise ValueError("Matched finite outcomes and controls required")
    ranked = [rankdata(np.round(x, 12)) for x in arrays]
    design = np.column_stack([np.ones(len(arrays[0])), *ranked[2:]])
    residuals = [x - design @ np.linalg.lstsq(design, x, rcond=None)[0] for x in ranked[:2]]
    denominator = np.linalg.norm(residuals[0]) * np.linalg.norm(residuals[1])
    reference = np.linalg.norm(ranked[0] - np.mean(ranked[0])) * np.linalg.norm(ranked[1] - np.mean(ranked[1]))
    if denominator <= max(1e-15, reference * 1e-10):
        return None
    return float(np.clip(residuals[0] @ residuals[1] / denominator, -1, 1))


def center_gram(gram):
    return gram - gram.mean(0)[None, :] - gram.mean(1)[:, None] + gram.mean()


def unbiased_hsic(k, l):
    """U-statistic HSIC with zero diagonals (n > 3); may be negative."""
    k, l = np.asarray(k).copy(), np.asarray(l).copy()
    n = len(k)
    if k.shape != l.shape or k.shape != (n, n) or n <= 3:
        raise ValueError("Matched Gram matrices with n > 3 required")
    def ucenter(g):
        np.fill_diagonal(g, 0)
        sums = g.sum(0)
        centered = g - sums[None] / (n - 2) - sums[:, None] / (n - 2)
        centered += g.sum() / ((n - 1) * (n - 2))
        np.fill_diagonal(centered, 0)
        return centered
    # Explicit U-centering avoids cancellation of large scalar sums in nearly
    # constant cosine Gram matrices (including fixed terminal-token controls).
    return float(np.sum(ucenter(k) * ucenter(l)) / (n * (n - 3)))


def cka_from_grams(k, l):
    a, b = center_gram(k), center_gram(l)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    biased = None if denominator <= 1e-20 else float(np.sum(a * b) / denominator)
    kk, ll = unbiased_hsic(k, k), unbiased_hsic(l, l)
    unbiased = None if min(kk, ll) <= 1e-20 else float(unbiased_hsic(k, l) / np.sqrt(kk * ll))
    return {"unit_gram_biased_cka": biased, "unit_gram_unbiased_cka": unbiased}


def spectrum_from_gram(gram):
    centered = center_gram(gram)
    eigenvalues = np.linalg.eigvalsh((centered + centered.T) / 2)
    # Numerical roundoff is reported rather than reinterpreted as dimensions.
    trace = float(np.trace(centered))
    positive = np.maximum(eigenvalues, 0)
    denominator = float(np.sum(positive ** 2))
    valid = trace > 1e-12 and denominator > 1e-24
    return {"unit_spread": max(0., trace / len(gram)),
            "effective_dimension": float(positive.sum() ** 2 / denominator) if valid else None,
            "first_pc_fraction": float(positive[-1] / positive.sum()) if valid else None,
            "negative_eigenvalue_mass": float(-eigenvalues[eigenvalues < 0].sum()),
            "maximum_centered_rank": len(gram) - 1}


def cloud_summary(values):
    values = np.asarray(values, dtype=np.float64)
    u = unit_rows(values)
    gram = u @ u.T
    norms = np.linalg.norm(values, axis=1)
    distances = upper(np.clip(1 - gram, 0, 2))
    return {**spectrum_from_gram(gram), "mean_pair_distance": float(distances.mean()),
            "sd_pair_distance": float(distances.std()), "median_norm": float(np.median(norms)),
            "norm_cv": float(norms.std() / norms.mean())}


def retrieval(a, b):
    """Symmetric nearest-candidate identity matching, with Q12 fractional ties."""
    a, b = unit_rows(a), unit_rows(b)
    if a.shape != b.shape:
        raise ValueError("Matched matrices required")
    scores = np.round(a @ b.T, 12)
    n = len(scores)
    directions = []
    for matrix in (scores, scores.T):
        hits, ranks = [], []
        for i, row in enumerate(matrix):
            ties = row == row.max()
            hits.append(float(ties[i] / ties.sum()))
            ranks.append(float(np.sum(row > row[i]) + (np.sum(row == row[i]) + 1) / 2))
        directions.append({"top1": float(np.mean(hits)), "mrr": float(np.mean(1 / np.asarray(ranks))),
                           "per_poem_rank": ranks, "per_poem_top1_credit": hits})
    return {"top1": float(np.mean([d["top1"] for d in directions])),
            "mrr": float(np.mean([d["mrr"] for d in directions])),
            "directions": directions, "chance_top1": 1 / n}


def neighbors(a, b, k=5):
    n = len(a)
    if a.shape != b.shape or a.shape != (n, n) or not 0 < k < n:
        raise ValueError("Matched RDMs and 0 < k < n required")
    sets = []
    for d in (a, b):
        rows = []
        for i, row in enumerate(np.round(d, 12)):
            indices = np.delete(np.arange(n), i)
            cutoff = np.partition(row[indices], k - 1)[k - 1]
            rows.append(set(indices[row[indices] <= cutoff]))
        sets.append(rows)
    overlap, adjusted, jaccard = [], [], []
    for x, y in zip(*sets, strict=True):
        common = len(x & y)
        expected = len(x) * len(y) / (n - 1)
        bound = min(len(x), len(y))
        overlap.append(common / np.sqrt(len(x) * len(y)))
        adjusted.append(None if bound <= expected else (common - expected) / (bound - expected))
        jaccard.append(common / len(x | y))
    return {"k": k, "mean_overlap_fraction": float(np.mean(overlap)),
            "mean_chance_adjusted_overlap": float(np.mean(adjusted)) if all(v is not None for v in adjusted) else None,
            "mean_jaccard": float(np.mean(jaccard)),
            "mean_neighbor_set_sizes": [float(np.mean([len(x) for x in rows])) for rows in sets]}


def paired_metrics(original, perturbed):
    a, b = unit_rows(original), unit_rows(perturbed)
    da, db = rdm(a), rdm(b)
    centered = [x - x.mean(0) for x in (a, b)]
    valid = all(np.min(np.linalg.norm(x, axis=1)) > 1e-12 for x in centered)
    displacement = np.clip(1 - np.sum(a * b, axis=1), 0, 2)
    return {"rdm_rho": rank_correlation(upper(da), upper(db)),
            **cka_from_grams(a @ a.T, b @ b.T), "neighbors": neighbors(da, db),
            "identity": retrieval(a, b),
            "condition_centered_identity": retrieval(*centered) if valid else None,
            "per_poem_cosine_displacement": displacement.tolist(),
            "median_cosine_displacement": float(np.median(displacement)),
            "per_poem_log_norm_ratio": np.log(np.linalg.norm(perturbed, axis=1) / np.linalg.norm(original, axis=1)).tolist(),
            "original_cloud": cloud_summary(original), "perturbed_cloud": cloud_summary(perturbed)}


def seed_repeatability(original, perturbations):
    u = unit_rows(original)
    changed = np.stack([unit_rows(x) for x in perturbations])
    rdms = [rdm(x) for x in changed]
    interseed = [rank_correlation(upper(rdms[i]), upper(rdms[j])) for i, j in combinations(range(len(rdms)), 2)]
    centroid = changed.mean(0)
    within = float(np.mean(np.sum((changed - centroid[None]) ** 2, axis=2)))
    between = float(np.mean(np.sum((centroid - centroid.mean(0)) ** 2, axis=1)))
    shifts = centroid - u
    total = float(np.mean(np.sum(shifts ** 2, axis=1)))
    common = float(np.sum(shifts.mean(0) ** 2))
    return {"interseed_rdm_rhos": interseed,
            "centroid_rdm_rho_to_original": rank_correlation(upper(rdm(u)), upper(rdm(centroid))),
            "shuffle_centroid_between_poem_variance": between, "within_poem_seed_variance": within,
            "between_fraction_of_total_shuffle_variance": between / (between + within) if between + within > 1e-20 else None,
            "mean_shift_energy": total, "common_shift_energy": common,
            "common_fraction_of_mean_shift_energy": common / total if total > 1e-20 else None}
