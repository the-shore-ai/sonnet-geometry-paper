"""Explicit pooling, descriptive geometry, and dependency-aware permutation tests.

These functions do not interpret geometric changes as literary significance.
Inputs are NumPy arrays; statistical sampling units are complete sonnets.
"""
from __future__ import annotations

from typing import Any, Sequence
import re

import numpy as np


def _finite_matrix(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("Expected a finite 2D matrix")
    return result


def cosine_similarity_matrix(representations: Any) -> np.ndarray:
    x = _finite_matrix(representations)
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0):
        raise ValueError("Cosine similarity is undefined for zero vectors")
    unit = x / norms[:, None]
    return np.clip(unit @ unit.T, -1, 1)


def prefix_representations(hidden: Any, token_ends: Sequence[int], *,
                           pooling: str, include_tokens: Sequence[bool] | None = None) -> np.ndarray:
    """Return [layer, prefix, feature] from [layer, token, feature].

    token_ends are exclusive, strictly increasing, exactly validated prefix
    boundaries. The caller must check tokenization equivalence first. Last-token
    selects the last included token; mean pools all included prefix tokens.
    """
    h = np.asarray(hidden, dtype=np.float64)
    if h.ndim != 3 or not np.isfinite(h).all():
        raise ValueError("hidden must be finite [layer, token, feature]")
    ends = list(token_ends)
    if not ends or any(not isinstance(e, (int, np.integer)) for e in ends) or any(
            e <= 0 or e > h.shape[1] for e in ends) or any(b <= a for a, b in zip(ends, ends[1:])):
        raise ValueError("token_ends must be valid increasing exclusive indices")
    mask = np.ones(h.shape[1], dtype=bool) if include_tokens is None else np.asarray(include_tokens, dtype=bool)
    if mask.shape != (h.shape[1],):
        raise ValueError("include_tokens must match the token axis")
    pooled = []
    for end in ends:
        selected = np.flatnonzero(mask[:end])
        if not len(selected):
            raise ValueError("A prefix has no included tokens")
        if pooling == "last_token":
            pooled.append(h[:, selected[-1], :])
        elif pooling == "cumulative_mean":
            pooled.append(h[:, selected, :].mean(axis=1))
        else:
            raise ValueError("pooling must be last_token or cumulative_mean")
    return np.stack(pooled, axis=1)


def trajectory_metrics(representations: Any) -> dict[str, np.ndarray]:
    """Return transitions [layer, boundary-1] and descriptive norms/angles.

    Zero-length vectors/steps make cosine/turn angles undefined (NaN), not zero.
    Turn angle is between consecutive displacement vectors, in radians.
    """
    x = np.asarray(representations, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite [layer, >=2 boundaries, feature]")
    displacement = np.diff(x, axis=1)
    norms = np.linalg.norm(x, axis=-1)
    distance = np.linalg.norm(displacement, axis=-1)
    denominator = norms[:, :-1] * norms[:, 1:]
    cosine = np.divide(np.sum(x[:, :-1] * x[:, 1:], axis=-1), denominator,
                       out=np.full_like(denominator, np.nan), where=denominator > 0)
    step_denominator = distance[:, :-1] * distance[:, 1:]
    step_cos = np.divide(np.sum(displacement[:, :-1] * displacement[:, 1:], axis=-1), step_denominator,
                         out=np.full_like(step_denominator, np.nan), where=step_denominator > 0)
    return {"cosine_distance": 1 - np.clip(cosine, -1, 1), "euclidean_distance": distance,
            "vector_norm": norms, "norm_change": np.diff(norms, axis=1),
            "path_length": distance.sum(axis=1), "turn_angle_radians": np.arccos(np.clip(step_cos, -1, 1))}


def whole_sonnet_metrics(representations: Any) -> dict[str, np.ndarray]:
    """Descriptive cosine RDM and mean distance anomaly score (no p-values)."""
    similarity = cosine_similarity_matrix(representations)
    if len(similarity) < 2:
        raise ValueError("At least two sonnets are required")
    distance = 1 - similarity
    np.fill_diagonal(distance, 0)
    return {"cosine_similarity": similarity, "cosine_distance": distance,
            "mean_distance_anomaly": distance.sum(axis=1) / (len(distance) - 1)}


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    while position < len(values):
        stop = position + 1
        while stop < len(values) and values[order[stop]] == values[order[position]]:
            stop += 1
        ranks[order[position:stop]] = (position + stop - 1) / 2
        position = stop
    return ranks


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator == 0:
        raise ValueError("Correlation is undefined for constant data")
    return float(np.clip(np.dot(x, y) / denominator, -1, 1))


def rsa(rdm_a: Any, rdm_b: Any, *, method: str = "spearman") -> float:
    """Correlate upper triangles descriptively; edges are not independent."""
    a, b = _finite_matrix(rdm_a), _finite_matrix(rdm_b)
    if a.shape != b.shape or a.shape[0] != a.shape[1] or len(a) < 3:
        raise ValueError("RDMs must be equally sized square matrices of >=3 sonnets")
    if not np.allclose(a, a.T) or not np.allclose(b, b.T):
        raise ValueError("RDMs must be symmetric")
    triangle = np.triu_indices(len(a), k=1)
    x, y = a[triangle], b[triangle]
    if method == "spearman":
        x, y = _ranks(x), _ranks(y)
    elif method != "pearson":
        raise ValueError("method must be spearman or pearson")
    return _correlation(x, y)


def linear_cka(features_a: Any, features_b: Any) -> float:
    """Biased centered linear CKA; feature widths may differ, samples must match.

    CKA can be inflated in high-dimensional small-sample settings. Calibrate
    with sample-label permutations and baselines; do not call this unbiased.
    """
    x, y = _finite_matrix(features_a), _finite_matrix(features_b)
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("CKA needs the same >=3 ordered sonnets")
    x, y = x - x.mean(axis=0), y - y.mean(axis=0)
    gram_x, gram_y = x @ x.T, y @ y.T
    denominator = np.linalg.norm(gram_x) * np.linalg.norm(gram_y)
    if denominator == 0:
        raise ValueError("CKA is undefined for constant representations")
    return float(np.sum(gram_x * gram_y) / denominator)


def _permutation(rng: np.random.Generator, n: int, blocks: Sequence[Any] | None) -> np.ndarray:
    if blocks is None:
        return rng.permutation(n)
    if len(blocks) != n:
        raise ValueError("Exchangeability blocks must match sonnet count")
    result = np.arange(n)
    # Preserve first occurrence order without sorting mixed-type labels.
    for label in dict.fromkeys(blocks):
        indices = np.array([i for i, block in enumerate(blocks) if block == label])
        result[indices] = rng.permutation(indices)
    return result


def permutation_agreement(a: Any, b: Any, *, statistic: str = "rsa", permutations: int = 999,
                          seed: int = 0, blocks: Sequence[Any] | None = None) -> dict[str, Any]:
    """One-sided greater-agreement null using complete sonnet-label permutations.

    For RSA both matrix axes are permuted together, preserving dependence of
    distances sharing a sonnet. Blocks can condition on preregistered length or
    edition strata. Exchangeability remains a scientific assumption. This does
    not independently shuffle edges, tokens, lines, or layers.
    """
    if permutations < 1:
        raise ValueError("At least one permutation is required")
    a, b = _finite_matrix(a), _finite_matrix(b)
    if statistic == "rsa":
        observed = rsa(a, b)
    elif statistic == "linear_cka":
        observed = linear_cka(a, b)
    else:
        raise ValueError("statistic must be rsa or linear_cka")
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for i in range(permutations):
        labels = _permutation(rng, len(a), blocks)
        null[i] = rsa(a, b[np.ix_(labels, labels)]) if statistic == "rsa" else linear_cka(a, b[labels])
    return {"statistic": statistic, "observed": observed,
            "p_value": float((1 + np.count_nonzero(null >= observed - 1e-12)) / (permutations + 1)),
            "alternative": "greater", "permutations": permutations, "seed": seed,
            "null_values": null, "null_mean": float(null.mean()),
            "exchangeability": "within supplied blocks" if blocks is not None else "all sonnet labels"}


def fdr_adjust(p_values: Any, *, method: str = "by") -> np.ndarray:
    """Benjamini–Yekutieli (default, arbitrary dependence) or BH adjusted p-values.

    Pass the complete prespecified comparison family, including all layers and
    metrics searched. BH requires independence/appropriate positive dependence.
    """
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must be a finite vector in [0,1]")
    if method not in {"bh", "by"}:
        raise ValueError("method must be bh or by")
    if not len(p):
        return p.copy()
    order = np.argsort(p)
    factor = np.sum(1 / np.arange(1, len(p) + 1)) if method == "by" else 1
    adjusted_sorted = np.minimum.accumulate((p[order] * len(p) * factor / np.arange(1, len(p) + 1))[::-1])[::-1]
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(adjusted_sorted, 1)
    return adjusted


def stylometric_features(texts: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Small transparent baseline; fit scaling/PCA only within training folds."""
    names = ["character_count", "word_count", "line_count", "mean_word_characters",
             "type_token_ratio", "punctuation_per_character", "apostrophe_per_word", "mean_line_characters"]
    rows = []
    for text in texts:
        words = re.findall(r"\b\w+(?:[’']\w+)*\b", text.lower())
        lines = text.splitlines()
        if not words or not lines:
            raise ValueError("Text must contain words")
        rows.append([len(text), len(words), len(lines), np.mean([len(w) for w in words]),
                     len(set(words)) / len(words), sum(c in ",.;:!?—-" for c in text) / len(text),
                     sum(c in "’'" for c in text) / len(words), np.mean([len(line) for line in lines])])
    return np.asarray(rows, dtype=np.float64), names
