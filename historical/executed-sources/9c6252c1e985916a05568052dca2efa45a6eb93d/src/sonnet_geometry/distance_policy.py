"""Explicit optional numerical tie sensitivity for derived cosine distances."""
from __future__ import annotations

from typing import Any

import numpy as np


def distance_policy(decimals: int | None) -> dict[str, Any]:
    if decimals is not None and (type(decimals) is not int or not 0 <= decimals <= 15):
        raise ValueError("distance_decimals must be None or an integer from 0 through 15")
    return {"decimals": decimals,
            "operation": "none" if decimals is None else "numpy.round applied uniformly before ranking or downstream distance summaries",
            "scope": "all selected model cosine RDMs and all conventional baseline cosine RDMs; integer length controls unchanged",
            "rank_ties": "average ranks",
            "interpretation": "Numerical sensitivity resolution, not a proof of exact mathematical ties or literary validity"}


def apply_distance_policy(matrix: np.ndarray, decimals: int | None, *, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    distance_policy(decimals)
    original = np.asarray(matrix, dtype=np.float64)
    result = original if decimals is None else np.round(original, decimals=decimals)
    triangle = np.triu_indices(len(original), 1)
    before, after = original[triangle], result[triangle]
    def counts(values):
        _, sizes = np.unique(values, return_counts=True)
        return {"unique_distances": len(sizes), "largest_exact_tie_group": int(sizes.max()) if len(sizes) else 0,
                "zero_distances": int(np.count_nonzero(values == 0))}
    return result, {"label": label, "upper_triangle_pairs": len(before),
                    "changed_distances": int(np.count_nonzero(before != after)),
                    "maximum_absolute_change": float(np.max(np.abs(before - after))) if len(before) else 0.0,
                    "before": counts(before), "after": counts(after)}
