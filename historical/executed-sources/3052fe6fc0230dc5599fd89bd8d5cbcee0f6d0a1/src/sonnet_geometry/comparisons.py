"""Cross-model RSA from derived RDMs with explicit sample/depth alignment."""
from __future__ import annotations

import hashlib
import itertools
import json
import re
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from .analysis import fdr_adjust, rsa
from .distance_policy import apply_distance_policy, distance_policy


DEFAULT_DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_POOLINGS = ("last_token", "cumulative_mean")


def _unique_ids(value: Any, label: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(x, str) or not x for x in value):
        raise ValueError(f"{label}: expected a sequence of nonempty sonnet IDs")
    ids = list(value)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label}: duplicate sonnet IDs")
    if len(ids) < 3:
        raise ValueError(f"{label}: at least three sonnets are required for RSA")
    return ids


def _alignment(source: list[str], target: list[str], *, allow_subset: bool, label: str) -> list[int]:
    missing = sorted(set(target) - set(source))
    extra = sorted(set(source) - set(target))
    if missing or (extra and not allow_subset):
        raise ValueError(f"{label}: sonnet identity mismatch; missing={missing}, extra={extra}. "
                         "Declare sonnet_ids explicitly to analyze a subset.")
    positions = {sonnet_id: i for i, sonnet_id in enumerate(source)}
    return [positions[sonnet_id] for sonnet_id in target]


def _rdm(value: Any, n: int, order: Sequence[int], *, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (n, n) or not np.isfinite(matrix).all():
        raise ValueError(f"{label}: expected a finite {n} by {n} distance matrix")
    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{label}: distance matrix must be symmetric")
    if not np.allclose(np.diag(matrix), 0, atol=1e-7, rtol=0):
        raise ValueError(f"{label}: distance matrix diagonal must be zero")
    return matrix[np.ix_(order, order)]


def residual_depths(layers: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Embedding is depth 0; block i output is (i+1)/number_of_blocks."""
    if "residual.embedding" not in layers:
        raise ValueError("Residual embedding must be present")
    blocks = sorted((int(match.group(1)), name) for name in layers
                    if (match := re.fullmatch(r"residual\.block\.(\d+)", name)))
    if not blocks or [i for i, _ in blocks] != list(range(len(blocks))):
        raise ValueError("Residual blocks must be contiguous, starting at zero")
    return [{"tensor": "residual.embedding", "residual_index": 0, "normalized_depth": 0.0}] + [
        {"tensor": name, "residual_index": i + 1, "normalized_depth": (i + 1) / len(blocks)}
        for i, name in blocks]


def select_depth(layers: Mapping[str, Any], target: float) -> dict[str, Any]:
    if not np.isfinite(target) or not 0 <= target <= 1:
        raise ValueError("Normalized depth must lie in [0, 1]")
    available = residual_depths(layers)
    # Floating division can make mathematical ties differ by an ulp.
    nearest = min(abs(layer["normalized_depth"] - target) for layer in available)
    selected = next(layer for layer in available
                    if np.isclose(abs(layer["normalized_depth"] - target), nearest, rtol=0, atol=1e-12))
    return {"requested_depth": float(target), **selected,
            "distance_to_requested_depth": abs(selected["normalized_depth"] - target)}


def _rank_rdm(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize upper-triangle ranks once; label permutations preserve ranks."""
    triangle = np.triu_indices(len(matrix), 1)
    ranks = rankdata(matrix[triangle], method="average")
    ranks -= ranks.mean()
    norm = np.linalg.norm(ranks)
    if norm == 0:
        raise ValueError("RSA is undefined for a constant distance matrix")
    ranks /= norm
    ranked_matrix = np.zeros_like(matrix)
    ranked_matrix[triangle] = ranks
    ranked_matrix[(triangle[1], triangle[0])] = ranks
    return ranks, ranked_matrix


def _label_permutations(n: int, count: int, seed: int,
                        blocks: Sequence[Any] | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permutations = np.empty((count, n), dtype=np.int64)
    for i in range(count):
        if blocks is None:
            permutations[i] = rng.permutation(n)
        else:
            permutations[i] = np.arange(n)
            for label in dict.fromkeys(blocks):
                indices = np.array([j for j, block in enumerate(blocks) if block == label], dtype=int)
                permutations[i, indices] = rng.permutation(indices)
    return permutations


def _rsa_permutations(a: np.ndarray, b: np.ndarray, permutations: np.ndarray) -> dict[str, Any]:
    ranks_a, _ = _rank_rdm(a)
    ranks_b, ranked_b = _rank_rdm(b)
    triangle = np.triu_indices(len(a), 1)
    observed = float(np.clip(ranks_a @ ranks_b, -1, 1))
    null = np.empty(len(permutations))
    # Simultaneously relabel rows and columns. Never shuffle independent edges.
    for i, labels in enumerate(permutations):
        permuted_ranks = ranked_b[labels[triangle[0]], labels[triangle[1]]]
        null[i] = np.clip(ranks_a @ permuted_ranks, -1, 1)
    return {"rsa_spearman": observed,
            "p_value": float((1 + np.count_nonzero(null >= observed - 1e-12)) / (len(null) + 1)),
            "null_mean": float(null.mean()), "null_standard_deviation": float(null.std()),
            "null_quantiles": dict(zip(["0.025", "0.5", "0.975"], np.quantile(null, [0.025, 0.5, 0.975]).tolist())),
            "null_values": null.tolist()}


def compare_models(reports: Sequence[dict[str, Any]], *, baseline_report: dict[str, Any] | None = None,
                   sonnet_ids: Sequence[str] | None = None, depths: Sequence[float] = DEFAULT_DEPTHS,
                   poolings: Sequence[str] = DEFAULT_POOLINGS, permutations: int = 999, seed: int = 999,
                   exchangeability_blocks: Mapping[str, Any] | None = None,
                   distance_decimals: int | None = None) -> dict[str, Any]:
    """Align exact identities, test every requested model pair/depth/pooling.

    By default every model and baseline must contain the same unique sonnet set.
    Explicit sonnet_ids permits a declared subset; all omitted IDs are recorded.
    BY correction covers one family containing every requested cross-model test,
    including undefined/constant-RDM cases conservatively entered as p=1.
    Baseline RSA is descriptive and contributes no inferential hypotheses.
    """
    if len(reports) < 1 or (len(reports) < 2 and baseline_report is None):
        raise ValueError("Require two model reports, or one model report with conventional baselines")
    numeric_policy, distance_audits = distance_policy(distance_decimals), []
    if not isinstance(permutations, int) or permutations < 1:
        raise ValueError("permutations must be a positive integer")
    requested_depths = list(depths)
    if not requested_depths or len(set(requested_depths)) != len(requested_depths):
        raise ValueError("Requested depths must be nonempty and unique")
    if any(not np.isfinite(depth) or not 0 <= depth <= 1 for depth in requested_depths):
        raise ValueError("Normalized depths must lie in [0, 1]")
    requested_poolings = list(poolings)
    if not requested_poolings or len(set(requested_poolings)) != len(requested_poolings):
        raise ValueError("Requested poolings must be nonempty and unique")
    target = _unique_ids(sonnet_ids if sonnet_ids is not None else reports[0]["sonnet_ids"], "analysis")
    identities = [(report["model_id"], report["revision"]) for report in reports]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate model/revision reports are not independent model pairs")
    if exchangeability_blocks is not None and set(exchangeability_blocks) != set(target):
        raise ValueError("Exchangeability blocks must identify exactly the analyzed sonnets")
    blocks = None if exchangeability_blocks is None else [exchangeability_blocks[sonnet] for sonnet in target]
    label_permutations = _label_permutations(len(target), permutations, seed, blocks)
    aligned: list[dict[str, Any]] = []
    selected_metadata = []
    for report, identity in zip(reports, identities):
        source = _unique_ids(report["sonnet_ids"], str(identity))
        order = _alignment(source, target, allow_subset=sonnet_ids is not None, label=str(identity))
        choices, distances = {}, {}
        for pooling in requested_poolings:
            if pooling not in report["pooling"]:
                raise ValueError(f"{identity}: missing pooling {pooling}")
            layers = report["pooling"][pooling]
            choices[pooling], distances[pooling] = [], []
            for depth in requested_depths:
                choice = select_depth(layers, depth)
                choices[pooling].append(choice)
                label = f"{identity}/{pooling}/depth={depth}/{choice['tensor']}"
                matrix = _rdm(layers[choice["tensor"]]["cosine_distance"], len(source), order, label=label)
                matrix, audit = apply_distance_policy(matrix, distance_decimals, label=label)
                distances[pooling].append(matrix)
                distance_audits.append(audit)
        aligned.append(distances)
        selected_metadata.append({"model_id": identity[0], "revision": identity[1], "source_sonnet_ids": source,
                                  "alignment_indices": order, "excluded_sonnet_ids": [x for x in source if x not in target],
                                  "selected_layers": choices, "source_run_ids": report.get("run_ids", [])})
    tests = []
    for i, j in itertools.combinations(range(len(reports)), 2):
        for pooling in requested_poolings:
            for depth_index, depth in enumerate(requested_depths):
                result = {"model_a": identities[i][0], "revision_a": identities[i][1],
                          "model_b": identities[j][0], "revision_b": identities[j][1],
                          "pooling": pooling, "requested_depth": depth,
                          "layer_a": selected_metadata[i]["selected_layers"][pooling][depth_index],
                          "layer_b": selected_metadata[j]["selected_layers"][pooling][depth_index]}
                try:
                    result.update(_rsa_permutations(aligned[i][pooling][depth_index], aligned[j][pooling][depth_index], label_permutations))
                    result["status"] = "computed"
                except ValueError as exc:
                    result.update({"status": "undefined", "reason": str(exc), "rsa_spearman": None,
                                   "p_value": None, "fdr_input_p_value": 1.0})
                tests.append(result)
    adjusted = fdr_adjust([test["p_value"] if test["p_value"] is not None else 1.0 for test in tests], method="by")
    for test, adjusted_p in zip(tests, adjusted):
        test["p_value_by"] = float(adjusted_p)
        test["fdr_input_p_value"] = test["p_value"] if test["p_value"] is not None else 1.0
    baseline_comparisons = []
    baseline_alignment = None
    if baseline_report is not None:
        source = _unique_ids(baseline_report["sonnet_ids"], "baselines")
        order = _alignment(source, target, allow_subset=sonnet_ids is not None, label="baselines")
        baseline_alignment = {"source_sonnet_ids": source, "alignment_indices": order,
                              "excluded_sonnet_ids": [x for x in source if x not in target],
                              "corpus_sha256": baseline_report.get("corpus_sha256"),
                              "lock_sha256": baseline_report.get("lock_sha256")}
        for name, baseline in baseline_report["baselines"].items():
            baseline_rdm = _rdm(baseline["cosine_distance"], len(source), order, label=f"baseline/{name}")
            baseline_rdm, audit = apply_distance_policy(baseline_rdm, distance_decimals, label=f"baseline/{name}")
            distance_audits.append(audit)
            for model_index, identity in enumerate(identities):
                for pooling in requested_poolings:
                    for depth_index, depth in enumerate(requested_depths):
                        item = {"baseline": name, "model_id": identity[0], "revision": identity[1],
                                "pooling": pooling, "requested_depth": depth,
                                "layer": selected_metadata[model_index]["selected_layers"][pooling][depth_index],
                                "inference": "Descriptive RSA only; no incremental-value or significance claim"}
                        try:
                            item["rsa_spearman"] = rsa(aligned[model_index][pooling][depth_index], baseline_rdm)
                            item["status"] = "computed"
                        except ValueError as exc:
                            item.update({"rsa_spearman": None, "status": "undefined", "reason": str(exc)})
                        baseline_comparisons.append(item)
    return {"schema_version": "cross-model-comparisons-v1", "sonnet_ids": target,
            "distance_policy": numeric_policy, "distance_audits": distance_audits,
            "selection_policy": "explicit subset" if sonnet_ids is not None else "identical complete sonnet sets",
            "depth_policy": "Nearest normalized residual depth: embedding=0; block i=(i+1)/L; ties choose shallower. Final normalized output excluded. Depth is not functional homology.",
            "requested_depths": requested_depths, "poolings": requested_poolings,
            "models": selected_metadata,
            "permutation": {"count": permutations, "seed": seed, "alternative": "greater",
                            "unit": "whole sonnet labels; simultaneous rows and columns of one model RDM",
                            "shared_across_comparison_family": True,
                            "exchangeability": "unrestricted sonnet labels" if blocks is None else "within explicit blocks",
                            "blocks": None if exchangeability_blocks is None else dict(exchangeability_blocks),
                            "indices_sha256": hashlib.sha256(json.dumps(label_permutations.tolist(), separators=(",", ":")).encode()).hexdigest(),
                            "minimum_p_value": 1 / (permutations + 1)},
            "multiple_comparisons": {"method": "Benjamini-Yekutieli", "family_size": len(tests),
                                     "family": "all requested model-pairs × pooling methods × normalized depth targets",
                                     "undefined_tests": "Retained in family with conservative p=1",
                                     "duplicate_nearest_layers": "Separate prespecified depth targets remain in family even if nearest layers coincide"},
            "cross_model_tests": tests, "baseline_alignment": baseline_alignment,
            "baseline_comparisons": baseline_comparisons,
            "interpretation": "Permutation agreement concerns sample correspondence under stated exchangeability; it does not establish literary structure, independence of pretraining data, or value beyond baselines."}
