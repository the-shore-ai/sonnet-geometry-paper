"""Isolated-line BERT changes as a descriptive comparator for cumulative models.

This uses previously saved float32 line embeddings. It never fits or reruns a
model and never labels isolated-line context as cumulative-prefix context.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .comparisons import DEFAULT_DEPTHS, DEFAULT_POOLINGS
from .line_comparisons import _aggregate, _coefficient, _summary_profile
from .storage import sha256_file


def isolated_line_profiles(prior_report: dict[str, Any], archive: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    """Map every saved vector to its poem/verse identity and adjacent distance."""
    embeddings = np.asarray(archive["line_embeddings"], dtype=np.float64)
    sonnets, lines = np.asarray(archive["sonnet_numbers"]), np.asarray(archive["line_numbers"])
    if embeddings.ndim != 2 or embeddings.shape[1] < 1 or not np.isfinite(embeddings).all():
        raise ValueError("Expected finite saved line embeddings")
    if sonnets.shape != (len(embeddings),) or lines.shape != sonnets.shape:
        raise ValueError("Identity arrays must match the saved embedding rows")
    if not np.issubdtype(sonnets.dtype, np.integer) or not np.issubdtype(lines.dtype, np.integer):
        raise ValueError("Sonnet and line identity arrays must contain integers")
    positions = {(int(number), int(line)): i for i, (number, line) in enumerate(zip(sonnets, lines))}
    if len(positions) != len(embeddings):
        raise ValueError("Duplicate sonnet/line identity in embedding archive")
    metrics = prior_report["sonnet_metrics"]
    if len({row["sonnet_id"] for row in metrics}) != len(metrics) or len({row["number"] for row in metrics}) != len(metrics):
        raise ValueError("Duplicate sonnet identities in the prior report")
    expected = {(row["number"], line) for row in metrics for line in range(1, row["num_lines"] + 1)}
    if set(positions) != expected:
        raise ValueError("Saved embeddings do not match every expected canonical verse line")
    result = {}
    for row in metrics:
        count = row["num_lines"]
        if count < 3:
            raise ValueError("At least three verse lines are needed for a change profile")
        x = embeddings[[positions[(row["number"], line)] for line in range(1, count + 1)]]
        norms = np.linalg.norm(x, axis=1)
        denominators = norms[:-1] * norms[1:]
        cosines = np.divide(np.sum(x[:-1] * x[1:], axis=1), denominators,
                            out=np.full(count - 1, np.nan), where=denominators > 0)
        distances = 1 - np.clip(cosines, -1, 1)
        result[row["sonnet_id"]] = {"number": row["number"], "line_count": count,
                                     "transition_labels": [[i, i + 1] for i in range(1, count)],
                                     "cosine_distances": [float(value) if np.isfinite(value) else None for value in distances],
                                     "zero_norm_line_numbers": (np.flatnonzero(norms == 0) + 1).tolist()}
    return result


def compare_isolated_line_baseline(reports: Sequence[dict[str, Any]], prior_report: dict[str, Any],
                                   archive: dict[str, np.ndarray], *, sonnet_ids: Sequence[str] | None = None,
                                   depths: Sequence[float] = DEFAULT_DEPTHS,
                                   poolings: Sequence[str] = DEFAULT_POOLINGS) -> dict[str, Any]:
    """Compare per-sonnet ranks of adjacent changes, preserving distinct contexts.

    No p-values or predictive evaluation are performed. Undefined comparisons
    remain explicit. BERT has one named final-layer isolated-line representation;
    it is not assigned a fictitious residual depth or cumulative pooling method.
    """
    if not reports or not depths or not poolings:
        raise ValueError("Require at least one model report, depth, and pooling")
    if len(set(depths)) != len(depths) or len(set(poolings)) != len(poolings):
        raise ValueError("Depth and pooling selections must be unique")
    baseline = isolated_line_profiles(prior_report, archive)
    target = list(baseline) if sonnet_ids is None else list(sonnet_ids)
    if not target or len(set(target)) != len(target) or not set(target).issubset(baseline):
        raise ValueError("Analysis IDs must be a unique nonempty set present in the baseline")
    identities, models, comparisons = [], [], []
    for report in reports:
        identity = (report["model_id"], report["revision"])
        if identity in identities:
            raise ValueError("Duplicate cumulative-model identity")
        identities.append(identity)
        catalog = {item["summary"]["sonnet_id"]: item for item in report["results"]}
        if len(catalog) != len(report["results"]):
            raise ValueError("Duplicate sonnet in execution report")
        missing, extra = set(target) - set(catalog), set(catalog) - set(target)
        if missing or (extra and sonnet_ids is None):
            raise ValueError(f"Sonnet identity mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        policy = catalog[target[0]]["summary"]["pooling_policy"]
        selected_layers = None
        for sonnet in target:
            item = catalog[sonnet]
            summary = item["summary"]
            if (summary["model_id"], summary["revision"]) != identity:
                raise ValueError("Summary identity disagrees with execution report")
            if summary["line_count"] != baseline[sonnet]["line_count"]:
                raise ValueError("Model/baseline canonical line counts differ")
            provenance_hash = item.get("manifest", {}).get("provenance", {}).get("corpus_sha256")
            if provenance_hash != prior_report["corpus_sha256"]:
                raise ValueError("Cumulative model and isolated baseline corpus hashes must match")
            if summary["pooling_policy"] != policy:
                raise ValueError("A cumulative-model report must use one consistent pooling policy")
            layer_names = sorted(summary["layers"])
            if selected_layers is not None and layer_names != selected_layers:
                raise ValueError("A cumulative-model report must use consistent residual layers")
            selected_layers = layer_names
        models.append({"model_id": identity[0], "revision": identity[1],
                       "context": "cumulative poem prefix at each verse boundary",
                       "source_sonnet_ids": list(catalog), "excluded_sonnet_ids": sorted(extra),
                       "pooling_policy": policy})
        for pooling in poolings:
            for depth in depths:
                rows = []
                for sonnet in target:
                    profile = _summary_profile(catalog[sonnet], pooling, depth)
                    isolated = baseline[sonnet]
                    if profile["transition_labels"] != isolated["transition_labels"]:
                        raise ValueError("Canonical adjacent-line identities disagree")
                    row = {"sonnet_id": sonnet, "line_count": isolated["line_count"],
                           "transition_labels": isolated["transition_labels"], "selected_cumulative_layer": profile["selected_layer"],
                           "cumulative_cosine_changes": profile["cosine_distances"],
                           "isolated_bert_cosine_changes": isolated["cosine_distances"], "spearman": None}
                    if profile["unusable_transitions"] or any(value is None for value in isolated["cosine_distances"]):
                        row.update({"status": "undefined", "reason": "unavailable cumulative transition or zero-norm baseline line",
                                    "unusable_cumulative_transitions": profile["unusable_transitions"],
                                    "zero_norm_baseline_lines": isolated["zero_norm_line_numbers"]})
                    else:
                        value = _coefficient(profile["cosine_distances"], isolated["cosine_distances"])
                        if value is None:
                            row.update({"status": "undefined", "reason": "constant change profile"})
                        else:
                            a, b = np.array(profile["cosine_distances"]), np.array(isolated["cosine_distances"])
                            peak_a = [isolated["transition_labels"][i][1] for i in np.flatnonzero(a == a.max())]
                            peak_b = [isolated["transition_labels"][i][1] for i in np.flatnonzero(b == b.max())]
                            row.update({"status": "computed", "spearman": value,
                                        "maximum_change_to_lines_cumulative": peak_a,
                                        "maximum_change_to_lines_isolated": peak_b,
                                        "both_peaks_unique": len(peak_a) == len(peak_b) == 1,
                                        "shared_maximum_to_lines": sorted(set(peak_a) & set(peak_b))})
                    rows.append(row)
                aggregate = _aggregate(rows)
                unique = [row for row in rows if row.get("both_peaks_unique")]
                aggregate["unique_peak_comparison_count"] = len(unique)
                aggregate["same_unique_peak_count"] = sum(bool(row["shared_maximum_to_lines"]) for row in unique)
                comparisons.append({"cumulative_model_id": identity[0], "cumulative_revision": identity[1],
                                    "cumulative_pooling": pooling, "requested_cumulative_depth": float(depth),
                                    "sonnets": rows, "aggregate": aggregate})
    return {"schema_version": "isolated-line-baseline-comparison-v1", "sonnet_ids": target,
            "selection_policy": "explicit subset" if sonnet_ids is not None else "complete matching corpus",
            "cumulative_models": models, "baseline": {"model": prior_report["model"],
                "context": "each verse line independently encoded with bidirectional BERT; no earlier poem context",
                "representation": "last hidden-layer mean excluding CLS/SEP/padding, no L2 normalization",
                "measurement": "cosine distance between adjacent isolated-line embeddings; not the prior angle-sum ratio",
                "corpus_sha256": prior_report["corpus_sha256"],
                "excluded_sonnet_ids": sorted(set(baseline) - set(target))},
            "comparisons": comparisons, "inferential_tests": 0,
            "interpretation": "Descriptive lexical/isolated-context comparator. Rank agreement or disagreement across different contexts does not establish incremental predictive value, literary significance, or a model advantage."}


def load_verified_archive(path: str | Path, prior_report: dict[str, Any]) -> dict[str, np.ndarray]:
    path = Path(path)
    if sha256_file(path) != prior_report["embedding_archive"]["sha256"]:
        raise ValueError("Saved isolated-line embedding checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in ["line_embeddings", "sonnet_numbers", "line_numbers"]}


def compact_baseline_comparison(report: dict[str, Any]) -> dict[str, Any]:
    """Keep per-sonnet scalar coefficients but omit large repeated line arrays."""
    result = {key: value for key, value in report.items() if key != "comparisons"}
    result["comparisons"] = []
    for comparison in report["comparisons"]:
        value = {key: item for key, item in comparison.items() if key != "sonnets"}
        value["sonnet_coefficients"] = [{key: row[key] for key in ["sonnet_id", "line_count", "status", "spearman"]} |
                                         ({"reason": row["reason"]} if "reason" in row else {})
                                         for row in comparison["sonnets"]]
        value["selected_cumulative_layer"] = comparison["sonnets"][0]["selected_cumulative_layer"]
        result["comparisons"].append(value)
    result["detail_level"] = "compact per-sonnet coefficients and aggregates; line arrays omitted"
    return result
