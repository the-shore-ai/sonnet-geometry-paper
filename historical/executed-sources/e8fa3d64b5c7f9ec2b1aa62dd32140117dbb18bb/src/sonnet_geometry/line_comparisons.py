"""Descriptive cross-model comparisons at matched cumulative verse boundaries.

Adjacent-transition profiles from saved summaries and within-poem prefix RDMs
are different artifacts. Neither is a between-sonnet whole-poem RDM.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import rankdata

from .analysis import cosine_similarity_matrix, prefix_representations, rsa
from .comparisons import DEFAULT_DEPTHS, DEFAULT_POOLINGS, select_depth
from .storage import sha256_file, verify_run


def _coefficient(x: Sequence[float], y: Sequence[float]) -> float | None:
    left, right = rankdata(x), rankdata(y)
    left, right = left - left.mean(), right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return None if denominator == 0 else float(np.clip(left @ right / denominator, -1, 1))


def _catalog(reports: Sequence[dict[str, Any]], sonnet_ids: Sequence[str] | None) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(reports) < 2:
        raise ValueError("At least two model reports are required")
    catalogs, metadata, model_keys = [], [], []
    for report in reports:
        key = (report["model_id"], report["revision"])
        if key in model_keys:
            raise ValueError("Duplicate model/revision reports")
        model_keys.append(key)
        catalog = {}
        for item in report["results"]:
            summary = item["summary"]
            sonnet_id = summary["sonnet_id"]
            if sonnet_id in catalog:
                raise ValueError("Duplicate sonnet identity in summaries")
            if (summary["model_id"], summary["revision"]) != key:
                raise ValueError("Summary model/revision disagrees with batch")
            catalog[sonnet_id] = item
        catalogs.append(catalog)
        metadata.append({"model_id": key[0], "revision": key[1], "source_sonnet_ids": list(catalog)})
    target = list(catalogs[0]) if sonnet_ids is None else list(sonnet_ids)
    if not target or len(set(target)) != len(target):
        raise ValueError("Analysis sonnet IDs must be nonempty and unique")
    for catalog, model in zip(catalogs, metadata):
        missing, extra = set(target) - set(catalog), set(catalog) - set(target)
        if missing or (extra and sonnet_ids is None):
            raise ValueError(f"Sonnet identity mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
        model["excluded_sonnet_ids"] = sorted(extra)
    return target, catalogs, metadata


def _summary_profile(item: dict[str, Any], pooling: str, depth: float) -> dict[str, Any]:
    summary = item["summary"]
    choice = select_depth(summary["layers"], depth)
    transitions = summary["layers"][choice["tensor"]]["pooling"][pooling]["lines"]["transitions"]
    expected = [(i, i + 1) for i in range(1, summary["line_count"])]
    observed = [(row["from_line"], row["to_line"]) for row in transitions]
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise ValueError("A summary must account for each canonical adjacent-line transition exactly once")
    by_label = dict(zip(observed, transitions))
    values, unusable = [], []
    for label in expected:
        row = by_label[label]
        value = row["cosine_distance"]
        if not row["usable"] or value is None or not np.isfinite(value):
            values.append(None)
            unusable.append(list(label))
        else:
            values.append(float(value))
    return {"selected_layer": choice, "transition_labels": [list(label) for label in expected],
            "cosine_distances": values, "unusable_transitions": unusable}


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["status"] == "computed"]
    scores = np.array([row["spearman"] for row in valid])
    result = {"total_sonnets": len(rows), "computed_sonnets": len(valid),
              "undefined_sonnets": [{"sonnet_id": row["sonnet_id"], "reason": row["reason"]}
                                    for row in rows if row["status"] != "computed"],
              "weighting": "one equal-weight coefficient per sonnet; no independent-edge or independent-line inference"}
    result["coefficient_summary"] = None if not len(scores) else {
        "mean": float(scores.mean()), "median": float(np.median(scores)),
        "minimum": float(scores.min()), "maximum": float(scores.max()),
        "quartiles": np.quantile(scores, [0.25, 0.75]).tolist()}
    return result


def compare_transition_profiles(reports: Sequence[dict[str, Any]], *, sonnet_ids: Sequence[str] | None = None,
                                depths: Sequence[float] = DEFAULT_DEPTHS,
                                poolings: Sequence[str] = DEFAULT_POOLINGS) -> dict[str, Any]:
    """Compare ranks of corresponding adjacent-line changes within each poem.

    All poems and transition identities must align. An unusable boundary marks
    that poem/depth/pooling comparison unusable; boundaries are never dropped or
    bridged. Constant profiles are undefined. No significance tests are supplied.
    """
    if not depths or len(set(depths)) != len(depths) or not poolings or len(set(poolings)) != len(poolings):
        raise ValueError("Depth and pooling selections must be nonempty and unique")
    target, catalogs, metadata = _catalog(reports, sonnet_ids)
    prepared = []
    for catalog in catalogs:
        model = {}
        for sonnet_id in target:
            item = catalog[sonnet_id]
            summary = item["summary"]
            if summary["line_count"] < 3:
                raise ValueError("At least three verse lines are needed for transition profiles")
            model[sonnet_id] = {pool: [_summary_profile(item, pool, depth) for depth in depths] for pool in poolings}
        prepared.append(model)
    for sonnet_id in target:
        summaries = [catalog[sonnet_id]["summary"] for catalog in catalogs]
        if len({summary["line_count"] for summary in summaries}) != 1:
            raise ValueError(f"Line-count mismatch for {sonnet_id}")
        policies = [summary["pooling_policy"] for summary in summaries]
        keys = ("special_tokens", "textual_separator_tokens", "prefix_endpoint")
        if any(any(policy.get(key) != policies[0].get(key) for key in keys) for policy in policies[1:]):
            raise ValueError(f"Prefix or token-inclusion policy mismatch for {sonnet_id}")
        hashes = [catalog[sonnet_id].get("manifest", {}).get("request", {}).get("record_sha256") for catalog in catalogs]
        present = [digest for digest in hashes if digest is not None]
        if present and (len(present) != len(hashes) or len(set(present)) != 1):
            raise ValueError(f"Source-record hash mismatch or partial provenance for {sonnet_id}")
    comparisons = []
    for i, j in itertools.combinations(range(len(reports)), 2):
        for pooling in poolings:
            for depth_index, depth in enumerate(depths):
                rows = []
                for sonnet_id in target:
                    a, b = prepared[i][sonnet_id][pooling][depth_index], prepared[j][sonnet_id][pooling][depth_index]
                    if a["transition_labels"] != b["transition_labels"]:
                        raise ValueError("Canonical transition labels disagree")
                    row = {"sonnet_id": sonnet_id, "line_count": len(a["transition_labels"]) + 1,
                           "transition_labels": a["transition_labels"], "layer_a": a["selected_layer"],
                           "layer_b": b["selected_layer"], "cosine_changes_a": a["cosine_distances"],
                           "cosine_changes_b": b["cosine_distances"], "spearman": None}
                    if a["unusable_transitions"] or b["unusable_transitions"]:
                        row.update({"status": "undefined", "reason": "unusable canonical prefix transition",
                                    "unusable_a": a["unusable_transitions"], "unusable_b": b["unusable_transitions"]})
                    else:
                        coefficient = _coefficient(a["cosine_distances"], b["cosine_distances"])
                        if coefficient is None:
                            row.update({"status": "undefined", "reason": "constant transition profile"})
                        else:
                            left, right = np.array(a["cosine_distances"]), np.array(b["cosine_distances"])
                            peaks_a = [a["transition_labels"][k][1] for k in np.flatnonzero(left == left.max())]
                            peaks_b = [b["transition_labels"][k][1] for k in np.flatnonzero(right == right.max())]
                            row.update({"status": "computed", "spearman": coefficient,
                                        "maximum_change_to_lines_a": peaks_a, "maximum_change_to_lines_b": peaks_b,
                                        "both_peaks_unique": len(peaks_a) == len(peaks_b) == 1,
                                        "shared_maximum_to_lines": sorted(set(peaks_a) & set(peaks_b))})
                    rows.append(row)
                aggregate = _aggregate(rows)
                unique = [row for row in rows if row.get("both_peaks_unique")]
                aggregate["unique_peak_comparison_count"] = len(unique)
                aggregate["same_unique_peak_count"] = sum(bool(row["shared_maximum_to_lines"]) for row in unique)
                comparisons.append({"model_a": metadata[i]["model_id"], "revision_a": metadata[i]["revision"],
                                    "model_b": metadata[j]["model_id"], "revision_b": metadata[j]["revision"],
                                    "pooling": pooling, "requested_depth": float(depth), "sonnets": rows,
                                    "aggregate": aggregate})
    return {"schema_version": "matched-line-transitions-v1", "analysis_type": "within-poem adjacent-line change-profile agreement",
            "sonnet_ids": target, "models": metadata, "depths": list(depths), "poolings": list(poolings),
            "selection_policy": "explicit subset" if sonnet_ids is not None else "identical complete sonnet sets",
            "comparisons": comparisons, "inferential_tests": 0,
            "interpretation": "Descriptive same-line rank agreement. Position, cumulative length, terminal tokens, and within-poem dependence are uncontrolled. No p-values, authorship inference, or claim of literary significance.",
            "not_computed": "Within-poem line RDMs cannot be reconstructed from adjacent distances or whole-poem pooled vectors."}


def transition_aggregates(report: dict[str, Any]) -> dict[str, Any]:
    """Compact review artifact; retain explicit undefined counts/identities.

    Per-sonnet coefficients and line measurements remain reproducible from the
    source batch reports. This does not hide undefined poems from aggregates.
    """
    if report.get("schema_version") != "matched-line-transitions-v1":
        raise ValueError("Expected a matched line-transition report")
    result = {key: value for key, value in report.items() if key != "comparisons"}
    result["comparisons"] = [{key: value for key, value in comparison.items() if key != "sonnets"}
                             for comparison in report["comparisons"]]
    result["detail_level"] = "aggregate only; raw profiles and per-sonnet coefficients omitted from this compact view"
    return result


def derive_line_geometry(run_dir: str | Path, *, depths: Sequence[float] = DEFAULT_DEPTHS,
                         poolings: Sequence[str] = DEFAULT_POOLINGS) -> dict[str, Any]:
    """Derive small within-poem cumulative-prefix RDMs from local verified raw data.

    This CPU-only operation needs no model weights. It is available for future
    near-storage derivation; current GPU extraction does not call this function.
    """
    from safetensors import safe_open

    path = Path(run_dir)
    manifest = verify_run(path)
    record = json.loads((path / "record.json").read_text())
    tokens = json.loads((path / "tokens.json").read_text())
    canonical_lines = [span["line_number"] for span in record["line_spans"]]
    checks = {check["line_number"]: check for check in tokens["prefix_checks"]}
    if sorted(checks) != canonical_lines or len(checks) != len(tokens["prefix_checks"]):
        raise ValueError("Prefix checks must cover every canonical line")
    usable = [line for line in canonical_lines if checks[line]["exact_token_prefix"]]
    ends = [checks[line]["full_pass_token_end"] for line in usable]
    for line, end in zip(usable, ends):
        if end is None or tokens["input_ids"][:end] != checks[line]["prefix_input_ids"]:
            raise ValueError("Inconsistent exact-prefix token IDs")
    mask = np.logical_not(tokens["special_tokens_mask"])
    choices = [select_depth(manifest["tensors"], depth) for depth in depths]
    result = {"schema_version": "within-poem-line-rdm-v1", "model_id": manifest["model_id"],
              "revision": manifest["revision"], "run_id": manifest["run_id"], "sonnet_id": manifest["sonnet_id"],
              "number": record["number"], "text_sha256": record.get("text_sha256"),
              "source_manifest_sha256": sha256_file(path / "manifest.json"),
              "canonical_line_numbers": canonical_lines, "usable_line_numbers": usable,
              "unusable_line_numbers": [line for line in canonical_lines if line not in usable],
              "boundary_policy": tokens.get("boundary_policy", "legacy_line_content_end"),
              "pooling_policy": "exclude special tokens; include textual separators; cumulative prefixes",
              "depths": list(depths), "pooling": {pool: [] for pool in poolings}}
    with safe_open(path / "activations.safetensors", framework="pt", device="cpu") as handle:
        for choice in choices:
            states = handle.get_tensor(choice["tensor"]).double().numpy()
            for pooling in poolings:
                row = {"selected_layer": choice, "line_numbers": usable, "cosine_distance": None}
                if len(usable) < 3:
                    row.update({"status": "undefined", "reason": "fewer than three usable cumulative prefixes"})
                else:
                    prefixes = prefix_representations(states[None], ends, pooling=pooling, include_tokens=mask)[0]
                    try:
                        matrix = 1 - cosine_similarity_matrix(prefixes)
                        np.fill_diagonal(matrix, 0)
                        row.update({"status": "computed", "cosine_distance": matrix.tolist()})
                    except ValueError as exc:
                        row.update({"status": "undefined", "reason": str(exc)})
                result["pooling"][pooling].append(row)
    return result


def compare_line_rdms(groups: Sequence[Sequence[dict[str, Any]]]) -> dict[str, Any]:
    """Compare actual within-poem RDM artifacts; require every canonical prefix.

    Missing/invalid boundaries remain an explicit undefined comparison, rather
    than silently changing which literary lines are compared across models.
    """
    if len(groups) < 2 or any(not group for group in groups):
        raise ValueError("At least two nonempty model groups are required")
    catalogs, identities = [], []
    for group in groups:
        identity = (group[0]["model_id"], group[0]["revision"])
        if identity in identities:
            raise ValueError("Duplicate model groups")
        if any((record["model_id"], record["revision"]) != identity for record in group):
            raise ValueError("Mixed model identities within a line-RDM group")
        catalog = {record["sonnet_id"]: record for record in group}
        if len(catalog) != len(group):
            raise ValueError("Duplicate sonnets in line-RDM group")
        catalogs.append(catalog)
        identities.append(identity)
    target = list(catalogs[0])
    if any(set(catalog) != set(target) for catalog in catalogs):
        raise ValueError("Sonnet identity mismatch between line-RDM groups")
    comparisons = []
    for i, j in itertools.combinations(range(len(groups)), 2):
        for sonnet in target:
            a, b = catalogs[i][sonnet], catalogs[j][sonnet]
            for key in ["canonical_line_numbers", "boundary_policy", "pooling_policy", "depths", "text_sha256"]:
                if a[key] != b[key]:
                    raise ValueError(f"Line-RDM {key} mismatch for {sonnet}")
            if set(a["pooling"]) != set(b["pooling"]):
                raise ValueError("Line-RDM pooling mismatch")
            for pooling in a["pooling"]:
                if len(a["pooling"][pooling]) != len(a["depths"]) or len(b["pooling"][pooling]) != len(b["depths"]):
                    raise ValueError("Line-RDM depth count mismatch")
                for depth, left, right in zip(a["depths"], a["pooling"][pooling], b["pooling"][pooling]):
                    row = {"model_a": identities[i][0], "revision_a": identities[i][1],
                           "model_b": identities[j][0], "revision_b": identities[j][1], "sonnet_id": sonnet,
                           "pooling": pooling, "requested_depth": depth, "spearman": None,
                           "layer_a": left["selected_layer"], "layer_b": right["selected_layer"]}
                    if left["line_numbers"] != a["canonical_line_numbers"] or right["line_numbers"] != b["canonical_line_numbers"]:
                        row.update({"status": "undefined", "reason": "at least one canonical line prefix is unavailable"})
                    elif left["status"] != "computed" or right["status"] != "computed":
                        row.update({"status": "undefined", "reason": "source line geometry is undefined"})
                    else:
                        expected_shape = (len(a["canonical_line_numbers"]), len(a["canonical_line_numbers"]))
                        if np.asarray(left["cosine_distance"]).shape != expected_shape or np.asarray(right["cosine_distance"]).shape != expected_shape:
                            raise ValueError("Line-RDM matrix dimensions disagree with canonical line identities")
                        try:
                            row.update({"status": "computed", "spearman": rsa(left["cosine_distance"], right["cosine_distance"])})
                        except ValueError as exc:
                            row.update({"status": "undefined", "reason": str(exc)})
                    comparisons.append(row)
    keys = sorted(set((row["model_a"], row["revision_a"], row["model_b"], row["revision_b"], row["pooling"], row["requested_depth"])
                      for row in comparisons))
    aggregates = []
    for key in keys:
        rows = [row for row in comparisons if tuple(row[field] for field in
                ["model_a", "revision_a", "model_b", "revision_b", "pooling", "requested_depth"]) == key]
        aggregates.append({"model_a": key[0], "revision_a": key[1], "model_b": key[2], "revision_b": key[3],
                           "pooling": key[4], "requested_depth": key[5], **_aggregate(rows)})
    return {"schema_version": "within-poem-line-rdm-comparison-v1", "sonnet_ids": target,
            "analysis_type": "within-poem cumulative-prefix RDM agreement", "comparisons": comparisons,
            "aggregates": aggregates, "inferential_tests": 0,
            "interpretation": "Descriptive within-poem geometry. Shared positions and increasing cumulative context are not controlled; no literary significance claim."}


def plan_line_derivation(batch_report: dict[str, Any]) -> dict[str, Any]:
    """List generation-pinned GCS objects required for future verified CPU work.

    No network request is made. Existing verify_run checks all recorded files,
    so the plan includes the full immutable checkpoint instead of only tensors.
    """
    runs = []
    for result in batch_report["results"]:
        receipt = result["storage_receipt"]
        objects = [{"uri": f"gs://{receipt['bucket']}/{item['name']}", "generation": item["generation"],
                    "sha256": item["sha256"], "bytes": item["bytes"]} for item in receipt["objects"]]
        runs.append({"sonnet_id": result["summary"]["sonnet_id"], "objects": objects,
                     "total_bytes": sum(item["bytes"] for item in objects)})
    return {"schema_version": "line-derivation-plan-v1", "model_id": batch_report["model_id"],
            "revision": batch_report["revision"], "runs": runs,
            "total_download_bytes": sum(run["total_bytes"] for run in runs),
            "execution": "Plan only. Derive on CPU near private GCS, one verified poem at a time; no model weights or GPU needed.",
            "download_verification": "Pin each download to listed generation and verify SHA256; run storage.verify_run on the complete local checkpoint before derive_line_geometry."}
