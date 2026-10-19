"""Post hoc descriptive position checks on saved adjacent-prefix changes."""
from __future__ import annotations

import itertools
from typing import Any, Sequence

import numpy as np

from .comparisons import DEFAULT_DEPTHS, DEFAULT_POOLINGS
from .line_comparisons import _aggregate, _coefficient, _summary_profile


IDS = [f"sonnet-{number:03d}" for number in range(1, 155)]
EXCEPTIONS = {
    "sonnet-099": "15 lines; its opening stanza has five lines, so conventional 14-line entry positions do not align",
    "sonnet-126": "12 lines arranged as six couplets; conventional three-quatrain-plus-couplet positions do not apply",
}
STANDARD_IDS = [sonnet for sonnet in IDS if sonnet not in EXCEPTIONS]


def describe(values: Sequence[float] | np.ndarray) -> dict[str, Any] | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    if not np.isfinite(values).all():
        raise ValueError("Descriptive values must be finite")
    return {"count": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "standard_deviation": float(values.std()), "minimum": float(values.min()),
            "maximum": float(values.max()), "quartiles": np.quantile(values, [.25, .75]).tolist()}


def peaks(profile: dict[str, Any]) -> list[int]:
    if profile["unusable_transitions"]:
        return []
    values = np.asarray(profile["cosine_distances"])
    return [profile["transition_labels"][i][1] for i in np.flatnonzero(values == values.max())]


def peak_distribution(profiles: dict[str, dict[str, Any]], sonnet_ids: Sequence[str]) -> dict[str, Any]:
    incidence, fractional, unique = {}, {}, {}
    final_count = final_unique = second_count = second_unique = valid = 0
    final_fraction = second_fraction = 0.0
    undefined = []
    for sonnet in sonnet_ids:
        profile = profiles[sonnet]
        maxima = peaks(profile)
        if not maxima:
            undefined.append(sonnet)
            continue
        valid += 1
        final = profile["transition_labels"][-1][1]
        for position in maxima:
            key = str(position)
            incidence[key] = incidence.get(key, 0) + 1
            fractional[key] = fractional.get(key, 0.0) + 1 / len(maxima)
            if len(maxima) == 1:
                unique[key] = unique.get(key, 0) + 1
        final_count += final in maxima
        second_count += 2 in maxima
        final_fraction += (final in maxima) / len(maxima)
        second_fraction += (2 in maxima) / len(maxima)
        final_unique += maxima == [final]
        second_unique += maxima == [2]
    return {"total_sonnets": len(sonnet_ids), "usable_sonnets": valid, "unusable_sonnet_ids": undefined,
            "maximum_incidence_by_to_line": dict(sorted(incidence.items(), key=lambda item: int(item[0]))),
            "fractional_maximum_mass_by_to_line": dict(sorted(fractional.items(), key=lambda item: int(item[0]))),
            "unique_maximum_count_by_to_line": dict(sorted(unique.items(), key=lambda item: int(item[0]))),
            "final_line_maximum_count": final_count, "final_line_unique_maximum_count": final_unique,
            "final_line_fractional_maximum_mass": final_fraction,
            "line_2_maximum_count": second_count, "line_2_unique_maximum_count": second_unique,
            "line_2_fractional_maximum_mass": second_fraction,
            "ties": "All exact maxima count in incidence; fractional mass assigns each usable poem total weight one; unique counts exclude ties"}


def individual_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {"selected_layer": profile["selected_layer"], "transition_to_lines": [x[1] for x in profile["transition_labels"]],
            "cosine_changes": profile["cosine_distances"], "unusable_transitions": profile["unusable_transitions"],
            "maximum_to_lines": peaks(profile)}


def center_positions(matrix: np.ndarray) -> np.ndarray:
    """Remove in-sample position means without ranking float64 subtraction noise."""
    centered = matrix - matrix.mean(axis=0)
    tolerance = 16 * np.finfo(np.float64).eps * np.max(np.abs(matrix))
    centered[np.abs(centered) <= tolerance] = 0
    for row in centered:
        if np.ptp(row) <= 2 * tolerance:
            row[:] = row.mean()
    return centered


def analyze_line_positions(reports: Sequence[dict[str, Any]], *, depths: Sequence[float] = DEFAULT_DEPTHS,
                           poolings: Sequence[str] = DEFAULT_POOLINGS) -> dict[str, Any]:
    """Summarize mechanics and compare profiles after in-sample position centering.

    Every model must retain all 154 canonical poem IDs and expected verse counts.
    Position centering is computed only for complete 152-poem standard-length
    matrices. It is an exploratory, fitted-on-the-same-corpus description.
    """
    if not reports or not depths or not poolings or len(set(depths)) != len(depths) or len(set(poolings)) != len(poolings):
        raise ValueError("Require reports and unique nonempty depth/pooling selections")
    identities = [(report["model_id"], report["revision"]) for report in reports]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate model/revision report")
    catalogs, source_hashes = [], {}
    reference_policy = None
    for report in reports:
        catalog = {item["summary"]["sonnet_id"]: item for item in report["results"]}
        if len(catalog) != len(report["results"]) or set(catalog) != set(IDS):
            raise ValueError("Require all 154 canonical sonnets exactly once per model")
        first_policy = catalog[IDS[0]]["summary"]["pooling_policy"]
        policy_keys = ("special_tokens", "textual_separator_tokens", "prefix_endpoint")
        if reference_policy is not None and any(first_policy.get(key) != reference_policy.get(key) for key in policy_keys):
            raise ValueError("Cross-model prefix or token-inclusion policy mismatch")
        reference_policy = first_policy
        for sonnet in IDS:
            item = catalog[sonnet]
            summary = item["summary"]
            if (summary["model_id"], summary["revision"]) != (report["model_id"], report["revision"]):
                raise ValueError("Summary model identity disagrees with execution report")
            expected_count = 15 if sonnet == "sonnet-099" else 12 if sonnet == "sonnet-126" else 14
            if summary["line_count"] != expected_count:
                raise ValueError(f"Incorrect canonical verse count for {sonnet}")
            if summary["pooling_policy"] != first_policy:
                raise ValueError("One model report must use a consistent prefix/pooling policy")
            digest = item.get("manifest", {}).get("request", {}).get("record_sha256")
            if not digest or (sonnet in source_hashes and source_hashes[sonnet] != digest):
                raise ValueError(f"Missing or mismatched source record checksum for {sonnet}")
            source_hashes[sonnet] = digest
        catalogs.append(catalog)
    model_outputs, prepared = [], []
    for identity, catalog in zip(identities, catalogs):
        settings, matrices = [], {}
        for pool in poolings:
            for depth in depths:
                profiles = {sonnet: _summary_profile(catalog[sonnet], pool, depth) for sonnet in IDS}
                choices = [profiles[sonnet]["selected_layer"] for sonnet in IDS]
                if any(choice != choices[0] for choice in choices[1:]):
                    raise ValueError("Residual-depth selections must be consistent within a model report")
                valid = [sonnet for sonnet in STANDARD_IDS if not profiles[sonnet]["unusable_transitions"]]
                matrix = np.asarray([profiles[sonnet]["cosine_distances"] for sonnet in valid], dtype=np.float64)
                entry_columns, other_columns = [3, 7, 11], [i for i in range(13) if i not in {3, 7, 11}]
                entry = {"standard_form_sonnets": 152, "usable_sonnets": len(valid),
                         "unusable_sonnet_ids": [sonnet for sonnet in STANDARD_IDS if sonnet not in valid],
                         "entry_to_lines": [5, 9, 13], "other_to_lines": [i + 2 for i in other_columns],
                         "positions": {str(line): describe(matrix[:, line - 2]) if len(valid) else None for line in [5, 9, 13]},
                         "mean_entry_change_per_poem": describe(matrix[:, entry_columns].mean(axis=1)) if len(valid) else None,
                         "mean_other_change_per_poem": describe(matrix[:, other_columns].mean(axis=1)) if len(valid) else None,
                         "entry_minus_other_per_poem": describe(matrix[:, entry_columns].mean(axis=1) - matrix[:, other_columns].mean(axis=1)) if len(valid) else None,
                         "weighting": "One summary per poem; each poem's entry mean weights three positions equally and other mean weights ten positions equally"}
                position_profile = {"transition_to_lines": list(range(2, 15)), "usable_sonnets": len(valid),
                                    "mean_cosine_change": matrix.mean(axis=0).tolist() if len(valid) else None,
                                    "median_cosine_change": np.median(matrix, axis=0).tolist() if len(valid) else None}
                settings.append({"pooling": pool, "requested_depth": float(depth), "selected_layer": choices[0],
                                 "all_sonnet_peak_distribution": peak_distribution(profiles, IDS),
                                 "standard_form_peak_distribution": peak_distribution(profiles, STANDARD_IDS),
                                 "standard_form_position_profile": position_profile, "conventional_entries": entry,
                                 "sonnet_130": individual_profile(profiles["sonnet-130"]),
                                 "form_exceptions": {sonnet: {"exclusion_reason": reason, **individual_profile(profiles[sonnet])}
                                                     for sonnet, reason in EXCEPTIONS.items()}})
                matrices[(pool, depth)] = matrix if len(valid) == 152 else None
        model_outputs.append({"model_id": identity[0], "revision": identity[1],
                              "pooling_policy": catalog[IDS[0]]["summary"]["pooling_policy"], "settings": settings})
        prepared.append(matrices)
    comparisons = []
    for i, j in itertools.combinations(range(len(reports)), 2):
        for pool in poolings:
            for depth in depths:
                left, right = prepared[i][(pool, depth)], prepared[j][(pool, depth)]
                comparison = {"model_a": identities[i][0], "revision_a": identities[i][1],
                              "model_b": identities[j][0], "revision_b": identities[j][1],
                              "pooling": pool, "requested_depth": float(depth)}
                if left is None or right is None:
                    comparison.update({"status": "undefined", "reason": "At least one standard-form profile is incomplete; position centering requires all 152 poems"})
                else:
                    comparison["status"] = "computed"
                    for label, a, b in [("raw", left, right), ("position_centered", center_positions(left), center_positions(right))]:
                        rows = []
                        for sonnet, av, bv in zip(STANDARD_IDS, a, b):
                            value = _coefficient(av, bv)
                            rows.append({"sonnet_id": sonnet, "spearman": value, "status": "computed" if value is not None else "undefined",
                                         **({"reason": "constant change profile"} if value is None else {})})
                        comparison[label] = _aggregate(rows)
                comparisons.append(comparison)
    return {"schema_version": "line-position-description-v1", "status": "post hoc descriptive mechanics check",
            "sonnet_ids": IDS, "standard_form_sonnet_ids": STANDARD_IDS,
            "form_exclusions": EXCEPTIONS,
            "standard_form_note": "152 poems have fourteen lines; this includes tetrameter Sonnet 145 and does not assert metrical uniformity",
            "models": model_outputs, "position_centered_cross_model_agreement": comparisons, "inferential_tests": 0,
            "centering": "For each model/depth/pooling, subtract the across-152-poem mean at each transition position, then compute within-poem Spearman agreement. Fitted on the same corpus; no held-out or causal interpretation. Negative centered values are deviations, not negative cosine distances. Values within 16 float64 eps times the setting's maximum absolute original change are zeroed; row ranges within twice that tolerance are constant to avoid ranking subtraction roundoff.",
            "interpretation": "Position, cumulative length, tokenization, punctuation and endpoint-type differences are possible confounds. Newline-inclusive boundaries may merge preceding punctuation with the separator; earlier endpoint token IDs are not assumed equal. A final endpoint lacks a following separator. Cumulative averaging changes the relative weight of additions. These summaries do not isolate any causal effect, test a volta, or establish literary meaning.",
            "source_record_sha256": source_hashes}
