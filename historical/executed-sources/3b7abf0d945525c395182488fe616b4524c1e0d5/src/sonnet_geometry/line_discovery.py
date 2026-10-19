"""Exploratory contextual-line reconstruction and poem-level cross-model checks.

Only saved sufficient statistics and immutable token metadata are used. Added
line means describe token states from the same full causal forward pass.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy.stats import rankdata

from .analysis import fdr_adjust
from .comparisons import DEFAULT_DEPTHS, select_depth
from .line_positions import EXCEPTIONS, IDS, STANDARD_IDS
from .storage import canonical_json, sha256_file


METRICS = ("cumulative_cosine_change", "line_context_cosine_distance", "log_line_to_prefix_norm_ratio")
SEED = 20261018
RANK_DECIMALS = 12


def reconstruct_line(previous_norm: float, current_norm: float, cosine_distance: float,
                     previous_count: int, current_count: int, euclidean_distance: float | None = None) -> dict[str, Any]:
    """Recover ||L|| and cos(L,A) from B=(N*A+n*L)/(N+n).

    Uses a nonnegative sum of a squared radial difference and angular term,
    avoiding subtraction of two large squared cumulative norms.
    """
    a, b, delta = map(float, (previous_norm, current_norm, cosine_distance))
    if not np.isfinite([a, b, delta]).all() or min(a, b) <= 0 or not 0 <= delta <= 2:
        return {"usable": False, "reason": "nonfinite/zero norm or invalid cosine distance"}
    if type(previous_count) is not int or type(current_count) is not int or not 0 < previous_count < current_count:
        raise ValueError("Included-token counts must be positive, integer, and strictly increasing")
    n, total, previous = current_count - previous_count, current_count, previous_count
    radial = total * b - previous * a
    numerator_squared = radial**2 + 2 * total * previous * a * b * delta
    scale = (total * b + previous * a)**2
    condition = scale / numerator_squared if numerator_squared > 0 else None
    error_bound = 64 * np.finfo(np.float64).eps * condition if condition is not None else None
    if numerator_squared <= 0 or error_bound is None or error_bound > 1e-8:
        return {"usable": False, "reason": "zero reconstructed line norm or cancellation bound exceeds 1e-8",
                "condition_proxy": condition, "roundoff_bound_proxy": error_bound}
    numerator_norm = np.sqrt(numerator_squared)
    line_norm = numerator_norm / n
    cosine = (radial - total * b * delta) / numerator_norm
    naive = total**2 * b**2 + previous**2 * a**2 - 2 * total * previous * a * b * (1 - delta)
    naive_disagreement = abs(naive - numerator_squared) / numerator_squared
    euclidean_disagreement = None
    if euclidean_distance is not None:
        observed_squared = float(euclidean_distance)**2
        expected_squared = (a - b)**2 + 2 * a * b * delta
        euclidean_disagreement = abs(observed_squared - expected_squared) / max(a*a + b*b + observed_squared, np.finfo(float).tiny)
    if abs(cosine) > 1 + 1e-8 or (euclidean_disagreement is not None and euclidean_disagreement > 1e-10):
        return {"usable": False, "reason": "inconsistent saved sufficient statistics",
                "cosine_before_clipping": float(cosine), "euclidean_relative_disagreement": euclidean_disagreement}
    return {"usable": True, "line_norm": float(line_norm), "line_context_cosine_distance": float(1 - np.clip(cosine, -1, 1)),
            "log_line_to_prefix_norm_ratio": float(np.log(line_norm / a)),
            "condition_proxy": float(condition), "roundoff_bound_proxy": float(error_bound),
            "naive_relative_disagreement": float(naive_disagreement),
            "euclidean_relative_disagreement": euclidean_disagreement}


def prefix_metadata(tokens: dict, line_count: int, summary: dict) -> list[dict]:
    ids, mask = tokens["input_ids"], tokens["special_tokens_mask"]
    if len(ids) != len(mask) or len(ids) != summary["token_count"] or any(x not in (0, 1) for x in mask):
        raise ValueError("Token IDs/special mask disagree with saved summary")
    if tokens["boundary_policy"] != "include_following_line_separator":
        raise ValueError("Only the declared newline-inclusive boundary policy is supported")
    checks = {row["line_number"]: row for row in tokens["prefix_checks"]}
    if len(checks) != line_count or len(tokens["prefix_checks"]) != line_count or set(checks) != set(range(1, line_count + 1)):
        raise ValueError("Prefix metadata must identify every canonical line exactly once")
    result, previous = [], 0
    for line in range(1, line_count + 1):
        check = checks[line]
        end = check["full_pass_token_end"]
        if not check["exact_token_prefix"] or check["prefix_input_ids"] != ids[:end]:
            raise ValueError("Unverified or inexact cumulative token prefix")
        included = [index for index in range(end) if not mask[index]]
        if len(included) <= previous:
            raise ValueError("Every added verse must contribute included tokens")
        last = included[-1]
        result.append({"line_number": line, "included_tokens": len(included), "added_tokens": len(included) - previous,
                       "endpoint_token_id": ids[last], "endpoint_text_slice": tokens["tokens"][last]["text_slice"]})
        previous = len(included)
    if previous != summary["included_token_count"] or previous != sum(not x for x in mask):
        raise ValueError("Final included-token count disagrees with saved summary")
    return result


def prepare_model(report: dict, records: list[dict], token_directory: Path, depths=DEFAULT_DEPTHS) -> dict:
    corpus = {record["sonnet_id"]: record for record in records}
    items = {item["summary"]["sonnet_id"]: item for item in report["results"]}
    if len(corpus) != len(records) or len(items) != len(report["results"]) or set(items) != set(IDS) or set(corpus) != set(IDS):
        raise ValueError("Require the complete 154-sonnet corpus exactly once")
    identity = (report["model_id"], report["revision"])
    poems, audits = {}, []
    policy = items[IDS[0]]["summary"]["pooling_policy"]
    for sonnet in IDS:
        record, item = corpus[sonnet], items[sonnet]
        summary, manifest = item["summary"], item["manifest"]
        if (summary["model_id"], summary["revision"]) != identity or summary["pooling_policy"] != policy:
            raise ValueError("Inconsistent model identity or pooling policy")
        if hashlib.sha256(canonical_json(record)).hexdigest() != manifest["request"]["record_sha256"]:
            raise ValueError("Canonical record differs from the extracted source")
        path = token_directory / f"{sonnet}.json"
        digest = sha256_file(path)
        artifact = manifest["artifacts"]["tokens.json"]
        if digest != artifact["sha256"] or path.stat().st_size != artifact["bytes"]:
            raise ValueError("Token metadata does not match the immutable extraction manifest")
        count = len(record["lines"])
        if summary["line_count"] != count:
            raise ValueError("Canonical line count mismatch")
        metadata = prefix_metadata(json.loads(path.read_text()), count, summary)
        settings = []
        for depth in depths:
            layer = select_depth(summary["layers"], depth)
            saved = summary["layers"][layer["tensor"]]["pooling"]["cumulative_mean"]["lines"]
            boundaries = {row["ending_line_number"]: row for row in saved["boundaries"]}
            transitions = {(row["from_line"], row["to_line"]): row for row in saved["transitions"]}
            if (len(boundaries) != len(saved["boundaries"]) or len(transitions) != len(saved["transitions"])
                    or set(boundaries) != set(range(1, count + 1))
                    or set(transitions) != {(line - 1, line) for line in range(2, count + 1)}):
                raise ValueError("Saved line boundaries or transitions are not canonical")
            values = {metric: [] for metric in METRICS}
            reconstructions, norms = [], []
            for line in range(2, count + 1):
                a, b, change = boundaries[line - 1], boundaries[line], transitions[(line - 1, line)]
                if not a["usable"] or not b["usable"] or not change["usable"]:
                    reconstruction = {"usable": False, "reason": "unusable saved cumulative boundary"}
                else:
                    reconstruction = reconstruct_line(a["vector_norm"], b["vector_norm"], change["cosine_distance"],
                        metadata[line - 2]["included_tokens"], metadata[line - 1]["included_tokens"], change["euclidean_distance"])
                reconstructions.append(reconstruction)
                norms.append([a["vector_norm"], b["vector_norm"]])
                values[METRICS[0]].append(change["cosine_distance"] if change["usable"] else None)
                for metric in METRICS[1:]:
                    values[metric].append(reconstruction.get(metric))
            settings.append({"requested_depth": float(depth), "layer": layer, "values": values,
                             "prefix_norms": norms, "reconstruction_audits": reconstructions})
        poems[sonnet] = {"line_count": count, "token_prefixes": metadata, "settings": settings}
        audits.append({"sonnet_id": sonnet, "tokens_sha256": digest, "record_sha256": manifest["request"]["record_sha256"]})
    return {"model_id": identity[0], "revision": identity[1], "pooling_policy": policy, "poems": poems,
            "token_directory": str(token_directory), "source_verification": audits}


def profile_ranks(matrix: np.ndarray, *, residual: bool = False) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    result = np.full(matrix.shape, np.nan)
    for i, row in enumerate(matrix):
        if not np.isfinite(row).all() or (residual and np.ptp(row) <= 1e-9):
            continue
        ranks = rankdata(np.round(row, RANK_DECIMALS), method="average")
        ranks -= ranks.mean()
        norm = np.linalg.norm(ranks)
        if norm > 0:
            result[i] = ranks / norm
    return result


def poem_folds(ids: list[str], seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    folds = np.empty(len(ids), dtype=int)
    for fold, indices in enumerate(np.array_split(order, 5)):
        folds[indices] = fold
    if "sonnet-153" in ids and "sonnet-154" in ids:
        folds[ids.index("sonnet-154")] = folds[ids.index("sonnet-153")]
    return folds


def control_inputs(model: dict, records: dict[str, dict], depth_index: int, ids=STANDARD_IDS) -> tuple[np.ndarray, np.ndarray]:
    continuous, categorical = [], []
    punctuation = ".,;:!?"
    word_pattern = re.compile(r"\b\w+(?:[’']\w+)*\b")
    for sonnet in ids:
        poem, record = model["poems"][sonnet], records[sonnet]
        numeric_rows, category_rows = [], []
        for i in range(1, len(record["lines"])):
            previous, current = poem["token_prefixes"][i - 1:i + 1]
            a, b = poem["settings"][depth_index]["prefix_norms"][i - 1]
            texts = [record["lines"][i - 1], record["lines"][i]]
            numeric = [np.log(current["added_tokens"]), np.log(previous["included_tokens"]),
                       len(texts[1]), len(word_pattern.findall(texts[1])), len(texts[0]), len(word_pattern.findall(texts[0])),
                       np.log(a), np.log(b)]
            for value in texts:
                suffix = value.rstrip()[-2:]
                numeric.extend(float(mark in suffix) for mark in punctuation)
                numeric.append(float(any(mark in suffix for mark in "’'\"")))
            numeric_rows.append(numeric)
            category_rows.append([str(previous["endpoint_token_id"]), str(current["endpoint_token_id"])])
        continuous.append(numeric_rows)
        categorical.append(category_rows)
    return np.asarray(continuous, dtype=float), np.asarray(categorical, dtype=str)


def crossfit_nuisance(target: np.ndarray, continuous: np.ndarray, categories: np.ndarray,
                      embedding_control: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """OLS controls with all fitting/standardization/category selection on training poems."""
    target, continuous, embedding_control = map(np.asarray, (target, continuous, embedding_control))
    if target.shape != embedding_control.shape or continuous.shape[:2] != target.shape or categories.shape[:2] != target.shape:
        raise ValueError("Cross-fit inputs must align by poem and canonical transition")
    if not all(np.isfinite(array).all() for array in (target, continuous, embedding_control)):
        raise ValueError("Cross-fitting requires complete finite profiles")
    n, transitions = target.shape
    if folds.shape != (n,) or len(set(folds)) < 2:
        raise ValueError("Cross-fitting requires at least two poem folds")
    numeric = np.concatenate([continuous, embedding_control[..., None]], axis=2)
    predictions, audits = np.full_like(target, np.nan), []
    for fold in sorted(set(folds)):
        training, held = folds != fold, folds == fold
        if not training.any() or not held.any():
            raise ValueError("A fold cannot have an empty training or held-out set")
        train_flat = numeric[training].reshape(-1, numeric.shape[-1])
        mean, scale = train_flat.mean(axis=0), train_flat.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        columns = [np.ones((n, transitions, 1)), (numeric - mean) / scale,
                   np.broadcast_to(np.eye(transitions)[:, 1:], (n, transitions, transitions - 1))]
        categorical_audit = []
        for column in range(categories.shape[-1]):
            counts = Counter(categories[training, :, column].ravel())
            vocabulary = sorted(value for value, count in counts.items() if count >= 10)
            # Rare and unseen values share an explicit fallback; one category is
            # omitted because the design already includes an intercept.
            levels = [*vocabulary, "__OTHER__"]
            mapped = np.where(np.isin(categories[:, :, column], vocabulary), categories[:, :, column], "__OTHER__")
            for level in levels[1:]:
                columns.append((mapped == level)[..., None].astype(float))
            categorical_audit.append({"column": column, "training_levels_at_least_10": vocabulary,
                                      "fallback": "__OTHER__", "reference_level": levels[0],
                                      "held_out_fallback_count": int(np.count_nonzero(mapped[held] == "__OTHER__"))})
        design = np.concatenate(columns, axis=2)
        x_train = design[training].reshape(-1, design.shape[-1])
        beta, _, rank, singular = np.linalg.lstsq(x_train, target[training].ravel(), rcond=1e-10)
        predictions[held] = design[held] @ beta
        audits.append({"fold": int(fold), "training_poems": int(training.sum()), "held_out_poems": int(held.sum()),
                       "design_columns": design.shape[-1], "design_rank": int(rank),
                       "training_means": mean.tolist(), "training_scales": scale.tolist(), "coefficients": beta.tolist(),
                       "smallest_singular_value": float(singular[-1]), "categorical_encoding": categorical_audit})
    return target - predictions, predictions, audits


def profile_permutations(ids: list[str], folds: np.ndarray, original_blocks: dict[str, str], count: int, seed: int) -> tuple[np.ndarray, dict]:
    if type(count) is not int or count < 1 or set(ids) - set(original_blocks):
        raise ValueError("Require positive permutation count and complete original block mapping")
    labels = [f"fold{fold}|{original_blocks[sonnet]}" if sonnet not in {"sonnet-153", "sonnet-154"}
              else f"fixed-near-duplicate|{sonnet}" for sonnet, fold in zip(ids, folds)]
    groups = [np.flatnonzero(np.asarray(labels) == label) for label in dict.fromkeys(labels)]
    rng = np.random.default_rng(seed)
    permutations = np.tile(np.arange(len(ids)), (count, 1))
    for row in permutations:
        for group in groups:
            row[group] = rng.permutation(group)
    return permutations, {"blocks": dict(zip(ids, labels)), "group_sizes": {label: labels.count(label) for label in dict.fromkeys(labels)},
                          "movable_poems": sum(len(group) for group in groups if len(group) > 1),
                          "fixed_sonnet_ids": [ids[group[0]] for group in groups if len(group) == 1],
                          "indices_sha256": hashlib.sha256(permutations.astype("<i8").tobytes()).hexdigest()}


def permutation_profile_agreement(left: np.ndarray, right: np.ndarray, permutations: np.ndarray) -> tuple[dict, np.ndarray]:
    if left.shape != right.shape or left.ndim != 2 or permutations.shape[1] != len(left):
        raise ValueError("Profiles and whole-poem label permutations must align")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("No silent intersection: every tested poem profile must be finite")
    coefficients = np.einsum("ij,ij->i", left, right)
    observed = float(coefficients.mean())
    null = np.empty(len(permutations))
    for start in range(0, len(permutations), 128):
        subset = permutations[start:start + 128]
        null[start:start + len(subset)] = np.einsum("ij,kij->k", left, right[subset]) / len(left)
    p = (1 + np.count_nonzero(null >= observed - 1e-12)) / (len(null) + 1)
    return {"status": "computed", "mean_within_poem_spearman": observed, "per_poem_spearman": coefficients.tolist(),
            "median_within_poem_spearman": float(np.median(coefficients)), "p_value": float(p),
            "null_mean": float(null.mean()), "null_standard_deviation": float(null.std()),
            "null_quantiles": dict(zip(["0.025", "0.5", "0.975"], np.quantile(null, [.025, .5, .975]).tolist())),
            "observed_minus_null_mean": float(observed - null.mean())}, null


def run_line_discovery(models: list[dict], records: list[dict], blocks: dict[str, str], *, permutations: int = 9999,
                       seed: int = SEED, depths=DEFAULT_DEPTHS) -> tuple[dict, dict, list[dict], np.ndarray]:
    if tuple(depths) != tuple(DEFAULT_DEPTHS):
        raise ValueError("This declared family uses exactly the fixed five-depth grid")
    if len(models) != 3 or len({(m["model_id"], m["revision"]) for m in models}) != 3:
        raise ValueError("The declared study requires three distinct complete models")
    policy_keys = ("special_tokens", "textual_separator_tokens", "prefix_endpoint")
    for model in models:
        if set(model["poems"]) != set(IDS) or any(model["pooling_policy"].get(key) != models[0]["pooling_policy"].get(key) for key in policy_keys):
            raise ValueError("Models must match complete corpus and prefix/token inclusion policy")
    corpus = {record["sonnet_id"]: record for record in records}
    if len(corpus) != len(records) or set(corpus) != set(IDS):
        raise ValueError("Require the complete canonical corpus exactly once")
    ids, folds = list(STANDARD_IDS), poem_folds(list(STANDARD_IDS), seed)
    label_permutations, permutation_audit = profile_permutations(ids, folds, blocks, permutations, seed)
    ranked, controls, fitting_audits, matrices = {}, {}, [], {}
    for model_index, model in enumerate(models):
        for depth_index, depth in enumerate(depths):
            continuous, categories = control_inputs(model, corpus, depth_index)
            for metric in METRICS:
                key = (model_index, float(depth), metric)
                matrix = np.asarray([model["poems"][sonnet]["settings"][depth_index]["values"][metric] for sonnet in ids], dtype=float)
                matrices[key] = matrix
                ranked[key] = profile_ranks(matrix)
                if depth == 0:
                    controls[key] = np.full(matrix.shape, np.nan)
                    continue
                embedding = ranked[(model_index, 0.0, metric)]
                if not np.isfinite(ranked[key]).all() or not np.isfinite(embedding).all() or not np.isfinite(continuous).all():
                    controls[key] = np.full(matrix.shape, np.nan)
                    fitting_audits.append({"model_id": model["model_id"], "depth": depth, "metric": metric,
                                           "status": "undefined", "reason": "Incomplete target or embedding-control profile"})
                    continue
                residual, predictions, fit = crossfit_nuisance(ranked[key], continuous, categories, embedding, folds)
                controls[key] = profile_ranks(residual, residual=True)
                fitting_audits.append({"model_id": model["model_id"], "depth": depth, "metric": metric, "status": "computed", "fits": fit,
                                       "out_of_fold_mean_squared_error": float(np.mean(residual**2))})
    tests, nulls = [], []
    for i, j in itertools.combinations(range(3), 2):
        for metric in METRICS:
            for depth in depths:
                for view, source in [("raw", ranked), ("cross_fitted_controls", controls)]:
                    row = {"model_a": models[i]["model_id"], "model_b": models[j]["model_id"],
                           "metric": metric, "requested_depth": float(depth), "view": view}
                    a, b = source[(i, float(depth), metric)], source[(j, float(depth), metric)]
                    invalid = [ids[k] for k in range(len(ids)) if not np.isfinite(a[k]).all() or not np.isfinite(b[k]).all()]
                    if view == "cross_fitted_controls" and depth == 0:
                        row.update(status="undefined", reason="Structural tautology: embedding target is its own embedding nuisance control", p_value=None)
                        null = np.full(permutations, np.nan)
                    elif invalid:
                        row.update(status="undefined", reason="Constant or unusable poem profile; no silent intersection", p_value=None,
                                   unusable_sonnet_ids=invalid)
                        null = np.full(permutations, np.nan)
                    else:
                        measured, null = permutation_profile_agreement(a, b, label_permutations)
                        row.update(measured)
                    row["fdr_input_p_value"] = row["p_value"] if row["p_value"] is not None else 1.0
                    tests.append(row)
                    nulls.append(null)
    expected = 3 * len(METRICS) * len(depths) * 2
    if len(tests) != expected:
        raise ValueError("Incomplete requested test family")
    for row, q in zip(tests, fdr_adjust([row["fdr_input_p_value"] for row in tests], method="by")):
        row["p_value_by"] = float(q)
    candidates = []
    selected_depths = [float(depth) for depth in depths if depth > 0]
    metric = "line_context_cosine_distance"
    for poem_index, sonnet in enumerate(ids):
        for position in range(13):
            ranks, details = [], []
            for depth in selected_depths:
                depth_ranks = []
                for model_index, model in enumerate(models):
                    profile = controls[(model_index, depth, metric)][poem_index]
                    percentile = float((rankdata(profile)[position] - 1) / 12) if np.isfinite(profile).all() else None
                    depth_ranks.append(percentile)
                    raw_distance = matrices[(model_index, depth, metric)][poem_index, position]
                    details.append({"model_id": model["model_id"], "depth": depth, "controlled_within_poem_percentile": percentile,
                                    "raw_line_context_distance": float(raw_distance) if np.isfinite(raw_distance) else None})
                ranks.append(depth_ranks)
            available = all(value is not None for values in ranks for value in values)
            metadata = [model["poems"][sonnet]["token_prefixes"][position + 1] for model in models]
            row = {"sonnet_id": sonnet, "to_line": position + 2, "previous_line_text": corpus[sonnet]["lines"][position],
                   "line_text": corpus[sonnet]["lines"][position + 1], "status": "ranked" if available else "unusable",
                   "selection_warning": "Selected exploratory transition; no independent literary annotation or candidate-level significance test",
                   "depths_all_models_at_least_75th_percentile": sum(min(values) >= .75 for values in ranks) if available else None,
                   "mean_weakest_model_percentile": float(np.mean([min(values) for values in ranks])) if available else None,
                   "mean_all_model_depth_percentile": float(np.mean(ranks)) if available else None,
                   "per_model_depth": details,
                   "token_controls": [{"model_id": model["model_id"], **meta} for model, meta in zip(models, metadata)]}
            candidates.append(row)
    candidates.sort(key=lambda row: (row["status"] != "ranked", -(row["depths_all_models_at_least_75th_percentile"] or 0),
                                    -(row["mean_weakest_model_percentile"] or 0), -(row["mean_all_model_depth_percentile"] or 0),
                                    row["sonnet_id"], row["to_line"]))
    for index, row in enumerate(candidates, 1):
        row["exploratory_rank"] = index
    report = {"schema_version": "line-discovery-v2", "status": "post hoc exploratory analysis of existing data",
              "sonnet_ids": ids, "exceptions_retained_descriptively": EXCEPTIONS,
              "models": [{"model_id": model["model_id"], "revision": model["revision"], "pooling_policy": model["pooling_policy"],
                           "layers": [setting["layer"] for setting in model["poems"][IDS[0]]["settings"]]} for model in models],
              "metrics": list(METRICS), "depths": list(depths), "ranking_decimals": RANK_DECIMALS,
              "folds": dict(zip(ids, folds.tolist())),
              "permutation": {"count": permutations, "seed": seed, "alternative": "greater", **permutation_audit,
                              "unit": "whole thirteen-transition poem profiles, retaining position; relabel only within held-out fold and original word-count/punctuation block; 153/154 fixed"},
              "multiple_comparisons": {"method": "Benjamini-Yekutieli", "family_size": len(tests),
                                       "family": "3 metrics x 5 depths x 3 model pairs x raw/poem-cross-fitted views",
                                       "undefined_tests": "Retained conservatively with p=1"},
              "tests": tests, "cross_fit_audits": fitting_audits,
              "control_policy": "Unregularized OLS, rcond=1e-10. Training-only continuous standardization and native endpoint-ID categorical vocabularies (at least10 training rows; unseen/rare fallback). Position, token counts, current/previous text lengths and punctuation, current/previous cumulative norms, and same-metric embedding-stage rank profile. Embedding controlled tests structurally undefined. Residual profile ranges<=1e-9 are undefined rather than ranked.",
              "interpretation": "Conditional exploratory correspondence, assuming exchangeability within declared blocks. Cross-fitting does not prove nuisance sufficiency. Token states are from the original full causal pass; reconstructed line means include contextualized token states and textual separators. No literary understanding, causal interpretation, candidate significance, or incremental predictive utility is established.",
              "top_selected_cases": candidates[:20]}
    detail = {"schema_version": "line-discovery-reconstruction-v2", "models": models,
              "cross_fit_rank_profiles": [{"model_id": models[i]["model_id"], "depth": depth, "metric": metric,
                  "sonnet_ids": ids, "values": [[float(x) if np.isfinite(x) else None for x in row] for row in array]}
                  for (i, depth, metric), array in controls.items()]}
    return report, detail, candidates, np.asarray(nulls)
