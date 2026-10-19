"""Derived CPU reports from verified immutable extraction artifacts.

Reporting never writes into or alters a completed extraction directory.
Residual blocks and the final normalized output retain distinct identities.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .analysis import prefix_representations, rsa, stylometric_features, trajectory_metrics, whole_sonnet_metrics
from .storage import verify_run


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _boundary_report(states: dict[int, np.ndarray], line_numbers: Sequence[int],
                     labels: Sequence[int]) -> dict[str, Any]:
    boundaries = [{"boundary_number": label, "ending_line_number": line,
                   "usable": line in states,
                   "vector_norm": float(np.linalg.norm(states[line])) if line in states else None}
                  for label, line in zip(labels, line_numbers)]
    transitions = []
    for i in range(1, len(line_numbers)):
        previous, current = line_numbers[i - 1:i + 1]
        valid = previous in states and current in states
        transition = {"from_boundary": labels[i - 1], "to_boundary": labels[i],
                      "from_line": previous, "to_line": current, "usable": valid,
                      "cosine_distance": None, "euclidean_distance": None, "norm_change": None}
        if valid:
            metrics = trajectory_metrics(np.stack([states[previous], states[current]])[None, :, :])
            for key in ("cosine_distance", "euclidean_distance", "norm_change"):
                transition[key] = float(metrics[key][0, 0])
        else:
            transition["reason"] = "At least one cumulative prefix is not exactly reusable"
        transitions.append(transition)
    return {"boundaries": boundaries, "transitions": transitions}


def summarize_run(run_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (summary, pooled embeddings), both strict-JSON-serializable dicts.

    Whole-poem pooling excludes all special tokens and includes textual newline
    tokens. Prefix statistics require exact tokenization equivalence; unusable
    boundaries are reported, never joined across a missing line. CPU float64
    arithmetic creates derived values without modifying primary precision.
    """
    from safetensors import safe_open

    path = Path(run_dir)
    manifest = verify_run(path)
    record = json.loads((path / "record.json").read_text())
    tokens = json.loads((path / "tokens.json").read_text())
    mask = np.logical_not(tokens["special_tokens_mask"])
    if mask.shape != (len(tokens["input_ids"]),) or not np.any(mask):
        raise ValueError("Token inclusion mask is invalid or empty")
    checks = {check["line_number"]: check for check in tokens["prefix_checks"]}
    line_numbers = [span["line_number"] for span in record["line_spans"]]
    if sorted(checks) != line_numbers:
        raise ValueError("Prefix checks must cover every verse line exactly once")
    usable = [line for line in line_numbers if checks[line]["exact_token_prefix"]]
    ends = [checks[line]["full_pass_token_end"] for line in usable]
    for line, end in zip(usable, ends):
        check = checks[line]
        if end is None or tokens["input_ids"][:end] != check["prefix_input_ids"]:
            raise ValueError("Inconsistent exact-prefix metadata")
    stanza_lines = [span["line_numbers"][-1] for span in record["stanza_spans"]]
    stanza_labels = [span["stanza_number"] for span in record["stanza_spans"]]
    identity = {key: manifest[key] for key in ("run_id", "sonnet_id", "model_id", "revision")}
    boundary_policy = tokens.get("boundary_policy", "legacy_line_content_end")
    policy = {"special_tokens": "excluded", "textual_separator_tokens": "included",
              "numeric_analysis_dtype": "float64", "prefix_endpoint": boundary_policy,
              "last_token": "last non-special token in the exact prefix",
              "cumulative_mean": "mean of all non-special tokens in the exact prefix"}
    summary: dict[str, Any] = {
        "schema_version": "run-report-v1", **identity, "number": record["number"],
        "token_count": len(mask), "included_token_count": int(mask.sum()),
        "line_count": len(line_numbers), "pooling_policy": policy,
        "exact_prefix_lines": usable, "boundary_policy": boundary_policy,
        "content_end_exact_prefix_lines": [check["line_number"] for check in tokens.get("prefix_checks_content_end", [])
                                            if check["exact_token_prefix"]],
        "unusable_prefix_lines": [line for line in line_numbers if line not in usable],
        "prefix_numerical_validation": "Token-ID equivalence only; consult separate architecture validation",
        "extraction_metrics": manifest.get("metrics", {}), "layers": {},
    }
    pooled: dict[str, Any] = {"schema_version": "pooled-embeddings-v1", **identity,
                              "number": record["number"], "text_sha256": record.get("text_sha256"),
                              "pooling_policy": policy,
                              "representations": {"last_token": {}, "cumulative_mean": {}},
                              "residual_order": [], "separate_output": "final_normalized"}
    with safe_open(path / "activations.safetensors", framework="pt", device="cpu") as handle:
        residual_names = ["residual.embedding"] + sorted(key for key in handle.keys() if key.startswith("residual.block."))
        if "residual.embedding" not in handle.keys():
            raise ValueError("Missing embedding residual")
        names = residual_names + (["final_normalized"] if "final_normalized" in handle.keys() else [])
        pooled["residual_order"] = residual_names
        for name in names:
            states = handle.get_tensor(name).double().numpy()
            if states.ndim != 2 or states.shape[0] != len(mask):
                raise ValueError(f"Unexpected residual dimensions: {name}")
            layer = {"kind": "final_normalized" if name == "final_normalized" else "residual_stream",
                     "source_tensor": manifest["tensors"][name], "pooling": {}}
            for pooling in ("last_token", "cumulative_mean"):
                whole = prefix_representations(states[None, :, :], [len(mask)], pooling=pooling, include_tokens=mask)
                pooled["representations"][pooling][name] = whole[0, 0]
                prefixes = prefix_representations(states[None, :, :], ends, pooling=pooling,
                                                  include_tokens=mask)[0] if ends else []
                by_line = dict(zip(usable, prefixes))
                layer["pooling"][pooling] = {
                    "lines": _boundary_report(by_line, line_numbers, line_numbers),
                    "stanzas": _boundary_report(by_line, stanza_lines, stanza_labels)}
            summary["layers"][name] = layer
    return _json_safe(summary), _json_safe(pooled)


def summarize_batch(pooled_reports: Sequence[dict[str, Any]], *,
                    records: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Cosine matrices and descriptive anomaly rankings for one model/revision.

    Input report order is retained explicitly. This returns no significance
    claims and requires distinct sonnets with identical representation keys.
    """
    if len(pooled_reports) < 2:
        raise ValueError("At least two pooled reports are needed")
    first = pooled_reports[0]
    identities = [(report["model_id"], report["revision"]) for report in pooled_reports]
    if len(set(identities)) != 1:
        raise ValueError("Batch cosine geometry requires one model and revision")
    sonnet_ids = [report["sonnet_id"] for report in pooled_reports]
    if len(set(sonnet_ids)) != len(sonnet_ids):
        raise ValueError("Duplicate sonnets must not enter a batch as independent items")
    reference_keys = {pool: set(layers) for pool, layers in first["representations"].items()}
    for report in pooled_reports:
        if report["pooling_policy"] != first["pooling_policy"] or {
            pool: set(layers) for pool, layers in report["representations"].items()} != reference_keys:
            raise ValueError("Batch pooling policies and representation keys must match")
    result: dict[str, Any] = {"schema_version": "batch-report-v1", "model_id": first["model_id"],
                              "revision": first["revision"], "sonnet_ids": sonnet_ids,
                              "run_ids": [r["run_id"] for r in pooled_reports], "pooling": {},
                              "inference": "Descriptive only; no calibrated literary or authorship claim"}
    for pool, names in reference_keys.items():
        result["pooling"][pool] = {}
        for name in sorted(names):
            x = np.array([r["representations"][pool][name] for r in pooled_reports])
            metrics = whole_sonnet_metrics(x)
            scores = metrics["mean_distance_anomaly"]
            order = np.argsort(-scores, kind="stable")
            result["pooling"][pool][name] = {**metrics, "anomaly_ranking": [
                {"sonnet_id": sonnet_ids[i], "mean_cosine_distance": scores[i]} for i in order]}
    if records is not None:
        by_id = {record["sonnet_id"]: record for record in records}
        x, names = stylometric_features([by_id[sonnet_id]["text"] for sonnet_id in sonnet_ids])
        standard_deviation = x.std(axis=0)
        standardized = np.divide(x - x.mean(axis=0), standard_deviation,
                                 out=np.zeros_like(x), where=standard_deviation > 0)
        rdm = np.linalg.norm(standardized[:, None, :] - standardized[None, :, :], axis=-1)
        result["stylometric_baseline"] = {"feature_names": names, "raw_features": x,
                                           "distance": "Euclidean after within-batch population-z-score; constant features zeroed",
                                           "rdm": rdm,
                                           "status": "Descriptive corpus-level fit only; no held-out predictive evaluation"}
        for layers in result["pooling"].values():
            for measurements in layers.values():
                try:
                    measurements["stylometric_rsa_spearman"] = rsa(measurements["cosine_distance"], rdm)
                except ValueError:
                    measurements["stylometric_rsa_spearman"] = None
                measurements["stylometric_rsa_inference"] = "Descriptive correlation; no p-value or incremental-value claim"
    return _json_safe(result)


def export_batch_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    """Save a new report directory, labelled CSV matrices, and anomaly tables."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=False)
    (path / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for pooling, layers in report["pooling"].items():
        for name, measurements in layers.items():
            filename = f"{pooling}.{name}"
            with (path / f"{filename}.cosine.csv").open("w", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["sonnet_id", *report["sonnet_ids"]])
                writer.writerows([sonnet, *row] for sonnet, row in
                                 zip(report["sonnet_ids"], measurements["cosine_similarity"]))
            with (path / f"{filename}.anomalies.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sonnet_id", "mean_cosine_distance"], lineterminator="\n")
                writer.writeheader()
                writer.writerows(measurements["anomaly_ranking"])
    baseline = report.get("stylometric_baseline")
    if baseline:
        with (path / "stylometric_features.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["sonnet_id", *baseline["feature_names"]])
            writer.writerows([sonnet, *row] for sonnet, row in zip(report["sonnet_ids"], baseline["raw_features"]))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline matrices and rankings from pooled JSON artifacts")
    parser.add_argument("--pooled", type=Path, nargs="+", required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New report directory; never overwritten")
    args = parser.parse_args()
    from .corpus import load_corpus

    pooled = [json.loads(path.read_text()) for path in args.pooled]
    report = summarize_batch(pooled, records=load_corpus(args.corpus) if args.corpus else None)
    path = export_batch_report(report, args.output)
    print(json.dumps({"output": str(path), "sonnets": len(pooled), "inference": "descriptive only"}))


if __name__ == "__main__":
    main()
