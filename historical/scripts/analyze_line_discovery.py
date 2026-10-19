"""Run declared v2 line reconstruction and whole-poem exploratory tests locally."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from sonnet_geometry.corpus import load_corpus
from sonnet_geometry.line_discovery import prepare_model, run_line_discovery
from sonnet_geometry.storage import sha256_file, write_exclusive


def save_json(path: Path, value) -> None:
    write_exclusive(path, (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs=3, type=Path, required=True)
    parser.add_argument("--token-dirs", nargs=3, type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("datasets/normalized/sonnets.jsonl"))
    parser.add_argument("--blocks", type=Path, default=Path("experiments/permutation-blocks.json"))
    parser.add_argument("--token-receipt", type=Path, default=Path("experiments/deep-study-v2/token-metadata-receipt.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/deep-study-v2"))
    parser.add_argument("--null-output", type=Path, default=Path("outputs/deep-study-v2/line-permutation-null.npz"))
    args = parser.parse_args()
    paths = [args.output_dir / name for name in ["line-discovery.json", "line-reconstruction.json", "line-selected-candidates.json"]]
    if any(path.exists() for path in [*paths, args.null_output]):
        raise FileExistsError("All v2 line outputs must be new; existing evidence is never overwritten")
    started = time.monotonic()
    records = load_corpus(args.corpus)
    models = []
    for report_path, token_directory in zip(args.reports, args.token_dirs):
        report = json.loads(report_path.read_text())
        model = prepare_model(report, records, token_directory)
        models.append(model)
        del report
        print(json.dumps({"prepared_model": model["model_id"], "verified_token_files": len(model["source_verification"])}), flush=True)
    result, reconstruction, candidates, nulls = run_line_discovery(models, records, json.loads(args.blocks.read_text()))
    audits = []
    for model in models:
        for depth_index in range(5):
            rows = [(sonnet, index + 2, row) for sonnet, poem in model["poems"].items()
                    for index, row in enumerate(poem["settings"][depth_index]["reconstruction_audits"])]
            valid = [row for _, _, row in rows if row["usable"]]
            audits.append({"model_id": model["model_id"], "depth": model["poems"]["sonnet-001"]["settings"][depth_index]["requested_depth"],
                           "total_transitions": len(rows), "usable_transitions": len(valid),
                           "unusable": [{"sonnet_id": sonnet, "to_line": line, **row} for sonnet, line, row in rows if not row["usable"]],
                           "maximum_condition_proxy": max((row["condition_proxy"] for row in valid), default=None),
                           "maximum_roundoff_bound_proxy": max((row["roundoff_bound_proxy"] for row in valid), default=None),
                           "maximum_naive_relative_disagreement": max((row["naive_relative_disagreement"] for row in valid), default=None),
                           "maximum_euclidean_relative_disagreement": max((row["euclidean_relative_disagreement"] for row in valid), default=None)})
    result["reconstruction_summary"] = audits
    result["source_files"] = [{"path": str(path.resolve()), "sha256": sha256_file(path)}
                              for path in [*args.reports, args.corpus, args.blocks, args.token_receipt]]
    result["implementation_sources"] = [{"path": str(path), "sha256": sha256_file(path)} for path in
                                        [Path("src/sonnet_geometry/line_discovery.py"), Path(__file__)]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.null_output.parent.mkdir(parents=True, exist_ok=True)
    with args.null_output.open("xb") as handle:
        np.savez_compressed(handle, null_statistics=nulls,
                            test_index=np.arange(len(result["tests"])),
                            sonnet_ids=np.asarray(result["sonnet_ids"]))
    result["full_null_archive"] = {"path": str(args.null_output), "sha256": sha256_file(args.null_output),
                                    "shape": list(nulls.shape), "undefined_rows": "NaN; retained in family with p=1"}
    save_json(paths[1], reconstruction)
    save_json(paths[2], {"selection": "All ranked/unusable standard-form transitions retained; no candidate-level tests",
                         "candidate_count": len(candidates), "candidates": candidates})
    result["detail_artifacts"] = [{"path": str(path), "sha256": sha256_file(path)} for path in paths[1:]]
    result["runtime_seconds"] = time.monotonic() - started
    save_json(paths[0], result)
    print(json.dumps({"output": str(paths[0]), "test_family": len(result["tests"]),
                      "computed": sum(row["status"] == "computed" for row in result["tests"]),
                      "by_q_at_most_0_05": sum(row["p_value_by"] <= .05 for row in result["tests"]),
                      "selected_candidates_retained": len(candidates), "seconds": result["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
