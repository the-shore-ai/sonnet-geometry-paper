"""Compare already-downloaded batch-analysis JSON files; no tensor downloads."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sonnet_geometry.comparisons import compare_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True,
                        help="Derived batch-analysis/report.json from each model")
    parser.add_argument("--baselines", type=Path)
    parser.add_argument("--sonnet-ids", nargs="+", help="Explicit analysis subset; omitted IDs are reported")
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--blocks", type=Path, help="Optional JSON mapping each analyzed sonnet ID to a block")
    parser.add_argument("--distance-decimals", type=int, help="Optional uniform rounding of all cosine RDM inputs; omitted preserves distances")
    parser.add_argument("--output", type=Path, required=True, help="New comparison JSON; never overwritten")
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.reports]
    result = compare_models(reports, baseline_report=json.loads(args.baselines.read_text()) if args.baselines else None,
                            sonnet_ids=args.sonnet_ids, permutations=args.permutations, seed=args.seed,
                            exchangeability_blocks=json.loads(args.blocks.read_text()) if args.blocks else None,
                            distance_decimals=args.distance_decimals)
    sources = list(args.reports) + ([args.baselines] if args.baselines else []) + ([args.blocks] if args.blocks else [])
    result["source_files"] = [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                              for path in sources]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "sonnets": len(result["sonnet_ids"]),
                      "model_pairs_depth_pooling_tests": len(result["cross_model_tests"]),
                      "baseline_descriptive_comparisons": len(result["baseline_comparisons"]),
                      "fdr": "BY over the complete requested family"}))


if __name__ == "__main__":
    main()
