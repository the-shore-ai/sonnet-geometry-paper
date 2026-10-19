"""Use saved isolated-line BERT embeddings; no model run or network access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sonnet_geometry.line_baselines import (compact_baseline_comparison, compare_isolated_line_baseline,
                                           load_verified_archive)
from sonnet_geometry.storage import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--prior", type=Path, default=Path("experiments/prior-method-replica.json"))
    parser.add_argument("--embeddings", type=Path, default=Path("outputs/prior-replica/line-embeddings.npz"))
    parser.add_argument("--sonnet-ids", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-detail", action="store_true", help="Include repeated per-line arrays; default is compact")
    args = parser.parse_args()
    prior = json.loads(args.prior.read_text())
    archive = load_verified_archive(args.embeddings, prior)
    reports = [json.loads(path.read_text()) for path in args.reports]
    result = compare_isolated_line_baseline(reports, prior, archive, sonnet_ids=args.sonnet_ids)
    if not args.full_detail:
        result = compact_baseline_comparison(result)
    result["source_files"] = [{"path": str(path), "sha256": sha256_file(path)}
                              for path in [*args.reports, args.prior, args.embeddings]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size,
                      "comparisons": len(result["comparisons"]), "inferential_tests": 0}))


if __name__ == "__main__":
    main()
