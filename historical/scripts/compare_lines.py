"""Compare existing per-line summaries without downloading primary tensors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sonnet_geometry.line_comparisons import compare_transition_profiles, plan_line_derivation, transition_aggregates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True, help="Full execution batch reports containing results[].summary")
    parser.add_argument("--sonnet-ids", nargs="+", help="Explicit subset; omitted IDs are recorded")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-raw-derivation", action="store_true", help="Emit a generation-pinned download plan only; no network")
    parser.add_argument("--aggregates-only", action="store_true", help="Small review artifact without per-sonnet/line arrays")
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.reports]
    if args.plan_raw_derivation:
        result = {"plans": [plan_line_derivation(report) for report in reports]}
    else:
        result = compare_transition_profiles(reports, sonnet_ids=args.sonnet_ids)
        if args.aggregates_only:
            result = transition_aggregates(result)
    result["source_files"] = [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                              for path in args.reports]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size,
                      "mode": "plan only" if args.plan_raw_derivation else "descriptive matched line-transition profiles"}))


if __name__ == "__main__":
    main()
