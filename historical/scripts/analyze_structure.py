"""Produce descriptive clustering and nuisance summaries from private reports."""
import argparse
import hashlib
import json
from pathlib import Path

from sonnet_geometry.corpus import load_corpus
from sonnet_geometry.structure import summarize_structure

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reports", type=Path, nargs="+", required=True)
parser.add_argument("--baselines", type=Path, required=True)
parser.add_argument("--corpus", type=Path, default=Path("datasets/normalized/sonnets.jsonl"))
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--distance-decimals", type=int, help="Optional uniform rounding of all cosine RDM inputs; omitted preserves distances")
args = parser.parse_args()
result = summarize_structure([json.loads(path.read_text()) for path in args.reports],
                             json.loads(args.baselines.read_text()), load_corpus(args.corpus),
                             distance_decimals=args.distance_decimals)
result["sources"] = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                     for path in [*args.reports, args.baselines, args.corpus]]
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as handle:
    json.dump(result, handle, indent=2, allow_nan=False)
    handle.write("\n")
print(json.dumps({"output": str(args.output), "models": len(result["models"]), "sonnets": len(result["sonnet_ids"])}))
