"""Analyze complete, frozen v4 artifacts offline; never launch model inference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from sonnet_geometry.confirmation_analysis import FAMILIES, analyze_runs, file_receipt, load_run
from sonnet_geometry.storage import canonical_json, write_exclusive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/confirmation-v4/execution"))
    parser.add_argument("--preflight-root", type=Path, default=Path("experiments/confirmation-v4/preflights"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/confirmation-v4/protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/confirmation-v4/analysis.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol_receipt = file_receipt(args.protocol)
    protocol = json.loads(args.protocol.read_text())
    required = [args.input_root/family/name for family in FAMILIES for name in ("report.json", "pooled.safetensors")]
    required += [args.preflight_root/f"{family}.json" for family in FAMILIES]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Complete actual execution and frozen preflights required; no output created: {missing}")
    runs = {family: load_run(args.input_root/family/"report.json", args.input_root/family/"pooled.safetensors",
                            args.preflight_root/f"{family}.json", family, protocol, protocol_receipt["sha256"])
            for family in FAMILIES}
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol_receipt,
              "source_files": [file_receipt(__file__), file_receipt("src/sonnet_geometry/confirmation_analysis.py"),
                               file_receipt("src/sonnet_geometry/text_control_analysis.py")],
              "storage_validation": "Local pooled bytes match execution receipts; full primary-artifact remote readback requires its separate storage audit.",
              **analyze_runs(runs, protocol)}
    write_exclusive(args.output, canonical_json(result))
    print(json.dumps({"output": str(args.output), "sha256": file_receipt(args.output)["sha256"],
                      "status": result["status"], "claims": len(result["claims"])}))


if __name__ == "__main__":
    main()
