"""Analyze local, checksum-validated frozen control reports without model inference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from sonnet_geometry.storage import canonical_json, write_exclusive
from sonnet_geometry.text_control_analysis import FAMILIES, analyze_runs, file_receipt, load_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/followup-v3/text-controls"))
    parser.add_argument("--protocol", type=Path, default=Path("experiments/followup-v3/text-control-protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/followup-v3/text-control-analysis.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = file_receipt(args.protocol)
    protocol = json.loads(args.protocol.read_text())
    runs, missing = {}, []
    for family in FAMILIES:
        report, pooled = args.input_root/family/"report.json", args.input_root/family/"pooled.safetensors"
        if not report.exists() and not pooled.exists():
            missing.append(family)
            continue
        if not report.exists() or not pooled.exists():
            raise FileNotFoundError(f"Both report and pooled artifact required for {family}")
        runs[family] = load_run(report, pooled, family, protocol, receipt["sha256"])
    if not runs:
        raise FileNotFoundError("No actual completed or partial execution reports; no scientific output created")
    result = {"schema_version": "text-control-analysis-v3", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": receipt, "planned_models": list(FAMILIES), "missing_model_reports": missing,
        "source_files": [file_receipt(__file__), file_receipt("src/sonnet_geometry/text_control_analysis.py")],
        "interpretation": "Descriptive effects of frozen interventions on12postdiscovery poems. No semantic mechanism or independent literary-utility claim.",
        "storage_validation": "Local pooled bytes match generation-identified execution receipts. This is not a substitute for full remote primary-artifact readback.",
        **analyze_runs(runs, protocol)}
    write_exclusive(args.output, canonical_json(result))
    print(json.dumps({"output": str(args.output), "sha256": file_receipt(args.output)["sha256"],
                      "loaded_models": list(runs), "missing_models": missing,
                      "within_model_comparisons": len(result["within_model"]),
                      "cross_model_comparisons": len(result["cross_model"])}))


if __name__ == "__main__":
    main()
