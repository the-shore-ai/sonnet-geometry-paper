"""Launch exactly one bounded v3 control run in the existing model allocation."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from sonnet_geometry.budget import (CONTROL_PROFILE, CONTROL_ENVELOPE, CONTROL_RESOURCE_RATE,
        resolve_control_budget, check_capacity, write_ledger_atomic, settle_finished)
    from sonnet_geometry.corpus import load_corpus
    from sonnet_geometry.text_controls import SONNETS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["qwen", "gemma", "muse"])
    args = parser.parse_args()
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        parser.error("Private GCP authentication required")
    policy, reservation, ledger_path = resolve_control_budget(ROOT, args.model)
    lockfile = ledger_path.with_suffix(".lock").open("a")
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger = json.loads(ledger_path.read_text())
    accounted = check_capacity(ledger["attempts"], policy.allocation_usd, reservation)
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty: parser.error("Commit tested source before paid execution")
    candidates = json.loads((ROOT / "configs/models-candidates.lock.json").read_text())["models"]
    candidates += json.loads((ROOT / "configs/models.lock.json").read_text())["models"]
    model_id = {"qwen": "Qwen/Qwen3-8B", "gemma": "google/gemma-4-31B-it", "muse": "meta-models/Muse-Glimmer-30B"}[args.model]
    model_lock = next(m for m in candidates if m["model_id"] == model_id)
    records = [r for r in load_corpus(ROOT / "datasets/normalized/sonnets.jsonl") if r["number"] in SONNETS]
    experiment_id = "v3-controls-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    protocol_path = ROOT / "experiments/followup-v3/text-control-protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["sonnets"] != list(SONNETS) or protocol["resource_envelope"] != CONTROL_ENVELOPE:
        parser.error("Code and frozen protocol differ")
    config = {"experiment_id": experiment_id, "model": args.model, "model_id": model_id,
              "revision": model_lock["revision"], "bucket": "the-shore-508003-shakespeare-activation-geometry",
              "resource_envelope": CONTROL_ENVELOPE, "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "accounted_before_usd": accounted, "model_allocation_usd": policy.allocation_usd,
              "reserved_usd": reservation, "compute_ceiling_usd": 25,
              "corpus_sha256": hashlib.sha256((ROOT / "datasets/normalized/sonnets.jsonl").read_bytes()).hexdigest()}
    preflight_path = ROOT / f"experiments/followup-v3/text-preflight-{args.model}.json"
    config["preflight"] = json.loads(preflight_path.read_text())
    config["preflight_sha256"] = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    attempt = {"experiment_id": experiment_id, "model": args.model, "budget_profile": CONTROL_PROFILE,
               "resource_envelope": CONTROL_ENVELOPE, "reserved_usd": reservation, "status": "reserved"}
    ledger["attempts"].append(attempt); write_ledger_atomic(ledger_path, ledger)
    os.environ["SHORE_MODEL_RUNTIME"] = "standard" if args.model == "qwen" else "modern"
    started = time.monotonic()
    try:
        import modal
        from infra.modal_controls import app, prepare, run_controls
        with modal.enable_output(), app.run():
            preparation = prepare.remote(model_id, model_lock["revision"])
            result = run_controls.remote(config, records)
        result["preparation"] = preparation; result["launcher_seconds"] = time.monotonic() - started
        destination = ROOT / "outputs" / experiment_id; destination.mkdir(exist_ok=False)
        report_path = destination / "report.json"
        report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        attempt.update(report_path=str(report_path), completed_cases=len(result["report"]["cases"]))
        settle_finished(attempt, result["launcher_seconds"], CONTROL_RESOURCE_RATE)
        print(json.dumps({"report": str(report_path), "complete": result["report"]["complete"],
                          "conservative_accounted_usd": attempt["conservative_compute_bound_usd"]}))
    except BaseException:
        attempt["status"] = "failed-reservation-retained"; raise
    finally:
        write_ledger_atomic(ledger_path, ledger); lockfile.close()


if __name__ == "__main__": main()
