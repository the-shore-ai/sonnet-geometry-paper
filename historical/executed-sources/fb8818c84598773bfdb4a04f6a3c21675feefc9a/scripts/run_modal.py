"""Launch one bounded experiment. Run from repository root; no deployment persists."""
from __future__ import annotations

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonnets", default="1,18,29,55,73,99,116,126,130,145")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--stage", default="pilot")
    parser.add_argument("--model", choices=["qwen", "gemma", "muse"], default="qwen")
    parser.add_argument("--validate-first", action="store_true", help="Stop batch if first sonnet fails fixed-shape causality")
    parser.add_argument("--auth-json", type=Path, help="Optional private local Modal credentials file; never uploaded")
    parser.add_argument("--budget-usd", type=float, help="Optional lower cap within the fixed model allocation")
    parser.add_argument("--reserve-usd", type=float, help="At least $5 for Qwen, $7.50 for Gemma/Muse")
    parser.add_argument("--ledger", type=Path, help="Must be the model's canonical repository ledger")
    args = parser.parse_args()
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        parser.error("Set GOOGLE_APPLICATION_CREDENTIALS to the private local GCP service-account file")
    from sonnet_geometry.budget import (GPU_RESOURCE_RATE, TOTAL_ALLOCATION_USD, check_capacity,
                                       resolve_model_budget, settle_finished, write_ledger_atomic)
    try:
        policy, args.budget_usd, args.reserve_usd, args.ledger = resolve_model_budget(
            ROOT, args.model, budget_usd=args.budget_usd, reserve_usd=args.reserve_usd, ledger=args.ledger)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger_lock = args.ledger.with_suffix(".lock").open("a")
    try:
        fcntl.flock(ledger_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Another launcher holds this cost ledger; wait for that attempt to finish")
    ledger = json.loads(args.ledger.read_text()) if args.ledger.exists() else {"attempts": []}
    try:
        accounted_before = check_capacity(ledger["attempts"], args.budget_usd, args.reserve_usd)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(f"Cannot reserve this attempt: {exc}")
    ledger.update(ceiling_usd=policy.allocation_usd, model_allocation=args.model,
                  total_approved_allocation_usd=TOTAL_ALLOCATION_USD)
    os.environ["SHORE_MODEL_RUNTIME"] = "standard" if args.model == "qwen" else "modern"
    import modal
    from sonnet_geometry.corpus import load_corpus
    from infra.modal_app import app, extract_batch, prepare_model

    if args.model == "qwen":
        lock = json.loads((ROOT / "configs/models.lock.json").read_text())["models"][0]
        capture_layers = [0, 18, 35]
        runtime_dir = ROOT
    else:
        identifier = {"gemma": "google/gemma-4-31B-it", "muse": "meta-models/Muse-Glimmer-30B"}[args.model]
        lock = next(m for m in json.loads((ROOT / "configs/models-candidates.lock.json").read_text())["models"]
                    if m["model_id"] == identifier)
        capture_layers = [0, 30, 58] if args.model == "gemma" else [0, 26, 51]
        runtime_dir = ROOT / "infra/modern_runtime"
    records = load_corpus(ROOT / "datasets/normalized/sonnets.jsonl")
    selected = set(range(1, 155)) if args.all else {int(n) for n in args.sonnets.split(",")}
    records = [r for r in records if r["number"] in selected]
    if not records or len(records) != len(selected):
        parser.error("Sonnet selection must be within 1–154")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = f"v1-{args.stage}-{stamp}-{uuid.uuid4().hex[:8]}"
    config = {"schema_version": 1, "experiment_id": experiment_id, "stage": args.stage,
              "model_id": lock["model_id"], "revision": lock["revision"],
              "bucket": "the-shore-508003-shakespeare-activation-geometry", "seed": 0,
              "sonnet_numbers": sorted(selected), "dtype": "bfloat16", "attention": "eager",
              "capture": ["q_proj", "k_proj", "v_proj", "attention_output", "mlp_intermediate", "mlp_output"],
              "capture_layers": capture_layers, "soft_timeout_seconds": policy.soft_timeout_seconds,
              "runtime_profile": args.model, "validate_first": args.validate_first,
              "fp32_prefix_diagnostic": args.stage == "validation" and args.model == "qwen",
              "resource_rate_usd_per_second": GPU_RESOURCE_RATE,
              "resource_timing": {"gpu_timeout_seconds": policy.gpu_timeout_seconds,
                                  "startup_timeout_seconds": policy.startup_timeout_seconds,
                                  "download_timeout_seconds": policy.download_timeout_seconds,
                                  "soft_timeout_seconds": policy.soft_timeout_seconds,
                                  "scaledown_window_seconds": 2, "gpu": "H200", "cpu": 4, "memory_mib": 32768},
              "budget_allocation": {"model": args.model, "allocation_usd": policy.allocation_usd,
                                    "effective_budget_usd": args.budget_usd,
                                    "canonical_ledger": args.ledger.name,
                                    "accounted_before_usd": accounted_before,
                                    "minimum_reservation_usd": policy.minimum_reservation_usd},
              "corpus_sha256": hashlib.sha256((ROOT / "datasets/normalized/sonnets.jsonl").read_bytes()).hexdigest(),
              "lock_sha256": hashlib.sha256((runtime_dir / "uv.lock").read_bytes()).hexdigest(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
              "approved_compute_ceiling_usd": TOTAL_ALLOCATION_USD, "reserved_usd": args.reserve_usd}
    if config["source_dirty"]:
        parser.error("Commit the tested source before cloud execution")
    attempt = {"experiment_id": experiment_id, "model": args.model,
               "model_allocation_usd": policy.allocation_usd, "reserved_usd": args.reserve_usd, "status": "reserved"}
    ledger["attempts"].append(attempt)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    write_ledger_atomic(args.ledger, ledger)
    client = None
    if args.auth_json:
        auth = json.loads(args.auth_json.read_text())
        client = modal.Client.from_credentials(auth["token_id"], auth["token_secret"])
    started = time.monotonic()
    try:
        with modal.enable_output(), app.run(client=client):
            preparation = prepare_model.remote(config["model_id"], config["revision"])
            report = extract_batch.remote(config, records)
        report["preparation"] = preparation
        report["launcher_wall_seconds"] = time.monotonic() - started
        destination = ROOT / "outputs" / experiment_id
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        attempt.update(status="finished", completed=report["completed"], report_path=str(destination / "report.json"),
                       function_estimate_usd=report["estimated_resource_cost_usd"])
        settle_finished(attempt, report["launcher_wall_seconds"], config["resource_rate_usd_per_second"])
        print(json.dumps({"report": str(destination / "report.json"), "completed": report["completed"],
                          "validation_passed": report.get("validation_passed"),
                          "estimated_resource_cost_usd": report["estimated_resource_cost_usd"]}))
    except BaseException:
        attempt["status"] = "failed-reservation-retained"
        raise
    finally:
        write_ledger_atomic(args.ledger, ledger)
        ledger_lock.close()


if __name__ == "__main__":
    main()
