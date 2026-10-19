"""Validate frozen confirmation inputs; optionally execute one approved bounded model run."""
import argparse
from copy import deepcopy
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

from infra.modal_confirmation import (
    ExecutionDisabled, build_execution_plan, create_modal_backend,
    read_budget_snapshot, require_execution_enabled, validate_preflight,
)
from sonnet_geometry.storage import canonical_json, sha256_file, write_exclusive


def resolve_operational_policy(root: Path, protocol: dict, protocol_hash: str, model: str,
                               amendment_path: Path | None = None) -> tuple[dict, dict | None]:
    """Validate one explicit operational recovery without changing scientific inputs.

    Call while the target model's canonical ledger lock is held. The returned
    policy changes allocations only; every historical reservation remains charged.
    """
    from sonnet_geometry.budget import resolve_confirmation_budget
    execution = deepcopy(protocol["execution"])
    histories = {}
    for family in ("qwen", "gemma", "muse"):
        _, _, path = resolve_confirmation_budget(root, family, execution)
        histories[family] = json.loads(path.read_text())["attempts"]
    matching = [a for a in histories[model] if a.get("protocol_sha256") == protocol_hash
                and a.get("budget_profile") == "bounded-confirmation-v4"]
    if amendment_path is None:
        if matching:
            raise ValueError("This model already has a frozen-protocol attempt; an explicit bounded recovery amendment is required")
        return execution, None
    payload = amendment_path.read_bytes()
    amendment = json.loads(payload)
    if amendment.get("schema_version") == "confirmation-operational-recovery-v2":
        return resolve_measured_recovery(root, protocol, protocol_hash, model,
                                         amendment_path, payload, amendment, histories)
    aggregate_cap = execution["prior_conservative_accounting_usd"] + execution["maximum_new_reservations_usd"]
    required = {
        "schema_version": "confirmation-budget-amendment-v4", "protocol_sha256": protocol_hash,
        "model": "muse", "reason_code": "modal_app_create_rate_limit",
        "approved_total_usd": 40, "allocations_usd": {"qwen": 12, "gemma": 12, "muse": 16},
        "reservation_per_model_usd": 5, "aggregate_accounting_ceiling_usd": aggregate_cap,
        "maximum_recovery_attempts": 1, "manual_recovery_only": True,
    }
    if model != "muse" or execution["approved_total_usd"] != 40:
        raise ValueError("This narrow amendment applies only to the approved Muse app-creation recovery")
    if any(amendment.get(key) != value for key, value in required.items()):
        raise ValueError("Recovery amendment changes the protocol identity, allocation, reservation, or aggregate scope")
    failed_id = amendment["failed_attempt_id"]
    failed = [a for a in matching if a["experiment_id"] == failed_id]
    if len(failed) != 1 or failed[0].get("status") != "failed-reservation-retained" or failed[0].get("reserved_usd") != 5:
        raise ValueError("Original failed Muse reservation must remain present and fully retained")
    if "conservative_compute_bound_usd" in failed[0]:
        raise ValueError("Failed reservation cannot be reduced for recovery")
    if any(a.get("recovery_of_failed_attempt_id") == failed_id for a in histories[model]):
        raise ValueError("The one manual recovery for this failed attempt has already been reserved")
    settled = amendment["settled_attempts"]
    if set(settled) != {"qwen", "gemma"}:
        raise ValueError("Exact Qwen and Gemma attempt identities are required")
    for family, attempt_id in settled.items():
        candidates = [a for a in histories[family] if a["experiment_id"] == attempt_id
                      and a.get("protocol_sha256") == protocol_hash
                      and a.get("budget_profile") == "bounded-confirmation-v4"]
        if (len(candidates) != 1 or candidates[0].get("status") != "finished"
                or "conservative_compute_bound_usd" not in candidates[0]):
            raise ValueError("Qwen and Gemma must settle their exact existing attempts before recovery")
        if any(a.get("protocol_sha256") == protocol_hash and a.get("status") == "reserved"
               for a in histories[family]):
            raise ValueError("Another confirmation worker still has a live reservation")
    evidence = amendment["failure_evidence"]
    evidence_path = (root / evidence["path"]).resolve()
    if not evidence_path.is_relative_to((root / "outputs/confirmation-v4").resolve()):
        raise ValueError("Recovery failure evidence must be the local confirmation execution log")
    if sha256_file(evidence_path) != evidence["sha256"]:
        raise ValueError("Recovery failure evidence checksum mismatch")
    evidence_text = evidence_path.read_text()
    if ("ResourceExhaustedError" not in evidence_text or "App create rate limit exceeded" not in evidence_text
            or "/modal.client.ModalClient/AppCreate" not in evidence_text
            or "Initialized" in evidence_text or "View app" in evidence_text):
        raise ValueError("Evidence does not establish the narrow pre-app creation rate-limit failure")
    execution["allocations_usd"] = dict(amendment["allocations_usd"])
    # Central validation retains the exact resource bounds, ledger filenames,
    # approved cumulative ceiling and $5 reservation from the scientific protocol.
    for family in histories:
        resolve_confirmation_budget(root, family, execution)
    receipt = {"path": str(amendment_path), "sha256": hashlib.sha256(payload).hexdigest(),
               "amendment": amendment}
    return execution, receipt


def resolve_measured_recovery(root, protocol, protocol_hash, model, amendment_path,
                             payload, amendment, histories):
    """One sequential, evidence-bound recovery per family within the accepted $40.

    This is an explicit operational deviation, never a rewrite of the original
    three-attempt protocol. Scientific inputs and criteria retain their raw hash.
    """
    from infra.modal_confirmation import UPLOAD_PIPELINE_POLICY
    from sonnet_geometry.budget import resolve_confirmation_budget
    execution = deepcopy(protocol["execution"])
    required = {
        "protocol_sha256": protocol_hash, "model": model,
        "approved_total_usd": 40, "aggregate_accounting_ceiling_usd": 40,
        "reservation_per_model_usd": 5, "maximum_recovery_attempts_per_model": 1,
        "manual_recovery_only": True, "sequential_execution_only": True,
        "operational_execution_policy": UPLOAD_PIPELINE_POLICY,
    }
    if execution["approved_total_usd"] != 40 or any(amendment.get(k) != v for k, v in required.items()):
        raise ValueError("Recovery changes the accepted ceiling, frozen identity, or bounded upload policy")
    if not amendment.get("authorization_basis"):
        raise ValueError("Recovery must record the owner's continued authorization")
    if any(a.get("status") == "reserved" for rows in histories.values() for a in rows):
        raise ValueError("Sequential recovery requires every prior worker to settle")
    snapshots = amendment["ledger_snapshots"]
    if set(snapshots) != set(histories):
        raise ValueError("Every canonical ledger snapshot is required")
    for family in histories:
        _, _, path = resolve_confirmation_budget(root, family, execution)
        if snapshots[family] != sha256_file(path):
            raise ValueError("Recovery ledger snapshot is stale or changed")
    matching = [a for a in histories[model] if a.get("protocol_sha256") == protocol_hash
                and a.get("budget_profile") == "bounded-confirmation-v4"]
    if (len(matching) != 1 or matching[0]["experiment_id"] != amendment["failed_attempt_id"]
            or matching[0].get("reserved_usd") != 5):
        raise ValueError("Exactly the original unrecovered $5 attempt is required")
    previous = matching[0]
    evidence = amendment["failure_evidence"]
    evidence_path = (root / evidence["path"]).resolve()
    if not evidence_path.is_relative_to((root / "outputs").resolve()):
        raise ValueError("Recovery evidence must be an immutable local execution artifact")
    if sha256_file(evidence_path) != evidence["sha256"]:
        raise ValueError("Recovery evidence checksum mismatch")
    if model == "muse":
        evidence_text = evidence_path.read_text()
        if (amendment["reason_code"] != "modal_app_create_rate_limit"
                or previous.get("status") != "failed-reservation-retained"
                or "conservative_compute_bound_usd" in previous
                or "ResourceExhaustedError" not in evidence_text
                or "App create rate limit exceeded" not in evidence_text
                or "/modal.client.ModalClient/AppCreate" not in evidence_text
                or "Initialized" in evidence_text or "View app" in evidence_text):
            raise ValueError("Muse recovery requires the fully retained pre-app failure")
    else:
        wrapper = json.loads(evidence_path.read_text())
        report = wrapper["report"]
        if (amendment["reason_code"] != "measured_pilot_timeout_projection"
                or previous.get("status") != "finished"
                or "conservative_compute_bound_usd" not in previous
                or report["experiment_id"] != previous["experiment_id"]
                or report["model"] != model or report["config"]["protocol_sha256"] != protocol_hash
                or report["complete"] is not False or len(report["cases"]) != 64
                or report["stop_reason"] != "Measured two-poem both-precision pilot cannot finish within frozen soft timeout"):
            raise ValueError("Recovery requires the exact preserved 64-case timing stop")
    execution["allocations_usd"] = amendment["allocations_usd"]
    execution["maximum_new_reservations_usd"] = 40 - execution["prior_conservative_accounting_usd"]
    execution["scope"] = "Explicit sequential recovery; one additional attempt per model, cumulative ceiling $40"
    for family in histories:
        resolve_confirmation_budget(root, family, execution)
    return execution, {"path": str(amendment_path), "sha256": hashlib.sha256(payload).hexdigest(),
                       "amendment": amendment}


def enforce_aggregate_attempt_cap(snapshot: dict, protocol: dict, reservation: float) -> float:
    execution = protocol["execution"]
    cap = execution["prior_conservative_accounting_usd"] + execution["maximum_new_reservations_usd"]
    if snapshot["accounted_total_usd"] + reservation > min(snapshot["approved_total_usd"], cap):
        raise ValueError("New reservation exceeds the approved aggregate attempt scope")
    return cap


def execute_model(protocol_path: Path, model: str, preflight_path: Path,
                  amendment_path: Path | None = None) -> dict:
    """One canonical locked reservation; interrupted attempts keep the entire $5."""
    from sonnet_geometry.budget import (
        CONTROL_RESOURCE_RATE, check_capacity, resolve_confirmation_budget,
        settle_finished, write_ledger_atomic,
    )
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    require_execution_enabled(protocol, ROOT)
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    preflight_bytes = preflight_path.read_bytes()
    preflight = json.loads(preflight_bytes)
    validate_preflight(ROOT, protocol, protocol_hash, preflight)
    if preflight["model"] != model:
        raise ValueError("Selected model/preflight mismatch")
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise ValueError("Private GCP authentication required")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        raise ValueError("Commit tested source before paid execution")
    policy, reservation, ledger_path = resolve_confirmation_budget(ROOT, model, protocol["execution"])
    # Hold across the full run so recovery reservations cannot race across families.
    global_lock = (ROOT / "outputs/.confirmation-execution.lock").open("a")
    fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lockfile = ledger_path.with_suffix(".lock").open("a")
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        execution, amendment_receipt = resolve_operational_policy(ROOT, protocol, protocol_hash, model, amendment_path)
        policy, reservation, resolved_path = resolve_confirmation_budget(ROOT, model, execution)
        if resolved_path != ledger_path:
            raise ValueError("Operational amendment changed the canonical ledger path")
        ledger = json.loads(ledger_path.read_text())
        accounted = check_capacity(ledger["attempts"], policy.allocation_usd, reservation)
        snapshot = read_budget_snapshot(ROOT, execution)
        aggregate_cap = enforce_aggregate_attempt_cap(snapshot, {"execution": execution}, reservation)
        experiment_id = "v4-confirmation-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        config = {"experiment_id": experiment_id, "model": model, "model_id": preflight["model_id"],
                  "revision": preflight["revision"], "hidden_size": preflight["hidden_size"],
                  "bucket": "the-shore-508003-shakespeare-activation-geometry",
                  "protocol": protocol, "protocol_sha256": protocol_hash,
                  "preflight_sha256": hashlib.sha256(preflight_bytes).hexdigest(),
                  "preflight_canonical_sha256": hashlib.sha256(canonical_json(preflight)).hexdigest(),
                  "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "resource_envelope": protocol["execution"]["resource_envelopes"][model],
                  "accounted_before_usd": accounted, "model_allocation_usd": policy.allocation_usd,
                  "reserved_usd": reservation, "compute_ceiling_usd": snapshot["approved_total_usd"],
                  "authorization_validated": True, "operational_budget_policy": execution,
                  "aggregate_accounting_ceiling_usd": aggregate_cap,
                  "budget_amendment": amendment_receipt,
                  "precision_source": "same loaded BF16 checkpoint values promoted to FP32"}
        if amendment_receipt is not None:
            pipeline = amendment_receipt["amendment"].get("operational_execution_policy")
            if pipeline is not None:
                config["operational_execution_policy"] = pipeline
        attempt = {"experiment_id": experiment_id, "model": model, "budget_profile": "bounded-confirmation-v4",
                   "resource_envelope": config["resource_envelope"], "reserved_usd": reservation,
                   "status": "reserved", "protocol_sha256": protocol_hash,
                   "approved_compute_ceiling_usd": snapshot["approved_total_usd"],
                   "model_allocation_usd": policy.allocation_usd,
                   "aggregate_accounting_ceiling_usd": aggregate_cap}
        if amendment_receipt is not None:
            attempt.update(budget_amendment_sha256=amendment_receipt["sha256"],
                           recovery_of_failed_attempt_id=amendment_receipt["amendment"]["failed_attempt_id"])
        ledger["attempts"].append(attempt)
        write_ledger_atomic(ledger_path, ledger)
        started = time.monotonic()
        try:
            app, prepare, worker = create_modal_backend(ROOT, config)
            import modal
            with modal.enable_output(), app.run():
                preparation = prepare.remote(config["model_id"], config["revision"])
                result = worker.remote(config, preflight)
            result["preparation"] = preparation
            result["launcher_seconds"] = time.monotonic() - started
            destination = ROOT / "outputs" / experiment_id
            destination.mkdir(exist_ok=False)
            report_path = destination / "report.json"
            write_exclusive(report_path, canonical_json(result))
            attempt.update(report_path=str(report_path), completed_cases=len(result["report"]["cases"]))
            settle_finished(attempt, result["launcher_seconds"], CONTROL_RESOURCE_RATE)
            return {"report": str(report_path), "report_sha256": sha256_file(report_path),
                    "complete": result["report"]["complete"], "completed_cases": len(result["report"]["cases"]),
                    "conservative_accounted_usd": attempt["conservative_compute_bound_usd"]}
        except BaseException:
            attempt["status"] = "failed-reservation-retained"
            raise
        finally:
            write_ledger_atomic(ledger_path, ledger)
    finally:
        lockfile.close()
        global_lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "experiments/confirmation-v4/protocol.json")
    parser.add_argument("--preflight", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--output", type=Path, help="Create-only local JSON plan; omit to print")
    parser.add_argument("--execute", action="store_true", help="Execute only with a matching approved frozen policy")
    parser.add_argument("--model", choices=("qwen", "gemma", "muse"))
    parser.add_argument("--budget-amendment", type=Path, help="Explicit one-time operational recovery; scientific protocol remains frozen")
    args = parser.parse_args(argv)
    preflights = {}
    for argument in args.preflight:
        model, separator, value = argument.partition("=")
        if separator != "=" or not value or model in preflights:
            parser.error("Use one --preflight MODEL=PATH per model")
        preflights[model] = Path(value)
    try:
        if args.execute:
            if args.model not in preflights or len(preflights) != 1 or args.output:
                parser.error("Execution requires --model and exactly its one --preflight; --output is plan-only")
            result = execute_model(args.protocol, args.model, preflights[args.model], args.budget_amendment)
            sys.stdout.buffer.write(canonical_json(result))
        else:
            if args.model or args.budget_amendment:
                parser.error("--model and --budget-amendment are only used with --execute")
            plan = build_execution_plan(ROOT, args.protocol, preflights)
            data = canonical_json(plan)
            if args.output:
                write_exclusive(args.output, data)
            else:
                sys.stdout.buffer.write(data)
    except (ExecutionDisabled, KeyError, ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
