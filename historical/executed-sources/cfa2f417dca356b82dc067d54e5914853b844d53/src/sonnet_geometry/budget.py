"""Conservative launcher accounting; estimates are never described as invoices."""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile


TOTAL_ALLOCATION_USD = 25.0
GPU_RESOURCE_RATE = 0.001261 + 4 * 0.0000131 + 32 * 0.00000222
CONTROL_PROFILE = "bounded-controls-v3"
CONTROL_RESERVATION_USD = 2.60
CONTROL_ENVELOPE = {"gpu_timeout_seconds": 300, "startup_timeout_seconds": 120,
                    "download_timeout_seconds": 600, "soft_timeout_seconds": 240,
                    "scaledown_window_seconds": 2, "max_cpu": 4, "max_memory_gib": 96,
                    "max_containers": 1, "retries": 0}
CONTROL_RESOURCE_RATE = 0.001261 + 4 * 0.0000131 + 96 * 0.00000222


@dataclass(frozen=True)
class ModelBudget:
    allocation_usd: float
    ledger_filename: str
    minimum_reservation_usd: float
    gpu_timeout_seconds: int
    soft_timeout_seconds: int
    startup_timeout_seconds: int = 600
    download_timeout_seconds: int = 1200


MODEL_BUDGETS = {
    "qwen": ModelBudget(10.0, "cost-ledger.json", 5.0, 1800, 1500),
    "gemma": ModelBudget(7.5, "cost-ledger-gemma.json", 7.5, 3600, 3000),
    "muse": ModelBudget(7.5, "cost-ledger-muse.json", 7.5, 3600, 3000),
}


def resolve_model_budget(root: Path, model: str, *, budget_usd=None, reserve_usd=None, ledger=None):
    """Resolve a fixed allocation and its sole ledger; arbitrary ledgers cannot reset spending."""
    if model not in MODEL_BUDGETS:
        raise ValueError("Unknown model budget allocation")
    policy = MODEL_BUDGETS[model]
    budget = policy.allocation_usd if budget_usd is None else budget_usd
    reservation = policy.minimum_reservation_usd if reserve_usd is None else reserve_usd
    if not all(math.isfinite(value) for value in (budget, reservation)):
        raise ValueError("Budget and reservation must be finite")
    if not 0 < budget <= policy.allocation_usd:
        raise ValueError(f"{model} has a fixed ${policy.allocation_usd:.2f} allocation; only a lower budget is allowed")
    if reservation < policy.minimum_reservation_usd or reservation > budget:
        raise ValueError(f"{model} requires a reservation of at least ${policy.minimum_reservation_usd:.2f} within its budget")
    canonical = Path(root).resolve() / "outputs" / policy.ledger_filename
    if canonical.resolve() != canonical:
        raise ValueError("Canonical budget ledger must not resolve through a symlink")
    if ledger is not None and Path(ledger).resolve() != canonical:
        raise ValueError(f"{model} must use its canonical ledger: {canonical}")
    return policy, budget, reservation, canonical


def check_capacity(attempts: list[dict], budget_usd: float, reservation_usd: float) -> float:
    """Pending/failed work retains its full reservation, including concurrent model allocations."""
    if not all(math.isfinite(value) and value > 0 for value in (budget_usd, reservation_usd)):
        raise ValueError("Invalid budget or reservation")
    accounted = sum(accounted_cost(attempt) for attempt in attempts)
    if accounted + reservation_usd > budget_usd:
        raise ValueError("Cumulative conservative accounting plus this reservation exceeds the model allocation")
    return accounted


def resolve_control_budget(root: Path, model: str):
    """A separate, substantially shorter envelope; retain the original allocations/ledgers."""
    policy, _, _, canonical = resolve_model_budget(root, model)
    return policy, CONTROL_RESERVATION_USD, canonical


def resolve_confirmation_budget(root: Path, model: str, execution: dict):
    """Use an explicitly authorized v4 policy without resetting historical ledgers.

    The old runners retain their original allocations. Only a frozen v4 protocol
    containing the owner's new bounded authorization can use the larger limits.
    """
    if (execution.get("authorization_status") != "user_approved_bounded_compute"
            or execution.get("no_paid_execution_this_pass") is not False):
        raise ValueError("A new bounded owner authorization is required")
    total = execution["approved_total_usd"]
    allocations = execution["allocations_usd"]
    reserve = execution["reservation_per_model_usd"]
    if (not math.isfinite(total) or total <= TOTAL_ALLOCATION_USD
            or set(allocations) != set(MODEL_BUDGETS)
            or any(not math.isfinite(v) or v <= 0 for v in allocations.values())
            or sum(allocations.values()) > total or reserve != 5.0):
        raise ValueError("Invalid confirmation ceiling, allocations, or reservation")
    old, _, _, canonical = resolve_model_budget(root, model)
    if any(allocations[m] < MODEL_BUDGETS[m].allocation_usd for m in allocations):
        raise ValueError("Historical model allocations cannot shrink in this confirmation policy")
    envelopes = execution["resource_envelopes"]
    if set(envelopes) != set(MODEL_BUDGETS):
        raise ValueError("All models need a bounded resource envelope")
    for envelope in envelopes.values():
        fixed = {"startup_timeout_seconds": 120, "download_timeout_seconds": 600,
                 "scaledown_window_seconds": 2, "max_cpu": 4, "max_memory_gib": 96,
                 "max_containers": 1, "min_containers": 0, "retries": 0}
        if any(envelope.get(k) != v for k, v in fixed.items()):
            raise ValueError("Confirmation worker resource limits differ from policy")
        gpu, soft = envelope["gpu_timeout_seconds"], envelope["soft_timeout_seconds"]
        if type(gpu) is not int or type(soft) is not int or not 0 < soft <= gpu - 60:
            raise ValueError("Invalid GPU timeout or serialization margin")
        # Include separate startup/scale-down lifecycles for both functions.
        cpu_rate = 4 * .0000131 + 96 * .00000222
        lifecycle_bound = (gpu + 120 + 2) * CONTROL_RESOURCE_RATE + (600 + 120 + 2) * cpu_rate + 1
        coarse_bound = (gpu + 120 + 600 + 2) * CONTROL_RESOURCE_RATE + 1
        bound = max(lifecycle_bound, coarse_bound)
        if bound > reserve:
            raise ValueError("Timeout resource bound exceeds the reserved amount")
    envelope = envelopes[model]
    policy = ModelBudget(allocations[model], old.ledger_filename, reserve,
                         envelope["gpu_timeout_seconds"], envelope["soft_timeout_seconds"], 120, 600)
    return policy, reserve, canonical


def write_ledger_atomic(path: Path, ledger: dict) -> None:
    """Publish one complete ledger snapshot while its model-specific lock is held."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(ledger, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def accounted_cost(attempt: dict) -> float:
    reserved = attempt["reserved_usd"]
    charged = attempt.get("conservative_compute_bound_usd", reserved)
    minimum = 5.0
    if attempt.get("budget_profile") == CONTROL_PROFILE:
        if attempt.get("resource_envelope") != CONTROL_ENVELOPE:
            raise ValueError("Control reservation requires the exact short resource envelope")
        minimum = CONTROL_RESERVATION_USD
    if not all(math.isfinite(x) for x in (reserved, charged)) or reserved < minimum or not 1 <= charged <= reserved:
        raise ValueError("Invalid compute ledger reservation or bound")
    if "conservative_compute_bound_usd" in attempt and attempt.get("status") != "finished":
        raise ValueError("Unfinished attempts must retain their full reservation")
    return charged


def settle_finished(attempt: dict, launcher_wall_seconds: float, resource_rate: float) -> None:
    if not math.isfinite(launcher_wall_seconds) or launcher_wall_seconds <= 0:
        raise ValueError("A positive measured launcher duration is required")
    if not math.isfinite(resource_rate) or resource_rate <= 0:
        raise ValueError("A positive resource rate is required")
    # Charge the FULL GPU+CPU+RAM rate even during CPU download/build/queue time,
    # plus $1 build overhead. Never lower a failed/interrupted reservation.
    # Each model's reservation bounds its configured timeout-limited attempt.
    attempt.update(status="finished", launcher_wall_seconds=launcher_wall_seconds,
                   conservative_compute_bound_usd=min(attempt["reserved_usd"], launcher_wall_seconds * resource_rate + 1),
                   accounting_note="Conservative compute bound: full H200 resource rate across entire launcher wall time plus $1 overhead, capped at the original timeout-based reservation; not invoice cost. Storage and transfer are separate.")
