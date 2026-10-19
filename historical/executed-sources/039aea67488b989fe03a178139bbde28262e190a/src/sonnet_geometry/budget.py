"""Conservative launcher accounting; estimates are never described as invoices."""
import math


def accounted_cost(attempt: dict) -> float:
    reserved = attempt["reserved_usd"]
    charged = attempt.get("conservative_compute_bound_usd", reserved)
    if not all(math.isfinite(x) for x in (reserved, charged)) or reserved < 5 or not 1 <= charged <= reserved:
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
    # The separate $5 reservation bounds the configured timeout-limited attempt.
    attempt.update(status="finished", launcher_wall_seconds=launcher_wall_seconds,
                   conservative_compute_bound_usd=min(attempt["reserved_usd"], launcher_wall_seconds * resource_rate + 1),
                   accounting_note="Conservative compute bound: full H200 resource rate across entire launcher wall time plus $1 overhead, capped at the original timeout-based reservation; not invoice cost. Storage and transfer are separate.")
