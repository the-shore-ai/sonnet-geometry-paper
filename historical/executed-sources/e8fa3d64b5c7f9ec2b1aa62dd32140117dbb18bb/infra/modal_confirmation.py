"""Confirmation planning and explicitly gated, bounded Modal execution.

Importing this module or planning makes no cloud objects or network calls.
Paid execution requires a frozen protocol matching a separately approved policy.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean

from sonnet_geometry.budget import (
    CONTROL_RESOURCE_RATE,
    MODEL_BUDGETS,
    TOTAL_ALLOCATION_USD,
    accounted_cost,
    resolve_model_budget,
)
from sonnet_geometry.storage import canonical_json, sha256_file

MODELS = ("qwen", "gemma", "muse")
MODEL_IDS = {
    "qwen": "Qwen/Qwen3-8B",
    "gemma": "google/gemma-4-31B-it",
    "muse": "meta-models/Muse-Glimmer-30B",
}
CONTROLS = ("word_shuffle", "line_shuffle", "token_shuffle")
STATES = ("input", "early", "depth75", "final_raw", "final_normalized")
POOLINGS = ("mean", "last", "last_lexical")
PRECISIONS = ("bf16", "fp32")
BF16_CHECKPOINT_PROMOTION = (
    "Load the same pinned BF16 checkpoint once; promote those exact checkpoint "
    "values to FP32 for all original and perturbed inputs. Disable TF32 and "
    "autocast, use evaluation/inference mode, and record actual parameter, "
    "capture and arithmetic settings. This does not recover higher-precision weights."
)
FUTURE_WORKER_POLICY = {
    "gpu": "H200", "max_containers": 1, "min_containers": 0,
    "retries": 0, "scaledown_window_seconds": 2, "max_cpu": 4,
    "max_memory_gib": 96, "startup_timeout_seconds": 120,
    "download_timeout_seconds": 600,
    "gpu_timeout_seconds": None,
    "timeout_status": "requires a measured v4 pilot and a validated bounded reservation",
}


class ExecutionDisabled(RuntimeError):
    """This pass cannot reserve or consume any paid compute."""


def study_namespace(protocol: dict) -> str:
    """Keep the new, explicitly authorized population out of historical paths."""
    name = protocol.get("study_namespace", "confirmation-v4")
    if name not in {"confirmation-v4", "confirmation-v5-spenser"}:
        raise ValueError("Unrecognized confirmation study namespace")
    return name


def require_execution_enabled(protocol: dict, root: Path | None = None) -> None:
    """Require the frozen paid scope and centrally validated approved budget policy."""
    execution = protocol.get("execution", {})
    if execution.get("no_paid_execution_this_pass") is not False:
        raise ExecutionDisabled("Paid execution is disabled in the frozen protocol")
    if "study_namespace" in protocol:
        study_namespace(protocol)
        if not validate_operational_execution_policy(execution) or not validate_durable_finalization_policy(execution):
            raise ExecutionDisabled("The new population requires durable detached execution")
    from sonnet_geometry.budget import resolve_confirmation_budget
    root = root or Path(__file__).resolve().parents[1]
    for model in MODELS:
        try:
            resolve_confirmation_budget(root, model, execution)
        except (ValueError, KeyError) as error:
            raise ExecutionDisabled(str(error)) from error
        envelope = execution["resource_envelopes"][model]
        expected = 1800
        if envelope.get("gpu_timeout_seconds") != expected or envelope.get("soft_timeout_seconds") != expected - 60:
            raise ExecutionDisabled("Execution must retain its fixed bounded timeout")
        for key, value in FUTURE_WORKER_POLICY.items():
            if key in {"gpu_timeout_seconds", "timeout_status", "gpu"}:
                continue
            if envelope.get(key) != value:
                raise ExecutionDisabled(f"Execution resource bound changed: {key}")


def validate_protocol(protocol: dict) -> dict:
    if protocol.get("schema_version") != "frozen-depth-confirmation-v4":
        raise ValueError("Unsupported confirmation protocol schema")
    if protocol.get("execution", {}).get("no_paid_execution_this_pass") is not True:
        require_execution_enabled(protocol)
    sonnets = protocol["sonnets"]
    if len(sonnets) != 24 or len(set(sonnets)) != 24 or any(
        type(n) is not int or not 1 <= n <= 154 for n in sonnets
    ):
        raise ValueError("Exactly 24 unique sonnets are required")
    if set(protocol["models"]) != set(MODELS) or len(protocol["models"]) != 3:
        raise ValueError("All three frozen model families are required")
    seeds = protocol["randomization"]["seeds"]
    if len(seeds) != 5 or len(set(seeds)) != 5 or any(type(seed) is not int for seed in seeds):
        raise ValueError("Exactly five distinct integer randomization seeds are required")
    if set(protocol["states"]) != set(STATES) or protocol["poolings"] != list(POOLINGS):
        raise ValueError("Five frozen states and three frozen poolings are required")
    pad_length = protocol["pad_length"]
    if type(pad_length) is not int or pad_length <= 0:
        raise ValueError("A positive fixed padded forward length is required")
    conditions = protocol["conditions"]
    expected = {(control, seed, precision) for control in CONTROLS for seed in seeds for precision in PRECISIONS}
    expected.update(("original", None, precision) for precision in PRECISIONS)
    if len(conditions) != 32 or {(c["intervention"], c["seed"], c["precision"]) for c in conditions} != expected:
        raise ValueError("Every original and five replicates of each control must appear in both precisions")
    for condition in conditions:
        prefix = condition["intervention"]
        if condition["seed"] is not None:
            prefix += f"_seed{condition['seed']}"
        if condition["name"] != f"{prefix}_padded_{condition['precision']}":
            raise ValueError("Noncanonical condition name")
    return {"poems": len(sonnets), "randomization_seeds": len(seeds), "conditions": len(conditions),
            "unique_inputs_per_model": len(sonnets) * len(conditions) // 2,
            "forwards_per_model": len(sonnets) * len(conditions),
            "forwards_total": len(sonnets) * len(conditions) * len(MODELS)}


def forward_payload(case: dict) -> dict:
    """The exact arrays sent to a future forward, including masked padding IDs."""
    ids = case["tokenization"]["input_ids"]
    length, pad_id = case["forward_length"], case["pad_token_id"]
    if type(length) is not int or not ids or len(ids) > length:
        raise ValueError("Empty or truncated forward input")
    if any(type(token) is not int or token < 0 for token in ids) or type(pad_id) is not int or pad_id < 0:
        raise ValueError("Token IDs must be nonnegative integers")
    mask, positions = case["attention_mask"], case["position_ids"]
    if mask != [1] * len(ids) + [0] * (length - len(ids)):
        raise ValueError("Invalid real-token/padding attention mask")
    if positions != list(range(length)):
        raise ValueError("Position IDs must match the frozen padded positions")
    return {"input_ids": ids + [pad_id] * (length - len(ids)),
            "attention_mask": mask, "position_ids": positions}


def forward_sha256(case: dict) -> str:
    return hashlib.sha256(canonical_json(forward_payload(case))).hexdigest()


def model_lock(root: Path, model: str) -> dict:
    candidates = []
    for name in ("models.lock.json", "models-candidates.lock.json"):
        candidates.extend(json.loads((root / "configs" / name).read_text())["models"])
    matches = [entry for entry in candidates if entry["model_id"] == MODEL_IDS[model]]
    revisions = {entry["revision"] for entry in matches}
    if len(revisions) != 1 or not re.fullmatch(r"[a-f0-9]{40}", next(iter(revisions), "")):
        raise ValueError("Missing or conflicting pinned model revision")
    return {"model_id": MODEL_IDS[model], "revision": next(iter(revisions))}


def validate_preflight(root: Path, protocol: dict, protocol_sha256: str, manifest: dict) -> dict:
    counts = validate_protocol(protocol)
    model = manifest["model"]
    if model not in MODELS or manifest["protocol_sha256"] != protocol_sha256:
        raise ValueError("Preflight model/protocol mismatch")
    if any(manifest.get(key) != value for key, value in model_lock(root, model).items()):
        raise ValueError("Preflight checkpoint differs from the pinned model")
    tokenizer_hash = manifest["tokenizer_sha256"]
    if not re.fullmatch(r"[a-f0-9]{64}", tokenizer_hash):
        raise ValueError("A pinned tokenizer backend hash is required")
    cases = manifest["cases"]
    conditions = {condition["name"]: condition for condition in protocol["conditions"]}
    expected = {(n, condition) for n in protocol["sonnets"] for condition in conditions}
    actual = [(case["sonnet"], case["condition"]) for case in cases]
    if len(cases) != counts["forwards_per_model"] or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Preflight must contain exactly every poem/condition pair")
    pairs = {}
    for case in cases:
        if case["forward_length"] != protocol["pad_length"]:
            raise ValueError("Preflight changed the frozen padded length")
        payload = forward_payload(case)
        if forward_sha256(case) != case["forward_sha256"]:
            raise ValueError("Forward hash mismatch")
        if "input_ids_sha256" in case and hashlib.sha256(canonical_json(case["tokenization"]["input_ids"])).hexdigest() != case["input_ids_sha256"]:
            raise ValueError("Real-token ID hash mismatch")
        condition = conditions[case["condition"]]
        key = (case["sonnet"], condition["intervention"], condition["seed"])
        pairs.setdefault(key, {})[condition["precision"]] = (payload, case["tokenization"])
    for pair in pairs.values():
        if set(pair) != set(PRECISIONS) or pair["bf16"] != pair["fp32"]:
            raise ValueError("BF16/FP32 input arrays and tokenization must be identical")
    return {"model": model, "complete": True, "cases": len(cases), "precision_pairs": len(pairs),
            "protocol_sha256": protocol_sha256, "tokenizer_sha256": tokenizer_hash,
            **model_lock(root, model)}


def read_budget_snapshot(root: Path, execution: dict | None = None) -> dict:
    """Read only the existing canonical ledgers; never create or reset a ledger."""
    if execution is not None:
        require_execution_enabled({"execution": execution})
    ceiling = TOTAL_ALLOCATION_USD if execution is None else execution["approved_total_usd"]
    models = []
    for model in MODELS:
        if execution is None:
            policy, _, _, ledger_path = resolve_model_budget(root, model)
        else:
            from sonnet_geometry.budget import resolve_confirmation_budget
            policy, _, ledger_path = resolve_confirmation_budget(root, model, execution)
        payload = ledger_path.read_bytes()  # Missing history is a hard failure, not an empty ledger.
        ledger = json.loads(payload)
        attempts = ledger["attempts"]
        if not isinstance(attempts, list) or not attempts:
            raise ValueError("Canonical ledger history must exist")
        accounted = sum(accounted_cost(attempt) for attempt in attempts)
        allocation = policy.allocation_usd if execution is None else execution["allocations_usd"][model]
        if accounted > allocation:
            raise ValueError("Canonical accounting already exceeds its fixed allocation")
        models.append({"model": model, "ledger": str(ledger_path.relative_to(root)),
                       "ledger_sha256": hashlib.sha256(payload).hexdigest(), "attempts": len(attempts),
                       "allocation_usd": allocation, "original_allocation_usd": MODEL_BUDGETS[model].allocation_usd,
                       "accounted_usd": accounted, "remaining_usd": allocation - accounted})
    total = sum(row["accounted_usd"] for row in models)
    return {"approved_total_usd": ceiling, "accounted_total_usd": total,
            "remaining_total_usd": ceiling - total, "models": models,
            "accounting_is_invoice": False, "storage_and_transfer_included": False,
            "ledger_writes": 0, "reservations_created": 0}


def measured_v3_projection(root: Path, model: str, forwards_per_precision: int, budget: dict) -> dict:
    """Historical timing extrapolation, never a validated v4 budget/reservation."""
    index_path = root / "experiments/followup-v3/text-execution-index.json"
    index = json.loads(index_path.read_text())
    record, = [entry for entry in index["executions"] if entry["model"] == model]
    report_path = root / "outputs/followup-v3/text-controls" / model / "report.json"
    digest = sha256_file(report_path)
    if digest != record["local_report_sha256"]:
        raise ValueError("Historical timing report checksum mismatch")
    wrapper = json.loads(report_path.read_text())
    report = wrapper["report"]
    if not report["complete"] or report["fp32_status"] != "complete" or report["model"] != model:
        raise ValueError("Incomplete or wrong-model historical timing report")
    groups = {precision: [] for precision in PRECISIONS}
    for case in report["cases"]:
        if "native" in case["condition"]:
            continue
        precision = case["condition"].rsplit("_", 1)[1]
        elapsed = case["elapsed_seconds"]
        if precision not in groups or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Invalid historical case timing")
        groups[precision].append(elapsed)
    if {key: len(value) for key, value in groups.items()} != {"bf16": 48, "fp32": 12}:
        raise ValueError("Historical padded timing coverage changed")
    # Case durations include capture/upload; startup/preparation are accounted separately.
    case_sum = sum(case["elapsed_seconds"] for case in report["cases"])
    noncase_function = max(0.0, report["function_seconds"] - case_sum)
    nongpu_launcher = max(0.0, wrapper["launcher_seconds"] - report["function_seconds"])
    typical_cases = forwards_per_precision * sum(mean(values) for values in groups.values())
    stressed_cases = 2 * forwards_per_precision * sum(max(values) for values in groups.values())
    estimated_function_seconds = stressed_cases + noncase_function + 60
    proposed_timeout = math.ceil(estimated_function_seconds / 60) * 60
    envelope_seconds = (proposed_timeout + FUTURE_WORKER_POLICY["startup_timeout_seconds"]
                        + FUTURE_WORKER_POLICY["download_timeout_seconds"] + 2)
    illustrative_reservation = 1 + CONTROL_RESOURCE_RATE * envelope_seconds
    point_estimate = 1 + CONTROL_RESOURCE_RATE * (typical_cases + noncase_function + nongpu_launcher)
    return {"status": "historical_extrapolation_only_not_validated_for_execution",
            "source": str(report_path.relative_to(root)), "source_sha256": digest,
            "timing_samples": {key: len(value) for key, value in groups.items()},
            "mean_case_seconds": {key: mean(value) for key, value in groups.items()},
            "max_case_seconds": {key: max(value) for key, value in groups.items()},
            "projected_forwards_per_precision": forwards_per_precision,
            "historical_launcher_nonfunction_seconds": nongpu_launcher,
            "historical_function_noncase_seconds": noncase_function,
            "point_accounting_estimate_usd": point_estimate,
            "planning_margin": "2x each precision's maximum historical case duration plus 60s serialization",
            "illustrative_gpu_timeout_seconds": proposed_timeout,
            "illustrative_reservation_usd": illustrative_reservation,
            "fixed_allocation_remaining_usd": budget["remaining_usd"],
            "illustrative_reservation_fits_fixed_allocation": illustrative_reservation <= budget["remaining_usd"],
            "minimum_one_dollar_accounting_overhead_fits": budget["remaining_usd"] >= 1,
            "resource_rate_usd_per_second": CONTROL_RESOURCE_RATE,
            "rate_source": "frozen v3 rate snapshot; must reverify before any future execution",
            "validated_budget_feasible": False,
            "limitations": ["No measured v4 pilot; new population and perturbed FP32 inputs are unmeasured",
                            "CPU/image/download/queue duration and storage behavior may differ",
                            "Illustrative resource bounds are not a reservation or provider invoice"]}


def storage_upper_bound(root: Path, model: str, counts: dict, pad_length: int) -> dict:
    if model == "qwen":
        width = 4096
    else:
        config_path = root / "configs" / ({"gemma": "google--gemma-4-31B-it.config.json",
                                          "muse": "meta-models--Muse-Glimmer-30B.config.json"}[model])
        config = json.loads(config_path.read_text())
        width = config.get("text_config", config)["hidden_size"]
    forwards = counts["unique_inputs_per_model"]
    # Five states per input, retained in original 2/4-byte output dtypes.
    primary = forwards * pad_length * width * len(STATES) * (2 + 4)
    pooled = counts["forwards_per_model"] * width * len(STATES) * len(POOLINGS) * 8
    return {"hidden_width": width, "maximum_real_tokens_per_input": pad_length,
            "selected_state_tensor_upper_bytes": primary,
            "float64_pooled_tensor_bytes": pooled,
            "metadata_and_container_overhead_included": False,
            "note": "Save real-token states only, preserving BF16/FP32 capture dtype; metadata is additional"}


def build_execution_plan(root: Path, protocol_path: Path, preflights: dict[str, Path] | None = None) -> dict:
    root = root.resolve()
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    digest = hashlib.sha256(protocol_bytes).hexdigest()
    counts = validate_protocol(protocol)
    execution = protocol["execution"] if protocol["execution"]["no_paid_execution_this_pass"] is False else None
    before = read_budget_snapshot(root, execution)
    validations = {}
    for model, path in (preflights or {}).items():
        manifest = json.loads(path.read_text())
        if manifest["model"] != model:
            raise ValueError("Preflight argument/model mismatch")
        validations[model] = {**validate_preflight(root, protocol, digest, manifest),
                              "path": str(path), "sha256": sha256_file(path)}
    projections = []
    for budget in before["models"]:
        projection = measured_v3_projection(root, budget["model"], counts["unique_inputs_per_model"], budget)
        projection.update(model=budget["model"],
                          storage=storage_upper_bound(root, budget["model"], counts, protocol["pad_length"]))
        projections.append(projection)
    after = read_budget_snapshot(root, execution)
    if before != after:
        raise ValueError("Canonical ledgers changed while planning; regenerate the local plan")
    return {"schema_version": "confirmation-execution-plan-v4", "status": "local_plan_no_cloud_actions",
            "protocol_sha256": digest, "counts": counts, "budget": before,
            "worker_policy": FUTURE_WORKER_POLICY, "precision_semantics": BF16_CHECKPOINT_PROMOTION,
            "preflight_validations": validations, "preflights_complete": set(validations) == set(MODELS),
            "projections": projections, "paid_execution_enabled": execution is not None, "gpu_backend_implemented": True,
            "validated_budget_feasible": execution is not None and all(
                row["remaining_usd"] >= execution["reservation_per_model_usd"] for row in before["models"]),
            "approved_reservation_total_usd": 0 if execution is None else 3 * execution["reservation_per_model_usd"],
            "approved_resource_envelopes": None if execution is None else execution["resource_envelopes"],
            "cloud_apps_created": 0, "cloud_objects_written": 0,
            "illustrative_total_reservation_usd": sum(p["illustrative_reservation_usd"] for p in projections),
            "selected_state_storage_upper_bytes": sum(p["storage"]["selected_state_tensor_upper_bytes"] for p in projections),
            "future_requirements": [
                "Explicitly revise the current no-paid-execution scope before any paid work",
                "Measured v4 pilot and conservative bounded budget fitting each unchanged model allocation",
                "Backend runs one worker/model, no retries, hard timeout, scale to zero",
                "Reserve only in each canonical historical ledger, with locks and retained failures",
                "Hash-check every frozen input; never retokenize during execution",
                "Both precisions for every original and perturbed case, TF32/autocast off",
                "Private create-only GCS artifacts, source dtype, full-readback hashes and immutable receipts",
                "Complete CPU checks and clean committed source before any future execution",
            ]}


def promote_precision(model):
    """Promote exact loaded values, retaining original mixed buffer/parameter dtypes."""
    original = {
        "parameter_dtypes": {name: parameter.dtype for name, parameter in model.named_parameters()},
        "buffers": {name: buffer.detach().clone() for name, buffer in model.named_buffers()},
    }
    import torch
    if any(tensor.dtype in {torch.float64, torch.complex64, torch.complex128}
           for tensor in (*model.parameters(), *model.buffers())):
        raise ValueError("Unexpected higher-precision or complex source parameters/buffers")
    model.float()
    return original


def restore_precision(model, original):
    """Restore pilot BF16 weights without downcasting originally FP32 buffers."""
    for name, parameter in model.named_parameters():
        parameter.data = parameter.data.to(dtype=original["parameter_dtypes"][name])
    for name, buffer in original["buffers"].items():
        parent, _, leaf = name.rpartition(".")
        module = model.get_submodule(parent) if parent else model
        module._buffers[leaf] = buffer


def execution_phases(protocol: dict, preflight: dict):
    """Disjoint pilot/remainder partitions; each frozen forward occurs exactly once."""
    pilot_poems = set(protocol["sonnets"][:2])
    conditions = {condition["name"]: condition for condition in protocol["conditions"]}
    phases = []
    for phase, precision, pilot in (("pilot_bf16", "bf16", True), ("pilot_fp32", "fp32", True),
                                    ("remaining_bf16", "bf16", False), ("remaining_fp32", "fp32", False)):
        cases = [case for case in preflight["cases"] if (case["sonnet"] in pilot_poems) == pilot
                 and conditions[case["condition"]]["precision"] == precision]
        phases.append((phase, precision, cases))
    keys = [(case["sonnet"], case["condition"]) for _, _, cases in phases for case in cases]
    if len(keys) != len(preflight["cases"]) or len(set(keys)) != len(keys):
        raise ValueError("Execution phases must partition all frozen cases exactly once")
    return phases


UPLOAD_PIPELINE_POLICY = {
    "schema_version": "confirmation-upload-pipeline-v1", "max_pending_uploads": 4,
    "pilot_projection": "twice_drained_phase_seconds_per_case", "finalization_margin_seconds": 60,
}

DURABLE_FINALIZATION_POLICY = {
    "schema_version": "confirmation-durable-finalization-v1",
    "runtime_checkpoint_before_each_phase": True,
    "runtime_checkpoint_phases": ["pilot_bf16", "pilot_fp32", "remaining_bf16", "remaining_fp32"],
    "capture_completion_checkpoint_before_pooling": True,
    "create_only_checkpoint_writes": True,
    "detached_app": True,
    "single_spawn": True,
}


def validate_durable_finalization_policy(config: dict) -> bool:
    """Only an exact explicit operational policy enables the extra evidence."""
    if "durable_finalization_policy" not in config:
        return False
    policy = config["durable_finalization_policy"]
    # Canonical bytes distinguish actual booleans from numerically equal 0/1.
    if type(policy) is not dict or canonical_json(policy) != canonical_json(DURABLE_FINALIZATION_POLICY):
        raise ValueError("Unrecognized durable finalization policy")
    return True


def validate_operational_execution_policy(config: dict) -> bool:
    policy = config.get("operational_execution_policy")
    if policy is None:
        return False
    if policy != UPLOAD_PIPELINE_POLICY:
        raise ValueError("Unrecognized operational upload pipeline or pilot projection policy")
    return True


class BoundedUploadPipeline:
    """At most four immutable byte jobs; every error propagates after draining."""
    def __init__(self, writer, max_pending=4):
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        if type(max_pending) is not int or not 1 <= max_pending <= 4:
            raise ValueError("Upload concurrency must be bounded between one and four")
        self.writer = writer
        self.max_pending = max_pending
        self.executor = ThreadPoolExecutor(max_workers=max_pending, thread_name_prefix="confirmation-upload")
        self.pending = deque()
        self.ready = []
        self.failures = []
        self.closed = False
        self.maximum_observed_pending = 0

    def _consume_oldest(self):
        future = self.pending.popleft()
        try:
            self.ready.append(future.result())
        except Exception as error:
            self.failures.append(error)

    def submit(self, states_bytes, metadata_bytes, *context):
        if self.closed:
            raise RuntimeError("Upload pipeline is closed")
        if type(states_bytes) is not bytes or type(metadata_bytes) is not bytes:
            raise TypeError("Queued state and metadata payloads must own immutable bytes")
        if len(self.pending) >= self.max_pending:
            self._consume_oldest()
        if self.failures:
            self.drain()  # Join all accepted jobs and propagate every failure before accepting another.
        self.pending.append(self.executor.submit(self.writer, states_bytes, metadata_bytes, *context))
        self.maximum_observed_pending = max(self.maximum_observed_pending, len(self.pending))

    def drain(self):
        while self.pending:
            self._consume_oldest()
        if self.failures:
            raise ExceptionGroup("Confirmation upload jobs failed", list(self.failures))
        result, self.ready = self.ready, []
        return result

    def close(self):
        try:
            return self.drain()
        finally:
            self.closed = True
            self.executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def throughput_pilot_decision(phases, remaining_counts, elapsed_seconds, soft_timeout_seconds):
    """Two times fully drained phase throughput, with an explicit finalization reserve."""
    if not all(math.isfinite(value) and value >= 0 for value in (elapsed_seconds, soft_timeout_seconds)):
        raise ValueError("Pilot timing must be finite and nonnegative")
    measured = {}
    for precision in PRECISIONS:
        matches = [phase for phase in phases if phase["phase"] == "pilot_" + precision]
        if len(matches) != 1:
            raise ValueError("Both complete precision pilot phases are required")
        phase = matches[0]
        if (phase["completed"] != 32 or phase.get("uploads_drained") is not True
                or not math.isfinite(phase["seconds"]) or phase["seconds"] <= 0):
            raise ValueError("Pilot phase must include all 32 cases and completed metadata uploads")
        if type(remaining_counts.get(precision)) is not int or remaining_counts[precision] < 0:
            raise ValueError("A fixed nonnegative remaining case count is required")
        measured[precision] = phase["seconds"] / phase["completed"]
    projected = 2 * sum(measured[precision] * remaining_counts[precision] for precision in PRECISIONS)
    margin = UPLOAD_PIPELINE_POLICY["finalization_margin_seconds"]
    return {"projection_method": UPLOAD_PIPELINE_POLICY["pilot_projection"],
            "phase_seconds_per_case": measured, "remaining_case_counts": dict(remaining_counts),
            "remaining_projection_seconds": projected, "elapsed_seconds": elapsed_seconds,
            "finalization_margin_seconds": margin, "soft_timeout_seconds": soft_timeout_seconds,
            "proceed": elapsed_seconds + projected + margin < soft_timeout_seconds}


def create_modal_backend(root: Path, config: dict):
    """Create bounded cloud resources only after the launcher validates authorization.

    This function is never called by plan mode. All imports and resource creation
    live inside this function so importing this module has no cloud side effects.
    """
    import os
    validate_operational_execution_policy(config)
    validate_durable_finalization_policy(config)
    require_execution_enabled(config["protocol"])
    import modal

    execution = config["protocol"]["execution"]
    if execution.get("no_paid_execution_this_pass") is not False or not config.get("authorization_validated"):
        raise ExecutionDisabled("Paid execution requires a separately validated approved policy")
    envelope = config["resource_envelope"]
    if any(envelope.get(key) != value for key, value in {
        "max_containers": 1, "min_containers": 0, "retries": 0,
        "scaledown_window_seconds": 2, "max_cpu": 4, "max_memory_gib": 96,
    }.items()):
        raise ValueError("Unbounded or altered worker policy")
    runtime = root if config["model"] == "qwen" else root / "infra/modern_runtime"
    app = modal.App("shore-shakespeare-" + study_namespace(config["protocol"]))
    image = (modal.Image.debian_slim(python_version="3.12")
             .uv_sync(str(runtime), frozen=True, uv_version="0.9.8")
             .env({"PYTHONPATH": "/research:/research/src", "HF_HOME": "/model-cache",
                   "TOKENIZERS_PARALLELISM": "false", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "4"})
             .add_local_dir(root / "src", "/research/src")
             .add_local_dir(root / "infra", "/research/infra"))
    cache = modal.Volume.from_name("shore-shakespeare-confirmation-cache", create_if_missing=True)
    gcp = modal.Secret.from_dict({"GCP_SERVICE_ACCOUNT_JSON": Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).read_text()})
    hf = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN")})

    @app.function(image=image, serialized=True, volumes={"/model-cache": cache}, secrets=[hf], cpu=(2, 4),
                  memory=(4096, 98304), timeout=envelope["download_timeout_seconds"],
                  startup_timeout=envelope["startup_timeout_seconds"], retries=0,
                  min_containers=0, max_containers=1, scaledown_window=2)
    def prepare(model_id, revision):
        import time
        from huggingface_hub import snapshot_download
        from sonnet_geometry.extraction import require_revision
        started = time.monotonic()
        snapshot_download(model_id, revision=require_revision(revision),
                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
        cache.commit()
        return {"seconds": time.monotonic() - started, "gpu_seconds": 0}

    @app.function(image=image, serialized=True, volumes={"/model-cache": cache}, secrets=[gcp, hf], gpu="H200",
                  cpu=(4, 4), memory=(32768, 98304), timeout=envelope["gpu_timeout_seconds"],
                  startup_timeout=envelope["startup_timeout_seconds"], retries=0,
                  min_containers=0, max_containers=1, scaledown_window=2)
    def run_confirmation(run_config, preflight):
        import tempfile
        import threading
        import time
        import torch
        from google.cloud import storage
        from google.oauth2 import service_account
        from safetensors.torch import save_file
        from sonnet_geometry.adapters import get_adapter
        from sonnet_geometry.extraction import load_model, _environment, _tokenizer_metadata
        from sonnet_geometry.text_controls import capture_selected, pool_selected

        started = time.monotonic()
        pipeline_enabled = validate_operational_execution_policy(run_config)
        durable_finalization = validate_durable_finalization_policy(run_config)
        protocol = run_config["protocol"]
        if protocol["execution"].get("no_paid_execution_this_pass") is not False:
            raise ExecutionDisabled("Frozen protocol still disallows paid execution")
        if preflight["protocol_sha256"] != run_config["protocol_sha256"]:
            raise ValueError("Runtime preflight/protocol mismatch")
        if hashlib.sha256(canonical_json(preflight)).hexdigest() != run_config["preflight_canonical_sha256"]:
            raise ValueError("Runtime preflight hash mismatch")
        expected = {(n, c["name"]) for n in protocol["sonnets"] for c in protocol["conditions"]}
        actual = [(c["sonnet"], c["condition"]) for c in preflight["cases"]]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Runtime preflight completeness mismatch")
        for case in preflight["cases"]:
            if forward_sha256(case) != case["forward_sha256"]:
                raise ValueError("Frozen runtime forward input hash mismatch")
        creds = service_account.Credentials.from_service_account_info(json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"]))
        bucket = storage.Client(project=creds.project_id, credentials=creds).bucket(run_config["bucket"])
        bucket.reload()
        if (bucket.iam_configuration.public_access_prevention != "enforced"
                or not bucket.iam_configuration.uniform_bucket_level_access_enabled):
            raise ValueError("Private storage required")
        prefix = f"{study_namespace(protocol)}/{run_config['model']}/{run_config['experiment_id']}"
        receipts = []
        receipts_lock = threading.Lock()

        def put_bytes(name, data):
            digest = hashlib.sha256(data).hexdigest()
            blob = bucket.blob(f"{prefix}/{name}")
            blob.metadata = {"sha256": digest}
            blob.upload_from_string(data, if_generation_match=0, checksum="auto")
            receipt = {"name": blob.name, "uri": f"gs://{bucket.name}/{blob.name}",
                       "generation": str(blob.generation), "bytes": len(data), "sha256": digest,
                       "full_readback_verified": False}
            with receipts_lock:
                receipts.append(receipt)
            return receipt

        put_bytes("config.json", canonical_json(run_config))
        preflight_receipt = put_bytes("preflight.json", canonical_json(preflight))
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        model, tokenizer = load_model(run_config["model_id"], run_config["revision"], dtype="bfloat16", device="cuda")
        model.eval()
        backend_bytes = tokenizer.backend_tokenizer.to_str().encode()
        if hashlib.sha256(backend_bytes).hexdigest() != preflight["tokenizer_sha256"]:
            raise ValueError("Loaded tokenizer backend differs from preflight")
        env_bf16 = _environment(torch, model)
        put_bytes("model-config.json", canonical_json(model.config.to_dict()))
        put_bytes("adapter.json", canonical_json(get_adapter(model).describe()))
        put_bytes("loading-info.json", canonical_json(getattr(model, "_sonnet_geometry_loading_info", {})))
        put_bytes("tokenizer-backend.json", backend_bytes)
        blocks = {state: protocol["states"][state][run_config["model"]]
                  for state in ("early", "depth75", "final_raw")}
        condition_map = {condition["name"]: condition for condition in protocol["conditions"]}
        completed, pooled, phases, environments = [], {}, [], {"bf16": env_bf16}
        runtime_checkpoints = []
        upload_stage_timings = []
        maximum_pending_uploads = 0
        pilot_poems = set(protocol["sonnets"][:2])
        stop_reason = None
        soft_stop = run_config["resource_envelope"]["soft_timeout_seconds"]

        def persist_case(state_bytes, metadata_bytes, key, case_started, submitted):
            upload_started = time.monotonic()
            metadata = json.loads(metadata_bytes)
            state_started = time.monotonic()
            states = put_bytes(f"{key}/states.safetensors", state_bytes)
            state_seconds = time.monotonic() - state_started
            metadata["primary_states"] = states
            metadata["elapsed_seconds"] = time.monotonic() - case_started
            metadata["stage_timings"].update(state_upload_seconds=state_seconds,
                                             upload_queue_wait_seconds=upload_started - submitted)
            metadata_started = time.monotonic()
            put_bytes(f"{key}/metadata.json", canonical_json(metadata))
            upload_timing = {"sonnet": metadata["sonnet"], "condition": metadata["condition"],
                             "state_bytes": len(state_bytes),
                             "metadata_upload_seconds": time.monotonic() - metadata_started,
                             "upload_worker_seconds": time.monotonic() - upload_started,
                             "total_case_wall_seconds": time.monotonic() - case_started}
            return metadata, upload_timing

        def accept_completed(results):
            for metadata, upload_timing in results:
                completed.append(metadata)
                upload_stage_timings.append(upload_timing)
                print(json.dumps({"sonnet": metadata["sonnet"], "condition": metadata["condition"],
                                  "completed": len(completed), "seconds": metadata["elapsed_seconds"],
                                  "stage_timings": metadata["stage_timings"]}), flush=True)

        def execute(case, pipeline=None):
            before = time.monotonic()
            if forward_sha256(case) != case["forward_sha256"]:
                raise ValueError("Runtime input mutated after frozen validation")
            # Only this main thread performs model operations. Upload jobs own bytes.
            capture_started = time.monotonic()
            with torch.autocast(device_type="cuda", enabled=False):
                autocast_enabled = torch.is_autocast_enabled("cuda")
                captures = capture_selected(model, case["tokenization"], blocks=blocks,
                                            pad_to=case["forward_length"], pad_id=case["pad_token_id"])
            capture_seconds = time.monotonic() - capture_started
            expected_dtype = torch.bfloat16 if condition_map[case["condition"]]["precision"] == "bf16" else torch.float32
            if any(tensor.dtype != expected_dtype for tensor in captures.values()):
                raise ValueError("Captured source precision differs from the frozen condition")
            pool_started = time.monotonic()
            for metric, vector in pool_selected(captures, case["tokenization"]).items():
                pooled[f"{case['sonnet']:03d}.{case['condition']}.{metric}"] = torch.from_numpy(vector.copy())
            pool_seconds = time.monotonic() - pool_started
            key = f"sonnet-{case['sonnet']:03d}/{case['condition']}"
            serialize_started = time.monotonic()
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "states.safetensors"
                save_file(captures, str(path), metadata={"selected_states_only": "true", "padding_tokens_stored": "false"})
                state_bytes = path.read_bytes()
            serialize_seconds = time.monotonic() - serialize_started
            metadata = {**case, "blocks": blocks, "dtypes": {k: str(v.dtype) for k, v in captures.items()},
                        "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
                        "tf32_cudnn_allowed": torch.backends.cudnn.allow_tf32, "autocast_enabled": autocast_enabled,
                        "source_dtype": str(next(model.parameters()).dtype),
                        "stage_timings": {"capture_seconds": capture_seconds, "pool_seconds": pool_seconds,
                                          "serialize_seconds": serialize_seconds}}
            metadata_bytes = canonical_json(metadata)
            submitted = time.monotonic()
            if pipeline is None:
                accept_completed([persist_case(state_bytes, metadata_bytes, key, before, submitted)])
            else:
                pipeline.submit(state_bytes, metadata_bytes, key, before, submitted)

        original_precision = None
        # Both precisions on two complete poems are measured before scaling.
        for phase, precision, phase_cases in execution_phases(protocol, preflight):
            if phase == "pilot_fp32":
                original_precision = promote_precision(model)
                torch.cuda.empty_cache()
                environments["fp32"] = _environment(torch, model)
            elif phase == "remaining_bf16":
                restore_precision(model, original_precision)
                original_precision = None
                torch.cuda.empty_cache()
                pilot_times = {p: [case["elapsed_seconds"] for case in completed
                                   if condition_map[case["condition"]]["precision"] == p] for p in PRECISIONS}
                remaining_counts = Counter(condition_map[case["condition"]]["precision"]
                                           for case in preflight["cases"] if case["sonnet"] not in pilot_poems)
                if pipeline_enabled:
                    decision = throughput_pilot_decision(phases, remaining_counts,
                                                         time.monotonic() - started, soft_stop)
                    decision.update(pilot_poems=sorted(pilot_poems), completed_cases=len(completed))
                else:
                    predicted_remaining = 2 * sum(max(pilot_times[p]) * remaining_counts[p] for p in PRECISIONS)
                    decision = {"pilot_poems": sorted(pilot_poems), "completed_cases": len(completed),
                                "remaining_case_counts": dict(remaining_counts),
                                "max_case_seconds": {p: max(pilot_times[p]) for p in PRECISIONS},
                                "remaining_projection_seconds": predicted_remaining,
                                "elapsed_seconds": time.monotonic() - started,
                                "soft_timeout_seconds": soft_stop,
                                "proceed": time.monotonic() - started + predicted_remaining < soft_stop}
                put_bytes("pilot-decision.json", canonical_json(decision))
                if not decision["proceed"]:
                    stop_reason = "Measured two-poem both-precision pilot cannot finish within frozen soft timeout"
                    break
            elif phase == "remaining_fp32":
                model.float()
                torch.cuda.empty_cache()
            if durable_finalization:
                # Persist actual model-wide evidence before any forward in this
                # precision phase. It must survive loss of the launcher or final
                # serialization; another run's evidence cannot replace it.
                checkpoint = {
                    "schema_version": "confirmation-runtime-checkpoint-v4",
                    "experiment_id": run_config["experiment_id"], "model": run_config["model"],
                    "model_id": run_config["model_id"], "revision": run_config["revision"],
                    "protocol_sha256": run_config["protocol_sha256"],
                    "preflight_sha256": run_config["preflight_sha256"],
                    "phase": phase, "precision": precision,
                    "environment": _environment(torch, model), "tokenizer": _tokenizer_metadata(tokenizer),
                    "model_training": model.training,
                    "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
                    "tf32_cudnn_allowed": torch.backends.cudnn.allow_tf32,
                    "autocast_enabled": torch.is_autocast_enabled("cuda"),
                    "completed_case_count_before_phase": len(completed),
                    "function_elapsed_seconds": time.monotonic() - started,
                }
                checkpoint_receipt = put_bytes(f"runtime/{phase}.json", canonical_json(checkpoint))
                runtime_checkpoints.append({"receipt": checkpoint_receipt, "payload": checkpoint})
            phase_started = time.monotonic()
            count_before = len(completed)
            pipeline = BoundedUploadPipeline(persist_case, max_pending=4) if pipeline_enabled else None
            try:
                for case in phase_cases:
                    if time.monotonic() - started > soft_stop:
                        stop_reason = "Soft timeout reached; partial immutable results retained without automatic retry"
                        break
                    execute(case, pipeline)
            finally:
                if pipeline is not None:
                    # A phase ends only after both state and metadata writes finish.
                    accept_completed(pipeline.close())
                    maximum_pending_uploads = max(maximum_pending_uploads, pipeline.maximum_observed_pending)
            phases.append({"phase": phase, "completed": len(completed) - count_before,
                           "seconds": time.monotonic() - phase_started, "uploads_drained": True})
            if stop_reason:
                break
        capture_completion_checkpoint = None
        if durable_finalization:
            # This records all drained captures before the large pooled upload.
            # It is observed partial/complete evidence, not a substitute report.
            checkpoint = {
                "schema_version": "confirmation-capture-completion-v4",
                "experiment_id": run_config["experiment_id"], "model": run_config["model"],
                "model_id": run_config["model_id"], "revision": run_config["revision"],
                "protocol_sha256": run_config["protocol_sha256"],
                "preflight_sha256": run_config["preflight_sha256"],
                "environment_bf16": environments["bf16"], "environment_fp32": environments.get("fp32"),
                "tokenizer": _tokenizer_metadata(tokenizer), "cases": completed, "phases": phases,
                "complete": len(completed) == len(expected), "stop_reason": stop_reason,
                "upload_stage_timings": upload_stage_timings, "maximum_pending_uploads": maximum_pending_uploads,
                "runtime_checkpoint_receipts": [item["receipt"] for item in runtime_checkpoints],
                "objects": list(receipts), "function_seconds_at_capture_completion": time.monotonic() - started,
            }
            checkpoint_receipt = put_bytes("capture-completion.json", canonical_json(checkpoint))
            capture_completion_checkpoint = {"receipt": checkpoint_receipt, "payload": checkpoint}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pooled.safetensors"
            save_file(pooled, str(path))
            pooled_receipt = put_bytes("pooled.safetensors", path.read_bytes())
        report = {"schema_version": "confirmation-execution-v4", "experiment_id": run_config["experiment_id"],
                  "model": run_config["model"], "model_id": run_config["model_id"], "revision": run_config["revision"],
                  "hidden_size": preflight["hidden_size"],
                  "config": run_config, "preflight": preflight_receipt,
                  "environment_bf16": environments["bf16"], "environment_fp32": environments.get("fp32"),
                  "tokenizer": _tokenizer_metadata(tokenizer), "cases": completed, "pooled": pooled_receipt,
                  "objects": list(receipts), "function_seconds": time.monotonic() - started,
                  "complete": len(completed) == len(expected), "stop_reason": stop_reason, "phases": phases,
                  "operational_execution_policy": run_config.get("operational_execution_policy"),
                  "upload_stage_timings": upload_stage_timings, "maximum_pending_uploads": maximum_pending_uploads,
                  "fp32_status": "complete" if sum(condition_map[c["condition"]]["precision"] == "fp32" for c in completed) == len(expected) // 2 else "partial",
                  "fp32_semantics": BF16_CHECKPOINT_PROMOTION,
                  "precision_settings": {"fp32_checkpoint_source": "bf16_promoted", "tf32": False, "autocast": False},
                  "tf32_matmul_allowed": False, "tf32_cudnn_allowed": False, "autocast_enabled": False}
        if durable_finalization:
            report["runtime_checkpoints"] = runtime_checkpoints
            report["capture_completion_checkpoint"] = capture_completion_checkpoint
        report_receipt = put_bytes("report.json", canonical_json(report))
        return {"report": report, "report_receipt": report_receipt}

    return app, prepare, run_confirmation
