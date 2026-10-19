"""Single-forward, token-level causal activation extraction without generation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import re
import time
from typing import Any, Mapping, Sequence

from .adapters import MODEL_CLASSES, OPTIONAL_CAPTURES, get_adapter
from .storage import RunStore, canonical_json, verify_run


def require_revision(revision: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision or ""):
        raise ValueError("An immutable 40-character Hugging Face commit SHA is required")
    return revision.lower()


def load_model(
    model_id: str,
    revision: str,
    *,
    dtype: str = "bfloat16",
    device: str = "cuda",
    attention_implementation: str = "eager",
):
    """Load a pinned native base decoder and fast tokenizer; no LM output head.

    HF_TOKEN is consumed by Hugging Face's standard authentication. It is never
    accepted into or serialized with experiment metadata. No remote code runs.
    """
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    revision = require_revision(revision)
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("Choose a captured floating precision, not quantization")
    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    if config.model_type in {"gemma4", "muse_glimmer"}:
        from .modern_adapters import load_modern_model
        return load_modern_model(model_id, revision, dtype=dtype, device=device,
                                 attention_implementation=attention_implementation)
    if config.model_type not in MODEL_CLASSES:
        raise ValueError(f"No audited adapter for {config.model_type}; model weights have not been loaded")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer offsets are required")
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        config=config,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=getattr(torch, dtype),
        attn_implementation=attention_implementation,
    ).to(device).eval()
    get_adapter(model)
    model._sonnet_geometry_provenance = {
        "model_id": model_id, "revision": revision, "weights_source": "huggingface_pretrained_safetensors",
        "trust_remote_code": False, "quantized": False,
    }
    return model, tokenizer


def _tokenizer_metadata(tokenizer: Any) -> dict[str, Any]:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for permanent character offsets")
    backend_json = tokenizer.backend_tokenizer.to_str()
    # Do not serialize init_kwargs: some Transformers versions retain auth kwargs.
    settings = {
        "backend_sha256": hashlib.sha256(backend_json.encode()).hexdigest(),
        "special_tokens_map": {k: str(v) for k, v in tokenizer.special_tokens_map.items()},
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "model_max_length": tokenizer.model_max_length,
        "add_bos_token": getattr(tokenizer, "add_bos_token", None),
        "add_eos_token": getattr(tokenizer, "add_eos_token", None),
    }
    return {
        "class": type(tokenizer).__name__,
        "name_or_path": tokenizer.name_or_path,
        "is_fast": True,
        "vocab_size": len(tokenizer),
        "fingerprint_sha256": hashlib.sha256(canonical_json(settings)).hexdigest(),
        "settings": settings,
    }


def _environment(torch: Any, model: Any) -> dict[str, Any]:
    versions = {}
    for package in ("torch", "transformers", "tokenizers", "safetensors", "huggingface-hub", "numpy", "google-cloud-storage"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    device = next(model.parameters()).device
    result = {
        "python": platform.python_version(), "platform": platform.platform(), "packages": versions,
        "device": str(device), "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_initial_seed": torch.initial_seed(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result["gpu"] = {"name": props.name, "total_memory_bytes": props.total_memory, "capability": [props.major, props.minor]}
    return result


def extract_sonnet(
    model: Any,
    tokenizer: Any,
    record: Mapping[str, Any],
    output_dir: str | Path,
    *,
    model_id: str,
    revision: str,
    run_id: str | None = None,
    capture: Sequence[str] = (),
    capture_layers: Sequence[int] | None = (),
    provenance: Mapping[str, Any] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Capture every residual token state and selected audited intermediates.

    Returns the immutable manifest plus a convenience ``run_dir`` key (the latter
    is not persisted). Files are stored at ``output_dir/run_id``. Resume requires
    an explicit run_id and a complete validated checkpoint with matching request.
    Models should come from load_model. Random CPU test models must declare
    provenance={"weights_source": "random_initialization"} explicitly.
    """
    import torch
    from .corpus import tokenize_with_alignment

    started = time.perf_counter()
    revision = require_revision(revision)
    adapter = get_adapter(model)
    capture = tuple(sorted(set(capture)))
    selected_layers = tuple(sorted(set(capture_layers or ())))
    if set(capture) - set(OPTIONAL_CAPTURES):
        raise ValueError(f"Unsupported captures: {sorted(set(capture) - set(OPTIONAL_CAPTURES))}")
    if capture and not selected_layers:
        raise ValueError("Explicit capture_layers are required for optional tensors")
    if any(not isinstance(layer, int) or not 0 <= layer < len(adapter.layers) for layer in selected_layers):
        raise ValueError("capture_layers contains an invalid layer index")
    if "attention_weights" in capture and model.config._attn_implementation != "eager":
        raise ValueError("Attention probabilities require explicit eager attention")
    for layer in selected_layers:
        for name in capture:
            adapter.module_for(layer, name)
    provenance = dict(provenance or {})
    loaded = getattr(model, "_sonnet_geometry_provenance", None)
    if loaded:
        if loaded["model_id"] != model_id or loaded["revision"] != revision:
            raise ValueError("Requested identity differs from the model loaded by load_model")
        weights_source = loaded["weights_source"]
    elif provenance.get("weights_source") == "random_initialization":
        weights_source = "random_initialization"
    else:
        raise ValueError("Use load_model for provenance-checked weights, or explicitly label random_initialization")
    if getattr(model, "is_quantized", False):
        raise ValueError("Quantized model loading is excluded from primary extraction")
    text = record.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError("A nonempty sonnet text is required")
    sonnet_id = str(record.get("sonnet_id") or record.get("id") or f"sonnet-{int(record['number']):03d}")
    encoding = tokenize_with_alignment(tokenizer, record)
    token_ids = encoding["input_ids"]
    if not token_ids:
        raise ValueError("Tokenization produced no tokens")
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is not None and len(token_ids) > max_positions:
        raise ValueError(f"Token count {len(token_ids)} exceeds model context {max_positions}; truncation is forbidden")
    tokenizer_metadata = _tokenizer_metadata(tokenizer)
    config_dict = model.config.to_dict()
    config_hash = hashlib.sha256(canonical_json(config_dict)).hexdigest()
    source_file = inspect.getsourcefile(type(model))
    source_hash = hashlib.sha256(Path(source_file).read_bytes()).hexdigest() if source_file else None
    request = {
        "model_id": model_id, "revision": revision, "weights_source": weights_source,
        "sonnet_id": sonnet_id, "record_sha256": hashlib.sha256(canonical_json(dict(record))).hexdigest(),
        "config_sha256": config_hash, "tokenizer_fingerprint": tokenizer_metadata["fingerprint_sha256"],
        "capture": list(capture), "capture_layers": list(selected_layers),
        "attention_implementation": model.config._attn_implementation,
        "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "implementation_source_sha256": source_hash,
    }
    parent_config = getattr(model, "_sonnet_geometry_parent_config", None)
    loading_info = getattr(model, "_sonnet_geometry_loading_info", None)
    if parent_config is not None:
        request["parent_config_sha256"] = hashlib.sha256(canonical_json(parent_config)).hexdigest()
    if weights_source == "random_initialization":
        weights_hash = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            weights_hash.update(name.encode())
            weights_hash.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        request["random_weights_sha256"] = weights_hash.hexdigest()
    if resume:
        if not run_id:
            raise ValueError("Resume requires an explicit run_id")
        existing_dir = Path(output_dir) / run_id
        if existing_dir.exists():
            previous = verify_run(existing_dir)
            if previous["request"] != request:
                raise ValueError("Existing run request differs; use a new run_id")
            return {**previous, "run_dir": str(existing_dir.resolve())}
    store = RunStore(output_dir, run_id)
    tensors: dict[str, Any] = {}
    semantics: dict[str, dict[str, Any]] = {}
    handles = []
    token_count = len(token_ids)

    def save(name: str, value: Any, meaning: str, *, attention: bool = False) -> None:
        if name in tensors:
            raise RuntimeError(f"Hook fired twice: {name}; unsupported execution path")
        expected_ndim = 4 if attention else 3
        if not isinstance(value, torch.Tensor) or value.ndim != expected_ndim or value.shape[0] != 1:
            raise RuntimeError(f"Unexpected tensor shape for {name}: {getattr(value, 'shape', None)}")
        if attention:
            if tuple(value.shape[-2:]) != (token_count, token_count):
                raise RuntimeError(f"Attention axes do not match token sequence: {name}")
        elif value.shape[1] != token_count:
            raise RuntimeError(f"Token axis mismatch: {name}")
        captured = value[0].detach().to(device="cpu", copy=True).contiguous()
        if not torch.isfinite(captured).all():
            raise RuntimeError(f"Nonfinite activations: {name}")
        tensors[name] = captured
        semantics[name] = {"shape": list(captured.shape), "dtype": str(captured.dtype), "meaning": meaning,
                           "axes": ["query_head", "query_token", "key_token"] if attention else ["token", "feature"]}

    def input_hook(name: str, meaning: str):
        def hook(_module, args, kwargs):
            value = args[0] if args else kwargs.get("hidden_states", kwargs.get("input"))
            save(name, value, meaning)
        return hook

    def output_hook(name: str, meaning: str, *, attention: bool = False):
        def hook(_module, _args, value):
            if attention:
                if not isinstance(value, tuple) or len(value) < 2 or value[1] is None:
                    raise RuntimeError(f"Attention module did not expose probabilities: {name}")
                value = value[1]
            elif isinstance(value, tuple):
                value = value[0]
            save(name, value, meaning, attention=attention)
        return hook

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        handles.append(adapter.layers[0].register_forward_pre_hook(input_hook("residual.embedding", adapter.describe()["embedding_semantics"]), with_kwargs=True))
        for index, layer in enumerate(adapter.layers):
            handles.append(layer.register_forward_hook(output_hook(f"residual.block.{index:03d}", adapter.describe()["block_semantics"])))
        handles.append(adapter.norm.register_forward_hook(output_hook("final_normalized", adapter.describe()["final_normalized_semantics"])))
        for index in selected_layers:
            for capture_name in capture:
                module, mode, meaning = adapter.module_for(index, capture_name)
                name = f"intermediate.block.{index:03d}.{capture_name}"
                if mode == "input":
                    handles.append(module.register_forward_pre_hook(input_hook(name, meaning), with_kwargs=True))
                else:
                    handles.append(module.register_forward_hook(output_hook(name, meaning, attention=mode == "weights")))
        input_ids = torch.tensor([token_ids], device=device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(token_count, device=device, dtype=torch.long).unsqueeze(0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        forward_start = time.perf_counter()
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                            use_cache=False, output_hidden_states=False,
                            output_attentions="attention_weights" in capture, return_dict=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - forward_start
        if not torch.equal(outputs.last_hidden_state[0].cpu(), tensors["final_normalized"]):
            raise RuntimeError("Final norm hook disagrees with model last_hidden_state")
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    expected_names = {"residual.embedding", "final_normalized"} | {f"residual.block.{i:03d}" for i in range(len(adapter.layers))}
    expected_names |= {f"intermediate.block.{i:03d}.{name}" for i in selected_layers for name in capture}
    if set(tensors) != expected_names:
        raise RuntimeError(f"Incomplete hooks: {sorted(expected_names - set(tensors))}")
    store.write_json("record.json", dict(record))
    store.write_json("tokens.json", encoding)
    store.write_json("model-config.json", config_dict)
    if parent_config is not None:
        store.write_json("parent-model-config.json", parent_config)
    if loading_info is not None:
        store.write_json("loading-info.json", loading_info)
    store.write_json("tokenizer.json", tokenizer_metadata)
    store.write_bytes("tokenizer-backend.json", tokenizer.backend_tokenizer.to_str().encode())
    store.write_tensors(tensors, {"schema_version": "1", "model_id": model_id, "revision": revision, "sonnet_id": sonnet_id})
    metrics = {
        "forward_seconds_including_hook_transfers": forward_seconds,
        "elapsed_seconds_before_manifest": time.perf_counter() - started,
        "token_count": token_count,
        "tensor_bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
    }
    if device.type == "cuda":
        metrics["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        metrics["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    manifest = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sonnet_id": sonnet_id, "model_id": model_id, "revision": revision,
        "request": request, "adapter": adapter.describe(), "tokenizer": tokenizer_metadata,
        "tensors": semantics, "environment": _environment(torch, model), "metrics": metrics,
        "provenance": provenance,
        "model_loading": loaded or {"weights_source": weights_source},
        "execution": {"batch_size": 1, "eval_mode": True, "inference_mode": True, "use_cache": False,
                      "generation": False, "logits": False, "truncation": False, "add_special_tokens": True,
                      "pooling": "none; raw token states retained", "padding": "none",
                      "prefix_policy": "Use tokens.json prefix_checks; only exact token prefixes have mathematical causal equivalence. Floating arithmetic can differ between sequence lengths."},
    }
    complete = store.complete(manifest)
    return {**complete, "run_dir": str(store.path.resolve())}

# Public spelling used by the infrastructure runner.
validate_revision = require_revision


def validate_causal_prefixes(
    model: Any,
    record: Mapping[str, Any],
    run_dir: str | Path,
    *,
    line_numbers: Sequence[int] = (1, 4, 8, 12),
    fp32_diagnostic: bool = False,
) -> dict[str, Any]:
    """Separate causal invariance from variable-length floating-point sensitivity.

    The pass criterion is bitwise invariance to changed future token IDs at the
    same full sequence length, mask and positions. Independently recomputed
    shorter prefixes retain their original dtype-specific numerical thresholds
    and a separate equivalence result. Neither result is silently substituted
    for the other. Primary tensors and older validation reports are immutable.
    """
    import torch
    from safetensors.torch import load_file

    run_dir = Path(run_dir)
    manifest = verify_run(run_dir)
    if hashlib.sha256(canonical_json(dict(record))).hexdigest() != manifest["request"]["record_sha256"]:
        raise ValueError("Prefix validation record differs from the captured run")
    adapter = get_adapter(model)
    dtype = next(model.parameters()).dtype
    atol, rtol = {
        torch.float32: (1e-5, 1e-4),
        torch.float16: (5e-3, 5e-3),
        torch.bfloat16: (5e-2, 2e-2),
    }[dtype]
    encoded = json.loads((run_dir / "tokens.json").read_text())
    full = load_file(run_dir / "activations.safetensors")
    device = next(model.parameters()).device
    results = []
    fixed_results = []
    full_token_ids = encoded["input_ids"]
    vocab_size = model.get_input_embeddings().num_embeddings
    if vocab_size < 2:
        raise ValueError("Future-token intervention requires at least two valid token IDs")

    def forward_residuals(ids):
        captured_last = []

        def save_last(_module, _args, output):
            value = output[0] if isinstance(output, tuple) else output
            captured_last.append(value[0].detach().to("cpu", copy=True))

        handle = adapter.layers[-1].register_forward_hook(save_last)
        try:
            inputs = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.inference_mode():
                output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                               position_ids=torch.arange(len(ids), device=device).unsqueeze(0),
                               use_cache=False, output_hidden_states=True, output_attentions=False,
                               return_dict=True)
            if len(output.hidden_states) != len(adapter.layers) + 1 or len(captured_last) != 1:
                raise RuntimeError("Unexpected hidden-state API during prefix validation")
            candidates = {"residual.embedding": output.hidden_states[0][0]}
            candidates.update({f"residual.block.{i:03d}": output.hidden_states[i + 1][0]
                               for i in range(len(adapter.layers) - 1)})
            candidates[f"residual.block.{len(adapter.layers) - 1:03d}"] = captured_last[0]
            candidates["final_normalized"] = output.last_hidden_state[0]
            return {name: value.detach().to("cpu", copy=True) for name, value in candidates.items()}
        finally:
            handle.remove()

    def measure(candidates, token_count, *, baseline=None, absolute_tolerance=None, relative_tolerance=None):
        baseline = full if baseline is None else baseline
        absolute_tolerance = atol if absolute_tolerance is None else absolute_tolerance
        relative_tolerance = rtol if relative_tolerance is None else relative_tolerance
        measurements = {}
        for name, value in candidates.items():
            actual_native = value[:token_count]
            reference_native = baseline[name][:token_count]
            actual, reference = actual_native.float(), reference_native.float()
            difference = actual - reference
            relative_l2 = (torch.linalg.vector_norm(difference) /
                           torch.linalg.vector_norm(reference).clamp_min(1e-12)).item()
            measurements[name] = {
                "max_absolute_difference": difference.abs().max().item(),
                "mean_absolute_difference": difference.abs().mean().item(),
                "relative_l2_difference": relative_l2,
                "bitwise_equal": torch.equal(actual_native, reference_native),
                "allclose": torch.allclose(actual, reference, atol=absolute_tolerance, rtol=relative_tolerance),
                "relative_l2_within_tolerance": relative_l2 <= relative_tolerance,
            }
        return measurements

    was_training = model.training
    model.eval()
    started = time.perf_counter()
    fp32_report = None
    try:
        for line_number in line_numbers:
            check = next((p for p in encoded["prefix_checks"] if p["line_number"] == line_number), None)
            if check is None or not check["exact_token_prefix"]:
                results.append({"line_number": line_number, "eligible": False,
                                "reason": "missing line" if check is None else "prefix retokenization or boundary-crossing token"})
                fixed_results.append(dict(results[-1]))
                continue
            token_count = check["full_pass_token_end"]
            measurements = measure(forward_residuals(check["prefix_input_ids"]), token_count)
            passed = all(m["allclose"] and m["relative_l2_within_tolerance"] for m in measurements.values())
            results.append({"line_number": line_number, "eligible": True, "token_count": token_count,
                            "passed": passed, "layers": measurements})
            if token_count == len(full_token_ids):
                fixed_results.append({"line_number": line_number, "eligible": False,
                                      "reason": "complete sequence has no future suffix to intervene on"})
                continue
            intervention_ids = full_token_ids[:token_count] + [(token + 1) % vocab_size for token in full_token_ids[token_count:]]
            fixed_measurements = measure(forward_residuals(intervention_ids), token_count)
            fixed_results.append({
                "line_number": line_number, "eligible": True, "token_count": token_count,
                "full_sequence_length": len(full_token_ids),
                "changed_future_tokens": len(full_token_ids) - token_count,
                "intervention_input_ids_sha256": hashlib.sha256(canonical_json(intervention_ids)).hexdigest(),
                "passed": all(m["bitwise_equal"] for m in fixed_measurements.values()),
                "layers": fixed_measurements,
            })
        if fp32_diagnostic:
            diagnostic_started = time.perf_counter()
            original_parameter_dtypes = {name: parameter.dtype for name, parameter in model.named_parameters()}
            original_buffers = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
            previous_tf32 = torch.backends.cuda.matmul.allow_tf32
            fp32_results = []
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                model.float()
                # A fresh FP32 baseline is essential: comparing a promoted
                # prefix against the stored BF16 full pass would confound dtype.
                fp32_full = forward_residuals(full_token_ids)
                for check in encoded["prefix_checks"]:
                    if check["line_number"] not in line_numbers or not check["exact_token_prefix"]:
                        continue
                    token_count = check["full_pass_token_end"]
                    measured = measure(forward_residuals(check["prefix_input_ids"]), token_count,
                                       baseline=fp32_full, absolute_tolerance=1e-5, relative_tolerance=1e-4)
                    fp32_results.append({
                        "line_number": check["line_number"], "token_count": token_count, "layers": measured,
                        "passed": all(m["allclose"] and m["relative_l2_within_tolerance"] for m in measured.values()),
                    })
                fp32_report = {
                    "enabled": True, "weights": f"Same loaded {dtype} parameter values promoted to FP32; no checkpoint reload",
                    "baseline": "fresh full-sequence FP32 forward", "atol": 1e-5, "rtol": 1e-4,
                    "tf32_matmul_allowed": False, "results": fp32_results,
                    "passed": bool(fp32_results) and all(r["passed"] for r in fp32_results),
                    "elapsed_seconds": time.perf_counter() - diagnostic_started,
                    "raw_fp32_tensors_persisted": False,
                    "interpretation": "A separate numerical sensitivity diagnostic, not replacement primary activations or proof of a unique source of BF16 discrepancies",
                }
            finally:
                # Restore each parameter and exact original buffer separately.
                # A blanket .to(original_dtype) would corrupt originally FP32
                # RoPE buffers by downcasting them along with BF16 parameters.
                for name, parameter in model.named_parameters():
                    parameter.data = parameter.data.to(dtype=original_parameter_dtypes[name])
                for name, buffer in original_buffers.items():
                    parent_name, _, leaf = name.rpartition(".")
                    module = model.get_submodule(parent_name) if parent_name else model
                    module._buffers[leaf] = buffer
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    finally:
        model.train(was_training)
    eligible = [r for r in results if r["eligible"]]
    fixed_eligible = [r for r in fixed_results if r["eligible"]]
    fixed_passed = bool(fixed_eligible) and all(r["passed"] for r in fixed_eligible)
    variable_passed = bool(eligible) and all(r["passed"] for r in eligible)
    return {"schema_version": 2, "dtype": str(dtype), "atol": atol, "rtol": rtol,
            "relative_l2_tolerance": rtol, "elapsed_seconds": time.perf_counter() - started,
            "requested_lines": list(line_numbers), "eligible_count": len(eligible),
            "passed": fixed_passed, "pass_criterion": "bitwise_future_token_invariance_at_fixed_full_sequence_shape",
            "variable_length_equivalent": variable_passed, "results": results,
            "fp32_diagnostic": fp32_report or {"enabled": False},
            "fixed_shape_causality": {"passed": fixed_passed, "eligible_count": len(fixed_eligible),
                                      "comparison": "bitwise equality; no tolerance relaxation",
                                      "intervention": "Replace each future ID with (ID+1) modulo vocabulary size; keep all-one mask, positions and full sequence length",
                                      "results": fixed_results},
            "note": "Pass validates causal future-token invariance at fixed shape. Variable-length recomputation is a separate sensitivity result with unchanged tolerances; a failed variable_length_equivalent result means shorter BF16 forwards are not numerically interchangeable with saved full-pass states. Ineligible boundaries are never rounded."}
