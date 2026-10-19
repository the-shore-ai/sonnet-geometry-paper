"""Short, capped follow-up inference. Never deployed as a service."""
import json
import os
from pathlib import Path
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
runtime = ROOT if os.environ.get("SHORE_MODEL_RUNTIME", "standard") == "standard" else ROOT / "infra/modern_runtime"
app = modal.App("shore-shakespeare-controls-v3")
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_sync(str(runtime), frozen=True, uv_version="0.9.8")
         .env({"PYTHONPATH": "/research/src", "HF_HOME": "/model-cache", "TOKENIZERS_PARALLELISM": "false",
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "4"})
         .add_local_dir(ROOT / "src", "/research/src"))
cache = modal.Volume.from_name("shore-shakespeare-controls-cache", create_if_missing=True)
gcp = modal.Secret.from_dict({"GCP_SERVICE_ACCOUNT_JSON": Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).read_text()}
                            if modal.is_local() and os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") else {})
hf = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN")})


@app.function(image=image, volumes={"/model-cache": cache}, secrets=[hf], cpu=(2, 4),
              memory=(4096, 98304), timeout=600, startup_timeout=120, retries=0,
              min_containers=0, max_containers=1, scaledown_window=2)
def prepare(model_id, revision):
    from huggingface_hub import snapshot_download
    from sonnet_geometry.extraction import require_revision
    started = time.monotonic()
    snapshot_download(model_id, revision=require_revision(revision),
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
    cache.commit()
    return {"seconds": time.monotonic() - started, "gpu_seconds": 0}


@app.function(image=image, volumes={"/model-cache": cache}, secrets=[gcp, hf], gpu="H200",
              cpu=(4, 4), memory=(32768, 98304), timeout=300, startup_timeout=120, retries=0,
              min_containers=0, max_containers=1, scaledown_window=2)
def run_controls(config, records):
    import hashlib
    import tempfile
    import torch
    from google.cloud import storage
    from google.oauth2 import service_account
    from safetensors.torch import save_file
    from sonnet_geometry.budget import CONTROL_ENVELOPE
    from sonnet_geometry.extraction import load_model, _environment, _tokenizer_metadata
    from sonnet_geometry.adapters import get_adapter
    from sonnet_geometry.storage import canonical_json
    from sonnet_geometry.text_controls import (case_inputs, capture_selected, pool_selected,
        EARLY_BLOCK, DEPTH75_BLOCK, LAST_BLOCK, PAD_LENGTH, SONNETS)

    if config["resource_envelope"] != CONTROL_ENVELOPE or [r["number"] for r in records] != list(SONNETS):
        raise ValueError("Run does not match the frozen bounded protocol")
    started = time.monotonic()
    creds = service_account.Credentials.from_service_account_info(json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"]))
    bucket = storage.Client(project=creds.project_id, credentials=creds).bucket(config["bucket"])
    bucket.reload()
    if bucket.iam_configuration.public_access_prevention != "enforced" or not bucket.iam_configuration.uniform_bucket_level_access_enabled:
        raise ValueError("Private storage required")
    prefix = f"controls-v3/{config['model']}/{config['experiment_id']}"
    receipts = []
    def put_bytes(name, data):
        digest = hashlib.sha256(data).hexdigest()
        blob = bucket.blob(f"{prefix}/{name}"); blob.metadata = {"sha256": digest}
        blob.upload_from_string(data, if_generation_match=0, checksum="auto")
        receipt = {"name": blob.name, "uri": f"gs://{bucket.name}/{blob.name}", "generation": str(blob.generation),
                   "bytes": len(data), "sha256": digest, "full_readback_verified": False}
        receipts.append(receipt); return receipt
    put_bytes("config.json", canonical_json(config))
    torch.manual_seed(0); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model, tokenizer = load_model(config["model_id"], config["revision"], dtype="bfloat16", device="cuda")
    env_bf16 = _environment(torch, model)
    put_bytes("model-config.json", canonical_json(model.config.to_dict()))
    put_bytes("adapter.json", canonical_json(get_adapter(model).describe()))
    put_bytes("loading-info.json", canonical_json(getattr(model, "_sonnet_geometry_loading_info", {})))
    put_bytes("tokenizer-backend.json", tokenizer.backend_tokenizer.to_str().encode())
    blocks = {"early": EARLY_BLOCK[config["model"]], "depth75": DEPTH75_BLOCK[config["model"]], "final_raw": LAST_BLOCK[config["model"]]}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    cases, pooled, original = [], {}, {}
    stopped = None
    def execute(record, name, encoding, shape):
        nonlocal pooled
        before = time.monotonic()
        expected_name = "original_padded_bf16" if name == "original_padded_fp32" else name
        expected, = [x for x in config["preflight"]["cases"] if x["sonnet"] == record["number"] and x["condition"] == expected_name]
        if hashlib.sha256(canonical_json(encoding["input_ids"])).hexdigest() != expected["input_ids_sha256"]:
            raise ValueError("Runtime input differs from frozen preflight")
        key = f"sonnet-{record['number']:03d}/{name}"
        captures = capture_selected(model, encoding, blocks=blocks, pad_to=shape, pad_id=pad_id)
        for metric, vector in pool_selected(captures, encoding).items():
            pooled[f"{record['number']:03d}.{name}.{metric}"] = torch.from_numpy(vector.copy())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "states.safetensors"
            save_file(captures, str(path), metadata={"selected_states_only": "true", "padding_tokens_stored": "false"})
            state_receipt = put_bytes(f"{key}/states.safetensors", path.read_bytes())
        metadata = {"sonnet": record["number"], "condition": name, "tokenization": encoding,
                    "forward_length": shape or len(encoding["input_ids"]), "pad_token_id": pad_id,
                    "attention_mask": [1] * len(encoding["input_ids"]) + [0] * ((shape or len(encoding["input_ids"]))-len(encoding["input_ids"])),
                    "position_ids": list(range(shape or len(encoding["input_ids"]))),
                    "blocks": blocks, "dtypes": {k: str(v.dtype) for k,v in captures.items()},
                    "elapsed_seconds": time.monotonic() - before, "primary_states": state_receipt}
        put_bytes(f"{key}/metadata.json", canonical_json(metadata))
        cases.append(metadata)
        print(json.dumps({"sonnet": record["number"], "condition": name, "seconds": metadata["elapsed_seconds"]}), flush=True)
    # First two poems form a measured pilot; the rest proceed only within the same fixed timeout.
    for record in records:
        for name, encoding, shape in case_inputs(record, tokenizer):
            if time.monotonic() - started > 190:
                stopped = "BF16 soft stop preserves time for serialization; no automatic retry"; break
            execute(record, name, encoding, shape)
            if name == "original_padded_bf16": original[record["number"]] = encoding
        if stopped: break
    fp32_status, env_fp32 = "not_started_due_to_time", None
    if not stopped and time.monotonic() - started < 190:
        # Values of checkpoint BF16 weights are preserved exactly; arithmetic is promoted.
        model.float(); torch.cuda.empty_cache()
        env_fp32 = _environment(torch, model)
        fp32_status = "complete"
        for record in records:
            if time.monotonic() - started > 240:
                stopped = "FP32 soft stop; incomplete precision diagnostics retained"; fp32_status = "partial"; break
            execute(record, "original_padded_fp32", original[record["number"]], PAD_LENGTH)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "pooled.safetensors"; save_file(pooled, str(path))
        pooled_receipt = put_bytes("pooled.safetensors", path.read_bytes())
    report = {"schema_version": "text-controls-execution-v3", "experiment_id": config["experiment_id"],
              "model": config["model"], "model_id": config["model_id"], "revision": config["revision"],
              "config": config, "environment_bf16": env_bf16, "environment_fp32": env_fp32,
              "tokenizer": _tokenizer_metadata(tokenizer), "cases": cases, "pooled": pooled_receipt,
              "objects": list(receipts), "function_seconds": time.monotonic() - started,
              "fp32_status": fp32_status, "stop_reason": stopped,
              "complete": len(cases) == len(records)*6, "fp32_semantics": "BF16 checkpoint values promoted to FP32 arithmetic; no higher-precision source weights inferred"}
    report_receipt = put_bytes("report.json", canonical_json(report))
    return {"report": report, "report_receipt": report_receipt}
