"""Bounded H200 extraction. Secrets enter only via Modal Secret, never images."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("shore-shakespeare-activation-geometry")
runtime_dir = ROOT / "infra/modern_runtime" if os.environ.get("SHORE_MODEL_RUNTIME") == "modern" else ROOT
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(runtime_dir), frozen=True, uv_version="0.9.8")
    .env({"PYTHONPATH": "/research/src", "HF_HOME": "/model-cache", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(ROOT / "src", "/research/src")
)
cache = modal.Volume.from_name("shore-shakespeare-model-cache", create_if_missing=True)
# The local launcher sets this path; only the credential contents are sent in an
# encrypted ephemeral Modal Secret. No credential is copied into the image.
gcp_secret = modal.Secret.from_dict(
    {"GCP_SERVICE_ACCOUNT_JSON": Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).read_text()}
    if modal.is_local() and os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") else {}
)
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN")})


@app.function(image=image, volumes={"/model-cache": cache}, secrets=[hf_secret],
              cpu=2, memory=4096, timeout=1200, retries=0, max_containers=1, scaledown_window=2)
def prepare_model(model_id: str, revision: str) -> dict:
    """Download model on CPU before allocating a GPU; immutable revision only."""
    from huggingface_hub import snapshot_download
    from sonnet_geometry.extraction import validate_revision

    validate_revision(revision)
    started = time.monotonic()
    snapshot = snapshot_download(model_id, revision=revision,
                                 allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
    cache.commit()
    return {"model_id": model_id, "revision": revision, "download_seconds": time.monotonic() - started,
            "snapshot": snapshot, "gpu_seconds": 0}


@app.function(image=image, gpu="H200", volumes={"/model-cache": cache}, secrets=[gcp_secret, hf_secret],
              cpu=4, memory=32768, timeout=1800, startup_timeout=600, retries=0,
              min_containers=0, max_containers=1, scaledown_window=2)
def extract_batch(config: dict, records: list[dict]) -> dict:
    import hashlib
    import tempfile

    import torch
    from google.cloud import storage
    from google.oauth2 import service_account
    from sonnet_geometry.extraction import extract_sonnet, load_model, validate_causal_prefixes
    from sonnet_geometry.storage import canonical_json, upload_run_gcs

    started = time.monotonic()
    credentials = service_account.Credentials.from_service_account_info(
        json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"]))
    client = storage.Client(project=credentials.project_id, credentials=credentials)
    bucket = client.bucket(config["bucket"])
    # Verify bucket protections before capturing data.
    bucket.reload()
    if bucket.iam_configuration.public_access_prevention != "enforced":
        raise RuntimeError("Refusing to write research data to a bucket without public access prevention")
    if not bucket.iam_configuration.uniform_bucket_level_access_enabled:
        raise RuntimeError("Refusing to write research data to a bucket with per-object ACLs")
    prefix = f"models/{config['model_id'].replace('/', '--')}/{config['revision']}/experiments/{config['experiment_id']}"
    experiment_blob = bucket.blob(f"{prefix}/experiment.json")
    experiment_blob.upload_from_string(canonical_json(config), content_type="application/json", if_generation_match=0)
    torch.manual_seed(config.get("seed", 0))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, tokenizer = load_model(config["model_id"], config["revision"], dtype="bfloat16", device="cuda",
                                  attention_implementation="eager")
    load_seconds = time.monotonic() - started
    results = []
    pooled_reports = []
    batch_peak_allocated_bytes = torch.cuda.max_memory_allocated()
    for record in records:
        if time.monotonic() - started > config.get("soft_timeout_seconds", 1500):
            break
        sonnet_started = time.monotonic()
        with tempfile.TemporaryDirectory() as temporary:
            manifest = extract_sonnet(
                model, tokenizer, record, temporary, model_id=config["model_id"], revision=config["revision"],
                run_id=f"sonnet-{record['number']:03d}", capture=config.get("capture", []),
                capture_layers=config.get("capture_layers"), provenance=config,
            )
            run_dir = Path(temporary) / f"sonnet-{record['number']:03d}"
            receipt = upload_run_gcs(run_dir, config["bucket"], f"{prefix}/sonnets/{record['number']:03d}", client=client)
            # Analyze on the worker so large primary tensors never need local download.
            from sonnet_geometry.reporting import summarize_run
            summary, pooled = summarize_run(run_dir)
            result = {"sonnet_number": record["number"], "manifest": manifest,
                      "storage_receipt": receipt, "summary": summary,
                      "elapsed_seconds_including_upload": time.monotonic() - sonnet_started}
            if config.get("stage") == "validation" or (config.get("validate_first") and not results):
                validation = validate_causal_prefixes(model, record, run_dir,
                                                      fp32_diagnostic=config.get("fp32_prefix_diagnostic", False))
                result["prefix_validation"] = validation
                validation_blob = bucket.blob(f"{prefix}/derived/sonnet-{record['number']:03d}-prefix-validation.json")
                validation_blob.upload_from_string(canonical_json(validation), content_type="application/json",
                                                   if_generation_match=0)
                if config.get("validate_first") and not validation["passed"]:
                    raise RuntimeError("Fixed-shape causality failed; raw first-sonnet data retained, batch stopped")
            batch_peak_allocated_bytes = max(batch_peak_allocated_bytes, torch.cuda.max_memory_allocated())
            blob = bucket.blob(f"{prefix}/derived/sonnet-{record['number']:03d}.json")
            blob.upload_from_string(canonical_json(summary), content_type="application/json", if_generation_match=0)
            # Derived vectors remain in private storage; never return hundreds of
            # megabytes of JSON through the Modal RPC response.
            blob = bucket.blob(f"{prefix}/derived/pooled-sonnet-{record['number']:03d}.json")
            blob.upload_from_string(canonical_json(pooled), content_type="application/json", if_generation_match=0)
            pooled_reports.append(pooled)
            result["elapsed_seconds_including_upload"] = time.monotonic() - sonnet_started
            results.append(result)
            print(json.dumps({"sonnet": record["number"], "seconds": result["elapsed_seconds_including_upload"],
                              "bytes": sum(o["bytes"] for o in receipt["objects"])}), flush=True)
    torch.cuda.synchronize()
    batch_analysis = None
    if len(pooled_reports) > 1:
        from sonnet_geometry.reporting import summarize_batch
        batch_analysis = summarize_batch(pooled_reports, records=records[:len(results)])
        blob = bucket.blob(f"{prefix}/derived/batch-analysis.json")
        blob.upload_from_string(canonical_json(batch_analysis), content_type="application/json", if_generation_match=0)
    elapsed = time.monotonic() - started
    report = {"experiment_id": config["experiment_id"], "model_id": config["model_id"],
              "revision": config["revision"], "prefix": prefix, "bucket": config["bucket"],
              "completed": len(results), "requested": len(records), "load_seconds": load_seconds,
              "function_seconds": elapsed, "gpu": torch.cuda.get_device_name(),
              "peak_allocated_bytes": batch_peak_allocated_bytes,
              "validation_passed": all(r.get("prefix_validation", {}).get("passed", False) for r in results)
                  if config.get("stage") == "validation" and results else None,
              "estimated_resource_cost_usd": elapsed * config["resource_rate_usd_per_second"],
              "cost_note": "Timed in-function estimate including extraction, validation, analysis and their uploads; excludes startup, final-report upload tail, image builds, transfer charges, storage and credits. Not invoice cost.",
              "batch_analysis_uri": f"gs://{config['bucket']}/{prefix}/derived/batch-analysis.json" if batch_analysis else None,
              "results": results}
    report_bytes = canonical_json(report)
    blob = bucket.blob(f"{prefix}/batch-report.json")
    blob.metadata = {"sha256": hashlib.sha256(report_bytes).hexdigest()}
    blob.upload_from_string(report_bytes, content_type="application/json", if_generation_match=0)
    return report
