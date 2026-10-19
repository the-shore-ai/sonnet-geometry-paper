"""Local validation and cost estimation; GPU execution is explicit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    corpus = commands.add_parser("validate-corpus")
    corpus.add_argument("--path", type=Path, default=Path("datasets/normalized/sonnets.jsonl"))
    verify = commands.add_parser("verify-run")
    verify.add_argument("path", type=Path)
    estimate = commands.add_parser("estimate")
    estimate.add_argument("--corpus", type=Path, default=Path("datasets/normalized/sonnets.jsonl"))
    estimate.add_argument("--lock", type=Path, default=Path("configs/models.lock.json"))
    estimate.add_argument("--output", type=Path, default=Path("experiments/preflight.json"))
    args = parser.parse_args()
    if args.command == "verify-run":
        from .storage import verify_run
        manifest = verify_run(args.path)
        print(json.dumps({"status": "verified", "run_id": manifest["run_id"], "artifacts": len(manifest["artifacts"])}))
        return
    from .corpus import load_corpus, tokenize_with_alignment
    records = load_corpus(args.path if args.command == "validate-corpus" else args.corpus)
    if [r["number"] for r in records] != list(range(1, 155)):
        raise ValueError("Corpus must contain all 154 sonnets in order")
    for record in records:
        if len(record["lines"]) != {99: 15, 126: 12}.get(record["number"], 14):
            raise ValueError("Unexpected line count")
        if hashlib.sha256(record["text"].encode()).hexdigest() != record["text_sha256"]:
            raise ValueError("Normalized sonnet hash mismatch")
    if args.command == "validate-corpus":
        print(json.dumps({"status": "verified", "sonnets": len(records), "lines": sum(len(r["lines"]) for r in records)}))
        return
    from transformers import AutoTokenizer
    lock = json.loads(args.lock.read_text())["models"][0]
    tokenizer = AutoTokenizer.from_pretrained(lock["model_id"], revision=lock["revision"], use_fast=True,
                                             trust_remote_code=False)
    stats = []
    for record in records:
        encoding = tokenize_with_alignment(tokenizer, record)
        stats.append({"sonnet": record["number"], "tokens": len(encoding["input_ids"]),
                      "exact_prefixes": sum(p["exact_token_prefix"] for p in encoding["prefix_checks"]),
                      "exact_content_end_prefixes": sum(p["exact_token_prefix"] for p in encoding["prefix_checks_content_end"]),
                      "line_count": len(record["lines"])})
    config = lock["architecture"]
    total_tokens = sum(s["tokens"] for s in stats)
    # Embedding + every block + separately stored final-normalized state.
    per_token_residual_bytes = (config["num_hidden_layers"] + 2) * config["hidden_size"] * 2
    # Qwen3 projected Q width = 32*128; K/V = 8*128; MLP intermediate 12288.
    per_token_selected_bytes = 3 * (4096 + 1024 + 1024 + 4096 + 12288 + 4096) * 2
    report = {"model_id": lock["model_id"], "revision": lock["revision"], "sonnets": len(records),
              "total_tokens": total_tokens, "max_tokens": max(s["tokens"] for s in stats),
              "captured_dtype": "bfloat16", "bytes_per_element": 2,
              "raw_residual_bytes": total_tokens * per_token_residual_bytes,
              "selected_intermediate_bytes": total_tokens * per_token_selected_bytes,
              "selected_blocks": [0, 18, 35], "projected_total_gib": total_tokens * (per_token_residual_bytes + per_token_selected_bytes) / 2**30,
              "weight_memory_estimate_gib": 8.2e9 * 2 / 2**30,
              "h200_memory_gb": 141, "execution_seconds_unmeasured_range": [60, 1800],
              "h200_gpu_usd_per_second": 0.001261,
              "resource_usd_per_second": 0.001261 + 4 * 0.0000131 + 32 * 0.00000222,
              "gpu_worker_timeout_seconds": 1800, "gpu_startup_timeout_seconds": 600,
              "conservative_attempt_reservation_usd": 5,
              "approved_total_compute_ceiling_usd": 25,
              "price_source": "https://modal.com/pricing", "price_checked_local_date": "2026-10-17",
              "storage_note": "Uncompressed lossless primary tensors; excludes metadata/model cache; recurring GCS storage and transfer billed separately.",
              "runtime_note": "Planning range only, not measured throughput. Replace with pilot observation.", "per_sonnet": stats}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_sonnet"}, indent=2))


if __name__ == "__main__":
    main()
