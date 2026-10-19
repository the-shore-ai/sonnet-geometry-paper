"""Freeze tokenizer-exact repeated controls; no checkpoint weights or paid execution."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from sonnet_geometry.confirmation_inputs import conditions, build_cases, audit_cases
from sonnet_geometry.extraction import _tokenizer_metadata
from sonnet_geometry.storage import canonical_json, sha256_file, write_exclusive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["qwen", "gemma", "muse"])
    parser.add_argument("--protocol", type=Path, default=Path("experiments/confirmation-v4/protocol.json"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    if protocol["conditions"] != conditions():
        raise ValueError("Protocol conditions differ from code")
    source = Path(protocol["population"]["path"])
    if sha256_file(source) != protocol["population"]["sha256"]:
        raise ValueError("Corpus hash mismatch")
    records = [json.loads(line) for line in source.read_text().splitlines() if line]
    if [record["number"] for record in records] != protocol["sonnets"]:
        raise ValueError("Population differs")
    from transformers import AutoTokenizer
    lock = protocol["model_locks"][args.model]
    tokenizer = AutoTokenizer.from_pretrained(lock["model_id"], revision=lock["revision"],
                                             trust_remote_code=False, use_fast=True, local_files_only=True)
    cases = [case for record in records for case in build_cases(record, tokenizer, pad_length=protocol["pad_length"])]
    audit = audit_cases(cases)
    tokenizer_metadata = _tokenizer_metadata(tokenizer)
    estimate = sum(len(case["tokenization"]["input_ids"]) * len(protocol["states"]) * lock["hidden_size"] *
                   (2 if case["precision"] == "bf16" else 4) for case in cases)
    collisions = []
    for record in records:
        for control in ("word_shuffle", "line_shuffle", "token_shuffle"):
            selected = [case for case in cases if case["sonnet"] == record["number"] and
                        case["intervention"] == control and case["precision"] == "bf16"]
            collisions.append({"sonnet": record["number"], "intervention": control,
                               "unique_input_count": len({c["input_ids_sha256"] for c in selected}),
                               "seeds": len(selected)})
    result = {"schema_version": "confirmation-token-preflight-v4",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": sha256_file(args.protocol), "corpus_sha256": sha256_file(source),
              "model": args.model, **lock, "tokenizer": tokenizer_metadata,
              "tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
              "case_count": len(cases), "unique_inputs": len(cases) // 2,
              "max_tokens": max(len(c["tokenization"]["input_ids"]) for c in cases),
              "pad_length": protocol["pad_length"], "primary_tensor_bytes_estimate": estimate,
              "new_model_inference": False, "cases": cases, "audit": audit,
              "randomization_collisions": collisions}
    # Compact exact inputs keep the reproducibility artifact small enough for Git.
    write_exclusive(args.output, (json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode())
    print(json.dumps({"path": str(args.output), "sha256": sha256_file(args.output),
                      "cases": len(cases), "max_tokens": result["max_tokens"],
                      "tensor_bytes_estimate": estimate, "new_model_inference": False}))


if __name__ == "__main__":
    main()
