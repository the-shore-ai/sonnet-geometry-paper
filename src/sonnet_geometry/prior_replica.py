"""Methodological replica of the inspected isolated-line BERT geometry recipe.

This substitutes pinned BERT-base for unavailable specialized checkpoints and
uses this project's corpus. It is not a reproduction of the prior findings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

from .corpus import load_corpus


def prior_trajectory_metrics(embeddings: Any) -> dict[str, Any]:
    """Vectorized cell-10/12 formula, including its epsilon/degenerate rules.

    The source's norm+1e-8 adjustment and near-closed-path return of zero are
    preserved explicitly. Float64 metric arithmetic is a documented difference.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite [line, feature] embeddings")
    epsilon = 1e-8
    count = x.shape[0]
    steps = np.diff(x, axis=0)
    step_lengths = np.linalg.norm(steps, axis=1)
    directions = steps / (step_lengths[:, None] + epsilon)
    cosines = np.sum(directions[:-1] * directions[1:], axis=1)
    angles = np.arccos(np.clip(cosines, -1, 1))
    endpoint_distance = float(np.linalg.norm(x[-1] - x[0])) if count else 0.0
    angle_sum = float(angles.sum())
    score = 0.0 if count < 2 or endpoint_distance < epsilon else angle_sum / endpoint_distance
    return {"num_lines": count, "angle_sum_radians": angle_sum,
            "first_last_euclidean_distance": endpoint_distance, "original_overall_tortuosity": score,
            "mean_step_distance": float(step_lengths.mean()) if len(steps) else 0.0,
            "max_step_distance": float(step_lengths.max()) if len(steps) else 0.0,
            "final_two_line_distance": float(step_lengths[-1]) if len(steps) else 0.0,
            "step_distances": step_lengths.tolist(), "turn_angles_radians": angles.tolist(),
            "steps_below_epsilon": int(np.count_nonzero(step_lengths < epsilon)),
            "endpoint_below_epsilon": endpoint_distance < epsilon, "epsilon": epsilon}


def run_replica(corpus_path: str | Path, lock_path: str | Path, *, cache_dir: str | Path,
                batch_size: int = 64, local_files_only: bool = True) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    corpus_path, lock_path = Path(corpus_path), Path(lock_path)
    lock = json.loads(lock_path.read_text())
    if not re.fullmatch(r"[0-9a-f]{40}", lock["model"]["revision"]):
        raise ValueError("The BERT checkpoint must be pinned to a full revision")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    records = load_corpus(corpus_path)
    lines = [line for record in records for line in record["lines"]]
    sonnet_numbers = [record["number"] for record in records for _ in record["lines"]]
    line_numbers = [i for record in records for i in range(1, len(record["lines"]) + 1)]
    kwargs = {"revision": lock["model"]["revision"], "cache_dir": str(cache_dir),
              "local_files_only": local_files_only, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(lock["model"]["model_id"], use_fast=True, **kwargs)
    token_counts = [len(ids) for ids in tokenizer(lines, truncation=False)["input_ids"]]
    if max(token_counts) > lock["model"]["max_tokens"]:
        raise ValueError("An isolated verse line exceeds the context limit; truncation is forbidden")
    model = AutoModel.from_pretrained(lock["model"]["model_id"], use_safetensors=True, **kwargs).cpu().eval()
    if max(token_counts) > model.config.max_position_embeddings:
        raise ValueError("A line exceeds the model's position table")
    started = time.perf_counter()
    vectors = []
    with torch.inference_mode():
        for offset in range(0, len(lines), batch_size):
            encoded = tokenizer(lines[offset:offset + batch_size], padding=True, truncation=False,
                                return_special_tokens_mask=True, return_tensors="pt")
            include = encoded["attention_mask"].bool() & ~encoded.pop("special_tokens_mask").bool()
            if not torch.all(include.sum(dim=1) > 0):
                raise ValueError("A verse line contains no nonspecial tokens")
            states = model(**encoded).last_hidden_state
            mask = include.unsqueeze(-1).to(states.dtype)
            pooled = (states * mask).sum(dim=1) / mask.sum(dim=1)
            vectors.append(pooled.cpu().numpy())
    matrix = np.concatenate(vectors)
    forward_seconds = time.perf_counter() - started
    metrics, offset = [], 0
    for record in records:
        count = len(record["lines"])
        metrics.append({"sonnet_id": record["sonnet_id"], "number": record["number"],
                        "token_counts": token_counts[offset:offset + count],
                        **prior_trajectory_metrics(matrix[offset:offset + count])})
        offset += count
    scores = np.array([row["original_overall_tortuosity"] for row in metrics])
    report = {"schema_version": "prior-method-replica-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Methodological replica with BERT-base and our Gutenberg corpus; not a reproduction of EEBO-BERT/Poetry-BERT results",
              "source_recipe": lock["source_recipe"], "model": lock["model"],
              "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
              "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
              "sonnets": len(records), "verse_lines": len(lines), "embedding_dimension": matrix.shape[1],
              "execution": {"device": "cpu", "batch_size": batch_size, "eval_mode": True,
                            "pooling": "isolated line last hidden layer mean; exclude CLS/SEP/padding; no L2 normalization",
                            "truncation": False, "token_count_min": min(token_counts), "token_count_max": max(token_counts),
                            "forward_and_pooling_seconds": forward_seconds, "embedding_dtype": str(matrix.dtype),
                            "metric_dtype": "float64", "cpu_threads": torch.get_num_threads(),
                            "tokenizer_backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()},
              "environment": {package: importlib.metadata.version(package) for package in ["torch", "transformers", "numpy"]},
              "metric_summary": {"mean": float(scores.mean()), "population_sd": float(scores.std()),
                                 "min": float(scores.min()), "max": float(scores.max())},
              "sonnet_metrics": metrics,
              "ranking": [{"sonnet_id": records[i]["sonnet_id"], "original_overall_tortuosity": float(scores[i])}
                          for i in np.argsort(-scores, kind="stable")],
              "limitations": ["Specialized EEBO-BERT and Poetry-BERT checkpoints were not obtained or validated for this run.",
                              "Corpus edition and batched CPU execution can differ from the source notebook.",
                              "The ratio depends on feature scale and small endpoint distances; it is not a validated measure of literary complexity.",
                              "The original epsilon convention assigns pi/2 to a turn involving a zero displacement and returns zero for near-closed trajectories; these quirks are preserved and flagged."]}
    archive = {"line_embeddings": matrix, "sonnet_numbers": np.array(sonnet_numbers), "line_numbers": np.array(line_numbers)}
    return report, archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("datasets/normalized/sonnets.jsonl"))
    parser.add_argument("--lock", type=Path, default=Path("configs/prior-replica.lock.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/private/tmp/shore-hf-cache/hub"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embeddings-output", type=Path)
    args = parser.parse_args()
    report, archive = run_replica(args.corpus, args.lock, cache_dir=args.cache_dir, batch_size=args.batch_size,
                                  local_files_only=not args.allow_download)
    if args.embeddings_output:
        args.embeddings_output.parent.mkdir(parents=True, exist_ok=True)
        with args.embeddings_output.open("xb") as handle:
            np.savez_compressed(handle, **archive)
        report["embedding_archive"] = {"path": str(args.embeddings_output),
                                        "sha256": hashlib.sha256(args.embeddings_output.read_bytes()).hexdigest(),
                                        "format": "lossless NumPy NPZ; float32 line embeddings plus identity arrays"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "sonnets": report["sonnets"], "lines": report["verse_lines"],
                      "cpu_forward_seconds": report["execution"]["forward_and_pooling_seconds"], "scope": "methodological replica only"}))


if __name__ == "__main__":
    main()
