"""Pinned conventional baselines, with no silent truncation or GPU use."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time
from typing import Any, Sequence

import numpy as np

from .analysis import cosine_similarity_matrix, rsa, stylometric_features
from .corpus import load_corpus


def conventional_baselines(texts: Sequence[str], tfidf_config: dict[str, Any]) -> dict[str, Any]:
    """Fit descriptive corpus baselines; supervised reuse must refit by fold."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    config = dict(tfidf_config)
    config["ngram_range"] = tuple(config["ngram_range"])
    tfidf = TfidfVectorizer(**config, dtype=np.float64)
    sparse = tfidf.fit_transform(texts)
    similarity = (sparse @ sparse.T).toarray()
    if np.any(np.diag(similarity) <= 0):
        raise ValueError("TF-IDF produced an empty document representation")
    x, names = stylometric_features(texts)
    scale = x.std(axis=0)
    standardized = np.divide(x - x.mean(axis=0), scale, out=np.zeros_like(x), where=scale > 0)
    style_similarity = cosine_similarity_matrix(standardized)
    return {
        "tfidf_character_ngrams": {"configuration": tfidf_config, "feature_count": sparse.shape[1],
                                    "cosine_distance": np.clip(1 - similarity, 0, 2),
                                    "vocabulary_sha256": hashlib.sha256(json.dumps(tfidf.vocabulary_, sort_keys=True).encode()).hexdigest()},
        "standardized_stylometry": {"feature_names": names, "raw_features": x.tolist(),
                                    "population_means": x.mean(axis=0).tolist(), "population_scales": scale.tolist(),
                                    "constant_features": [name for name, value in zip(names, scale) if value == 0],
                                    "normalization": "column population z-score then row L2 cosine; constant columns zeroed",
                                    "cosine_distance": np.clip(1 - style_similarity, 0, 2)},
    }


def check_lengths(token_ids: Sequence[Sequence[int]], limit: int) -> list[int]:
    lengths = [len(ids) for ids in token_ids]
    if not lengths or min(lengths) < 1:
        raise ValueError("Empty baseline input")
    if max(lengths) > limit:
        raise ValueError(f"Baseline sequence exceeds locked limit {limit}; truncation is forbidden")
    return lengths


def minilm_embeddings(texts: Sequence[str], lock: dict[str, Any], *, cache_dir: str | Path,
                      batch_size: int = 16, local_files_only: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """Official attention-mask mean pooling including CLS/SEP, then L2 normalize."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not re.fullmatch(r"[0-9a-f]{40}", lock["revision"]):
        raise ValueError("Baseline checkpoint must have an exact 40-hex revision")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    kwargs = {"revision": lock["revision"], "cache_dir": str(cache_dir),
              "local_files_only": local_files_only, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(lock["model_id"], use_fast=True, **kwargs)
    encoded = tokenizer(list(texts), truncation=False, padding=False)
    lengths = check_lengths(encoded["input_ids"], lock["max_tokens_including_special"])
    model = AutoModel.from_pretrained(lock["model_id"], use_safetensors=True, **kwargs).cpu().eval()
    check_lengths(encoded["input_ids"], int(model.config.max_position_embeddings))
    started = time.perf_counter()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(list(texts[start:start + batch_size]), padding=True, truncation=False, return_tensors="pt")
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            rows.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
    vectors = np.concatenate(rows)
    return vectors, {"model_id": lock["model_id"], "revision": lock["revision"],
                     "device": "cpu", "eval_mode": True, "inference_mode": True,
                     "dtype": str(vectors.dtype), "pooling": lock["pooling"], "dimension": vectors.shape[1],
                     "batch_size": batch_size, "cpu_threads": torch.get_num_threads(),
                     "forward_and_pooling_seconds": time.perf_counter() - started,
                     "truncation": False, "token_counts": lengths,
                     "documented_limit": lock["max_tokens_including_special"],
                     "tokenizer_backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                     "model_config_sha256": hashlib.sha256(model.config.to_json_string().encode()).hexdigest()}


def run_baselines(corpus_path: str | Path, lock_path: str | Path, *, cache_dir: str | Path,
                  local_files_only: bool = True) -> dict[str, Any]:
    corpus_path, lock_path = Path(corpus_path), Path(lock_path)
    records = load_corpus(corpus_path)
    lock = json.loads(lock_path.read_text())
    texts = [record["text"] for record in records]
    results = conventional_baselines(texts, lock["tfidf"])
    vectors, metadata = minilm_embeddings(texts, lock["sentence_embedding"], cache_dir=cache_dir,
                                          local_files_only=local_files_only)
    results["minilm_sentence_embedding"] = {**metadata, "cosine_distance": np.clip(1 - cosine_similarity_matrix(vectors), 0, 2)}
    comparisons = []
    names = list(results)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            comparisons.append({"baseline_a": name, "baseline_b": other,
                                "rsa_spearman": rsa(results[name]["cosine_distance"], results[other]["cosine_distance"]),
                                "inference": "descriptive; no significance test"})
    for baseline in results.values():
        rdm = baseline["cosine_distance"]
        np.fill_diagonal(rdm, 0)
        scores = rdm.sum(axis=1) / (len(records) - 1)
        baseline["mean_distance_anomaly_ranking"] = [
            {"sonnet_id": records[i]["sonnet_id"], "mean_cosine_distance": float(scores[i])}
            for i in np.argsort(-scores, kind="stable")]
        # Explicit compact derived export; source data and code permit regeneration.
        baseline["cosine_distance"] = np.round(rdm, 8).tolist()
    return {"schema_version": "conventional-baselines-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
            "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
            "sonnet_ids": [r["sonnet_id"] for r in records], "baselines": results, "pairwise_rsa": comparisons,
            "rdm_export_decimal_places": 8,
            "environment": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "numpy", "scikit-learn"]},
            "inference": "Descriptive corpus geometry; no held-out performance or literary finding. TF-IDF and scaling fit on this corpus only."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("datasets/normalized/sonnets.jsonl"))
    parser.add_argument("--lock", type=Path, default=Path("configs/baselines.lock.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/private/tmp/shore-hf-cache/hub"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    report = run_baselines(args.corpus, args.lock, cache_dir=args.cache_dir, local_files_only=not args.allow_download)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    print(json.dumps({"sonnets": len(report["sonnet_ids"]), "baselines": list(report["baselines"]),
                      "output": str(args.output), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
