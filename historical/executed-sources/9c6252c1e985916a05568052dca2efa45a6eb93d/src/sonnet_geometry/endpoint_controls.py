"""Endpoint and pooling interventions on immutable saved token states."""
from __future__ import annotations

import hashlib
import json
import struct
import unicodedata

import numpy as np
from scipy.stats import rankdata


POOLINGS = ("last_original", "last_lexical", "mean_all", "mean_no_newline", "mean_lexical")


def token_masks(record: dict, metadata: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Classify complete source-offset tokens; do not try to split mixed tokens."""
    tokens = metadata["tokens"]
    special = np.asarray(metadata["special_tokens_mask"], dtype=bool)
    if not (len(tokens) == len(special) == len(metadata["input_ids"]) == len(metadata["offset_mapping"])):
        raise ValueError("Token metadata lengths disagree")
    lexical, newline = np.zeros(len(tokens), bool), np.zeros(len(tokens), bool)
    counts = dict(lexical=0, pure_punctuation=0, separator_only=0, mixed_lexical_punctuation=0,
                  mixed_lexical_newline=0, punctuation_newline=0, unmapped_nonspecial=0, special=int(special.sum()))
    for i, token in enumerate(tokens):
        if token["token_index"] != i or token["input_id"] != metadata["input_ids"][i] or token["special"] != bool(special[i]):
            raise ValueError("Token identity disagrees with authoritative metadata")
        start, end = metadata["offset_mapping"][i]
        if [start, end] != [token["char_start"], token["char_end"]] or not 0 <= start <= end <= len(record["text"]):
            raise ValueError("Token offsets are invalid")
        text = record["text"][start:end]
        if text != token["text_slice"]:
            raise ValueError("Source slice differs from canonical text")
        if special[i]:
            continue
        lexical[i] = any(unicodedata.category(c)[0] in "LN" for c in text)
        newline[i] = any(c in "\r\n" for c in text)
        punct = any(unicodedata.category(c).startswith("P") for c in text)
        counts["lexical"] += int(lexical[i])
        counts["pure_punctuation"] += int(punct and not lexical[i] and not any(c.isspace() for c in text))
        counts["separator_only"] += int(bool(text) and text.isspace())
        counts["mixed_lexical_punctuation"] += int(lexical[i] and punct)
        counts["mixed_lexical_newline"] += int(lexical[i] and newline[i])
        counts["punctuation_newline"] += int(punct and newline[i] and not lexical[i])
        counts["unmapped_nonspecial"] += int(not text)
    if counts["unmapped_nonspecial"]:
        raise ValueError("Nonspecial unmapped tokens cannot be classified safely")
    masks = {"mean_all": ~special, "mean_no_newline": ~special & ~newline,
             "mean_lexical": ~special & lexical}
    return masks, counts


def verified_prefix_ends(metadata: dict, line_count: int) -> list[int]:
    checks = sorted(metadata["prefix_checks"], key=lambda item: item["line_number"])
    if metadata["boundary_policy"] != "include_following_line_separator" or [c["line_number"] for c in checks] != list(range(1, line_count + 1)):
        raise ValueError("Noncanonical line-prefix metadata")
    ends = []
    for check in checks:
        end = check["full_pass_token_end"]
        if not check["exact_token_prefix"] or type(end) is not int or check["prefix_input_ids"] != metadata["input_ids"][:end]:
            raise ValueError("Unverified token prefix")
        if end <= (ends[-1] if ends else 0) or end > len(metadata["input_ids"]):
            raise ValueError("Invalid prefix endpoint")
        ends.append(end)
    if ends[-1] != len(metadata["input_ids"]):
        raise ValueError("Last prefix does not cover the full saved sequence")
    return ends


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 0:
        raise ValueError("Zero vector in saved-state control")
    return float(1 - np.clip(np.dot(a, b) / denominator, -1, 1))


def pool_prefixes(states: np.ndarray, masks: dict[str, np.ndarray], ends: list[int]) -> dict:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or not np.isfinite(states).all() or states.shape[0] != len(masks["mean_all"]):
        raise ValueError("Invalid saved tensor")
    pooled, line_context, indices = {}, {}, {}
    for name in POOLINGS:
        mask = masks["mean_lexical"] if name == "last_lexical" else masks["mean_all"] if name == "last_original" else masks[name]
        vectors, endpoint_indices, line_vectors = [], [], []
        previous = 0
        for end in ends:
            included = np.flatnonzero(mask[:end])
            added = np.flatnonzero(mask[previous:end]) + previous
            if not len(included) or not len(added):
                raise ValueError("A filtered line has no included tokens")
            endpoint_indices.append(int(included[-1]))
            vectors.append(states[included[-1]] if name.startswith("last_") else states[included].mean(axis=0))
            line_vectors.append(states[added].mean(axis=0))
            previous = end
        pooled[name] = np.asarray(vectors)
        indices[name] = endpoint_indices
        if name.startswith("mean_"):
            line_context[name] = [cosine_distance(line_vectors[i], vectors[i - 1]) for i in range(1, len(ends))]
    return {"pooled": pooled, "line_context": line_context, "endpoint_indices": indices}


def q12_rdm(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or np.any(norms <= 0):
        raise ValueError("Invalid whole-poem vectors")
    unit = vectors / norms[:, None]
    result = np.round(np.clip(1 - unit @ unit.T, 0, 2), 12)
    np.fill_diagonal(result, 0)
    return result


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Rank inputs must be finite and aligned")
    ar, br = rankdata(np.round(a, 12)), rankdata(np.round(b, 12))
    ar, br = ar - ar.mean(), br - br.mean()
    denominator = np.linalg.norm(ar) * np.linalg.norm(br)
    return None if denominator == 0 else float(ar @ br / denominator)


def neighbor_sets(rdm: np.ndarray, k: int = 5) -> list[set[int]]:
    n = len(rdm)
    if rdm.shape != (n, n) or not 0 < k < n:
        raise ValueError("Require square RDM and a valid neighbor count")
    sets = []
    for i, row in enumerate(rdm):
        candidates = np.asarray([j for j in range(n) if j != i])
        threshold = np.partition(row[candidates], k - 1)[k - 1]
        sets.append(set(map(int, candidates[row[candidates] <= threshold])))
    return sets


def compare_rdms(a: np.ndarray, b: np.ndarray) -> dict:
    upper = np.triu_indices(len(a), 1)
    sets_a, sets_b = neighbor_sets(a), neighbor_sets(b)
    jaccard = [len(x & y) / len(x | y) for x, y in zip(sets_a, sets_b)]
    return {"rdm_spearman": spearman(a[upper], b[upper]), "mean_five_neighbor_jaccard": float(np.mean(jaccard)),
            "per_poem_five_neighbor_jaccard": jaccard,
            "mean_neighbor_set_sizes": [float(np.mean([len(x) for x in sets_a])), float(np.mean([len(x) for x in sets_b]))]}


def decode_tensor(data: bytes, tensor: dict) -> np.ndarray:
    shape, dtype = tensor["shape"], tensor["dtype"]
    sizes = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}
    if dtype not in sizes or len(shape) != 2 or len(data) != int(np.prod(shape)) * sizes[dtype]:
        raise ValueError("Unsupported or truncated tensor range")
    if dtype == "BF16":
        # IEEE BF16 high 16 bits of IEEE float32; no arithmetic conversion.
        result = (np.frombuffer(data, dtype="<u2").astype("<u4") << 16).view("<f4")
    else:
        result = np.frombuffer(data, dtype={"F16": "<f2", "F32": "<f4", "F64": "<f8"}[dtype])
    return result.reshape(shape).astype(np.float64)


def parse_header(prefix: bytes) -> tuple[dict, int]:
    if len(prefix) < 8:
        raise ValueError("Missing safetensors length")
    size = struct.unpack("<Q", prefix[:8])[0]
    if size < 2 or size > 100_000_000 or len(prefix) < size + 8:
        raise ValueError("Truncated or invalid safetensors header")
    return json.loads(prefix[8:8 + size]), size + 8


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
