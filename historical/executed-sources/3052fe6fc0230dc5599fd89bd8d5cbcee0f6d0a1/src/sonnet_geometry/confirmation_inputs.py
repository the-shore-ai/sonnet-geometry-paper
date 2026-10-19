"""Frozen repeated interventions for the independent Sidney population.

This module prepares inputs only. It never imports a model or contacts a cloud.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import random

from .storage import canonical_json
from .text_controls import WORD

SEEDS = (2026101801, 2026101802, 2026101803, 2026101804, 2026101805)
CONTROLS = ("word_shuffle", "line_shuffle", "token_shuffle")
PRECISIONS = ("bf16", "fp32")


def conditions():
    return [
        {"name": "original_padded_" + precision, "intervention": "original", "seed": None,
         "precision": precision} for precision in PRECISIONS
    ] + [
        {"name": f"{control}_seed{seed}_padded_{precision}", "intervention": control,
         "seed": seed, "precision": precision}
        for control in CONTROLS for seed in SEEDS for precision in PRECISIONS
    ]


def _rng(record, seed, control):
    identity = record.get("sonnet_id")
    if not identity or seed not in SEEDS or control not in CONTROLS:
        raise ValueError("Stable poem identity and registered seed/control required")
    digest = hashlib.sha256(canonical_json(["confirmation-v4", identity, seed, control])).digest()
    return random.Random(int.from_bytes(digest, "big"))


def text_variants(record, seed):
    lines = record["text"].splitlines()
    if len(lines) != 14 or any(not line.strip() for line in lines):
        raise ValueError("Frozen population requires fourteen nonempty lines")
    rng = _rng(record, seed, "word_shuffle")
    shuffled = []
    for line in lines:
        matches = list(WORD.finditer(line))
        words = [m.group() for m in matches]
        replacement = words[:-1].copy()
        rng.shuffle(replacement)
        if words:
            replacement.append(words[-1])
        pieces, cursor = [], 0
        for match, word in zip(matches, replacement):
            pieces.extend([line[cursor:match.start()], word])
            cursor = match.end()
        pieces.append(line[cursor:])
        changed = "".join(pieces)
        if Counter(WORD.findall(changed)) != Counter(words) or len(changed) != len(line):
            raise ValueError("Word intervention changed inventory or character length")
        shuffled.append(changed)
    order = list(range(len(lines) - 2))
    _rng(record, seed, "line_shuffle").shuffle(order)
    order.extend([len(lines) - 2, len(lines) - 1])
    return {
        "original": {"text": record["text"], "line_order": list(range(1, 15))},
        "word_shuffle": {"text": "\n".join(shuffled), "line_order": list(range(1, 15))},
        "line_shuffle": {"text": "\n".join(lines[i] for i in order),
                         "line_order": [i + 1 for i in order]},
    }


def encode_text(tokenizer, text, pad_length):
    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer required")
    result = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True,
                       add_special_tokens=True, truncation=False)
    ids, specials = list(result["input_ids"]), list(result["special_tokens_mask"])
    offsets = [list(pair) for pair in result["offset_mapping"]]
    if not ids or len(ids) > pad_length:
        raise ValueError(f"Input of {len(ids)} tokens exceeds frozen shape {pad_length}; no truncation")
    if len(specials) != len(ids) or len(offsets) != len(ids):
        raise ValueError("Invalid tokenizer metadata")
    lexical = [i for i, ((a, b), s) in enumerate(zip(offsets, specials))
               if not s and any(c.isalpha() for c in text[a:b])]
    if not lexical:
        raise ValueError("No lexical token")
    return {"input_ids": ids, "offsets": offsets, "special_tokens_mask": specials,
            "lexical_indices": lexical, "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


def token_permutation(encoding, record, seed):
    result = deepcopy(encoding)
    text, groups = encoding["text"], {}
    for i in encoding["lexical_indices"]:
        a, b = encoding["offsets"][i]
        if "\n" not in text[a:b] and all(c.isalpha() or c.isspace() for c in text[a:b]):
            groups.setdefault(text[:a].count("\n"), []).append(i)
    permutation = list(range(len(encoding["input_ids"])))
    rng = _rng(record, seed, "token_shuffle")
    for group in groups.values():
        # Same eligibility as v3: the last eligible piece also stays fixed.
        movable = group[:-1]
        sources = movable.copy()
        rng.shuffle(sources)
        for dest, source in zip(movable, sources):
            permutation[dest] = source
    result["input_ids"] = [encoding["input_ids"][i] for i in permutation]
    result["source_token_indices"] = permutation
    result["offset_semantics"] = "destination slots in original text; content follows source_token_indices"
    result["text_is_original_slot_template_not_decoding"] = True
    if Counter(result["input_ids"]) != Counter(encoding["input_ids"]):
        raise ValueError("Token inventory changed")
    return result


def forward_payload(case):
    ids = case["tokenization"]["input_ids"]
    width = case["forward_length"]
    if not ids or len(ids) > width:
        raise ValueError("Invalid forward length")
    return {"input_ids": ids + [case["pad_token_id"]] * (width - len(ids)),
            "attention_mask": case["attention_mask"], "position_ids": case["position_ids"]}


def build_cases(record, tokenizer, *, pad_length):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    originals = encode_text(tokenizer, record["text"], pad_length)
    originals["original_line_order"] = list(range(1, 15))
    encodings = {("original", None): originals}
    for seed in SEEDS:
        variants = text_variants(record, seed)
        for control in ("word_shuffle", "line_shuffle"):
            encoding = encode_text(tokenizer, variants[control]["text"], pad_length)
            encoding["original_line_order"] = variants[control]["line_order"]
            encodings[(control, seed)] = encoding
        encodings[("token_shuffle", seed)] = token_permutation(originals, record, seed)
    rows = []
    for spec in conditions():
        encoding = deepcopy(encodings[(spec["intervention"], spec["seed"])])
        count = len(encoding["input_ids"])
        case = {"sonnet": record["number"], "poem_id": record["sonnet_id"], "condition": spec["name"],
                "intervention": spec["intervention"], "seed": spec["seed"], "precision": spec["precision"],
                "tokenization": encoding, "forward_length": pad_length, "pad_token_id": pad_id,
                "attention_mask": [1] * count + [0] * (pad_length - count),
                "position_ids": list(range(pad_length)),
                "input_ids_sha256": hashlib.sha256(canonical_json(encoding["input_ids"])).hexdigest()}
        case["forward_sha256"] = hashlib.sha256(canonical_json(forward_payload(case))).hexdigest()
        rows.append(case)
    return rows


def audit_cases(cases):
    by_poem = {}
    for case in cases:
        by_poem.setdefault(case["sonnet"], []).append(case)
    rows = []
    expected = {spec["name"] for spec in conditions()}
    for number, items in by_poem.items():
        named = {case["condition"]: case for case in items}
        if len(items) != 32 or set(named) != expected:
            raise ValueError("Missing or duplicate conditions")
        original = named["original_padded_bf16"]["tokenization"]
        for case in items:
            spec = case
            counterpart = named[case["condition"].rsplit("_", 1)[0] + "_" +
                                ("fp32" if case["precision"] == "bf16" else "bf16")]
            if case["tokenization"] != counterpart["tokenization"] or forward_payload(case) != forward_payload(counterpart):
                raise ValueError("Dtype pair inputs differ")
            if case["forward_sha256"] != hashlib.sha256(canonical_json(forward_payload(case))).hexdigest():
                raise ValueError("Forward hash mismatch")
            encoding = case["tokenization"]
            same_inventory = Counter(encoding["input_ids"]) == Counter(original["input_ids"])
            if spec["intervention"] == "token_shuffle" and not same_inventory:
                raise ValueError("Token inventory differs")
            if encoding["input_ids"][-1] != original["input_ids"][-1]:
                raise ValueError("Final real token differs")
            if encoding["input_ids"][encoding["lexical_indices"][-1]] != original["input_ids"][original["lexical_indices"][-1]]:
                raise ValueError("Final lexical token differs")
            if case["precision"] == "bf16":
                rows.append({"sonnet": number, "intervention": spec["intervention"], "seed": spec["seed"],
                             "token_count": len(encoding["input_ids"]),
                             "token_count_delta": len(encoding["input_ids"]) - len(original["input_ids"]),
                             "token_inventory_preserved": same_inventory,
                             "input_changed": encoding["input_ids"] != original["input_ids"],
                             "paired_precision_inputs_identical": True})
    return rows
