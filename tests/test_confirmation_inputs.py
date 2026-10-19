from collections import Counter
from copy import deepcopy
import re

import pytest

from sonnet_geometry.confirmation_inputs import (
    SEEDS, WORD, audit_cases, build_cases, conditions, forward_payload, text_variants,
)


class TinyTokenizer:
    is_fast = True
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, **_kwargs):
        matches = list(re.finditer(r"\w+|[^\w\s]", text))
        return {"input_ids": [2 + sum(ord(c) for c in m.group()) for m in matches],
                "special_tokens_mask": [0] * len(matches),
                "offset_mapping": [(m.start(), m.end()) for m in matches]}


def poem(number=3):
    return {"number": number, "sonnet_id": f"sidney-newman1591-{number:03d}",
            "text": "\n".join(f"Love makes my patient thoughts dwell near thee {chr(65+i)}."
                              for i in range(14))}


def test_repeated_controls_preserve_frozen_invariants_and_are_reproducible():
    record = poem()
    originals = record["text"].splitlines()
    outputs = [text_variants(record, seed) for seed in SEEDS]
    assert outputs == [text_variants(record, seed) for seed in SEEDS]
    for output in outputs:
        for before, after in zip(originals, output["word_shuffle"]["text"].splitlines()):
            assert Counter(WORD.findall(before)) == Counter(WORD.findall(after))
            assert len(before) == len(after)
            assert WORD.findall(before)[-1] == WORD.findall(after)[-1]
        lines = output["line_shuffle"]["text"].splitlines()
        assert Counter(lines) == Counter(originals)
        assert lines[-2:] == originals[-2:]
    for control in ("word_shuffle", "line_shuffle"):
        assert len({x[control]["text"] for x in outputs}) == 5
    assert text_variants(poem(5), SEEDS[0]) != outputs[0]


def test_all_32_inputs_have_exact_precision_pairs_and_token_inventory():
    cases = build_cases(poem(), TinyTokenizer(), pad_length=256)
    assert len(cases) == 32
    assert {c["condition"] for c in cases} == {s["name"] for s in conditions()}
    audited = audit_cases(cases)
    assert len(audited) == 16
    assert all(c["paired_precision_inputs_identical"] for c in audited)
    for a, b in zip(cases[::2], cases[1::2]):
        assert a["forward_sha256"] == b["forward_sha256"]
        assert forward_payload(a) == forward_payload(b)
    original = cases[0]["tokenization"]
    for case in cases:
        encoding = case["tokenization"]
        if case["intervention"] != "token_shuffle":
            continue
        assert Counter(encoding["input_ids"]) == Counter(original["input_ids"])
        for i, (start, end) in enumerate(original["offsets"]):
            if not original["text"][start:end].isalpha():
                assert encoding["source_token_indices"][i] == i


def test_missing_conditions_tampered_inputs_and_truncation_fail_closed():
    cases = build_cases(poem(), TinyTokenizer(), pad_length=256)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        audit_cases(cases[:-1])
    changed = deepcopy(cases)
    changed[1]["tokenization"]["input_ids"][0] += 1
    with pytest.raises(ValueError, match="Dtype pair"):
        audit_cases(changed)
    changed = deepcopy(cases)
    changed[0]["forward_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_cases(changed)
    with pytest.raises(ValueError, match="no truncation"):
        build_cases(poem(), TinyTokenizer(), pad_length=5)


def test_unchanged_shuffles_are_retained_without_resampling():
    record = {"number": 3, "sonnet_id": "sidney-newman1591-003", "text": "\n".join(["Love."] * 14)}
    cases = build_cases(record, TinyTokenizer(), pad_length=256)
    audit = audit_cases(cases)
    assert len(cases) == 32
    assert all(not row["input_changed"] for row in audit)


def test_population_shape_and_seed_restrictions():
    with pytest.raises(ValueError, match="fourteen"):
        text_variants({**poem(), "text": "short"}, SEEDS[0])
    with pytest.raises(ValueError, match="registered"):
        text_variants(poem(), 999)
