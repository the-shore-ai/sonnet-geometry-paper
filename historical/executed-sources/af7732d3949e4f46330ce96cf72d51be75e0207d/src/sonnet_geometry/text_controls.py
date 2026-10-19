"""Frozen text interventions and selected-state capture; no generated labels."""
from collections import Counter
import hashlib
import random
import re

import numpy as np


SEED = 20261019
SONNETS = (1, 18, 29, 53, 73, 75, 99, 116, 126, 130, 142, 153)
PAD_LENGTH = 256
EARLY_BLOCK = {"qwen": 1, "gemma": 4, "muse": 4}
DEPTH75_BLOCK = {"qwen": 26, "gemma": 44, "muse": 38}
LAST_BLOCK = {"qwen": 35, "gemma": 59, "muse": 51}
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)


def text_variants(record):
    """Preserve word spellings/counts and each line's final word in the word control."""
    lines = record["text"].splitlines()
    rng = random.Random(SEED + int(record["number"]))
    changed = []
    for line in lines:
        matches = list(WORD.finditer(line))
        words = [m.group() for m in matches]
        replacement = words[:-1].copy()
        rng.shuffle(replacement)
        if words:
            replacement.append(words[-1])
        parts, cursor = [], 0
        for match, word in zip(matches, replacement):
            parts.extend([line[cursor:match.start()], word]); cursor = match.end()
        parts.append(line[cursor:])
        new = "".join(parts)
        if Counter(WORD.findall(new)) != Counter(words) or len(new) != len(line):
            raise ValueError("Word control changed the registered lexical inventory")
        changed.append(new)
    order = list(range(len(lines) - 2))
    random.Random(SEED * 2 + int(record["number"])).shuffle(order)
    order += [len(lines) - 2, len(lines) - 1]
    return {"original": {"text": record["text"], "line_order": list(range(1, len(lines) + 1))},
            "word_shuffle": {"text": "\n".join(changed), "line_order": list(range(1, len(lines) + 1))},
            "line_shuffle": {"text": "\n".join(lines[i] for i in order), "line_order": [i + 1 for i in order]}}


def encode_text(tokenizer, text):
    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer with source offsets required")
    encoded = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True,
                        add_special_tokens=True, truncation=False)
    ids = list(encoded["input_ids"])
    specials = list(encoded["special_tokens_mask"])
    offsets = [list(x) for x in encoded["offset_mapping"]]
    if not ids or len(ids) > PAD_LENGTH:
        raise ValueError("Frozen 256-token forward shape cannot fit this input")
    lexical = [i for i, ((a, b), s) in enumerate(zip(offsets, specials))
               if not s and any(c.isalpha() for c in text[a:b])]
    if not lexical:
        raise ValueError("No lexical token")
    return {"input_ids": ids, "offsets": offsets, "special_tokens_mask": specials,
            "lexical_indices": lexical, "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


def token_permutation(encoding, number):
    """A separate token-inventory control, not grammatical word reordering.

    Keep line boundaries, punctuation-bearing tokens, special tokens, and each
    line's last lexical token fixed. Permute lexical token pieces in other slots.
    Offsets describe destination slots; the permutation retains source indices.
    """
    result = dict(encoding)
    text = encoding["text"]
    groups = {}
    for i in encoding["lexical_indices"]:
        a, b = encoding["offsets"][i]
        if "\n" not in text[a:b] and all(c.isalpha() or c.isspace() for c in text[a:b]):
            groups.setdefault(text[:a].count("\n"), []).append(i)
    permutation = list(range(len(encoding["input_ids"])))
    rng = random.Random(SEED * 3 + number)
    for group in groups.values():
        movable = group[:-1]
        source = movable.copy(); rng.shuffle(source)
        for dest, origin in zip(movable, source):
            permutation[dest] = origin
    result["input_ids"] = [encoding["input_ids"][i] for i in permutation]
    result["source_token_indices"] = permutation
    result["offset_semantics"] = "destination slots in original text; token content follows source_token_indices"
    result["text_is_original_slot_template_not_decoding"] = True
    assert Counter(result["input_ids"]) == Counter(encoding["input_ids"])
    return result


def case_inputs(record, tokenizer):
    variants = text_variants(record)
    enc = {key: encode_text(tokenizer, value["text"]) for key, value in variants.items()}
    for key, value in variants.items():
        enc[key]["original_line_order"] = value["line_order"]
    return [("original_native_bf16", enc["original"], None),
            ("original_padded_bf16", enc["original"], PAD_LENGTH),
            ("word_shuffle_padded_bf16", enc["word_shuffle"], PAD_LENGTH),
            ("line_shuffle_padded_bf16", enc["line_shuffle"], PAD_LENGTH),
            ("token_shuffle_padded_bf16", token_permutation(enc["original"], int(record["number"])), PAD_LENGTH)]


def capture_selected(model, encoding, *, blocks, pad_to=None, pad_id=0):
    """Keep source-precision real-token states; exclude masked padding from artifacts/pooling."""
    import torch
    from .adapters import get_adapter
    adapter = get_adapter(model)
    if len(set(blocks.values())) != len(blocks) or any(i >= len(adapter.layers) or i < 0 for i in blocks.values()):
        raise ValueError("Distinct valid block indices required")
    count = len(encoding["input_ids"])
    width = count if pad_to is None else pad_to
    if width < count:
        raise ValueError("Truncation forbidden")
    device = next(model.parameters()).device
    ids = torch.tensor([encoding["input_ids"] + [pad_id] * (width - count)], device=device)
    mask = torch.tensor([[1] * count + [0] * (width - count)], device=device)
    pos = torch.arange(width, device=device).unsqueeze(0)
    captures, handles = {}, []
    def save(name, value):
        if isinstance(value, tuple): value = value[0]
        if name in captures or value.ndim != 3 or value.shape[:2] != (1, width):
            raise ValueError("Unexpected selected hook shape or repetition")
        array = value[0, :count].detach().cpu().contiguous()
        if not torch.isfinite(array).all(): raise ValueError("Nonfinite capture")
        captures[name] = array
    def prehook(_module, args, kwargs):
        save("input", args[0] if args else kwargs["hidden_states"])
    def hook(name):
        return lambda _module, _args, out: save(name, out)
    try:
        handles.append(adapter.layers[0].register_forward_pre_hook(prehook, with_kwargs=True))
        for name, index in blocks.items():
            handles.append(adapter.layers[index].register_forward_hook(hook(name)))
        handles.append(adapter.norm.register_forward_hook(hook("final_normalized")))
        with torch.inference_mode():
            output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
                           output_hidden_states=False, output_attentions=False, return_dict=True)
        if not torch.equal(output.last_hidden_state[0, :count].cpu(), captures["final_normalized"]):
            raise ValueError("Final-normalization capture mismatch")
    finally:
        for handle in handles: handle.remove()
    if set(captures) != {"input", "final_normalized", *blocks}:
        raise ValueError("Incomplete selected capture")
    return captures


def pool_selected(captures, encoding):
    mask = np.logical_not(encoding["special_tokens_mask"])
    lexical = encoding["lexical_indices"][-1]
    if not mask.any(): raise ValueError("No text token for pooling")
    return {f"{state}.{pool}": value for state, tensor in captures.items()
            for pool, value in {"mean": tensor.double().numpy()[mask].mean(axis=0),
                                "last": tensor[-1].double().numpy(),
                                "last_lexical": tensor[lexical].double().numpy()}.items()}
