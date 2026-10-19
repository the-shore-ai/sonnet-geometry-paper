"""Versioned sonnet records and lossless token-to-source alignment.

All spans are half-open Unicode character offsets. Source offsets refer to the
UTF-8 decoded *unmodified* raw file (including CRLF); byte offsets are separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


SCHEMA_VERSION = "sonnets-v1"
SOURCE_URL = "https://www.gutenberg.org/cache/epub/1041/pg1041.txt"
STRUCTURE_SOURCE = "https://www.folger.edu/explore/shakespeares-works/shakespeares-sonnets/read/"


def _roman_number(value: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    numbers = [values[c] for c in value]
    return sum(-n if i + 1 < len(numbers) and n < numbers[i + 1] else n
               for i, n in enumerate(numbers))


def _stanzas(number: int, line_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sizes = [5, 4, 4, 2] if number == 99 else ([2] * 6 if number == 126 else [4, 4, 4, 2])
    result, cursor = [], 0
    for i, size in enumerate(sizes, 1):
        selected = line_spans[cursor:cursor + size]
        kind = {2: "couplet", 4: "quatrain", 5: "quintain"}[size]
        result.append({"stanza_number": i, "kind": kind,
                       "line_numbers": [line["line_number"] for line in selected],
                       "start": selected[0]["start"], "end": selected[-1]["end"],
                       "annotation_type": "editorial_structure",
                       "source_url": STRUCTURE_SOURCE})
        cursor += size
    return result


def parse_gutenberg(raw: bytes, *, retrieved_at: str | None = None) -> list[dict[str, Any]]:
    """Parse the pinned #1041 layout, fail closed on absent/extra poem lines.

    Only the wrapper, headings, empty rows and outer indentation are removed;
    spelling, punctuation and internal spaces are untouched. CRLF becomes LF.
    """
    source = raw.decode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    active = False
    current: dict[str, Any] | None = None
    records = []
    char_offset = byte_offset = 0
    for source_line_number, raw_line in enumerate(source.splitlines(keepends=True), 1):
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()
        if stripped.startswith("*** END OF THE PROJECT GUTENBERG EBOOK"):
            break
        if stripped.startswith("*** START OF THE PROJECT GUTENBERG EBOOK"):
            active = True
        elif active and re.fullmatch(r"[IVXLC]+", stripped):
            number = _roman_number(stripped)
            current = {"schema_version": SCHEMA_VERSION, "sonnet_id": f"sonnet-{number:03d}",
                       "number": number, "text": "", "lines": [], "line_spans": [],
                       "provenance": {"source_url": SOURCE_URL, "source_sha256": digest,
                                      "source_heading": stripped, "source_heading_line": source_line_number,
                                      "edition": "Project Gutenberg ebook 1041; updated 2024-10-29",
                                      "retrieved_at": retrieved_at,
                                      "normalization": "outer-line-whitespace-trim; CRLF-to-LF; no-final-LF"}}
            records.append(current)
        elif active and current is not None and stripped:
            leading = len(line) - len(line.lstrip())
            start = len(current["text"]) + bool(current["lines"])
            span = {"line_number": len(current["lines"]) + 1,
                    "start": start, "end": start + len(stripped),
                    "source_line_number": source_line_number,
                    "source_start": char_offset + leading,
                    "source_end": char_offset + leading + len(stripped),
                    "source_byte_start": byte_offset + len(line[:leading].encode("utf-8")),
                    "source_byte_end": byte_offset + len(line[:leading + len(stripped)].encode("utf-8"))}
            current["lines"].append(stripped)
            current["line_spans"].append(span)
            current["text"] = "\n".join(current["lines"])
        char_offset += len(raw_line)
        byte_offset += len(raw_line.encode("utf-8"))
    if [r["number"] for r in records] != list(range(1, 155)):
        raise ValueError("Expected exactly sonnets 1–154 in order")
    for record in records:
        expected = {99: 15, 126: 12}.get(record["number"], 14)
        if len(record["lines"]) != expected:
            raise ValueError(f"Sonnet {record['number']}: expected {expected} lines, found {len(record['lines'])}")
        record["stanza_spans"] = _stanzas(record["number"], record["line_spans"])
        record["text_sha256"] = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
        record["scholarly_annotations"] = {
            "meter": "iambic tetrameter" if record["number"] == 145 else "iambic pentameter (conventional; not line-level scansion)",
            "meter_source_url": STRUCTURE_SOURCE,
            "rhyme_scheme": "ABABA CDCD EFEF GG" if record["number"] == 99 else
                            ("AA BB CC DD EE FF" if record["number"] == 126 else "ABAB CDCD EFEF GG"),
            "rhyme_status": "conventional structural template; not phonologically verified",
            "themes": [], "volta_lines": [],
        }
        record["model_annotations"] = []
    return records


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL without implicitly changing the saved records."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def align_tokens(record: dict[str, Any], offsets: Iterable[Iterable[int]],
                 input_ids: Iterable[int] | None = None,
                 special_tokens_mask: Iterable[int] | None = None) -> list[dict[str, Any]]:
    """Keep every token, overlapping line/stanza memberships, and separator tokens.

    Token IDs and special-token flags come from the tokenizer, not heuristics.
    Unmapped zero-length offsets are explicit; no invented line is assigned.
    """
    offsets = [tuple(map(int, pair)) for pair in offsets]
    ids = [None] * len(offsets) if input_ids is None else list(input_ids)
    specials = [0] * len(offsets) if special_tokens_mask is None else list(special_tokens_mask)
    if len(ids) != len(offsets) or len(specials) != len(offsets):
        raise ValueError("Offsets, IDs and special mask lengths must match")
    result = []
    for i, ((start, end), token_id, special) in enumerate(zip(offsets, ids, specials)):
        if not 0 <= start <= end <= len(record["text"]):
            raise ValueError(f"Token {i} has invalid character offsets")
        lines = [] if special or start == end else [s for s in record["line_spans"] if start < s["end"] and end > s["start"]]
        stanzas = [] if special or start == end else [s for s in record["stanza_spans"] if start < s["end"] and end > s["start"]]
        pieces = []
        for line in lines:
            left, right = max(start, line["start"]), min(end, line["end"])
            pieces.append({"line_number": line["line_number"],
                           "line_char_start": left - line["start"], "line_char_end": right - line["start"],
                           "source_start": line["source_start"] + left - line["start"],
                           "source_end": line["source_start"] + right - line["start"]})
        result.append({"sonnet_id": record["sonnet_id"], "token_index": i,
                       "input_id": None if token_id is None else int(token_id),
                       "char_start": start, "char_end": end,
                       "text_slice": record["text"][start:end], "special": bool(special),
                       "line_numbers": [s["line_number"] for s in lines],
                       "stanza_numbers": [s["stanza_number"] for s in stanzas],
                       "source_pieces": pieces, "unmapped_offset": start == end,
                       "separator_only": not special and end > start and not lines,
                       "crosses_line_boundary": len(lines) > 1 or (bool(lines) and "\n" in record["text"][start:end])})
    return result


def tokenize_with_alignment(tokenizer: Any, record: dict[str, Any]) -> dict[str, Any]:
    """Encode once, audit all independently tokenized line prefixes exactly.

    A prefix is reusable only if its *entire* token-ID sequence equals the
    leading sequence of the full encoding, and no included token crosses its
    character boundary. EOS and boundary-sensitive merges can invalidate this.
    Primary boundaries include the following LF (except the final line); this
    is an explicit operational definition. Content-end checks are also saved.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer with character offsets is required")
    encoded = tokenizer(record["text"], return_offsets_mapping=True, return_special_tokens_mask=True,
                        add_special_tokens=True, truncation=False)
    ids = list(encoded["input_ids"])
    offsets = [list(pair) for pair in encoded["offset_mapping"]]
    specials = list(encoded["special_tokens_mask"])
    tokens = align_tokens(record, offsets, ids, specials)
    def prefix_check(span: dict[str, Any], *, include_separator: bool) -> dict[str, Any]:
        char_end = span["end"]
        if include_separator and record["text"][char_end:char_end + 1] == "\n":
            char_end += 1
        prefix_ids = list(tokenizer(record["text"][:char_end], add_special_tokens=True,
                                   truncation=False)["input_ids"])
        equivalent = ids[:len(prefix_ids)] == prefix_ids and all(
            special or end <= char_end for (_, end), special in zip(offsets[:len(prefix_ids)], specials))
        return {"line_number": span["line_number"], "char_end": char_end,
                "line_content_end": span["end"],
                "boundary_policy": "include_following_line_separator" if include_separator else "line_content_end",
                "prefix_input_ids": prefix_ids, "prefix_token_count": len(prefix_ids),
                "exact_token_prefix": equivalent,
                "full_pass_token_end": len(prefix_ids) if equivalent else None}
    checks = [prefix_check(span, include_separator=True) for span in record["line_spans"]]
    content_checks = [prefix_check(span, include_separator=False) for span in record["line_spans"]]
    return {"input_ids": ids, "attention_mask": list(encoded.get("attention_mask", [1] * len(ids))),
            "offset_mapping": offsets, "special_tokens_mask": specials,
            "tokens": tokens, "prefix_checks": checks, "prefix_checks_content_end": content_checks,
            "boundary_policy": "include_following_line_separator"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("datasets/raw/pg1041.txt"))
    parser.add_argument("--output", type=Path, default=Path("datasets/normalized/sonnets.jsonl"))
    parser.add_argument("--retrieved-at", default=None,
                        help="Actual retrieval timestamp; unspecified remains null")
    args = parser.parse_args()
    records = parse_gutenberg(args.raw.read_bytes(), retrieved_at=args.retrieved_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records), encoding="utf-8")
    print(json.dumps({"records": len(records), "output": str(args.output),
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
