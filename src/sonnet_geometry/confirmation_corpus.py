"""Freeze an outcome-independent Sidney population for the v4 confirmation.

This parses the pinned Gutenberg #56375 transcription of the Newman 1591 text.
IDs and ``number`` refer to its unnumbered main sequence's edition position,
not to the differently constituted, conventionally numbered 1598 sequence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


SOURCE_URL = "https://www.gutenberg.org/cache/epub/56375/pg56375.txt"
START = "SIR P. S. HIS ASTROPHEL AND _STELLA_."
END = "_Other Sonnets of variable verse._"
SELECTION_SEED = 20261018
SAMPLE_SIZE = 24
SCHEMA = "sidney-confirmation-corpus-v1"
ALIGNMENT_SOURCE = "datasets/confirmation-v4/alignment-source.txt"
RAW_SOURCE = "datasets/confirmation-v4/raw/pg56375.txt"


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_rows(source: str) -> list[dict[str, Any]]:
    rows, char_offset, byte_offset = [], 0, 0
    for number, raw_line in enumerate(source.splitlines(keepends=True), 1):
        content = raw_line.rstrip("\r\n")
        rows.append({"raw": content, "line_number": number,
                     "char_offset": char_offset, "byte_offset": byte_offset})
        char_offset += len(raw_line)
        byte_offset += len(raw_line.encode("utf-8"))
    return rows


def _plain_line(row: dict[str, Any]) -> tuple[str, list[dict[str, int]], list[int]]:
    """Remove only outer space and paired underscore emphasis delimiters."""
    raw = row["raw"]
    left = len(raw) - len(raw.lstrip())
    right = len(raw.rstrip())
    if raw[left:right].count("_") % 2:
        raise ValueError("Unpaired emphasis delimiter in verse line")
    chunks = list(re.finditer(r"[^_]+", raw[left:right]))
    pieces, delimiter_offsets, normalized_offset = [], [], 0
    for match in chunks:
        a, b = left + match.start(), left + match.end()
        value = raw[a:b]
        pieces.append({"line_char_start": normalized_offset,
                       "line_char_end": normalized_offset + len(value),
                       "source_start": row["char_offset"] + a,
                       "source_end": row["char_offset"] + b,
                       "source_byte_start": row["byte_offset"] + len(raw[:a].encode("utf-8")),
                       "source_byte_end": row["byte_offset"] + len(raw[:b].encode("utf-8"))})
        normalized_offset += len(value)
    delimiter_offsets = [row["char_offset"] + i for i in range(left, right) if raw[i] == "_"]
    return raw[left:right].replace("_", ""), pieces, delimiter_offsets


def parse_sidney(raw: bytes, *, retrieved_at: str | None = None) -> tuple[list[dict], str, list[dict]]:
    """Return all 107 main sonnets, alignment source, and excluded song inventory.

    The separate alignment source is plain verse joined by LF, with a blank
    line between poems. Generic source offsets refer to that named derivative;
    raw_source_segments provide the exact mapping back to unchanged PG bytes.
    """
    source = raw.decode("utf-8")
    rows = _source_rows(source)
    starts = [i for i, row in enumerate(rows) if row["raw"].strip() == START]
    ends = [i for i, row in enumerate(rows) if row["raw"].strip() == END]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError("Expected one ordered main-sequence boundary pair")
    blocks, block = [], []
    for row in rows[starts[0] + 1:ends[0]]:
        line = row["raw"].strip()
        if not line:
            if block:
                blocks.append(block)
                block = []
        elif line == "[Illustration]":
            if block:
                raise ValueError("Illustration inside verse block")
        else:
            block.append(row)
    if block:
        blocks.append(block)
    if len(blocks) != 107:
        raise ValueError(f"Expected 107 main sonnets; found {len(blocks)}")
    for position, block in enumerate(blocks, 1):
        if len(block) != 14:
            raise ValueError(f"Edition position {position}: expected 14 lines; found {len(block)}")

    records, source_parts, source_cursor, byte_cursor = [], [], 0, 0
    for position, block in enumerate(blocks, 1):
        lines, spans = [], []
        poem_cursor = 0
        for row in block:
            line, pieces, removed = _plain_line(row)
            spans.append({"line_number": len(lines) + 1,
                          "start": poem_cursor, "end": poem_cursor + len(line),
                          "source_start": source_cursor + poem_cursor,
                          "source_end": source_cursor + poem_cursor + len(line),
                          "source_byte_start": byte_cursor + len("\n".join(lines).encode("utf-8")) + bool(lines),
                          "source_byte_end": byte_cursor + len("\n".join(lines).encode("utf-8")) + bool(lines)
                                             + len(line.encode("utf-8")),
                          "source_path": ALIGNMENT_SOURCE,
                          "source_line_number": (position - 1) * 15 + len(lines) + 1,
                          "raw_source_path": RAW_SOURCE,
                          "raw_source_line_number": row["line_number"],
                          "raw_source_segments": pieces,
                          "removed_emphasis_source_offsets": removed})
            lines.append(line)
            poem_cursor += len(line) + 1
        text = "\n".join(lines)
        records.append({"schema_version": SCHEMA,
                        "sonnet_id": f"sidney-newman1591-{position:03d}",
                        "poem_id": f"sidney-newman1591-{position:03d}",
                        "number": position, "edition_position": position,
                        "numbering_basis": "ordinal position in unnumbered 1591 main sequence",
                        "canonical_1598_number": None,
                        "author": "Philip Sidney", "cycle": "Astrophel and Stella",
                        "text": text, "lines": lines, "line_spans": spans,
                        "stanza_spans": [],
                        "text_sha256": sha256(text.encode("utf-8")),
                        "provenance": {"source_url": SOURCE_URL,
                                       "raw_source_path": RAW_SOURCE,
                                       "raw_source_sha256": sha256(raw),
                                       "alignment_source_path": ALIGNMENT_SOURCE,
                                       "source_offset_basis": "alignment-source.txt; raw_source_segments map to PG bytes",
                                       "edition": "Project Gutenberg #56375; title page: Thomas Newman, London, 1591",
                                       "gutenberg_updated": "2023-03-12",
                                       "retrieved_at": retrieved_at,
                                       "normalization": "trim outer line whitespace; remove underscore emphasis delimiters; LF verse joins; no final LF",
                                       "source_transcriber_corrections": "PG preface and final note retained in raw source; no further spelling or punctuation corrections"},
                        "scholarly_annotations": {}, "model_annotations": []})
        source_parts.append(text)
        source_cursor += len(text) + 2
        byte_cursor += len(text.encode("utf-8")) + 2
    alignment = "\n\n".join(source_parts)
    for record in records:
        record["provenance"]["source_sha256"] = sha256(alignment.encode("utf-8"))

    finish = [i for i, row in enumerate(rows) if row["raw"].strip() == "Finis Syr P. S."]
    if len(finish) != 1 or finish[0] <= ends[0]:
        raise ValueError("Expected one final variable-verse terminator")
    excluded, current = [], None
    for row in rows[ends[0] + 1:]:
        line = row["raw"].strip()
        if line == "Finis Syr P. S.":
            break
        if re.fullmatch(r"_(?:The |the )?(?:First|Second|third|fourth|fifth|Sixth|Seventh|eighth|ninth|Tenth) Sonnet\._", line):
            current = {"poem_id": f"sidney-newman1591-variable-verse-{len(excluded) + 1:02d}",
                       "source_heading": line, "source_heading_line": row["line_number"],
                       "section": "Other Sonnets of variable verse", "eligible": False,
                       "exclusion_reason": "outside fixed-form main sonnet sequence (song/variable-verse section)",
                       "verse_line_count": 0, "source_first_verse_line": None, "source_last_verse_line": None}
            excluded.append(current)
        elif line and line != "[Illustration]":
            if current is None:
                raise ValueError("Unexpected content before first variable-verse heading")
            current["verse_line_count"] += 1
            if current["source_first_verse_line"] is None:
                current["source_first_verse_line"] = row["line_number"]
            current["source_last_verse_line"] = row["line_number"]
    if len(excluded) != 10 or any(not r["verse_line_count"] for r in excluded):
        raise ValueError("Expected exactly ten nonempty variable-verse works")
    return records, alignment, excluded


def select_population(records: list[dict], *, seed: int = SELECTION_SEED,
                      count: int = SAMPLE_SIZE) -> tuple[list[dict], list[dict]]:
    """Rank stable IDs by SHA-256(seed + LF + ID), then return corpus order."""
    ids = [r["sonnet_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Population IDs must be unique")
    if not 1 <= count <= len(records):
        raise ValueError("Sample size must be within population")
    ranked = sorted(records, key=lambda r: (sha256(f"{seed}\n{r['sonnet_id']}".encode()), r["sonnet_id"]))
    selected_ids = {r["sonnet_id"] for r in ranked[:count]}
    selection_rank = {r["sonnet_id"]: i + 1 for i, r in enumerate(ranked)}
    inventory = [{"poem_id": r["sonnet_id"], "edition_position": r["number"],
                  "text_sha256": r["text_sha256"], "line_count": len(r["lines"]),
                  "eligible": True, "eligibility_reason": "main sequence; exactly fourteen source verse lines",
                  "selected": r["sonnet_id"] in selected_ids,
                  "selection_rank": selection_rank[r["sonnet_id"]],
                  "selection_sha256": sha256(f"{seed}\n{r['sonnet_id']}".encode())}
                 for r in sorted(records, key=lambda r: r["number"])]
    return [r for r in sorted(records, key=lambda r: r["number"]) if r["sonnet_id"] in selected_ids], inventory


def write_corpus(directory: Path, *, retrieved_at: str) -> dict:
    raw = (directory / "raw/pg56375.txt").read_bytes()
    records, alignment, excluded = parse_sidney(raw, retrieved_at=retrieved_at)
    selected, inventory = select_population(records)
    outputs = {"alignment-source.txt": alignment,
               "all-sonnets.jsonl": "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records),
               "selected-sonnets.jsonl": "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in selected)}
    for name, content in outputs.items():
        (directory / name).write_text(content, encoding="utf-8")
    selection = {"schema_version": "sidney-confirmation-selection-v1", "seed": SELECTION_SEED,
                 "selection_rule": "ascending SHA256(UTF8(decimal seed + LF + stable poem ID)), tie-break stable ID; first 24; serialize in edition order",
                 "selection_inputs": ["fixed seed", "stable edition-position ID"],
                 "model_outputs_used": False, "eligible_population_size": len(records),
                 "selected_count": len(selected), "selected_ids": [r["sonnet_id"] for r in selected],
                 "selected_edition_positions": [r["number"] for r in selected],
                 "inventory": inventory, "excluded_variable_verse": excluded,
                 "source_limitations": ["107-sonnet 1591 edition, not the differently constituted 108-sonnet 1598 sequence",
                                        "New to this study; model pretraining exposure is unknown",
                                        "One author and one historical transcription; not representative of all poetry",
                                        "PG has documented editorial corrections; not a diplomatic facsimile"],
                 "raw_sha256": sha256(raw),
                 "files": {name: {"sha256": sha256((directory / name).read_bytes()),
                                  "bytes": (directory / name).stat().st_size} for name in outputs},
                 "retrieved_at": retrieved_at}
    (directory / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("datasets/confirmation-v4"))
    parser.add_argument("--retrieved-at", required=True)
    args = parser.parse_args()
    result = write_corpus(args.directory, retrieved_at=args.retrieved_at)
    print(json.dumps({k: result[k] for k in ("eligible_population_size", "selected_count", "selected_edition_positions")}))


if __name__ == "__main__":
    main()
