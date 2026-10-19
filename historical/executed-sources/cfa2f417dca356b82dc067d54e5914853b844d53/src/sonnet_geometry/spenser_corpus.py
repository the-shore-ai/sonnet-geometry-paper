"""Outcome-independent Amoretti corpus from the Spenser Archive's copy-text view.

Only verse divs enter the corpus. Original readings are retained, regularized
alternatives and line-number apparatus are excluded. The single 1595/1611
apparatus choice retains the explicitly labeled 1595 reading, including its typo.
"""
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re

from .confirmation_corpus import select_population, sha256
from .storage import write_exclusive

NAMESPACE = "confirmation-v5-spenser"
DIRECTORY = Path("datasets") / NAMESPACE
SOURCE_URL = "https://talus.artsci.wustl.edu/spenserArchivePrototype/html/am_ep_amoretti.html"
RAW_PATH = DIRECTORY / "raw/am_ep_amoretti.html"
ALIGNMENT_PATH = DIRECTORY / "alignment-source.txt"


class VerseParser(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=False)
        self.source, self.rows, self.stack, self.current = source, {}, [], None
        self.starts, offset = [], 0
        for line in source.splitlines(keepends=True):
            self.starts.append(offset)
            offset += len(line)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set(attrs.get("class", "").split())
        if self.current is None:
            if tag != "div" or "l" not in classes:
                return
            match = re.fullmatch(r"am_ep\.sonnets\.(\d+)\.(\d+)", attrs.get("linenumber", ""))
            if not match:
                raise ValueError("Unexpected verse identity")
            identity = tuple(map(int, match.groups()))
            if identity in self.rows:
                raise ValueError("Duplicate verse identity")
            self.current = {"identity": identity, "parts": [], "raw_source_segments": []}
            self.stack = [("div", False)]
            return
        if tag != "span":
            raise ValueError("Unexpected nested verse element")
        known = {"orig", "reg", "hi", "lineNumber", "app", "lem", "rdg"}
        if classes and not classes.intersection(known):
            raise ValueError("Unknown verse span semantics")
        if "rdg" in classes and "#E1595" not in classes:
            raise ValueError("Unknown apparatus witness")
        suppressed = self.stack[-1][1] or bool(classes.intersection({"reg", "lem", "lineNumber"}))
        self.stack.append((tag, suppressed))

    def handle_endtag(self, tag):
        if self.current is None:
            return
        if not self.stack or self.stack[-1][0] != tag:
            raise ValueError("Unbalanced verse markup")
        self.stack.pop()
        if not self.stack:
            row = self.current
            row["text"] = " ".join("".join(row.pop("parts")).split())
            if not row["text"]:
                raise ValueError("Empty verse line")
            self.rows[row["identity"]] = row
            self.current = None

    def append(self, raw, decoded):
        if self.current is None or self.stack[-1][1]:
            return
        line, column = self.getpos()
        start = self.starts[line - 1] + column
        end = start + len(raw)
        if self.source[start:end] != raw:
            raise ValueError("HTML source offset mismatch")
        self.current["parts"].append(decoded)
        self.current["raw_source_segments"].append({
            "source_start": start, "source_end": end,
            "source_byte_start": len(self.source[:start].encode()),
            "source_byte_end": len(self.source[:end].encode())})

    def handle_data(self, data):
        self.append(data, data)

    def handle_entityref(self, name):
        raw = f"&{name};"
        self.append(raw, unescape(raw))

    def handle_charref(self, name):
        raw = f"&#{name};"
        self.append(raw, unescape(raw))


def parse_spenser(raw, retrieved_at):
    source = raw.decode("utf-8")
    parser = VerseParser(source)
    parser.feed(source)
    parser.close()
    expected = {(n, line) for n in range(1, 90) for line in range(1, 15)}
    if parser.current is not None or set(parser.rows) != expected:
        raise ValueError("Expected exactly 89 sonnets, each with 14 distinct verse lines")
    records, parts, cursor, byte_cursor = [], [], 0, 0
    for n in range(1, 90):
        lines, spans, poem_cursor, poem_bytes = [], [], 0, 0
        for line in range(1, 15):
            row = parser.rows[n, line]
            value = row["text"]
            spans.append({"line_number": line, "start": poem_cursor, "end": poem_cursor + len(value),
                          "source_start": cursor + poem_cursor, "source_end": cursor + poem_cursor + len(value),
                          "source_byte_start": byte_cursor + poem_bytes,
                          "source_byte_end": byte_cursor + poem_bytes + len(value.encode()),
                          "source_path": str(ALIGNMENT_PATH), "source_anchor": f"am_ep.sonnets.{n}.{line}",
                          "raw_source_segments": row["raw_source_segments"]})
            lines.append(value)
            poem_cursor += len(value) + 1
            poem_bytes += len(value.encode()) + 1
        text = "\n".join(lines)
        identity = f"spenser-archive1595-{n:03d}"
        records.append({"schema_version": "spenser-confirmation-corpus-v1", "sonnet_id": identity,
                        "poem_id": identity, "number": n, "edition_position": n,
                        "numbering_basis": "Spenser Archive am_ep.sonnets identifier; original 1595 sequence",
                        "author": "Edmund Spenser", "cycle": "Amoretti", "text": text, "lines": lines,
                        "line_spans": spans, "stanza_spans": [], "text_sha256": sha256(text.encode()),
                        "provenance": {"source_url": SOURCE_URL, "raw_source_path": str(RAW_PATH),
                                       "raw_source_sha256": sha256(raw), "alignment_source_path": str(ALIGNMENT_PATH),
                                       "retrieved_at": retrieved_at,
                                       "edition": "Spenser Archive Prototype; 1595 original readings, not regularized or Oxford view",
                                       "normalization": "HTML entity decoding; original spans retained, reg/lineNumber/1611 lem excluded; 1595 rdg retained; collapse whitespace within verse lines; LF joins",
                                       "raw_segment_semantics": "Concatenate decoded raw fragments then collapse whitespace to reconstruct a line; generic offsets refer to alignment-source.txt"},
                        "scholarly_annotations": {}, "model_annotations": []})
        parts.append(text)
        cursor += len(text) + 2
        byte_cursor += len(text.encode()) + 2
    alignment = "\n\n".join(parts)
    for record in records:
        record["provenance"]["source_sha256"] = sha256(alignment.encode())
    return records, alignment


def write_corpus(retrieved_at):
    raw = RAW_PATH.read_bytes()
    records, alignment = parse_spenser(raw, retrieved_at)
    selected, inventory = select_population(records)
    files = {"all-sonnets.jsonl": records, "selected-sonnets.jsonl": selected}
    write_exclusive(ALIGNMENT_PATH, alignment.encode())
    for name, rows in files.items():
        write_exclusive(DIRECTORY / name, ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)).encode())
    paths = [ALIGNMENT_PATH, *(DIRECTORY / name for name in files)]
    result = {"schema_version": "spenser-confirmation-selection-v1", "seed": 20261018,
              "selection_rule": "ascending SHA256(decimal seed + LF + stable ID); first 24; serialize in original order",
              "source_url": SOURCE_URL, "raw_sha256": sha256(raw), "model_outputs_used": False,
              "eligible_population_size": 89, "selected_count": 24, "selected_ids": [r["sonnet_id"] for r in selected],
              "selected_numbers": [r["number"] for r in selected], "inventory": inventory,
              "excluded_poems": [], "files": {str(p): {"sha256": sha256(p.read_bytes()), "bytes": p.stat().st_size} for p in paths},
              "limits": ["One Elizabethan sonnet cycle; training exposure unknown", "Author, edition and population differences remain confounded", "Archive copy-text readings include original spelling and the 1595 tbou reading at 68:6; no spelling repair"]}
    write_exclusive(DIRECTORY / "selection.json", (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
    return result
