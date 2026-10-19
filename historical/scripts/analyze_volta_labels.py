"""Nested CPU diagnostic of externally annotated implicit line9 turns."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from sonnet_geometry.corpus import load_corpus
from sonnet_geometry.editorial_diagnostic import nested_diagnostic, paired_stability


class VoltaFeatures:
    def __init__(self, records, indices, embeddings, activation, activation_names):
        self.records, self.indices = records, indices
        self.pairs = np.column_stack([indices, indices])
        self.activation, self.names = activation, activation_names
        self.cache = {}
        self.new, self.previous, self.prefix, vectors, elementary = [], [], [], [], []
        numbers, lines, x = embeddings["sonnet_numbers"], embeddings["line_numbers"], embeddings["line_embeddings"].astype(float)
        for index in indices:
            text = records[index]["text"].splitlines()
            assert len(text) == 14
            self.new.append(text[8]); self.previous.append(text[7]); self.prefix.append("\n".join(text[:8]))
            verse = x[numbers == index + 1][np.argsort(lines[numbers == index + 1])]
            assert len(verse) == 14
            old, new, prefix = verse[7], verse[8], verse[:8].mean(axis=0)
            def cos(a, b):
                return float(1 - a @ b / (np.linalg.norm(a)*np.linalg.norm(b)))
            vectors.append(np.concatenate([old, new, prefix]))
            markers = ("but", "yet", "and", "then", "thus", "so", "therefore", "however", "nor", "or", "if", "when")
            first = re.findall(r"\b\w+\b", text[8].lower())[0]
            elementary.append([cos(old, new), cos(prefix, new), len(text[7]), len(text[8]),
                               len(self.prefix[-1]), len(re.findall(r"\b\w+\b", text[8])),
                               float(text[7].rstrip().endswith((".", "!", "?", ";", ":"))),
                               (index + 1) / 154, *[float(first == marker) for marker in markers]])
        self.vectors, self.elementary = np.asarray(vectors), np.asarray(elementary)

    def get(self, train, version):
        key = tuple(sorted(int(i) for i in train))
        if key not in self.cache:
            fit_text = [s for i in key for s in (self.previous[i], self.new[i], self.prefix[i])]
            tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2,
                                    lowercase=True, sublinear_tf=True, dtype=np.float64).fit(fit_text)
            old, new, prefix = [tfidf.transform(t) for t in (self.previous, self.new, self.prefix)]
            for matrix in (old, new, prefix):
                if np.any(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel() == 0):
                    raise ValueError("Empty text representation in fold")
            self.cache[key] = np.column_stack([1 - np.asarray(old.multiply(new).sum(axis=1)).ravel(),
                                              1 - np.asarray(prefix.multiply(new).sum(axis=1)).ravel(), self.elementary])
        base = self.cache[key]
        if version in ("text_vectors", "text_vectors_plus_all_depths"):
            base = np.column_stack([base, self.vectors])
        if version in ("conventional", "text_vectors"):
            return base
        if version == "plus_input":
            keep = [i for i, name in enumerate(self.names) if "/0.0/" in name]
        elif version == "plus_final":
            keep = [i for i, name in enumerate(self.names) if "/0.0/" in name or "/1.0/" in name]
        elif version in ("plus_all_depths", "text_vectors_plus_all_depths"):
            keep = list(range(len(self.names)))
        else:
            raise ValueError(version)
        return np.column_stack([base, self.activation[:, keep]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edition-sensitivity", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = [Path(p) for p in ["datasets/normalized/sonnets.jsonl", "experiments/deep-study-v2/volta-labels.json",
               "experiments/deep-study-v2/editorial-links.json", "experiments/deep-study-v2/line-reconstruction.json",
               "outputs/prior-replica/line-embeddings.npz", "docs/deep-study-plan.md"]]
    records = load_corpus(sources[0])
    labels = json.loads(sources[1].read_text())
    editorial = json.loads(sources[2].read_text())
    profiles = json.loads(sources[3].read_text())
    rows = [{**r, "source_sonnet_id": r["sonnet_id"],
             "sonnet_id": f"sonnet-{int(str(r['sonnet_id']).split('-')[-1]):03d}"}
            for r in labels["records"] if r["diagnostic_include"]]
    exclusions = []
    if args.edition_sensitivity:
        collation = Path("experiments/deep-study-v2/volta-edition-collation.json")
        sources.append(collation)
        exclusions = json.loads(collation.read_text())["potential_substantive_variant_ids"]
        rows = [r for r in rows if int(r["sonnet_id"].split("-")[-1]) not in exclusions]
    indices = np.array([int(r["sonnet_id"].split("-")[-1]) - 1 for r in rows])
    y = np.array([r["target_shadow"] for r in rows], int)
    assert (len(y), y.sum()) == ((91, 25) if args.edition_sensitivity else (96, 27))
    columns, names = [], []
    for model in profiles["models"]:
        settings = model["poems"][rows[0]["sonnet_id"]]["settings"]
        for setting in settings:
            depth = setting["requested_depth"]
            for metric in setting["values"]:
                values = []
                for row in rows:
                    match, = [s for s in model["poems"][row["sonnet_id"]]["settings"] if s["requested_depth"] == depth]
                    values.append(match["values"][metric][7])  # transition line8 -> line9
                columns.append(values); names.append(f"{model['model_id']}/{depth}/{metric}")
    activation = np.asarray(columns).T
    assert activation.shape == (len(y), 45) and np.isfinite(activation).all()
    features = VoltaFeatures(records, indices, np.load(sources[4]), activation, names)
    components = {n - 1: g["component_id"] for g in editorial["fold_components"] for n in g["sonnet_numbers"]}
    components[152] = components[153] = "positive-control-153-154"
    versions = ("prevalence", "conventional", "text_vectors", "plus_input", "plus_final",
                "plus_all_depths", "text_vectors_plus_all_depths")
    bands = indices // 22
    print(json.dumps({"phase": "volta_nested", "poems": len(y), "shadow": int(y.sum()), "versions": versions}), flush=True)
    result = nested_diagnostic(features, y, bands, components, versions=versions)
    result["conditional_block_stability"] = paired_stability(y, bands, result["predictions"], reference="text_vectors")
    result.update({"schema_version": "external-volta-diagnostic-v2", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "source_files": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
                   "feature_names": names,
                   "edition_sensitivity_exclusions": exclusions,
                   "rows": [{"sonnet_id": r["sonnet_id"], "source_type": r["source_type"], "target_shadow": int(y[i]),
                             "band": int(bands[i])} for i, r in enumerate(rows)],
                   "inference": "Externally sourced but subjective, unreplicated author labels; exploratory internal held-out prediction, not confirmed volta detection",
                   "no_model_inference": True})
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with args.output.open("x") as handle:
        handle.write(encoded)
    print(json.dumps({"output": str(args.output), "metrics": result["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
