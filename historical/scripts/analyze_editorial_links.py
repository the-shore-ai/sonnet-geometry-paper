"""Run the frozen externally sourced editorial relationship diagnostic on CPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from sonnet_geometry.corpus import load_corpus
from sonnet_geometry.depth_discovery import load_model_rdms
from sonnet_geometry.editorial_diagnostic import Features, VERSIONS, nested_diagnostic, paired_stability


def digest(path):
    return {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extended-final-baselines", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    corpus_path = Path("datasets/normalized/sonnets.jsonl")
    labels_path = Path("experiments/deep-study-v2/editorial-links.json")
    baselines_path = Path("experiments/baselines.json")
    lock_path = Path("configs/baselines.lock.json")
    records = load_corpus(corpus_path)
    ids = [r["sonnet_id"] for r in records]
    texts = [r["text"] for r in records]
    labels = json.loads(labels_path.read_text())
    baseline = json.loads(baselines_path.read_text())
    assert baseline["sonnet_ids"] == ids
    config = json.loads(lock_path.read_text())
    models = {f: load_model_rdms(f"outputs/full-{f}-v1/report.json", ids) for f in ("qwen", "gemma", "muse")}
    versions = VERSIONS + (("plus_final_block", "plus_final_normalized", "plus_both_final",
                            "final_plus_fixed_depths", "final_plus_all_blocks", "plus_early_blocks")
                           if args.extended_final_baselines else ())
    positive = {tuple(e["sonnet_numbers"]) for e in labels["positive_edges"]}
    control = tuple(labels["positive_control"]["sonnet_numbers"])
    positive.add(control)
    ambiguous = {tuple(e["sonnet_numbers"]) for e in labels["excluded_relations"]}
    components = {n - 1: group["component_id"] for group in labels["fold_components"] for n in group["sonnet_numbers"]}
    components[152] = components[153] = "positive-control-153-154"
    assert len(components) == 154 and components[152] == components[153]

    all_matrices = {}
    for family, model in models.items():
        for pooling, matrices in model.distances.items():
            all_matrices[f"{family}/{pooling}/input"] = matrices[0]
            for depth in (.25, .5, .75, 1.):
                index = min(range(len(model.layers)), key=lambda i: abs(model.layers[i]["normalized_depth"] - depth))
                all_matrices[f"{family}/{pooling}/fixed/{depth}"] = matrices[index]
            for i, layer in enumerate(model.layers[1:], 1):
                all_matrices[f"{family}/{pooling}/block/{layer['tensor']}"] = matrices[i]
            if args.extended_final_baselines:
                all_matrices[f"{family}/{pooling}/final_normalized"] = model.final_normalized[pooling]
    feature_names = list(all_matrices)
    minilm = np.asarray(baseline["baselines"]["minilm_sentence_embedding"]["cosine_distance"])
    results = []
    for exclude154 in (False, True):
        pairs = np.array([(i, i + 1) for i in range(153)
                          if i // 22 == (i + 1) // 22 and (i + 1, i + 2) not in ambiguous
                          and not (exclude154 and i == 152)], dtype=int)
        y = np.array([int((i + 1, j + 1) in positive) for i, j in pairs])
        bands = pairs[:, 0] // 22
        values = np.column_stack([matrix[pairs[:, 0], pairs[:, 1]] for matrix in all_matrices.values()])
        features = Features(texts, pairs, minilm, values, feature_names, config["tfidf"])
        print(json.dumps({"phase": "nested_editorial", "exclude154": exclude154,
                          "pairs": len(pairs), "positives": int(y.sum())}), flush=True)
        result = nested_diagnostic(features, y, bands, components, versions=versions)
        result["conditional_block_stability"] = paired_stability(
            y, bands, result["predictions"], reference="plus_both_final" if args.extended_final_baselines else "plus_input")
        result["exclude154"] = exclude154
        result["rows"] = [{"row": r, "sonnets": [int(a + 1), int(b + 1)], "band": int(bands[r]),
                            "editorially_linked": int(y[r]), "positive_control": (a + 1, b + 1) == control}
                           for r, (a, b) in enumerate(pairs)]
        results.append(result)
        print(json.dumps({"exclude154": exclude154, "metrics": result["metrics"]}), flush=True)

    # All explicitly annotated pairs, compared only with pairs at the same sequence gap.
    # This is descriptive: source groups overlap and unmentioned controls are not negatives.
    rank_sources = dict(all_matrices)
    rank_sources.update({f"baseline/{name}": np.asarray(item["cosine_distance"])
                         for name, item in baseline["baselines"].items()})
    pair_ranks = []
    for a, b in sorted(positive):
        gap = b - a
        row = {"sonnets": [a, b], "gap": gap, "positive_control": (a, b) == control,
               "same_gap_comparison_pairs": 154 - gap, "distance_midrank_percentiles": {}}
        for name, matrix in rank_sources.items():
            distances = np.round(np.diag(matrix, k=gap), 12)
            # Half-rank convention: a unique minimum among m has percentile .5/m.
            row["distance_midrank_percentiles"][name] = float((rankdata(distances, method="average")[a - 1] - .5) / len(distances))
        pair_ranks.append(row)
    result = {"schema_version": "editorial-diagnostic-v2", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Post hoc internal prediction of externally annotated links, not independent literary validation",
              "source_files": [digest(p) for p in [corpus_path, labels_path, baselines_path, lock_path, Path("docs/deep-study-plan.md")]] + [m.source for m in models.values()],
              "feature_names": feature_names, "versions": results,
              "descriptive_same_gap_pair_ranks": pair_ranks,
              "limitations": ["Unmentioned adjacent pairs are not verified semantic negatives.",
                              "Seven chronological held-out units; uncertainty is conditional and low-resolution.",
                              "Source corpus and initial geometry were previously observed; all results exploratory.",
                              "TF-IDF/IDF, stylometric scaling and supervised standardization are refit in every training split.",
                              "Descriptive same-gap ranks use original full-corpus baseline distances; they are separate from predictive fits.",
                              "Model training exposure, BF16 shape effects and incomplete editorial annotations remain unresolved."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with args.output.open("x") as handle:
        handle.write(encoded)
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
