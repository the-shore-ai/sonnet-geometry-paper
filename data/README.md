# Compact verification dataset

Approximately 54 MB (decimal, before packaging). All values are taken from
existing saved results; none is simulated or downcast for this release.

- `identity-final-means.npz`: lossless float64 pooled vectors from FP32 forwards,
  all 24 poems in each of Sidney and Spenser, all three models, original and
  five token shuffles at final-raw mean. Array names bind population/model/condition.
- `frozen-rdms.npz` and `frozen-rdm-index.json`: all 2,880 original frozen-study
  cosine RDM upper triangles, in NumPy `triu_indices(24, 1)` order; retain both
  precisions, five states, three poolings and all 32 conditions. Distances are
  stored unchanged; the verifier rounds to Q12 only before ranking.
- `frozen-claims.json`: original criterion definitions and summaries, with the
  24 edition positions per population. Pairwise edges and seeds are dependent.
- `response-displacements.json`: every retained per-poem scalar cosine and norm
  response and token-change fraction; not raw activation vectors.
- `selected-reference-results.json`: selected numerical results and complete
  declared search scope/failed-lead descriptions.
- `complete-exploratory-summary.json`: the full v6 numerical summary, retaining
  weak and undefined cells; only private location/code receipts were omitted.
- `manifest.json`: file sizes and SHA256 hashes. NPZ files contain numerical
  arrays only; load with `allow_pickle=False`.

Poem numbers are positions in the specified editions. No attribution claim is
implied. The source recipe and selected text hashes are in `datasets/`.

The compact verifier checks 246 numerical/decision quantities from these data,
including paired structural sensitivity ranges. The historical 33,552-check
private audit has a different scope. No human literary responses, complete
historical predictive-feature dataset, model weights or full token-level
activation tensors are included. This package does not permit verification of
every historical extraction from raw tensors without regenerating them.
