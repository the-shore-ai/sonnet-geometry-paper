# Reproduction levels and recipes

## 1. Verify the compact published results

Follow the three commands in README.md. The receipt reports exactly 246
numerical/decision comparisons and seven data-file hash checks. It recomputes
both directions of other-author identity calibration (including Q12 tie credit),
centered and raw relational stability, the twelve frozen criterion decisions,
line/token contrasts and omission ranges, and the six selected response
components. Regenerated charts intentionally use a simple independent layout.
The paper's original figure assets are also included. This level uses only
NumPy, SciPy and Matplotlib, one CPU, and the released data.

These checks are not independent extraction or human literary validation. The
complete summary reports additional selected metrics that this compact verifier
does not independently recalculate, including some nuisance fits and permutation
references. All cells remain explicitly exploratory. Historical literary-task
features and complete raw tensors are not in this compact package.

## 2. Regenerate source inputs

Use Python 3.12. For the extraction environments, install `uv`, then either:

```sh
uv sync --project environments --no-install-project --python 3.12 --frozen --extra dev
```

for Qwen (Transformers 4.57.1), or:

```sh
uv sync --project environments/infra/modern_runtime --python 3.12 --frozen --extra dev
```

for Gemma/Muse (Transformers 5.20.1). Both pin PyTorch 2.9.1. The two environments
are intentionally separate. These are the original dependency locks, not
substituted current versions. Use the selected environment's Python in the
following commands, from the repository root:

```sh
PYTHONPATH=src environments/.venv/bin/python recipes/prepare_corpora.py
```

The recipe downloads the unchanged credited Gutenberg 1041/56375 and Spenser
Archive original-reading sources, verifies raw SHA256 checksums, uses the exact
published parsing/selection code, and verifies every selected poem's input-text
hash. It retains source wrappers/terms in `outputs/corpora/`. If a remote source
has changed, it fails instead of silently substituting another edition. You can
supply exact previously downloaded sources using `--source-dir DIRECTORY`.
Source terms remain distinct from this release's software/data licenses.

## 3. Repeat model extraction using your own GPU

First inspect a plan (no model or cloud work):

```sh
PYTHONPATH=src environments/.venv/bin/python recipes/capture.py \
  --population sidney --model qwen --output outputs/captures/sidney/qwen
```

One-case capture requires explicit execution:

```sh
PYTHONPATH=src environments/.venv/bin/python recipes/capture.py \
  --population sidney --model qwen --output outputs/captures/sidney/qwen-pilot \
  --execute --max-cases 1 --retain-token-states
```

For a complete new capture, use a fresh output directory and `--execute --all-cases`;
repeat for Sidney and Spenser and all three models, using the modern environment's
Python for Gemma/Muse. Each complete population/model run contains 768 cases:
24 poems, 16 original/perturbation conditions, two arithmetic precisions. FP32
promotes the exact loaded BF16 checkpoint values, with TF32 disabled. The source
kernels retain raw input/early/depth-.75/final-raw/final-normalized states and
float64 nonspecial mean, final token and final lexical token pooling. Add
`--retain-token-states` for native BF16/FP32 token captures; otherwise save the
pooled results and actual runtime report. No truncation or generation is used.

For original Shakespeare extraction, choose `--population shakespeare` with the
same wrapper; this switches to the original all-native-depth extraction kernel,
including sampled Q/K/V/attention/MLP hooks, without padding. There are 154 poems.
The 12-poem v3 controls and historical CPU study scripts are preserved under
`historical/`; their fixed source paths and report schemas are historical and
may require adaptation to a new local capture layout. They are not advertised
as a fully automated public rerun of every old stage.

The public local wrapper is new. Byte-exact historical scientific kernels and
scheduler snapshots are identified by SHA256 in
`provenance/original-source-manifest.json`. Exact source files from nine recorded execution commits are separately listed in `provenance/executed-source-manifest.json`; the former manifest describes the reviewed baseline files, which can include later fixes. The wrapper substitutes local files
and the caller's existing CUDA device for private Modal/GCS orchestration. It
makes no cloud submissions and cannot use our private accounts. Capturing all
conditions is computational work charged to the caller's own resources.

## 4. Analyze complete newly generated pooled captures

With complete runs at `outputs/captures/{sidney,spenser}/{qwen,gemma,muse}/`,
rebuild the compact numerical inputs and compare with the published expectations:

```sh
PYTHONPATH=src environments/.venv/bin/python recipes/pack_capture.py \
  --runs-root outputs/captures --output outputs/regenerated-data
.venv/bin/python recipes/verify_release.py \
  --data-dir outputs/regenerated-data --output outputs/regenerated-comparison.json
```

Use an environment with safetensors for packing. Pooled vectors, all RDMs and
response magnitudes are recalculated. Reference-result files retain the original
published expectations; they are not reclassified as newly observed outcomes.
Token-change fractions are carried from the registered input design and are not
new model measurements. The comparison fails on a numerical discrepancy rather
than silently changing a threshold. This packing path was checked against the
existing six complete pooled files; model execution with the new public wrapper
has not been run. The full inference-to-paper route still needs external replication.

## Hardware, numerical and provenance limits

Original execution used a single H200 with 141 GB nominal memory. BF16 weights,
FP32 promotion, working memory and optional captures all require headroom;
Gemma/Muse are large. Historical original-corpus function times were approximately
10/36/19 minutes for Qwen/Gemma/Muse, excluding downloads and some startup.
The USD 42.97 cumulative conservative accounting includes all historical failed,
partial and successful attempts and is not an invoice or regeneration quote.
Storage, transfer, prices and runtime vary; measure a small local pilot first.

Q12 rounding defines metric ties; it is not a promise of bitwise equal forward
results across devices or libraries. BF16 cross-shape checks failed historically,
and Gemma's deeper BF16/FP32 proxies repeatedly failed the declared arithmetic
gate. The paired FP32 primary results remain separate. Do not loosen thresholds
to make a rerun agree. Record hardware, packages, actual dtypes, input/output
hashes and failures in a new run; do not overwrite historical evidence.

No paid model run was performed to validate the new public scheduling wrapper.
Source regeneration, CPU verification and engineering checks are recorded
separately in the release validation receipt. No external end-to-end inference
replication is claimed.

## Render the paper

With Pandoc, its citeproc support, XeLaTeX and the standard LaTeX packages
installed, run `sh recipes/build_paper.sh`. The PDF source, bibliography and all
six included figure assets are local. This recreates the document content;
PDF build timestamps can change its byte hash.
