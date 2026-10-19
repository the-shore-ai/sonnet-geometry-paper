# Recoverable Poem Identity and Changing Relational Geometry Across Transformer Depth

Timothy O'Brien, The Shore. Preprint v1.0.0; not peer reviewed.

Paper and compact companion deposit: https://doi.org/10.5281/zenodo.23464450 .

This public companion contains the paper, generation recipes, original source
kernels and a compact verification dataset. The complete research archive
remains private. No credentials, model weights, raw token-level activation
tensors or private Git history are included.

The main observation is that identifying an inventory-preserving shuffled poem,
preserving its relationships, agreeing across models and sharing sensitivity to
scrambles are different measurements. With calibration from Sidney, Qwen/Muse
identify token-shuffled Spenser poems at 83.3%/93.8% among 24 candidates, versus
6.3%/18.8% raw. This can reflect lexical identity and does not demonstrate
preserved meaning. The new measurements were selected after both population
outcomes were known. Neither Sidney nor Spenser meets the complete frozen
primary pattern. Literary usefulness remains unestablished.

Read [the paper source](paper/manuscript.md), [the PDF](output/pdf/sonnet-geometry-v1.0.0-preprint.pdf),
[reproduction instructions](REPRODUCING.md), [compact data description](data/README.md)
and [adversarial review](docs/publication-v1-review.md).

## Check the compact data on CPU

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-compact.txt
.venv/bin/python recipes/verify_release.py --figures
```

This verifies all data-file hashes and recomputes identity/relational results,
all six frozen criterion decisions in each population, structural contrasts
and selected shared-response components. It writes `outputs/public-verification.json`
and two independently rendered charts. No cloud account, model download or GPU
is needed. A successful check confirms arithmetic on the released derived
files, not an independent model-inference replication.

Full source/condition regeneration and local extraction use the separate locked
environments and explicit commands in REPRODUCING.md. They require your own
hardware/resources. Exact source snapshots from nine recorded execution commits and reviewed baseline kernels
are retained with hashes; the public local wrapper is new and is labeled as such.
No additional inference was run to prepare this release.

Original code: MIT. Paper, figures and original derived numeric data: CC BY 4.0.
Third-party sources keep their own terms; see [licenses](LICENSES.md). Extensive
AI assistance, audit limitations and supporting research are acknowledged in the paper.
