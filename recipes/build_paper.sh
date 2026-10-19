#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p output/pdf
pandoc paper/manuscript.md --from markdown --standalone --citeproc \
  --bibliography paper/references.bib --resource-path=paper:. \
  --pdf-engine=xelatex --lua-filter=paper/wrap-code.lua \
  --include-in-header=paper/pdf-header.tex --toc --toc-depth=2 \
  -V fontsize=10pt -V geometry:margin=0.85in -V colorlinks=true \
  -V urlcolor=blue -V linkcolor=blue \
  --output output/pdf/sonnet-geometry-v1.0.0-preprint.pdf
