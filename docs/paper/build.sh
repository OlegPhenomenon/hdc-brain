#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
pandoc paper.md \
  --from=markdown \
  --to=latex \
  --standalone \
  --citeproc \
  --bibliography=refs.bib \
  --number-sections \
  --metadata=title:"HDC-Brain: A 300M Hyperdimensional Language Model with Bipolar Codebook" \
  --metadata=author:"Oleg Hasjanov" \
  --pdf-engine=xelatex \
  -V mainfont="Times New Roman" \
  -V monofont="Menlo" \
  -V geometry:margin=1in \
  -o paper.pdf
