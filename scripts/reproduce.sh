#!/usr/bin/env bash
# Every result of the paper from scratch: build, run the stages into results/, then the tables, the
# numbers (results/numbers.txt) and the figures (figures/). The classes stage, whose largest LPs
# run for hours on one core each, runs alongside the others.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release
scripts/classes.sh &
classes=$!
scripts/grid.sh
scripts/edges.sh
scripts/gap.sh
scripts/bracket.sh
wait $classes
scripts/tables.sh
scripts/numbers.sh | tee results/numbers.txt
python3 plotting/figures.py
