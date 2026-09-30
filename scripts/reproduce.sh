#!/usr/bin/env bash
# Runs the steps 01-06 in order (or only those given, e.g. `scripts/reproduce.sh 4 5 6`), each also
# logged to $LOGS/<step>.log, then writes the figures, figures/reference_numbers.tex and
# figures/numbers.md with plotting/paper.py. Cost: about 1.5 h on 16 cores, most of it in step 01.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON=${PYTHON:-python3}
LOGS=${LOGS:-logs}
RESULTS=${RESULTS:-results}
steps=(01_grid 02_distributions 03_numerics 04_screen 05_intervals 06_scheduling)

usage() {
  echo "usage: scripts/reproduce.sh [STEP ...]    STEP: 1-6 (default: all)" >&2
  exit 2
}

selected=(0 0 0 0 0 0)
if (($# == 0)); then
  selected=(1 1 1 1 1 1)
fi
for arg in "$@"; do
  [[ $arg =~ ^0?([1-6])$ ]] || usage
  selected[BASH_REMATCH[1] - 1]=1
done

mkdir -p "$LOGS"
for i in "${!steps[@]}"; do
  ((selected[i])) || continue
  step=${steps[i]}
  echo "== $step"
  start=$SECONDS
  bash "scripts/$step.sh" 2>&1 | tee "$LOGS/$step.log"
  echo "== $step done in $((SECONDS - start)) s"
done

echo "== figures and numbers"
"$PYTHON" plotting/paper.py --results "$RESULTS"
