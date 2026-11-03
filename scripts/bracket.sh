#!/usr/bin/env bash
# Appendix A: the certified OJF interval against the exact miss probability of the closed-form
# case (exponential reconfiguration times with mu = N / taubar): two classes, mean 1/2 s, loads
# up to 0.45, thresholds 2 and 3 s, time-grid step 1/2000 s.
set -euo pipefail
cd "$(dirname "$0")/.."
T=target/release/tdf
R=results/bracket
mkdir -p $R

$T bracket --out $R --profiles exp --ns 2 --mean 1/2 --eps 1/100 --ds 2,3 \
  --rhos 1/10000,1/1000,1/100,3/100,1/20,1/10,1/5,3/10,2/5,9/20 --threads "${THREADS:-$(nproc)}" \
  > results/bracket.log 2>&1
