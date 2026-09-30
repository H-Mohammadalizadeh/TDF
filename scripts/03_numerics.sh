#!/usr/bin/env bash
# Single-setting checks, each run without a coefficient cache so that it reports its full cost:
# precision (64-192 bits), grid step (1/100-1/4000), restricted versus all class subsets of the
# window constraint, and run time against N (2-16). Feeds the cost and numerical-check numbers of
# "Limitations and Cost" and of the appendix. Cost: about 9 min on 16 cores.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
TDF=${TDF:-target/release/tdf}
THREADS=${THREADS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)}
RESULTS=${RESULTS:-results}
command -v "$TDF" >/dev/null || { echo "$TDF not found; build it with 'cargo build --release'" >&2; exit 1; }

out=$RESULTS/numerics
mkdir -p "$out"
point() { "$TDF" point --eps 1/100 --threads "$THREADS" "$@"; }

for profile in exp gamma tnorm lognormal pareto; do
  for bits in 64 96 128 192; do
    point --profile "$profile" --n 4 --rho 1/2 --d 4 --bits "$bits" \
      --out "$out/precision_${profile}_${bits}.json"
  done
done
for profile in exp gamma; do
  for step in 1/100 1/250 1/500 1/1000 1/2000 1/4000; do
    point --profile "$profile" --n 4 --rho 1/2 --d 4 --step "$step" \
      --out "$out/step_${profile}_${step//\//_}.json"
  done
done
for n in 5 6 8; do
  for rho in 1/2 17/20; do
    point --profile gamma --n "$n" --rho "$rho" --d 4 \
      --out "$out/subsets_pairs_N${n}_rho${rho//\//_}.json"
    point --profile gamma --n "$n" --rho "$rho" --d 4 --all-subsets \
      --out "$out/subsets_all_N${n}_rho${rho//\//_}.json"
  done
done
for n in 2 3 4 6 8 12 16; do
  point --profile gamma --n "$n" --rho 1/2 --d 4 --out "$out/runtime_N${n}.json"
done
