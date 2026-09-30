#!/usr/bin/env bash
# Certified intervals [L_safe, U+] of the symmetric case: the closed-form exponential case (also at
# g = 1/4000), light and finite load, the equal-moment gamma/Pareto pair, the refinement of the
# workload bound, and OJF simulated in the closed-form case. Feeds fig4_bracket.pdf, the enclosure
# table and the interval numbers of the evaluation and the appendix. Cost: about 3 min on 16 cores.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
TDF=${TDF:-target/release/tdf}
THREADS=${THREADS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)}
RESULTS=${RESULTS:-results}
CACHE=$RESULTS/coefficients
command -v "$TDF" >/dev/null || { echo "$TDF not found; build it with 'cargo build --release'" >&2; exit 1; }

bracket() { # bracket DIR OPTIONS...
  local dir=$1
  shift
  "$TDF" bracket --out "$RESULTS/bracket/$dir" "$@" --eps 1/100 --cache "$CACHE" --threads "$THREADS"
}

bracket benchmark --profiles exp --ns 2 --ds 2,3 \
  --rhos 1/10000,1/1000,1/100,3/100,1/20,1/10,1/5,3/10,2/5,9/20
bracket light_load --profiles det,gamma,pareto_heavy --ns 2,4 --ds 2 \
  --rhos 1/1000,3/1000,1/100,3/100,1/20
bracket benchmark_fine --profiles exp --ns 2 --ds 2,3 --rhos 1/10000,1/1000 --step 1/4000
bracket finite_load --profiles det,gamma,pareto_heavy --ns 2,4 --ds 2,3 \
  --rhos 1/100,1/20,1/10,1/5,3/10
bracket moments --profiles gamma,pareto --ns 4 --ds 15/8 --rhos 1/4000
for s in 500 1000 2000 4000 8000; do
  bracket "refinement/step_$s" --profiles det,gamma --ns 2 --ds 2 --rhos 1/10 --step "1/$s"
done

"$TDF" ojf --out "$RESULTS/ojf" --n 2 --profile exp --rhos 1/100,1/20,1/5,2/5 --d 2 \
  --thresholds 2,3 --seeds 32 --threads "$THREADS"
