#!/usr/bin/env bash
# Scheduler requirements: at N = 2, load 0.05, deadline 3, the target sweep with the rate point of
# OJF and the range of the total reconfiguration rate over a fine target grid; the three-class
# directional case with its service and reconfiguration changes. Feeds fig5_scheduling.pdf and the
# numbers of "Scheduler Requirements". Cost: about 1 min on 16 cores.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
TDF=${TDF:-target/release/tdf}
THREADS=${THREADS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)}
RESULTS=${RESULTS:-results}
CACHE=$RESULTS/coefficients
command -v "$TDF" >/dev/null || { echo "$TDF not found; build it with 'cargo build --release'" >&2; exit 1; }

"$TDF" sweep --out "$RESULTS/target_sweep" --n 2 --profile exp --rho 1/20 --d 3 \
  --epsilons 1/500,1/400,3/1000,1/250,9/2000,1/200,3/400,1/100,1/50,1/20 \
  --cache "$CACHE" --threads "$THREADS"
"$TDF" region --out "$RESULTS/rate_region" --n 2 --profile exp --rho 1/20 --d 3 \
  --cache "$CACHE" --threads "$THREADS"
"$TDF" directional --out "$RESULTS/directional" --rhos 1/20,1/10,1/5 --scales 1/2,3/4,1,5/4,3/2,2 \
  --cache "$CACHE" --threads "$THREADS"
