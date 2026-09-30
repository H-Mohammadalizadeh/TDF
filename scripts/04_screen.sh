#!/usr/bin/env bash
# Policy screen: EC and HOL-BMW, 32 runs each, against the certified bounds in 6 distributions x 4
# (N, load, deadline) settings at the 1 % target. Feeds the policy-versus-bound numbers of
# "Limitations and Cost" and the simulation protocol of the setup. Three runs, because a setting's
# bootstrap stream is seeded by its position in a run. Cost: about 2 min on 16 cores.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
TDF=${TDF:-target/release/tdf}
THREADS=${THREADS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)}
RESULTS=${RESULTS:-results}
CACHE=$RESULTS/coefficients
command -v "$TDF" >/dev/null || { echo "$TDF not found; build it with 'cargo build --release'" >&2; exit 1; }

screen() { # screen PART SETTINGS: settings are N:load:deadline triples
  "$TDF" screen --out "$RESULTS/screen/$1" --settings "$2" --policies ec,hol_bmw \
    --seeds 32 --seed-start 20270901 --record-seeds 2 --draws 10000 --max-wall-seconds 900 \
    --cache "$CACHE" --threads "$THREADS"
}

screen a 2:1/20:2,4:1/2:4
screen b 2:7/10:3
screen c 8:7/10:6
