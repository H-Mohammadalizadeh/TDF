#!/usr/bin/env bash
# Reconfiguration-time distributions and independent checks: `distributions` tabulates the seven
# profiles (the distribution table); `validate` checks the coefficient constructions and the
# simulator's distributions against independent references (the numerical checks of the appendix).
# Cost: under a minute on 16 cores.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
TDF=${TDF:-target/release/tdf}
THREADS=${THREADS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)}
RESULTS=${RESULTS:-results}
command -v "$TDF" >/dev/null || { echo "$TDF not found; build it with 'cargo build --release'" >&2; exit 1; }

"$TDF" distributions --out "$RESULTS/distributions"
"$TDF" validate --out "$RESULTS/validation" --threads "$THREADS"
