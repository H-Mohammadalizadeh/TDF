#!/usr/bin/env bash
# Evaluation grid, with the defaults of `tdf map`: 1890 settings (6 reconfiguration-time
# distributions x N in {2,3,4,6,8} x 9 loads x 7 deadlines) x 3 targets = 5670 exactly verified
# certificates. Feeds fig3_where.pdf, the evaluation-grid table, the exclusion counts, the example
# certificate, and the grid's cost and enclosure widths. Cost: about 76 min on 16 cores.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
TDF=${TDF:-target/release/tdf}
THREADS=${THREADS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)}
RESULTS=${RESULTS:-results}
CACHE=$RESULTS/coefficients
command -v "$TDF" >/dev/null || { echo "$TDF not found; build it with 'cargo build --release'" >&2; exit 1; }

"$TDF" map --out "$RESULTS/grid" --cache "$CACHE" --threads "$THREADS"
