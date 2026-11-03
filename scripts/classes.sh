#!/usr/bin/env bash
# Fig. 3 (Section VI-C): the certified edges against the number of classes, 2 to 16, for
# deterministic reconfiguration with mean 1/2 s, the 1 % target and loads 1/2 and 9/10, with
# thresholds as multiples of the mean time away I. The certified policies use the lattice step
# 1/40 s; two classes at load 9/10 need thresholds beyond 4 I.
set -euo pipefail
cd "$(dirname "$0")/.."
T=target/release/tdf
R=results/classes
J=${THREADS:-$(nproc)}
C="--profiles det --rhos 1/2,9/10 --ns 2,3,4,6,8,12,16"
mkdir -p $R

$T witness --out $R/witness $C --d-over-i 6/5,3/2,9/5,11/5,27/10,33/10,4 --step 1/40 --threads $J \
  > $R/witness.log 2>&1
$T witness --out $R/witness_n2 --profiles det --rhos 9/10 --ns 2 --d-over-i 9/2,5,11/2,6,7 --step 1/40 \
  --threads $J > $R/witness_n2.log 2>&1
$T map --out $R/map $C --d-over-i 1/4,7/20,1/2,13/20,4/5,1,5/4 --epsilons 1/100 --gap-rows \
  --cache $R/cache --threads $J > $R/map.log 2>&1
