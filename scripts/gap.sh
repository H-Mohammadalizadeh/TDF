#!/usr/bin/env bash
# Fig. 2 (Section VI-B): the gap between the two certificates for four classes at load 0.9, with
# thresholds as multiples of the mean time away I, at mean reconfiguration times 1/2 to 500 s; and
# EC simulated at d = I/4, I/2, 3I/4 and I for 1/2 s (I = 15.5 s) and 50 s (I = 1550 s).
set -euo pipefail
cd "$(dirname "$0")/.."
T=target/release/tdf
R=results/gap
J=${THREADS:-$(nproc)}
S="--profiles det,exp,pareto --ns 4 --rhos 9/10 --means 1/2,5,50,500
   --d-over-i 1/10,1/4,2/5,1/2,3/5,3/4,9/10,1,11/10,5/4"
SIM="--profiles det,exp,pareto --policies ec --seeds 16 --record-seeds 0 --draws 1000"
mkdir -p $R

$T map --out $R/map $S --epsilons 1/100 --gap-rows --cache $R/cache --threads $J > $R/map.log 2>&1
$T witness --out $R/witness $S --threads $J > $R/witness.log 2>&1
$T screen --out $R/sim_fast $SIM --settings 4:9/10:31/8,4:9/10:31/4,4:9/10:93/8,4:9/10:31/2 \
  --cache $R/cache_sim --threads $J > $R/sim_fast.log 2>&1
$T screen --out $R/sim_slow $SIM --settings 4:9/10:775/2,4:9/10:775,4:9/10:2325/2,4:9/10:1550 \
  --mean 50 --step 1/20 --warmup 20000 --measure-factor 4 --cache $R/cache_sim --threads $J \
  > $R/sim_slow.log 2>&1
