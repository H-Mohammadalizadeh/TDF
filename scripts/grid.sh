#!/usr/bin/env bash
# The evaluation grid of Table I (Sections VI-A and VI-D): 1,890 settings at three targets.
#   witness   certified upper bounds on the miss probabilities of EC and OJF
#   map       the feasibility test at its four constraint levels
#   verdicts  certified infeasible, certified feasible or undetermined
#   nogap     the test without the next-visit bound
#   halfstep  four classes at half the time-grid step
set -euo pipefail
cd "$(dirname "$0")/.."
T=target/release/tdf
R=results/grid
J=${THREADS:-$(nproc)}
G="--profiles det,exp,gamma,tnorm,lognormal,pareto --rhos 1/20,1/10,1/5,3/10,1/2,7/10,17/20,9/10,19/20
   --ds 1,2,3,4,6,8,12"
E="--epsilons 1/1000,1/100,1/20"
mkdir -p $R

$T witness --out $R/witness $G --ns 2,3,4,6,8 --threads $J > $R/witness.log 2>&1
$T map --out $R/map $G --ns 2,3,4,6,8 $E --gap-rows --cache $R/cache --threads $J > $R/map.log 2>&1
$T verdicts --map $R/map/map.csv --witness $R/witness/witness.csv --out $R/verdicts > $R/verdicts.log 2>&1
$T map --out $R/nogap $G --ns 2,3,4,6,8 $E --cache $R/cache --threads $J > $R/nogap.log 2>&1
$T map --out $R/halfstep $G --ns 4 $E --gap-rows --step 1/4000 --cache $R/cache --threads $J \
  > $R/halfstep.log 2>&1
