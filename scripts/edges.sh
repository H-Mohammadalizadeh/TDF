#!/usr/bin/env bash
# Fig. 1 (Section VI-A): the certified edges of four classes at the 1 % target. The test (`map`)
# and the certified policies (`witness`) are evaluated at thresholds that bracket each 1 % crossing:
# both at the mean reconfiguration time 1/2 s for deterministic and Pareto reconfiguration times,
# and the certified policies at 1/4 and 1/20 s (2x and 10x faster) for deterministic times. Four
# runs at a time, each on a quarter of the threads.
set -euo pipefail
cd "$(dirname "$0")/.."
THREADS=${THREADS:-$(nproc)}
export T=target/release/tdf R=results/edges J=$(( THREADS > 4 ? THREADS / 4 : 1 ))
mkdir -p $R

# name command profiles mean loads thresholds [options]
run() {
  local name=$1 cmd=$2 profiles=$3 mean=$4 rhos=$5 ds=$6
  shift 6
  local extra=()
  if [ "$cmd" = map ]; then extra=(--epsilons 1/100 --gap-rows --cache "$R/cache_$name"); fi
  "$T" "$cmd" --out "$R/$name" --profiles "$profiles" --ns 4 --means "$mean" --rhos "$rhos" --ds "$ds" \
    "${extra[@]}" --threads "$J" "$@" > "$R/$name.log" 2>&1
}
export -f run

grep -v '^#' <<'EOF' | xargs -P 4 -L 1 bash -c 'run "$@"' _
bound_a map det,pareto 1/2 1/20,1/10,3/20,1/5,1/4,3/10,7/20,2/5,9/20,1/2 1.2,1.4,1.6,1.8,2,2.2,2.5,3,3.5
bound_b map det,pareto 1/2 11/20,3/5,13/20,7/10,3/4,4/5 2,2.5,3,3.5,4,5,6,7,8,10
bound_c map det,pareto 1/2 17/20,9/10,19/20 8,10,12,14,16,20,24,28,34
w1_a witness det 1/2 1/20 1.6,1.7,1.8,1.9
w1_b witness det,pareto 1/2 1/20,1/10,3/20 2,2.5,3,3.5,4,5,6
w1_c witness det,pareto 1/2 1/5,1/4,3/10,7/20 3,3.5,4,5,6,7
w1_d witness det,pareto 1/2 2/5,9/20,1/2,11/20 4,5,6,7,8,10
w1_e witness det,pareto 1/2 3/5,13/20,7/10 6,7,8,10,12,14,16
w1_f witness det,pareto 1/2 3/4,4/5 10,12,14,16,20,24
w1_g witness det,pareto 1/2 17/20 16,20,24,28,34
w1_h witness det,pareto 1/2 9/10 28,32,36,40,46
w1_i witness det,pareto 1/2 19/20 66,72,80,90,100
w2_05 witness det 1/4 1/20 1.4,1.6,1.85,2.1,2.4,2.75,3.2
w2_10 witness det 1/4 1/10 1.55,1.8,2.05,2.35,2.7,3.1,3.6
w2_20 witness det 1/4 1/5 1.8,2.1,2.4,2.75,3.15,3.6,4.2
w2_30 witness det 1/4 3/10 2.15,2.5,2.85,3.25,3.75,4.3,5
w2_40 witness det 1/4 2/5 2.6,3,3.45,3.95,4.5,5.2,6.1
w2_50 witness det 1/4 1/2 3.1,3.55,4.1,4.75,5.45,6.25,7.3
w2_60 witness det 1/4 3/5 4,4.6,5.3,6.05,7,8,9.4
w2_70 witness det 1/4 7/10 5.4,6.2,7.15,8.2,9.4,10.8,12.7
w2_80 witness det 1/4 4/5 8.7,10,11.5,13.2,15.2,17.4,20.4 --step 1/200
w2_90 witness det 1/4 9/10 17,19.5,22.5,26,29.5,34,40 --step 1/200
w2_95 witness det 1/4 19/20 38,44,50,58,66,76,89 --step 1/200
w10_05 witness det 1/20 1/20 1.1,1.25,1.4,1.55,1.7,1.9,2.25
w10_10 witness det 1/20 1/10 1.15,1.3,1.45,1.6,1.8,2.05,2.4
w10_20 witness det 1/20 1/5 1.3,1.45,1.65,1.85,2.1,2.35,2.75
w10_30 witness det 1/20 3/10 1.5,1.7,1.9,2.15,2.4,2.7,3.2
w10_40 witness det 1/20 2/5 1.75,2,2.25,2.5,2.85,3.2,3.75
w10_50 witness det 1/20 1/2 2.1,2.35,2.65,3,3.4,3.8,4.5
w10_60 witness det 1/20 3/5 2.6,2.95,3.35,3.75,4.25,4.85,5.7
w10_70 witness det 1/20 7/10 3.45,3.95,4.45,5,5.7,6.45,7.6 --step 1/400
# the same point at d = 5 s on a finer lattice; the edges take the smaller certified bound
w10_70f witness det 1/20 7/10 5 --step 1/2000
w10_80 witness det 1/20 4/5 5.2,5.9,6.7,7.65,8.7,9.9,11.7 --step 1/400
w10_90 witness det 1/20 9/10 10.4,11.8,13.3,15.1,17.3,19.7,23.3 --step 1/400
w10_95 witness det 1/20 19/20 20.7,23.5,26.7,30.5,35.5,41,48 --step 1/400
EOF
