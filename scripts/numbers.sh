#!/usr/bin/env bash
# Every number that Section VI and Appendix A quote, recomputed from the results of the stage
# scripts and the tables of tables.sh (awk only).
set -euo pipefail
cd "$(dirname "$0")/.."
R=${RESULTS:-results}
TB=$R/tables
G=$R/grid
NUM='function num(s,  q) { split(s, q, "/"); return (q[2] == "") ? q[1] + 0 : q[1] / q[2] }'
COLS='NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }'

echo "== VI-A: verdicts on the grid of Table I"
awk -F, "$COLS"'
  { n[$c["eps"]]++; v[$c["eps"], $c["verdict"]]++ }
  END {
    split("1/1000 1/100 1/20", e, " ")
    for (k = 1; k <= 3; k++) {
      t = e[k]; inf = v[t, "certified_infeasible"]; fea = v[t, "certified_feasible"]; und = v[t, "undetermined"]
      printf "  target %-6s  decided %4d of %d (%.0f %%): infeasible %d (%.0f %%), feasible %d (%.0f %%), undetermined %d (%.0f %%)\n",
             t, inf + fea, n[t], 100 * (inf + fea) / n[t], inf, 100 * inf / n[t], fea, 100 * fea / n[t], und, 100 * und / n[t] } }' \
  "$G/verdicts/verdicts.csv"
awk -F, "$COLS"'
  $c["eps"] == "1/100" && $c["full_category"] == "certified_infeasible" { all++; single += ($c["base_category"] == "certified_infeasible") }
  END { printf "  1 %% target: %d of %d infeasible settings (%.0f %%) excluded by the single-class bound, %d (%.0f %%) need the reconfiguration constraints\n",
               single, all, 100 * single / all, all - single, 100 * (all - single) / all }' "$G/map/map.csv"
echo "  1 % target, exclusions by load: single-class bound | added by the window constraints | added by unavoidable misses | full test | added by the next-visit bound"
awk -F, "$COLS"'
  FNR == NR { if (FNR > 1 && $c["eps"] == "1/100") nogap[$c["id"]] = ($c["full_category"] == "certified_infeasible"); next }
  FNR == 1 { next }
  $c["eps"] == "1/100" {
    r = $c["rho_f64"]; inf = "certified_infeasible"; base = ($c["base_category"] == inf)
    b[r] += base; w[r] += ($c["base_windows_category"] == inf) - base; p[r] += ($c["base_phases_category"] == inf) - base
    f[r] += ($c["full_category"] == inf); gap[r] += ($c["full_category"] == inf) && !nogap[$c["id"]] }
  END { for (r in b) printf "    load %-5s %4d %4d %4d %4d %4d\n", r, b[r], w[r], p[r], f[r], gap[r] }' "$G/nogap/map.csv" "$G/map/map.csv" | sort -k2g
awk -F, "$COLS"'
  FNR == NR { if (FNR > 1) nogap[$c["id"]] = ($c["full_category"] == "certified_infeasible"); next }
  FNR == 1 { next }
  $c["full_category"] == "certified_infeasible" { all[$c["eps"]]++; gap[$c["eps"]] += !nogap[$c["id"]] }
  END { printf "  next-visit bound needed for"
        split("1/1000 1/100 1/20", e, " ")
        for (k = 1; k <= 3; k++) printf " %d of %d (%.0f %%) at %s;", gap[e[k]], all[e[k]], 100 * gap[e[k]] / all[e[k]], e[k]
        print "" }' "$G/nogap/map.csv" "$G/map/map.csv"

echo "== VI-A: edges of four classes at the 1 % target (Fig. 1), threshold in s"
awk -F, '
  function f(x) { return (x == "") ? "-" : sprintf("%.2f", x) }
  FNR == NR { if (FNR > 1) { nec[$1 + 0] = $4; suf[$1 + 0] = $5; if ($1 + 0 <= 0.95 && $4 > necmax) necmax = $4 } next }
  FNR > 1 && $8 == "bracketed" { e[$1, $2 + 0, $3 + 0, $4] = $5 }
  END {
    printf "    load  bound det  bound Pareto  feasible det  feasible Pareto  feasible det 2x  feasible det 10x  d_nec  d_suf\n"
    split("0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.85 0.9 0.95", L, " ")
    for (k = 1; k <= 12; k++) {
      r = L[k] + 0
      printf "    %-5s %9s %13s %13s %16s %16s %17s %6.3f %6.3f\n", L[k], f(e["det", 0.5, r, "bound"]), f(e["pareto", 0.5, r, "bound"]),
             f(e["det", 0.5, r, "witness"]), f(e["pareto", 0.5, r, "witness"]), f(e["det", 0.25, r, "witness"]), f(e["det", 0.05, r, "witness"]), nec[r], suf[r] }
    printf "  load 0.5: 2x faster reconfiguration lowers the feasible edge by %.0f %%, 10x by %.0f %%\n",
           100 * (1 - e["det", 0.25, 0.5, "witness"] / e["det", 0.5, 0.5, "witness"]), 100 * (1 - e["det", 0.05, 0.5, "witness"] / e["det", 0.5, 0.5, "witness"])
    printf "  mean time away I of four classes at load 0.9: %.1f s; largest d_nec up to load 0.95: %.3f s\n", (1 - 0.9 / 4) * 4 * 0.5 / 0.1, necmax }' \
  "$TB/edges_closed.csv" "$TB/edges.csv"

echo "== VI-A: same mean and variance, opposite verdicts (2 classes, load 0.05, d = 3 s, 0.1 % target)"
awk -F, "$COLS"'$c["n"] == 2 && $c["rho"] == "1/20" && $c["d"] == "3" && $c["eps"] == "1/1000" && ($c["profile"] == "gamma" || $c["profile"] == "pareto") {
  printf "  %-7s %s %s (lower bound %.4g %%, upper bound %.4g %%)\n", $c["profile"], $c["verdict"], $c["certified_by"], 100 * $c["full_lb_max"], 100 * $c["upper"] }' \
  "$G/verdicts/verdicts.csv"
awk 'BEGIN { pareto = (1 / 9) ^ 3; gamma = exp(-18) * (1 + 18 + 18 * 18 / 2)
  printf "  P(tau > 3 s): Pareto %.3g, gamma %.3g, ratio %.0f\n", pareto, gamma, pareto / gamma }'

echo "== VI-A: exclusions per distribution (of 315 settings each)"
awk -F, "$COLS"'$c["full_category"] == "certified_infeasible" { x[$c["eps"], $c["profile"]]++ }
  END { split("det tnorm gamma lognormal pareto exp", P, " ")
        for (k = 1; k <= 2; k++) { t = (k == 1) ? "1/1000" : "1/20"; printf "  target %-6s", t
          for (j = 1; j <= 6; j++) printf " %s %d", P[j], x[t, P[j]]; print "" } }' "$G/map/map.csv"

echo "== VI-B: the gap for four classes at load 0.9 (Fig. 2), in pp"
awk -F, "$NUM"'NR > 1 && $2 == "9/10" { v[$1, $3, $4] = $10; ex[$1, $3, $4] = $12; de[$1, $3, $4] = $13 }
  END {
    split("1/4 1/2 3/4", F, " ")
    for (m = 1; m <= 2; m++) { mean = (m == 1) ? "1/2" : "500"; lo = 1e9; hi = -1
      for (k = 1; k <= 3; k++) { w = v["det", mean, F[k]]; if (w < lo) lo = w; if (w > hi) hi = w }
      printf "  deterministic, I/4 <= d <= 3I/4, mean %s s: %.2f to %.2f\n", mean, lo, hi }
    for (m = 1; m <= 2; m++) { mean = (m == 1) ? "1/2" : "50"
      printf "  deterministic, d = I/2, mean %s s: gap %.2f = %.2f above the limit (EC) + %.2f below it (test)\n",
             mean, v["det", mean, "1/2"], ex["det", mean, "1/2"], de["det", mean, "1/2"] }
    printf "  d = I, mean 500 s: deterministic %.2f, exponential %.2f, Pareto %.2f\n", v["det", "500", "1"], v["exp", "500", "1"], v["pareto", "500", "1"]
    for (m = 1; m <= 2; m++) { mean = (m == 1) ? 0.5 : 50; lam = 0.9 * 4 / 4 * (1 - 0.9 / 4) * 4 * mean / 0.1
      printf "  class-i arrivals while away at %g s: %.4g, relative fluctuation %.2f %%\n", mean, lam, 100 / sqrt(lam) } }' "$TB/gap.csv"
awk -F, 'NR > 1 && $2 == "9/10" && ($3 == "1/4" || $3 == "1/2" || $3 == "3/4") { s = s sprintf(" %s %s %s;", $1, $3, $5) }
  END { print "  log-log slopes of the gap over the mean:" s }' "$TB/gap_slopes.csv"
awk -F, 'NR > 1 && $14 != "" { n++; ok += ($15 <= $9 && $9 <= $16); d = $9 - $14; if (d < 0) d = -d; if (d > dmax) dmax = d }
  END { printf "  certified EC bound inside the 95 %% interval of simulated EC in %d of %d settings; largest distance %.2f pp\n", ok, n, dmax }' \
  "$TB/gap.csv"

echo "== VI-C: classes (Fig. 3), deterministic reconfiguration with mean 0.5 s, 1 % target"
awk -F, 'NR > 1 { if ($9 == "bracketed" || $9 == "wide") e[$1 + 0, $2 + 0, $4] = $5; I[$1 + 0, $2 + 0] = $3; st[$1 + 0, $2 + 0, $4] = $9; lo[$1 + 0, $2 + 0, $4] = $7; hi[$1 + 0, $2 + 0, $4] = $8 }
  END {
    split("2 3 4 6 8 12 16", N, " ")
    for (l = 1; l <= 2; l++) { r = (l == 1) ? 0.5 : 0.9
      printf "  load %s (I grows by %g s per class)\n    N       I   infeasible edge   d/I   feasible edge   d/I\n", r, 0.5 / (1 - r)
      for (k = 1; k <= 7; k++) { n = N[k] + 0; b = st[r, n, "bound"]; w = st[r, n, "witness"]
        printf "    %-3d %7.2f %17s %5s %15s %5s\n", n, I[r, n],
               (b == "bracketed") ? sprintf("%.2f", e[r, n, "bound"]) : sprintf("%s-%s (%s)", lo[r, n, "bound"], hi[r, n, "bound"], b),
               (b == "bracketed") ? sprintf("%.2f", e[r, n, "bound"] / I[r, n]) : "",
               (w == "bracketed") ? sprintf("%.2f", e[r, n, "witness"]) : sprintf("> %s (%s)", lo[r, n, "witness"], w),
               (w == "bracketed") ? sprintf("%.2f", e[r, n, "witness"] / I[r, n]) : "" } } }' "$TB/classes.csv"

echo "== VI-D: limitations and cost"
awk -F, "$COLS"'
  FNR == NR { if (FNR > 1) { cat[$c["id"]] = $c["full_category"]; lb[$c["id"]] = $c["full_lb_max"] } next }
  FNR == 1 { next }
  { n++; changed += (cat[$c["id"]] != $c["full_category"]); a = lb[$c["id"]]; s = $c["full_lb_max"] - a; if (s < 0) s = -s
    if (s > smax) smax = s; if (a >= 0.001 && s / a > rmax) rmax = s / a }
  END { printf "  half the time-grid step (four classes): %d of %d verdicts change; the bound moves by at most %.3f pp,\n", changed, n, 100 * smax
        printf "  and by at most %.1f %% of its value where it is at least the smallest target, 0.1 %%\n", 100 * rmax }' \
  "$G/map/map.csv" "$G/halfstep/map.csv"
awk -F, "$COLS"'{ k = $c["n"]; m[k]++; ex[k] += ($c["full_exact_simplex"] == "true"); if ($c["full_rows"] + 0 > rows[k]) rows[k] = $c["full_rows"] + 0 }
  END { for (k in m) printf "  N = %d: at most %d constraints, exact simplex in %d of %d cases (%.1f %%)\n", k, rows[k], ex[k], m[k], 100 * ex[k] / m[k] }' \
  "$G/map/map.csv" | sort -t= -k2n
awk -F, "$COLS"'$c["n"] == 16 { if (lo == "" || $c["full_rows"] + 0 < lo) lo = $c["full_rows"] + 0; if ($c["full_rows"] + 0 > hi) hi = $c["full_rows"] + 0 }
  END { printf "  N = 16 (Fig. 3): %d to %d constraints\n", lo, hi }' "$R/classes/map/map.csv"

echo "== Appendix A: the OJF interval against the closed form (two classes, mean 0.5 s, step 0.5 ms)"
awk -F, "$COLS"'{ n++; ok += ($c["workload_lower"] <= $c["u_exact_lo"] && $c["u_exact_hi"] <= $c["u_workload"])
    d = $c["u_workload"] - $c["u_exact_hi"]; if (d > dmax) dmax = d }
  END { printf "  %d of %d intervals contain the exact miss probability; the upper end exceeds it by at most %.4f pp\n", ok, n, 100 * dmax }' \
  "$R/bracket/bracket.csv"
