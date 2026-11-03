#!/usr/bin/env bash
# The tables behind the figures, from the results of edges.sh, gap.sh and classes.sh (awk only),
# written to results/tables:
#   edges_points.csv    law,mean,rho,d,lb,lb_category,ec,ojf,upper            every certified point
#   edges.csv           law,mean,rho,side,d_cross,d_lo,d_hi,status,nonmonotone   1 % crossings (Fig. 1)
#   edges_closed.csv    rho,d_mm1,d_busy,d_nec,d_suf                          instantaneous reconfiguration
#   gap.csv             law,rho,mean,d_over_i,x,i_mean,d,bound,witness,width,limit,excess,deficit,
#                       sim,sim_lo,sim_hi                                      Fig. 2, in percent
#   gap_slopes.csv      law,rho,d_over_i,x,slope_width,slope_deficit,slope_excess
#   classes_points.csv  rho,n,d_over_i,d,lb,lb_category,ec,ojf,upper
#   classes.csv         rho,n,I,side,d_cross,r_cross,d_lo,d_hi,status,nonmonotone  (Fig. 3)
#   classes_slopes.csv  rho,side,points,s_per_class,i_per_class,slope_n,slope_i,ratio_min,ratio_max
#   classes_closed.csv  rho,n,I,d_mm1,d_busy,d_nec,d_suf
# lb is the certified lower bound of the test on the best worst-class miss probability; upper the
# smaller of the certified upper bounds of EC and OJF. A crossing interpolates log(miss) linearly in
# log(d) between the two grid points that bracket the target: the grid points are certified, the
# interpolated threshold is not. Before that, the bound is replaced by its running maximum from the
# right and the upper bound by its running minimum from the left; both stay certified, since the
# best miss probability and those of EC and OJF are nonincreasing in d.
set -euo pipefail
cd "$(dirname "$0")/.."
R=${RESULTS:-results}
OUT=$R/tables
mkdir -p "$OUT"
NUM='function num(s,  q) { split(s, q, "/"); return (q[2] == "") ? q[1] + 0 : q[1] / q[2] }
     function field(id, key,  f, n, k) { n = split(id, f, "|"); for (k = 1; k <= n; k++) if (index(f[k], key "=") == 1) return substr(f[k], length(key) + 2); return "" }
     function mean_of(id) { return (field(id, "tau") == "") ? "1/2" : field(id, "tau") }'

# ---- Fig. 1: edges of four classes at the 1 % target ----
{
  for f in "$R"/edges/*/map.csv; do
    awk -F, -v OFS=, "$NUM"'
      NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
      $c["eps"] == "1/100" { print "B", $c["profile"], mean_of($c["id"]), $c["rho"], $c["d"], $c["full_lb_max"], $c["full_category"] }' "$f"
  done
  for f in "$R"/edges/*/witness.csv; do
    awk -F, -v OFS=, "$NUM"'
      NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
      { print "W", $c["profile"], mean_of($c["id"]), $c["rho"], $c["d"], $c["hi_f64"], $c["ojf_hi"] }' "$f"
  done
} | awk -F, -v OFS=, "$NUM"'
  { key = $2 SUBSEP num($3) SUBSEP num($4) SUBSEP num($5); keys[key] = 1
    if ($1 == "B") {
      if ((key in lb) && lb[key] != $6) print "BOUND_DIFFERS " $2, $3, $4, $5 > "/dev/stderr"
      lb[key] = $6; cat[key] = $7
    } else {
      if (!(key in ec) || $6 + 0 < ec[key]) ec[key] = $6 + 0
      if ($7 != "" && (!(key in ojf) || $7 + 0 < ojf[key])) ojf[key] = $7 + 0
    } }
  END {
    print "law,mean,rho,d,lb,lb_category,ec,ojf,upper"
    for (key in keys) {
      split(key, p, SUBSEP)
      up = ""; if (key in ec) up = ec[key]; if ((key in ojf) && (up == "" || ojf[key] < up)) up = ojf[key]
      print p[1], p[2], p[3], p[4], (key in lb) ? lb[key] : "", (key in cat) ? cat[key] : "", \
            (key in ec) ? ec[key] : "", (key in ojf) ? ojf[key] : "", up } }' \
  | { IFS= read -r header; echo "$header"; sort -t, -k1,1 -k2,2g -k3,3g -k4,4g; } > "$OUT/edges_points.csv"

awk -F, -v OFS=, -v eps=0.01 '
  NR == 1 { next }
  { for (s = 1; s <= 2; s++) {
      val = (s == 1) ? $5 : $9; if (val == "") continue
      g = $1 SUBSEP $2 SUBSEP $3 SUBSEP s; n[g]++; D[g, n[g]] = $4 + 0; V[g, n[g]] = val + 0
      if (s == 1 && (($6 == "certified_infeasible") != (val + 0 > eps))) print "CATEGORY_MISMATCH " $1, $2, $3, $4 > "/dev/stderr" } }
  END {
    print "law,mean,rho,side,d_cross,d_lo,d_hi,status,nonmonotone"
    for (g in n) {
      split(g, p, SUBSEP); m = n[g]
      for (i = 1; i <= m; i++) { d[i] = D[g, i]; v[i] = V[g, i] }
      for (i = 2; i <= m; i++) { x = d[i]; y = v[i]; for (j = i - 1; j >= 1 && d[j] > x; j--) { d[j + 1] = d[j]; v[j + 1] = v[j] } d[j + 1] = x; v[j + 1] = y }
      bad = 0; for (i = 2; i <= m; i++) if (v[i] > v[i - 1] * (1 + 1e-9) + 1e-300) bad++
      if (p[4] == 2) { for (i = 2; i <= m; i++) if (v[i] > v[i - 1]) v[i] = v[i - 1] }
      else { for (i = m - 1; i >= 1; i--) if (v[i] < v[i + 1]) v[i] = v[i + 1] }
      status = "above"; dc = ""; lo = d[m]; hi = ""
      if (v[1] <= eps) { status = "below"; lo = ""; hi = d[1] }
      else for (i = 2; i <= m; i++) if (v[i] <= eps) {
        status = "bracketed"; lo = d[i - 1]; hi = d[i]
        if (v[i] > 0) dc = exp(log(lo) + (log(eps) - log(v[i - 1])) * (log(hi) - log(lo)) / (log(v[i]) - log(v[i - 1])))
        else dc = lo + (v[i - 1] - eps) / v[i - 1] * (hi - lo)
        break }
      print p[1], p[2], p[3], (p[4] == 1 ? "bound" : "witness"), dc, lo, hi, status, bad } }' "$OUT/edges_points.csv" \
  | { IFS= read -r header; echo "$header"; sort -t, -k1,1 -k2,2g -k4,4 -k3,3g; } > "$OUT/edges.csv"

# Instantaneous reconfiguration (Section V-C), mu = 4, N = 4: below d_nec, the larger root of the
# single-class bound and of misses behind the service in progress, no speed-up helps; above d_suf
# OJF meets the target once reconfiguration is fast enough.
awk -v OFS=, -v eps=0.01 'BEGIN {
  mu = 4; N = 4; L = log(1 / eps); print "rho,d_mm1,d_busy,d_nec,d_suf"
  for (k = 1; k <= 99; k++) {
    r = k / 100; mm1 = L / (mu * (1 - r / N)); suf = L / (mu * (1 - r))
    a = 0; b = 20; for (it = 0; it < 200; it++) { c = (a + b) / 2; if (exp(-mu * c) * (1 + r * mu * c) > eps) a = c; else b = c }
    busy = (a + b) / 2; nec = (busy > mm1) ? busy : mm1
    printf "%.2f,%.6f,%.6f,%.6f,%.6f\n", r, mm1, busy, nec, suf } }' > "$OUT/edges_closed.csv"

# ---- Fig. 2: the gap at load 0.9 (percent: width = witness - bound, excess = witness - limit,
# deficit = limit - bound, limit = (1 - d/I)+); sim = simulated EC with its 95 % interval ----
{
  awk -F, -v OFS=, "$NUM"'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
    { print "B", $c["profile"], $c["rho"], $c["mean"], $c["d_over_i"], $c["d"], $c["full_lb_max"] }' "$R/gap/map/map.csv"
  awk -F, -v OFS=, "$NUM"'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
    { print "W", $c["profile"], $c["rho"], $c["mean"], $c["d_over_i"], $c["hi"], $c["ojf_hi"] }' "$R/gap/witness/witness.csv"
  for s in fast:1/2 slow:50; do
    awk -F, -v OFS=, -v mean="${s#*:}" "$NUM"'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
      $c["policy"] == "ec" { n = $c["n"]; r = num($c["rho"]); I = (1 - r / n) * n * num(mean) / (1 - r)
        print "S", $c["profile"], $c["rho"], mean, num($c["d"]) / I, $c["worst_pooled"], $c["worst_lo"], $c["worst_hi"] }' \
      "$R/gap/sim_${s%%:*}/screen.csv"
  done
} | awk -F, -v OFS=, "$NUM"'
  function key(law, rho, mean, x) { return law SUBSEP sprintf("%.4f", rho) SUBSEP sprintf("%.4f", mean) SUBSEP sprintf("%.4f", x) }
  $1 == "B" { k = key($2, num($3), num($4), num($5)); bound[k] = 100 * $7; dfrac[k] = $5; rhofrac[k] = $3
              meanfrac[k] = $4; d[k] = num($6); next }
  $1 == "W" { k = key($2, num($3), num($4), num($5)); w = $6; if ($7 != "" && $7 + 0 < w + 0) w = $7
              wit[k] = 100 * w; next }
  $1 == "S" { k = key($2, num($3), num($4), $5); sim[k] = 100 * $6; simlo[k] = 100 * $7; simhi[k] = 100 * $8 }
  END {
    print "law,rho,mean,d_over_i,x,i_mean,d,bound,witness,width,limit,excess,deficit,sim,sim_lo,sim_hi"
    for (k in bound) {
      if (!(k in wit)) { print "MISSING WITNESS " k > "/dev/stderr"; continue }
      split(k, p, SUBSEP); x = p[4] + 0
      lim = (x < 1) ? 100 * (1 - x) : 0
      line = sprintf("%s,%s,%s,%s,%.4f,%.6g,%.6g,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f", p[1], rhofrac[k], meanfrac[k], dfrac[k], x,
                     d[k] / x, d[k], bound[k], wit[k], wit[k] - bound[k], lim, wit[k] - lim, lim - bound[k])
      line = line ((k in sim) ? sprintf(",%.6f,%.6f,%.6f", sim[k], simlo[k], simhi[k]) : ",,,")
      print line } }' \
  | { IFS= read -r header; echo "$header"; sort -t, -k1,1 -k2,2 -k5,5g -k3,3g; } > "$OUT/gap.csv"

# log-log least-squares slopes over the mean reconfiguration times, per (law, load, d/I)
awk -F, -v OFS=, "$NUM"'
  NR == 1 { next }
  { k = $1 SUBSEP $2 SUBSEP $4 SUBSEP $5; lx = log(num($3)) / log(10)
    n[k]++; sx[k] += lx; sxx[k] += lx * lx
    for (j = 1; j <= 3; j++) {
      y = (j == 1) ? $10 : (j == 2 ? $13 : $12)
      if (y > 1e-6) { ly = log(y) / log(10); sy[k, j] += ly; sxy[k, j] += lx * ly; ok[k, j]++ } } }
  END {
    print "law,rho,d_over_i,x,slope_width,slope_deficit,slope_excess"
    for (k in n) {
      split(k, p, SUBSEP); line = p[1] "," p[2] "," p[3] "," p[4]
      for (j = 1; j <= 3; j++) {
        if (ok[k, j] == n[k] && n[k] > 1)
          line = line sprintf(",%.3f", (n[k] * sxy[k, j] - sx[k] * sy[k, j]) / (n[k] * sxx[k] - sx[k] * sx[k]))
        else line = line ","
      }
      print line } }' "$OUT/gap.csv" | { IFS= read -r header; echo "$header"; sort -t, -k1,1 -k2,2 -k4,4g; } \
  > "$OUT/gap_slopes.csv"

# ---- Fig. 3: edges against the number of classes (deterministic, mean 1/2 s, 1 % target) ----
{
  awk -F, -v OFS=, "$NUM"'NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
    $c["eps"] == "1/100" { print "B", $c["rho"], $c["n"], field($c["id"], "d/I"), $c["d"], $c["full_lb_max"], $c["full_category"] }' \
    "$R/classes/map/map.csv"
  awk -F, -v OFS=, 'FNR == 1 { delete c; for (i = 1; i <= NF; i++) c[$i] = i; next }
    { print "W", $c["rho"], $c["n"], $c["d_over_i"], $c["d"], $c["hi_f64"], $c["ojf_hi"] }' \
    "$R/classes/witness/witness.csv" "$R/classes/witness_n2/witness.csv"
} | awk -F, -v OFS=, "$NUM"'
  { key = num($2) SUBSEP $3 SUBSEP num($5); keys[key] = 1; r[key] = num($4)
    if ($1 == "B") {
      if ((key in lb) && lb[key] != $6) print "BOUND_DIFFERS " $2, $3, $5 > "/dev/stderr"
      lb[key] = $6; cat[key] = $7
    } else {
      if (!(key in ec) || $6 + 0 < ec[key]) ec[key] = $6 + 0
      if ($7 != "" && (!(key in ojf) || $7 + 0 < ojf[key])) ojf[key] = $7 + 0
    } }
  END {
    print "rho,n,d_over_i,d,lb,lb_category,ec,ojf,upper"
    for (key in keys) {
      split(key, p, SUBSEP)
      up = ""; if (key in ec) up = ec[key]; if ((key in ojf) && (up == "" || ojf[key] < up)) up = ojf[key]
      print p[1], p[2], r[key], p[3], (key in lb) ? lb[key] : "", (key in cat) ? cat[key] : "", \
            (key in ec) ? ec[key] : "", (key in ojf) ? ojf[key] : "", up } }' \
  | { IFS= read -r header; echo "$header"; sort -t, -k1,1g -k2,2n -k4,4g; } > "$OUT/classes_points.csv"

# Where the value at the upper grid point is below eps/100, the bound falls about linearly to zero
# (at load 0.9 near d = I), and the crossing interpolates the miss probability linearly in d; a
# bracket wider than a factor 2 in d is marked "wide". I = (1 - rho/N) N mean / (1 - rho).
awk -F, -v OFS=, -v eps=0.01 -v mean=0.5 '
  NR == 1 { next }
  { for (s = 1; s <= 2; s++) {
      val = (s == 1) ? $5 : $9; if (val == "") continue
      g = $1 SUBSEP $2 SUBSEP s; n[g]++; D[g, n[g]] = $4 + 0; V[g, n[g]] = val + 0
      if (s == 1 && (($6 == "certified_infeasible") != (val + 0 > eps))) print "CATEGORY_MISMATCH " $1, $2, $4 > "/dev/stderr" } }
  END {
    print "rho,n,I,side,d_cross,r_cross,d_lo,d_hi,status,nonmonotone"
    for (g in n) {
      split(g, p, SUBSEP); m = n[g]; rho = p[1] + 0; N = p[2] + 0
      I = (1 - rho / N) * N * mean / (1 - rho)
      for (i = 1; i <= m; i++) { d[i] = D[g, i]; v[i] = V[g, i] }
      for (i = 2; i <= m; i++) { x = d[i]; y = v[i]; for (j = i - 1; j >= 1 && d[j] > x; j--) { d[j + 1] = d[j]; v[j + 1] = v[j] } d[j + 1] = x; v[j + 1] = y }
      bad = 0; for (i = 2; i <= m; i++) if (v[i] > v[i - 1] * (1 + 1e-9) + 1e-300) bad++
      if (p[3] == 2) { for (i = 2; i <= m; i++) if (v[i] > v[i - 1]) v[i] = v[i - 1] }
      else { for (i = m - 1; i >= 1; i--) if (v[i] < v[i + 1]) v[i] = v[i + 1] }
      status = "above"; dc = ""; lo = d[m]; hi = ""
      if (v[1] <= eps) { status = "below"; lo = ""; hi = d[1] }
      else for (i = 2; i <= m; i++) if (v[i] <= eps) {
        status = (d[i] > 2 * d[i - 1]) ? "wide" : "bracketed"; lo = d[i - 1]; hi = d[i]
        if (v[i] >= eps / 100) dc = exp(log(lo) + (log(eps) - log(v[i - 1])) * (log(hi) - log(lo)) / (log(v[i]) - log(v[i - 1])))
        else dc = lo + (v[i - 1] - eps) / (v[i - 1] - v[i]) * (hi - lo)
        break }
      printf "%s,%s,%.6f,%s,%s,%s,%s,%s,%s,%d\n", p[1], p[2], I, (p[3] == 1 ? "bound" : "witness"), \
             (dc == "" ? "" : sprintf("%.6f", dc)), (dc == "" ? "" : sprintf("%.6f", dc / I)), lo, hi, status, bad } }' \
  "$OUT/classes_points.csv" | { IFS= read -r header; echo "$header"; sort -t, -k1,1g -k4,4 -k2,2n; } > "$OUT/classes.csv"

# least-squares fits over N of the bracketed crossings, per load and side: seconds per added class,
# against the growth of I per class, mean/(1 - rho); log-log slopes against N and against I
awk -F, -v OFS=, -v mean=0.5 '
  NR == 1 || $9 != "bracketed" { next }
  { g = $1 SUBSEP $4; k[g]++; n = $2 + 0; x = log(n); z = log($3); y = log($5); dd = $5 + 0
    sx[g] += x; sy[g] += y; sxx[g] += x * x; sxy[g] += x * y; sz[g] += z; szz[g] += z * z; szy[g] += z * y
    sn[g] += n; snn[g] += n * n; sd[g] += dd; snd[g] += n * dd
    if (!(g in rmin) || $6 < rmin[g]) rmin[g] = $6; if (!(g in rmax) || $6 > rmax[g]) rmax[g] = $6 }
  END {
    print "rho,side,points,s_per_class,i_per_class,slope_n,slope_i,ratio_min,ratio_max"
    for (g in k) {
      split(g, p, SUBSEP); m = k[g]
      lin = (m * snd[g] - sn[g] * sd[g]) / (m * snn[g] - sn[g] * sn[g])
      bn = (m * sxy[g] - sx[g] * sy[g]) / (m * sxx[g] - sx[g] * sx[g])
      bi = (m * szy[g] - sz[g] * sy[g]) / (m * szz[g] - sz[g] * sz[g])
      printf "%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", p[1], p[2], m, lin, mean / (1 - p[1]), bn, bi, rmin[g], rmax[g] } }' \
  "$OUT/classes.csv" | { IFS= read -r header; echo "$header"; sort; } > "$OUT/classes_slopes.csv"

# the closed forms on a fine grid of N from 2 to 16 (the same roots as edges_closed.csv)
awk -v OFS=, -v eps=0.01 -v mean=0.5 'BEGIN {
  mu = 4; L = log(1 / eps); print "rho,n,I,d_mm1,d_busy,d_nec,d_suf"
  split("0.5 0.9", loads, " ")
  for (li = 1; li <= 2; li++) {
    r = loads[li] + 0
    a = 0; b = 20; for (it = 0; it < 200; it++) { c = (a + b) / 2; if (exp(-mu * c) * (1 + r * mu * c) > eps) a = c; else b = c }
    busy = (a + b) / 2; suf = L / (mu * (1 - r))
    for (k = 0; k <= 120; k++) {
      N = 2 * exp(log(8) * k / 120); I = (1 - r / N) * N * mean / (1 - r)
      mm1 = L / (mu * (1 - r / N)); nec = (busy > mm1) ? busy : mm1
      printf "%.2f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", r, N, I, mm1, busy, nec, suf } } }' > "$OUT/classes_closed.csv"
