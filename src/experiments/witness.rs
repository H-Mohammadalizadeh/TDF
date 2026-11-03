//! `tdf witness`: the certified enclosure `lo <= m <= hi` of the miss probability under cyclic
//! exhaustive polling ([`crate::witness`]) for the symmetric settings of `tdf map` (profiles x
//! mean reconfiguration times x numbers of classes x deadlines x loads), with the same options.
//! The enclosure does not depend on the target, so every setting is one row of `witness.csv`;
//! its `id` is the identifier of `tdf map` without the target (`|eps=...`), and `tdf verdicts`
//! joins the two tables on it. Every row also records the lattice step `g` and the truncation
//! depth `K` used, and with identical classes the certified upper bound on the miss
//! probability of OJF (oldest job first) from the reference analysis: the smallest of the exact
//! exponential benchmark where it applies, the baseline bound `U_0` where the tail
//! `P(tau + S > d)` has a closed form, and the workload bound `U_K^+`, on the grid of the
//! lattice step (any step gives a valid bound). The OJF bound is computed only where `hi`
//! exceeds `--ojf-above` (source `not_needed` otherwise), since below it every target from that
//! value up is already met. Output: `witness.csv`, `summary.json`.

use super::common::{grid_settings, manifest, parse_rational, profile_law, write_csv, write_json};
use crate::feasibility::distributions::{Distribution, Model};
use crate::feasibility::rational::{Q, qi, qr, to_f64};
use crate::feasibility::reference::Homogeneous;
use crate::witness::bound::{default_h0, sw_pre};
use crate::witness::laws::q_up;
use crate::witness::{Law, Opts, System, compute, default_step};
use num_traits::{One, ToPrimitive};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Enclose the miss probability under cyclic exhaustive polling on the grid of `tdf map`.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// Output directory.
    #[arg(long)]
    pub out: PathBuf,
    /// Reconfiguration-time profiles.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "det,exp,gamma,tnorm,lognormal,pareto"
    )]
    pub profiles: Vec<String>,
    /// Numbers of classes.
    #[arg(long, value_delimiter = ',', default_value = "2,3,4,6,8")]
    pub ns: Vec<usize>,
    /// Total loads.
    #[arg(long, value_delimiter = ',', default_value = "1/20,1/10,1/5,3/10,1/2,7/10,17/20,9/10,19/20", value_parser = parse_rational)]
    pub rhos: Vec<Q>,
    /// Deadlines in seconds.
    #[arg(long, value_delimiter = ',', default_value = "1,2,3,4,6,8,12", value_parser = parse_rational)]
    pub ds: Vec<Q>,
    /// Deadlines as multiples of `I = (1 - rho/N) N taubar / (1 - rho)`, rounded to the nearest
    /// multiple of the grid step of `tdf map` (`--map-step`), in place of `--ds`.
    #[arg(long, value_delimiter = ',', value_parser = parse_rational)]
    pub d_over_i: Vec<Q>,
    /// Mean reconfiguration times in seconds.
    #[arg(long = "means", alias = "mean", value_delimiter = ',', default_value = "1/2", value_parser = parse_rational)]
    pub means: Vec<Q>,
    /// Squared coefficient of variation of the reconfiguration time.
    #[arg(long, default_value = "1/3", value_parser = parse_rational)]
    pub cv2: Q,
    /// Lattice step in seconds (default `d / max(400, ceil(100 d / taubar))`, at most d/400 and
    /// about taubar/100).
    #[arg(long, value_parser = parse_rational)]
    pub step: Option<Q>,
    /// The `--step` given to `tdf map`, which rounds the deadlines of `--d-over-i` (default: the
    /// default step of `tdf map`).
    #[arg(long, value_parser = parse_rational)]
    pub map_step: Option<Q>,
    /// Truncation tolerance: `K` is the fewest past cycles with `E[R_K] <= tol E[I]`.
    #[arg(long, default_value_t = 1e-7)]
    pub tol: f64,
    /// Grid step of the OJF workload bound (default `d / clamp(ceil(500 mu d), 400, 32768)`,
    /// about 1/(500 mu)). The bound exceeds OJF's miss probability by an amount proportional to
    /// the step, so the default follows the service time, not the lattice of the witness.
    #[arg(long, value_parser = parse_rational)]
    pub ojf_step: Option<Q>,
    /// Compute the OJF bound only where `hi` exceeds this value (default 1/1000, the smallest
    /// default target of `tdf map`): at or below it, cyclic exhaustive polling already meets
    /// every target from this value up, so OJF cannot change a verdict there; 0 computes the
    /// bound everywhere.
    #[arg(long, default_value = "1/1000", value_parser = parse_rational)]
    pub ojf_above: Q,
    /// Leave out the OJF bound.
    #[arg(long)]
    pub no_ojf: bool,
    /// Worker threads (default: all cores).
    #[arg(long)]
    pub threads: Option<usize>,
}

/// The deadline `r I` rounded to the nearest multiple of `step` (at least one step), with
/// `I = (1 - rho/N) N mean / (1 - rho)`: the deadline of `tdf map --d-over-i`.
pub fn deadline_over_i(n: usize, rho: &Q, mean: &Q, r: &Q, step: &Q) -> Result<Q, String> {
    if *rho >= Q::one() {
        return Err(format!("--d-over-i needs a load below one, got {rho}"));
    }
    let nq = qi(n as i64);
    let intervisit = (Q::one() - rho / &nq) * &nq * mean / (Q::one() - rho);
    Ok((r * intervisit / step).round().max(Q::one()) * step)
}

/// The identifier of a setting: that of `tdf map` without the target, i.e. `profile|N=n|rho=..|d=..`
/// followed by `|tau=mean` when the mean is not 1/2 and by `|d/I=r` for a deadline given as a
/// multiple of `I`.
pub fn setting_id(p: &str, n: usize, rho: &Q, d: &Q, mean: &Q, over_i: Option<&Q>) -> String {
    let mut id = format!("{p}|N={n}|rho={rho}|d={d}");
    if *mean != qr(1, 2) {
        id.push_str(&format!("|tau={mean}"));
    }
    if let Some(r) = over_i {
        id.push_str(&format!("|d/I={r}"));
    }
    id
}

/// A `tdf map` identifier without its target segment (`eps=...`).
pub fn strip_target(id: &str) -> String {
    id.split('|')
        .filter(|s| !s.starts_with("eps="))
        .collect::<Vec<_>>()
        .join("|")
}

/// One setting of the grid.
struct Setting {
    id: String,
    profile: String,
    n: usize,
    rho: Q,
    mean: Q,
    d: Q,
    over_i: Option<Q>,
    dist: Distribution,
    law: Law,
    g: Q,
}

/// The settings of the grid in the order of `tdf map` (profile, mean, classes, deadline, load).
fn settings(args: &Args) -> Result<Vec<Setting>, String> {
    let mut out = Vec::new();
    let deadlines: Vec<(Option<Q>, Option<Q>)> = if args.d_over_i.is_empty() {
        args.ds.iter().map(|d| (Some(d.clone()), None)).collect()
    } else {
        args.d_over_i
            .iter()
            .map(|r| (None, Some(r.clone())))
            .collect()
    };
    for p in &args.profiles {
        for mean in &args.means {
            let dist = profile_law(p, mean, &args.cv2)?;
            let law = Law::from_distribution(&dist)?;
            for &n in &args.ns {
                // the step of `tdf map`, which rounds the deadlines given as multiples of I
                let map_step = match &args.map_step {
                    Some(s) => s.clone(),
                    None => grid_settings(n, None, None, None, false)?.step * mean * qi(2),
                };
                for (ds, over_i) in &deadlines {
                    for rho in &args.rhos {
                        if *rho >= Q::one() {
                            return Err(format!("the load must be below one, got {rho}"));
                        }
                        let d = match (ds, over_i) {
                            (Some(d), _) => d.clone(),
                            (None, Some(r)) => deadline_over_i(n, rho, mean, r, &map_step)?,
                            _ => unreachable!(),
                        };
                        let g = args.step.clone().unwrap_or_else(|| default_step(&d, mean));
                        let id = setting_id(p, n, rho, &d, mean, over_i.as_ref());
                        out.push(Setting {
                            id,
                            profile: p.clone(),
                            n,
                            rho: rho.clone(),
                            mean: mean.clone(),
                            d,
                            over_i: over_i.clone(),
                            dist: dist.clone(),
                            law: law.clone(),
                            g,
                        });
                    }
                }
            }
        }
    }
    Ok(out)
}

/// The certified upper bound on the miss probability of OJF in a symmetric setting.
struct Ojf {
    /// The smallest certified bound and the bound attaining it (`None`: OJF is not proved
    /// stable, or the classes are not identical).
    hi: Option<(Q, &'static str)>,
    /// Mixture terms `K` and grid step of the workload bound.
    k: Option<usize>,
    step: Option<Q>,
    ms: f64,
}

/// The default grid step of the OJF workload bound, `d / clamp(ceil(500 mu d), 400, 32768)`:
/// about `1/(500 mu)`, with at least 400 cells and at most 32768 (the convolutions cost the
/// square of the number of cells).
fn default_ojf_step(d: &Q, mu: &Q) -> Q {
    let cells = (d * mu * qi(500))
        .ceil()
        .to_integer()
        .to_i64()
        .unwrap_or(i64::MAX)
        .clamp(400, 32768);
    d / qi(cells)
}

/// The smallest of the exact exponential benchmark, the baseline bound `U_0` (with the
/// closed-form tail) and the workload bound `U_K^+` on the grid of step `step` (default
/// `default_ojf_step`), with `K` as in `tdf bracket` (the fewest terms with
/// `rho_F^{K+1} <= 10^-6`, at most 64).
fn ojf_bound(s: &Setting, step: Option<&Q>, bits: u32) -> Result<Ojf, String> {
    let t0 = Instant::now();
    // the target does not enter the bounds
    let model = Model::balanced(&s.id, &s.dist, s.n, &s.rho, &s.d, &qr(1, 100));
    let Ok(h) = Homogeneous::from_model(&model) else {
        return Ok(Ojf {
            hi: None,
            k: None,
            step: None,
            ms: 0.0,
        });
    };
    let step = step
        .cloned()
        .unwrap_or_else(|| default_ojf_step(&s.d, &h.mu));
    let step = &step;
    let mut best: Option<(Q, &'static str)> = None;
    let mut consider = |v: Q, source: &'static str| {
        if best.as_ref().is_none_or(|(b, _)| v < *b) {
            best = Some((v, source));
        }
    };
    if let Ok(Some(b)) = h.exact_exponential_benchmark(bits) {
        consider(b.miss.1.clone(), "benchmark");
    }
    if let Some(tail) = h.tail_closed_form(bits) {
        let floor = h.floor_from_tail(&tail, bits);
        if let Some(u) = Homogeneous::u_simple(&floor, &h.rho_f(bits), bits) {
            consider(u, "baseline");
        }
    }
    let k = h.workload_terms(bits, 1e-6, 64);
    let mut wstep = None;
    if let Some(k) = k {
        let w = h.workload_bound(step, k, bits)?;
        wstep = Some(w.step.clone());
        consider(w.u_k_plus.clone(), "workload");
    }
    Ok(Ojf {
        hi: best,
        k,
        step: wstep,
        ms: t0.elapsed().as_secs_f64() * 1e3,
    })
}

const WITNESS_CSV: [&str; 27] = [
    "id",
    "profile",
    "n",
    "rho",
    "mean",
    "d",
    "d_over_i",
    "step",
    "cells",
    "k",
    "hi",
    "lo",
    "hi_f64",
    "lo_f64",
    "e_rk",
    "r_del",
    "v_add",
    "nmax",
    "skipped",
    "convs",
    "admitted",
    "ms_witness",
    "ojf_hi",
    "ojf_source",
    "ojf_k",
    "ojf_step",
    "ms_ojf",
];

/// The quantile `q` of `v` (sorted in place), 0 for an empty vector.
fn quantile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q).round() as usize]
}

/// Enclose every setting (in parallel) and write the table; returns the summary.
pub fn run(args: &Args) -> Result<Value, String> {
    let t0 = Instant::now();
    fs::create_dir_all(&args.out).map_err(|e| e.to_string())?;
    let all = settings(args)?;
    // Warm the caches of the reconfiguration laws: one task per law and step, with the longest
    // lattice of its settings.
    let mut warm: BTreeMap<String, (Law, Q, usize)> = BTreeMap::new();
    for s in &all {
        let cells = (&s.d / &s.g).floor().to_integer().to_usize();
        let len = cells.ok_or_else(|| format!("{}: lattice too long", s.id))? + 1;
        let e = warm
            .entry(format!("{:?}|{}", s.law, s.g))
            .or_insert_with(|| (s.law.clone(), s.g.clone(), 0));
        e.2 = e.2.max(len);
    }
    let warm: Vec<(Law, Q, usize)> = warm.into_values().collect();
    warm.par_iter().for_each(|(law, g, len)| {
        sw_pre(law, g, *len, &default_h0(law));
    });
    let warm_seconds = t0.elapsed().as_secs_f64();
    eprintln!(
        "witness: {} settings, caches of {} laws warmed in {warm_seconds:.1} s",
        all.len(),
        warm.len()
    );
    let bits = grid_settings(2, None, None, None, false)?.bits;
    let rows: Vec<Result<(Vec<String>, Value), String>> = all
        .par_iter()
        .map(|s| {
            let sys = System::symmetric(s.n, &s.rho, &qi(4), &s.law);
            let opts = Opts {
                g: s.g.clone(),
                tol_trunc: args.tol,
                ..Opts::default()
            };
            let o = compute(&sys, 0, &s.d, &opts);
            // hi_f >= hi: at or below `ojf_above`, OJF cannot change a verdict
            let needed = Q::from_float(o.hi_f).is_none_or(|h| h > args.ojf_above);
            let ojf = if args.no_ojf || !needed {
                None
            } else {
                Some(ojf_bound(s, args.ojf_step.as_ref(), bits)?)
            };
            let ojf_hi = ojf.as_ref().and_then(|j| j.hi.as_ref());
            let source = match (&ojf, ojf_hi) {
                (_, Some((_, src))) => src.to_string(),
                (None, _) if !args.no_ojf => "not_needed".to_string(),
                _ => String::new(),
            };
            let opt = |v: Option<String>| v.unwrap_or_default();
            let csv = vec![
                s.id.clone(),
                s.profile.clone(),
                s.n.to_string(),
                s.rho.to_string(),
                s.mean.to_string(),
                s.d.to_string(),
                opt(s.over_i.as_ref().map(|r| r.to_string())),
                s.g.to_string(),
                o.m_cells.to_string(),
                o.k_cycles.to_string(),
                o.hi.clone(),
                o.lo.clone(),
                format!("{:e}", o.hi_f),
                format!("{:e}", o.lo_f),
                format!("{:.3e}", o.e_rk),
                format!("{:.3e}", o.r_del),
                format!("{:.3e}", o.v_add),
                o.nmax_max.to_string(),
                o.skipped.to_string(),
                o.convs.to_string(),
                o.admitted.to_string(),
                format!("{:.3}", o.seconds * 1e3),
                opt(ojf_hi.map(|(u, _)| format!("{:e}", q_up(u)))),
                source.clone(),
                opt(ojf.as_ref().and_then(|j| j.k).map(|k| k.to_string())),
                opt(ojf.as_ref().and_then(|j| j.step.as_ref()).map(|g| g.to_string())),
                opt(ojf.as_ref().map(|j| format!("{:.3}", j.ms))),
            ];
            let info = json!({
                "ms": o.seconds * 1e3, "admitted": o.admitted, "ojf_ms": ojf.as_ref().map(|j| j.ms),
                "ojf_source": (!source.is_empty()).then_some(source), "hi": o.hi_f, "ojf_hi": ojf_hi.map(|(u, _)| to_f64(u))
            });
            Ok((csv, info))
        })
        .collect();
    let mut csv_rows = Vec::with_capacity(rows.len());
    let mut infos = Vec::with_capacity(rows.len());
    for r in rows {
        let (c, i) = r?;
        csv_rows.push(c);
        infos.push(i);
    }
    write_csv(&args.out.join("witness.csv"), &WITNESS_CSV, &csv_rows)?;
    let col = |k: &str| -> Vec<f64> { infos.iter().filter_map(|i| i[k].as_f64()).collect() };
    let (mut ms, mut ojf_ms) = (col("ms"), col("ojf_ms"));
    let admitted: u64 = infos.iter().filter_map(|i| i["admitted"].as_u64()).sum();
    let mut sources: BTreeMap<String, usize> = BTreeMap::new();
    for i in &infos {
        if let Some(src) = i["ojf_source"].as_str() {
            *sources.entry(src.to_string()).or_default() += 1;
        }
    }
    let ojf_below = infos
        .iter()
        .filter(|i| {
            i["ojf_hi"]
                .as_f64()
                .is_some_and(|u| u < i["hi"].as_f64().unwrap_or(1.0))
        })
        .count();
    let summary = json!({
        "settings": all.len(), "total_seconds": t0.elapsed().as_secs_f64(), "warm_seconds": warm_seconds,
        "witness_ms": {"total": ms.iter().sum::<f64>(), "median": quantile(&mut ms, 0.5), "p90": quantile(&mut ms, 0.9), "max": quantile(&mut ms, 1.0)},
        "ojf_ms": {"total": ojf_ms.iter().sum::<f64>(), "median": quantile(&mut ojf_ms, 0.5), "p90": quantile(&mut ojf_ms, 0.9), "max": quantile(&mut ojf_ms, 1.0)},
        "admitted_total": admitted, "settings_with_admissions": infos.iter().filter(|i| i["admitted"].as_u64() != Some(0)).count(),
        "ojf_sources": sources, "ojf_below_witness": ojf_below,
        "manifest": manifest("witness", json!({
            "profiles": args.profiles, "ns": args.ns, "rhos": args.rhos.iter().map(|q| q.to_string()).collect::<Vec<_>>(),
            "ds": args.ds.iter().map(|q| q.to_string()).collect::<Vec<_>>(), "d_over_i": args.d_over_i.iter().map(|q| q.to_string()).collect::<Vec<_>>(),
            "means": args.means.iter().map(|q| q.to_string()).collect::<Vec<_>>(), "cv2": args.cv2.to_string(),
            "step": args.step.as_ref().map(|q| q.to_string()), "map_step": args.map_step.as_ref().map(|q| q.to_string()),
            "tol": args.tol, "ojf": !args.no_ojf, "ojf_step": args.ojf_step.as_ref().map(|q| q.to_string()),
            "ojf_above": args.ojf_above.to_string()
        }), None)
    });
    write_json(&args.out.join("summary.json"), &summary)?;
    let mut short = summary.clone();
    short.as_object_mut().unwrap().remove("manifest");
    Ok(short)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::common::point_id;

    #[test]
    fn identifiers_are_those_of_the_map_without_the_target() {
        let (rho, eps, d) = (qr(9, 10), qr(1, 100), qi(4));
        let map_id = point_id("det", 4, &rho, &d, &eps);
        assert_eq!(
            setting_id("det", 4, &rho, &d, &qr(1, 2), None),
            strip_target(&map_id)
        );
        assert_eq!(
            setting_id("det", 4, &rho, &d, &qr(1, 2), None),
            "det|N=4|rho=9/10|d=4"
        );
        let id = setting_id("det", 4, &qr(7, 10), &qi(275), &qi(50), Some(&qr(1, 2)));
        assert_eq!(id, "det|N=4|rho=7/10|d=275|tau=50|d/I=1/2");
        assert_eq!(
            strip_target("det|N=4|rho=7/10|d=275|eps=1/100|tau=50|d/I=1/2"),
            id
        );
    }

    #[test]
    fn deadlines_over_the_intervisit_time_match_the_map() {
        // the cases of the map's own test
        let d = deadline_over_i(4, &qr(7, 10), &qi(50), &qr(1, 2), &qr(1, 20)).unwrap();
        assert_eq!(d, qi(275));
        let d = deadline_over_i(2, &qr(1, 20), &qr(1, 2), &qr(1, 100), &qr(1, 10)).unwrap();
        assert_eq!(d, qr(1, 10));
        assert!(deadline_over_i(2, &qi(1), &qr(1, 2), &qi(1), &qr(1, 10)).is_err());
    }
}
