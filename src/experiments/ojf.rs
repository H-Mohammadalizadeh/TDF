//! `tdf ojf`: the reference policy OJF (oldest job first, global FCFS) simulated at symmetric
//! settings against its reference analysis. Every run keeps its sojourn times, so one run serves
//! every threshold (OJF does not read the deadline). Per load and threshold, the pooled tail over
//! all runs, with a whole-run percentile bootstrap, is reported with the exact exponential
//! benchmark, the no-waiting miss probability `m_0` (`floor_b`) and the baseline bound `U_0`
//! (`u_simple`) at that threshold, where closed forms exist; the observed reconfiguration rate and
//! phase fractions are reported with OJF's rate identities. Output: `manifest.json`,
//! `validate.json`, `tails.csv`.

use super::common::{
    grid_settings, manifest, parse_rational, point_id, profile_law, write_csv, write_json,
};
use crate::feasibility::distributions::Model;
use crate::feasibility::rational::{Q, qi, qr, to_f64};
use crate::feasibility::reference::Homogeneous;
use crate::sim::rng::Rng;
use crate::sim::runner::{Run, Window, run as simulate_run, x_grid};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Simulate OJF at symmetric settings against its reference analysis.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// Output directory.
    #[arg(long)]
    pub out: PathBuf,
    /// Number of classes.
    #[arg(long, default_value_t = 2)]
    pub n: usize,
    /// Reconfiguration-time profile (det, exp, gamma, tnorm, lognormal, pareto, pareto_heavy).
    #[arg(long, default_value = "exp")]
    pub profile: String,
    /// Total loads.
    #[arg(long, value_delimiter = ',', default_value = "1/100,1/20,1/5,2/5", value_parser = parse_rational)]
    pub rhos: Vec<Q>,
    /// Deadline in seconds; it sets the follow-up of the runs and the counted misses.
    #[arg(long, default_value = "2", value_parser = parse_rational)]
    pub d: Q,
    /// Sojourn-time thresholds of the reported tails, in seconds.
    #[arg(long, value_delimiter = ',', default_value = "2,3", value_parser = parse_rational)]
    pub thresholds: Vec<Q>,
    /// Replications (runs) per load, with consecutive seeds.
    #[arg(long, default_value_t = 32)]
    pub seeds: usize,
    /// Seed of the first replication.
    #[arg(long, default_value_t = 20271401)]
    pub seed_start: u64,
    /// Warm-up before the observation window, in seconds.
    #[arg(long, default_value_t = 2000.0)]
    pub warmup: f64,
    /// Factor on the observation-window length max(20000, ceil(5000 / lambda_min)) seconds.
    #[arg(long, default_value_t = 1.0)]
    pub measure_factor: f64,
    /// Bootstrap resamples.
    #[arg(long, default_value_t = 2000)]
    pub draws: usize,
    /// Significance level of the two-sided bootstrap intervals (0.05: 95 % intervals).
    #[arg(long, default_value_t = 0.05)]
    pub alpha: f64,
    /// Mean reconfiguration time in seconds.
    #[arg(long, default_value = "1/2", value_parser = parse_rational)]
    pub mean: Q,
    /// Squared coefficient of variation of the reconfiguration time.
    #[arg(long, default_value = "1/3", value_parser = parse_rational)]
    pub cv2: Q,
    /// Fractional bits of the rational enclosures of the analytical references (default 128).
    #[arg(long)]
    pub bits: Option<u32>,
    /// Worker threads (default: all cores).
    #[arg(long)]
    pub threads: Option<usize>,
}

/// Pooled tail estimate over runs with a whole-run percentile bootstrap; `estimate_with_censored`
/// counts every censored job as an exceedance.
struct Tail {
    arrivals: u64,
    exceed: u64,
    censored: u64,
    estimate: Option<f64>,
    estimate_with_censored: Option<f64>,
    per_class: Vec<Option<f64>>,
    interval: (Option<f64>, Option<f64>),
}

/// The pooled fraction of jobs with a completed sojourn time above `t` (censored jobs counted
/// separately), with a percentile interval from `draws` bootstrap resamples of whole runs; the
/// resampling stream is xoshiro256** seeded by `seed`.
fn tail_of(runs: &[Run], n: usize, t: f64, draws: usize, alpha: f64, seed: u64) -> Tail {
    let per_run: Vec<(u64, u64, u64, Vec<(u64, u64)>)> = runs
        .iter()
        .map(|r| {
            let mut a = 0u64;
            let mut e = 0u64;
            let mut c = 0u64;
            let mut classes = Vec::with_capacity(n);
            for q in 0..n {
                let aq = r.classes[q].arrivals;
                let eq = r
                    .sojourns
                    .as_ref()
                    .map(|s| s[q].iter().filter(|x| **x > t).count() as u64)
                    .unwrap_or(0);
                a += aq;
                e += eq;
                c += r.classes[q].censored;
                classes.push((aq, eq));
            }
            (a, e, c, classes)
        })
        .collect();
    let arrivals: u64 = per_run.iter().map(|r| r.0).sum();
    let exceed: u64 = per_run.iter().map(|r| r.1).sum();
    let censored: u64 = per_run.iter().map(|r| r.2).sum();
    let ratio = |num: u64, den: u64| (den > 0).then(|| num as f64 / den as f64);
    let per_class: Vec<Option<f64>> = (0..n)
        .map(|q| {
            let a: u64 = per_run.iter().map(|r| r.3[q].0).sum();
            let e: u64 = per_run.iter().map(|r| r.3[q].1).sum();
            ratio(e, a)
        })
        .collect();
    let interval = if draws > 0 && runs.len() > 1 && arrivals > 0 {
        let mut rng = Rng::xoshiro(seed);
        let m = runs.len();
        let mut vals: Vec<f64> = Vec::with_capacity(draws);
        for _ in 0..draws {
            let mut a = 0u64;
            let mut e = 0u64;
            for _ in 0..m {
                let k = rng.randrange(m as u64) as usize;
                a += per_run[k].0;
                e += per_run[k].1;
            }
            if a > 0 {
                vals.push(e as f64 / a as f64);
            }
        }
        vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let pick = |p: f64| {
            vals.get(((vals.len() as f64 - 1.0) * p).round() as usize)
                .copied()
        };
        (pick(alpha / 2.0), pick(1.0 - alpha / 2.0))
    } else {
        (None, None)
    };
    Tail {
        arrivals,
        exceed,
        censored,
        estimate: ratio(exceed, arrivals),
        estimate_with_censored: ratio(exceed + censored, arrivals),
        per_class,
        interval,
    }
}

/// Simulate every load and write the tables; returns a summary. The bootstrap of the `k`-th
/// threshold (from zero) uses the seed `20271500 + k` at every load.
pub fn run(args: &Args) -> Result<Value, String> {
    let out = &args.out;
    let (n, profile_name, d) = (args.n, args.profile.as_str(), &args.d);
    let (warmup, measure_factor, draws, alpha) =
        (args.warmup, args.measure_factor, args.draws, args.alpha);
    let seeds: Vec<u64> = (0..args.seeds as u64)
        .map(|k| args.seed_start + k)
        .collect();
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    let law = profile_law(profile_name, &args.mean, &args.cv2)?;
    let settings = grid_settings(n, None, None, args.bits, false)?;
    let xs = x_grid();
    write_json(
        &out.join("manifest.json"),
        &manifest(
            "ojf",
            json!({"n": n, "profile": profile_name, "rhos": args.rhos.iter().map(|q| q.to_string()).collect::<Vec<_>>(), "d": d.to_string(),
                   "thresholds": args.thresholds.iter().map(|q| q.to_string()).collect::<Vec<_>>(), "seeds": seeds, "warmup": warmup,
                   "measure_factor": measure_factor, "draws": draws, "alpha": alpha, "mean": args.mean.to_string(), "cv2": args.cv2.to_string(),
                   "policy": "ojf", "records": "every seed"}),
            Some(&settings),
        ),
    )?;
    let policy = json!({"name": "ojf", "params": {}});
    let mut rows = Vec::new();
    let mut csv_rows = Vec::new();
    for rho in &args.rhos {
        let eps = qr(1, 100);
        let id = point_id(profile_name, n, rho, d, &eps);
        let model = Model::balanced(&id, &law, n, rho, d, &eps);
        let h = Homogeneous::from_model(&model)?;
        let window = Window::for_model(&model, warmup, measure_factor, None);
        let t = Instant::now();
        let runs: Vec<Run> = seeds
            .par_iter()
            .map(|&s| simulate_run(&model, &policy, s, true, &window, &xs))
            .collect();
        let sim_seconds = t.elapsed().as_secs_f64();
        let completed = runs.iter().filter(|r| r.status == "completed").count();
        let growth = runs.iter().filter(|r| r.growth).count();
        let mean_of =
            |f: &dyn Fn(&Run) -> f64| runs.iter().map(f).sum::<f64>() / runs.len().max(1) as f64;
        let rates = h.reference_rates(settings.bits);
        let observed = json!({
            "switch_rate_mean": mean_of(&|r| r.switch_rate), "phase_service_mean": mean_of(&|r| r.phase.0),
            "phase_switch_mean": mean_of(&|r| r.phase.1), "phase_idle_mean": mean_of(&|r| r.phase.2),
            "expected": {"switch_rate": to_f64(&rates.nu_total), "phase_service": to_f64(&h.rho()),
                         "phase_switch": [to_f64(&(&rates.nu_total * &law.mean(settings.bits).0)), to_f64(&(&rates.nu_total * &law.mean(settings.bits).1))],
                         "phase_idle": [to_f64(&(qi(1) - &rates.rho_f.1)), to_f64(&(qi(1) - &rates.rho_f.0))]}
        });
        let mut tails = Vec::new();
        for (k, thr) in args.thresholds.iter().enumerate() {
            let tf = to_f64(thr);
            let tail = tail_of(&runs, n, tf, draws, alpha, 20_271_500 + k as u64);
            // The analytical references at this threshold: the same setting with deadline `thr`.
            let ht = Homogeneous {
                d: thr.clone(),
                ..h.clone()
            };
            let floor = ht
                .tail_closed_form(settings.bits)
                .map(|tl| ht.floor_from_tail(&tl, settings.bits));
            let rho_f = ht.rho_f(settings.bits);
            let u_simple = floor
                .as_ref()
                .and_then(|b| Homogeneous::u_simple(b, &rho_f, settings.bits));
            let exact = ht.exact_exponential_benchmark(settings.bits).ok().flatten();
            let inside_exact = match (&exact, tail.estimate) {
                (Some(b), Some(e)) => Some(
                    to_f64(&b.miss.0) <= e && e <= to_f64(&b.miss.1)
                        || tail.interval.0.is_some_and(|lo| lo <= to_f64(&b.miss.1))
                            && tail.interval.1.is_some_and(|hi| hi >= to_f64(&b.miss.0)),
                ),
                _ => None,
            };
            csv_rows.push(vec![
                id.clone(),
                rho.to_string(),
                to_f64(rho).to_string(),
                thr.to_string(),
                tail.arrivals.to_string(),
                tail.exceed.to_string(),
                tail.censored.to_string(),
                tail.estimate.map(|v| v.to_string()).unwrap_or_default(),
                tail.interval.0.map(|v| v.to_string()).unwrap_or_default(),
                tail.interval.1.map(|v| v.to_string()).unwrap_or_default(),
                exact
                    .as_ref()
                    .map(|b| to_f64(&b.miss.0).to_string())
                    .unwrap_or_default(),
                exact
                    .as_ref()
                    .map(|b| to_f64(&b.miss.1).to_string())
                    .unwrap_or_default(),
                floor
                    .as_ref()
                    .map(|b| to_f64(&b.0).to_string())
                    .unwrap_or_default(),
                u_simple
                    .as_ref()
                    .map(|u| to_f64(u).to_string())
                    .unwrap_or_default(),
                inside_exact.map(|b| b.to_string()).unwrap_or_default(),
            ]);
            tails.push(json!({
                "threshold": thr.to_string(), "threshold_f64": tf, "arrivals": tail.arrivals, "exceed": tail.exceed, "censored": tail.censored,
                "estimate": tail.estimate, "estimate_with_censored": tail.estimate_with_censored, "per_class": tail.per_class,
                "interval": [tail.interval.0, tail.interval.1], "interval_note": "whole-run percentile bootstrap of the pooled ratio; a finite-window statement, not a stationary bound",
                "exact_benchmark": exact.as_ref().map(|b| b.to_json()).unwrap_or(Value::Null),
                "floor_b": floor.as_ref().map(|b| json!([to_f64(&b.0), to_f64(&b.1)])).unwrap_or(Value::Null),
                "u_simple": u_simple.as_ref().map(to_f64), "estimate_consistent_with_exact": inside_exact
            }));
        }
        let per_run: Vec<Value> = runs
            .iter()
            .map(|r| {
                json!({"seed": r.seed, "status": r.status, "growth_flag": r.growth, "abort": r.abort_reason,
                       "arrivals": r.classes.iter().map(|c| c.arrivals).collect::<Vec<_>>(), "misses_at_d": r.classes.iter().map(|c| c.misses).collect::<Vec<_>>(),
                       "censored": r.classes.iter().map(|c| c.censored).collect::<Vec<_>>(), "unresolved": r.classes.iter().map(|c| c.unresolved).collect::<Vec<_>>(),
                       "switch_rate": r.switch_rate, "phase": [r.phase.0, r.phase.1, r.phase.2],
                       "runtime_seconds": r.runtime_seconds, "events": r.events})
            })
            .collect();
        rows.push(json!({
            "id": id, "rho": rho.to_string(), "rho_f64": to_f64(rho), "d": d.to_string(), "homogeneous": h.to_json(),
            "window": window.to_json(), "runs": runs.len(), "completed": completed, "growth_flags": growth,
            "reference_rates": rates.to_json(), "observed": observed, "tails": tails, "per_run": per_run, "simulation_seconds": sim_seconds
        }));
    }
    write_csv(
        &out.join("tails.csv"),
        &[
            "id",
            "rho",
            "rho_f64",
            "threshold",
            "arrivals",
            "exceed",
            "censored",
            "estimate",
            "boot_lo",
            "boot_hi",
            "exact_lo",
            "exact_hi",
            "floor_b_lo",
            "u_simple",
            "estimate_consistent_with_exact",
        ],
        &csv_rows,
    )?;
    let summary = json!({"settings": rows.len(), "runs_per_setting": seeds.len(), "total_seconds": t0.elapsed().as_secs_f64()});
    write_json(
        &out.join("validate.json"),
        &json!({"summary": summary, "rows": rows}),
    )?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_runs_the_tail_is_empty() {
        let t = tail_of(&[], 2, 1.0, 10, 0.05, 1);
        assert_eq!(t.arrivals, 0);
        assert!(t.estimate.is_none());
        assert!(t.interval.0.is_none());
    }
}
