//! `tdf screen`: EC and OJF simulated against the certified bounds. An *environment*
//! ([`Environment`]) is one profile at one `N:rho:d` setting; every policy runs there with the
//! same seeds (paired replications), and a whole-run bootstrap gives the intervals of the pooled
//! miss probabilities and delay tails (see [`simulate_screen`]). Output: `manifest.json`,
//! `screen.json`, `screen.csv` (one row per environment and policy), `runs.csv` (one row per run
//! and class), `tails.csv` (pooled tails with bands) and `records/` (sojourn times and rebuilt
//! occupations of the record runs).

use super::common::{
    file_stem, grid_settings, parse_rational, point_id, profile_law, write_json, write_lines,
};
use crate::feasibility::Evaluation;
use crate::feasibility::coefficients::PhaseCoefficients;
use crate::feasibility::distributions::Model;
use crate::feasibility::evaluate;
use crate::feasibility::grid::GridSettings;
use crate::feasibility::model::DiscreteModel;
use crate::feasibility::rational::{Q, parse_q, to_f64};
use crate::feasibility::relaxation::Level;
use crate::feasibility::windows::{WindowRow, default_subsets, default_windows, window_rows};
use crate::sim::rng::Rng;
use crate::sim::runner::{Run, Window, run as simulate_run, x_grid};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Simulate EC or OJF against the certified bounds.
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
    /// Settings as `N:rho:d` triples (number of classes, total load, deadline in seconds).
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "2:1/20:2,2:7/10:3,4:1/2:4,8:7/10:6"
    )]
    pub settings: Vec<String>,
    /// Policies to simulate: ec (exhaustive cyclic polling), ojf (oldest job first).
    #[arg(long, value_delimiter = ',', default_value = "ec")]
    pub policies: Vec<String>,
    /// Replications (runs) per profile, setting and policy, with consecutive seeds.
    #[arg(long, default_value_t = 32)]
    pub seeds: usize,
    /// Seed of the first replication.
    #[arg(long, default_value_t = 20270901)]
    pub seed_start: u64,
    /// Number of leading replications whose runs keep complete event records.
    #[arg(long, default_value_t = 2)]
    pub record_seeds: usize,
    /// Bootstrap resamples.
    #[arg(long, default_value_t = 10_000)]
    pub draws: usize,
    /// Significance level of the two-sided bootstrap intervals (0.05: 95 % intervals).
    #[arg(long, default_value_t = 0.05)]
    pub alpha: f64,
    /// Wall-clock limit per run in seconds; a run that reaches it is aborted (default: none).
    #[arg(long)]
    pub max_wall_seconds: Option<f64>,
    /// Warm-up before the observation window, in seconds.
    #[arg(long, default_value_t = 2000.0)]
    pub warmup: f64,
    /// Factor on the observation-window length max(20000, ceil(5000 / lambda_min)) seconds.
    #[arg(long, default_value_t = 1.0)]
    pub measure_factor: f64,
    /// Mean reconfiguration time in seconds.
    #[arg(long, default_value = "1/2", value_parser = parse_rational)]
    pub mean: Q,
    /// Squared coefficient of variation of the reconfiguration time.
    #[arg(long, default_value = "1/3", value_parser = parse_rational)]
    pub cv2: Q,
    /// Target miss probability of every class.
    #[arg(long, default_value = "1/100", value_parser = parse_rational)]
    pub eps: Q,
    /// Service-time distribution of the simulated network: `exponential` (default),
    /// `deterministic` or `lognormal:<cv>`; the last two lie outside the model, where the
    /// certified bounds serve only as a reference.
    #[arg(long)]
    pub service: Option<String>,
    /// Mean geometric batch size of the arrivals of every class, outside the model (default:
    /// single arrivals).
    #[arg(long)]
    pub batch_mean: Option<String>,
    /// Grid step of the certificates in seconds (default 1/2000 up to 4 classes, 1/1000 up to 8,
    /// 1/500 beyond).
    #[arg(long)]
    pub step: Option<String>,
    /// Refinement sweeps of the upper bound on the completion probability (default 3).
    #[arg(long)]
    pub sweeps: Option<usize>,
    /// Directory of cached phase coefficients, read when present and written otherwise
    /// (default: no cache).
    #[arg(long)]
    pub cache: Option<PathBuf>,
    /// Worker threads (default: all cores).
    #[arg(long)]
    pub threads: Option<usize>,
}

/// The policies of the screen.
fn known_policies() -> Vec<(String, Value)> {
    vec![
        ("ec".into(), json!({"name": "ec", "params": {}})),
        ("ojf".into(), json!({"name": "ojf", "params": {}})),
    ]
}

fn select_policies(labels: &[String]) -> Result<Vec<(String, Value)>, String> {
    let all = known_policies();
    labels
        .iter()
        .map(|l| {
            all.iter()
                .find(|(label, _)| label == l)
                .cloned()
                .ok_or_else(|| format!("unknown policy {l}"))
        })
        .collect()
}

/// One environment of the screen: a profile at one `N:rho:d` setting.
#[derive(Clone, Debug)]
pub struct Environment {
    pub id: String,
    pub profile: String,
    pub model: Model,
    pub window: Window,
    /// Grid settings of the certificate: the per-`N` defaults of `tdf map` unless overridden, so
    /// that the screen reuses the cached coefficients of the evaluation grid and reports the same
    /// bounds.
    pub settings: GridSettings,
}

/// Checks a `--service` specification: `exponential`, `deterministic` or `lognormal:<cv>` with a
/// positive coefficient of variation.
fn check_service(spec: &str) -> Result<(), String> {
    let valid = match spec.strip_prefix("lognormal:") {
        Some(cv) => cv.parse::<f64>().is_ok_and(|v| v.is_finite() && v > 0.0),
        None => spec == "exponential" || spec == "deterministic",
    };
    if valid {
        Ok(())
    } else {
        Err(format!(
            "--service {spec:?}: expected exponential, deterministic or lognormal:<cv> with cv > 0"
        ))
    }
}

/// Set up the environments, write the manifest and run the screen.
pub fn run(args: &Args) -> Result<Value, String> {
    if let Some(spec) = &args.service {
        check_service(spec)?;
    }
    let out = &args.out;
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let (step, sweeps) = (args.step.as_deref(), args.sweeps);
    let mut envs = Vec::new();
    for t in &args.settings {
        let parts: Vec<&str> = t.split(':').collect();
        if parts.len() != 3 {
            return Err(format!("setting {t:?} is not N:rho:d"));
        }
        let n: usize = parts[0].parse().map_err(|_| format!("bad N in {t}"))?;
        let rho = parse_q(parts[1])?;
        let d = parse_q(parts[2])?;
        for p in &args.profiles {
            let law = profile_law(p, &args.mean, &args.cv2)?;
            let id = point_id(p, n, &rho, &d, &args.eps);
            let mut model = Model::balanced(&id, &law, n, &rho, &d, &args.eps);
            model.service = args.service.clone();
            model.batch_mean = args.batch_mean.as_deref().map(parse_q).transpose()?;
            let window = Window::for_model(
                &model,
                args.warmup,
                args.measure_factor,
                args.max_wall_seconds,
            );
            envs.push(Environment {
                id,
                profile: p.clone(),
                settings: grid_settings(model.n, step, sweeps, None, false)?,
                model,
                window,
            });
        }
    }
    let policies = select_policies(&args.policies)?;
    let n_max = envs.iter().map(|e| e.model.n).max().unwrap_or(2);
    let settings = grid_settings(n_max, step, sweeps, None, false)?;
    let seeds: Vec<u64> = (0..args.seeds as u64)
        .map(|k| args.seed_start + k)
        .collect();
    let (service, batch_mean) = (args.service.as_deref(), args.batch_mean.as_deref());
    let manifest = json!({
        "environments": envs.iter().map(|e| json!({"id": e.id, "profile": e.profile, "window": e.window.to_json()})).collect::<Vec<_>>(),
        "policies": policies.iter().map(|(l, _)| l.clone()).collect::<Vec<_>>(), "seeds": seeds, "record_seeds": args.record_seeds,
        "draws": args.draws, "alpha": args.alpha, "max_wall_seconds": args.max_wall_seconds, "warmup": args.warmup, "measure_factor": args.measure_factor,
        "mean": args.mean.to_string(), "cv2": args.cv2.to_string(), "eps": args.eps.to_string(), "grid": settings.to_json(),
        "service": service, "batch_mean": batch_mean, "outside_model": service.is_some_and(|x| x != "exponential") || batch_mean.is_some()
    });
    write_json(&out.join("manifest.json"), &manifest)?;
    simulate_screen(
        &envs,
        &policies,
        &seeds,
        args.record_seeds,
        &settings,
        args.cache.as_deref(),
        args.draws,
        args.alpha,
        out,
    )
}

/// Pooled per-class miss rates, the worst class, and the pooled delay tails.
#[derive(Clone, Debug)]
pub struct Pooled {
    pub arrivals: Vec<u64>,
    pub misses: Vec<u64>,
    /// Pooled miss rate per class; `None` without arrivals.
    pub rate: Vec<Option<f64>>,
    pub worst: Option<f64>,
    pub worst_class: Option<usize>,
    /// `ccdf[i][k]`: pooled fraction of class-`i` jobs with sojourn time `D_i > x_k d_i`.
    pub ccdf: Vec<Vec<Option<f64>>>,
    /// `max_i ccdf[i][k]` (the worst-class envelope, not the tail of a maximum).
    pub envelope: Vec<Option<f64>>,
}

/// Pool the counts of `runs` (`n` classes, `nx` tail grid points).
pub fn pooled_of(runs: &[&Run], n: usize, nx: usize) -> Pooled {
    let mut arrivals = vec![0u64; n];
    let mut misses = vec![0u64; n];
    let mut exceed = vec![vec![0u64; nx]; n];
    for r in runs {
        for i in 0..n {
            arrivals[i] += r.classes[i].arrivals;
            misses[i] += r.classes[i].misses;
            for k in 0..nx {
                exceed[i][k] += r.classes[i].exceed[k];
            }
        }
    }
    let rate: Vec<Option<f64>> = (0..n)
        .map(|i| (arrivals[i] > 0).then(|| misses[i] as f64 / arrivals[i] as f64))
        .collect();
    let (mut worst, mut worst_class) = (None, None);
    for (i, r) in rate.iter().enumerate() {
        if let Some(v) = r
            && worst.is_none_or(|w: f64| *v > w)
        {
            worst = Some(*v);
            worst_class = Some(i);
        }
    }
    let ccdf: Vec<Vec<Option<f64>>> = (0..n)
        .map(|i| {
            (0..nx)
                .map(|k| (arrivals[i] > 0).then(|| exceed[i][k] as f64 / arrivals[i] as f64))
                .collect()
        })
        .collect();
    let envelope: Vec<Option<f64>> = (0..nx)
        .map(|k| {
            (0..n)
                .filter_map(|i| ccdf[i][k])
                .fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))))
        })
        .collect();
    Pooled {
        arrivals,
        misses,
        rate,
        worst,
        worst_class,
        ccdf,
        envelope,
    }
}

/// The element of `sorted` at the rounded rank `(n - 1) p`.
fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let k = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    Some(sorted[k.min(sorted.len() - 1)])
}

/// Two-sided `1 - alpha` percentile interval of the finite values.
fn interval(values: &mut Vec<f64>, alpha: f64) -> (Option<f64>, Option<f64>) {
    values.retain(|v| v.is_finite());
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (
        percentile(values, alpha / 2.0),
        percentile(values, 1.0 - alpha / 2.0),
    )
}

/// Bootstrap of whole runs with the same resampled run indices for every
/// policy and class (the seeds are paired); every pooled quantity is
/// recomputed in each resample.
#[derive(Clone, Debug)]
pub struct Bootstrap {
    pub draws: usize,
    pub seed: u64,
    pub alpha: f64,
    /// Number of tests over which the joint level is spread (classes times policies).
    pub joint_tests: usize,
    /// Per policy: pointwise interval of the worst-class rate.
    pub worst: Vec<(Option<f64>, Option<f64>)>,
    /// Per policy: the same interval at the Bonferroni level `alpha / joint_tests`.
    pub worst_joint: Vec<(Option<f64>, Option<f64>)>,
    /// Per policy, per class: pointwise interval of the pooled rate.
    pub rate: Vec<Vec<(Option<f64>, Option<f64>)>>,
    /// Per policy, per grid point: pointwise band of the worst-class envelope.
    pub envelope: Vec<Vec<(Option<f64>, Option<f64>)>>,
    /// Per policy, per class, per grid point: pointwise band of the pooled tail.
    pub ccdf: Vec<Vec<Vec<(Option<f64>, Option<f64>)>>>,
    /// Per policy: interval of the paired difference of its worst-class rate from that of the
    /// first policy.
    pub worst_vs_first: Vec<(Option<f64>, Option<f64>)>,
    /// Classes with zero observed misses per policy (their intervals are degenerate).
    pub zero_observed: Vec<Vec<usize>>,
}

/// The paired whole-run bootstrap of `runs_by_policy` (one run list per
/// policy, all of the same length), with a xoshiro256** stream seeded by
/// `seed`: each resample draws one vector of run indices and applies it to
/// every policy.
pub fn bootstrap(
    runs_by_policy: &[Vec<Run>],
    n: usize,
    nx: usize,
    draws: usize,
    alpha: f64,
    seed: u64,
) -> Bootstrap {
    let p = runs_by_policy.len();
    let m = runs_by_policy[0].len();
    assert!(runs_by_policy.iter().all(|r| r.len() == m), "paired runs");
    let mut rng = Rng::xoshiro(seed);
    let mut worst: Vec<Vec<f64>> = vec![Vec::with_capacity(draws); p];
    let mut rate: Vec<Vec<Vec<f64>>> = vec![vec![Vec::with_capacity(draws); n]; p];
    let mut envelope: Vec<Vec<Vec<f64>>> = vec![vec![Vec::with_capacity(draws); nx]; p];
    let mut ccdf: Vec<Vec<Vec<Vec<f64>>>> = vec![vec![vec![Vec::with_capacity(draws); nx]; n]; p];
    let mut diff: Vec<Vec<f64>> = vec![Vec::with_capacity(draws); p];
    let mut idx = vec![0usize; m];
    for _ in 0..draws {
        for slot in idx.iter_mut() {
            *slot = rng.randrange(m as u64) as usize;
        }
        let mut worst_first = None;
        for q in 0..p {
            let chosen: Vec<&Run> = idx.iter().map(|&k| &runs_by_policy[q][k]).collect();
            let pooled = pooled_of(&chosen, n, nx);
            worst[q].push(pooled.worst.unwrap_or(f64::NAN));
            for i in 0..n {
                rate[q][i].push(pooled.rate[i].unwrap_or(f64::NAN));
                for k in 0..nx {
                    ccdf[q][i][k].push(pooled.ccdf[i][k].unwrap_or(f64::NAN));
                }
            }
            for k in 0..nx {
                envelope[q][k].push(pooled.envelope[k].unwrap_or(f64::NAN));
            }
            if q == 0 {
                worst_first = pooled.worst;
            }
            diff[q].push(match (pooled.worst, worst_first) {
                (Some(a), Some(b)) => a - b,
                _ => f64::NAN,
            });
        }
    }
    let joint_tests = (n * p).max(1);
    let joint_alpha = alpha / joint_tests as f64;
    let zero_observed: Vec<Vec<usize>> = runs_by_policy
        .iter()
        .map(|runs| {
            (0..n)
                .filter(|&i| runs.iter().map(|r| r.classes[i].misses).sum::<u64>() == 0)
                .collect()
        })
        .collect();
    Bootstrap {
        draws,
        seed,
        alpha,
        joint_tests,
        worst: worst.iter_mut().map(|v| interval(v, alpha)).collect(),
        worst_joint: worst.iter_mut().map(|v| interval(v, joint_alpha)).collect(),
        rate: rate
            .iter_mut()
            .map(|per| per.iter_mut().map(|v| interval(v, alpha)).collect())
            .collect(),
        envelope: envelope
            .iter_mut()
            .map(|per| per.iter_mut().map(|v| interval(v, alpha)).collect())
            .collect(),
        ccdf: ccdf
            .iter_mut()
            .map(|per| {
                per.iter_mut()
                    .map(|cls| cls.iter_mut().map(|v| interval(v, alpha)).collect())
                    .collect()
            })
            .collect(),
        worst_vs_first: diff.iter_mut().map(|v| interval(v, alpha)).collect(),
        zero_observed,
    }
}

/// The consistency check of the record runs. The phase occupations and reconfiguration rates,
/// averaged over the completed runs with records, and their pooled miss rates are checked
/// against the phase constraint (evaluated with the lower and with the upper coefficients), the
/// window constraint and the time budget; residuals are also given in standard errors of the
/// miss rates.
pub fn consistency(
    model: &DiscreteModel,
    coef_lo: &PhaseCoefficients,
    coef_hi: &PhaseCoefficients,
    windows: &[WindowRow],
    runs: &[Run],
) -> Value {
    let n = model.n;
    let recs: Vec<&Run> = runs
        .iter()
        .filter(|r| r.status == "completed" && r.reconstruction.is_some())
        .collect();
    if recs.is_empty() {
        return json!({"record_runs": 0});
    }
    let k = recs.len() as f64;
    let mean = |f: &dyn Fn(&crate::sim::runner::Reconstruction) -> f64| {
        recs.iter()
            .map(|r| f(r.reconstruction.as_ref().unwrap()))
            .sum::<f64>()
            / k
    };
    let rho: Vec<f64> = (0..n).map(|j| mean(&|r| r.rho[j])).collect();
    let z: Vec<f64> = (0..n).map(|j| mean(&|r| r.z[j])).collect();
    let nu: Vec<Vec<f64>> = (0..n)
        .map(|u| (0..n).map(|v| mean(&|r| r.nu[u][v])).collect())
        .collect();
    let switch_fraction = mean(&|r| r.switch_time_fraction);
    let mut rate = vec![None; n];
    let mut se = vec![0.0; n];
    for i in 0..n {
        let arrivals: u64 = recs.iter().map(|r| r.classes[i].arrivals).sum();
        let violations: u64 = recs.iter().map(|r| r.classes[i].misses).sum();
        if arrivals > 0 {
            let p = violations as f64 / arrivals as f64;
            rate[i] = Some(p);
            se[i] = (p * (1.0 - p) / arrivals as f64).sqrt();
        }
    }
    let nu_total: f64 = model.edges().iter().map(|(u, v)| nu[*u][*v]).sum();
    let mean_tau_switch: f64 = model
        .edges()
        .iter()
        .map(|(u, v)| nu[*u][*v] * to_f64(&coef_lo.mean_tau[*u][*v]))
        .sum();
    let phase_sum = rho.iter().sum::<f64>() + switch_fraction + z.iter().sum::<f64>();
    let time_accounting = rho.iter().sum::<f64>() + mean_tau_switch + z.iter().sum::<f64>();
    let mut phase_rows = Vec::new();
    for i in 0..n {
        if !model.active(i) {
            continue;
        }
        let rhs = |c: &PhaseCoefficients| {
            let mut s = 0.0;
            for j in 0..n {
                s += rho[j] * to_f64(&c.a[j][i]) + z[j] * to_f64(&c.b[j][i]);
            }
            for (u, v) in model.edges() {
                s += nu[u][v] * to_f64(&c.c[u][v][i]);
            }
            s
        };
        let (lo, hi) = (rhs(coef_lo), rhs(coef_hi));
        let residual = rate[i].map(|r| r - lo);
        phase_rows.push(json!({"queue": i, "observed_rate": rate[i], "se": se[i], "bound_lower_coefficients": lo, "bound_upper_coefficients": hi,
            "residual": residual, "residual_in_se": residual.map(|r| if se[i] > 0.0 { r / se[i] } else { f64::INFINITY }),
            "violated_beyond_3se": residual.is_some_and(|r| r < -3.0 * se[i])}));
    }
    let mut window_worst: Option<Value> = None;
    for w in windows {
        let lhs_rate: f64 = w
            .subset
            .iter()
            .filter_map(|&i| rate[i].map(|r| to_f64(&w.lam_a[i]) * r))
            .sum();
        let lhs = to_f64(&w.h) * nu_total + lhs_rate;
        let se_w: f64 = w
            .subset
            .iter()
            .map(|&i| (to_f64(&w.lam_a[i]) * se[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        let residual = lhs - to_f64(&w.rhs);
        let in_se = if se_w > 0.0 {
            residual / se_w
        } else {
            f64::INFINITY
        };
        let row = json!({"h": w.h.to_string(), "subset": w.subset, "lhs": lhs, "rhs": to_f64(&w.rhs), "residual": residual, "residual_in_se": in_se});
        if window_worst
            .as_ref()
            .is_none_or(|b| in_se < b["residual_in_se"].as_f64().unwrap_or(f64::INFINITY))
        {
            window_worst = Some(row);
        }
    }
    json!({
        "record_runs": recs.len(), "rho": rho, "z": z, "nu": nu, "nu_total": nu_total, "switch_time_fraction": switch_fraction,
        "phase_sum": phase_sum, "time_accounting_sum": time_accounting,
        "phase_bounds": phase_rows,
        "any_phase_bound_violated_beyond_3se": phase_rows.iter().any(|r| r["violated_beyond_3se"] == true),
        "window_rows": windows.len(), "window_worst": window_worst
    })
}

fn opt(v: Option<f64>) -> Value {
    v.map(|x| json!(x)).unwrap_or(Value::Null)
}

fn pair(v: &(Option<f64>, Option<f64>)) -> Value {
    json!([opt(v.0), opt(v.1)])
}

/// Run every policy on every environment with paired seeds, then pool, bootstrap and check the
/// record runs; write the tables into `out` and return a summary.
///
/// Per class, the misses of all runs are pooled, and exceedances of the sojourn time are counted
/// on a grid of multiples of the deadline for the delay tails. The intervals come from a
/// bootstrap over whole runs that resamples the same run indices for every policy and class, so
/// the pairing is preserved; its seed is `20260922 + e` for the environment index `e` in this
/// invocation. The runs of the first `record_seeds` seeds keep complete records, from which the
/// phase occupations and reconfiguration rates are rebuilt and checked against the necessary
/// conditions ([`consistency`]). Completed, censored, unresolved and aborted runs are counted
/// separately, and undefined rates are written as null, never as zero.
#[allow(clippy::too_many_arguments)]
pub fn simulate_screen(
    envs: &[Environment],
    policies: &[(String, Value)],
    seeds: &[u64],
    record_seeds: usize,
    settings: &GridSettings,
    cache: Option<&Path>,
    draws: usize,
    alpha: f64,
    out: &Path,
) -> Result<Value, String> {
    let t0 = Instant::now();
    let xs = x_grid();
    let nx = xs.len();
    fs::create_dir_all(out.join("records")).map_err(|e| e.to_string())?;
    // Certify every environment before simulating (coefficients from the cache when present).
    let certs: Vec<Result<Evaluation, String>> = envs
        .iter()
        .map(|e| evaluate(&e.model, &e.settings, &Level::all(), cache))
        .collect();
    let mut cert_ok = Vec::new();
    for c in certs {
        cert_ok.push(c?);
    }
    let jobs: Vec<(usize, usize, usize)> = (0..envs.len())
        .flat_map(|e| {
            (0..policies.len()).flat_map(move |q| (0..seeds.len()).map(move |k| (e, q, k)))
        })
        .collect();
    let total = jobs.len();
    eprintln!(
        "screen: {} settings x {} policies x {} seeds = {total} runs",
        envs.len(),
        policies.len(),
        seeds.len()
    );
    let done = AtomicUsize::new(0);
    let results: Vec<((usize, usize, usize), Run)> = jobs
        .par_iter()
        .map(|&(e, q, k)| {
            let env = &envs[e];
            let (_, policy) = &policies[q];
            let r = simulate_run(
                &env.model,
                policy,
                seeds[k],
                k < record_seeds,
                &env.window,
                &xs,
            );
            let c = done.fetch_add(1, Ordering::SeqCst) + 1;
            if c.is_multiple_of(50) || c == total {
                eprintln!(
                    "screen: {c}/{total} runs ({:.0} s)",
                    t0.elapsed().as_secs_f64()
                );
            }
            ((e, q, k), r)
        })
        .collect();
    let mut by_env: Vec<Vec<Vec<Option<Run>>>> = (0..envs.len())
        .map(|_| {
            (0..policies.len())
                .map(|_| (0..seeds.len()).map(|_| None).collect())
                .collect()
        })
        .collect();
    for ((e, q, k), r) in results {
        by_env[e][q][k] = Some(r);
    }
    let by_env: Vec<Vec<Vec<Run>>> = by_env
        .into_iter()
        .map(|per| {
            per.into_iter()
                .map(|runs| {
                    runs.into_iter()
                        .map(|r| r.expect("every run present"))
                        .collect()
                })
                .collect()
        })
        .collect();
    let sim_seconds = t0.elapsed().as_secs_f64();
    // Analysis per environment.
    let analyses: Vec<Value> = by_env
        .par_iter()
        .enumerate()
        .map(|(e, runs_by_policy)| {
            let env = &envs[e];
            let n = env.model.n;
            let ev = &cert_ok[e];
            let shell = env.model.lp_shell(env.settings.bits);
            let windows: Vec<WindowRow> = window_rows(&shell, &default_windows(&shell), &default_subsets(&shell), env.settings.bits);
            let boot = bootstrap(runs_by_policy, n, nx, draws, alpha, 20260922 + e as u64);
            let full = ev.evaluation.level(Level::Full);
            let lb = full.lower_bound_max();
            let mut per_policy = Vec::new();
            for (q, (label, _)) in policies.iter().enumerate() {
                let runs = &runs_by_policy[q];
                let all: Vec<&Run> = runs.iter().collect();
                let pooled = pooled_of(&all, n, nx);
                let cons = consistency(&shell, &ev.lower, &ev.upper, &windows, runs);
                let statuses = runs.iter().fold(BTreeMap::<String, usize>::new(), |mut m, r| {
                    *m.entry(r.status.clone()).or_default() += 1;
                    m
                });
                let aborts: Vec<Value> = runs
                    .iter()
                    .filter(|r| r.status == "aborted")
                    .map(|r| json!({"seed": r.seed, "reason": r.abort_reason}))
                    .collect();
                // Record runs: sojourn times and rebuilt occupations.
                for r in runs.iter().filter(|r| r.sojourns.is_some()) {
                    let name = format!("{}__{}__{}.json", file_stem(&env.id), label, r.seed);
                    let rec = json!({
                        "environment": env.id, "policy": label, "seed": r.seed, "status": r.status,
                        "completed_sojourns": r.sojourns, "classes": r.classes.iter().map(|c| json!({"arrivals": c.arrivals, "misses": c.misses, "unresolved": c.unresolved, "censored": c.censored})).collect::<Vec<_>>(),
                        "reconstruction": r.reconstruction.as_ref().map(|rc| json!({"rho": rc.rho, "z": rc.z, "nu": rc.nu, "switch_time_fraction": rc.switch_time_fraction, "window": rc.window})).unwrap_or(Value::Null)
                    });
                    let _ = fs::write(out.join("records").join(name), serde_json::to_string(&rec).unwrap());
                }
                let decisions: u64 = runs.iter().map(|r| r.decisions).sum();
                let decision_seconds: f64 = runs.iter().map(|r| r.decision_seconds).sum();
                per_policy.push(json!({
                    "policy": label, "runs": runs.len(),
                    "statuses": statuses, "growth_flags": runs.iter().filter(|r| r.growth).count(), "aborts": aborts,
                    "unresolved_jobs": runs.iter().map(|r| r.classes.iter().map(|c| c.unresolved).sum::<u64>()).sum::<u64>(),
                    "censored_jobs": runs.iter().map(|r| r.classes.iter().map(|c| c.censored).sum::<u64>()).sum::<u64>(),
                    "pooled": {"arrivals": pooled.arrivals, "misses": pooled.misses, "rate": pooled.rate.iter().map(|v| opt(*v)).collect::<Vec<_>>(),
                               "worst": opt(pooled.worst), "worst_class": pooled.worst_class,
                               "worst_interval": pair(&boot.worst[q]), "worst_interval_joint": pair(&boot.worst_joint[q]),
                               "rate_intervals": boot.rate[q].iter().map(pair).collect::<Vec<_>>(),
                               "worst_minus_first_policy": pair(&boot.worst_vs_first[q]),
                               "zero_observed_classes": boot.zero_observed[q]},
                    "gap_to_certified_pp": match (pooled.worst, lb) { (Some(w), Some(b)) => json!(100.0 * (w - b)), _ => Value::Null },
                    "tails": {"x": xs, "envelope": pooled.envelope.iter().map(|v| opt(*v)).collect::<Vec<_>>(),
                              "envelope_band": boot.envelope[q].iter().map(pair).collect::<Vec<_>>(),
                              "per_class": pooled.ccdf.iter().map(|c| c.iter().map(|v| opt(*v)).collect::<Vec<_>>()).collect::<Vec<_>>(),
                              "per_class_band": boot.ccdf[q].iter().map(|c| c.iter().map(pair).collect::<Vec<_>>()).collect::<Vec<_>>()},
                    "per_run": runs.iter().map(|r| json!({"seed": r.seed, "status": r.status, "growth": r.growth, "runtime_seconds": r.runtime_seconds, "events": r.events,
                                                          "decision_seconds": r.decision_seconds, "decisions": r.decisions, "max_backlog": r.max_backlog,
                                                          "phase": {"service": r.phase.0, "switch": r.phase.1, "idle": r.phase.2}, "switch_rate": r.switch_rate,
                                                          "classes": r.classes.iter().map(|c| json!({"arrivals": c.arrivals, "misses": c.misses, "unresolved": c.unresolved, "censored": c.censored, "p50": opt(c.p50), "p99": opt(c.p99), "mean_oldest_age": opt(c.mean_oldest_age)})).collect::<Vec<_>>()})).collect::<Vec<_>>(),
                    "decision_cpu": {"total_seconds": decision_seconds, "decisions": decisions, "mean_seconds_per_decision": if decisions > 0 { json!(decision_seconds / decisions as f64) } else { Value::Null }},
                    "mean_runtime_seconds": runs.iter().map(|r| r.runtime_seconds).sum::<f64>() / runs.len().max(1) as f64,
                    "consistency": cons
                }));
            }
            json!({
                "id": env.id, "profile": env.profile, "model": env.model.to_json(), "laws": env.model.describe_laws(env.settings.bits), "window": env.window.to_json(), "grid": env.settings.to_json(),
                "certified": {"full_category": full.category(), "full_lb_max": lb, "full_lb_avg": full.lower_bound_avg(),
                              "levels": ev.evaluation.levels.iter().map(|l| json!({"level": l.level.name(), "category": l.category(), "lb_max": l.lower_bound_max()})).collect::<Vec<_>>()},
                "bootstrap": {"draws": boot.draws, "seed": boot.seed, "alpha": boot.alpha, "joint_tests": boot.joint_tests},
                "policies": per_policy
            })
        })
        .collect();
    // Tables.
    let mut summary_csv = vec!["environment,profile,n,rho,d,law,full_category,full_lb_max,policy,runs,completed,censored_runs,unresolved_runs,aborted_runs,growth_flags,worst_pooled,worst_lo,worst_hi,worst_joint_lo,worst_joint_hi,gap_pp,worst_minus_ec_lo,worst_minus_ec_hi,zero_observed_classes,mean_runtime_s,decision_cpu_per_decision_s".to_string()];
    let mut runs_csv = vec!["environment,policy,seed,class,arrivals,misses,unresolved,censored,status,growth,runtime_seconds,decision_seconds,decisions,p50,p99".to_string()];
    let mut tails_csv = vec!["environment,policy,class,x,pooled,lo,hi".to_string()];
    for a in &analyses {
        let m = &a["model"];
        for p in a["policies"].as_array().unwrap() {
            let st = &p["statuses"];
            let f = |v: &Value| v.as_f64().map(|x| x.to_string()).unwrap_or_default();
            summary_csv.push(format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                a["id"].as_str().unwrap(),
                a["profile"].as_str().unwrap(),
                m["n"],
                m["rho"].as_str().unwrap_or(""),
                m["deadlines"][0].as_str().unwrap_or(""),
                a["laws"][0]["id"].as_str().unwrap_or("").replace(',', ";"),
                a["certified"]["full_category"].as_str().unwrap_or(""),
                f(&a["certified"]["full_lb_max"]),
                p["policy"].as_str().unwrap(),
                p["runs"],
                st["completed"].as_u64().unwrap_or(0),
                st["censored"].as_u64().unwrap_or(0),
                st["deadline_unresolved"].as_u64().unwrap_or(0),
                st["aborted"].as_u64().unwrap_or(0),
                p["growth_flags"],
                f(&p["pooled"]["worst"]),
                f(&p["pooled"]["worst_interval"][0]),
                f(&p["pooled"]["worst_interval"][1]),
                f(&p["pooled"]["worst_interval_joint"][0]),
                f(&p["pooled"]["worst_interval_joint"][1]),
                f(&p["gap_to_certified_pp"]),
                f(&p["pooled"]["worst_minus_first_policy"][0]),
                f(&p["pooled"]["worst_minus_first_policy"][1]),
                p["pooled"]["zero_observed_classes"]
                    .as_array()
                    .map(|z| z.len())
                    .unwrap_or(0),
                f(&p["mean_runtime_seconds"]),
                f(&p["decision_cpu"]["mean_seconds_per_decision"])
            ));
            for r in p["per_run"].as_array().unwrap() {
                for (i, c) in r["classes"].as_array().unwrap().iter().enumerate() {
                    runs_csv.push(format!(
                        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                        a["id"].as_str().unwrap(),
                        p["policy"].as_str().unwrap(),
                        r["seed"],
                        i,
                        c["arrivals"],
                        c["misses"],
                        c["unresolved"],
                        c["censored"],
                        r["status"].as_str().unwrap_or(""),
                        r["growth"],
                        f(&r["runtime_seconds"]),
                        f(&r["decision_seconds"]),
                        r["decisions"],
                        f(&c["p50"]),
                        f(&c["p99"])
                    ));
                }
            }
            let xs_v = p["tails"]["x"].as_array().unwrap();
            for (k, x) in xs_v.iter().enumerate() {
                tails_csv.push(format!(
                    "{},{},worst,{},{},{},{}",
                    a["id"].as_str().unwrap(),
                    p["policy"].as_str().unwrap(),
                    x,
                    f(&p["tails"]["envelope"][k]),
                    f(&p["tails"]["envelope_band"][k][0]),
                    f(&p["tails"]["envelope_band"][k][1])
                ));
                for (i, cls) in p["tails"]["per_class"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    tails_csv.push(format!(
                        "{},{},{},{},{},{},{}",
                        a["id"].as_str().unwrap(),
                        p["policy"].as_str().unwrap(),
                        i,
                        x,
                        f(&cls[k]),
                        f(&p["tails"]["per_class_band"][i][k][0]),
                        f(&p["tails"]["per_class_band"][i][k][1])
                    ));
                }
            }
        }
    }
    write_lines(&out.join("screen.csv"), &summary_csv)?;
    write_lines(&out.join("runs.csv"), &runs_csv)?;
    write_lines(&out.join("tails.csv"), &tails_csv)?;
    let value = json!({
        "environments": analyses, "seeds": seeds, "record_seeds": record_seeds, "policies": policies.iter().map(|(l, p)| json!({"label": l, "spec": p})).collect::<Vec<_>>(),
        "settings": settings.to_json(), "x_grid": xs, "bootstrap_draws": draws, "alpha": alpha,
        "simulation_seconds": sim_seconds, "total_seconds": t0.elapsed().as_secs_f64(), "runs": total
    });
    fs::write(
        out.join("screen.json"),
        serde_json::to_string(&value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let brief: Vec<Value> = analyses
        .iter()
        .map(|a| {
            json!({"id": a["id"], "full_category": a["certified"]["full_category"], "full_lb_max": a["certified"]["full_lb_max"],
                   "policies": a["policies"].as_array().unwrap().iter().map(|p| json!({"policy": p["policy"], "worst": p["pooled"]["worst"], "interval": p["pooled"]["worst_interval"], "gap_pp": p["gap_to_certified_pp"], "statuses": p["statuses"], "growth_flags": p["growth_flags"], "runtime_s": p["mean_runtime_seconds"]})).collect::<Vec<_>>()})
        })
        .collect();
    Ok(
        json!({"out": out, "runs": total, "simulation_seconds": sim_seconds, "total_seconds": t0.elapsed().as_secs_f64(), "environments": brief}),
    )
}
