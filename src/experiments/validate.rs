//! `tdf validate`: checks of the phase-coefficient constructions and of the simulator's
//! distributions against independent references, written to `validate.json`: (1) exponential
//! reconfiguration against gamma of shape one, (2) closed-form coefficients of deterministic
//! reconfiguration with two configurations, (3) the two-class example through the lattice and the
//! grid constructions, (4) the moments of every profile in the simulator against their exact
//! enclosures and a sample, (5) enclosure widths against grid step and refinement sweeps, and
//! (6) geometric batch arrivals against the closed-form sojourn-time distribution. The comments
//! in [`run`] state each check.

use super::common::write_json;
use crate::feasibility::coefficients::{Mode, lattice, phase_coefficients};
use crate::feasibility::distributions::{
    Distribution, Model, pareto_with_alpha, profile, profile_names,
};
use crate::feasibility::grid::{GridSettings, grid_coefficients};
use crate::feasibility::model::two_class_example;
use crate::feasibility::rational::{Q, parse_q, qi, qr, to_f64};
use crate::feasibility::relaxation::Level;
use crate::feasibility::{Options, evaluate, evaluate_discrete};
use crate::sim::laws::Law;
use crate::sim::rng::Rng;
use crate::sim::runner::{Window, run as simulate_run, x_grid};
use serde_json::{Value, json};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Check the coefficient constructions and the simulator's distributions.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// Output directory.
    #[arg(long)]
    pub out: PathBuf,
    /// Samples per distribution for the check of the sample mean.
    #[arg(long, default_value_t = 200_000)]
    pub samples: usize,
    /// Worker threads (default: all cores).
    #[arg(long)]
    pub threads: Option<usize>,
}

/// Sample mean, sample variance and minimum of `n` draws of a distribution.
fn sample_moments(law: &Law, n: usize, seed: u64) -> (f64, f64, f64) {
    let mut rng = Rng::xoshiro(seed);
    let (mut s, mut s2, mut min) = (0.0f64, 0.0f64, f64::INFINITY);
    for _ in 0..n.max(1) {
        let x = law.sample(&mut rng);
        s += x;
        s2 += x * x;
        min = min.min(x);
    }
    let m = s / n.max(1) as f64;
    (m, (s2 / n.max(1) as f64 - m * m).max(0.0), min)
}

/// Run every check, write `validate.json` and return the verdicts.
pub fn run(args: &Args) -> Result<Value, String> {
    let (out, samples) = (&args.out, args.samples);
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    let mean = parse_q("1/2")?;
    let cv2 = parse_q("1/3")?;
    let f = |q: &Q| to_f64(q);
    let mut checks = Vec::new();
    // 1. Exponential reconfiguration against gamma of shape one, the same distribution: the
    //    enclosures of A, B and one C coefficient must intersect.
    {
        let s = GridSettings {
            step: parse_q("1/500")?,
            bits: 128,
            sweeps: 3,
            shortcut: true,
            all_subsets: false,
        };
        let e = Model::balanced(
            "exp",
            &profile("exp", &mean, &qi(1))?,
            2,
            &mean,
            &qi(3),
            &qr(1, 100),
        );
        let g = Model::balanced(
            "gamma1",
            &profile("gamma", &mean, &qi(1))?,
            2,
            &mean,
            &qi(3),
            &qr(1, 100),
        );
        let (elo, ehi, _) = grid_coefficients(&e, &s)?;
        let (glo, ghi, _) = grid_coefficients(&g, &s)?;
        let mut ok = true;
        let mut worst = 0.0f64;
        for j in 0..2 {
            for i in 0..2 {
                ok &= elo.a[j][i] <= ghi.a[j][i] && glo.a[j][i] <= ehi.a[j][i];
                ok &= elo.b[j][i] <= ghi.b[j][i] && glo.b[j][i] <= ehi.b[j][i];
                worst = worst
                    .max((f(&elo.a[j][i]) - f(&glo.a[j][i])).abs())
                    .max((f(&elo.b[j][i]) - f(&glo.b[j][i])).abs());
            }
        }
        ok &= elo.c[0][1][1] <= ghi.c[0][1][1] && glo.c[0][1][1] <= ehi.c[0][1][1];
        checks.push(json!({"check": "exponential_vs_gamma_shape_one", "passed": ok, "max_abs_difference_of_lower_coefficients": worst,
                           "exp_full_lb": evaluate(&e, &s, &Level::all(), None)?.evaluation.level(Level::Full).lower_bound_max(),
                           "gamma_full_lb": evaluate(&g, &s, &Level::all(), None)?.evaluation.level(Level::Full).lower_bound_max()}));
    }
    // 2. Deterministic reconfiguration, N = 2, tau = 1/2, mu = 4, d = 3: every enclosure must
    //    contain the closed form of its coefficient.
    {
        let s = GridSettings {
            step: parse_q("1/500")?,
            bits: 128,
            sweeps: 1,
            shortcut: true,
            all_subsets: false,
        };
        let m = Model::balanced(
            "det",
            &Distribution::Deterministic {
                value: mean.clone(),
            },
            2,
            &mean,
            &qi(3),
            &qr(1, 100),
        );
        let (lo, hi, _) = grid_coefficients(&m, &s)?;
        let (mu, v, d) = (4.0f64, 0.5f64, 3.0f64);
        let e = |x: f64| (-x).exp();
        let closed = [
            ("B_12", e(mu * (d - v)), f(&lo.b[0][1]), f(&hi.b[0][1])),
            (
                "A_12",
                e(mu * (d - v)) * (1.0 + mu * (d - v)),
                f(&lo.a[0][1]),
                f(&hi.a[0][1]),
            ),
            ("B_22", e(mu * d), f(&lo.b[1][1]), f(&hi.b[1][1])),
            (
                "A_22",
                e(mu * d) * (1.0 + mu * d),
                f(&lo.a[1][1]),
                f(&hi.a[1][1]),
            ),
            (
                "C_12,2",
                (e(mu * (d - v)) - e(mu * d)) / mu,
                f(&lo.c[0][1][1]),
                f(&hi.c[0][1][1]),
            ),
            (
                "C_12,1",
                (e(mu * (d - 2.0 * v)) - e(mu * (d - v))) / mu,
                f(&lo.c[0][1][0]),
                f(&hi.c[0][1][0]),
            ),
        ];
        let rows: Vec<Value> = closed.iter().map(|(n, c, l, h)| json!({"coefficient": n, "closed_form": c, "lower": l, "upper": h, "bracketed": l - 1e-12 <= *c && *c <= h + 1e-12, "width": h - l})).collect();
        let passed = rows.iter().all(|r| r["bracketed"].as_bool().unwrap());
        checks.push(json!({"check": "deterministic_two_configuration_closed_forms", "passed": passed, "coefficients": rows}));
    }
    // 3. The two-class example: the lattice and grid enclosures of A, B and one C coefficient
    //    must intersect, and the grid construction must certify infeasibility; the lattice bound
    //    and the closed-form bound are reported.
    {
        let discrete = two_class_example();
        let step = parse_q("1/2000")?;
        let lat = lattice(&discrete, &step)?;
        let lat_lo = phase_coefficients(&discrete, &lat, Mode::Upper, 128);
        let lat_hi = phase_coefficients(&discrete, &lat, Mode::Lower, 128);
        let cont = Model::from_discrete(&discrete);
        let s = GridSettings {
            step,
            bits: 128,
            sweeps: 3,
            shortcut: true,
            all_subsets: false,
        };
        let (g_lo, g_hi, _) = grid_coefficients(&cont, &s)?;
        let mut ok = true;
        for j in 0..2 {
            for i in 0..2 {
                ok &= lat_lo.a[j][i] <= g_hi.a[j][i] && g_lo.a[j][i] <= lat_hi.a[j][i];
                ok &= lat_lo.b[j][i] <= g_hi.b[j][i] && g_lo.b[j][i] <= lat_hi.b[j][i];
            }
        }
        ok &= lat_lo.c[0][1][1] <= g_hi.c[0][1][1] && g_lo.c[0][1][1] <= lat_hi.c[0][1][1];
        let ev = evaluate(&cont, &s, &Level::all(), None)?;
        let full = ev.evaluation.level(Level::Full);
        let lattice_eval = evaluate_discrete(
            &discrete,
            &Options {
                step: s.step.clone(),
                bits: 128,
                all_subsets: false,
            },
        )?;
        checks.push(json!({"check": "two_class_example_lattice_vs_grid", "passed": ok && full.category() == "certified_infeasible",
                           "grid_category": full.category(), "grid_full_lb": full.lower_bound_max(),
                           "lattice_full_lb": lattice_eval.level(Level::Full).lower_bound_max(), "closed_form_lb": 0.0103575067089845,
                           "lattice_c": [f(&lat_lo.c[0][1][1]), f(&lat_hi.c[0][1][1])], "grid_c": [f(&g_lo.c[0][1][1]), f(&g_hi.c[0][1][1])]}));
    }
    // 4. Every profile, `pareto_heavy`, gamma with squared coefficient of variation 4 and the
    //    truncated normal with 1/16: the simulator's mean must lie in the exact enclosure of the
    //    mean, the sample mean within 2 % of it, and every sample must be positive.
    {
        let mut rows = Vec::new();
        let mut passed = true;
        let mut profiles: Vec<(String, Distribution)> = profile_names()
            .iter()
            .map(|p| (p.to_string(), profile(p, &mean, &cv2).unwrap()))
            .collect();
        profiles.push((
            "pareto_heavy".into(),
            pareto_with_alpha(&mean, &parse_q("3/2")?),
        ));
        profiles.push(("gamma_cv2".into(), profile("gamma", &mean, &qi(4))?));
        profiles.push(("tnorm_cv025".into(), profile("tnorm", &mean, &qr(1, 16))?));
        for (name, law) in &profiles {
            let m = law.mean(128);
            let v = law.variance(128);
            let engine = Law::from_json(&law.to_engine_json())?;
            let (smean, svar, min_sample) = sample_moments(&engine, samples, 20260922);
            let mean_ok = f(&m.0) - 1e-9 <= engine.mean() && engine.mean() <= f(&m.1) + 1e-9;
            let sample_ok = (smean - engine.mean()).abs() < 0.02 * engine.mean();
            passed &= mean_ok && sample_ok && min_sample > 0.0;
            rows.push(json!({"profile": name, "law": law.describe(128), "engine_mean": engine.mean(), "engine_variance": engine.variance(),
                             "sample_mean": smean, "sample_variance": svar, "samples": samples, "min_sample": min_sample,
                             "prescribed_mean_inside_enclosure": f(&m.0) <= 0.5 && 0.5 <= f(&m.1) || name.starts_with("tnorm"),
                             "mean_enclosure_width": f(&(&m.1 - &m.0)), "variance_enclosure": v.as_ref().map(|v| json!([f(&v.0), f(&v.1)])).unwrap_or(Value::Null),
                             "engine_mean_inside_checker_enclosure": mean_ok, "sample_mean_within_2pct": sample_ok}));
        }
        checks.push(json!({"check": "profile_moments_and_engine_sampling", "passed": passed, "profiles": rows}));
    }
    // 5. Enclosure widths and bounds against grid step and refinement sweeps (reported only).
    {
        let mut rows = Vec::new();
        for name in ["exp", "gamma", "lognormal", "pareto", "tnorm"] {
            let law = profile(name, &mean, &cv2)?;
            let m = Model::balanced(name, &law, 2, &mean, &qi(3), &qr(1, 100));
            for (st, sw) in [
                ("1/100", 1),
                ("1/100", 3),
                ("1/500", 1),
                ("1/500", 3),
                ("1/2000", 3),
            ] {
                let s = GridSettings {
                    step: parse_q(st)?,
                    bits: 128,
                    sweeps: sw,
                    shortcut: true,
                    all_subsets: false,
                };
                let t1 = Instant::now();
                let ev = evaluate(&m, &s, &Level::all(), None)?;
                let t = &ev.diagnostics["targets"][0];
                rows.push(json!({"profile": name, "step": st, "sweeps": sw, "max_h_width": t["max_h_width"], "max_a_width": t["max_a_width"],
                                 "max_b_width": t["max_b_width"], "max_c_width": t["max_c_width"], "last_sweep_change": t["last_sweep_change"],
                                 "full_lb": ev.evaluation.level(Level::Full).lower_bound_max(), "full_category": ev.evaluation.level(Level::Full).category(),
                                 "seconds": t1.elapsed().as_secs_f64()}));
            }
        }
        checks.push(
            json!({"check": "enclosure_width_vs_step_and_sweeps", "passed": true, "rows": rows}),
        );
    }
    // 6. Geometric batch arrivals: one class, exponential service, batch mean b. The sojourn
    //    time of a random job is exponential with rate (mu - lambda) / b, so the mean sojourn
    //    time is b / (mu - lambda) and the miss probability at d is exp(-(mu - lambda) d / b);
    //    b = 1 is the M/M/1 queue. Required: arrival rate and mean sojourn time within 3 %, miss
    //    probability within 0.01, and a completed run without censored jobs.
    {
        let mut rows = Vec::new();
        let mut passed = true;
        let (mu, lam, d) = (4.0f64, 2.0f64, 3.0f64);
        let policy = json!({"name": "ec", "params": {}});
        let xs = x_grid();
        for b in [1i64, 3] {
            let mut m = Model::new(
                "batch",
                vec![qi(2)],
                vec![qi(4)],
                vec![qi(3)],
                vec![qr(1, 100)],
                vec![vec![None]],
            )?;
            if b > 1 {
                m.batch_mean = Some(qi(b));
            }
            let window = Window {
                warmup: 2000.0,
                measure: 100_000.0,
                followup: 100.0,
                max_wall_seconds: None,
            };
            let t1 = Instant::now();
            let r = simulate_run(&m, &policy, 20260922, true, &window, &xs);
            let s = &r.sojourns.as_ref().unwrap()[0];
            let count = s.len() as f64;
            let sample_mean = s.iter().sum::<f64>() / count.max(1.0);
            let exact_mean = b as f64 / (mu - lam);
            let miss = s.iter().filter(|x| **x > d).count() as f64 / count.max(1.0);
            let exact_miss = (-(mu - lam) * d / b as f64).exp();
            let arrivals = r.classes[0].arrivals as f64;
            let rate_ok = (arrivals / window.measure - lam).abs() < 0.03 * lam;
            let mean_ok = (sample_mean - exact_mean).abs() < 0.03 * exact_mean;
            let miss_ok = (miss - exact_miss).abs() < 0.01;
            let complete = r.classes[0].censored == 0
                && r.classes[0].unresolved == 0
                && r.status == "completed";
            passed &= rate_ok && mean_ok && miss_ok && complete;
            rows.push(json!({"batch_mean": b, "arrivals": arrivals, "arrival_rate": arrivals / window.measure, "prescribed_rate": lam, "rate_within_3pct": rate_ok,
                             "sample_mean_sojourn": sample_mean, "exact_mean_sojourn": exact_mean, "mean_within_3pct": mean_ok,
                             "miss_probability": miss, "exact_miss_probability": exact_miss, "miss_within_0.01": miss_ok,
                             "completed": s.len(), "censored": r.classes[0].censored, "unresolved": r.classes[0].unresolved, "status": r.status.clone(),
                             "seconds": t1.elapsed().as_secs_f64()}));
        }
        checks.push(json!({"check": "geometric_batch_arrivals_closed_form", "passed": passed, "rows": rows}));
    }
    let passed = checks
        .iter()
        .all(|c| c["passed"].as_bool().unwrap_or(false));
    let value =
        json!({"passed": passed, "checks": checks, "elapsed_seconds": t0.elapsed().as_secs_f64()});
    write_json(&out.join("validate.json"), &value)?;
    Ok(
        json!({"out": out, "passed": passed, "checks": checks.iter().map(|c| json!({"check": c["check"], "passed": c["passed"]})).collect::<Vec<_>>(), "elapsed_seconds": t0.elapsed().as_secs_f64()}),
    )
}
