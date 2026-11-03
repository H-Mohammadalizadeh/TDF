//! Tests of the feasibility test with continuous reconfiguration-time distributions: the grid
//! construction of the phase coefficients against closed forms (exponential against gamma of
//! shape one, deterministic reconfiguration with two configurations) and against the lattice
//! construction (the two-class example), the two time-budget rows of an enclosed mean,
//! refinement of grid step and sweeps, the four constraint levels for every profile, the
//! simulator's distributions against the exact enclosures, and geometric batch arrivals against
//! the closed-form sojourn-time distribution.

use serde_json::json;
use tdf::feasibility::coefficients::{Mode, lattice, phase_coefficients};
use tdf::feasibility::distributions::{Distribution, Model, pareto_with_alpha, profile};
use tdf::feasibility::evaluate;
use tdf::feasibility::grid::{GridSettings, grid_coefficients};
use tdf::feasibility::model::two_class_example;
use tdf::feasibility::rational::{Q, parse_q, qi, qr, to_f64};
use tdf::feasibility::relaxation::{Ingredients, Level, Objective, build};
use tdf::feasibility::windows::{default_subsets, default_windows, window_rows};
use tdf::sim::laws::Law;
use tdf::sim::rng::Rng;
use tdf::sim::runner::{Window, run, x_grid};

fn settings(step: &str, sweeps: usize) -> GridSettings {
    GridSettings {
        step: parse_q(step).unwrap(),
        bits: 64,
        sweeps,
        shortcut: true,
        all_subsets: false,
        gap_rows: false,
    }
}

fn f(x: &Q) -> f64 {
    to_f64(x)
}

fn intersect(a: (&Q, &Q), b: (&Q, &Q)) -> bool {
    a.0 <= b.1 && b.0 <= a.1
}

#[test]
fn exponential_and_gamma_shape_one_give_intersecting_coefficient_enclosures() {
    let s = settings("1/200", 2);
    let mut coefs = Vec::new();
    for name in ["exp", "gamma"] {
        let law = profile(name, &qr(1, 2), &qi(1)).unwrap();
        let m = Model::balanced(name, &law, 2, &qr(1, 2), &qi(3), &qr(1, 100));
        let (lo, hi, _) = grid_coefficients(&m, &s).unwrap();
        coefs.push((lo, hi));
    }
    let (e_lo, e_hi) = &coefs[0];
    let (g_lo, g_hi) = &coefs[1];
    for j in 0..2 {
        for i in 0..2 {
            assert!(
                intersect(
                    (&e_lo.a[j][i], &e_hi.a[j][i]),
                    (&g_lo.a[j][i], &g_hi.a[j][i])
                ),
                "A[{j}][{i}]"
            );
            assert!(
                intersect(
                    (&e_lo.b[j][i], &e_hi.b[j][i]),
                    (&g_lo.b[j][i], &g_hi.b[j][i])
                ),
                "B[{j}][{i}]"
            );
        }
    }
    assert!(intersect(
        (&e_lo.c[0][1][1], &e_hi.c[0][1][1]),
        (&g_lo.c[0][1][1], &g_hi.c[0][1][1])
    ));
    assert_eq!(e_lo.mean_tau[0][1], g_lo.mean_tau[0][1]);
}

#[test]
fn deterministic_two_configuration_closed_forms_are_bracketed() {
    // tau = v on both pairs, a common mu, deadline d > 2v.
    let (mu, v, d) = (4.0f64, 0.5f64, 3.0f64);
    let law = Distribution::Deterministic { value: qr(1, 2) };
    let m = Model::balanced("det", &law, 2, &qr(1, 2), &qi(3), &qr(1, 100));
    let (lo, hi, diag) = grid_coefficients(&m, &settings("1/100", 1)).unwrap();
    let e = |x: f64| (-x).exp();
    let b12 = e(mu * (d - v));
    let a12 = e(mu * (d - v)) * (1.0 + mu * (d - v));
    let b22 = e(mu * d);
    let a22 = e(mu * d) * (1.0 + mu * d);
    let c12_2 = (e(mu * (d - v)) - e(mu * d)) / mu;
    let c12_1 = (e(mu * (d - 2.0 * v)) - e(mu * (d - v))) / mu;
    let check = |name: &str, lo_q: &Q, hi_q: &Q, value: f64| {
        assert!(
            f(lo_q) - 1e-12 <= value && value <= f(hi_q) + 1e-12,
            "{name}: [{}, {}] vs {value}",
            f(lo_q),
            f(hi_q)
        );
        // Each enclosure is narrower than two grid steps (step 1/100): one direction places the
        // point mass at the right end of its cell, at most one cell from its position.
        assert!(
            f(hi_q) - f(lo_q) < 2.0 * 0.01 + 1e-9,
            "{name} width {}",
            f(hi_q) - f(lo_q)
        );
    };
    check("B_12", &lo.b[0][1], &hi.b[0][1], b12);
    check("A_12", &lo.a[0][1], &hi.a[0][1], a12);
    check("B_22", &lo.b[1][1], &hi.b[1][1], b22);
    check("A_22", &lo.a[1][1], &hi.a[1][1], a22);
    check("C_12,2", &lo.c[0][1][1], &hi.c[0][1][1], c12_2);
    check("C_12,1", &lo.c[0][1][0], &hi.c[0][1][0], c12_1);
    assert_eq!(lo.mean_tau[0][1], qr(1, 2));
    assert!(diag["symmetric_shortcut"].as_bool().unwrap());
}

#[test]
fn the_two_class_example_is_certified_infeasible_through_the_grid_path() {
    let discrete = two_class_example();
    let step = parse_q("1/1000").unwrap();
    let lat = lattice(&discrete, &step).unwrap();
    let lat_lo = phase_coefficients(&discrete, &lat, Mode::Upper, 64);
    let lat_hi = phase_coefficients(&discrete, &lat, Mode::Lower, 64);
    let cont = Model::from_discrete(&discrete);
    let s = settings("1/1000", 2);
    let (g_lo, g_hi, _) = grid_coefficients(&cont, &s).unwrap();
    for j in 0..2 {
        for i in 0..2 {
            assert!(
                intersect(
                    (&lat_lo.a[j][i], &lat_hi.a[j][i]),
                    (&g_lo.a[j][i], &g_hi.a[j][i])
                ),
                "A[{j}][{i}]: lattice [{}, {}] grid [{}, {}]",
                f(&lat_lo.a[j][i]),
                f(&lat_hi.a[j][i]),
                f(&g_lo.a[j][i]),
                f(&g_hi.a[j][i])
            );
            assert!(
                intersect(
                    (&lat_lo.b[j][i], &lat_hi.b[j][i]),
                    (&g_lo.b[j][i], &g_hi.b[j][i])
                ),
                "B[{j}][{i}]"
            );
        }
    }
    assert!(
        intersect(
            (&lat_lo.c[0][1][1], &lat_hi.c[0][1][1]),
            (&g_lo.c[0][1][1], &g_hi.c[0][1][1])
        ),
        "C: lattice [{}, {}] grid [{}, {}]",
        f(&lat_lo.c[0][1][1]),
        f(&lat_hi.c[0][1][1]),
        f(&g_lo.c[0][1][1]),
        f(&g_hi.c[0][1][1])
    );
    let ev = evaluate(&cont, &s, &Level::all(), None).unwrap();
    let full = ev.evaluation.level(Level::Full);
    assert_eq!(full.category(), "certified_infeasible");
    let lb = full.lower_bound_max().unwrap();
    assert!(lb > 0.0102 && lb < 0.010358, "grid-path bound {lb}");
    let base = ev.evaluation.level(Level::Base);
    assert_eq!(base.category(), "undetermined");
    assert!(base.lower_bound_max().unwrap() <= lb);
}

#[test]
fn a_time_budget_with_an_enclosed_mean_produces_two_rows() {
    let law = profile("tnorm", &qr(1, 2), &qr(1, 3)).unwrap();
    let m = Model::balanced("tnorm", &law, 2, &qr(1, 2), &qi(3), &qr(1, 100));
    let s = settings("1/200", 2);
    let (lo, _, diag) = grid_coefficients(&m, &s).unwrap();
    assert!(lo.mean_tau[0][1] < lo.mean_tau_hi[0][1]);
    let mean_width = diag["law_tables"][0]["mean_width"].as_f64().unwrap();
    assert!(
        mean_width > 0.0 && mean_width < 1e-12,
        "mean width {mean_width}"
    );
    let shell = m.lp_shell(64);
    let windows = window_rows(
        &shell,
        &default_windows(&shell),
        &default_subsets(&shell),
        64,
    );
    let ing = Ingredients {
        model: &shell,
        coef: &lo,
        windows: &windows,
        gaps: None,
    };
    let r = build(&ing, Level::Full, Objective::Feasibility);
    let labels: Vec<&str> = r.lp.rows.iter().map(|row| row.label.as_str()).collect();
    assert!(
        labels.contains(&"time_accounting_lower_mean")
            && labels.contains(&"time_accounting_upper_mean")
    );
    assert!(!labels.contains(&"time_accounting"));
    let ev = evaluate(&m, &s, &Level::all(), None).unwrap();
    for l in &ev.evaluation.levels {
        assert_ne!(
            l.category(),
            "numerically_inconclusive",
            "{}",
            l.level.name()
        );
    }
}

#[test]
fn enclosures_tighten_with_finer_grids_and_sweeps_do_not_loosen() {
    let law = profile("gamma", &qr(1, 2), &qr(1, 3)).unwrap();
    let m = Model::balanced("gamma", &law, 2, &qr(1, 2), &qi(2), &qr(1, 100));
    let width = |s: &GridSettings| -> (f64, f64, f64) {
        let (_, _, d) = grid_coefficients(&m, s).unwrap();
        let t = &d["targets"][0];
        (
            t["max_a_width"].as_f64().unwrap(),
            t["max_b_width"].as_f64().unwrap(),
            t["max_c_width"].as_f64().unwrap(),
        )
    };
    let coarse = width(&settings("1/50", 2));
    let fine = width(&settings("1/400", 2));
    assert!(
        fine.0 < coarse.0 && fine.1 < coarse.1 && fine.2 < coarse.2,
        "coarse {coarse:?} fine {fine:?}"
    );
    let (lo1, _, _) = grid_coefficients(&m, &settings("1/100", 1)).unwrap();
    let (lo3, _, _) = grid_coefficients(&m, &settings("1/100", 3)).unwrap();
    // More sweeps only lower the upper bound on the completion probability H, so the certified
    // coefficients only grow.
    assert!(
        lo3.b[0][1] >= lo1.b[0][1]
            && lo3.a[0][1] >= lo1.a[0][1]
            && lo3.c[0][1][1] >= lo1.c[0][1][1]
    );
}

#[test]
fn every_profile_is_evaluated_at_the_four_levels() {
    let s = settings("1/100", 2);
    for name in ["det", "exp", "gamma", "tnorm", "lognormal", "pareto"] {
        let law = profile(name, &qr(1, 2), &qr(1, 3)).unwrap();
        let m = Model::balanced(name, &law, 3, &qr(1, 2), &qi(2), &qr(1, 100));
        let ev = evaluate(&m, &s, &Level::all(), None).unwrap();
        let mut last = -1.0;
        for l in &ev.evaluation.levels {
            assert_ne!(
                l.category(),
                "numerically_inconclusive",
                "{name} {}",
                l.level.name()
            );
            let lb = l.lower_bound_max().unwrap();
            assert!(
                lb >= last - 1e-15,
                "{name}: bounds not monotone across levels"
            );
            last = lb;
        }
    }
    let heavy = pareto_with_alpha(&qr(1, 2), &qr(3, 2));
    let m = Model::balanced("heavy", &heavy, 2, &qr(1, 2), &qi(2), &qr(1, 100));
    let ev = evaluate(&m, &s, &Level::all(), None).unwrap();
    assert!(ev.evaluation.level(Level::Full).lower_bound_max().unwrap() >= 0.0);
}

#[test]
fn the_simulator_laws_of_the_profiles_match_the_checker_enclosures() {
    for name in ["det", "exp", "gamma", "tnorm", "lognormal", "pareto"] {
        let law = profile(name, &qr(1, 2), &qr(1, 3)).unwrap();
        let engine = Law::from_json(&law.to_engine_json()).unwrap();
        let m = law.mean(64);
        assert!(
            f(&m.0) - 1e-9 <= engine.mean() && engine.mean() <= f(&m.1) + 1e-9,
            "{name} mean"
        );
        if let Some(v) = law.variance(64) {
            assert!(
                f(&v.0) - 1e-9 <= engine.variance() && engine.variance() <= f(&v.1) + 1e-9,
                "{name} variance"
            );
        }
        // The simulator's survival function lies inside the exact enclosure.
        for x in ["1/10", "1/2", "1", "3"] {
            let xq = parse_q(x).unwrap();
            let s = law.survival(&xq, 64);
            let sv = engine.survival(to_f64(&xq));
            assert!(
                f(&s.0) - 1e-9 <= sv && sv <= f(&s.1) + 1e-9,
                "{name} survival at {x}: {sv} not in [{}, {}]",
                f(&s.0),
                f(&s.1)
            );
        }
    }
    let mut rng = Rng::xoshiro(7);
    let law =
        Law::from_json(&json!({"family": "gamma", "shape": 3.0, "scale": 1.0 / 6.0})).unwrap();
    let mean: f64 = (0..100_000).map(|_| law.sample(&mut rng)).sum::<f64>() / 100_000.0;
    assert!((mean - 0.5).abs() < 0.01);
}

#[test]
fn geometric_batch_arrivals_reproduce_the_closed_form() {
    // One class, exponential service, geometric batches of mean b: the sojourn time of a random
    // job is exponential with rate (mu - lambda) / b. With b = 1 the arrivals are the model's
    // Poisson stream.
    let policy = json!({"name": "ec", "params": {}});
    let xs = x_grid();
    let (mu, lam, d) = (4.0f64, 2.0f64, 3.0f64);
    for b in [1i64, 3] {
        let mut m = Model::new(
            "batch",
            vec![qi(2)],
            vec![qi(4)],
            vec![qi(3)],
            vec![qr(1, 100)],
            vec![vec![None]],
        )
        .unwrap();
        if b > 1 {
            m.batch_mean = Some(qi(b));
        }
        let sys = m.system_json(1000.0, 30_000.0, 100.0, true);
        assert_eq!(sys["batch_mean"].as_f64(), (b > 1).then_some(b as f64));
        let window = Window {
            warmup: 1000.0,
            measure: 30_000.0,
            followup: 100.0,
            max_wall_seconds: None,
        };
        let r = run(&m, &policy, 20260922, true, &window, &xs);
        assert_eq!(r.status, "completed");
        assert_eq!(r.classes[0].censored, 0);
        let s = &r.sojourns.as_ref().unwrap()[0];
        let count = s.len() as f64;
        let rate = r.classes[0].arrivals as f64 / window.measure;
        assert!((rate - lam).abs() < 0.05 * lam, "b={b}: job rate {rate}");
        let mean = s.iter().sum::<f64>() / count;
        let exact_mean = b as f64 / (mu - lam);
        assert!(
            (mean - exact_mean).abs() < 0.05 * exact_mean,
            "b={b}: mean sojourn {mean} vs {exact_mean}"
        );
        let miss = s.iter().filter(|x| **x > d).count() as f64 / count;
        let exact_miss = (-(mu - lam) * d / b as f64).exp();
        assert!(
            (miss - exact_miss).abs() < 0.015,
            "b={b}: miss {miss} vs {exact_miss}"
        );
    }
    // The JSON round trip of a model keeps the batch mean.
    let mut m = Model::balanced(
        "rt",
        &profile("exp", &qr(1, 2), &qi(1)).unwrap(),
        2,
        &qr(1, 2),
        &qi(3),
        &qr(1, 100),
    );
    m.batch_mean = Some(qr(5, 2));
    let back = Model::from_json(&m.to_json()).unwrap();
    assert_eq!(back.batch_mean, Some(qr(5, 2)));
}
