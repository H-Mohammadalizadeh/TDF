//! Tests of the feasibility test on models with finite-support reconfiguration-time
//! distributions (the lattice construction): the two-class example certified infeasible with an
//! exactly verified Farkas witness, the single-class bound against the exact M/M/1 miss
//! probability `exp(-(mu - lambda) d)`, monotonicity of the bounds over the four constraint
//! levels, relaxed targets, routes through intermediate configurations, inactive classes,
//! heterogeneous classes, and simulated EC and HOL-BMW runs against the necessary conditions,
//! including the pathwise coupling with a dedicated M/M/1 queue per class.

use serde_json::json;
use tdf::experiments::screen::consistency;
use tdf::feasibility::coefficients::{Mode, h_table, lattice, phase_coefficients};
use tdf::feasibility::distributions::Model;
use tdf::feasibility::model::{DiscreteLaw, DiscreteModel, two_class_example};
use tdf::feasibility::rational::{Q, parse_q, qi, qr, to_f64};
use tdf::feasibility::relaxation::{Ingredients, Level, Objective, build};
use tdf::feasibility::simplex::{certify_feasibility, certify_minimum, verify_farkas};
use tdf::feasibility::windows::{default_subsets, default_windows, window_rows};
use tdf::feasibility::{Options, evaluate_discrete, lattice_coefficients};
use tdf::sim::runner::{Run, Window, run, run_job, x_grid};

fn opts(step: &str) -> Options {
    Options {
        all_subsets: false,
        step: parse_q(step).unwrap(),
        bits: 128,
    }
}

/// 0.1 or 0.9 with equal probability (mean 0.5, variance 0.16).
fn law_a() -> DiscreteLaw {
    DiscreteLaw::new(vec![(qr(1, 10), qr(1, 2)), (qr(9, 10), qr(1, 2))]).unwrap()
}

/// 0.45 with probability 64/65, 3.7 with probability 1/65 (mean 0.5,
/// variance 0.16).
fn law_b() -> DiscreteLaw {
    DiscreteLaw::new(vec![(qr(9, 20), qr(64, 65)), (qr(37, 10), qr(1, 65))]).unwrap()
}

/// A point mass at 0.5.
fn law_det() -> DiscreteLaw {
    DiscreteLaw::point(qr(1, 2))
}

/// `n` classes with `mu_i = 4`, `lambda_i = 4 c / n`, deadline `d`, target
/// 1 %, and the reconfiguration-time distribution `law` on every pair.
fn balanced(n: usize, law: &DiscreteLaw, c: &Q, d: &Q) -> DiscreteModel {
    let laws = (0..n)
        .map(|u| (0..n).map(|v| (u != v).then(|| law.clone())).collect())
        .collect();
    DiscreteModel::new(
        "balanced",
        vec![c * qi(4) / qi(n as i64); n],
        vec![qi(4); n],
        vec![d.clone(); n],
        vec![qr(1, 100); n],
        laws,
    )
    .unwrap()
}

/// `n` classes with `mu_i = 4`, load shares `2 (n - i) / (n (n + 1))`, deadline `d`, target 1 %,
/// `law_a` on the pairs `i -> i + 1 mod n` and `law_b` on the others.
fn ring(n: usize, c: &Q, d: &Q) -> DiscreteModel {
    let lam = (0..n)
        .map(|i| c * qi(4) * qr(2 * (n as i64 - i as i64), (n * (n + 1)) as i64))
        .collect();
    let laws = (0..n)
        .map(|u| {
            (0..n)
                .map(|v| (u != v).then(|| if v == (u + 1) % n { law_a() } else { law_b() }))
                .collect()
        })
        .collect();
    DiscreteModel::new(
        "ring",
        lam,
        vec![qi(4); n],
        vec![d.clone(); n],
        vec![qr(1, 100); n],
        laws,
    )
    .unwrap()
}

/// `mu = (2, 4, 6, 8)`, load shares `(0.4, 0.3, 0.2, 0.1)`, deadlines `4 d / mu_i`, target 1 %,
/// and the reconfiguration times of `ring`, doubled on the pairs `u > v`.
fn hetero(c: &Q, d: &Q) -> DiscreteModel {
    let n = 4;
    let mu: Vec<Q> = [2, 4, 6, 8].iter().map(|m| qi(*m)).collect();
    let p: Vec<Q> = vec![qr(2, 5), qr(3, 10), qr(1, 5), qr(1, 10)];
    let lam: Vec<Q> = (0..n).map(|i| c * &p[i] * &mu[i]).collect();
    let dl: Vec<Q> = (0..n).map(|i| qi(4) * d / &mu[i]).collect();
    let laws = (0..n)
        .map(|u| {
            (0..n)
                .map(|v| {
                    (u != v).then(|| {
                        let base = if v == (u + 1) % n { law_a() } else { law_b() };
                        if u > v { base.scaled(&qi(2)) } else { base }
                    })
                })
                .collect()
        })
        .collect();
    DiscreteModel::new("hetero", lam, mu, dl, vec![qr(1, 100); n], laws).unwrap()
}

#[test]
fn the_two_class_example_is_certified_infeasible_with_a_verified_witness() {
    let m = two_class_example();
    let e = evaluate_discrete(&m, &opts("1/4000")).unwrap();
    let full = e.level(Level::Full);
    assert_eq!(
        full.category(),
        "certified_infeasible",
        "{}",
        full.feasibility.note
    );
    // The witness is verified against the exact rows.
    let (_, coef) = lattice_coefficients(&m, &opts("1/4000")).unwrap();
    let windows = window_rows(&m, &default_windows(&m), &default_subsets(&m), 128);
    let ing = Ingredients {
        model: &m,
        coef: &coef,
        windows: &windows,
    };
    let r = build(&ing, Level::Full, Objective::Feasibility);
    let w: Vec<Q> =
        r.lp.rows
            .iter()
            .map(|row| {
                full.feasibility
                    .witness
                    .as_ref()
                    .unwrap()
                    .iter()
                    .find(|(l, _)| *l == row.label)
                    .map(|(_, v)| v.clone())
                    .unwrap_or_else(|| Q::from_integer(0.into()))
            })
            .collect();
    assert!(verify_farkas(&r.lp, &w).is_ok());
    // Neither the window nor the phase constraint alone excludes the 1 % target.
    assert_eq!(e.level(Level::Base).category(), "undetermined");
    assert_eq!(e.level(Level::BaseWindows).category(), "undetermined");
    assert_eq!(e.level(Level::BasePhases).category(), "undetermined");
    // The arrival-weighted lower bound lies at most 1 % below the closed-form bound 0.0103575.
    let avg = full.lower_bound_avg().unwrap();
    assert!(avg <= 0.010357506708984512 && avg > 0.0103, "{avg}");
    let mx = full.lower_bound_max().unwrap();
    assert!(mx >= avg - 1e-12 && mx > 0.0103, "{mx}");
}

#[test]
fn the_single_class_bound_never_exceeds_the_exact_mm1_miss_probability() {
    for (lam, mu, d) in [
        ("1", "4", "3"),
        ("1/2", "1", "2"),
        ("3", "4", "1"),
        ("1/10", "1", "5"),
        ("7/2", "4", "1/2"),
    ] {
        let (lam, mu, d) = (
            parse_q(lam).unwrap(),
            parse_q(mu).unwrap(),
            parse_q(d).unwrap(),
        );
        let m = DiscreteModel::new(
            "mm1",
            vec![lam.clone()],
            vec![mu.clone()],
            vec![d.clone()],
            vec![qr(1, 100)],
            vec![vec![None]],
        )
        .unwrap();
        let e = evaluate_discrete(&m, &opts("1/2000")).unwrap();
        let exact = (-(to_f64(&mu) - to_f64(&lam)) * to_f64(&d)).exp();
        for l in Level::all() {
            let lr = e.level(l);
            assert_ne!(lr.category(), "numerically_inconclusive");
            if let Some(b) = lr.lower_bound_max() {
                assert!(
                    b <= exact * (1.0 + 1e-9),
                    "{}: bound {b} exceeds exact {exact}",
                    l.name()
                );
            }
            assert_eq!(
                lr.category() == "certified_infeasible",
                exact > 0.01 && lr.lower_bound_max().is_some_and(|b| b > 0.01),
                "{}",
                l.name()
            );
        }
        // With one class the time budget fixes z = 1 - rho, and the phase constraint gives
        // e^{-mu d} (1 + lambda d) from below and within 2 % (the lattice discretizes the
        // residual service time).
        let (_, coef) = lattice_coefficients(&m, &opts("1/2000")).unwrap();
        let rho = m.rho(0);
        let b =
            to_f64(&(&rho * &coef.a[0][0] + (Q::from_integer(1.into()) - &rho) * &coef.b[0][0]));
        let closed = (-to_f64(&mu) * to_f64(&d)).exp() * (1.0 + to_f64(&lam) * to_f64(&d));
        assert!(b <= closed && b > 0.98 * closed, "{b} vs {closed}");
    }
}

#[test]
fn adding_constraints_never_weakens_and_relaxing_targets_never_breaks_feasibility() {
    for (name, m) in [
        ("balanced2_A", balanced(2, &law_a(), &qr(1, 2), &qi(3))),
        ("ring4", ring(4, &qr(1, 2), &qi(6))),
        ("hetero4", hetero(&qr(3, 10), &qi(4))),
        ("balanced4_det", balanced(4, &law_det(), &qr(7, 10), &qi(3))),
    ] {
        let e = evaluate_discrete(&m, &opts("1/500")).unwrap();
        let b = |l: Level| e.level(l).lower_bound_max().unwrap_or(0.0);
        assert!(b(Level::Full) >= b(Level::BaseWindows) - 1e-12, "{name}");
        assert!(b(Level::Full) >= b(Level::BasePhases) - 1e-12, "{name}");
        assert!(b(Level::BasePhases) >= b(Level::Base) - 1e-12, "{name}");
        assert!(b(Level::BaseWindows) >= b(Level::Base) - 1e-12, "{name}");
        // Relaxing every target to 0.5 leaves the relaxation nonempty.
        let mut relaxed = m.clone();
        relaxed.eps = vec![qr(1, 2); m.n];
        let er = evaluate_discrete(&relaxed, &opts("1/500")).unwrap();
        assert_eq!(er.level(Level::Full).category(), "undetermined", "{name}");
        // The verdict agrees with the certified bound against the target.
        let full = e.level(Level::Full);
        if full.category() == "certified_infeasible" {
            assert!(full.lower_bound_max().unwrap() > 0.01 - 1e-12, "{name}");
        } else {
            assert!(full.lower_bound_max().unwrap() <= 0.01 + 1e-12, "{name}");
        }
    }
}

#[test]
fn inactive_classes_and_asymmetric_pairs_are_handled() {
    // Class 1 is inactive; the direct pair 0 -> 2 is slow and the route through 1 fast; service
    // rates and deadlines differ.
    let n = 3;
    let laws: Vec<Vec<Option<DiscreteLaw>>> = (0..n)
        .map(|u| {
            (0..n)
                .map(|v| {
                    (u != v).then(|| {
                        if (u, v) == (0, 2) {
                            law_b().scaled(&qi(3))
                        } else {
                            law_a()
                        }
                    })
                })
                .collect()
        })
        .collect();
    let m = DiscreteModel::new(
        "asym",
        vec![qi(1), Q::from_integer(0.into()), qr(1, 2)],
        vec![qi(4), qi(2), qi(6)],
        vec![qi(3), qi(1), qi(2)],
        vec![qr(1, 100); 3],
        laws,
    )
    .unwrap();
    let e = evaluate_discrete(&m, &opts("1/500")).unwrap();
    for l in Level::all() {
        assert_ne!(
            e.level(l).category(),
            "numerically_inconclusive",
            "{}",
            l.name()
        );
    }
    // The optimistic completion probability from 0 to 2 uses the two-hop route: H_02(2) > 0.5.
    let lat = lattice(&m, &parse_q("1/500").unwrap()).unwrap();
    let ht = h_table(&m, &lat, 2, Mode::Upper, 128);
    assert!(to_f64(&ht.h[0][lat.nd[2]]) > 0.5);
    let coef = phase_coefficients(&m, &lat, Mode::Upper, 128);
    // With the direct route only, B_02 = 1 - Pr(tau_02 + S <= 2) with tau_02 >= 1.35 would be
    // much larger.
    assert!(to_f64(&coef.b[0][2]) < 0.5);
}

#[test]
fn the_minimum_worst_class_bound_agrees_with_the_feasibility_verdict() {
    let m = balanced(4, &law_b(), &qr(9, 10), &qi(2));
    let (_, coef) = lattice_coefficients(&m, &opts("1/500")).unwrap();
    let windows = window_rows(&m, &default_windows(&m), &default_subsets(&m), 128);
    let ing = Ingredients {
        model: &m,
        coef: &coef,
        windows: &windows,
    };
    let f = certify_feasibility(&build(&ing, Level::Full, Objective::Feasibility).lp);
    let mm = certify_minimum(&build(&ing, Level::Full, Objective::MinMax).lp);
    assert_eq!(mm.status, "certified_lower_bound", "{}", mm.note);
    let bound = to_f64(mm.certified_value.as_ref().unwrap());
    assert_eq!(
        f.status == "certified_infeasible",
        bound > 0.01,
        "{} {}",
        f.status,
        bound
    );
}

#[test]
fn simulated_ec_at_the_two_class_example_satisfies_the_necessary_conditions() {
    let m = two_class_example();
    let o = opts("1/2000");
    let lat = lattice(&m, &o.step).unwrap();
    let lo = phase_coefficients(&m, &lat, Mode::Upper, o.bits);
    let hi = phase_coefficients(&m, &lat, Mode::Lower, o.bits);
    let windows = window_rows(&m, &default_windows(&m), &default_subsets(&m), o.bits);
    let model = Model::from_discrete(&m);
    let window = Window {
        warmup: 2000.0,
        measure: 3000.0,
        followup: 100.0,
        max_wall_seconds: None,
    };
    let xs = x_grid();
    let runs: Vec<Run> = [31u64, 32]
        .iter()
        .map(|&seed| {
            run(
                &model,
                &json!({"name": "ec", "params": {}}),
                seed,
                true,
                &window,
                &xs,
            )
        })
        .collect();
    for r in &runs {
        assert_eq!(r.status, "completed", "{}", r.abort_reason);
    }
    let c = consistency(&m, &lo, &hi, &windows, &runs);
    assert_eq!(c["record_runs"], 2);
    // The phases sum to one and the time budget closes.
    assert!(
        (c["phase_sum"].as_f64().unwrap() - 1.0).abs() < 1e-6,
        "{}",
        c["phase_sum"]
    );
    assert!(
        (c["time_accounting_sum"].as_f64().unwrap() - 1.0).abs() < 0.02,
        "{}",
        c["time_accounting_sum"]
    );
    // The observed miss rate is not below the phase bound by more than three standard errors.
    assert_eq!(
        c["any_phase_bound_violated_beyond_3se"], false,
        "{}",
        c["phase_bounds"]
    );
    for row in c["phase_bounds"].as_array().unwrap() {
        assert!(row["observed_rate"].as_f64().unwrap() > 0.03, "{row}");
    }
    // The window constraints hold within three standard errors.
    assert!(
        c["window_worst"]["residual_in_se"].as_f64().unwrap() > -3.0,
        "{}",
        c["window_worst"]
    );
}

#[test]
fn the_dedicated_queue_coupling_holds_pathwise_and_gives_the_baseline_level() {
    // Every class-i completion time in the network is at least the completion time of the same
    // job in a dedicated FCFS M/M/1 queue with the same arrival times and service times:
    // C_k >= max(a_k, C_{k-1}) + S_k.
    let m = two_class_example();
    let model = Model::from_discrete(&m);
    for (policy, seed) in [
        (json!({"name": "ec", "params": {}}), 41u64),
        (json!({"name": "hol_bmw", "params": {"alpha": 0.1}}), 43),
    ] {
        let system = model.system_json(2000.0, 2000.0, 100.0, true);
        let r = run_job(&json!({"system": system, "policy": policy, "seed": seed}));
        assert_eq!(r["status"], "completed", "{}", r["abort_reason"]);
        for q in 0..2usize {
            let completions: Vec<(f64, f64, f64)> = r["completion_trace"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|e| e["queue"].as_u64() == Some(q as u64))
                .map(|e| {
                    (
                        e["arrival"].as_f64().unwrap(),
                        e["time"].as_f64().unwrap(),
                        e["service_duration"].as_f64().unwrap(),
                    )
                })
                .collect();
            assert!(
                completions.len() > 1000,
                "{policy} q{q}: {}",
                completions.len()
            );
            let mut dedicated = 0.0f64;
            let (mut miss_network, mut miss_dedicated) = (0u64, 0u64);
            for (k, (a, c, s)) in completions.iter().enumerate() {
                dedicated = a.max(dedicated) + s;
                assert!(
                    *c >= dedicated - 1e-9,
                    "{policy} q{q} job {k}: network {c} < dedicated {dedicated}"
                );
                if c - a > 3.0 {
                    miss_network += 1;
                }
                if dedicated - a > 3.0 {
                    miss_dedicated += 1;
                }
            }
            assert!(
                miss_network >= miss_dedicated,
                "{policy} q{q}: {miss_network} < {miss_dedicated}"
            );
        }
    }
    // The baseline level gives the enclosed single-class bound exp(-(mu - lambda) d) = e^{-9}.
    let e = evaluate_discrete(&m, &opts("1/2000")).unwrap();
    let base = e.level(Level::Base);
    assert_eq!(base.category(), "undetermined");
    let b = base.lower_bound_max().unwrap();
    assert!((b - (-9.0f64).exp()).abs() < 1e-12, "{b}");
    // The combined test keeps its larger bound (here the phase constraint dominates the
    // single-class bound).
    let full = e.level(Level::Full);
    assert!(full.lower_bound_max().unwrap() > 0.0103);
    // At high load the single-class bound alone excludes 1 %: lambda = 1.8, mu = 4, d = 2 give
    // e^{-4.4} = 1.2 %.
    let high = balanced(
        2,
        &law_det(),
        &parse_q("9/10").unwrap(),
        &parse_q("2").unwrap(),
    );
    let eh = evaluate_discrete(&high, &opts("1/2000")).unwrap();
    assert_eq!(eh.level(Level::Base).category(), "certified_infeasible");
    assert!((eh.level(Level::Base).lower_bound_max().unwrap() - (-4.4f64).exp()).abs() < 1e-12);
}
