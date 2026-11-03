//! Tests of the reference analysis of the symmetric case: OJF in the simulator against its
//! effective-service recursion, pathwise on the simulator's own draws; the exact exponential
//! benchmark against an independent Lindley simulation and its transform identity; the no-waiting
//! miss probability (`floor`), the benchmark and the baseline bound `U_simple` against closed
//! forms and the relaxation; OJF's rate identities and the certified rate ranges; the workload
//! bound; the membership of OJF's long-run vector in the relaxation;
//! detours in the completion table; and the rejection of settings outside the symmetric case.

use serde_json::json;
use tdf::feasibility::coefficients::{Mode, h_table, lattice};
use tdf::feasibility::distributions::{Model, profile};
use tdf::feasibility::gaps::gap_rows;
use tdf::feasibility::grid::GridSettings;
use tdf::feasibility::model::{DiscreteLaw, DiscreteModel};
use tdf::feasibility::rational::{Q, exp_neg, parse_q, qi, qr, to_f64};
use tdf::feasibility::reference::{Homogeneous, Verdict, bracket, verdict};
use tdf::feasibility::relaxation::{Ingredients, Level, Objective, build};
use tdf::feasibility::simplex::verify_primal;
use tdf::feasibility::windows::{default_subsets, default_windows, window_rows};
use tdf::feasibility::{Options, evaluate, rate_intervals};
use tdf::sim::rng::Rng;
use tdf::sim::runner::run_job;

fn settings(step: &str) -> GridSettings {
    GridSettings {
        step: parse_q(step).unwrap(),
        bits: 64,
        sweeps: 2,
        shortcut: true,
        all_subsets: false,
        gap_rows: false,
    }
}

fn options(s: &GridSettings) -> Options {
    Options {
        step: s.step.clone(),
        bits: s.bits,
        all_subsets: s.all_subsets,
        gap_rows: s.gap_rows,
    }
}

/// Check that OJF's exact long-run vector (`nu_uv` and `z_j` from its rate identities, `x_i` at
/// the upper end of the enclosure of the exact miss probability) satisfies every row of the full
/// test, without and with the visit-gap constraints; returns the labels of the visit-gap rows.
fn ojf_vector_satisfies_every_row(model: &Model, s: &GridSettings) -> Vec<String> {
    let ev = evaluate(model, s, &[Level::Full], None).unwrap();
    let h = Homogeneous::from_model(model).unwrap();
    let bench = h.exact_exponential_benchmark(s.bits).unwrap().unwrap();
    let rates = h.reference_rates(s.bits);
    let shell = model.lp_shell(s.bits);
    let windows = window_rows(
        &shell,
        &default_windows(&shell),
        &default_subsets(&shell),
        s.bits,
    );
    let gaps = gap_rows(&shell, &ev.lower.mean_tau, &default_windows(&shell), s.bits);
    let mut labels = Vec::new();
    for with_gaps in [false, true] {
        let ing = Ingredients {
            model: &shell,
            coef: &ev.lower,
            windows: &windows,
            gaps: with_gaps.then_some(gaps.as_slice()),
        };
        let rel = build(&ing, Level::Full, Objective::Feasibility);
        let mut y = vec![Q::from_integer(0.into()); rel.lp.nvars];
        for row in &rel.nu {
            for k in row.iter().flatten() {
                y[*k] = rates.nu_pair.clone();
            }
        }
        for &k in &rel.z {
            y[k] = rates.z.0.clone();
        }
        for &k in &rel.x {
            y[k] = bench.miss.1.clone();
        }
        verify_primal(&rel.lp, &y).unwrap_or_else(|e| {
            panic!("OJF's long-run vector violates the relaxation (gap rows {with_gaps}): {e}")
        });
        if with_gaps {
            labels = rel
                .lp
                .rows
                .iter()
                .filter(|r| r.label.starts_with("gap_"))
                .map(|r| r.label.clone())
                .collect();
        }
    }
    labels
}

/// `N = 2`, `mu = 4`, `tau ~ Exp(2)`, load `rho`, deadline `d`, target 1 %.
fn benchmark_model(rho: &str, d: &str) -> Model {
    let law = profile("exp", &qr(1, 2), &qi(1)).unwrap();
    Model::balanced(
        "benchmark",
        &law,
        2,
        &parse_q(rho).unwrap(),
        &parse_q(d).unwrap(),
        &qr(1, 100),
    )
}

#[test]
fn ojf_departures_follow_the_effective_service_recursion_pathwise() {
    // Three classes, exponential reconfiguration times, moderate load; every draw comes from the
    // engine.
    let law = profile("exp", &qr(1, 2), &qi(1)).unwrap();
    let model = Model::balanced("oracle", &law, 3, &qr(1, 5), &qi(2), &qr(1, 100));
    let system = model.system_json(0.0, 2000.0, 100.0, true);
    let job = json!({"system": system, "policy": {"name": "ojf", "params": {}}, "seed": 7, "compact": false});
    let r = run_job(&job);
    assert_eq!(r["status"], "completed", "{}", r["abort_reason"]);
    let mut jobs: Vec<(f64, usize, f64, f64)> = r["completion_trace"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            (
                e["arrival"].as_f64().unwrap(),
                e["queue"].as_u64().unwrap() as usize,
                e["time"].as_f64().unwrap(),
                e["service_duration"].as_f64().unwrap(),
            )
        })
        .collect();
    jobs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut switches: Vec<(usize, usize, f64, f64)> = r["switches"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["completed"] == true)
        .map(|s| {
            (
                s["from"].as_u64().unwrap() as usize,
                s["to"].as_u64().unwrap() as usize,
                s["start"].as_f64().unwrap(),
                s["end"].as_f64().unwrap(),
            )
        })
        .collect();
    switches.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    assert!(
        jobs.len() > 1000 && switches.len() > 100,
        "{} jobs, {} switches",
        jobs.len(),
        switches.len()
    );
    let tol = 1e-7;
    let (mut prev_class, mut prev_completion, mut k) = (0usize, 0.0f64, 0usize);
    let mut last_service_start = f64::NEG_INFINITY;
    for &(a, q, c, s) in &jobs {
        let start_eff = prev_completion.max(a);
        let service_start = c - s;
        // Service in global arrival order, also after idle periods.
        assert!(
            service_start >= last_service_start - tol,
            "service order is not the arrival order"
        );
        last_service_start = service_start;
        if q != prev_class {
            // The reconfiguration to the selected job's class starts at once, and that job is
            // served when it ends.
            assert!(
                k < switches.len(),
                "missing reconfiguration before a class change"
            );
            let (f, t, st, en) = switches[k];
            k += 1;
            assert_eq!(
                (f, t),
                (prev_class, q),
                "reconfiguration goes from the previous class to the selected class"
            );
            assert!(
                (st - start_eff).abs() < tol,
                "reconfiguration starts when the job is selected: {st} vs {start_eff}"
            );
            assert!(
                (en - service_start).abs() < tol,
                "service starts when the reconfiguration ends"
            );
            assert!(
                (c - (start_eff + (en - st) + s)).abs() < tol,
                "D = V + tau + S"
            );
        } else {
            // No reconfiguration and no idling between consecutive jobs of the same class.
            assert!(
                (service_start - start_eff).abs() < tol,
                "same-class job served at once: {service_start} vs {start_eff}"
            );
            assert!((c - (start_eff + s)).abs() < tol, "D = V + S");
        }
        prev_class = q;
        prev_completion = c;
    }
    // Every completed reconfiguration precedes a completed job, except possibly the last one.
    assert!(
        k + 1 >= switches.len(),
        "{} of {} reconfigurations unaccounted for",
        switches.len() - k,
        switches.len()
    );
}

#[test]
fn lindley_recursion_reproduces_the_exact_exponential_tail() {
    // Independent implementation of the reduction: interarrival times X ~ Exp(Lambda), uniform
    // class labels, effective service time Y = S + I tau.
    let (lambda, mu, rate, d) = (0.2f64, 4.0f64, 2.0f64, 2.0f64);
    let mut rng = Rng::xoshiro(11);
    let (mut v, mut prev, mut miss, n) = (0.0f64, 0u64, 0u64, 400_000u64);
    for _ in 0..n {
        let label = rng.randrange(2);
        let y = rng.expovariate(mu)
            + if label != prev {
                rng.expovariate(rate)
            } else {
                0.0
            };
        prev = label;
        let sojourn = v + y;
        if sojourn > d {
            miss += 1;
        }
        let x = rng.expovariate(lambda);
        v = (v + y - x).max(0.0);
    }
    let estimate = miss as f64 / n as f64;
    let exact = (-(rate - lambda) * d).exp();
    assert!(
        (estimate - exact).abs() < 0.003,
        "estimate {estimate} vs exact {exact}"
    );
}

#[test]
fn exponential_cancellation_identity_holds_exactly() {
    // Laplace transform of the effective service time of the benchmark (N = 2, mu = 4,
    // tau ~ Exp(2)): E[exp(-sY)] = (4/(4+s)) [1/2 + (1/2) 2/(2+s)] = 2/(2+s).
    for s in [qr(1, 3), qi(2), qi(7), qr(11, 5)] {
        let lhs = qi(4) / (qi(4) + &s) * (qr(1, 2) + qr(1, 2) * (qi(2) / (qi(2) + &s)));
        let rhs = qi(2) / (qi(2) + &s);
        assert_eq!(lhs, rhs);
    }
}

#[test]
fn exact_benchmark_and_floor_enclose_the_closed_forms_and_bracket_the_lp_bound() {
    let s = settings("1/200");
    let model = benchmark_model("1/20", "2");
    let ev = evaluate(&model, &s, &[Level::Full], None).unwrap();
    let full = ev.evaluation.level(Level::Full);
    let h = Homogeneous::from_model(&model).unwrap();
    assert!(h.benchmark_applies());
    let br = bracket(
        &h,
        full.min_max.certified_value.as_ref(),
        &full.min_max.status,
        full.category() == "certified_infeasible",
        &s,
    )
    .unwrap();
    // The no-waiting miss probability (`floor`) is exp(-2 d) = exp(-4); m_F = exp(-3.6).
    let (b_lo, b_hi) = (to_f64(&br.floor.0), to_f64(&br.floor.1));
    assert!(
        b_lo <= (-4.0f64).exp() && (-4.0f64).exp() <= b_hi,
        "floor [{b_lo}, {b_hi}]"
    );
    assert!(b_hi - b_lo < 1e-3, "floor enclosure width {}", b_hi - b_lo);
    let bench = br.benchmark.as_ref().unwrap();
    // m_F = exp(-3.6) ~ 0.0273237: the enclosure is tight and contains the double value.
    let m_f = (-3.6f64).exp();
    assert!(
        to_f64(&bench.miss.0) <= m_f + 1e-15 && m_f - 1e-15 <= to_f64(&bench.miss.1),
        "[{}, {}] vs {m_f}",
        to_f64(&bench.miss.0),
        to_f64(&bench.miss.1)
    );
    assert!((to_f64(&bench.miss.0) - 0.0273237).abs() < 1e-7);
    assert_eq!(bench.rho_f, qr(1, 10));
    // The closed-form and grid enclosures of P(tau + S > d) intersect.
    let c = br.tail.closed_form.as_ref().unwrap();
    assert!(c.0 <= br.tail.grid.1 && br.tail.grid.0 <= c.1);
    // L_LP <= m* <= U_exact <= U_simple, and the interval is ordered.
    let l_lp = br.lp_lower.as_ref().unwrap();
    assert!(
        l_lp <= &bench.miss.1,
        "L_LP {} above the achievable bound {}",
        to_f64(l_lp),
        to_f64(&bench.miss.1)
    );
    assert!(br.u_simple.as_ref().unwrap() >= &bench.miss.0);
    assert!(&br.l_safe <= br.u_plus.as_ref().unwrap());
    assert!(br.l_safe >= br.floor.0);
    // The LP bound of this infeasible setting is about 2 %, and the verdict at 1 % is exclusion.
    assert!(to_f64(l_lp) > 0.015, "{}", to_f64(l_lp));
    assert_eq!(br.verdict.as_ref().unwrap(), &Verdict::CertifiedInfeasible);
}

#[test]
fn the_feasible_benchmark_point_lies_in_the_relaxation_and_is_certified_feasible() {
    let s = settings("1/200");
    let model = benchmark_model("1/20", "3");
    let ev = evaluate(&model, &s, &[Level::Full], None).unwrap();
    let full = ev.evaluation.level(Level::Full);
    assert_ne!(full.category(), "certified_infeasible");
    let h = Homogeneous::from_model(&model).unwrap();
    let br = bracket(
        &h,
        full.min_max.certified_value.as_ref(),
        &full.min_max.status,
        false,
        &s,
    )
    .unwrap();
    // m_F = exp(-5.4) ~ 0.00451658 < 1 %.
    let bench = br.benchmark.as_ref().unwrap();
    let m_f = (-5.4f64).exp();
    assert!(
        to_f64(&bench.miss.0) <= m_f + 1e-15 && m_f - 1e-15 <= to_f64(&bench.miss.1),
        "[{}, {}] vs {m_f}",
        to_f64(&bench.miss.0),
        to_f64(&bench.miss.1)
    );
    assert!((to_f64(&bench.miss.0) - 0.00451658).abs() < 1e-7);
    assert_eq!(br.verdict.as_ref().unwrap(), &Verdict::CertifiedFeasible);
    // OJF's exact long-run vector, with x_i at the upper end of the enclosure of m_F, satisfies
    // every row.
    let shell = model.lp_shell(s.bits);
    let windows = window_rows(
        &shell,
        &default_windows(&shell),
        &default_subsets(&shell),
        s.bits,
    );
    let ing = Ingredients {
        model: &shell,
        coef: &ev.lower,
        windows: &windows,
        gaps: None,
    };
    let rel = build(&ing, Level::Full, Objective::Feasibility);
    let rates = h.reference_rates(s.bits);
    let mut y = vec![Q::from_integer(0.into()); rel.lp.nvars];
    for u in 0..2 {
        for v in 0..2 {
            if let Some(k) = rel.nu[u][v] {
                y[k] = rates.nu_pair.clone();
            }
        }
    }
    assert_eq!(rates.nu_pair, qr(1, 20));
    // The exponential mean is exact, so rho_F = 1/10 and z = 9/20 are exact.
    assert_eq!(rates.rho_f, (qr(1, 10), qr(1, 10)));
    assert_eq!(rates.z, (qr(9, 20), qr(9, 20)));
    for j in 0..2 {
        y[rel.z[j]] = rates.z.0.clone();
    }
    for i in 0..2 {
        y[rel.x[i]] = bench.miss.1.clone();
    }
    verify_primal(&rel.lp, &y)
        .expect("the reference policy's long-run vector is a point of the relaxation");
    // The same vector satisfies the visit-gap constraints; at this load only the window rows of
    // the gap group are not vacuous.
    let labels = ojf_vector_satisfies_every_row(&model, &s);
    assert!(!labels.is_empty() && labels.iter().all(|l| l.starts_with("gap_window[")));
}

#[test]
fn ojf_satisfies_the_visit_gap_rows_at_a_high_closed_form_load() {
    // N = 2, tau ~ Exp(2), mu = 4, rho = 9/20: rho_F = 9/10 and m_F = exp(-(2 - 9/5) 3) = 0.549.
    // The target 3/5 lies above m_F, so the target rows hold as well.
    let s = settings("1/200");
    let law = profile("exp", &qr(1, 2), &qi(1)).unwrap();
    let model = Model::balanced("high", &law, 2, &qr(9, 20), &qi(3), &qr(3, 5));
    let h = Homogeneous::from_model(&model).unwrap();
    let bench = h.exact_exponential_benchmark(s.bits).unwrap().unwrap();
    assert!((to_f64(&bench.miss.0) - (-0.6f64).exp()).abs() < 1e-12);
    let labels = ojf_vector_satisfies_every_row(&model, &s);
    for family in ["gap_window[", "gap_weak[", "gap_strong["] {
        assert!(
            labels.iter().any(|l| l.starts_with(family)),
            "no {family} rows: {labels:?}"
        );
    }
}

#[test]
fn rate_projections_bracket_the_reference_rates_with_certified_endpoints() {
    let s = settings("1/200");
    let model = benchmark_model("1/20", "3");
    let ev = evaluate(&model, &s, &[Level::Full], None).unwrap();
    let shell = model.lp_shell(s.bits);
    let ivs = rate_intervals(&shell, &ev.lower, &options(&s), Level::Full);
    assert_eq!(ivs.len(), 3);
    let h = Homogeneous::from_model(&model).unwrap();
    let rates = h.reference_rates(s.bits);
    for (k, iv) in ivs.iter().enumerate() {
        assert!(!iv.empty(), "{}", iv.name);
        assert_eq!(
            iv.lower.status, "certified_lower_bound",
            "{}: {}",
            iv.name, iv.lower.note
        );
        assert_eq!(
            iv.upper.status, "certified_lower_bound",
            "{}: {}",
            iv.name, iv.upper.note
        );
        assert!(
            iv.lower.witness.is_some() && iv.upper.witness.is_some(),
            "dual multipliers are part of the certificate"
        );
        let (lo, hi) = (iv.lo().unwrap(), iv.hi().unwrap());
        let reference = if k == 0 { &rates.nu_total } else { &rates.eta };
        assert!(
            &lo <= reference && reference <= &hi,
            "{}: [{}, {}] must contain {}",
            iv.name,
            to_f64(&lo),
            to_f64(&hi),
            to_f64(reference)
        );
        assert!(lo >= Q::from_integer(0.into()));
    }
    assert_eq!(rates.nu_total, qr(1, 10));
    assert_eq!(rates.eta, qr(1, 20));
    assert_eq!(rates.beta, qi(2));
}

#[test]
fn the_symmetric_case_rejects_heterogeneous_parameters_and_unstable_loads() {
    let law = profile("exp", &qr(1, 2), &qi(1)).unwrap();
    let base = Model::balanced("h", &law, 2, &qr(1, 5), &qi(2), &qr(1, 100));
    assert!(Homogeneous::from_model(&base).is_ok());
    let mut m = base.clone();
    m.lam[0] = qr(1, 3);
    assert!(
        Homogeneous::from_model(&m)
            .unwrap_err()
            .contains("arrival rates")
    );
    let mut m = base.clone();
    m.mu[1] = qi(5);
    assert!(
        Homogeneous::from_model(&m)
            .unwrap_err()
            .contains("service rates")
    );
    let mut m = base.clone();
    m.d[1] = qi(3);
    assert!(
        Homogeneous::from_model(&m)
            .unwrap_err()
            .contains("deadlines")
    );
    let mut m = base.clone();
    m.eps[0] = qr(1, 50);
    assert!(Homogeneous::from_model(&m).unwrap_err().contains("targets"));
    let mut m = base.clone();
    m.laws[0][1] = Some(profile("det", &qr(1, 2), &qi(1)).unwrap());
    assert!(
        Homogeneous::from_model(&m)
            .unwrap_err()
            .contains("distributions differ")
    );
    let mut m = base.clone();
    m.service = Some("deterministic".into());
    assert!(Homogeneous::from_model(&m).is_err());
    // rho = 1/2 gives Lambda = 2, the rate of the effective service time: no stationary
    // benchmark and no U_simple.
    let unstable = Model::balanced("u", &law, 2, &qr(1, 2), &qi(2), &qr(1, 100));
    let h = Homogeneous::from_model(&unstable).unwrap();
    assert!(h.benchmark_applies());
    assert!(h.exact_exponential_benchmark(64).is_err());
    let rho_f = h.rho_f(64);
    assert!(rho_f.0 >= Q::from_integer(1.into()));
    assert!(Homogeneous::u_simple(&(qr(1, 10), qr(1, 10)), &rho_f, 64).is_none());
    // The verdict never resolves a contradiction by a tolerance.
    assert!(verdict(&qr(1, 100), true, &qr(1, 200), Some(&qr(1, 200))).is_err());
    assert_eq!(
        verdict(&qr(1, 100), false, &qr(1, 200), Some(&qr(1, 200))).unwrap(),
        Verdict::CertifiedFeasible
    );
    assert_eq!(
        verdict(&qr(1, 100), false, &qr(1, 50), None).unwrap(),
        Verdict::CertifiedInfeasible
    );
    assert_eq!(
        verdict(&qr(1, 100), false, &qr(1, 200), Some(&qr(1, 50))).unwrap(),
        Verdict::Undetermined
    );
}

#[test]
fn reference_rates_close_the_time_budget_for_every_profile() {
    for (name, cv2) in [
        ("det", "1/3"),
        ("exp", "1"),
        ("gamma", "1/3"),
        ("lognormal", "1/3"),
        ("pareto", "1/3"),
    ] {
        let law = profile(name, &qr(1, 2), &parse_q(cv2).unwrap()).unwrap();
        for n in [2usize, 4] {
            let model = Model::balanced(name, &law, n, &qr(1, 10), &qi(2), &qr(1, 100));
            let h = Homogeneous::from_model(&model).unwrap();
            let r = h.reference_rates(64);
            assert!(r.time_budget_contains_one, "{name}, N = {n}");
            assert_eq!(&r.nu_pair * qi((n * (n - 1)) as i64), r.nu_total);
            assert_eq!(&r.eta * qi(n as i64), r.nu_total);
            assert_eq!(&r.beta * &r.eta, h.lambda_class());
        }
    }
}

#[test]
fn closed_form_and_grid_tail_enclosures_intersect_for_the_profiles() {
    let s = settings("1/200");
    for (name, cv2) in [("det", "1/3"), ("exp", "1"), ("gamma", "1/3")] {
        let law = profile(name, &qr(1, 2), &parse_q(cv2).unwrap()).unwrap();
        let model = Model::balanced(name, &law, 2, &qr(1, 10), &qi(2), &qr(1, 100));
        let h = Homogeneous::from_model(&model).unwrap();
        let tail = h.tail(&s).unwrap();
        let c = tail
            .closed_form
            .as_ref()
            .unwrap_or_else(|| panic!("{name} has a closed form"));
        assert!(
            c.0 <= tail.grid.1 && tail.grid.0 <= c.1,
            "{name}: closed [{}, {}] grid [{}, {}]",
            to_f64(&c.0),
            to_f64(&c.1),
            to_f64(&tail.grid.0),
            to_f64(&tail.grid.1)
        );
        assert!(
            to_f64(&c.1) - to_f64(&c.0) < 1e-9,
            "{name}: closed-form width"
        );
    }
    // The deterministic tail is exact: P(1/2 + S > 2) = exp(-6).
    let det = profile("det", &qr(1, 2), &qi(1)).unwrap();
    let model = Model::balanced("det", &det, 2, &qr(1, 10), &qi(2), &qr(1, 100));
    let c = Homogeneous::from_model(&model)
        .unwrap()
        .tail_closed_form(64)
        .unwrap();
    let e6 = exp_neg(&qi(6), 64);
    assert!(c.0 <= e6.1 && e6.0 <= c.1);
    // Pareto has no closed form and uses the grid enclosure alone.
    let pareto = profile("pareto", &qr(1, 2), &qr(1, 3)).unwrap();
    let model = Model::balanced("pareto", &pareto, 2, &qr(1, 10), &qi(2), &qr(1, 100));
    let h = Homogeneous::from_model(&model).unwrap();
    assert!(h.tail_closed_form(64).is_none());
    let tail = h.tail(&s).unwrap();
    assert_eq!(tail.used, tail.grid);
}

#[test]
fn detours_through_intermediate_configurations_are_kept_in_the_completion_table() {
    // Three configurations, deterministic reconfiguration times: 0 -> 2 takes 3, 0 -> 1 -> 2
    // takes 2; with deadline 5/2 only the detour can complete the isolated job.
    let point = |v: i64| Some(DiscreteLaw::point(qi(v)));
    let laws = vec![
        vec![None, point(1), point(3)],
        vec![point(1), None, point(1)],
        vec![point(3), point(1), None],
    ];
    let m = DiscreteModel::new(
        "detour",
        vec![qr(1, 10); 3],
        vec![qi(4); 3],
        vec![qr(5, 2); 3],
        vec![qr(1, 100); 3],
        laws,
    )
    .unwrap();
    let lat = lattice(&m, &qr(1, 2)).unwrap();
    let nd = lat.nd[2];
    let direct_only = 0.0; // tau_02 = 3 > 5/2
    let detour = 1.0 - (-4.0f64 * 0.5).exp(); // two units of reconfiguration, half a unit of service
    for mode in [Mode::Upper, Mode::Lower] {
        let ht = h_table(&m, &lat, 2, mode, 64);
        let value = to_f64(&ht.h[0][nd]);
        assert!(
            value > direct_only + 0.5,
            "{mode:?}: H_02(5/2) = {value} must use the detour"
        );
        assert!(
            (value - detour).abs() < 1e-9,
            "{mode:?}: {value} vs {detour}"
        );
    }
}

#[test]
fn the_exact_benchmark_generalizes_to_every_n_with_mu_equal_n_r() {
    // N = 4, tau ~ Exp(1) (mean 1), mu = 4 = N r: Y ~ Exp(1) and m_F = exp(-(1 - Lambda) d).
    let law = profile("exp", &qi(1), &qi(1)).unwrap();
    let model = Model::balanced("n4", &law, 4, &qr(1, 20), &qi(2), &qr(1, 100));
    let h = Homogeneous::from_model(&model).unwrap();
    assert!(h.benchmark_applies());
    let b = h.exact_exponential_benchmark(64).unwrap().unwrap();
    assert_eq!(b.rho_f, qr(1, 5)); // Lambda / r = 0.2 / 1
    let m_f = (-(1.0f64 - 0.2) * 2.0).exp();
    assert!(to_f64(&b.miss.0) <= m_f + 1e-15 && m_f - 1e-15 <= to_f64(&b.miss.1));
    // With mu != N r there is no benchmark.
    let model = Model::balanced("n3", &law, 3, &qr(1, 20), &qi(2), &qr(1, 100));
    assert!(!Homogeneous::from_model(&model).unwrap().benchmark_applies());
}

#[test]
fn workload_bound_encloses_the_exact_tail_and_starts_at_u_simple() {
    let s = settings("1/200");
    let model = benchmark_model("1/20", "3");
    let h = Homogeneous::from_model(&model).unwrap();
    let b = h.exact_exponential_benchmark(s.bits).unwrap().unwrap();
    let m_f = to_f64(&b.miss.1);
    let floor = h.floor_from_tail(&h.tail_closed_form(s.bits).unwrap(), s.bits);
    let u_simple = to_f64(&Homogeneous::u_simple(&floor, &h.rho_f(s.bits), s.bits).unwrap());
    // K = 0 gives the baseline bound U_simple, up to the grid rounding of Y.
    let w0 = h
        .workload_bound(&parse_q("1/1000").unwrap(), 0, s.bits)
        .unwrap();
    let u0 = to_f64(&w0.u_k_plus);
    assert!(
        u0 >= u_simple - 1e-12 && u0 - u_simple < 8e-3,
        "U_0 = {u0} vs U_simple = {u_simple}"
    );
    // With more terms the bound stays above the exact value, is nonincreasing in K and
    // approaches the exact value.
    let mut last = u0;
    for k in [2usize, 6, 12] {
        let w = h
            .workload_bound(&parse_q("1/1000").unwrap(), k, s.bits)
            .unwrap();
        let u = to_f64(&w.u_k_plus);
        assert!(
            u >= m_f,
            "K = {k}: U_K^+ = {u} below the exact miss probability {m_f}"
        );
        assert!(u <= last + 1e-15, "K = {k}: not nonincreasing");
        last = u;
    }
    assert!(last - m_f < 4e-3, "K = 12: U_K^+ - m_F = {}", last - m_f);
    assert!(
        to_f64(
            &h.workload_bound(&parse_q("1/1000").unwrap(), 12, s.bits)
                .unwrap()
                .truncation
        ) < 1e-12
    );
    assert_eq!(h.workload_terms(s.bits, 1e-6, 64), Some(5)); // rho_F = 0.1: 0.1^6 <= 1e-6
    // An unstable reduction (rho >= 1/2 here) has no workload bound.
    let unstable = benchmark_model("1/2", "3");
    let hu = Homogeneous::from_model(&unstable).unwrap();
    assert!(hu.workload_terms(s.bits, 1e-6, 64).is_none());
    assert!(
        hu.workload_bound(&parse_q("1/1000").unwrap(), 4, s.bits)
            .is_err()
    );
}

#[test]
fn the_rate_membership_check_agrees_with_the_verdicts() {
    use tdf::feasibility::point_in_relaxation;
    let s = settings("1/200");
    // OJF's rates with x_i = eps lie in the relaxation at eps = 1/250 although OJF misses that
    // target (exp(-5.4) ~ 0.452 %): membership is only necessary.
    let model = benchmark_model("1/20", "3");
    let ev = evaluate(&model, &s, &[Level::Full], None).unwrap();
    let shell = model.lp_shell(s.bits);
    let h = Homogeneous::from_model(&model).unwrap();
    let rates = h.reference_rates(s.bits);
    let eps = qr(1, 250);
    assert!(
        point_in_relaxation(
            &shell,
            &ev.lower,
            &options(&s),
            Level::Full,
            &rates.nu_pair,
            &rates.z.0,
            &eps
        )
        .is_ok()
    );
    let b = h.exact_exponential_benchmark(s.bits).unwrap().unwrap();
    assert!(b.miss.0 > eps);
    // At a target below the certified no-waiting miss probability the same point is rejected.
    assert!(
        point_in_relaxation(
            &shell,
            &ev.lower,
            &options(&s),
            Level::Full,
            &rates.nu_pair,
            &rates.z.0,
            &qr(1, 500)
        )
        .is_err()
    );
}
