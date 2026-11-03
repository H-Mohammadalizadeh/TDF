//! Tests of the visit-gap constraints: the symmetric optimum of the relaxation with only the time
//! budget, flow balance and visit-gap rows against the closed forms and values of the symmetric
//! analysis (`I = (1 - rho/N) N taubar / (1 - rho)`, optimum at zero idling), for the window,
//! weak and strong rows and for the U rows (weak U-row and S-U row); the full test with the
//! visit-gap constraints: bounds that never decrease, verdicts that never change from excluded
//! to not excluded, the other levels unchanged and the groups recorded; and the membership of the
//! exact long-run vector of OJF in every visit-gap row, the U rows included.

use tdf::feasibility::coefficients::{Mode, lattice, phase_coefficients};
use tdf::feasibility::distributions::{Model, profile};
use tdf::feasibility::gaps::{GapRow, gap_rows};
use tdf::feasibility::grid::GridSettings;
use tdf::feasibility::model::{DiscreteLaw, DiscreteModel};
use tdf::feasibility::rational::{Q, parse_q, qi, qr, to_f64};
use tdf::feasibility::reference::Homogeneous;
use tdf::feasibility::relaxation::{Ingredients, Level, Objective, build, build_groups};
use tdf::feasibility::simplex::{certify_minimum, verify_primal};
use tdf::feasibility::windows::{default_subsets, default_windows, window_rows};
use tdf::feasibility::{Options, evaluate, evaluate_discrete};

/// The row families of the visit-gap constraints by label prefix: window, weak and strong rows,
/// and the U rows.
const OLD_FAMILIES: [&str; 3] = ["gap_window[", "gap_weak[", "gap_strong["];
const WEAK_U: &str = "gap_weak_u[";
const STRONG_U: &str = "gap_strong_u[";

/// `n` classes with `mu = 4`, `lambda = 4 rho / n`, deadline `d`, target 1 % and the
/// deterministic reconfiguration time 1/2 on every pair.
fn symmetric(n: usize, rho: &str, d: &str) -> DiscreteModel {
    symmetric_tau(n, rho, d, "1/2")
}

/// As [`symmetric`] with the deterministic reconfiguration time `tau`.
fn symmetric_tau(n: usize, rho: &str, d: &str, tau: &str) -> DiscreteModel {
    let rho = parse_q(rho).unwrap();
    let tau = parse_q(tau).unwrap();
    let laws = (0..n)
        .map(|u| {
            (0..n)
                .map(|v| (u != v).then(|| DiscreteLaw::point(tau.clone())))
                .collect()
        })
        .collect();
    DiscreteModel::new(
        "symmetric",
        vec![&rho * qi(4) / qi(n as i64); n],
        vec![qi(4); n],
        vec![parse_q(d).unwrap(); n],
        vec![qr(1, 100); n],
        laws,
    )
    .unwrap()
}

fn opts(gap_rows: bool) -> Options {
    Options {
        step: qr(1, 100),
        bits: 128,
        all_subsets: false,
        gap_rows,
    }
}

/// Whether a visit-gap row belongs to one of the families `prefixes`.
fn family(r: &GapRow, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| r.label.starts_with(p))
}

/// The certified minimum of the worst-class miss probability over the relaxation with only the
/// constraint groups `groups` (and the bounds `x_i <= 1`), with the visit-gap rows of the
/// families `prefixes`, rounded with `bits` fractional bits.
fn min_max(m: &DiscreteModel, groups: &[&str], prefixes: &[&str], bits: u32) -> f64 {
    let lat = lattice(m, &qr(1, 10)).unwrap();
    let coef = phase_coefficients(m, &lat, Mode::Upper, 64);
    let gaps: Vec<GapRow> = gap_rows(m, &coef.mean_tau, &default_windows(m), bits)
        .into_iter()
        .filter(|r| family(r, prefixes))
        .collect();
    let ing = Ingredients {
        model: m,
        coef: &coef,
        windows: &[],
        gaps: Some(&gaps),
    };
    let c = certify_minimum(&build_groups(&ing, groups, Objective::MinMax).lp);
    assert_eq!(c.status, "certified_lower_bound", "{}", c.note);
    to_f64(c.certified_value.as_ref().unwrap())
}

#[test]
fn the_symmetric_optimum_of_the_visit_gap_rows_matches_the_closed_forms() {
    let window = ["time", "flow", "gap_window"];
    let gap = ["time", "flow", "gap"];
    let both = ["time", "flow", "gap_window", "gap"];
    // (N, rho, d, groups, value): window rows alone at N = 4, rho = 0.9, d = 4 give 23.44 %; the
    // strong row (1 - d/I - (2/mu) sqrt(2 lambda / I) - 2 (1 - rho/N) / sqrt(lambda I))^+ gives
    // 32.96 % at N = 4, rho = 0.95, d = 8 and 19.85 % at d = 12; the weak row with the decision
    // part gives 16.68 % at N = 4, rho = 0.9, d = 4; with both groups the window rows give
    // 10.51 % at N = 4, rho = 0.95, d = 16, and no window, weak or strong row is positive at
    // (8, 0.7, 6) and (4, 0.9, 12). The U rows are tested separately below.
    let cases: [(usize, &str, &str, &[&str], f64); 8] = [
        (4, "9/10", "4", &window, 0.2344),
        (4, "19/20", "8", &gap, 0.3296),
        (4, "19/20", "12", &gap, 0.1985),
        (4, "9/10", "4", &gap, 0.1668),
        (4, "19/20", "16", &both, 0.1051),
        (4, "19/20", "4", &both, 0.4608),
        (8, "7/10", "6", &both, 0.0),
        (4, "9/10", "12", &both, 0.0),
    ];
    for (n, rho, d, groups, value) in cases {
        let m = symmetric(n, rho, d);
        let v = min_max(&m, groups, &OLD_FAMILIES, 128);
        // Finitely many window lengths and tangent points only weaken the rows; the values above
        // are rounded to four places.
        assert!(
            v <= value + 5e-5 && v >= value - 2e-3,
            "N = {n}, rho = {rho}, d = {d}, {groups:?}: {v} vs {value}"
        );
    }
}

/// `v` in percent, rounded down to two decimals, equals `floor` (also in percent).
fn floors_to(v: f64, floor: f64) -> bool {
    let cents = (1e4 * v).floor();
    cents == (100.0 * floor).round()
}

#[test]
fn the_symmetric_optimum_of_the_s_u_rows_reproduces_the_certified_floors() {
    let gap = ["time", "flow", "gap"];
    // (N, rho, d, taubar, floor %): the S-U values of the symmetric analysis (law-free), each
    // re-certified with interval arithmetic and rounded down: 53.4328, 28.1091, 32.5878, 12.6494,
    // 44.6059, 20.5539 and 6.8764 % (the last at t -> 0, here t = 10^-4).
    let cases = [
        (4, "19/20", "8", "1/2", 53.43),
        (4, "19/20", "16", "1/2", 28.10),
        (4, "7/10", "28", "5", 32.58),
        (4, "7/10", "41", "5", 12.64),
        (4, "7/10", "275", "50", 44.60),
        (4, "7/10", "413", "50", 20.55),
        (4, "7/10", "495", "50", 6.87),
    ];
    for (n, rho, d, tau, floor) in cases {
        let m = symmetric_tau(n, rho, d, tau);
        let v = min_max(&m, &gap, &[STRONG_U], 32);
        assert!(
            floors_to(v, floor),
            "N = {n}, rho = {rho}, d = {d}, taubar = {tau}: {} % vs {floor} %",
            100.0 * v
        );
        // At these settings the S-U rows are at least as strong as the strong rows.
        assert!(v >= min_max(&m, &gap, &["gap_strong["], 32) - 1e-9);
    }
}

#[test]
fn the_symmetric_optimum_of_the_weak_u_rows_reproduces_the_closed_form() {
    let gap = ["time", "flow", "gap"];
    // (N, rho, d, taubar, floor %): the closed-form weak U-row
    // max over eps = k/400 of (1 - eps)(f - eta d) - eta / (lambda beta(eps)) at zero idling,
    // f = 1 - 1/N + eta taubar, eta = (1 - rho) / (N taubar), evaluated independently in
    // doubles (9.68119, 5.56814, 39.94526, 23.34549, 28.06086, 12.25818, 37.11668, 17.65462 and
    // 6.44246 %), rounded down. The rows of the symmetric analysis with sharper certified U values
    // reach 11.41 % at (8, 0.7, 6) and 5.57 % at (4, 0.9, 12); the closed form stays below them.
    let cases = [
        (8, "7/10", "6", "1/2", 9.68),
        (4, "9/10", "12", "1/2", 5.56),
        (4, "19/20", "8", "1/2", 39.94),
        (4, "19/20", "16", "1/2", 23.34),
        (4, "7/10", "28", "5", 28.06),
        (4, "7/10", "41", "5", 12.25),
        (4, "7/10", "275", "50", 37.11),
        (4, "7/10", "413", "50", 17.65),
        (4, "7/10", "495", "50", 6.44),
    ];
    for (n, rho, d, tau, floor) in cases {
        let m = symmetric_tau(n, rho, d, tau);
        let weak = min_max(&m, &gap, &[WEAK_U], 32);
        assert!(
            floors_to(weak, floor),
            "N = {n}, rho = {rho}, d = {d}, taubar = {tau}: {} % vs {floor} %",
            100.0 * weak
        );
        // Both U families together give the larger of the two values.
        let strong = min_max(&m, &gap, &[STRONG_U], 32);
        let both = min_max(&m, &gap, &[WEAK_U, STRONG_U], 32);
        assert!(
            (both - weak.max(strong)).abs() < 1e-9,
            "{both} vs {weak}, {strong}"
        );
    }
    // Where few jobs arrive per gap the closed form is negative at the symmetric optimum
    // (-0.116 % at (2, 0.7, 3)); its rows are positive only at longer mean gaps and bound nothing.
    let m = symmetric(2, "7/10", "3");
    assert_eq!(min_max(&m, &gap, &[WEAK_U], 32), 0.0);
}

#[test]
fn the_exact_long_run_vector_of_ojf_satisfies_every_visit_gap_row_including_the_u_rows() {
    // N = 2, tau ~ Exp(2), mu = 4, rho = 9/20, d = 3, target 3/5, above m_F = exp(-0.6) = 0.549:
    // OJF's exact rates (rate identities) and the upper end of the enclosure of its miss
    // probability satisfy every row of the full test with the visit-gap rows as generated by the
    // pipeline (32 fractional bits), and the U rows of both kinds are among them.
    let s = GridSettings {
        step: qr(1, 200),
        bits: 64,
        sweeps: 2,
        shortcut: true,
        all_subsets: false,
        gap_rows: false,
    };
    let law = profile("exp", &qr(1, 2), &qi(1)).unwrap();
    let model = Model::balanced("high", &law, 2, &qr(9, 20), &qi(3), &qr(3, 5));
    let ev = evaluate(&model, &s, &[Level::Full], None).unwrap();
    let h = Homogeneous::from_model(&model).unwrap();
    let bench = h.exact_exponential_benchmark(s.bits).unwrap().unwrap();
    let rates = h.reference_rates(s.bits);
    let shell = model.lp_shell(s.bits);
    let windows = window_rows(
        &shell,
        &default_windows(&shell),
        &default_subsets(&shell),
        s.bits,
    );
    let gaps = gap_rows(&shell, &ev.lower.mean_tau, &default_windows(&shell), 32);
    for prefix in [WEAK_U, STRONG_U] {
        assert!(
            gaps.iter().any(|r| r.label.starts_with(prefix)),
            "no {prefix} rows"
        );
    }
    let ing = Ingredients {
        model: &shell,
        coef: &ev.lower,
        windows: &windows,
        gaps: Some(&gaps),
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
    verify_primal(&rel.lp, &y)
        .unwrap_or_else(|e| panic!("OJF's long-run vector violates the relaxation: {e}"));
}

#[test]
fn the_visit_gap_rows_raise_the_full_test_and_change_no_other_level() {
    for (n, rho, d) in [(4, "9/10", "4"), (4, "19/20", "8"), (3, "1/2", "2")] {
        let m = symmetric(n, rho, d);
        let off = evaluate_discrete(&m, &opts(false)).unwrap();
        let on = evaluate_discrete(&m, &opts(true)).unwrap();
        assert!(off.gaps.is_none() && on.gaps.is_some_and(|(rows, _)| rows > 0));
        for (a, b) in off.levels.iter().zip(&on.levels) {
            assert_eq!(a.level, b.level);
            let (la, lb) = (a.lower_bound_max().unwrap(), b.lower_bound_max().unwrap());
            if a.level == Level::Full {
                assert!(lb >= la, "{n} {rho} {d}: {lb} < {la}");
                assert!(b.rows > a.rows);
                assert!(b.groups.contains(&"gap_window") && b.groups.contains(&"gap"));
                assert!(!a.groups.contains(&"gap_window") && !a.groups.contains(&"gap"));
                if a.category() == "certified_infeasible" {
                    assert_eq!(b.category(), "certified_infeasible");
                }
            } else {
                assert_eq!(la, lb);
                assert_eq!(a.rows, b.rows);
                assert_eq!(a.groups, b.groups);
                assert_eq!(a.category(), b.category());
            }
        }
        let full = on.level(Level::Full);
        assert_ne!(full.category(), "numerically_inconclusive");
        assert_eq!(full.to_json()["groups"], serde_json::json!(full.groups));
    }
    // At N = 4, rho = 0.9, d = 4 the window rows of the gap group raise the full bound above the
    // symmetric value 23.4 % of the visit-gap rows alone.
    let on = evaluate_discrete(&symmetric(4, "9/10", "4"), &opts(true)).unwrap();
    let b = on.level(Level::Full).lower_bound_max().unwrap();
    assert!(b >= 0.2344 - 2e-3, "{b}");
}
