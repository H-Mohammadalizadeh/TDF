//! Tests of the witness (cyclic exhaustive polling): the closed forms with one class, the
//! light-traffic limit, a nonincreasing upper end in the truncation depth, the limit of large
//! deterministic reconfiguration times, the admission rules of the directed rounding, a
//! regression against the evaluation grid, and the identifiers and verdicts of `tdf witness` and
//! `tdf verdicts` against `tdf map`.

use tdf::experiments::common::profile_law;
use tdf::experiments::{map, verdicts, witness};
use tdf::feasibility::rational::{Q, parse_q, qi, qr, to_f64};
use tdf::witness::fp::{Dir, FLOOR, MAX_DIVISOR, MIN_CONST, take_admitted};
use tdf::witness::{Law, Opts, Out, System, compute, default_step};

fn q(s: &str) -> Q {
    parse_q(s).unwrap()
}

/// The enclosure of class 0 of `n` identical classes (`mu = 4`, total load `rho`) at step `g`.
fn enclose(n: usize, rho: &str, law: &Law, d: &str, g: &Q, lower: bool) -> Out {
    let sys = System::symmetric(n, &q(rho), &qi(4), law);
    let opts = Opts {
        g: g.clone(),
        lower,
        ..Opts::default()
    };
    compute(&sys, 0, &q(d), &opts)
}

#[test]
fn one_class_matches_the_closed_forms() {
    // With one class, I = tau and m = P(Exp(theta) + tau^R > d), tau^R the equilibrium
    // residual of tau: theta = mu - lambda = 4 (1 - rho).
    for (family, mean) in [("det", "1/2"), ("exp", "1/2"), ("det", "5"), ("exp", "5")] {
        let c = q(mean);
        let law = if family == "det" {
            Law::Det { c: c.clone() }
        } else {
            Law::Exp { rate: c.recip() }
        };
        for (rho, d) in [("1/2", "1"), ("1/2", "2"), ("1/5", "3"), ("1/2", "8")] {
            let o = enclose(1, rho, &law, d, &default_step(&q(d), &qr(1, 2)), true);
            let theta = 4.0 * (1.0 - to_f64(&q(rho)));
            let (df, cf) = (to_f64(&q(d)), to_f64(&c));
            let exact = if family == "det" {
                // (1/c) int_0^c P(Exp(theta) > d - r) dr
                if df >= cf {
                    (-theta * df).exp() * ((theta * cf).exp() - 1.0) / (theta * cf)
                } else {
                    ((-theta * df).exp() * ((theta * df).exp() - 1.0) / theta + (cf - df)) / cf
                }
            } else {
                // tau^R ~ Exp(b): P(Exp(theta) + Exp(b) > d), an Erlang-2 tail when b = theta
                let b = 1.0 / cf;
                if (b - theta).abs() < 1e-12 {
                    (-theta * df).exp() * (1.0 + theta * df)
                } else {
                    (b * (-theta * df).exp() - theta * (-b * df).exp()) / (b - theta)
                }
            };
            let what = format!(
                "{family} mean {mean} rho {rho} d {d}: [{}, {}] vs {exact}",
                o.lo, o.hi
            );
            assert!(o.lo_f <= exact * (1.0 + 1e-12), "{what}");
            assert!(exact <= o.hi_f * (1.0 + 1e-12), "{what}");
            assert!(o.hi_f - exact <= 1e-6, "{what}");
            assert_eq!(o.admitted, 0, "{what}");
        }
    }
}

#[test]
fn light_traffic_approaches_the_reconfiguration_only_limit() {
    // N = 4, deterministic 1/2: as rho -> 0, I -> 2 and m -> P(Exp(4) + U(0, 2) > 3)
    // = e^-4 (1 - e^-8) / 8; the miss probability increases with the load.
    let limit = (-4.0f64).exp() * (1.0 - (-8.0f64).exp()) / 8.0;
    let law = Law::Det { c: qr(1, 2) };
    let o5 = enclose(4, "1/100000", &law, "3", &qr(1, 200), true);
    let o6 = enclose(4, "1/1000000", &law, "3", &qr(1, 200), true);
    assert!(
        limit <= o5.hi_f && o5.hi_f - limit <= 1e-6,
        "{} vs {limit}",
        o5.hi
    );
    assert!(
        limit <= o6.hi_f && o6.hi_f - limit <= 0.5 * (o5.hi_f - limit),
        "{} vs {limit}",
        o6.hi
    );
    assert!(o5.lo_f <= o5.hi_f && o6.lo_f <= o6.hi_f);
}

#[test]
fn hi_does_not_increase_with_the_truncation_depth() {
    // N = 2, rho = 9/10, exponential reconfiguration with mean 1/2, d = 12 at K past cycles
    let sys = System::symmetric(2, &qr(9, 10), &qi(4), &Law::Exp { rate: qi(2) });
    let expected = [
        (5, "0.170733806677"),
        (10, "0.109876632873"),
        (20, "0.100961599565"),
        (40, "0.100798471373"),
        (80, "0.100798418549"),
    ];
    let mut prev = f64::INFINITY;
    for (k, hi) in expected {
        let opts = Opts {
            g: qr(1, 200),
            tol_trunc: 0.0,
            kmin: k,
            kmax: k,
            lower: false,
            ..Opts::default()
        };
        let o = compute(&sys, 0, &qi(12), &opts);
        assert_eq!(o.k_cycles, k);
        assert!(o.hi_f <= prev, "K = {k}: {} after {prev}", o.hi);
        assert_eq!(o.hi, hi, "K = {k}");
        prev = o.hi_f;
    }
}

#[test]
fn large_deterministic_reconfiguration_times_give_the_uniform_limit() {
    // N = 4, deterministic 50: every intervisit time is at least 200 > d, so
    // m = 1 - W(d) / E[I] = 1 - x + (1 - e^{-theta d}) / (theta E[I]) with x = d / E[I].
    for rho in ["1/10", "9/10"] {
        for d in ["50", "100", "150"] {
            let o = enclose(4, rho, &Law::Det { c: qi(50) }, d, &qr(1, 2), true);
            let r = to_f64(&q(rho));
            let theta = 4.0 - r;
            let ei = (1.0 - r / 4.0) * 200.0 / (1.0 - r);
            let df = to_f64(&q(d));
            let exact = 1.0 - df / ei + (1.0 - (-theta * df).exp()) / (theta * ei);
            assert_eq!(o.mass_le_d, 0.0);
            assert!(
                (o.hi_f - exact).abs() <= 1e-11,
                "rho {rho} d {d}: {} vs {exact}",
                o.hi
            );
            assert!(
                o.lo_f <= exact + 1e-12,
                "rho {rho} d {d}: {} vs {exact}",
                o.lo
            );
        }
    }
}

#[test]
fn admission_rules_keep_the_bound_directions() {
    take_admitted();
    assert_eq!(Dir::Up.constant(0.0), 0.0);
    assert_eq!(Dir::Up.constant(0.5), 0.5);
    assert_eq!(Dir::Up.constant(1e-50), MIN_CONST);
    assert_eq!(Dir::Dn.constant(1e-50), 0.0);
    assert_eq!(Dir::Dn.constant(-1e-300), 0.0);
    assert_eq!(Dir::Up.stored(1e-100), 1e-100);
    assert_eq!(Dir::Up.stored(1e-200), FLOOR);
    assert_eq!(Dir::Dn.stored(1e-200), 0.0);
    assert_eq!(Dir::Up.divisor(3.0), 3.0);
    assert_eq!(Dir::Up.divisor(1e60), MAX_DIVISOR);
    assert_eq!(Dir::Dn.divisor(1e60), f64::INFINITY);
    assert_eq!(take_admitted(), 7);
    assert_eq!(take_admitted(), 0);
    assert!(Dir::Up.constant(f64::NAN).is_nan() && Dir::Dn.stored(f64::NAN).is_nan());
}

/// A setting of the evaluation grid at mean 1/2 with the enclosure of the validated reference
/// implementation (its `grid_fast.csv`): profile, N, rho, d, K, hi, lo and hi as an f64.
type GridRow = (
    &'static str,
    usize,
    &'static str,
    &'static str,
    usize,
    &'static str,
    &'static str,
    &'static str,
);

const GRID: [GridRow; 9] = [
    (
        "det",
        2,
        "1/20",
        "2",
        2,
        "0.007835047190",
        "0.007834166510",
        "7.835047189729489e-3",
    ),
    (
        "det",
        6,
        "7/10",
        "4",
        20,
        "0.579404322903",
        "0.579371200562",
        "5.794043229020304e-1",
    ),
    (
        "exp",
        2,
        "1/20",
        "1",
        2,
        "0.420002658621",
        "0.419998373049",
        "4.200026586200342e-1",
    ),
    (
        "exp",
        4,
        "7/10",
        "3",
        18,
        "0.531581929568",
        "0.531542352533",
        "5.315819295677178e-1",
    ),
    (
        "gamma",
        3,
        "7/10",
        "6",
        16,
        "0.082733448679",
        "0.082678932692",
        "8.273344867875099e-2",
    ),
    (
        "gamma",
        2,
        "1/10",
        "6",
        3,
        "0.000000523442",
        "0.000000000000",
        "5.234412416884476e-7",
    ),
    (
        "tnorm",
        3,
        "7/10",
        "3",
        16,
        "0.373481955446",
        "0.373430964341",
        "3.7348195544576834e-1",
    ),
    (
        "lognormal",
        4,
        "1/5",
        "3",
        5,
        "0.082405985843",
        "0.082375958068",
        "8.240598584256834e-2",
    ),
    (
        "pareto",
        3,
        "3/10",
        "8",
        6,
        "0.001140177325",
        "0.001108429203",
        "1.1401773246508728e-3",
    ),
];

#[test]
fn the_evaluation_grid_is_reproduced() {
    for (p, n, rho, d, k, hi, lo, hi_f) in GRID {
        let law = Law::from_distribution(&profile_law(p, &qr(1, 2), &qr(1, 3)).unwrap()).unwrap();
        let o = enclose(n, rho, &law, d, &default_step(&q(d), &qr(1, 2)), true);
        let what = format!("{p} N={n} rho={rho} d={d}");
        assert_eq!(o.k_cycles, k, "{what}");
        assert_eq!(o.hi, hi, "{what}");
        assert_eq!(o.lo, lo, "{what}");
        assert_eq!(format!("{:e}", o.hi_f), hi_f, "{what}");
        assert_eq!(o.admitted, 0, "{what}");
    }
}

#[test]
fn witness_rows_join_the_map_and_the_verdicts_are_consistent() {
    let root = std::env::temp_dir().join(format!("tdf-witness-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    // deadlines in seconds at mean 1/2, and multiples of I at mean 5 (rounded by the map's step)
    let grids = [
        ("seconds", qr(1, 2), vec![qi(1), qi(2)], vec![]),
        ("over_i", qi(5), vec![qi(1)], vec![qr(1, 2), qi(1)]),
    ];
    let profiles = vec!["det".to_string(), "exp".to_string()];
    let (ns, rhos) = (vec![2], vec![qr(1, 20), qr(3, 10)]);
    for (name, mean, ds, d_over_i) in grids {
        let dir = root.join(name);
        map::run(&map::Args {
            out: dir.join("map"),
            profiles: profiles.clone(),
            ns: ns.clone(),
            rhos: rhos.clone(),
            ds: ds.clone(),
            d_over_i: d_over_i.clone(),
            epsilons: vec![qr(1, 1000), qr(1, 100), qr(1, 20)],
            means: vec![mean.clone()],
            cv2: qr(1, 3),
            step: None,
            sweeps: None,
            bits: None,
            all_subsets: false,
            gap_rows: false,
            cache: None,
            threads: None,
        })
        .unwrap();
        let w = witness::run(&witness::Args {
            out: dir.join("witness"),
            profiles: profiles.clone(),
            ns: ns.clone(),
            rhos: rhos.clone(),
            ds,
            d_over_i,
            means: vec![mean],
            cv2: qr(1, 3),
            step: None,
            map_step: None,
            tol: 1e-7,
            ojf_step: None,
            ojf_above: qr(1, 1000),
            no_ojf: false,
            threads: None,
        })
        .unwrap();
        assert_eq!(w["settings"].as_u64(), Some(8), "{name}");
        assert_eq!(w["admitted_total"].as_u64(), Some(0), "{name}");
        let s = verdicts::run(&verdicts::Args {
            map: dir.join("map").join("map.csv"),
            witness: dir.join("witness").join("witness.csv"),
            out: dir.join("verdicts"),
            no_ojf: false,
        })
        .unwrap();
        assert_eq!(s["map_rows"].as_u64(), Some(24), "{name}");
        for key in [
            "missing_witness",
            "unused_witness_rows",
            "impossible",
            "lb_above_upper",
            "ojf_left_out_but_needed",
        ] {
            assert_eq!(s[key].as_u64(), Some(0), "{name}: {key}");
        }
    }
    // det, N = 2, rho = 1/20, d = 2: hi = 0.0078350... meets 1/100 but not 1/1000
    let text = std::fs::read_to_string(root.join("seconds/verdicts/verdicts.csv")).unwrap();
    let verdict = |id: &str| {
        let line = text
            .lines()
            .find(|l| l.starts_with(&format!("{id},")))
            .unwrap();
        line.split(',').nth(11).unwrap().to_string()
    };
    assert_eq!(
        verdict("det|N=2|rho=1/20|d=2|eps=1/100"),
        "certified_feasible"
    );
    assert_ne!(
        verdict("det|N=2|rho=1/20|d=2|eps=1/1000"),
        "certified_feasible"
    );
    let _ = std::fs::remove_dir_all(&root);
}
