//! Tests of the two-sided workload enclosure of OJF's M/G/1 reduction with load `rho_F` and
//! truncation order `K` (`Homogeneous::workload_bound`): `U_K^- = 1 - C_K^+ - rho_F^{K+1}` and
//! `U_K^+ = 1 - C_K^-` must contain the stationary miss probability `m_F` of OJF. On the exact
//! exponential benchmark the value is known in closed form; for deterministic and gamma
//! reconfiguration times every enclosure contains the same unknown value, so the enclosures of
//! different grid steps must overlap, and their numerical width must shrink with the step.

use tdf::feasibility::distributions::{Model, profile};
use tdf::feasibility::grid::GridSettings;
use tdf::feasibility::rational::{parse_q, qi, qr, to_f64};
use tdf::feasibility::reference::{Homogeneous, WorkloadBound};

fn homogeneous(name: &str, n: usize, rho: &str, d: &str) -> Homogeneous {
    let law = if name == "exp" {
        profile("exp", &qr(1, 2), &qi(1)).unwrap()
    } else {
        profile(name, &qr(1, 2), &qr(1, 3)).unwrap()
    };
    let model = Model::balanced(
        "workload",
        &law,
        n,
        &parse_q(rho).unwrap(),
        &parse_q(d).unwrap(),
        &qr(1, 100),
    );
    Homogeneous::from_model(&model).unwrap()
}

fn enclosure(h: &Homogeneous, step: &str) -> WorkloadBound {
    let bits = GridSettings::default_for(h.n).bits;
    let k = h.workload_terms(bits, 1e-6, 64).unwrap();
    h.workload_bound(&parse_q(step).unwrap(), k, bits).unwrap()
}

#[test]
fn the_enclosure_contains_the_exact_benchmark_and_narrows_with_the_step() {
    // N = 2, mu = 4, tau ~ Exp(2): m_F = exp(-(2 - 4 rho) d).
    for (rho, d) in [("1/100", "2"), ("1/20", "2"), ("1/5", "3"), ("2/5", "2")] {
        let h = homogeneous("exp", 2, rho, d);
        let bits = GridSettings::default_for(2).bits;
        let b = h.exact_exponential_benchmark(bits).unwrap().unwrap();
        let mut last_width = f64::INFINITY;
        for step in ["1/500", "1/1000", "1/2000", "1/4000"] {
            let w = enclosure(&h, step);
            assert!(
                w.u_k_minus <= b.miss.0,
                "rho = {rho}, d = {d}, g = {step}: U_K^- = {} above m_F",
                to_f64(&w.u_k_minus)
            );
            assert!(
                w.u_k_plus >= b.miss.1,
                "rho = {rho}, d = {d}, g = {step}: U_K^+ = {} below m_F",
                to_f64(&w.u_k_plus)
            );
            let width = to_f64(&w.numeric_width);
            assert!(
                width < last_width,
                "rho = {rho}, d = {d}, g = {step}: width {width} did not shrink"
            );
            last_width = width;
        }
    }
}

#[test]
fn deterministic_and_gamma_enclosures_overlap_across_refinements_and_narrow() {
    for name in ["det", "gamma"] {
        for (n, rho, d) in [(2usize, "1/10", "2"), (4, "1/20", "3")] {
            let h = homogeneous(name, n, rho, d);
            let steps = ["1/500", "1/1000", "1/2000", "1/4000"];
            let ws: Vec<WorkloadBound> = steps.iter().map(|s| enclosure(&h, s)).collect();
            for (i, a) in ws.iter().enumerate() {
                assert!(
                    a.u_k_minus <= a.u_k_plus,
                    "{name} N={n}: inverted enclosure at {}",
                    steps[i]
                );
                for b in &ws {
                    // Both contain m_F, so they intersect.
                    assert!(
                        a.u_k_minus <= b.u_k_plus,
                        "{name} N={n} rho={rho} d={d}: enclosures of different steps are disjoint"
                    );
                }
            }
            for i in 1..ws.len() {
                assert!(
                    ws[i].numeric_width < ws[i - 1].numeric_width,
                    "{name} N={n}: width did not shrink at {}",
                    steps[i]
                );
                // A finer grid never loosens the upper end by more than rounding noise.
                assert!(
                    to_f64(&ws[i].u_k_plus) <= to_f64(&ws[i - 1].u_k_plus) + 1e-12,
                    "{name} N={n}: U_K^+ rose at {}",
                    steps[i]
                );
            }
        }
    }
}

#[test]
fn the_upper_end_of_a_deterministic_setting_is_pinned() {
    // Regression value: U_K^+ for deterministic reconfiguration at N = 2, rho = 1/10, d = 2,
    // g = 1/2000, a setting of the finite-load results.
    let h = homogeneous("det", 2, "1/10", "2");
    let w = enclosure(&h, "1/2000");
    assert_eq!(to_f64(&w.u_k_plus).to_string(), "0.010690506475432904");
    assert!(w.u_k_minus <= w.u_k_plus);
}
