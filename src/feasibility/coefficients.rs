//! Phase coefficients and their lattice construction for finite-support
//! reconfiguration-time distributions (laws, in identifiers). `H_ji(t)` is the
//! largest probability that an isolated class-`i` job, with the network idle
//! in configuration `j`, completes within `t` over every sequence of
//! reconfigurations; the coefficients `A`, `B`, `C` and the service-only
//! bound follow from it. Every value is rounded in the direction given by
//! [`Mode`]; [`PhaseCoefficients`] is also the result of the grid
//! construction (`grid.rs`).

use super::model::DiscreteModel;
use super::rational::{Q, exp_neg, exp_neg_powers, rational_gcd, round_down, round_up};
use num_traits::{One, Zero};
use rayon::prelude::*;
use serde_json::{Value, json};

/// The approximation direction. `Upper` bounds `H` from above and thus every
/// unavoidable-miss coefficient from below: the certificate direction, in
/// which the approximated region contains the exact one. `Lower` gives the
/// opposite bounds, which measure the numerical looseness and enclose the
/// coefficients from both sides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Upper,
    Lower,
}

impl Mode {
    /// The label stored with the coefficients.
    pub fn name(&self) -> &'static str {
        match self {
            Mode::Upper => "upper_h_lower_coefficients",
            Mode::Lower => "lower_h_upper_coefficients",
        }
    }
}

/// The time lattice: every support point and every deadline is an integer
/// multiple of `g`, and `g` is at most the requested step.
#[derive(Clone, Debug)]
pub struct Lattice {
    /// Lattice step.
    pub g: Q,
    /// `d_i / g`.
    pub nd: Vec<usize>,
    /// `steps[u][v]`: the distribution of the pair `u -> v` as
    /// `(support point / g, probability)`.
    pub steps: Vec<Vec<Vec<(usize, Q)>>>,
}

/// The lattice of `model` with step `g = g0 / ceil(g0 / target_step)`, where
/// `g0` is the rational gcd of the deadlines and the support points.
pub fn lattice(model: &DiscreteModel, target_step: &Q) -> Result<Lattice, String> {
    let mut values: Vec<Q> = model.d.clone();
    for (u, v) in model.edges() {
        for (s, _) in &model.law(u, v).atoms {
            values.push(s.clone());
        }
    }
    let g0 = rational_gcd(&values);
    if g0.is_zero() {
        return Err("empty lattice".into());
    }
    let ratio = &g0 / target_step;
    let m = ratio.ceil().to_integer();
    let m: i64 = num_traits::ToPrimitive::to_i64(&m).ok_or("lattice refinement too large")?;
    let g = &g0 / Q::from_integer(num_bigint::BigInt::from(m.max(1)));
    let count = |x: &Q| -> Result<usize, String> {
        let r = x / &g;
        if !r.is_integer() {
            return Err(format!("{x} is not a lattice multiple"));
        }
        num_traits::ToPrimitive::to_usize(&r.to_integer())
            .ok_or_else(|| "lattice index overflow".into())
    };
    let nd = model.d.iter().map(count).collect::<Result<Vec<_>, _>>()?;
    let n = model.n;
    let mut steps = vec![vec![Vec::new(); n]; n];
    for (u, v) in model.edges() {
        for (s, p) in &model.law(u, v).atoms {
            steps[u][v].push((count(s)?, p.clone()));
        }
    }
    Ok(Lattice { g, nd, steps })
}

/// `h[j][m]`: a bound on `H_j,target(m g)` for `m = 0..=nd[target]`.
#[derive(Clone, Debug)]
pub struct HTable {
    pub h: Vec<Vec<Q>>,
}

/// Dynamic program over the lattice: `H_ii(t) = 1 - exp(-mu_i t)`,
/// `H_ji(t) = max_k E[H_ki(t - tau_jk)]`, `H = 0` for `t <= 0`, with `i` the
/// target class. Every support point is at least one lattice step, so one
/// pass in increasing `t` suffices. Each value is rounded in the direction of
/// `mode` after it is computed, and the maximum of upper (lower) bounds is an
/// upper (lower) bound of the maximum, so the whole table keeps the direction.
pub fn h_table(
    model: &DiscreteModel,
    lat: &Lattice,
    target: usize,
    mode: Mode,
    bits: u32,
) -> HTable {
    let n = model.n;
    let nd = lat.nd[target];
    let (lo_pow, hi_pow) = exp_neg_powers(&(&model.mu[target] * &lat.g), nd, bits);
    let round = |x: Q| match mode {
        Mode::Upper => round_up(&x, bits),
        Mode::Lower => round_down(&x, bits),
    };
    let mut h = vec![vec![Q::zero(); nd + 1]; n];
    for m in 1..=nd {
        h[target][m] = match mode {
            Mode::Upper => round_up(&(Q::one() - &lo_pow[m]), bits),
            Mode::Lower => round_down(&(Q::one() - &hi_pow[m]), bits),
        };
    }
    for m in 1..=nd {
        for j in 0..n {
            if j == target {
                continue;
            }
            let mut best = Q::zero();
            for k in 0..n {
                if k == j {
                    continue;
                }
                let mut val = Q::zero();
                for (steps, p) in &lat.steps[j][k] {
                    if *steps < m {
                        val += p * &h[k][m - steps];
                    }
                }
                if val > best {
                    best = val;
                }
            }
            h[j][m] = round(best);
        }
    }
    HTable { h }
}

/// Phase coefficients of one model (independent of the arrival rates):
/// `a[j][i] = A_ji`, `b[j][i] = B_ji`, `c[u][v][i] = C_uv,i`, the
/// service-only bound `exp(-mu_i d_i)` of every class and the mean
/// reconfiguration times.
#[derive(Clone, Debug)]
pub struct PhaseCoefficients {
    pub mode: Mode,
    pub bits: u32,
    /// Lattice or grid step.
    pub g: Q,
    pub a: Vec<Vec<Q>>,
    pub b: Vec<Vec<Q>>,
    pub c: Vec<Vec<Vec<Q>>>,
    pub service_only: Vec<Q>,
    /// Lower end of the mean reconfiguration time per pair (exact for
    /// finite-support distributions).
    pub mean_tau: Vec<Vec<Q>>,
    /// Upper end of the mean per pair; equal to `mean_tau` when the mean is exact.
    pub mean_tau_hi: Vec<Vec<Q>>,
}

impl PhaseCoefficients {
    /// Exact serialization (rationals as strings) for the coefficient cache.
    pub fn to_exact_json(&self) -> Value {
        let s = |x: &Q| x.to_string();
        json!({
            "mode": self.mode.name(), "bits": self.bits, "lattice_step": self.g.to_string(),
            "a": self.a.iter().map(|r| r.iter().map(s).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "b": self.b.iter().map(|r| r.iter().map(s).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "c": self.c.iter().map(|r| r.iter().map(|q| q.iter().map(s).collect::<Vec<_>>()).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "service_only": self.service_only.iter().map(s).collect::<Vec<_>>(),
            "mean_tau": self.mean_tau.iter().map(|r| r.iter().map(s).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "mean_tau_hi": self.mean_tau_hi.iter().map(|r| r.iter().map(s).collect::<Vec<_>>()).collect::<Vec<_>>()
        })
    }

    /// The inverse of [`PhaseCoefficients::to_exact_json`]; a missing
    /// `mean_tau_hi` defaults to `mean_tau`.
    pub fn from_exact_json(v: &Value) -> Result<Self, String> {
        let q = |x: &Value| -> Result<Q, String> {
            super::rational::parse_q(x.as_str().ok_or("rational expected")?)
        };
        let vec1 = |x: &Value| -> Result<Vec<Q>, String> {
            x.as_array().ok_or("list expected")?.iter().map(q).collect()
        };
        let vec2 = |x: &Value| -> Result<Vec<Vec<Q>>, String> {
            x.as_array()
                .ok_or("table expected")?
                .iter()
                .map(vec1)
                .collect()
        };
        let vec3 = |x: &Value| -> Result<Vec<Vec<Vec<Q>>>, String> {
            x.as_array()
                .ok_or("cube expected")?
                .iter()
                .map(vec2)
                .collect()
        };
        let mode = match v["mode"].as_str() {
            Some(m) if m == Mode::Upper.name() => Mode::Upper,
            Some(m) if m == Mode::Lower.name() => Mode::Lower,
            _ => return Err("unknown coefficient mode".into()),
        };
        Ok(PhaseCoefficients {
            mode,
            bits: v["bits"].as_u64().ok_or("bits")? as u32,
            g: q(&v["lattice_step"])?,
            a: vec2(&v["a"])?,
            b: vec2(&v["b"])?,
            c: vec3(&v["c"])?,
            service_only: vec1(&v["service_only"])?,
            mean_tau: vec2(&v["mean_tau"])?,
            mean_tau_hi: match v.get("mean_tau_hi") {
                Some(x) => vec2(x)?,
                None => vec2(&v["mean_tau"])?,
            },
        })
    }

    /// Decimal JSON for reports (18 places, rounded toward minus infinity).
    pub fn to_json(&self) -> Value {
        let s = |x: &Q| super::rational::dec(x, 18, false);
        json!({
            "mode": self.mode.name(), "bits": self.bits, "lattice_step": self.g.to_string(),
            "a": self.a.iter().map(|r| r.iter().map(s).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "b": self.b.iter().map(|r| r.iter().map(s).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "c": self.c.iter().map(|r| r.iter().map(|q| q.iter().map(s).collect::<Vec<_>>()).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "service_only": self.service_only.iter().map(s).collect::<Vec<_>>(),
            "mean_tau": self.mean_tau.iter().map(|r| r.iter().map(s).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "mean_tau_hi": self.mean_tau_hi.iter().map(|r| r.iter().map(s).collect::<Vec<_>>()).collect::<Vec<_>>()
        })
    }
}

/// `A_ji = E[1 - H_ji(d_i - R_j)]` with the residual service time
/// `R_j ~ Exp(mu_j)`, `B_ji = 1 - H_ji(d_i)` and
/// `C_uv,i = E[int_0^tau (1 - H_vi(d_i - r)) dr]`, computed on the lattice as
/// `E[tau] - int_0^{d_i} Pr(tau > r) H_vi(d_i - r) dr`.
///
/// Direction (`Mode::Upper`, lower bounds on the coefficients): on a cell
/// `[m g, (m + 1) g)` of the residual `R_j` the completion probability is at
/// most `H(d_i - m g)` and the cell probability at most
/// `hi(exp(-mu_j m g)) - lo(exp(-mu_j (m + 1) g))`; on a cell of the
/// residual reconfiguration time `Pr(tau > r) <= Pr(tau > m g)` and
/// `H(d_i - r) <= H(d_i - m g)`. `Mode::Lower` uses the opposite cell ends.
pub fn phase_coefficients(
    model: &DiscreteModel,
    lat: &Lattice,
    mode: Mode,
    bits: u32,
) -> PhaseCoefficients {
    let n = model.n;
    let g = lat.g.clone();
    let mut mean_tau = vec![vec![Q::zero(); n]; n];
    for (u, v) in model.edges() {
        mean_tau[u][v] = model.law(u, v).mean();
    }
    // One parallel task per target class: its H table and every coefficient
    // for that class.
    let per_target: Vec<(Q, Vec<Q>, Vec<Q>, Vec<Vec<Q>>)> = (0..n)
        .into_par_iter()
        .map(|i| target_coefficients(model, lat, mode, bits, i, &mean_tau))
        .collect();
    let mut a = vec![vec![Q::zero(); n]; n];
    let mut b = vec![vec![Q::zero(); n]; n];
    let mut c = vec![vec![vec![Q::zero(); n]; n]; n];
    let mut service_only = vec![Q::zero(); n];
    for (i, (so, ai, bi, ci)) in per_target.into_iter().enumerate() {
        service_only[i] = so;
        for j in 0..n {
            a[j][i] = ai[j].clone();
            b[j][i] = bi[j].clone();
            for v in 0..n {
                c[j][v][i] = ci[j][v].clone();
            }
        }
    }
    PhaseCoefficients {
        mode,
        bits,
        g,
        a,
        b,
        c,
        service_only,
        mean_tau_hi: mean_tau.clone(),
        mean_tau,
    }
}

/// `(service_only_i, a[.][i], b[.][i], c[.][.][i])` for the target class `i`.
fn target_coefficients(
    model: &DiscreteModel,
    lat: &Lattice,
    mode: Mode,
    bits: u32,
    i: usize,
    mean_tau: &[Vec<Q>],
) -> (Q, Vec<Q>, Vec<Q>, Vec<Vec<Q>>) {
    let n = model.n;
    let g = lat.g.clone();
    let round_coef = |x: Q| match mode {
        Mode::Upper => round_down(&x, bits),
        Mode::Lower => round_up(&x, bits),
    };
    let nd = lat.nd[i];
    let ht = h_table(model, lat, i, mode, bits);
    let (lo_d, hi_d) = exp_neg(&(&model.mu[i] * &model.d[i]), bits);
    let service_only = match mode {
        Mode::Upper => lo_d,
        Mode::Lower => hi_d,
    };
    let mut a = vec![Q::zero(); n];
    let mut b = vec![Q::zero(); n];
    let mut c = vec![vec![Q::zero(); n]; n];
    for j in 0..n {
        b[j] = round_coef(Q::one() - &ht.h[j][nd]);
        let (lo_pow, hi_pow) = exp_neg_powers(&(&model.mu[j] * &g), nd, bits);
        let mut sum = Q::zero();
        for m in 0..nd {
            match mode {
                Mode::Upper => {
                    let w = &hi_pow[m] - &lo_pow[m + 1];
                    sum += w * &ht.h[j][nd - m];
                }
                Mode::Lower => {
                    let w = &lo_pow[m] - &hi_pow[m + 1];
                    if w > Q::zero() {
                        sum += w * &ht.h[j][nd - m - 1];
                    }
                }
            }
        }
        a[j] = round_coef(Q::one() - sum);
    }
    for (u, v) in model.edges() {
        let law = model.law(u, v);
        let mut sum = Q::zero();
        let max_steps = law.max_support() / &g;
        let max_steps =
            num_traits::ToPrimitive::to_usize(&max_steps.ceil().to_integer()).unwrap_or(usize::MAX);
        for m in 0..nd.min(max_steps) {
            match mode {
                Mode::Upper => {
                    let s = law.survival(&(Q::from_integer(num_bigint::BigInt::from(m)) * &g));
                    if s.is_zero() {
                        break;
                    }
                    sum += s * &g * &ht.h[v][nd - m];
                }
                Mode::Lower => {
                    let s = law.survival(&(Q::from_integer(num_bigint::BigInt::from(m + 1)) * &g));
                    if s.is_zero() {
                        break;
                    }
                    sum += s * &g * &ht.h[v][nd - m - 1];
                }
            }
        }
        c[u][v] = round_coef(&mean_tau[u][v] - sum);
    }
    (service_only, a, b, c)
}

#[cfg(test)]
mod tests {
    use super::super::model::{DiscreteLaw, DiscreteModel, two_class_example};
    use super::super::rational::{parse_q, qi, qr, to_f64};
    use super::*;

    fn step() -> Q {
        parse_q("1/1000").unwrap()
    }

    #[test]
    fn lattice_of_the_two_class_example() {
        let m = two_class_example();
        let lat = lattice(&m, &step()).unwrap();
        assert_eq!(lat.g, qr(1, 1000));
        assert_eq!(lat.nd, vec![3000, 3000]);
        assert_eq!(lat.steps[0][1], vec![(450, qr(64, 65)), (3700, qr(1, 65))]);
    }

    #[test]
    fn h_direct_service_matches_exponential_and_deadline_equality_is_zero() {
        let m = two_class_example();
        let lat = lattice(&m, &step()).unwrap();
        let ht = h_table(&m, &lat, 0, Mode::Upper, 128);
        // H_00(3) = 1 - e^{-12}
        let h = to_f64(&ht.h[0][3000]);
        assert!((h - (1.0 - (-12.0f64).exp())).abs() < 1e-12);
        // H_10(t) = E[H_00(t - tau)]: at t = 0.45 the short reconfiguration ends
        // exactly at the deadline, so H_10(0.45) = 0.
        assert_eq!(ht.h[1][450], Q::zero());
        // at t = 3: (64/65)(1 - e^{-4 * 2.55})
        let expect = 64.0 / 65.0 * (1.0 - (-10.2f64).exp());
        assert!((to_f64(&ht.h[1][3000]) - expect).abs() < 1e-12);
        let lo = h_table(&m, &lat, 0, Mode::Lower, 128);
        assert!(lo.h[1][3000] <= ht.h[1][3000]);
    }

    #[test]
    fn multi_hop_route_beats_a_slow_direct_edge() {
        // 0 -> 2 direct takes 3.7 always; 0 -> 1 -> 2 takes 0.1 + 0.1.
        let n = 3;
        let slow = DiscreteLaw::point(qr(37, 10));
        let fast = DiscreteLaw::point(qr(1, 10));
        let laws: Vec<Vec<Option<DiscreteLaw>>> = (0..n)
            .map(|u| {
                (0..n)
                    .map(|v| {
                        (u != v).then(|| {
                            if (u, v) == (0, 2) {
                                slow.clone()
                            } else {
                                fast.clone()
                            }
                        })
                    })
                    .collect()
            })
            .collect();
        let m = DiscreteModel::new(
            "route",
            vec![qi(1); 3],
            vec![qi(4); 3],
            vec![qi(3); 3],
            vec![qr(1, 100); 3],
            laws,
        )
        .unwrap();
        let lat = lattice(&m, &step()).unwrap();
        let ht = h_table(&m, &lat, 2, Mode::Upper, 128);
        let two_hop = 1.0 - (-4.0f64 * 2.8).exp();
        assert!(
            (to_f64(&ht.h[0][3000]) - two_hop).abs() < 1e-12,
            "{}",
            to_f64(&ht.h[0][3000])
        );
        // a recursion restricted to direct reconfigurations would give zero here
        assert!(to_f64(&ht.h[0][3000]) > 0.99);
    }

    #[test]
    fn two_class_coefficients_bracket_the_closed_forms() {
        let m = two_class_example();
        let lat = lattice(&m, &qr(1, 4000)).unwrap();
        let lo = phase_coefficients(&m, &lat, Mode::Upper, 128);
        let hi = phase_coefficients(&m, &lat, Mode::Lower, 128);
        let e12 = (-12.0f64).exp();
        let e102 = (-10.2f64).exp();
        // B_00 = e^{-12}; B_10 = (64/65) e^{-10.2} + 1/65
        let b00 = e12;
        let b10 = 64.0 / 65.0 * e102 + 1.0 / 65.0;
        // A_00 = Pr(S1+S2>3) = 13 e^{-12}; A_10 = (64/65) 11.2 e^{-10.2} + 1/65
        let a00 = 13.0 * e12;
        let a10 = 64.0 / 65.0 * 11.2 * e102 + 1.0 / 65.0;
        // C_{01,1} = E int_0^tau Pr(S > 3 - r) dr ; C_{01,0} = E int_0^tau Pr(tau' + S > 3 - r) dr
        let c011 = 64.0 / 65.0 * (e12 * ((1.8f64).exp() - 1.0) / 4.0)
            + 1.0 / 65.0 * ((1.0 - e12) / 4.0 + 0.7);
        let c010 = 64.0 / 65.0 * (64.0 / 65.0 * e102 * ((1.8f64).exp() - 1.0) / 4.0 + 0.45 / 65.0)
            + 1.0 / 65.0 * (64.0 / 65.0 * ((1.0 - e102) / 4.0 + 1.15) + 3.7 / 65.0);
        for (name, lo_v, hi_v, exact) in [
            ("B00", &lo.b[0][0], &hi.b[0][0], b00),
            ("B10", &lo.b[1][0], &hi.b[1][0], b10),
            ("A00", &lo.a[0][0], &hi.a[0][0], a00),
            ("A10", &lo.a[1][0], &hi.a[1][0], a10),
            ("C011", &lo.c[0][1][1], &hi.c[0][1][1], c011),
            ("C010", &lo.c[0][1][0], &hi.c[0][1][0], c010),
        ] {
            let (l, h) = (to_f64(lo_v), to_f64(hi_v));
            assert!(
                l <= exact * (1.0 + 1e-12) && exact * (1.0 - 1e-12) <= h,
                "{name}: {l} <= {exact} <= {h}"
            );
            // the certificate direction (lower bound) is tight; the diagnostic upper
            // bound may lose one lattice cell of probability mass at every atom
            assert!(
                exact - l <= 0.005 * exact.max(1e-9) + 1e-12,
                "{name}: lower bound too loose {l} vs {exact}"
            );
        }
        assert!((to_f64(&lo.service_only[0]) - e12).abs() < 1e-14);
    }

    #[test]
    fn deterministic_switch_coefficients_are_closed_form() {
        let point = || Some(DiscreteLaw::point(qr(1, 2)));
        let m = DiscreteModel::new(
            "det",
            vec![qi(1); 2],
            vec![qi(4); 2],
            vec![qi(3); 2],
            vec![qr(1, 100); 2],
            vec![vec![None, point()], vec![point(), None]],
        )
        .unwrap();
        let lat = lattice(&m, &qr(1, 4000)).unwrap();
        let lo = phase_coefficients(&m, &lat, Mode::Upper, 128);
        let hi = phase_coefficients(&m, &lat, Mode::Lower, 128);
        let b10 = (-4.0f64 * 2.5).exp();
        assert!(to_f64(&lo.b[1][0]) <= b10 && b10 <= to_f64(&hi.b[1][0]));
        // C_{01,1} = int_0^{0.5} e^{-4(3 - r)} dr = e^{-12}(e^{2} - 1)/4
        let c = (-12.0f64).exp() * ((2.0f64).exp() - 1.0) / 4.0;
        assert!(to_f64(&lo.c[0][1][1]) <= c && c <= to_f64(&hi.c[0][1][1]));
    }
}
