//! Visit-gap constraints of the relaxation (gap rows, in identifiers): lower bounds on the miss
//! probability of a class from the time the network spends away from its configuration. A gap
//! of class `i` is a maximal interval outside configuration `i` (neither serving class `i` nor
//! idling in `i`). Gaps end at the entries into `i`, which occur at the rate
//! `eta_i = sum_{u != i} nu_ui`, and cover the fraction `f_i = 1 - rho_i - z_i` of the time, so
//! the Palm mean gap is `f_i / eta_i`. Two constraint groups:
//!
//! * `gap_window` ([`window_single`]): in a window of fixed length, a class-`i` arrival that
//!   finds the network away from configuration `i` misses unless `i` is entered in time;
//! * `gap` ([`weak`], [`strong`]): a job that arrives more than `d_i` before the end of its gap
//!   misses, and so does a job that arrives after the gap while the class-`i` work left by the
//!   gap still exceeds `d_i`. A policy may end a gap because class-`i` jobs have just
//!   arrived, so these counts are corrected by maximal inequalities for the Poisson arrivals at
//!   the policy's decisions; the corrected bounds are convex in `(eta_i, f_i)`, and their tangent
//!   planes at a declared set of mean gaps are the rows;
//! * `gap`, U rows ([`weak_u`], [`strong_u`]): the same two counts with the maximal inequalities
//!   replaced by optimal-stopping bounds, the expected supremum of a Poisson process with drift
//!   for the weak rows, and for the strong rows the value of stopping Page's CUSUM statistic of
//!   the arrivals (solved with the scale function, [`cusum_value_upper`]) and a bound on the
//!   drawdown of the class-`i` work. These rows are linear for fixed parameters; at each declared
//!   mean gap the parameters that maximize the row are taken from a fixed grid.
//!
//! Every row is linear in `(x_i, z_i, nu_ui)` and holds for every admissible policy. Each
//! approximated constant is rounded in the direction that weakens its row (`f_i >= 0` and
//! `nu_ui >= 0` at every policy), so the rows keep the exact region inside the relaxation.

use super::model::DiscreteModel;
use super::rational::{Q, exp_neg, qi, qr, round_down, round_up, to_f64};
use super::special::{exp_enclosure, sqrt_enclosure};
use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Fractional bits of the tangent parameters and window lengths, which are chosen in doubles
/// and enter the rows exactly.
const PARAM_BITS: u32 = 12;

/// Fractional bits of the rounded coefficients of the rows (see [`gap_rows`]).
pub const ROW_BITS: u32 = 32;

/// The tangent mean gaps of the weak and strong rows, `2^(k/2) G_i`; the U rows take their
/// parameters at the same mean gaps.
const TANGENT_EXPONENTS: [i32; 6] = [-1, 0, 1, 2, 3, 4];

/// The mean gaps `2^(k/2) G_i` at which the window rows get their best window length.
const WINDOW_EXPONENTS: [i32; 3] = [0, 1, 2];

/// Steps of the parameter grids of the U rows: `eps = k / U_STEPS` for the weak U-rows, and
/// `c = k / U_STEPS` and `delta = rho_i k / U_STEPS` for the S-U rows, `k = 1..U_STEPS`.
const U_STEPS: usize = 400;

/// The values of `t` of the S-U rows as fractions: the grid of the symmetric analysis, with its
/// end point `t = 0`, where the CUSUM bound does not apply, replaced by `10^-4` and `10^-3`.
const T_GRID: [(i64, i64); 16] = [
    (1, 10_000),
    (1, 1000),
    (1, 100),
    (1, 50),
    (1, 20),
    (1, 10),
    (3, 20),
    (1, 5),
    (3, 10),
    (2, 5),
    (1, 2),
    (3, 5),
    (7, 10),
    (4, 5),
    (9, 10),
    (1, 1),
];

/// One visit-gap constraint of class `class`,
/// `x x_i + z z_i + sum_{u != i} entry_u nu_ui >= rhs`.
#[derive(Clone, Debug)]
pub struct GapRow {
    /// The constraint group: `gap_window` or `gap`.
    pub group: &'static str,
    /// Family, class and tangent parameter.
    pub label: String,
    pub class: usize,
    /// Coefficient of `x_i`.
    pub x: Q,
    /// Coefficient of `z_i`.
    pub z: Q,
    /// Coefficient of `nu_ui` for every `u` (zero at `u = i`).
    pub entry: Vec<Q>,
    pub rhs: Q,
}

/// The row `cx x_i >= a f_i - sum_u b_u nu_ui` in the variables of the relaxation:
/// `cx x_i + a z_i + sum_u b_u nu_ui >= a (1 - rho_i)` (exact, since `f_i = 1 - rho_i - z_i`).
fn row(
    model: &DiscreteModel,
    group: &'static str,
    label: String,
    i: usize,
    cx: Q,
    a: Q,
    b: Vec<Q>,
) -> GapRow {
    let rhs = &a * (Q::one() - model.rho(i));
    GapRow {
        group,
        label,
        class: i,
        x: cx,
        z: a,
        entry: b,
        rhs,
    }
}

/// `v > 0` as a dyadic rational with `PARAM_BITS` fractional bits (rounded down), `None` when
/// that is zero or `v` is not finite.
fn param(v: f64) -> Option<Q> {
    let q = round_down(&Q::from_float(v)?, PARAM_BITS);
    q.is_positive().then_some(q)
}

/// Window rows of class `i` for the window lengths `xs` (group `gap_window`). At a fixed time,
/// the network is away from configuration `i` with probability `f_i`, independently of the
/// class-`i` arrivals of the next `x` time units, at least one of which occurs with probability
/// `1 - exp(-lambda_i x)`. The first of them either misses or completes by `x + d_i`, which
/// takes an entry into `i` within `x + d_i` time units. Entries occur at rate `eta_i`, and the
/// jobs arriving in the window miss at rate `lambda_i x_i`, so
///
/// ```text
/// lambda_i x x_i + (x + d_i) eta_i >= (1 - exp(-lambda_i x)) f_i.
/// ```
///
/// The exponential is enclosed with `bits` fractional bits and the lower end of
/// `1 - exp(-lambda_i x)` is used. With `x_j = h - d_j`, the rows of the classes of a subset
/// with `h > d_j` sum to a row that implies the window constraint of length `h` on that subset
/// (`windows.rs`), and also its form that counts only the entries into the subset.
pub fn window_single(model: &DiscreteModel, i: usize, xs: &[Q], bits: u32) -> Vec<GapRow> {
    let mut out = Vec::new();
    for x in xs {
        if !x.is_positive() {
            continue;
        }
        let lam_x = &model.lam[i] * x;
        let (_, hi) = exp_neg(&lam_x, bits);
        let p = Q::one() - hi;
        if !p.is_positive() {
            continue;
        }
        let entry = (0..model.n)
            .map(|u| if u == i { Q::zero() } else { x + &model.d[i] })
            .collect();
        out.push(row(
            model,
            "gap_window",
            format!("gap_window[{i}] x={x}"),
            i,
            lam_x,
            p,
            entry,
        ));
    }
    out
}

/// Weak gap rows of class `i` for the parameters `kappas` in `(0, 1)` (group `gap`). A
/// class-`i` job that arrives more than `d_i` before the end of its gap misses. The gap ends
/// when the reconfiguration into `i` started at the policy's decision time `c` completes; its
/// duration is independent of the past and of the later arrivals, so Doob's maximal inequality
/// is needed only for the arrivals up to `c`. With the Palm mean gap `f_i / eta_i`, the mean
/// time `(f_i - sum_u nu_ui E[tau_ui]) / eta_i` from the start of a gap to the decision, and
/// Jensen's inequality,
///
/// ```text
/// x_i >= f_i - eta_i d_i - 2 sqrt(eta_i (f_i - sum_u nu_ui E[tau_ui]) / lambda_i),
/// ```
///
/// and `2 sqrt(a b) <= kappa a + b / kappa` gives, for every `kappa > 0`,
///
/// ```text
/// x_i >= (1 - kappa) f_i - eta_i (d_i + 1 / (kappa lambda_i)) + kappa sum_u nu_ui E[tau_ui],
/// ```
///
/// which implies the row without the last sum, the visit-gap bound with the effective
/// threshold `d_i + 1 / (kappa lambda_i)`. `mean_tau[u][i]` must not exceed `E[tau_ui]`, and the
/// coefficients of `nu_ui` are rounded up with `bits` fractional bits.
pub fn weak(
    model: &DiscreteModel,
    mean_tau: &[Vec<Q>],
    i: usize,
    kappas: &[Q],
    bits: u32,
) -> Vec<GapRow> {
    let mut out = Vec::new();
    for kappa in kappas {
        if !kappa.is_positive() || *kappa >= Q::one() {
            continue;
        }
        let threshold = &model.d[i] + (kappa * &model.lam[i]).recip();
        let entry = (0..model.n)
            .map(|u| {
                if u == i {
                    Q::zero()
                } else {
                    round_up(&(&threshold - kappa * &mean_tau[u][i]), bits)
                }
            })
            .collect();
        out.push(row(
            model,
            "gap",
            format!("gap_weak[{i}] kappa={kappa}"),
            i,
            Q::one(),
            Q::one() - kappa,
            entry,
        ));
    }
    out
}

/// Strong gap rows of class `i` for the parameters `ss > 0` (group `gap`). A class-`i` job that
/// arrives after a gap while the class-`i` work exceeds `d_i` misses as well. Rate conservation
/// for this work over the visit that follows the gap counts `E(W - d_i)^+ / (1 - rho_i)` such
/// arrivals per unit of `lambda_i`, where `W` is the work that arrived during the gap; Doob's
/// inequality bounds the fluctuation of `W` around `rho_i` times the gap length and that of the
/// arrivals, which gives
///
/// ```text
/// x_i >= (f_i - eta_i d_i - (2 / mu_i) sqrt(2 lambda_i eta_i f_i)) / (1 - rho_i)
///        - 2 sqrt(eta_i f_i / lambda_i).
/// ```
///
/// With `A_i = 2 sqrt(2 lambda_i) / (mu_i (1 - rho_i)) + 2 / sqrt(lambda_i)` and
/// `sqrt(eta f) <= (eta s + f / s) / 2`, every `s > 0` gives the row tangent at the mean gap
/// `s^2`,
///
/// ```text
/// x_i >= (1 / (1 - rho_i) - A_i / (2 s)) f_i - (d_i / (1 - rho_i) + A_i s / 2) eta_i.
/// ```
///
/// `A_i` is replaced by an upper bound, the coefficient of `f_i` is rounded down and that of
/// `eta_i` up, with `bits` fractional bits. Rows whose coefficient of `f_i` is not positive are
/// vacuous and dropped; there are none for `rho_i >= 1`.
pub fn strong(model: &DiscreteModel, i: usize, ss: &[Q], bits: u32) -> Vec<GapRow> {
    let rho_i = model.rho(i);
    if rho_i >= Q::one() {
        return Vec::new();
    }
    let lam = &model.lam[i];
    let one_minus_rho = Q::one() - &rho_i;
    let work = bits + 16;
    let root_2lam = sqrt_enclosure(&(qi(2) * lam), work).1;
    let root_inv_lam = sqrt_enclosure(&lam.recip(), work).1;
    let a_up = round_up(
        &(qi(2) * root_2lam / (&model.mu[i] * &one_minus_rho) + qi(2) * root_inv_lam),
        bits,
    );
    let inv = one_minus_rho.recip();
    let mut out = Vec::new();
    for s in ss {
        if !s.is_positive() {
            continue;
        }
        let a = round_down(&(&inv - &a_up / (qi(2) * s)), bits);
        if !a.is_positive() {
            continue;
        }
        let b = round_up(&(&model.d[i] * &inv + &a_up * s / qi(2)), bits);
        let entry = (0..model.n)
            .map(|u| if u == i { Q::zero() } else { b.clone() })
            .collect();
        out.push(row(
            model,
            "gap",
            format!("gap_strong[{i}] s={s}"),
            i,
            Q::one(),
            a,
            entry,
        ));
    }
    out
}

/// The positive root `beta(eps)` of `1 - exp(-beta) = (1 - eps) beta` for `0 < eps < 1`, in
/// doubles (bisection).
fn prophet_rate_f64(eps: f64) -> f64 {
    let (mut lo, mut hi) = (0.0f64, 1.0 / (1.0 - eps));
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if 1.0 - (-mid).exp() - (1.0 - eps) * mid > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// A certified lower bound on `beta(eps)`, the positive root of `1 - exp(-beta) = (1 - eps) beta`
/// for `0 < eps < 1`, dyadic with 64 fractional bits. The function `1 - exp(-b) - (1 - eps) b`
/// is concave and zero at `b = 0` with slope `eps > 0`, so it is positive exactly on
/// `(0, beta)`: a point `b > 0` at which it stays positive with the upper enclosure of
/// `exp(-b)` lies below `beta`. The root is located in doubles and the point checked exactly.
pub fn prophet_rate_lower(eps: &Q) -> Q {
    assert!(
        eps.is_positive() && *eps < Q::one(),
        "eps must lie in (0, 1)"
    );
    let one_minus = Q::one() - eps;
    let mut b = prophet_rate_f64(to_f64(eps)) * (1.0 - 1e-9);
    loop {
        if let Some(q) = Q::from_float(b).map(|q| round_down(&q, 64))
            && q.is_positive()
        {
            let (_, e_hi) = exp_neg(&q, 96);
            if (Q::one() - e_hi - &one_minus * &q).is_positive() {
                return q;
            }
        }
        b *= 0.999;
    }
}

/// The lower bound `(m - d_i)^+` on `E(tau_ui - d_i)^+` (Jensen) when the lower bounds
/// `m = mean_tau[u][i]` of the mean reconfiguration times into `i` agree for every origin `u`,
/// and zero otherwise.
fn reconfiguration_excess(model: &DiscreteModel, mean_tau: &[Vec<Q>], i: usize) -> Q {
    let mut means = (0..model.n).filter(|u| *u != i).map(|u| &mean_tau[u][i]);
    match means.next() {
        Some(m) if means.all(|x| x == m) && *m > model.d[i] => m - &model.d[i],
        _ => Q::zero(),
    }
}

/// Weak U-rows of class `i` for the parameters `epss` in `(0, 1)` (group `gap`): the count of
/// the weak rows with the maximal inequality replaced by the expected supremum of a Poisson
/// process with drift. Let a gap of class `i` start at `a`, and let the reconfiguration that
/// ends it start at the decision time `c` from configuration `u` and end at `b = c + tau_ui`.
/// Every class-`i` job that arrives in `(a, b - d_i]` misses. For `gamma = eps lambda_i` the
/// supremum `S = sup_{s >= 0} [(lambda_i - gamma) s - N((a, a + s])]` of the class-`i`
/// arrivals `N` is exponential with rate `beta(eps)`, the positive root of
/// `1 - exp(-beta) = (1 - eps) beta`, since the arrivals after `a` are Poisson whatever the
/// policy did before. Pathwise, at least `(lambda_i - gamma) (b - d_i - a) - S` jobs arrive in
/// `(a, b - d_i]` when `tau_ui <= d_i`, and at least `(lambda_i - gamma) (c - a) - S` in
/// `(a, c]` otherwise, when the arrivals in `(c, b - d_i]` add `lambda_i E(tau_ui - d_i)^+` in
/// expectation, as `c` is a stopping time and `tau_ui` is drawn independently of the past and
/// of the later arrivals. With the Palm mean gap `f_i / eta_i`, for every `eps`,
///
/// ```text
/// x_i >= (1 - eps) (f_i - eta_i d_i) + eps sum_u nu_ui E(tau_ui - d_i)^+ - eta_i / (lambda_i beta(eps)).
/// ```
///
/// With one law into `i` this is the closed-form U-row of the symmetric analysis. The term
/// with `E(tau_ui - d_i)^+` uses the lower bound of `reconfiguration_excess` (zero unless the
/// mean bounds of all origins agree, which only weakens the row), `beta(eps)` is replaced by
/// the lower bound [`prophet_rate_lower`], and the coefficients of `nu_ui` are rounded up with
/// `bits` fractional bits.
pub fn weak_u(
    model: &DiscreteModel,
    mean_tau: &[Vec<Q>],
    i: usize,
    epss: &[Q],
    bits: u32,
) -> Vec<GapRow> {
    let mut out = Vec::new();
    if !model.lam[i].is_positive() {
        return out;
    }
    let excess = reconfiguration_excess(model, mean_tau, i);
    for eps in epss {
        if !eps.is_positive() || *eps >= Q::one() {
            continue;
        }
        let beta = prophet_rate_lower(eps);
        let one_minus = Q::one() - eps;
        let b = round_up(
            &(&one_minus * &model.d[i] - eps * &excess + (&model.lam[i] * &beta).recip()),
            bits,
        );
        let entry = (0..model.n)
            .map(|u| if u == i { Q::zero() } else { b.clone() })
            .collect();
        out.push(row(
            model,
            "gap",
            format!("gap_weak_u[{i}] eps={eps}"),
            i,
            Q::one(),
            one_minus,
            entry,
        ));
    }
    out
}

/// The scale function `W` of `beta s - N(s)` for a Poisson process `N` of rate one, in doubles
/// (it only places the S-U rows): `W(0) = 1 / beta`, `W(n) = e^a sum_{k<n} (-a)^k / k!
/// W(n - 1 - k)` at the integers and `W(n + y) = e^(a y) sum_{k<=n} (-a y)^k / k! W(n - k)`
/// for `0 <= y < 1`, `a = 1 / beta`.
struct ScaleF64 {
    a: f64,
    beta: f64,
    wn: Vec<f64>,
}

impl ScaleF64 {
    fn new(beta: f64) -> Self {
        ScaleF64 {
            a: 1.0 / beta,
            beta,
            wn: vec![1.0 / beta],
        }
    }

    fn w(&mut self, x: f64) -> f64 {
        let n = x.floor() as usize;
        for m in self.wn.len()..=n {
            let (mut s, mut term) = (0.0, 1.0);
            for k in 0..m {
                s += term * self.wn[m - 1 - k];
                term *= -self.a / (k as f64 + 1.0);
                if term == 0.0 {
                    break;
                }
            }
            self.wn.push(self.a.exp() * s);
        }
        let y = x - n as f64;
        let (mut s, mut term) = (0.0, 1.0);
        for k in 0..=n {
            s += term * self.wn[n - k];
            term *= -self.a * y / (k as f64 + 1.0);
            if term == 0.0 {
                break;
            }
        }
        (self.a * y).exp() * s
    }
}

/// `(V, r)` in doubles for the CUSUM stopping problem of [`cusum_value_upper`] at `(c, t)`: the
/// point `r` with `delta W(r) = 1`, `delta = c t`, and `V = delta Wdot(r)`.
fn cusum_value_f64(c: f64, t: f64) -> (f64, f64) {
    let (beta, delta) = (1.0 - c * (1.0 - t), c * t);
    let mut sc = ScaleF64::new(beta);
    let mut hi = 1.0f64;
    while delta * sc.w(hi) < 1.0 {
        hi *= 2.0;
        if hi > 1e5 {
            return (f64::INFINITY, f64::INFINITY);
        }
    }
    let mut lo = 0.0f64;
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if delta * sc.w(mid) < 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let wdot: f64 = (0..=hi.floor() as usize)
        .map(|j| sc.beta * sc.w(hi - j as f64) - 1.0)
        .sum();
    (delta * wdot, hi)
}

/// A real number enclosed as `[lo, hi] 2^-p` (fixed point, for the scale-function series).
type Fixed = (BigInt, BigInt);

/// The enclosure of the rational `x` with `p` fractional bits.
fn fixed(x: &Q, p: u32) -> Fixed {
    let s = x * Q::from_integer(BigInt::one() << p);
    (s.floor().to_integer(), s.ceil().to_integer())
}

/// The product of two enclosures of nonnegative numbers, rounded outward.
fn fixed_mul(a: &Fixed, b: &Fixed, p: u32) -> Fixed {
    assert!(!a.0.is_negative() && !b.0.is_negative());
    let hi = &a.1 * &b.1 + ((BigInt::one() << p) - 1u32);
    ((&a.0 * &b.0) >> p, hi >> p)
}

/// The quotient of an enclosure of a nonnegative number by a positive integer, rounded outward.
fn fixed_div(a: &Fixed, k: u64) -> Fixed {
    (&a.0 / k, (&a.1 + (k - 1)) / k)
}

/// Enclosures of `beta W(r - i)` for `i = 0..=n`, `n = floor(r)`, with `p` fractional bits, by
/// the explicit form of the scale function at rate one,
/// `beta W(r - i) = sum_{k=i}^{n} (-1)^(k-i) z_k^(k-i) / (k-i)! e^(z_k)`, `z_k = a (r - k)`,
/// `a = 1 / beta`. The terms are nonnegative and enclosed with outward rounding (the
/// exponentials from [`exp_enclosure`], `e^(z_k) = e^(z_n) (e^a)^(n-k)`), so the alternating
/// sums are enclosed; the cancellation costs about `2.9 a r` bits, which `p` must cover.
fn beta_w_series(a: &Q, r: &Q, n: usize, p: u32) -> Vec<Fixed> {
    let enclose = |e: (Q, Q)| (fixed(&e.0, p).0, fixed(&e.1, p).1);
    let zn = a * (r - qi(n as i64));
    let ea = enclose(exp_enclosure(a, p));
    let mut ek = enclose(exp_enclosure(&zn, p));
    let one: Fixed = (BigInt::one() << p, BigInt::one() << p);
    let mut out: Vec<Fixed> = vec![(BigInt::zero(), BigInt::zero()); n + 1];
    for k in (0..=n).rev() {
        if k < n {
            ek = fixed_mul(&ek, &ea, p);
        }
        let zk = fixed(&(&zn + a * qi((n - k) as i64)), p);
        let mut power = one.clone();
        for j in 0..=k {
            if j > 0 {
                power = fixed_div(&fixed_mul(&power, &zk, p), j as u64);
            }
            let term = fixed_mul(&power, &ek, p);
            let o = &mut out[k - j];
            if j % 2 == 0 {
                o.0 += &term.0;
                o.1 += &term.1;
            } else {
                o.0 -= &term.1;
                o.1 -= &term.0;
            }
        }
    }
    out
}

/// The certified CUSUM values by `(c, t)` (shared by all classes and models of a process).
fn cusum_cache() -> &'static Mutex<HashMap<(Q, Q), Option<Q>>> {
    static CACHE: OnceLock<Mutex<HashMap<(Q, Q), Option<Q>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A certified upper bound on `V(c, t) = sup_sigma E[R(sigma) - c sigma]` for `0 < c < 1` and
/// `0 < t <= 1` (`None` when it cannot be certified), where `R(s) = sup_{u <= s} [N((u, s]) -
/// beta (s - u)]` is Page's CUSUM statistic of a Poisson process `N` of rate one with
/// `beta = 1 - c (1 - t)` and `sigma` ranges over the stopping times with finite mean of any
/// filtration in which `N` has independent increments. In units of `1 / lambda_i`, this is the
/// value for the class-`i` arrivals with `gamma' = c lambda_i` and `beta' = beta lambda_i`.
/// Two bounds are certified and the smaller is returned:
///
/// * Theorem C (verification with the scale function `W` of `beta s - N(s)`): with
///   `delta = beta - 1 + c = c t > 0`, every `r > 0` with `delta W(r) >= 1` gives
///   `V <= delta Wdot(r)`, where `Wdot(r) = sum_{j <= r} (beta W(r - j) - 1)`. The point `r` is
///   found in doubles slightly above the root of `delta W(r) = 1`; `beta W(r - j)` is enclosed
///   by `beta_w_series`, and the condition and the bound use only the certified ends. The
///   precision grows until the enclosure of `W(r)` is narrow, and `r` until the condition
///   holds. The series has about `r^2` terms, so this bound is computed only for
///   `r <= SERIES_CAP`.
/// * the prophet bound, for `t < 1`: `E[R(sigma) - c sigma] = E sup_{s <= sigma} (beta s - N(s))
///   - delta E sigma` is at most the mean `1 / Phi` of the overall supremum of `beta s - N(s)`,
///   exponential with the rate `Phi = beta(c (1 - t))` of [`prophet_rate_lower`] (the root of
///   `beta Phi = 1 - exp(-Phi)`). It is close to the first bound for small `t`, where `r` is large.
///
/// Values are cached per `(c, t)`.
pub fn cusum_value_upper(c: &Q, t: &Q) -> Option<Q> {
    let key = (c.clone(), t.clone());
    if let Some(v) = cusum_cache().lock().unwrap().get(&key) {
        return v.clone();
    }
    let v = cusum_value_certified(c, t);
    cusum_cache().lock().unwrap().insert(key, v.clone());
    v
}

/// The largest root `r` of `delta W(r) = 1` for which the CUSUM bound is computed.
const SERIES_CAP: f64 = 600.0;

fn cusum_value_certified(c: &Q, t: &Q) -> Option<Q> {
    if !(c.is_positive() && *c < Q::one() && t.is_positive() && *t <= Q::one()) {
        return None;
    }
    let prophet = (*t < Q::one()).then(|| prophet_rate_lower(&(c * (Q::one() - t))).recip());
    match (prophet, cusum_series_bound(c, t)) {
        (Some(p), Some(s)) => Some(if p < s { p } else { s }),
        (p, s) => p.or(s),
    }
}

/// `V(c, t)` in doubles as [`cusum_value_upper`] certifies it: the CUSUM value when its root
/// `r` is at most `SERIES_CAP`, and the prophet bound for `t < 1`, whichever is smaller.
fn cusum_value_used_f64(c: f64, t: f64) -> f64 {
    let prophet = if t < 1.0 {
        1.0 / prophet_rate_f64(c * (1.0 - t))
    } else {
        f64::INFINITY
    };
    let (v, r) = cusum_value_f64(c, t);
    if r <= SERIES_CAP {
        v.min(prophet)
    } else {
        prophet
    }
}

/// The certified CUSUM bound of [`cusum_value_upper`] (`None` beyond `SERIES_CAP`).
fn cusum_series_bound(c: &Q, t: &Q) -> Option<Q> {
    let beta = Q::one() - c * (Q::one() - t);
    let delta = c * t;
    let a = beta.recip();
    let (_, r_f) = cusum_value_f64(to_f64(c), to_f64(t));
    if r_f.is_nan() || r_f > SERIES_CAP {
        return None;
    }
    let mut r = round_up(&Q::from_float(r_f * (1.0 + 1e-9) + 1e-12)?, 40);
    let mut p = 0u32;
    let mut step = 1e-6;
    for _ in 0..48 {
        let n = r.floor().to_integer().to_usize()?;
        if n as f64 > 2.0 * SERIES_CAP {
            return None;
        }
        let bits_n = usize::BITS - (n + 1).leading_zeros();
        p = p.max(96 + (2.9 * to_f64(&(&a * &r))).ceil() as u32 + 2 * bits_n);
        let bw = beta_w_series(&a, &r, n, p);
        let unit = Q::from_integer(BigInt::one() << p);
        let w_lo = Q::from_integer(bw[0].0.clone()) / &unit;
        if &delta * &w_lo >= beta {
            let sum_hi: BigInt = bw.iter().map(|x| &x.1).sum();
            let wdot_hi = Q::from_integer(sum_hi) / &unit - qi(n as i64 + 1);
            return Some(round_up(&(&delta * wdot_hi), 64));
        }
        let w_hi = Q::from_integer(bw[0].1.clone()) / &unit;
        if to_f64(&(&w_hi - &w_lo)) > 1e-12 * to_f64(&w_hi) {
            if p > 1 << 16 {
                return None;
            }
            p *= 2;
        } else {
            r = round_up(&(&r * Q::from_float(1.0 + step)?), 40);
            step *= 2.0;
        }
    }
    None
}

/// S-U rows of class `i` for the parameters `(c, t, delta)` with `0 < c < 1`, `0 < t <= 1` and
/// `0 < delta < rho_i` (group `gap`): the count of the strong rows with both maximal
/// inequalities replaced by optimal-stopping bounds. Consider, under the Palm distribution at
/// the gap starts, a gap `(a, b)` of mean `G = f_i / eta_i` and the visit that follows. Let
/// `Y(v)` be the class-`i` work arrived in `(a, a + v)` minus `rho_i v`, `Y_min <= 0` its
/// minimum over the gap and `z = d_i - rho_i (b - a) - Y_min`. The jobs that arrive in `(a, T]`,
/// where `b - T <= z^+ / (1 - rho_i)`, miss, and so do the jobs that arrive in the following
/// visit while the class-`i` work exceeds `d_i`; rate conservation for that work counts at least
/// `lambda_i E(-z)^+ / (1 - rho_i)` of them, so the late jobs `L` of the cycle satisfy
/// `E L >= E N((a, T]) + lambda_i E(-z)^+ / (1 - rho_i)`. The arrivals in `(T, b]` are at most
/// `beta' (b - T)` plus Page's CUSUM statistic of the arrivals at drift
/// `beta' = lambda_i (1 - c (1 - t))` at time `b`, a stopping time, whose mean is at most
/// `c lambda_i G + V(c, t)` ([`cusum_value_upper`]); `t <= 1` keeps `beta' <= lambda_i`, which
/// gives `-beta' z^+ + lambda_i (-z)^+ >= -beta' z`. The drawdown `-Y_min` of the work, compound
/// Poisson with exponential jumps of mean `1 / mu_i`, has mean at most `delta G + C_W(delta)`,
/// `C_W(delta) = (rho_i^2 - delta^2) / (2 mu_i delta rho_i)` (a verification function for its
/// reflected version). With `s = 1 - c (1 - t)` this gives
///
/// ```text
/// x_i >= (1 - c + s (rho_i - delta) / (1 - rho_i)) f_i
///        - (s (d_i + C_W(delta)) / (1 - rho_i) + V(c, t) / lambda_i) eta_i
/// ```
///
/// for every admissible policy and every reconfiguration-time law. `V` is replaced by its
/// certified upper bound, the coefficient of `f_i` is rounded down and that of `eta_i` up with
/// `bits` fractional bits; rows whose coefficient of `f_i` is not positive or whose `V` is not
/// certified are dropped, and there are none for `rho_i >= 1`.
pub fn strong_u(model: &DiscreteModel, i: usize, params: &[(Q, Q, Q)], bits: u32) -> Vec<GapRow> {
    let rho_i = model.rho(i);
    let lam = &model.lam[i];
    if rho_i >= Q::one() || !lam.is_positive() {
        return Vec::new();
    }
    let one_minus_rho = Q::one() - &rho_i;
    let mut out = Vec::new();
    for (c, t, delta) in params {
        if !(delta.is_positive() && *delta < rho_i) {
            continue;
        }
        let Some(v) = cusum_value_upper(c, t) else {
            continue;
        };
        let s = Q::one() - c * (Q::one() - t);
        let a = round_down(
            &(Q::one() - c + &s * (&rho_i - delta) / &one_minus_rho),
            bits,
        );
        if !a.is_positive() {
            continue;
        }
        let cw = (&rho_i * &rho_i - delta * delta) / (qi(2) * &model.mu[i] * delta * &rho_i);
        let b = round_up(&(&s * (&model.d[i] + cw) / &one_minus_rho + v / lam), bits);
        let entry = (0..model.n)
            .map(|u| if u == i { Q::zero() } else { b.clone() })
            .collect();
        out.push(row(
            model,
            "gap",
            format!("gap_strong_u[{i}] c={c} t={t} delta={delta}"),
            i,
            Q::one(),
            a,
            entry,
        ));
    }
    out
}

/// `1 / beta(k / U_STEPS)` in doubles for `k = 1..U_STEPS` (parameter choice only).
fn prophet_table() -> &'static [f64] {
    static TABLE: OnceLock<Vec<f64>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (1..U_STEPS)
            .map(|k| 1.0 / prophet_rate_f64(k as f64 / U_STEPS as f64))
            .collect()
    })
}

/// `V(c, t)` in doubles, as certified ([`cusum_value_used_f64`]), at `c = k / U_STEPS`,
/// `k = 1..U_STEPS`, and every `t` of [`T_GRID`] (parameter choice only; computed once per
/// process, sequentially).
fn cusum_table() -> &'static [Vec<f64>] {
    static TABLE: OnceLock<Vec<Vec<f64>>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (1..U_STEPS)
            .map(|k| {
                T_GRID
                    .iter()
                    .map(|&(tn, td)| {
                        cusum_value_used_f64(k as f64 / U_STEPS as f64, tn as f64 / td as f64)
                    })
                    .collect()
            })
            .collect()
    })
}

/// The grid index `k` of the weak U-row parameter `eps = k / U_STEPS` that maximizes the row at
/// the mean gap `g`, per unit entry rate `(1 - eps) (g - d) + eps excess - 1 / (lambda
/// beta(eps))`; `None` when no grid point makes it positive.
fn best_weak_u(lambda: f64, d: f64, excess: f64, g: f64) -> Option<usize> {
    let inv_beta = prophet_table();
    let value = |k: usize| {
        let eps = k as f64 / U_STEPS as f64;
        (1.0 - eps) * (g - d) + eps * excess - inv_beta[k - 1] / lambda
    };
    let k = (1..U_STEPS).max_by(|a, b| value(*a).total_cmp(&value(*b)))?;
    (value(k) > 0.0).then_some(k)
}

/// The grid indices `(c, t, delta)` of the S-U row that maximize it at the mean gap `g`, per unit
/// entry rate `(1 - c) g + s (phi(delta) - d) / (1 - rho) - V(c, t) / lambda` with
/// `phi(delta) = (rho - delta) g - C_W(delta)`; `None` when no grid point makes it positive.
/// `phi` is concave with its maximum at `rho / sqrt(2 mu rho g - 1)` (at `rho` when
/// `mu rho g <= 1`), so the best grid `delta` is a neighbour of that point.
fn best_strong_u(lambda: f64, mu: f64, d: f64, g: f64) -> Option<(usize, usize, usize)> {
    let rho = lambda / mu;
    if rho >= 1.0 {
        return None;
    }
    let steps = U_STEPS as f64;
    let phi = |kd: usize| {
        let dl = rho * kd as f64 / steps;
        (rho - dl) * g - (rho * rho - dl * dl) / (2.0 * mu * dl * rho)
    };
    let x = 2.0 * mu * rho * g - 1.0;
    let k0 = if x > 1.0 { steps / x.sqrt() } else { steps };
    let k0 = (k0.round() as usize).clamp(1, U_STEPS - 1);
    let kd = [k0.saturating_sub(1).max(1), k0, (k0 + 1).min(U_STEPS - 1)]
        .into_iter()
        .max_by(|a, b| phi(*a).total_cmp(&phi(*b)))?;
    let work = (phi(kd) - d) / (1.0 - rho);
    let table = cusum_table();
    let mut best: Option<((usize, usize, usize), f64)> = None;
    for kc in 1..U_STEPS {
        let c = kc as f64 / steps;
        for (ti, &(tn, td)) in T_GRID.iter().enumerate() {
            let s = 1.0 - c * (1.0 - tn as f64 / td as f64);
            let h = (1.0 - c) * g + s * work - table[kc - 1][ti] / lambda;
            if h.is_finite() && best.is_none_or(|b| h > b.1) {
                best = Some(((kc, ti, kd), h));
            }
        }
    }
    best.filter(|b| b.1 > 0.0).map(|b| b.0)
}

/// The reference mean gap `G_i = (1 - rho_i) K tau_min / (1 - rho)` of class `i`: the mean gap
/// when the `K` active classes are entered equally often and all time not spent serving goes to
/// reconfigurations of the smallest mean duration `tau_min` (in the symmetric case the mean
/// intervisit time `I` of cyclic exhaustive service). `None` when `rho >= 1` or no mean is
/// positive. It only places the tangent points.
pub fn reference_gap(model: &DiscreteModel, mean_tau: &[Vec<Q>], i: usize) -> Option<f64> {
    let rho = to_f64(&model.total_rho());
    let tau_min = model
        .edges()
        .iter()
        .map(|(u, v)| to_f64(&mean_tau[*u][*v]))
        .fold(f64::INFINITY, f64::min);
    let k = (0..model.n).filter(|j| model.active(*j)).count() as f64;
    (rho < 1.0 && tau_min > 0.0 && tau_min.is_finite())
        .then(|| (1.0 - to_f64(&model.rho(i))) * k * tau_min / (1.0 - rho))
}

/// The window length `x = u / lambda` that maximizes the window row at the mean gap `g > d`,
/// where `u > 0` solves `(1 + u) exp(-u) = 1 - d / g` (bisection in doubles).
fn best_window(lambda: f64, d: f64, g: f64) -> Option<f64> {
    if !(g > d && lambda > 0.0) {
        return None;
    }
    let target = 1.0 - d / g;
    let (mut lo, mut hi) = (0.0f64, 64.0f64);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if (1.0 + mid) * (-mid).exp() > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi) / lambda)
}

/// The visit-gap constraints of every active class of `model` (none with fewer than two
/// configurations): window rows at `x = h - d_i` for every window length `h > d_i` of
/// `windows` and at the best window length for the mean gaps `G_i`, `sqrt(2) G_i` and `2 G_i`;
/// weak and strong rows tangent at the mean gaps `2^(k/2) G_i`, `k = -1..=4`, with
/// `kappa = 1 / sqrt(lambda_i g)` and `s = sqrt(g)`, where `G_i` is [`reference_gap`]; and at
/// the same mean gaps the weak U-row and the S-U row with the grid parameters that maximize
/// them there (`eps = k / 400`; `c = k / 400`, `t` in `T_GRID`, `delta = rho_i k / 400`),
/// where positive. The tangent points and the parameter choice only decide where the rows are
/// tight; each row is valid on its own. The coefficients are rounded with `bits` fractional
/// bits; [`ROW_BITS`] weakens a row by about `2^-32` at most and keeps the rationals of the
/// exact certificates short.
pub fn gap_rows(
    model: &DiscreteModel,
    mean_tau: &[Vec<Q>],
    windows: &[Q],
    bits: u32,
) -> Vec<GapRow> {
    let mut out = Vec::new();
    if model.n < 2 {
        return out;
    }
    for i in (0..model.n).filter(|i| model.active(*i)) {
        let (lam, d) = (to_f64(&model.lam[i]), to_f64(&model.d[i]));
        let g_ref = reference_gap(model, mean_tau, i);
        let mut xs: Vec<Q> = windows
            .iter()
            .map(|h| h - &model.d[i])
            .filter(|x| x.is_positive())
            .collect();
        let mut kappas = Vec::new();
        let mut ss = Vec::new();
        let mut weak_u_grid = Vec::new();
        let mut strong_u_grid = Vec::new();
        if let Some(g0) = g_ref {
            let excess = to_f64(&reconfiguration_excess(model, mean_tau, i));
            let mu = to_f64(&model.mu[i]);
            for k in WINDOW_EXPONENTS {
                let g = g0 * 2f64.powf(k as f64 / 2.0);
                xs.extend(best_window(lam, d, g).and_then(param));
            }
            for k in TANGENT_EXPONENTS {
                let g = g0 * 2f64.powf(k as f64 / 2.0);
                kappas.extend(param(1.0 / (lam * g).sqrt()));
                ss.extend(param(g.sqrt()));
                weak_u_grid.extend(best_weak_u(lam, d, excess, g));
                strong_u_grid.extend(best_strong_u(lam, mu, d, g));
            }
        }
        xs.sort();
        xs.dedup();
        weak_u_grid.sort_unstable();
        weak_u_grid.dedup();
        strong_u_grid.sort_unstable();
        strong_u_grid.dedup();
        let steps = U_STEPS as i64;
        let epss: Vec<Q> = weak_u_grid.iter().map(|k| qr(*k as i64, steps)).collect();
        let rho_i = model.rho(i);
        let params: Vec<(Q, Q, Q)> = strong_u_grid
            .iter()
            .map(|&(kc, ti, kd)| {
                let (tn, td) = T_GRID[ti];
                (
                    qr(kc as i64, steps),
                    qr(tn, td),
                    &rho_i * qr(kd as i64, steps),
                )
            })
            .collect();
        out.extend(window_single(model, i, &xs, bits));
        out.extend(weak(model, mean_tau, i, &kappas, bits));
        out.extend(strong(model, i, &ss, bits));
        out.extend(weak_u(model, mean_tau, i, &epss, bits));
        out.extend(strong_u(model, i, &params, bits));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::model::{DiscreteLaw, DiscreteModel};
    use super::super::rational::{parse_q, qr};
    use super::*;

    /// `n` classes with `mu = 4`, `lambda = 4 rho / n`, deadline `d` and the reconfiguration
    /// time 1/2 on every pair.
    fn symmetric(n: usize, rho: &str, d: &str) -> DiscreteModel {
        let rho = parse_q(rho).unwrap();
        let laws = (0..n)
            .map(|u| {
                (0..n)
                    .map(|v| (u != v).then(|| DiscreteLaw::point(qr(1, 2))))
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

    fn means(m: &DiscreteModel) -> Vec<Vec<Q>> {
        (0..m.n)
            .map(|u| {
                (0..m.n)
                    .map(|v| {
                        if u == v {
                            Q::zero()
                        } else {
                            m.law(u, v).mean()
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// The bound on `x_i` that `r` gives at the symmetric point with fraction `f` away from the
    /// configuration and entry rate `eta`, split equally over the other configurations.
    fn implied(m: &DiscreteModel, r: &GapRow, f: f64, eta: f64) -> f64 {
        let z = 1.0 - to_f64(&m.rho(r.class)) - f;
        let nu = eta / (m.n - 1) as f64;
        let entries: f64 = r.entry.iter().map(|b| to_f64(b) * nu).sum();
        (to_f64(&r.rhs) - to_f64(&r.z) * z - entries) / to_f64(&r.x)
    }

    #[test]
    fn window_rows_follow_the_closed_form_from_below() {
        let m = symmetric(4, "9/10", "4");
        let (f, eta, lam, d) = (0.775, 0.05, 0.9f64, 4.0);
        let rows = window_single(&m, 0, &[qi(1), qr(1, 4), qi(8)], 128);
        assert_eq!(rows.len(), 3);
        for (r, x) in rows.iter().zip([1.0, 0.25, 8.0]) {
            let exact = (f * (1.0 - (-lam * x).exp()) - (x + d) * eta) / (lam * x);
            let b = implied(&m, r, f, eta);
            assert!(
                b <= exact + 1e-12 && exact - b < 1e-12,
                "x = {x}: {b} vs {exact}"
            );
            assert_eq!(r.group, "gap_window");
        }
        // The best window at the mean intervisit time 15.5 s gives the symmetric value 23.44 %.
        let x = best_window(lam, d, 15.5).unwrap();
        assert!((x - 1.092).abs() < 0.01, "{x}");
        let r = &window_single(&m, 0, &[param(x).unwrap()], 128)[0];
        assert!((implied(&m, r, f, eta) - 0.2344).abs() < 1e-4);
    }

    #[test]
    fn tangent_rows_touch_the_convex_bounds_at_their_mean_gap_and_stay_below() {
        let m = symmetric(4, "19/20", "8");
        let tau = means(&m);
        let (lam, mu, d, rho_i) = (0.95, 4.0, 8.0, 0.2375);
        let weak_bound =
            |f: f64, eta: f64| f - eta * d - 2.0 * (eta * (f - eta * 0.5) / lam).sqrt();
        let strong_bound = |f: f64, eta: f64| {
            (f - eta * d - (2.0 / mu) * (2.0 * lam * eta * f).sqrt()) / (1.0 - rho_i)
                - 2.0 * (eta * f / lam).sqrt()
        };
        for g in [7.0f64, 30.5, 61.0] {
            let w = &weak(
                &m,
                &tau,
                0,
                &[param(1.0 / (lam * (g - 0.5)).sqrt()).unwrap()],
                128,
            )[0];
            let s = &strong(&m, 0, &[param(g.sqrt()).unwrap()], 128)[0];
            for (f, eta) in [
                (0.7625, 0.7625 / g),
                (0.5, 0.5 / g),
                (0.7, 0.01),
                (0.3, 0.2),
            ] {
                let (bw, bs) = (implied(&m, w, f, eta), implied(&m, s, f, eta));
                assert!(bw <= weak_bound(f, eta) + 1e-12, "weak at {f}, {eta}");
                assert!(bs <= strong_bound(f, eta) + 1e-12, "strong at {f}, {eta}");
                if (f / eta - g).abs() < 1e-9 {
                    // Tangent: equal up to the rounding of the tangent parameter.
                    assert!(weak_bound(f, eta) - bw < 1e-5, "weak at g = {g}");
                    assert!(strong_bound(f, eta) - bs < 1e-5, "strong at g = {g}");
                }
            }
        }
        // The strong row at the mean intervisit time reproduces the symmetric value 32.96 %.
        let s = &strong(&m, 0, &[param(30.5f64.sqrt()).unwrap()], 128)[0];
        assert!((implied(&m, s, 0.7625, 0.025) - 0.3296).abs() < 1e-4);
    }

    #[test]
    fn the_certified_constants_of_the_u_rows_enclose_the_values_of_the_symmetric_analysis() {
        // The prophet rate: a lower bound within 1e-8 of the root, above 2 eps (Phi(c) >= 2c).
        for (n, d) in [(1, 400), (1, 4), (1, 2), (399, 400)] {
            let eps = qr(n, d);
            let lo = to_f64(&prophet_rate_lower(&eps));
            let root = prophet_rate_f64(to_f64(&eps));
            assert!(
                lo <= root && root - lo < 1e-8 * root,
                "eps = {eps}: {lo} vs {root}"
            );
            assert!(lo > 2.0 * to_f64(&eps), "eps = {eps}: {lo}");
        }
        // The CUSUM values V(c, t) at the parameters of the S-U rows of the symmetric analysis
        // (six decimals), and close to the prophet limit 1 / Phi(0.11) = 4.206 as t -> 0.
        let cases = [
            (37, (1, 1), 2.697564),
            (57, (1, 5), 2.416257),
            (49, (1, 5), 2.845073),
            (13, (2, 5), 9.847065),
            (26, (1, 100), 7.087175),
            (44, (1, 10_000), 4.2022),
        ];
        for (kc, (tn, td), v) in cases {
            let (c, t) = (qr(kc, 400), qr(tn, td));
            let hi = to_f64(&cusum_value_upper(&c, &t).unwrap());
            let (vf, _) = cusum_value_f64(to_f64(&c), to_f64(&t));
            let tol = if td == 10_000 { 1e-4 } else { 1e-6 };
            assert!((hi - v).abs() < tol, "c = {c}, t = {t}: {hi} vs {v}");
            assert!(
                hi >= vf - 1e-9 && hi - vf < 1e-7,
                "c = {c}, t = {t}: {hi} vs {vf}"
            );
        }
        // Where the series would be long (small c and t) the prophet bound is used alone.
        let (c, t) = (qr(1, 400), qr(1, 10_000));
        assert!(cusum_value_f64(to_f64(&c), to_f64(&t)).1 > SERIES_CAP);
        let prophet = prophet_rate_lower(&(&c * (Q::one() - &t))).recip();
        assert_eq!(cusum_value_upper(&c, &t).unwrap(), prophet);
        let limit = 1.0 / prophet_rate_f64(0.11);
        let near = to_f64(&cusum_value_upper(&qr(44, 400), &qr(1, 10_000)).unwrap());
        assert!(near < limit && limit - near < 0.01, "{near} vs {limit}");
    }

    #[test]
    fn the_u_rows_follow_their_closed_forms_from_below() {
        let m = symmetric(4, "19/20", "8");
        let tau = means(&m);
        let (lam, mu, d, rho_i) = (0.95f64, 4.0, 8.0, 0.2375);
        let eps = qr(1, 10);
        let w = &weak_u(&m, &tau, 0, std::slice::from_ref(&eps), 64)[0];
        let (c, t, delta) = (qr(37, 400), qr(1, 1), qr(1, 32));
        let s = &strong_u(&m, 0, &[(c.clone(), t.clone(), delta.clone())], 64)[0];
        let beta = prophet_rate_f64(0.1);
        let (v, _) = cusum_value_f64(to_f64(&c), to_f64(&t));
        let (cf, dl) = (to_f64(&c), to_f64(&delta));
        let cw = (rho_i * rho_i - dl * dl) / (2.0 * mu * dl * rho_i);
        for (f, eta) in [(0.7625, 0.025), (0.5, 0.01), (0.3, 0.2)] {
            let weak_value = 0.9 * (f - eta * d) - eta / (lam * beta);
            let strong_value = (1.0 - cf + (rho_i - dl) / (1.0 - rho_i)) * f
                - eta * ((d + cw) / (1.0 - rho_i) + v / lam);
            let (bw, bs) = (implied(&m, w, f, eta), implied(&m, s, f, eta));
            assert!(
                bw <= weak_value + 1e-12 && weak_value - bw < 1e-8,
                "{bw} vs {weak_value}"
            );
            assert!(
                bs <= strong_value + 1e-9 && strong_value - bs < 1e-6,
                "{bs} vs {strong_value}"
            );
        }
        // The S-U row at the parameters of the symmetric analysis gives 53.43 % at the mean
        // intervisit time (c = 0.0925, t = 1, delta = 0.1325 rho_i).
        let s = &strong_u(&m, 0, &[(c, t, &m.rho(0) * qr(53, 400))], 64)[0];
        let b = implied(&m, s, 0.7625, 0.025);
        assert!((0.5343..0.5344).contains(&b), "{b}");
        assert!(w.label.starts_with("gap_weak_u[0]") && s.label.starts_with("gap_strong_u[0]"));
    }

    #[test]
    fn the_declared_rows_cover_every_active_class_and_skip_vacuous_ones() {
        let m = symmetric(3, "1/2", "2");
        let tau = means(&m);
        assert!(
            (reference_gap(&m, &tau, 0).unwrap() - (1.0 - 1.0 / 6.0) * 1.5 / 0.5).abs() < 1e-12
        );
        let windows = super::super::windows::default_windows(&m);
        let rows = gap_rows(&m, &tau, &windows, 64);
        for i in 0..3 {
            assert!(rows.iter().any(|r| r.class == i && r.group == "gap_window"));
            assert!(
                rows.iter()
                    .any(|r| r.class == i && r.label.starts_with("gap_weak"))
            );
        }
        for r in &rows {
            assert!(r.rhs.is_positive(), "{}", r.label);
            assert!(r.entry[r.class].is_zero());
        }
        // An inactive class gets no rows, and a single class none at all.
        let mut idle = m.clone();
        idle.lam[1] = Q::zero();
        assert!(
            gap_rows(&idle, &tau, &windows, 64)
                .iter()
                .all(|r| r.class != 1)
        );
        let one = DiscreteModel::new(
            "one",
            vec![qi(1)],
            vec![qi(4)],
            vec![qi(3)],
            vec![qr(1, 100)],
            vec![vec![None]],
        )
        .unwrap();
        assert!(gap_rows(&one, &[vec![Q::zero()]], &[qi(5)], 64).is_empty());
    }
}
