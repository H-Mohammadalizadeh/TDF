//! The reference analysis of the symmetric case (identical classes and one
//! reconfiguration-time distribution on every pair; `Homogeneous` in
//! identifiers), where OJF reduces to an M/G/1 queue. It encloses the
//! no-waiting lower bound `b(d)` (`floor`), the achievable upper bound, the
//! workload bound and the closed-form benchmark of OJF, and it assigns target
//! verdicts from the certified interval `L_safe <= m* <= U^+` (`Bracket`) of
//! the best achievable worst-class miss probability `m*`. Every number is a
//! rational enclosure with directed rounding, not a simulation estimate.

use super::distributions::{Distribution, Model};
use super::grid::{GridSettings, grid_coefficients, q_to_f64_down, q_to_f64_up};
use super::rational::{Q, dec, exp_neg, exp_neg_powers, qi, round_down, round_up, to_f64};
use super::special::exp_enclosure;
use num_traits::{One, Signed, ToPrimitive, Zero};
use serde_json::{Value, json};
use std::time::Instant;

/// A rational enclosure `lo <= value <= hi`.
pub type Iv = (Q, Q);

/// Both ends of an enclosure clamped to `[0, 1]`.
fn clip01(e: Iv) -> Iv {
    (
        e.0.max(Q::zero()).min(Q::one()),
        e.1.max(Q::zero()).min(Q::one()),
    )
}

/// `e` rounded outward to `bits` fractional bits.
fn round_out(e: Iv, bits: u32) -> Iv {
    (round_down(&e.0, bits), round_up(&e.1, bits))
}

/// `s * e` for a scalar `s` of either sign.
fn scale(e: &Iv, s: &Q) -> Iv {
    if s.is_negative() {
        (&e.1 * s, &e.0 * s)
    } else {
        (&e.0 * s, &e.1 * s)
    }
}

/// Sum of two enclosures.
fn add(a: &Iv, b: &Iv) -> Iv {
    (&a.0 + &b.0, &a.1 + &b.1)
}

/// Product of two enclosures of nonnegative quantities.
fn mul_nonneg(a: &Iv, b: &Iv) -> Iv {
    (&a.0 * &b.0, &a.1 * &b.1)
}

/// `x^k`.
fn powi(x: &Q, k: u32) -> Q {
    let mut v = Q::one();
    for _ in 0..k {
        v *= x;
    }
    v
}

/// `k!`.
fn factorial(k: u32) -> Q {
    (1..=k).fold(Q::one(), |acc, j| acc * qi(j as i64))
}

fn iv_json(e: &Iv) -> Value {
    json!({"lo": e.0.to_string(), "hi": e.1.to_string(), "lo_f64": to_f64(&e.0), "hi_f64": to_f64(&e.1), "width": to_f64(&(&e.1 - &e.0))})
}

fn q_json(x: &Q) -> Value {
    json!({"exact": x.to_string(), "f64": to_f64(x), "pct_up": dec(&(x * qi(100)), 6, true)})
}

/// The symmetric case of a model: `N >= 2` classes with equal arrival rates,
/// service rates, deadlines and targets, and one reconfiguration-time
/// distribution on every ordered pair. OJF (oldest job first, i.e. global
/// FCFS) serves jobs in arrival order, and consecutive jobs belong to
/// different classes with probability `q = 1 - 1/N`, so under OJF the network
/// is an M/G/1 queue (the reference queue) with effective service time
/// `Y = S + I tau`, `P(I = 1) = q`, and load `rho_F = Lambda E[Y]`.
#[derive(Clone, Debug)]
pub struct Homogeneous {
    pub n: usize,
    /// Total arrival rate `Lambda = sum_i lambda_i`.
    pub lambda_total: Q,
    pub mu: Q,
    pub d: Q,
    pub eps: Q,
    /// The one reconfiguration-time distribution of every ordered pair.
    pub law: Distribution,
}

/// The closed-form benchmark: `tau ~ Exp(r)` and `mu = N r` make the
/// effective service time exponential with rate `r`, so the stationary miss
/// probability of OJF is `exp(-(r - Lambda) d)` for `Lambda < r`.
#[derive(Clone, Debug)]
pub struct Benchmark {
    /// The rate `r` of the effective service time.
    pub rate: Q,
    /// `rho_F = Lambda / r`, exact.
    pub rho_f: Q,
    /// Enclosure of `exp(-(r - Lambda) d)`.
    pub miss: Iv,
}

/// The two valid enclosures of `P(tau + S > d)` and their intersection.
#[derive(Clone, Debug)]
pub struct TailEnclosure {
    /// The closed-form enclosure, where one exists.
    pub closed_form: Option<Iv>,
    /// The grid enclosure.
    pub grid: Iv,
    /// The intersection of both, or the grid enclosure alone.
    pub used: Iv,
    pub grid_diagnostics: Value,
}

/// Long-run rates and fractions of OJF in the symmetric case.
#[derive(Clone, Debug)]
pub struct ReferenceRates {
    /// `nu_ij = Lambda / N^2` for every ordered pair.
    pub nu_pair: Q,
    /// `nu = Lambda q`.
    pub nu_total: Q,
    /// `eta_i = Lambda q / N`.
    pub eta: Q,
    /// Average jobs served per visit, `N / (N - 1)`.
    pub beta: Q,
    /// `z_i = (1 - rho_F) / N`, enclosed through the mean enclosure.
    pub z: Iv,
    /// Enclosure of `rho_F`.
    pub rho_f: Iv,
    /// Whether the enclosed time budget `rho + nu taubar + sum z` contains one.
    pub time_budget_contains_one: bool,
}

impl Homogeneous {
    /// The symmetric case of `m`, or the reason `m` is not one: fewer than two
    /// classes, unequal classes or distributions, or a service or arrival
    /// setting outside the model.
    pub fn from_model(m: &Model) -> Result<Homogeneous, String> {
        if m.n < 2 {
            return Err("the homogeneous analysis needs N >= 2".into());
        }
        if m.service.is_some() || m.batch_mean.is_some() {
            return Err("the homogeneous analysis applies to the model only (exponential service, single Poisson arrivals)".into());
        }
        let same = |v: &[Q], what: &str| -> Result<Q, String> {
            if v.iter().all(|x| x == &v[0]) {
                Ok(v[0].clone())
            } else {
                Err(format!(
                    "{what} differ across classes; the homogeneous analysis does not apply"
                ))
            }
        };
        let lam = same(&m.lam, "arrival rates")?;
        if !lam.is_positive() {
            return Err("arrival rates must be positive".into());
        }
        let mu = same(&m.mu, "service rates")?;
        let d = same(&m.d, "deadlines")?;
        let eps = same(&m.eps, "targets")?;
        let ids: Vec<String> = m.edges().iter().map(|(u, v)| m.law(*u, *v).id()).collect();
        if !ids.iter().all(|i| i == &ids[0]) {
            return Err("reconfiguration-time distributions differ across pairs; the homogeneous analysis does not apply".into());
        }
        Ok(Homogeneous {
            n: m.n,
            lambda_total: lam * qi(m.n as i64),
            mu,
            d,
            eps,
            law: m.law(0, 1).clone(),
        })
    }

    /// `q = 1 - 1/N`, the probability that consecutive arrivals differ in class.
    pub fn q(&self) -> Q {
        Q::one() - qi(self.n as i64).recip()
    }

    /// The arrival rate `Lambda / N` of every class.
    pub fn lambda_class(&self) -> Q {
        &self.lambda_total / qi(self.n as i64)
    }

    /// The total offered load `rho = Lambda / mu`.
    pub fn rho(&self) -> Q {
        &self.lambda_total / &self.mu
    }

    /// `rho_F = Lambda (1/mu + q E[tau])`, enclosed through the mean
    /// enclosure; exact (both ends equal, no rounding) when the mean is exact.
    pub fn rho_f(&self, bits: u32) -> Iv {
        let (lo, hi) = self.law.mean(bits);
        let q = self.q();
        let f = |m: &Q| &self.lambda_total * (self.mu.recip() + &q * m);
        if lo == hi {
            let v = f(&lo);
            return (v.clone(), v);
        }
        round_out((f(&lo), f(&hi)), bits)
    }

    /// `P(tau + S > d)` in closed form for a point mass, an exponential
    /// distribution or a gamma distribution of integer shape; `None` for the
    /// other distributions.
    pub fn tail_closed_form(&self, bits: u32) -> Option<Iv> {
        let (mu, d) = (&self.mu, &self.d);
        let work = bits + 16;
        let e: Iv = match &self.law {
            Distribution::Deterministic { value } => {
                if value >= d {
                    (Q::one(), Q::one())
                } else {
                    exp_neg(&(mu * (d - value)), work)
                }
            }
            Distribution::Exponential { rate } => {
                let er = exp_neg(&(rate * d), work);
                let em = exp_neg(&(mu * d), work);
                if rate == mu {
                    // P(tau + S > d) = e^{-mu d} (1 + mu d) for an Erlang-2 sum
                    let f = Q::one() + mu * d;
                    (&em.0 * &f, &em.1 * &f)
                } else if mu > rate {
                    // (mu e^{-r d} - r e^{-mu d}) / (mu - r)
                    let den = mu - rate;
                    (
                        (mu * &er.0 - rate * &em.1) / &den,
                        (mu * &er.1 - rate * &em.0) / &den,
                    )
                } else {
                    // (r e^{-mu d} - mu e^{-r d}) / (r - mu)
                    let den = rate - mu;
                    (
                        (rate * &em.0 - mu * &er.1) / &den,
                        (rate * &em.1 - mu * &er.0) / &den,
                    )
                }
            }
            Distribution::Gamma { shape, rate } if shape.is_integer() => {
                let k = shape.to_integer().to_u32()?;
                if k == 0 {
                    return None;
                }
                // P(tau + S > d)
                //     = P(tau > d) + (r^k / (k-1)!) e^{-mu d} int_0^d t^{k-1} e^{-(r-mu) t} dt
                let surv = self.law.survival(d, work);
                let a = rate - mu;
                let integral: Iv = if a.is_zero() {
                    let v = powi(d, k) / qi(k as i64);
                    (v.clone(), v)
                } else {
                    // for a != 0, int_0^d t^{k-1} e^{-a t} dt
                    //     = (k-1)! / a^k [1 - e^{-a d} sum_{j<k} (a d)^j / j!]
                    let ad = &a * d;
                    let e_ad = exp_enclosure(&(-&ad), work);
                    let mut p = Q::zero();
                    let mut term = Q::one();
                    for j in 0..k {
                        if j > 0 {
                            term = term * &ad / qi(j as i64);
                        }
                        p += &term;
                    }
                    let ep = scale(&e_ad, &p);
                    let bracket = (Q::one() - &ep.1, Q::one() - &ep.0);
                    let c = factorial(k - 1) / powi(&a, k);
                    scale(&bracket, &c)
                };
                let integral = (integral.0.max(Q::zero()), integral.1.max(Q::zero()));
                let coef = powi(rate, k) / factorial(k - 1);
                let em = exp_neg(&(mu * d), work);
                add(&surv, &mul_nonneg(&scale(&em, &coef), &integral))
            }
            _ => return None,
        };
        Some(round_out(clip01(e), bits))
    }

    /// `P(tau + S > d)` through the grid enclosure of `B_jk = 1 - H_jk(d)`
    /// (`grid.rs`) on a two-configuration model with the same distribution,
    /// valid for every distribution: with identical distributions on all
    /// pairs, reconfiguring directly to `k` is optimal for the isolated job,
    /// so `H_jk(d) = P(tau + S <= d)` for `j != k`.
    pub fn tail_grid(&self, settings: &GridSettings) -> Result<(Iv, Value), String> {
        let laws = (0..2)
            .map(|u| (0..2).map(|v| (u != v).then(|| self.law.clone())).collect())
            .collect();
        let m = Model::new(
            "reference-tail",
            vec![self.lambda_class(); 2],
            vec![self.mu.clone(); 2],
            vec![self.d.clone(); 2],
            vec![self.eps.clone(); 2],
            laws,
        )?;
        let (lower, upper, diag) = grid_coefficients(&m, settings)?;
        Ok(((lower.b[0][1].clone(), upper.b[0][1].clone()), diag))
    }

    /// Both enclosures of `P(tau + S > d)` and their intersection; an empty
    /// intersection is a correctness failure and returned as an error.
    pub fn tail(&self, settings: &GridSettings) -> Result<TailEnclosure, String> {
        let closed_form = self.tail_closed_form(settings.bits);
        let (grid, grid_diagnostics) = self.tail_grid(settings)?;
        let used = match &closed_form {
            Some(c) => {
                let lo = c.0.clone().max(grid.0.clone());
                let hi = c.1.clone().min(grid.1.clone());
                if lo > hi {
                    return Err(format!(
                        "closed form [{}, {}] and grid enclosure [{}, {}] of P(tau + S > d) do not intersect: correctness failure",
                        to_f64(&c.0),
                        to_f64(&c.1),
                        to_f64(&grid.0),
                        to_f64(&grid.1)
                    ));
                }
                (lo, hi)
            }
            None => grid.clone(),
        };
        Ok(TailEnclosure {
            closed_form,
            grid,
            used,
            grid_diagnostics,
        })
    }

    /// The no-waiting lower bound `b(d) = P(Y > d)`, i.e.
    /// `(1/N) e^{-mu d} + q P(tau + S > d)`, from an enclosure of the tail.
    pub fn floor_from_tail(&self, tail: &Iv, bits: u32) -> Iv {
        let e = exp_neg(&(&self.mu * &self.d), bits + 16);
        let inv_n = qi(self.n as i64).recip();
        let q = self.q();
        round_out(
            clip01((&inv_n * &e.0 + &q * &tail.0, &inv_n * &e.1 + &q * &tail.1)),
            bits,
        )
    }

    /// The achievable upper bound `U_simple = b + (1 - b) rho_F`, evaluated at
    /// `b^+` and `rho_F^+` (it increases in both) and rounded up; `None` unless
    /// `rho_F^+ < 1`, where OJF is proved stable.
    pub fn u_simple(b: &Iv, rho_f: &Iv, bits: u32) -> Option<Q> {
        if rho_f.1 >= Q::one() {
            return None;
        }
        Some(round_up(&(&b.1 + (Q::one() - &b.1) * &rho_f.1), bits).min(Q::one()))
    }

    /// Whether the closed-form benchmark applies: an exponential distribution
    /// of rate `r` with `mu = N r`, for any `N >= 2`; then
    /// `E[exp(-sY)] = (Nr/(Nr+s)) [1/N + (1-1/N) r/(r+s)] = r/(r+s)`, so
    /// `Y ~ Exp(r)`, `rho_F = Lambda / r` and `m_F = exp(-(r - Lambda) d)`.
    pub fn benchmark_applies(&self) -> bool {
        matches!(&self.law, Distribution::Exponential { rate } if self.mu == rate * qi(self.n as i64))
    }

    /// `Ok(None)` when the benchmark does not apply, `Err` when it applies
    /// but `Lambda >= r` (no stationary miss probability), otherwise the
    /// closed-form benchmark with the enclosure of `exp(-(r - Lambda) d)`.
    pub fn exact_exponential_benchmark(&self, bits: u32) -> Result<Option<Benchmark>, String> {
        if !self.benchmark_applies() {
            return Ok(None);
        }
        let Distribution::Exponential { rate } = &self.law else {
            unreachable!()
        };
        if &self.lambda_total >= rate {
            return Err(format!(
                "exact benchmark: Lambda = {} is not below the effective service rate {}; global FCFS is not stable there",
                self.lambda_total, rate
            ));
        }
        Ok(Some(Benchmark {
            rate: rate.clone(),
            rho_f: &self.lambda_total / rate,
            miss: exp_neg(&((rate - &self.lambda_total) * &self.d), bits),
        }))
    }

    /// The long-run rates and fractions of OJF, with the check that the
    /// enclosed time budget contains one.
    pub fn reference_rates(&self, bits: u32) -> ReferenceRates {
        let n = qi(self.n as i64);
        let q = self.q();
        let rho_f = self.rho_f(bits);
        let z = if rho_f.0 == rho_f.1 {
            let v = (Q::one() - &rho_f.0) / &n;
            (v.clone(), v)
        } else {
            (
                round_down(&((Q::one() - &rho_f.1) / &n), bits),
                round_up(&((Q::one() - &rho_f.0) / &n), bits),
            )
        };
        let (mlo, mhi) = self.law.mean(bits);
        let nu_total = &self.lambda_total * &q;
        let budget_lo = self.rho() + &nu_total * &mlo + &n * &z.0;
        let budget_hi = self.rho() + &nu_total * &mhi + &n * &z.1;
        ReferenceRates {
            nu_pair: &self.lambda_total / (&n * &n),
            nu_total,
            eta: &self.lambda_total * &q / &n,
            beta: &n / (&n - Q::one()),
            z,
            rho_f,
            time_budget_contains_one: budget_lo <= Q::one() && Q::one() <= budget_hi,
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "n": self.n, "lambda_total": self.lambda_total.to_string(), "lambda_class": self.lambda_class().to_string(),
            "mu": self.mu.to_string(), "d": self.d.to_string(), "eps": self.eps.to_string(), "q": self.q().to_string(),
            "rho": to_f64(&self.rho()), "law": self.law.id()
        })
    }
}

impl ReferenceRates {
    pub fn to_json(&self) -> Value {
        json!({
            "nu_pair": q_json(&self.nu_pair), "nu_total": q_json(&self.nu_total), "eta": q_json(&self.eta),
            "beta_jobs_per_visit": q_json(&self.beta), "z": iv_json(&self.z), "rho_f": iv_json(&self.rho_f),
            "time_budget_contains_one": self.time_budget_contains_one
        })
    }
}

impl Benchmark {
    pub fn to_json(&self) -> Value {
        json!({"effective_rate": self.rate.to_string(), "rho_f": q_json(&self.rho_f), "miss": iv_json(&self.miss)})
    }
}

impl TailEnclosure {
    pub fn to_json(&self) -> Value {
        json!({
            "closed_form": self.closed_form.as_ref().map(iv_json).unwrap_or(Value::Null),
            "grid": iv_json(&self.grid), "used": iv_json(&self.used),
            "grid_step": self.grid_diagnostics["step"].clone(), "grid_cells": self.grid_diagnostics["cells"].clone()
        })
    }
}

/// The workload bound of the reference queue, from the compound-geometric
/// form of the M/G/1 workload. With `H` the distribution of `Y`, `H_e` its
/// equilibrium residual distribution and
/// `C_K(d) = (1 - rho_F) sum_{n=0}^{K} rho_F^n (H * H_e^{*n})(d)`, the
/// stationary miss probability `m_F` of OJF satisfies
/// `1 - C_K(d) - rho_F^{K+1} <= m_F <= 1 - C_K(d)`; `K = 0` gives the
/// achievable upper bound `1 - (1 - rho_F) H(d) = b + (1 - b) rho_F = U_simple`
/// with `b = P(Y > d)`. The convolutions are enclosed on a grid: every
/// duration is rounded up to the end of its cell (a stochastically larger
/// variable), cell masses and mixture weights are lower enclosures, and every
/// operation rounds down, so the computed `C_K^-` is a lower bound and
/// `U_K^+ = 1 - C_K^-` is valid.
///
/// The mirror computation encloses `C_K` from above: every duration is
/// represented by the lower end of its cell, a term counts when the sum of
/// the lower ends stays below `d`, cell masses and mixture weights are upper
/// enclosures and every operation rounds up. Then
/// `U_K^- = 1 - C_K^+ - rho_F^{K+1} <= m_F <= U_K^+`, and
/// `C_K^+ - C_K^-` is the numerical width of the enclosure of `1 - C_K(d)`,
/// separate from the omitted mixture mass `rho_F^{K+1}`.
#[derive(Clone, Debug)]
pub struct WorkloadBound {
    /// Number of residual terms `K`.
    pub k: usize,
    /// Grid step.
    pub step: Q,
    /// Grid cells up to the deadline.
    pub cells: usize,
    /// Lower enclosure `C_K^- <= C_K(d)`.
    pub c_k_lower: Q,
    /// `U_K^+ = 1 - C_K^-`, a certified upper bound on `m_F`.
    pub u_k_plus: Q,
    /// Upper enclosure `C_K^+ >= C_K(d)`.
    pub c_k_upper: Q,
    /// `U_K^- = max(0, 1 - C_K^+ - rho_F^{K+1})`, a certified lower bound on `m_F`.
    pub u_k_minus: Q,
    /// `C_K^+ - C_K^-`: the numerical width of the enclosure of `1 - C_K(d)`.
    pub numeric_width: Q,
    /// `rho_F^{K+1}` (upper end): the omitted mixture mass.
    pub truncation: Q,
    /// Largest width of the enclosure of the distribution function of `Y` on the grid.
    pub y_cdf_width: f64,
    /// Computation time.
    pub seconds: f64,
}

impl WorkloadBound {
    pub fn to_json(&self) -> Value {
        json!({"k": self.k, "step": self.step.to_string(), "cells": self.cells,
               "c_k_lower": to_f64(&self.c_k_lower), "u_k_plus": q_json(&self.u_k_plus),
               "c_k_upper": to_f64(&self.c_k_upper), "u_k_minus": q_json(&self.u_k_minus),
               "numeric_width": to_f64(&self.numeric_width),
               "truncation": to_f64(&self.truncation), "y_cdf_width": self.y_cdf_width, "seconds": self.seconds})
    }
}

/// `x` lowered by one ulp and clamped at zero: a lower bound after a
/// round-to-nearest operation on nonnegative values.
fn dn(x: f64) -> f64 {
    x.next_down().max(0.0)
}

/// `x` raised by one ulp: an upper bound after a round-to-nearest operation.
fn up(x: f64) -> f64 {
    x.next_up()
}

impl Homogeneous {
    /// The smallest `K` with `rho_F^{K+1} <= tol`, evaluated in doubles at
    /// `rho_F^+` and capped at `cap` (any `K` gives valid bounds); `None` when
    /// `rho_F^+ >= 1` (OJF is not proved stable).
    pub fn workload_terms(&self, bits: u32, tol: f64, cap: usize) -> Option<usize> {
        let r = to_f64(&self.rho_f(bits).1);
        if r >= 1.0 {
            return None;
        }
        if r <= 0.0 {
            return Some(0);
        }
        let terms = (tol.ln() / r.ln()).ceil().max(1.0) as usize;
        Some(terms.saturating_sub(1).min(cap))
    }

    /// The workload bound from `K + 1` mixture terms on the grid of step
    /// `d / ceil(d / step)`, at most `step`; an error when OJF is not proved
    /// stable or the two enclosures of `C_K` are inverted.
    pub fn workload_bound(&self, step: &Q, k: usize, bits: u32) -> Result<WorkloadBound, String> {
        let t0 = Instant::now();
        let rho_f = self.rho_f(bits);
        if rho_f.1 >= Q::one() {
            return Err("the reference policy is not proved stable (rho_F^+ >= 1)".into());
        }
        let cells = (&self.d / step)
            .ceil()
            .to_integer()
            .to_usize()
            .ok_or("cells")?
            .max(1);
        let g = &self.d / qi(cells as i64);
        // 1/N and q = 1 - 1/N with directed rounding (both exact when N is a power of two)
        let q_exact = self.q();
        let inv_n = Q::one() / qi(self.n as i64);
        let (q_lo, q_hi) = (q_to_f64_down(&q_exact), q_to_f64_up(&q_exact));
        let (inv_n_lo, inv_n_hi) = (q_to_f64_down(&inv_n), q_to_f64_up(&inv_n));
        // survival of tau at the grid points and the cell masses
        let mut s_lo = Vec::with_capacity(cells + 1);
        let mut s_hi = Vec::with_capacity(cells + 1);
        for m in 0..=cells {
            let (lo, hi) = self.law.survival(&(&g * qi(m as i64)), bits);
            s_lo.push(q_to_f64_down(&lo).clamp(0.0, 1.0));
            s_hi.push(q_to_f64_up(&hi).clamp(0.0, 1.0));
        }
        let p_lo: Vec<f64> = (0..=cells)
            .map(|m| {
                if m == 0 {
                    0.0
                } else {
                    dn(s_lo[m - 1] - s_hi[m])
                }
            })
            .collect();
        let p_hi: Vec<f64> = (0..=cells)
            .map(|m| {
                if m == 0 {
                    0.0
                } else {
                    up(s_hi[m - 1] - s_lo[m]).min(1.0)
                }
            })
            .collect();
        // an atom of tau at zero, P(tau = 0) in [1 - s_hi[0], 1 - s_lo[0]]; both ends are exactly
        // zero for every distribution with P(tau > 0) = 1, as the model requires, so the atom
        // terms below vanish for them
        let atom_lo = dn(1.0 - s_hi[0]);
        let atom_hi = if s_lo[0] < 1.0 {
            up(1.0 - s_lo[0]).min(1.0)
        } else {
            0.0
        };
        // exp(-mu j g), j = 0..=cells + 1
        let (e_lo_q, e_hi_q) = exp_neg_powers(&(&self.mu * &g), cells + 1, bits);
        let e_lo: Vec<f64> = e_lo_q
            .iter()
            .map(|x| q_to_f64_down(x).clamp(0.0, 1.0))
            .collect();
        let e_hi: Vec<f64> = e_hi_q
            .iter()
            .map(|x| q_to_f64_up(x).clamp(0.0, 1.0))
            .collect();
        // distribution function of Y at the grid points:
        // (1/N)(1 - e^{-mu n g}) + q P(tau + S <= n g)
        let mut f_lo = vec![0.0f64; cells + 1];
        let mut f_hi = vec![0.0f64; cells + 1];
        let mut width = 0.0f64;
        for n in 1..=cells {
            let mut lo = 0.0f64;
            let mut hi = 0.0f64;
            for m in 1..=n {
                // a duration in ((m-1)g, mg] leaves at least (n-m)g and at most
                // (n-m+1)g for the service
                lo = dn(lo + dn(p_lo[m] * dn(1.0 - e_hi[n - m])));
                hi = up(hi + up(p_hi[m] * up(1.0 - e_lo[n - m + 1])));
            }
            f_lo[n] = dn(dn(dn(1.0 - e_hi[n]) * inv_n_lo) + dn(q_lo * lo)).min(1.0);
            f_hi[n] = up(up(1.0 - e_lo[n]) * inv_n_hi + up(q_hi * hi)).min(1.0);
            if atom_lo > 0.0 {
                f_lo[n] = dn(f_lo[n] + dn(q_lo * dn(atom_lo * dn(1.0 - e_hi[n])))).min(1.0);
            }
            if atom_hi > 0.0 {
                f_hi[n] = up(f_hi[n] + up(q_hi * up(atom_hi * up(1.0 - e_lo[n])))).min(1.0);
            }
            width = width.max(f_hi[n] - f_lo[n]);
        }
        // Lower masses of Ybar >= Y, which rounds the reconfiguration time and the service time
        // up to the grid separately: Ybar = ceil(S/g) g with probability 1/N and
        // ceil(tau/g) g + ceil(S/g) g with probability q. Unlike lower bounds on the cell masses
        // of Y itself, which lose about mu g of the probability (a duration that ends inside the
        // cell where tau ends), Ybar keeps all of it, and Ybar >= Y can only lower
        // P(Ybar + Rbar_1 + ... + Rbar_n <= d). Its mass at ng is
        // (1/N)(e^{-mu(n-1)g} - e^{-mu n g}) + q sum_{m<n} P(tau in cell m) e^{-mu(n-m-1)g}(1 - e^{-mu g}).
        // (Differences of the distribution-function enclosures would lose the enclosure width
        // at every cell.)
        let one_minus_eg_lo = dn(1.0 - e_hi[1]);
        let mut h_lo = vec![0.0f64; cells + 1];
        for n in 1..=cells {
            let mut lo = 0.0f64;
            for m in 1..n {
                lo = dn(lo + dn(p_lo[m] * dn(e_lo[n - m - 1] * one_minus_eg_lo)));
            }
            h_lo[n] = dn(dn(dn(e_lo[n - 1] - e_hi[n]) * inv_n_lo) + dn(q_lo * lo));
            if atom_lo > 0.0 {
                h_lo[n] = dn(h_lo[n] + dn(q_lo * dn(atom_lo * dn(e_lo[n - 1] - e_hi[n]))));
            }
        }
        let (mlo, mhi) = self.law.mean(bits);
        let ey_hi = q_to_f64_up(&(self.mu.recip() + &self.q() * mhi));
        let ey_lo = q_to_f64_down(&(self.mu.recip() + &self.q() * mlo));
        let gf = q_to_f64_down(&g);
        let r_lo: Vec<f64> = (0..=cells)
            .map(|n| {
                if n == 0 {
                    0.0
                } else {
                    dn(dn(gf * dn(1.0 - f_hi[n])) / ey_hi)
                }
            })
            .collect();
        // convolutions: lower bounds on P(Ybar + Rbar_1 + ... + Rbar_n <= d), where the bars
        // denote durations rounded up to the end of their cells
        let mut conv = h_lo.clone();
        let mut terms: Vec<f64> = Vec::with_capacity(k + 1);
        terms.push(conv.iter().fold(0.0, |acc, x| dn(acc + x)));
        for _ in 0..k {
            let mut next = vec![0.0f64; cells + 1];
            for i in 1..=cells {
                if conv[i] == 0.0 {
                    continue;
                }
                for j in (i + 1)..=cells {
                    next[j] = dn(next[j] + dn(conv[i] * r_lo[j - i]));
                }
            }
            conv = next;
            terms.push(conv.iter().fold(0.0, |acc, x| dn(acc + x)));
        }
        let rho_lo = q_to_f64_down(&rho_f.0).max(0.0);
        let rho_hi = q_to_f64_up(&rho_f.1);
        let mut c = 0.0f64;
        let mut pow = 1.0f64;
        for term in &terms {
            c = dn(c + dn(pow * term));
            pow = dn(pow * rho_lo);
        }
        let c_k = dn(dn(1.0 - rho_hi) * c).clamp(0.0, 1.0);
        let c_k_q = Q::from_float(c_k).ok_or("c_k")?;
        let mut trunc = Q::one();
        for _ in 0..=k {
            trunc = round_up(&(&trunc * &rho_f.1), bits);
        }

        // ---- the mirror: an upper enclosure C_K^+ >= C_K(d) ----
        // Upper cell masses of Y: (1/N)(e^{-mu(n-1)g} - e^{-mu n g})
        // + q sum_{m <= n} P(tau in cell m) sup_t P(S in ((n-1)g - t, ng - t]), where a duration
        // t in ((m-1)g, mg] gives at most e^{-mu(n-1-m)g}(1 - e^{-mu g}) for m < n and at most
        // 1 - e^{-mu g} for m = n.
        let one_minus_eg_hi = up(1.0 - e_lo[1]).min(1.0);
        let mut h_hi = vec![0.0f64; cells + 1];
        for n in 1..=cells {
            let mut hi = 0.0f64;
            for m in 1..n {
                hi = up(hi + up(p_hi[m] * up(e_hi[n - 1 - m] * one_minus_eg_hi)));
            }
            hi = up(hi + up(p_hi[n] * one_minus_eg_hi));
            let s_cell = up((e_hi[n - 1] - e_lo[n]).max(0.0));
            let mut v = up(up(s_cell * inv_n_hi) + up(q_hi * hi));
            if atom_hi > 0.0 {
                v = up(v + up(q_hi * up(atom_hi * s_cell)));
            }
            h_hi[n] = v.min(1.0);
        }
        // Upper equilibrium cell masses: P(R in ((l-1)g, lg]) <= g P(Y > (l-1)g) / E[Y], since
        // P(Y > u) is nonincreasing; P(Y > 0) = 1.
        let g_up = q_to_f64_up(&g);
        let r_hi: Vec<f64> = (0..=cells)
            .map(|l| {
                if l == 0 {
                    0.0
                } else {
                    up(up(g_up * up(1.0 - f_lo[l - 1])) / ey_lo).min(1.0)
                }
            })
            .collect();
        // Every variable in cell l exceeds (l-1)g, so a combination of cells can finish by d
        // only when the sum of the lower ends, a g with a = sum (l - 1), is below d, i.e.
        // a <= cells - 1.
        let mut conv_hi: Vec<f64> = h_hi[1..=cells].to_vec();
        let mut terms_hi: Vec<f64> = Vec::with_capacity(k + 1);
        terms_hi.push(conv_hi.iter().fold(0.0, |acc, x| up(acc + x)).min(1.0));
        for _ in 0..k {
            let mut next = vec![0.0f64; cells];
            for a in 0..cells {
                if conv_hi[a] == 0.0 {
                    continue;
                }
                for l in 1..=(cells - a) {
                    let b = a + l - 1;
                    next[b] = up(next[b] + up(conv_hi[a] * r_hi[l]));
                }
            }
            conv_hi = next;
            terms_hi.push(conv_hi.iter().fold(0.0, |acc, x| up(acc + x)).min(1.0));
        }
        // mixture weights (1 - rho_F) rho_F^n <= (1 - rho_F^-) (rho_F^+)^n
        let mut c_up = 0.0f64;
        let mut pow_up = 1.0f64;
        for term in &terms_hi {
            c_up = up(c_up + up(pow_up * term));
            pow_up = up(pow_up * rho_hi);
        }
        let c_k_hi = up(up(1.0 - rho_lo) * c_up).clamp(0.0, 1.0);
        let c_k_hi_q = Q::from_float(c_k_hi).ok_or("c_k_hi")?;
        if c_k_hi_q < c_k_q {
            return Err(format!(
                "workload enclosure inverted: C_K^+ = {c_k_hi} < C_K^- = {c_k}"
            ));
        }
        let u_k_minus = (Q::one() - &c_k_hi_q - &trunc).max(Q::zero());
        Ok(WorkloadBound {
            k,
            step: g,
            cells,
            c_k_lower: c_k_q.clone(),
            u_k_plus: (Q::one() - &c_k_q).min(Q::one()),
            numeric_width: &c_k_hi_q - &c_k_q,
            c_k_upper: c_k_hi_q,
            u_k_minus,
            truncation: trunc,
            y_cdf_width: width,
            seconds: t0.elapsed().as_secs_f64(),
        })
    }
}

/// The verdict on a common target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// A verified lower bound or witness excludes the target.
    CertifiedInfeasible,
    /// OJF, proved stable, has a certified upper bound at or below the target.
    CertifiedFeasible,
    /// Neither is established.
    Undetermined,
}

impl Verdict {
    /// The name used in output records.
    pub fn name(&self) -> &'static str {
        match self {
            Verdict::CertifiedInfeasible => "certified_infeasible",
            Verdict::CertifiedFeasible => "certified_feasible",
            Verdict::Undetermined => "undetermined",
        }
    }
}

/// The verdict on the common target `eps` from a verified exclusion
/// (`lp_infeasible`: the relaxation with the target rows is certified
/// empty), the certified lower bound `l_safe` and the certified achievable
/// bound `u_plus`. A target that is both excluded and met is a correctness
/// failure, returned as an error rather than resolved by a tolerance.
pub fn verdict(
    eps: &Q,
    lp_infeasible: bool,
    l_safe: &Q,
    u_plus: Option<&Q>,
) -> Result<Verdict, String> {
    let excluded = lp_infeasible || l_safe > eps;
    let met = u_plus.is_some_and(|u| u <= eps);
    if excluded && met {
        return Err(format!(
            "correctness failure: a certified lower bound ({}, witness: {lp_infeasible}) excludes the target {} while a certified achievable bound {} meets it",
            to_f64(l_safe),
            to_f64(eps),
            u_plus.map(to_f64).unwrap_or(f64::NAN)
        ));
    }
    Ok(if excluded {
        Verdict::CertifiedInfeasible
    } else if met {
        Verdict::CertifiedFeasible
    } else {
        Verdict::Undetermined
    })
}

/// The certified interval `L_safe <= m* <= U^+` of one setting of the
/// symmetric case.
#[derive(Clone, Debug)]
pub struct Bracket {
    pub homogeneous: Homogeneous,
    /// The certified lower bound `L_LP` of the relaxation (`min max_i x_i`,
    /// targets dropped).
    pub lp_lower: Option<Q>,
    pub lp_status: String,
    /// Whether the relaxation with the target rows is certified empty.
    pub lp_infeasible_at_target: bool,
    pub tail: TailEnclosure,
    /// Enclosure of the no-waiting lower bound `b(d)`.
    pub floor: Iv,
    pub rho_f: Iv,
    pub u_simple: Option<Q>,
    pub benchmark: Option<Benchmark>,
    /// The workload bound, when OJF is proved stable and the enclosure succeeds.
    pub workload: Option<WorkloadBound>,
    /// The sharpest certified achievable bound, `min(U_simple^+, U_exact^+, U_K^+)`.
    pub u_plus: Option<Q>,
    /// `max(L_LP, b^-)`.
    pub l_safe: Q,
    /// `U^+ - L_safe`.
    pub gap_safe: Option<Q>,
    /// `U^+ - L_LP`.
    pub gap_raw: Option<Q>,
    /// The verdict, or the correctness failure.
    pub verdict: Result<Verdict, String>,
}

/// Assemble the certified interval of a setting of the symmetric case from
/// its relaxation results. The workload bound uses the smallest `K` with
/// `rho_F^{K+1} <= 10^-6`, at most 64.
pub fn bracket(
    h: &Homogeneous,
    lp_lower: Option<&Q>,
    lp_status: &str,
    lp_infeasible_at_target: bool,
    settings: &GridSettings,
) -> Result<Bracket, String> {
    let bits = settings.bits;
    let tail = h.tail(settings)?;
    let floor = h.floor_from_tail(&tail.used, bits);
    let rho_f = h.rho_f(bits);
    let u_simple = Homogeneous::u_simple(&floor, &rho_f, bits);
    let benchmark = h.exact_exponential_benchmark(bits).unwrap_or_default();
    let workload = h
        .workload_terms(bits, 1e-6, 64)
        .and_then(|k| h.workload_bound(&settings.step, k, bits).ok());
    let candidates: Vec<Q> = u_simple
        .iter()
        .cloned()
        .chain(benchmark.iter().map(|b| b.miss.1.clone()))
        .chain(workload.iter().map(|w| w.u_k_plus.clone()))
        .collect();
    let u_plus = candidates.into_iter().min();
    let l_safe = lp_lower
        .cloned()
        .unwrap_or_else(Q::zero)
        .max(floor.0.clone());
    let gap_safe = u_plus.as_ref().map(|u| u - &l_safe);
    let gap_raw = match (&u_plus, lp_lower) {
        (Some(u), Some(l)) => Some(u - l),
        _ => None,
    };
    let verdict = verdict(&h.eps, lp_infeasible_at_target, &l_safe, u_plus.as_ref());
    Ok(Bracket {
        homogeneous: h.clone(),
        lp_lower: lp_lower.cloned(),
        lp_status: lp_status.into(),
        lp_infeasible_at_target,
        tail,
        floor,
        rho_f,
        u_simple,
        benchmark,
        workload,
        u_plus,
        l_safe,
        gap_safe,
        gap_raw,
        verdict,
    })
}

impl Bracket {
    pub fn to_json(&self) -> Value {
        let opt = |x: &Option<Q>| x.as_ref().map(q_json).unwrap_or(Value::Null);
        json!({
            "homogeneous": self.homogeneous.to_json(),
            "lp_lower": opt(&self.lp_lower), "lp_status": self.lp_status, "lp_infeasible_at_target": self.lp_infeasible_at_target,
            "tail": self.tail.to_json(), "floor_b": iv_json(&self.floor), "rho_f": iv_json(&self.rho_f),
            "u_simple": opt(&self.u_simple), "benchmark": self.benchmark.as_ref().map(|b| b.to_json()).unwrap_or(Value::Null),
            "u_workload": self.workload.as_ref().map(|w| w.to_json()).unwrap_or(Value::Null),
            "u_plus": opt(&self.u_plus), "l_safe": q_json(&self.l_safe),
            "gap_safe": opt(&self.gap_safe), "gap_raw": opt(&self.gap_raw),
            "verdict": match &self.verdict { Ok(v) => json!(v.name()), Err(e) => json!({"error": e}) }
        })
    }
}
