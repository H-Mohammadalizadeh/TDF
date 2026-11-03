//! Reconfiguration-time distributions of the witness with exact rational
//! parameters, built from the checker's [`Distribution`]: rigorous enclosures
//! of the mean, of Psi(x) = E(x - tau)^+ (the integrated distribution
//! function), the mean-preserving spread ("randomized rounding") lattice laws
//! built from Psi, and the partitions used for the mixed Poisson weights of
//! reconfigurations outside the target's intervisit period.
//!
//! Spread of tau onto points y_0 < y_1 < ... (Strassen 1965): tau in
//! [y_l, y_{l+1}] goes to y_l with probability (y_{l+1} - tau)/(y_{l+1} - y_l)
//! and to y_{l+1} otherwise, so E[tau~ | tau] = tau and tau <=_cx tau~. Its
//! law is the second divided difference of Psi:
//! P(tau~ = y_l) = slope_l - slope_{l-1}, slope_l = (Psi(y_{l+1}) - Psi(y_l)) / (y_{l+1} - y_l).

use crate::feasibility::distributions::Distribution;
use crate::feasibility::rational::{Q, exp_neg, qi, qr, round_down, round_up, to_f64};
use crate::feasibility::special::{
    iadd, idiv, imul, isub, ln_enclosure, norm_cdf_enclosure, point, pow_rational_enclosure,
    sqrt_enclosure,
};
use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};

/// Working precision (fractional bits) of the rational enclosures.
pub const BITS: u32 = 128;

/// Largest f64 not above the rational `x`.
pub fn q_dn(x: &Q) -> f64 {
    let mut f = to_f64(x);
    if !f.is_finite() {
        f = if x.is_negative() { f64::MIN } else { f64::MAX };
    }
    loop {
        let fq = Q::from_float(f).expect("finite");
        if fq <= *x {
            return f;
        }
        f = f.next_down();
    }
}

/// Smallest f64 not below the rational `x`.
pub fn q_up(x: &Q) -> f64 {
    let mut f = to_f64(x);
    if !f.is_finite() {
        f = if x.is_negative() { f64::MIN } else { f64::MAX };
    }
    loop {
        let fq = Q::from_float(f).expect("finite");
        if fq >= *x {
            return f;
        }
        f = f.next_up();
    }
}

/// Exact rational value of an f64 (panics on infinity or NaN, so that an
/// overflowed bound is never used).
pub fn fq(x: f64) -> Q {
    Q::from_float(x).expect("finite")
}

#[derive(Clone, Debug, PartialEq)]
pub enum Law {
    /// Point mass at c.
    Det { c: Q },
    /// Mean 1 / rate.
    Exp { rate: Q },
    /// Erlang: sum of k independent Exp(rate).
    Erlang { k: u32, rate: Q },
    /// N(mu0, sigma0^2) conditioned on a positive value.
    TNorm { mu0: Q, sigma0: Q },
    /// Lognormal with the given mean and squared coefficient of variation.
    LogN { mean: Q, cv2: Q },
    /// P(tau > x) = (xm / x)^alpha for x >= xm.
    Pareto { alpha: Q, xm: Q },
}

impl Law {
    /// The law of a checker distribution with the same exact parameters; a
    /// gamma distribution needs an integer shape (Erlang), and finite-support
    /// distributions other than a point mass are not supported.
    pub fn from_distribution(d: &Distribution) -> Result<Law, String> {
        Ok(match d {
            Distribution::Deterministic { value } => Law::Det { c: value.clone() },
            Distribution::Exponential { rate } => Law::Exp { rate: rate.clone() },
            Distribution::Gamma { shape, rate } => {
                let k = shape
                    .is_integer()
                    .then(|| shape.to_integer().to_u32())
                    .flatten()
                    .filter(|&k| k >= 1)
                    .ok_or_else(|| {
                        format!("the witness needs an integer gamma shape, got {shape}")
                    })?;
                Law::Erlang {
                    k,
                    rate: rate.clone(),
                }
            }
            Distribution::TruncNormal { mu0, sigma0 } => Law::TNorm {
                mu0: mu0.clone(),
                sigma0: sigma0.clone(),
            },
            Distribution::Lognormal { mean, cv2 } => Law::LogN {
                mean: mean.clone(),
                cv2: cv2.clone(),
            },
            Distribution::Pareto { alpha, xm } => Law::Pareto {
                alpha: alpha.clone(),
                xm: xm.clone(),
            },
            Distribution::Discrete(_) => {
                return Err(format!("the witness does not support {}", d.id()));
            }
        })
    }

    /// Enclosure of the mean (exact except for the truncated normal).
    pub fn mean(&self) -> (Q, Q) {
        match self {
            Law::Det { c } => point(c),
            Law::Exp { rate } => point(&rate.recip()),
            Law::Erlang { k, rate } => point(&(qi(*k as i64) / rate)),
            Law::LogN { mean, .. } => point(mean),
            Law::Pareto { alpha, xm } => point(&(alpha * xm / (alpha - Q::one()))),
            Law::TNorm { mu0, sigma0 } => {
                // mu0 + sigma0 phi(a) / (1 - Phi(a)), a = -mu0/sigma0
                let w = BITS + 32;
                let a = -mu0 / sigma0;
                let (cdf, pdf) = normal_at(&a, w);
                let tail = (Q::one() - &cdf.1, Q::one() - &cdf.0);
                let lam = idiv(&pdf, &tail, w);
                iadd(&point(mu0), &imul(&lam, &point(sigma0), w), BITS)
            }
        }
    }

    /// Upper bound of E[tau^2] (infinite for Pareto with alpha <= 2).
    pub fn second_moment_up(&self) -> f64 {
        match self {
            Law::Det { c } => q_up(&(c * c)),
            Law::Exp { rate } => q_up(&(qi(2) / (rate * rate))),
            Law::Erlang { k, rate } => {
                let k = qi(*k as i64);
                q_up(&(&k * (&k + Q::one()) / (rate * rate)))
            }
            Law::LogN { mean, cv2 } => q_up(&(mean * mean * (Q::one() + cv2))),
            Law::Pareto { alpha, xm } => {
                if *alpha <= qi(2) {
                    f64::INFINITY
                } else {
                    q_up(&(alpha * xm * xm / (alpha - qi(2))))
                }
            }
            Law::TNorm { mu0, sigma0 } => {
                // E[tau^2] <= E[N(mu0,sigma0)^2] / P(N > 0) = (mu0^2 + sigma0^2) / (1 - Phi(a))
                let a = -mu0 / sigma0;
                let cdf = norm_cdf_enclosure(&a, BITS);
                let tail = Q::one() - &cdf.1;
                q_up(&((mu0 * mu0 + sigma0 * sigma0) / tail))
            }
        }
    }

    /// Constants of Psi that do not depend on x.
    pub fn prepare(&self) -> Prep {
        let w = BITS + 32;
        match self {
            Law::TNorm { mu0, sigma0 } => {
                let a = -mu0 / sigma0;
                let (pa, da) = normal_at(&a, w);
                let den = (Q::one() - &pa.1, Q::one() - &pa.0);
                Prep {
                    tn: Some((pa, da, den)),
                    ln: None,
                }
            }
            Law::LogN { mean, cv2 } => {
                let s2 = ln_enclosure(&(Q::one() + cv2), w);
                let sigma = (sqrt_enclosure(&s2.0, w).0, sqrt_enclosure(&s2.1, w).1);
                let mu = isub(&ln_enclosure(mean, w), &imul(&s2, &point(&qr(1, 2)), w), w);
                Prep {
                    tn: None,
                    ln: Some((sigma, mu)),
                }
            }
            _ => Prep { tn: None, ln: None },
        }
    }

    /// Enclosure of Psi(x) = E(x - tau)^+ for rational x.
    pub fn psi(&self, x: &Q, prep: &Prep) -> (Q, Q) {
        if !x.is_positive() {
            return (Q::zero(), Q::zero());
        }
        let w = BITS + 32;
        match self {
            Law::Det { c } => {
                let v = if x > c { x - c } else { Q::zero() };
                point(&v)
            }
            Law::Exp { rate } => {
                // x - (1 - e^{-rate x}) / rate, increasing in e^{-rate x}
                let (elo, ehi) = exp_neg(&(rate * x), w);
                let f = |e: &Q| x - (Q::one() - e) / rate;
                (round_down(&f(&elo), BITS), round_up(&f(&ehi), BITS))
            }
            Law::Erlang { k, rate } => {
                // x P(k, y) - (k/rate) P(k+1, y), y = rate x, P(k,y) = 1 - e^{-y} sum_{j<k} y^j/j!
                let y = rate * x;
                let mut poly_k = Q::zero();
                let mut term = Q::one();
                for j in 0..*k {
                    if j > 0 {
                        term = term * &y / qi(j as i64);
                    }
                    poly_k += &term;
                }
                let term_k = term * &y / qi(*k as i64);
                let poly_k1 = &poly_k + &term_k;
                let kr = qi(*k as i64) / rate;
                // Psi = (x - k/rate) + e (k/rate poly_{k+1} - x poly_k)
                let coef = &kr * &poly_k1 - x * &poly_k;
                let (elo, ehi) = exp_neg(&y, w);
                let base = x - &kr;
                let a = &base + &coef * &elo;
                let b = &base + &coef * &ehi;
                let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                (round_down(&lo, BITS), round_up(&hi, BITS))
            }
            Law::Pareto { alpha, xm } => {
                if x <= xm {
                    return (Q::zero(), Q::zero());
                }
                let am1 = alpha - Q::one();
                let base = x - alpha * xm / &am1;
                let r = xm / x;
                if *alpha == qi(3) {
                    let v = &base + xm * &r * &r / qi(2);
                    return point(&v);
                }
                let p = pow_rational_enclosure(&r, &am1, w);
                let c = xm / &am1;
                (
                    round_down(&(&base + &c * &p.0), BITS),
                    round_up(&(&base + &c * &p.1), BITS),
                )
            }
            Law::TNorm { mu0, sigma0 } => {
                let (pa, da, den) = prep.tn.as_ref().expect("prepared");
                let beta = (x - mu0) / sigma0;
                let (pb, db) = normal_at(&beta, w);
                // [(x - mu0)(Phi(b) - Phi(a)) + sigma0 (phi(b) - phi(a))] / (1 - Phi(a))
                let t1 = imul(&point(&(x - mu0)), &isub(&pb, pa, w), w);
                let t2 = imul(&point(sigma0), &isub(&db, da, w), w);
                let num = iadd(&t1, &t2, w);
                let v = idiv(&num, den, BITS);
                (v.0.max(Q::zero()), v.1.max(Q::zero()))
            }
            Law::LogN { mean, .. } => {
                let (sigma, mu) = prep.ln.as_ref().expect("prepared");
                let z = idiv(&isub(&ln_enclosure(x, w), mu, w), sigma, w);
                let zs = isub(&z, sigma, w);
                let phi_iv = |z: &(Q, Q)| -> (Q, Q) {
                    // Phi increasing with slope at most 1/sqrt(2 pi) < 2/5
                    let base = norm_cdf_enclosure(&round_down(&z.0, w), w);
                    let slack = round_up(&(qr(2, 5) * (&z.1 - &round_down(&z.0, w))), w);
                    (base.0, (base.1 + slack).min(Q::one()))
                };
                let pz = phi_iv(&z);
                let pzs = phi_iv(&zs);
                let v = isub(&imul(&point(x), &pz, w), &imul(&point(mean), &pzs, w), BITS);
                (v.0.max(Q::zero()), v.1.max(Q::zero()))
            }
        }
    }

    /// Enclosure of `E(tau - t)^+ = E[tau] - t + Psi(t)`.
    pub fn tail_mean(&self, t: &Q, prep: &Prep) -> (Q, Q) {
        let m = self.mean();
        let p = self.psi(t, prep);
        let lo = &m.0 - t + &p.0;
        let hi = &m.1 - t + &p.1;
        (lo.max(Q::zero()), hi.max(Q::zero()))
    }

    /// A cap T with E(tau - T)^+ tiny relative to the mean (rel ~ 1e-13).
    pub fn cap(&self) -> Q {
        let m = to_f64(&self.mean().1);
        let v = match self {
            Law::Det { c } => to_f64(c),
            Law::Exp { .. } => 40.0 * m,
            Law::Erlang { .. } => 30.0 * m,
            Law::TNorm { mu0, sigma0 } => to_f64(mu0) + 10.0 * to_f64(sigma0),
            Law::LogN { cv2, .. } => {
                let s2 = (1.0 + to_f64(cv2)).ln();
                let s = s2.sqrt();
                m * (0.5 * s2 + 7.6 * s).exp()
            }
            Law::Pareto { alpha, xm } => {
                // xm^a T^(1-a) / (a-1) = 1e-13 m
                let a = to_f64(alpha);
                let xm = to_f64(xm);
                ((xm.powf(a) / ((a - 1.0) * 1e-13 * m)).powf(1.0 / (a - 1.0))).min(1e30)
            }
        };
        dyadic(v, 20)
    }
}

/// Enclosures of Phi(b) and phi(b) for rational b, evaluated at the dyadic
/// roundings b_lo <= b <= b_hi (w fractional bits) so that the series work on
/// short numbers: Phi is increasing; phi is increasing below 0 and decreasing above.
pub fn normal_at(b: &Q, w: u32) -> ((Q, Q), (Q, Q)) {
    let bl = round_down(b, w);
    let bh = round_up(b, w);
    let pl = norm_cdf_fast(&bl, w);
    let ph = if bh == bl {
        pl.clone()
    } else {
        norm_cdf_fast(&bh, w)
    };
    let dl = norm_pdf_fast(&bl, w);
    let dh = if bh == bl {
        dl.clone()
    } else {
        norm_pdf_fast(&bh, w)
    };
    let dens = if !bl.is_negative() {
        (dh.0.clone(), dl.1.clone())
    } else if !bh.is_positive() {
        (dl.0.clone(), dh.1.clone())
    } else {
        let d0 = norm_pdf_fast(&Q::zero(), w);
        (dl.0.clone().min(dh.0.clone()), d0.1)
    };
    ((pl.0, ph.1), dens)
}

/// e^f scaled by 2^w for 0 <= f <= 1 in fixed point: (lo, hi) integers with
/// lo <= 2^w e^f <= hi. Terms f^k/k! are rounded toward their bound; the tail
/// after the last kept term (once terms are below one unit and k >= 2) is at
/// most twice the last term.
fn exp_unit_fixed(f: &Q, w: u32) -> (BigInt, BigInt) {
    use num_integer::Integer;
    let scale: BigInt = BigInt::one() << w;
    let flo: BigInt = (f.numer() * &scale).div_floor(f.denom());
    let fhi: BigInt = (f.numer() * &scale).div_ceil(f.denom());
    let (mut slo, mut shi) = (scale.clone(), scale.clone());
    let (mut tlo, mut thi) = (scale.clone(), scale.clone());
    let mut k: u64 = 1;
    loop {
        let den = BigInt::from(k) << w;
        tlo = (&tlo * &flo).div_floor(&den);
        thi = (&thi * &fhi).div_ceil(&den);
        slo += &tlo;
        shi += &thi;
        k += 1;
        if thi <= BigInt::one() && k > 3 {
            shi += BigInt::from(3u32);
            return (slo, shi);
        }
        assert!(k < 10_000);
    }
}

/// Enclosure of e^{-x} for rational x >= 0 with fixed-point series (fast for
/// arguments with long denominators).
pub fn exp_neg_fast(x: &Q, bits: u32) -> (Q, Q) {
    assert!(!x.is_negative());
    let w = bits + 40;
    let m = x.floor();
    let f = x - &m;
    let m = m
        .to_integer()
        .to_u64()
        .expect("exp_neg_fast argument too large");
    let scale = Q::from_integer(BigInt::one() << w);
    let (elo, ehi) = exp_unit_fixed(&f, w);
    // e^{-f} in [2^w / ehi, 2^w / elo]
    let mut lo = &scale / Q::from_integer(ehi);
    let mut hi = &scale / Q::from_integer(elo);
    if m > 0 {
        let (e1lo, e1hi) = exp_unit_fixed(&Q::one(), w);
        let inv_lo = round_down(&(&scale / Q::from_integer(e1hi)), w); // <= e^{-1}
        let inv_hi = round_up(&(&scale / Q::from_integer(e1lo)), w);
        let (mut plo, mut phi) = (Q::one(), Q::one());
        let (mut blo, mut bhi) = (inv_lo, inv_hi);
        let mut e = m;
        while e > 0 {
            if e & 1 == 1 {
                plo = round_down(&(&plo * &blo), w);
                phi = round_up(&(&phi * &bhi), w);
            }
            blo = round_down(&(&blo * &blo), w);
            bhi = round_up(&(&bhi * &bhi), w);
            e >>= 1;
        }
        lo *= plo;
        hi *= phi;
    }
    (round_down(&lo, bits), round_up(&hi, bits))
}

/// Enclosure of Phi(b) at dyadic b: the series of the reference crate for
/// |b| < 8, the Mills-ratio bounds x/(1+x^2) phi(x) <= 1 - Phi(x) <= phi(x)/x
/// (x = |b| >= 8) otherwise.
pub fn norm_cdf_fast(b: &Q, w: u32) -> (Q, Q) {
    let x = b.abs();
    if x < qi(8) {
        return norm_cdf_enclosure(b, w);
    }
    let d = norm_pdf_fast(&x, w + 8);
    let t_hi = round_up(&(&d.1 / &x), w);
    let t_lo = round_down(&(&d.0 * &x / (Q::one() + &x * &x)), w);
    if b.is_positive() {
        (Q::one() - t_hi, Q::one() - t_lo)
    } else {
        (t_lo, t_hi)
    }
}

/// Enclosure of the standard normal density at dyadic b.
pub fn norm_pdf_fast(b: &Q, w: u32) -> (Q, Q) {
    let t = b * b / qi(2);
    let tl = round_down(&t, w + 8);
    let th = round_up(&t, w + 8);
    let e = (exp_neg_fast(&th, w + 8).0, exp_neg_fast(&tl, w + 8).1);
    let c = crate::feasibility::special::sqrt_two_pi(w + 16);
    let lo = round_down(&(&e.0 / &c.1), w);
    let hi = round_up(&(&e.1 / &c.0), w);
    (lo, hi)
}

/// A dyadic rational close to v with `bits` fractional bits (at least 2^-bits).
pub fn dyadic(v: f64, bits: u32) -> Q {
    let s = (v * (1u64 << bits) as f64).round().max(1.0);
    Q::new(BigInt::from(s as u128), BigInt::one() << bits)
}

/// Prepared constants of Psi.
#[derive(Clone, Debug)]
pub struct Prep {
    /// truncated normal: Phi(a), phi(a), 1 - Phi(a)
    tn: Option<((Q, Q), (Q, Q), (Q, Q))>,
    /// lognormal: sigma, mu
    ln: Option<((Q, Q), (Q, Q))>,
}

/// Spread of tau onto the lattice g Z (points m g, m = 0..len-1): lower and
/// upper bounds of P(tau~ = m g). Mass beyond the last point is not returned.
pub fn spread_lattice(law: &Law, g: &Q, len: usize) -> (Vec<f64>, Vec<f64>) {
    let mut lo = vec![0.0; len];
    let mut hi = vec![0.0; len];
    match law {
        Law::Det { c } => {
            // atoms at floor(c/g) and +1
            let t = c / g;
            let k = t.floor();
            let f = &t - &k;
            let k = k.to_integer().to_usize().unwrap();
            if k < len {
                let v = Q::one() - &f;
                lo[k] = q_dn(&v);
                hi[k] = q_up(&v);
            }
            if k + 1 < len && f.is_positive() {
                lo[k + 1] = q_dn(&f);
                hi[k + 1] = q_up(&f);
            }
        }
        _ => {
            let prep = law.prepare();
            // Psi at x_{m} for m = 0..len (Psi(-g) = Psi(0) = 0)
            let psi: Vec<(Q, Q)> = (0..=len)
                .map(|m| law.psi(&(g * qi(m as i64)), &prep))
                .collect();
            let zero = (Q::zero(), Q::zero());
            for m in 0..len {
                let a = if m == 0 { &zero } else { &psi[m - 1] };
                let b = &psi[m];
                let c = &psi[m + 1];
                let l = (&c.0 - qi(2) * &b.1 + &a.0) / g;
                let h = (&c.1 - qi(2) * &b.0 + &a.1) / g;
                lo[m] = q_dn(&l).clamp(0.0, 1.0);
                hi[m] = q_up(&h).clamp(0.0, 1.0);
            }
        }
    }
    (lo, hi)
}

/// Partition for the mixed Poisson weights: nodes y_0 = 0 < y_1 < ... with
/// step h0 up to y0_end, then a geometric step of about 1 %, up to the cap.
#[derive(Clone, Debug)]
pub struct Partition {
    pub y: Vec<f64>,        // upper f64 bounds of the node values
    pub y_lo: Vec<f64>,     // lower f64 bounds of the node values
    pub w_lo: Vec<f64>,     // lower bounds of P(tau~ = y_l)
    pub w_hi: Vec<f64>,     // upper bounds
    pub cap: Q,             // last node; tau is replaced by min(tau, cap)
    pub tail_mean_hi: f64,  // upper bound of E(tau - cap)^+ (deleted mean)
    pub spread_var_hi: f64, // upper bound of E[Var(tau~ | tau)]
}

pub fn partition(law: &Law, h0: &Q, y0_end: &Q) -> Partition {
    let prep = law.prepare();
    let cap = law.cap().max(y0_end.clone());
    let mut nodes: Vec<Q> = vec![Q::zero()];
    let n0 = (y0_end / h0).ceil().to_integer().to_usize().unwrap();
    for l in 1..=n0 {
        nodes.push(h0 * qi(l as i64));
    }
    while *nodes.last().unwrap() < cap {
        let y = nodes.last().unwrap().clone();
        let step = dyadic(to_f64(&y) * 0.01, 30).max(h0.clone());
        nodes.push(&y + step);
    }
    let cap = nodes.last().unwrap().clone();
    let psi: Vec<(Q, Q)> = nodes.iter().map(|y| law.psi(y, &prep)).collect();
    let nn = nodes.len();
    // slopes on [y_l, y_{l+1}], then slope 1 beyond the cap
    let mut s_lo = Vec::with_capacity(nn);
    let mut s_hi = Vec::with_capacity(nn);
    for l in 0..nn - 1 {
        let dy = &nodes[l + 1] - &nodes[l];
        s_lo.push((&psi[l + 1].0 - &psi[l].1) / &dy);
        s_hi.push((&psi[l + 1].1 - &psi[l].0) / &dy);
    }
    s_lo.push(Q::one());
    s_hi.push(Q::one());
    let mut w_lo = Vec::with_capacity(nn);
    let mut w_hi = Vec::with_capacity(nn);
    for l in 0..nn {
        let (pl, ph) = if l == 0 {
            (Q::zero(), Q::zero())
        } else {
            (s_lo[l - 1].clone(), s_hi[l - 1].clone())
        };
        w_lo.push(q_dn(&(&s_lo[l] - &ph)).clamp(0.0, 1.0));
        w_hi.push(q_up(&(&s_hi[l] - &pl)).clamp(0.0, 1.0));
    }
    let tm = law.tail_mean(&cap, &prep);
    // E[Var(tau~|tau)] <= h0^2/4 + E[(0.0102 tau)^2]/4 on the geometric part
    let m2 = law.second_moment_up().min(q_up(&(&cap * &law.mean().1)));
    let h0f = q_up(h0);
    let var = h0f * h0f / 4.0 + 0.0102f64.powi(2) / 4.0 * m2;
    Partition {
        y: nodes.iter().map(q_up).collect(),
        y_lo: nodes.iter().map(q_dn).collect(),
        w_lo,
        w_hi,
        cap,
        tail_mean_hi: q_up(&tm.1),
        spread_var_hi: var * (1.0 + 1e-9),
    }
}
