//! Rigorous rational enclosures of the special functions that the continuous
//! reconfiguration-time distributions need: `ln`, square and `q`-th roots,
//! `pi`, `exp`, the standard normal distribution function and density, the
//! gamma function and the regularized lower incomplete gamma function. Each
//! returns `(lo, hi)` with `lo <= f <= hi`, dyadic with `bits` fractional
//! bits. Intermediate results carry `GUARD` extra bits, truncated series carry
//! explicit remainder bounds, partial sums are rounded toward their bound, and
//! a monotone function of an enclosed or rounded argument is evaluated at both ends.

use super::rational::{Q, exp_neg, qi, qr, round_down, round_up};
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Extra working bits of intermediate results.
const GUARD: u32 = 32;

/// Constants (`ln 2`, `pi`, `sqrt(2 pi)`) enclosed once per precision.
fn constant_cache() -> &'static Mutex<HashMap<(&'static str, u32), (Q, Q)>> {
    static CACHE: OnceLock<Mutex<HashMap<(&'static str, u32), (Q, Q)>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The cached enclosure of the constant `name` at `bits`, computed on first use.
fn memo(name: &'static str, bits: u32, compute: impl FnOnce() -> (Q, Q)) -> (Q, Q) {
    if let Some(v) = constant_cache().lock().unwrap().get(&(name, bits)) {
        return v.clone();
    }
    let v = compute();
    constant_cache()
        .lock()
        .unwrap()
        .insert((name, bits), v.clone());
    v
}

/// `2^-bits`.
fn tol(bits: u32) -> Q {
    Q::new(BigInt::one(), BigInt::one() << bits)
}

/// `(lo, hi)` rounded outward to `bits` fractional bits.
fn rounded(lo: Q, hi: Q, bits: u32) -> (Q, Q) {
    debug_assert!(lo <= hi);
    (round_down(&lo, bits), round_up(&hi, bits))
}

/// Interval addition with directed rounding at `bits`.
pub fn iadd(a: &(Q, Q), b: &(Q, Q), bits: u32) -> (Q, Q) {
    rounded(&a.0 + &b.0, &a.1 + &b.1, bits)
}

/// Interval difference with directed rounding at `bits`.
pub fn isub(a: &(Q, Q), b: &(Q, Q), bits: u32) -> (Q, Q) {
    rounded(&a.0 - &b.1, &a.1 - &b.0, bits)
}

/// Interval product with directed rounding (any signs).
pub fn imul(a: &(Q, Q), b: &(Q, Q), bits: u32) -> (Q, Q) {
    let c = [&a.0 * &b.0, &a.0 * &b.1, &a.1 * &b.0, &a.1 * &b.1];
    let lo = c.iter().min().unwrap().clone();
    let hi = c.iter().max().unwrap().clone();
    rounded(lo, hi, bits)
}

/// Interval reciprocal for an interval that does not contain zero.
pub fn irecip(a: &(Q, Q), bits: u32) -> (Q, Q) {
    assert!(
        a.0.is_positive() || a.1.is_negative(),
        "reciprocal of an interval containing zero"
    );
    rounded(a.1.recip(), a.0.recip(), bits)
}

/// Interval quotient for a divisor that does not contain zero.
pub fn idiv(a: &(Q, Q), b: &(Q, Q), bits: u32) -> (Q, Q) {
    imul(a, &irecip(b, bits + GUARD), bits)
}

/// Interval negation (exact).
pub fn ineg(a: &(Q, Q)) -> (Q, Q) {
    (-a.1.clone(), -a.0.clone())
}

/// The degenerate interval `[x, x]`.
pub fn point(x: &Q) -> (Q, Q) {
    (x.clone(), x.clone())
}

/// `lo <= sqrt(a) <= hi` for rational `a >= 0`.
pub fn sqrt_enclosure(a: &Q, bits: u32) -> (Q, Q) {
    assert!(!a.is_negative(), "sqrt of a negative rational");
    let scale: BigInt = BigInt::one() << (2 * bits);
    let n_lo: BigInt = (a.numer() * &scale).div_floor(a.denom());
    let n_hi: BigInt = (a.numer() * &scale).div_ceil(a.denom());
    let s_lo = n_lo.sqrt();
    let s_hi = n_hi.sqrt() + BigInt::one();
    let den: BigInt = BigInt::one() << bits;
    (Q::new(s_lo, den.clone()), Q::new(s_hi, den))
}

/// `lo <= a^(1/q) <= hi` for rational `a >= 0` and integer `q >= 1`.
pub fn root_enclosure(a: &Q, q: u32, bits: u32) -> (Q, Q) {
    assert!(!a.is_negative() && q >= 1);
    if q == 1 {
        return rounded(a.clone(), a.clone(), bits);
    }
    let scale: BigInt = BigInt::one() << (q * bits);
    let n_lo: BigInt = (a.numer() * &scale).div_floor(a.denom());
    let n_hi: BigInt = (a.numer() * &scale).div_ceil(a.denom());
    let s_lo = n_lo.nth_root(q);
    let s_hi = n_hi.nth_root(q) + BigInt::one();
    let den: BigInt = BigInt::one() << bits;
    (Q::new(s_lo, den.clone()), Q::new(s_hi, den))
}

/// `lo <= a^(p/q) <= hi` for rational `a >= 0` and rational exponent
/// `p/q >= 0`: the power `a^p` exactly, then its `q`-th root.
pub fn pow_rational_enclosure(a: &Q, exponent: &Q, bits: u32) -> (Q, Q) {
    assert!(!a.is_negative() && !exponent.is_negative());
    let p = exponent.numer().to_u32().expect("exponent numerator");
    let q = exponent.denom().to_u32().expect("exponent denominator");
    let ap = num_traits::pow::pow(a.clone(), p as usize);
    root_enclosure(&ap, q, bits)
}

/// `2 atanh(y) = ln((1 + y) / (1 - y)) = 2 sum_{k>=0} y^(2k+1) / (2k+1)`
/// for rational `0 <= y < 1`; the remainder after the last kept term is at
/// most the next term divided by `1 - y^2`. Powers and terms are integers
/// scaled by `2^w` (`w = bits + GUARD + 64`), rounded in the direction of
/// the bound they belong to.
fn two_atanh(y: &Q, bits: u32) -> (Q, Q) {
    assert!(!y.is_negative() && *y < Q::one());
    let work = bits + GUARD;
    let w = work + 64;
    let scale: BigInt = BigInt::one() << w;
    let y_lo_fp: BigInt = (y.numer() * &scale).div_floor(y.denom());
    let y_hi_fp: BigInt = (y.numer() * &scale).div_ceil(y.denom());
    let two_w: BigInt = BigInt::one() << (2 * w);
    let y2_lo: BigInt = &y_lo_fp * &y_lo_fp; // scaled by 2^(2w)
    let y2_hi: BigInt = &y_hi_fp * &y_hi_fp;
    assert!(y2_hi < two_w, "two_atanh argument too close to one");
    let tolerance: BigInt = BigInt::one() << 64; // 2^-work in units of 2^-w
    let (mut power_lo, mut power_hi) = (y_lo_fp, y_hi_fp); // y^(2k+1)
    let (mut lo, mut hi) = (BigInt::zero(), BigInt::zero());
    let mut k: u64 = 0;
    loop {
        lo += power_lo.div_floor(&BigInt::from(2 * k + 1));
        hi += power_hi.div_ceil(&BigInt::from(2 * k + 1));
        power_lo = (&power_lo * &y2_lo).div_floor(&two_w);
        power_hi = (&power_hi * &y2_hi).div_ceil(&two_w);
        k += 1;
        let next_hi = power_hi.div_ceil(&BigInt::from(2 * k + 1));
        if next_hi < tolerance {
            // sum_{j >= k} y^(2j+1) / (2j+1) <= next / (1 - y^2)
            let remainder = (&next_hi * &two_w).div_ceil(&(&two_w - &y2_hi));
            return rounded(
                Q::new(BigInt::from(2u64) * lo, scale.clone()),
                Q::new(BigInt::from(2u64) * (hi + remainder), scale),
                bits,
            );
        }
        if k > 100_000 {
            panic!("two_atanh did not converge");
        }
    }
}

/// `ln 2 = 2 atanh(1/3)`.
pub fn ln2_enclosure(bits: u32) -> (Q, Q) {
    memo("ln2", bits, || two_atanh(&qr(1, 3), bits))
}

/// `lo <= ln x <= hi` for rational `x > 0`. The argument is reduced to
/// `x = 2^e r` with `r` near one; `ln r = 2 atanh((r - 1) / (r + 1))`.
pub fn ln_enclosure(x: &Q, bits: u32) -> (Q, Q) {
    assert!(x.is_positive(), "ln of a nonpositive rational");
    let work = bits + GUARD;
    let e: i64 = {
        let f = x.to_f64().unwrap_or(1.0);
        if f.is_finite() && f > 0.0 {
            f.log2().round() as i64
        } else {
            0
        }
    };
    let r = if e >= 0 {
        x / Q::new(BigInt::one() << (e as u32), BigInt::one())
    } else {
        x * Q::new(BigInt::one() << ((-e) as u32), BigInt::one())
    };
    let y = (&r - Q::one()) / (&r + Q::one());
    // ln r is increasing in y; evaluate the series at both rounded ends of y.
    let series = |y: &Q| -> (Q, Q) {
        if y.is_negative() {
            let (lo, hi) = two_atanh(&(-y.clone()), work);
            (-hi, -lo)
        } else {
            two_atanh(y, work)
        }
    };
    let y_lo = round_down(&y, work);
    let y_hi = round_up(&y, work);
    let a = series(&y_lo);
    let b = series(&y_hi);
    let ln_r = (a.0.min(b.0), a.1.max(b.1));
    let ln2 = ln2_enclosure(work);
    let scaled = if e >= 0 {
        imul(&ln2, &point(&qi(e)), work)
    } else {
        ineg(&imul(&ln2, &point(&qi(-e)), work))
    };
    iadd(&ln_r, &scaled, bits)
}

/// `arctan(1/n)` by its alternating series (remainder at most the next term).
fn arctan_inverse(n: i64, bits: u32) -> (Q, Q) {
    let work = bits + GUARD;
    let t = tol(work);
    let inv = qr(1, n);
    let inv2 = &inv * &inv;
    let mut power = inv.clone();
    let (mut lo, mut hi) = (Q::zero(), Q::zero());
    let mut k: i64 = 0;
    loop {
        let term = &power / qi(2 * k + 1);
        let signed = if k % 2 == 0 {
            term.clone()
        } else {
            -term.clone()
        };
        lo = round_down(&(&lo + &signed), work);
        hi = round_up(&(&hi + &signed), work);
        power *= &inv2;
        k += 1;
        let next = &power / qi(2 * k + 1);
        if next < t {
            return rounded(lo - &next, hi + &next, bits);
        }
    }
}

/// `pi = 16 arctan(1/5) - 4 arctan(1/239)` (Machin's formula).
pub fn pi_enclosure(bits: u32) -> (Q, Q) {
    memo("pi", bits, || {
        let work = bits + GUARD;
        let a = arctan_inverse(5, work);
        let b = arctan_inverse(239, work);
        let sixteen_a = imul(&a, &point(&qi(16)), work);
        let four_b = imul(&b, &point(&qi(4)), work);
        isub(&sixteen_a, &four_b, bits)
    })
}

/// `exp(x)` for rational `x` of either sign. For `x > 0`, `x = m + f` with
/// integer `m` gives `exp(x) = exp(f) e^m`, both factors from reciprocals of
/// enclosures of `exp(-f)` and `exp(-1)`.
pub fn exp_enclosure(x: &Q, bits: u32) -> (Q, Q) {
    if !x.is_positive() {
        return exp_neg(&(-x.clone()), bits);
    }
    let work = bits + GUARD;
    let m: BigInt = x.floor().to_integer();
    let f = x - Q::from_integer(m.clone());
    let m = m.to_u64().expect("exp argument too large");
    let ef = irecip(&exp_neg(&f, work), work);
    let e1 = irecip(&exp_neg(&Q::one(), work), work);
    let mut acc = ef;
    for _ in 0..m {
        acc = imul(&acc, &e1, work);
    }
    rounded(acc.0, acc.1, bits)
}

/// `sqrt(2 pi)`.
pub fn sqrt_two_pi(bits: u32) -> (Q, Q) {
    memo("sqrt_two_pi", bits, || {
        let work = bits + GUARD;
        let pi = pi_enclosure(work);
        let two_pi = imul(&pi, &point(&qi(2)), work);
        let lo = sqrt_enclosure(&two_pi.0, work).0;
        let hi = sqrt_enclosure(&two_pi.1, work).1;
        rounded(lo, hi, bits)
    })
}

/// `int_0^x exp(-t^2/2) dt` for dyadic `x >= 0` by the alternating series
/// `sum_n (-1)^n x^(2n+1) / (n! 2^n (2n+1))`; the terms decrease in
/// magnitude once `n >= x^2 / 2`, from where the remainder is bounded by the
/// first omitted term. Partial sums, powers and terms are integers scaled by
/// `2^w` with `w = bits + GUARD + 64`; every power and term is rounded in
/// the direction of the bound it belongs to (no rational normalization).
fn gauss_integral(x: &Q, bits: u32) -> (Q, Q) {
    assert!(!x.is_negative());
    let work = bits + GUARD;
    let w = work + 64;
    let scale: BigInt = BigInt::one() << w;
    let x_num = x.numer() * &scale;
    assert!(
        x_num.is_multiple_of(x.denom()),
        "gauss_integral needs a dyadic argument with at most {w} fractional bits"
    );
    let x_fp: BigInt = x_num / x.denom(); // x, scaled by 2^w (exact)
    let x2_fp: BigInt = &x_fp * &x_fp; // x^2, scaled by 2^(2w) (exact)
    let two_w: BigInt = BigInt::one() << (2 * w);
    let decreasing_from = x2_fp
        .div_ceil(&(&two_w * BigInt::from(2u64)))
        .to_usize()
        .expect("gauss_integral argument too large")
        + 1;
    let tolerance: BigInt = BigInt::one() << 64; // 2^-work in units of 2^-w
    let (mut power_lo, mut power_hi) = (x_fp.clone(), x_fp); // x^(2n+1), scaled by 2^w
    let mut nfact_2n: BigInt = BigInt::one(); // n! 2^n
    let (mut lo, mut hi) = (BigInt::zero(), BigInt::zero());
    let mut n: usize = 0;
    loop {
        let den = &nfact_2n * BigInt::from(2 * n as u64 + 1);
        let term_lo = power_lo.div_floor(&den);
        let term_hi = power_hi.div_ceil(&den);
        if n.is_multiple_of(2) {
            lo += &term_lo;
            hi += &term_hi;
        } else {
            lo -= &term_hi;
            hi -= &term_lo;
        }
        power_lo = (&power_lo * &x2_fp).div_floor(&two_w);
        power_hi = (&power_hi * &x2_fp).div_ceil(&two_w);
        n += 1;
        nfact_2n *= BigInt::from(2 * n as u64);
        let next_hi = power_hi.div_ceil(&(&nfact_2n * BigInt::from(2 * n as u64 + 1)));
        if n >= decreasing_from && next_hi < tolerance {
            return rounded(
                Q::new(lo - &next_hi, scale.clone()),
                Q::new(hi + &next_hi, scale),
                bits,
            );
        }
        if n > 4000 {
            panic!("gauss_integral did not converge");
        }
    }
}

/// `1/sqrt(2 pi) exp(-x^2/2)`, the standard normal density, for rational `x`.
pub fn norm_pdf_enclosure(x: &Q, bits: u32) -> (Q, Q) {
    let work = bits + GUARD;
    let e = exp_neg(&(x * x / qi(2)), work);
    let c = irecip(&sqrt_two_pi(work), work);
    imul(&e, &c, bits)
}

/// The standard normal distribution function `Phi(x)` for rational `x`:
/// `Phi(x) = 1/2 + I(x) / sqrt(2 pi)` with `I` from `gauss_integral`, and
/// for `|x| >= 8` the tail bound `1 - Phi(x) <= phi(x) / x`.
pub fn norm_cdf_enclosure(x: &Q, bits: u32) -> (Q, Q) {
    let work = bits + GUARD;
    if x.is_negative() {
        let (lo, hi) = norm_cdf_enclosure(&(-x.clone()), work);
        return rounded(Q::one() - hi, Q::one() - lo, bits);
    }
    if *x >= qi(8) {
        let pdf = norm_pdf_enclosure(x, work);
        let tail_hi = round_up(&(&pdf.1 / x), work);
        return rounded(Q::one() - tail_hi, Q::one(), bits);
    }
    // Phi is increasing: the lower bound at the rounded-down x; the integrand
    // is at most one, so I(x_hi) <= I(x_lo) + (x_hi - x_lo).
    let xb = work / 2;
    let x_lo = round_down(x, xb);
    let x_hi = round_up(x, xb);
    let i = gauss_integral(&x_lo, work);
    let i_lo = i.0;
    let i_hi = round_up(&(i.1 + (&x_hi - &x_lo)), work);
    let c = irecip(&sqrt_two_pi(work), work);
    let scaled = imul(&(i_lo, i_hi), &c, work);
    rounded(qr(1, 2) + scaled.0, qr(1, 2) + scaled.1, bits)
}

/// Bernoulli numbers `B_2 .. B_24`.
fn bernoulli() -> Vec<Q> {
    vec![
        qr(1, 6),
        qr(-1, 30),
        qr(1, 42),
        qr(-1, 30),
        qr(5, 66),
        qr(-691, 2730),
        qr(7, 6),
        qr(-3617, 510),
        qr(43867, 798),
        qr(-174611, 330),
        qr(854513, 138),
        qr(-236364091, 2730),
    ]
}

/// `ln Gamma(z)` for rational `z >= 100` by the Stirling series with the
/// terms up to `B_22`; the first omitted term (`B_24`) bounds the remainder.
fn ln_gamma_large(z: &Q, bits: u32) -> (Q, Q) {
    assert!(*z >= qi(100));
    let work = bits + GUARD;
    let ln_z = ln_enclosure(z, work);
    let ln_two_pi = {
        let pi = pi_enclosure(work);
        let a = ln_enclosure(&(qi(2) * &pi.0), work).0;
        let b = ln_enclosure(&(qi(2) * &pi.1), work).1;
        (a, b)
    };
    // (z - 1/2) ln z - z + ln(2 pi) / 2
    let mut acc = imul(&point(&(z - qr(1, 2))), &ln_z, work);
    acc = isub(&acc, &point(z), work);
    acc = iadd(&acc, &imul(&ln_two_pi, &point(&qr(1, 2)), work), work);
    let b = bernoulli();
    let mut zpow = z.clone(); // z^(2k-1)
    let z2 = z * z;
    let terms_used = b.len() - 1;
    for (idx, bk) in b.iter().enumerate().take(terms_used) {
        let k = (idx + 1) as i64;
        let term = bk / (qi(2 * k) * qi(2 * k - 1) * &zpow);
        acc = iadd(&acc, &point(&term), work);
        zpow *= &z2;
    }
    let k = (terms_used + 1) as i64;
    let remainder = round_up(
        &(b[terms_used].abs() / (qi(2 * k) * qi(2 * k - 1) * &zpow)),
        work,
    );
    rounded(acc.0 - &remainder, acc.1 + &remainder, bits)
}

/// `ln Gamma(a)` for rational `a > 0`: shift to `z = a + n >= 100`, then
/// `ln Gamma(a) = ln Gamma(z) - ln prod_{j<n} (a + j)`.
pub fn ln_gamma_enclosure(a: &Q, bits: u32) -> (Q, Q) {
    assert!(a.is_positive(), "ln Gamma needs a positive argument");
    let work = bits + GUARD;
    let shift = if *a >= qi(100) {
        0
    } else {
        (qi(100) - a).ceil().to_integer().to_u64().unwrap()
    };
    let z = a + qi(shift as i64);
    let mut acc = ln_gamma_large(&z, work);
    let mut prod = Q::one();
    for j in 0..shift {
        prod *= a + qi(j as i64);
    }
    if shift > 0 {
        let ln_prod = ln_enclosure(&prod, work);
        acc = isub(&acc, &ln_prod, work);
    }
    rounded(acc.0, acc.1, bits)
}

/// `Gamma(a) = exp(ln Gamma(a))` for rational `a > 0`.
pub fn gamma_enclosure(a: &Q, bits: u32) -> (Q, Q) {
    let work = bits + GUARD;
    let lg = ln_gamma_enclosure(a, work);
    let lo = exp_enclosure(&lg.0, work).0;
    let hi = exp_enclosure(&lg.1, work).1;
    rounded(lo, hi, bits)
}

/// The regularized lower incomplete gamma function `P(a, x)` for rational
/// `a > 0` and `x >= 0`, from the series of positive terms
/// `P(a, x) = x^a exp(-x) / Gamma(a) sum_{n>=0} c_n` with
/// `c_n = x^n / (a (a+1) ... (a+n))`. Summation stops at the first omitted
/// term `c_n` once `c_n < 2^-(bits + GUARD)` and
/// `c_{n+1} / c_n = x / (a + n + 1) <= 1/2`. Every later ratio is smaller,
/// so the omitted terms sum to at most `2 c_n`, which the upper bound adds;
/// the terms are rounded down for the lower bound and up for the upper one.
pub fn gamma_lower_regularized_enclosure(a: &Q, x: &Q, bits: u32) -> (Q, Q) {
    assert!(a.is_positive() && !x.is_negative());
    if x.is_zero() {
        return (Q::zero(), Q::zero());
    }
    let work = bits + GUARD;
    let t = tol(work);
    let xb = work / 2;
    // P is increasing in x: evaluate at the rounded ends.
    let series = |x: &Q, upper: bool| -> Q {
        let mut coef = Q::one() / a; // 1/a
        let mut sum = Q::zero();
        let mut n: i64 = 0;
        loop {
            sum = if upper {
                round_up(&(&sum + &coef), work)
            } else {
                round_down(&(&sum + &coef), work)
            };
            n += 1;
            let next = &coef * x / (a + qi(n));
            coef = if upper {
                round_up(&next, work + 64)
            } else {
                round_down(&next, work + 64)
            };
            let ratio = x / (a + qi(n + 1));
            if ratio <= qr(1, 2) && coef < t {
                return if upper { sum + qi(2) * coef } else { sum };
            }
            if n > 100_000 {
                panic!("incomplete gamma series did not converge");
            }
        }
    };
    let x_lo = round_down(x, xb);
    let x_hi = round_up(x, xb);
    let s_lo = series(&x_lo, false);
    let s_hi = series(&x_hi, true);
    let xa_lo = pow_rational_enclosure(&x_lo, a, work).0;
    let xa_hi = pow_rational_enclosure(&x_hi, a, work).1;
    let e_lo = exp_neg(&x_hi, work).0;
    let e_hi = exp_neg(&x_lo, work).1;
    let g = gamma_enclosure(a, work);
    let lo = round_down(&(&xa_lo * &e_lo * &s_lo / &g.1), work);
    let hi = round_up(&(&xa_hi * &e_hi * &s_hi / &g.0), work);
    rounded(lo.max(Q::zero()), hi.min(Q::one()), bits)
}

#[cfg(test)]
mod tests {
    use super::super::rational::{parse_q, to_f64};
    use super::*;

    fn width(e: &(Q, Q)) -> f64 {
        (&e.1 - &e.0).to_f64().unwrap_or(f64::NAN)
    }

    fn contains(e: &(Q, Q), v: f64) -> bool {
        to_f64(&e.0) <= v + 1e-15 && v - 1e-15 <= to_f64(&e.1)
    }

    #[test]
    fn roots_and_logarithms_enclose_the_double_values() {
        let s = sqrt_enclosure(&qi(2), 128);
        assert!(contains(&s, 2f64.sqrt()) && width(&s) < 1e-30);
        let r = root_enclosure(&qi(81), 4, 128);
        assert!(contains(&r, 3.0) && width(&r) < 1e-30);
        let p = pow_rational_enclosure(&qr(1, 3), &qr(3, 2), 128);
        assert!(contains(&p, (1.0f64 / 3.0).powf(1.5)) && width(&p) < 1e-30);
        for x in ["2", "1/2", "1000", "1/1000", "3/7", "1"] {
            let q = parse_q(x).unwrap();
            let e = ln_enclosure(&q, 128);
            assert!(contains(&e, to_f64(&q).ln()), "ln {x}");
            assert!(width(&e) < 1e-30, "ln width {x}");
        }
        let two = ln2_enclosure(128);
        assert!(contains(&two, std::f64::consts::LN_2));
    }

    #[test]
    fn pi_exp_and_the_normal_law_enclose_the_double_values() {
        let pi = pi_enclosure(128);
        assert!(contains(&pi, std::f64::consts::PI) && width(&pi) < 1e-30);
        for x in ["0", "1", "-1", "5/2", "-7/3", "20"] {
            let q = parse_q(x).unwrap();
            let e = exp_enclosure(&q, 128);
            assert!(contains(&e, to_f64(&q).exp()), "exp {x}");
        }
        let expected = [
            ("0", 0.5),
            ("1", 0.841344746068543),
            ("-1", 0.158655253931457),
            ("2", 0.977249868051821),
            ("-3", 0.001349898031630),
            ("5/2", 0.993790334674224),
            ("9", 1.0 - 1.128588405e-19),
        ];
        for (x, phi) in expected {
            let q = parse_q(x).unwrap();
            let e = norm_cdf_enclosure(&q, 128);
            assert!(
                contains(&e, phi),
                "Phi({x}) = {:?}",
                (to_f64(&e.0), to_f64(&e.1))
            );
            assert!(width(&e) < 1e-15, "Phi width {x} = {}", width(&e));
        }
        let d = norm_pdf_enclosure(&Q::zero(), 128);
        assert!(contains(&d, 0.398942280401433) && width(&d) < 1e-30);
    }

    #[test]
    fn gamma_functions_enclose_known_values() {
        let g4 = gamma_enclosure(&qi(4), 128);
        assert!(
            contains(&g4, 6.0) && width(&g4) < 1e-28,
            "Gamma(4) {:?}",
            (to_f64(&g4.0), to_f64(&g4.1))
        );
        let gh = gamma_enclosure(&qr(1, 2), 128);
        assert!(contains(&gh, std::f64::consts::PI.sqrt()) && width(&gh) < 1e-28);
        let gq = gamma_enclosure(&qr(1, 4), 128);
        assert!(contains(&gq, 3.625609908221908) && width(&gq) < 1e-28);
        // Erlang-3 with rate 6: P(3, 6t) = 1 - exp(-6t)(1 + 6t + 18t^2).
        for t in ["1/10", "1/2", "1", "3", "12"] {
            let tq = parse_q(t).unwrap();
            let x = qi(6) * &tq;
            let p = gamma_lower_regularized_enclosure(&qi(3), &x, 128);
            let xf = to_f64(&x);
            let closed = 1.0 - (-xf).exp() * (1.0 + xf + xf * xf / 2.0);
            assert!(
                contains(&p, closed),
                "P(3,{x}) = {:?} vs {closed}",
                (to_f64(&p.0), to_f64(&p.1))
            );
            assert!(width(&p) < 1e-15, "P width at {t}: {}", width(&p));
        }
        // non-integer shapes: against the double-precision regularized incomplete gamma of statrs
        for (a, x) in [
            (qr(1, 4), qi(1)),
            (qr(1, 4), qr(1, 10)),
            (qr(1, 4), qi(20)),
            (qr(7, 3), qr(5, 2)),
        ] {
            let p = gamma_lower_regularized_enclosure(&a, &x, 128);
            let reference = statrs::function::gamma::gamma_lr(to_f64(&a), to_f64(&x));
            assert!(
                to_f64(&p.0) - 1e-12 <= reference && reference <= to_f64(&p.1) + 1e-12,
                "P({a},{x}) = {:?} vs {reference}",
                (to_f64(&p.0), to_f64(&p.1))
            );
            assert!(width(&p) < 1e-15, "P width {}", width(&p));
        }
    }
}
