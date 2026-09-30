//! Exact rationals (`Q`) with parsing and decimal output, directed rounding
//! to dyadic rationals, and rigorous enclosures of `exp(-x)`: the base
//! arithmetic of the enclosures and certificates of the checker.

use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use serde_json::Value;

/// Exact rational numbers.
pub type Q = BigRational;

/// The integer `n` as a rational.
pub fn qi(n: i64) -> Q {
    Q::from_integer(BigInt::from(n))
}

/// The fraction `n / d`.
pub fn qr(n: i64, d: i64) -> Q {
    Q::new(BigInt::from(n), BigInt::from(d))
}

/// A double approximation of `x` (`NaN` when the conversion fails).
pub fn to_f64(x: &Q) -> f64 {
    x.to_f64().unwrap_or(f64::NAN)
}

/// Parse an integer, a fraction `a/b` or a decimal `12.345` exactly.
pub fn parse_q(s: &str) -> Result<Q, String> {
    let s = s.trim();
    if let Some((a, b)) = s.split_once('/') {
        let b = parse_q(b)?;
        if b.is_zero() {
            return Err(format!("division by zero in {s}"));
        }
        return Ok(parse_q(a)? / b);
    }
    let (negative, body) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let (int_part, frac_part) = body.split_once('.').unwrap_or((body, ""));
    let digits = format!("{int_part}{frac_part}");
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return Err(format!("not a rational: {s}"));
    }
    let numer: BigInt = digits.parse().map_err(|_| format!("not a rational: {s}"))?;
    let denom = BigInt::from(10u32).pow(frac_part.len() as u32);
    let value = Q::new(numer, denom);
    Ok(if negative { -value } else { value })
}

/// A JSON string (`"9/20"`, `"0.45"`) or number as an exact rational. A JSON
/// number is taken as the exact binary value of the double, so decimal
/// parameters should be written as strings.
pub fn q_from_json(v: &Value) -> Result<Q, String> {
    match v {
        Value::String(s) => parse_q(s),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(qi(i))
            } else {
                Q::from_float(n.as_f64().ok_or("bad number")?)
                    .ok_or_else(|| "non-finite number".to_string())
            }
        }
        other => Err(format!("expected a rational, got {other}")),
    }
}

/// Decimal string with `digits` places, rounded toward minus infinity
/// (`up = false`) or plus infinity (`up = true`).
pub fn dec(x: &Q, digits: u32, up: bool) -> String {
    let scale = BigInt::from(10u32).pow(digits);
    let scaled = x * Q::from_integer(scale);
    let n = if up {
        scaled.numer().div_ceil(scaled.denom())
    } else {
        scaled.numer().div_floor(scaled.denom())
    };
    let negative = n.is_negative();
    let s = n.abs().to_string();
    let (ip, fp) = if s.len() > digits as usize {
        s.split_at(s.len() - digits as usize)
    } else {
        ("", s.as_str())
    };
    let ip = if ip.is_empty() { "0" } else { ip };
    let fp = format!("{:0>width$}", fp, width = digits as usize);
    if digits == 0 {
        format!("{}{}", if negative { "-" } else { "" }, ip)
    } else {
        format!("{}{}.{}", if negative { "-" } else { "" }, ip, fp)
    }
}

/// The largest dyadic rational with `bits` fractional bits not above `x`.
pub fn round_down(x: &Q, bits: u32) -> Q {
    let scale: BigInt = BigInt::one() << bits;
    Q::new((x.numer() * &scale).div_floor(x.denom()), scale)
}

/// The smallest dyadic rational with `bits` fractional bits not below `x`.
pub fn round_up(x: &Q, bits: u32) -> Q {
    let scale: BigInt = BigInt::one() << bits;
    Q::new((x.numer() * &scale).div_ceil(x.denom()), scale)
}

/// `lo <= exp(-x) <= hi` for rational `x >= 0`, both dyadic with `bits`
/// fractional bits. The argument is split as `x = m + f` with integer `m`
/// and `0 <= f < 1`; `exp(-f)` and `exp(-1)` come from the alternating
/// Taylor series with the first omitted term as the remainder bound, and
/// the power `exp(-1)^m` from repeated multiplication with directed
/// rounding at `bits + 24` working bits.
pub fn exp_neg(x: &Q, bits: u32) -> (Q, Q) {
    assert!(!x.is_negative(), "exp_neg needs x >= 0");
    let work = bits + 24;
    let m: BigInt = x.numer().div_floor(x.denom());
    let f = x - Q::from_integer(m.clone());
    let (flo, fhi) = exp_neg_unit(&f, work);
    let m = m.to_u64().expect("exponent too large for exp_neg");
    let (mut plo, mut phi) = (Q::one(), Q::one());
    if m > 0 {
        let (elo, ehi) = exp_neg_unit(&Q::one(), work);
        for _ in 0..m {
            plo = round_down(&(plo * &elo), work);
            phi = round_up(&(phi * &ehi), work);
        }
    }
    (round_down(&(plo * flo), bits), round_up(&(phi * fhi), bits))
}

/// Alternating Taylor series of `exp(-f)` for `0 <= f <= 1`: the terms are
/// nonincreasing in magnitude, so the remainder after the last kept term is
/// at most the first omitted term in magnitude, and the partial sum plus or
/// minus that term encloses `exp(-f)`.
fn exp_neg_unit(f: &Q, bits: u32) -> (Q, Q) {
    assert!(!f.is_negative() && *f <= Q::one());
    let tol = Q::new(BigInt::one(), BigInt::one() << (bits + 4));
    let mut sum = Q::zero();
    let mut term = Q::one();
    let mut k: i64 = 0;
    loop {
        sum += &term;
        k += 1;
        term = -(term * f) / qi(k);
        if term.abs() < tol || k > 2000 {
            break;
        }
    }
    let r = term.abs();
    (round_down(&(&sum - &r), bits), round_up(&(&sum + &r), bits))
}

/// Enclosures of `exp(-n x)` for `n = 0..=nmax`: `lo[n] <= exp(-n x) <= hi[n]`,
/// dyadic with `bits + 16` fractional bits, from the enclosure of `exp(-x)`
/// by repeated multiplication with directed rounding (all factors are
/// nonnegative, so the products of the bounds enclose the product).
pub fn exp_neg_powers(x: &Q, nmax: usize, bits: u32) -> (Vec<Q>, Vec<Q>) {
    let (lo1, hi1) = exp_neg(x, bits + 16);
    let mut lo = Vec::with_capacity(nmax + 1);
    let mut hi = Vec::with_capacity(nmax + 1);
    lo.push(Q::one());
    hi.push(Q::one());
    for n in 1..=nmax {
        lo.push(round_down(&(&lo[n - 1] * &lo1), bits + 16));
        hi.push(round_up(&(&hi[n - 1] * &hi1), bits + 16));
    }
    (lo, hi)
}

/// Greatest common divisor of rationals: the largest rational of which every
/// entry is an integer multiple. Zero entries are ignored; the result is zero
/// when every entry is zero.
pub fn rational_gcd(values: &[Q]) -> Q {
    let mut g = Q::zero();
    for v in values {
        let v = v.abs();
        if v.is_zero() {
            continue;
        }
        if g.is_zero() {
            g = v;
            continue;
        }
        // gcd(a/b, c/d) = gcd(a d, c b) / (b d)
        let num = (g.numer() * v.denom()).gcd(&(v.numer() * g.denom()));
        let den = g.denom() * v.denom();
        g = Q::new(num, den);
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_decimals_and_fractions_exactly() {
        assert_eq!(parse_q("0.45").unwrap(), qr(9, 20));
        assert_eq!(parse_q("64/65").unwrap(), qr(64, 65));
        assert_eq!(parse_q("-3.7").unwrap(), qr(-37, 10));
        assert_eq!(parse_q("12").unwrap(), qi(12));
        assert!(parse_q("abc").is_err());
    }

    #[test]
    fn rounding_is_directed() {
        let x = qr(1, 3);
        let lo = round_down(&x, 10);
        let hi = round_up(&x, 10);
        assert!(lo <= x && x <= hi);
        assert!(&hi - &lo <= qr(1, 1024));
    }

    #[test]
    fn exp_neg_encloses_the_double_value() {
        for x in ["0", "0.25", "1", "1.8", "2", "10.2", "12", "37.5", "96"] {
            let x = parse_q(x).unwrap();
            let (lo, hi) = exp_neg(&x, 128);
            let value = (-to_f64(&x)).exp();
            assert!(
                to_f64(&lo) <= value * (1.0 + 1e-14) && value * (1.0 - 1e-14) <= to_f64(&hi),
                "{x}"
            );
            let width = &hi - &lo;
            assert!(
                width < Q::new(BigInt::one(), BigInt::one() << 100),
                "{x}: width {width}"
            );
        }
    }

    #[test]
    fn powers_stay_ordered() {
        let (lo, hi) = exp_neg_powers(&parse_q("0.004").unwrap(), 3000, 128);
        for n in [1usize, 10, 100, 1000, 3000] {
            let value = (-0.004 * n as f64).exp();
            assert!(
                to_f64(&lo[n]) <= value * (1.0 + 1e-13) && value * (1.0 - 1e-13) <= to_f64(&hi[n])
            );
        }
    }

    #[test]
    fn rational_gcd_of_support_points() {
        let g = rational_gcd(&[qr(9, 20), qr(37, 10), qr(1, 10), qr(9, 10), qr(1, 2), qi(3)]);
        assert_eq!(g, qr(1, 20));
    }

    #[test]
    fn decimal_truncation_directions() {
        let x = qr(2, 3);
        assert_eq!(dec(&x, 4, false), "0.6666");
        assert_eq!(dec(&x, 4, true), "0.6667");
        assert_eq!(dec(&qr(-2, 3), 2, false), "-0.67");
    }
}
