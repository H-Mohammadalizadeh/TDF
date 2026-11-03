//! Directed rounding in binary64 and the nonnegative kernels of the lattice
//! computation.
//!
//! Every kernel works on nonnegative numbers. A value computed in
//! round-to-nearest (RN) and then moved one ulp up (`next_up`) is an upper
//! bound of the exact result of that operation; one ulp down gives a lower
//! bound. A sum of `n` nonnegative products evaluated in RN in any order
//! satisfies `|fl(S) - S| <= gamma_n S`, `gamma_n = n u / (1 - n u)`,
//! `u = 2^-53`, provided no underflow occurs (Higham, Accuracy and Stability
//! of Numerical Algorithms, 2nd ed., 2002, Sec. 3.1). Hence
//! `S <= fl(S) (1 + 4 (n + 2) u)` and `S >= fl(S) (1 - 4 (n + 2) u)` for
//! `(n + 2) u < 1/8`.
//!
//! Underflow is excluded by construction. Every operand of a kernel product
//! is a stored value or a constant, and each term multiplies at most two
//! stored values and one constant.
//!
//! - A stored value is zero or at least `FLOOR = 2^-400`. The upper pass
//!   raises every positive result to at least `FLOOR` (an upper bound stays
//!   an upper bound), the lower pass flushes every result below `FLOOR` to
//!   zero (a lower bound stays a lower bound). Values that enter a kernel
//!   without such a result (lattice and partition weights, the first term of
//!   a recursion) pass through [`Dir::stored`], which applies the same rule.
//! - A constant is zero or at least `MIN_CONST = 2^-150`, and a divisor is at
//!   most `1 / MIN_CONST`. [`Dir::constant`] raises a positive upper-bound
//!   constant below `MIN_CONST` to `MIN_CONST` and flushes a lower-bound
//!   constant below `MIN_CONST` to zero; [`Dir::divisor`] lowers an
//!   upper-pass divisor above `1 / MIN_CONST` to `1 / MIN_CONST` and replaces
//!   a lower-pass one by infinity.
//!
//! Every lattice probability is nondecreasing in each constant and stored
//! operand and nonincreasing in each divisor (all terms are nonnegative), so
//! the admitted operands keep the direction of every bound. Every nonzero
//! product is then at least `2^-950`, a normal number, and exact zeros stay
//! exact zeros: a sum or product of nonnegative upper bounds is zero only if
//! the exact value is zero.
//!
//! Two scalar sums of the lower end (the Chernoff bound of the capped jumps
//! and the mean excess of the jump count) multiply factors that may be
//! subnormal. A product that underflows loses at most `2^-1075`; once the
//! rounded sum is at least `TINY = 2^-1000`, the relative factor of the
//! rounding exceeds these losses, and below `TINY` the absolute allowance
//! `2^-1074` per term is added.
//!
//! The rules and allowances change a value only where the computation
//! without them could underflow; [`take_admitted`] counts the changes, and a
//! count of zero means that the computation is the one without the rules.

use std::cell::Cell;

pub const U: f64 = 1.1102230246251565e-16; // 2^-53

/// 2^-400.
pub const FLOOR: f64 = f64::from_bits((1023u64 - 400) << 52);

/// Smallest positive constant allowed to multiply stored values (2^-150).
pub const MIN_CONST: f64 = f64::from_bits((1023u64 - 150) << 52);

/// Largest divisor allowed for stored values (2^150).
pub const MAX_DIVISOR: f64 = f64::from_bits((1023u64 + 150) << 52);

/// Sums below this value get the absolute underflow allowance (2^-1000).
pub const TINY: f64 = f64::from_bits((1023u64 - 1000) << 52);

/// The smallest positive subnormal number, 2^-1074.
pub const SUBNORMAL_MIN: f64 = f64::from_bits(1);

thread_local! {
    static ADMITTED: Cell<u64> = const { Cell::new(0) };
}

/// The number of operands changed by the admission rules (and of underflow
/// allowances added) on this thread since the last call, which resets it.
pub fn take_admitted() -> u64 {
    ADMITTED.with(|c| c.replace(0))
}

/// Count one admission or underflow allowance.
#[cold]
pub(crate) fn note_admitted() {
    ADMITTED.with(|c| c.set(c.get() + 1));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Up,
    Dn,
}

impl Dir {
    /// Bound of a nonnegative value `x` computed in RN by one operation.
    #[inline(always)]
    pub fn p(self, x: f64) -> f64 {
        match self {
            Dir::Up => pu(x),
            Dir::Dn => pd(x),
        }
    }
    /// Factor that turns an RN sum of at most `n` nonnegative products into a
    /// bound (before the final `p`).
    #[inline(always)]
    pub fn k(self, n: usize) -> f64 {
        let e = 4.0 * (n as f64 + 2.0) * U;
        match self {
            Dir::Up => 1.0 + e,
            Dir::Dn => 1.0 - e,
        }
    }
    /// Directed rounding of a (signed) scalar result computed in RN.
    #[inline(always)]
    pub fn r(self, x: f64) -> f64 {
        match self {
            Dir::Up => x.next_up(),
            Dir::Dn => x.next_down(),
        }
    }
    /// The end of an interval `(lo, hi)` that bounds in this direction.
    #[inline(always)]
    pub fn pick(self, iv: (f64, f64)) -> f64 {
        match self {
            Dir::Up => iv.1,
            Dir::Dn => iv.0,
        }
    }
    /// The end of an interval that bounds in the opposite direction.
    #[inline(always)]
    pub fn pick_rev(self, iv: (f64, f64)) -> f64 {
        match self {
            Dir::Up => iv.0,
            Dir::Dn => iv.1,
        }
    }
    /// A bound `c` of a nonnegative constant, admitted to be zero or at least
    /// `MIN_CONST`: the upper pass raises it, the lower pass flushes it (and
    /// any negative lower bound) to zero.
    #[inline(always)]
    pub fn constant(self, c: f64) -> f64 {
        self.admit(c, MIN_CONST)
    }
    /// A bound `x` of a nonnegative value, admitted as a stored value: zero
    /// or at least `FLOOR`, by the rule of [`Dir::constant`].
    #[inline(always)]
    pub fn stored(self, x: f64) -> f64 {
        self.admit(x, FLOOR)
    }
    #[inline(always)]
    fn admit(self, x: f64, min: f64) -> f64 {
        match self {
            Dir::Up => {
                debug_assert!(x >= 0.0 || x.is_nan(), "negative upper bound {x}");
                if x > 0.0 && x < min {
                    note_admitted();
                    return min;
                }
            }
            Dir::Dn => {
                if x < min && x != 0.0 {
                    note_admitted();
                    return 0.0;
                }
            }
        }
        x
    }
    /// A positive bound `x` of a divisor admitted to be at most `MAX_DIVISOR`
    /// (lowered in the upper pass, replaced by infinity in the lower pass).
    #[inline(always)]
    pub fn divisor(self, x: f64) -> f64 {
        if x > MAX_DIVISOR && x < f64::INFINITY {
            note_admitted();
            return match self {
                Dir::Up => MAX_DIVISOR,
                Dir::Dn => f64::INFINITY,
            };
        }
        if x == f64::INFINITY && self == Dir::Up {
            note_admitted();
            return MAX_DIVISOR;
        }
        x
    }
    /// An admitted vector of stored values.
    pub fn stored_vec(self, v: &[f64]) -> Vec<f64> {
        v.iter().map(|&x| self.stored(x)).collect()
    }
}

/// Upper bound of a nonnegative RN result: exact zero stays zero, a positive
/// value moves one ulp up and is raised to at least `FLOOR`.
#[inline(always)]
pub fn pu(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        let y = x.next_up();
        if y < FLOOR { FLOOR } else { y }
    }
}

/// Lower bound of a nonnegative RN result: values below `FLOOR` become zero.
#[inline(always)]
pub fn pd(x: f64) -> f64 {
    if x < FLOOR { 0.0 } else { x.next_down() }
}

#[inline(always)]
pub fn ru(x: f64) -> f64 {
    x.next_up()
}
#[inline(always)]
pub fn rd(x: f64) -> f64 {
    x.next_down()
}

// ---------------------------------------------------------------------------
// scalar interval helpers (f64 intervals [lo, hi], directed rounding)
// ---------------------------------------------------------------------------

pub type Iv = (f64, f64);

#[inline]
pub fn iv_add(a: Iv, b: Iv) -> Iv {
    (rd(a.0 + b.0), ru(a.1 + b.1))
}
#[inline]
pub fn iv_sub(a: Iv, b: Iv) -> Iv {
    (rd(a.0 - b.1), ru(a.1 - b.0))
}
/// Product of intervals of nonnegative quantities. A negative lower end (such
/// as `rd(0 + 0) = -2^-1074`) stands for zero.
#[inline]
pub fn iv_mul(a: Iv, b: Iv) -> Iv {
    debug_assert!(a.1 >= 0.0 && b.1 >= 0.0, "iv_mul {a:?} {b:?}");
    (rd(a.0.max(0.0) * b.0.max(0.0)).max(0.0), ru(a.1 * b.1))
}
/// Quotient of intervals of nonnegative quantities, the divisor positive; a
/// negative lower end of the dividend stands for zero.
#[inline]
pub fn iv_div(a: Iv, b: Iv) -> Iv {
    debug_assert!(a.1 >= 0.0 && b.0 > 0.0, "iv_div {a:?} {b:?}");
    (rd(a.0.max(0.0) / b.1).max(0.0), ru(a.1 / b.0))
}
#[inline]
pub fn iv_sqrt(a: Iv) -> Iv {
    (rd(a.0.max(0.0).sqrt()).max(0.0), ru(a.1.sqrt()))
}
#[inline]
pub fn iv_pt(x: f64) -> Iv {
    (x, x)
}

// ---------------------------------------------------------------------------
// elementary functions with rigorous bounds
// ---------------------------------------------------------------------------

/// The f64 neighbours of ln 2.
pub const LN2_DN: f64 = 0.6931471805599452;
pub const LN2_UP: f64 = 0.6931471805599454;

/// Lower and upper bounds of exp(r) for 0 <= r <= 1 from the Taylor series
/// (all terms positive): the partial sum rounded down is a lower bound; the
/// partial sum rounded up plus twice the first omitted term is an upper bound.
fn exp_unit(r: f64) -> (f64, f64) {
    debug_assert!((0.0..=1.0).contains(&r), "exp_unit {r}");
    let (mut slo, mut shi) = (1.0f64, 1.0f64);
    let (mut tlo, mut thi) = (1.0f64, 1.0f64);
    for j in 1..=27 {
        tlo = rd(rd(tlo * r) / j as f64).max(0.0);
        thi = ru(ru(thi * r) / j as f64);
        slo = rd(slo + tlo);
        shi = ru(shi + thi);
    }
    // first omitted term r^28/28! <= thi r / 28; the tail is at most twice it
    let tail = ru(ru(ru(thi * r) / 28.0) * 2.0);
    (slo, ru(shi + tail))
}

/// Bounds of exp(-x), x >= 0, as a mantissa interval and a binary exponent:
/// exp(-x) is in 2^e [m_lo, m_hi], from x = k ln 2 + r with 0 <= r <= 1.
pub fn exp_neg_split(x: f64) -> ((f64, f64), i64) {
    debug_assert!(x >= 0.0);
    if x == 0.0 {
        return ((1.0, 1.0), 0);
    }
    if x >= 1e12 {
        // e^{-x} = 2^{-x log2 e} <= 2^{-x} <= 2^{-10^12}
        return ((0.0, 1.0), -1_000_000_000_000);
    }
    let k = (x / LN2_UP).floor().max(0.0);
    // r = x - k ln 2 lies in [r_lo, r_hi]
    let r_lo = rd(x - ru(k * LN2_UP)).max(0.0);
    let r_hi = ru(x - rd(k * LN2_DN)).max(r_lo);
    assert!(
        r_hi <= 1.0,
        "exp_neg_split: argument reduction failed for {x}"
    );
    let m_hi = ru(1.0 / exp_unit(r_lo).0); // exp(-r) <= exp(-r_lo)
    let m_lo = rd(1.0 / exp_unit(r_hi).1); // exp(-r) >= exp(-r_hi)
    ((m_lo, m_hi), -(k as i64))
}

/// 2^e as f64 for e in [-1074, 1023] (exact).
#[inline]
pub fn pow2(e: i64) -> f64 {
    assert!((-1074..=1023).contains(&e));
    if e >= -1022 {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (e + 1074))
    }
}

/// Bounds of exp(-x) for x >= 0 (the lower bound is 0 when the value is
/// below 2^-1000).
pub fn exp_neg_bounds(x: f64) -> (f64, f64) {
    let ((mlo, mhi), e) = exp_neg_split(x);
    if e < -1000 {
        return (0.0, pow2(-999));
    }
    let s = pow2(e);
    (rd(mlo * s).max(0.0), ru(mhi * s))
}

/// Upper bound of exp(x) for 0 <= x <= 700.
pub fn exp_pos_up(x: f64) -> f64 {
    assert!((0.0..=700.0).contains(&x));
    let (lo, _) = exp_neg_bounds(x);
    assert!(lo > 0.0);
    ru(1.0 / lo)
}

// ---------------------------------------------------------------------------
// Poisson probabilities with an extended exponent
// ---------------------------------------------------------------------------

/// Bounds of the Poisson probabilities e^{-t} t^n / n!, n = 0..up.len()-1, for
/// every t in [t_lo, t_hi] (0 <= t_lo <= t_hi): `up[n] >= e^{-t_lo} t_hi^n / n!`
/// and `dn[n] <= e^{-t_hi} t_lo^n / n!`. Values below FLOOR are returned as
/// FLOOR (upper) or 0 (lower).
pub fn poisson_bounds(t_lo: f64, t_hi: f64, up: &mut [f64], dn: &mut [f64]) {
    debug_assert!(t_lo >= 0.0 && t_hi >= t_lo);
    let len = up.len();
    if t_hi == 0.0 {
        up.fill(0.0);
        dn.fill(0.0);
        up[0] = 1.0;
        dn[0] = 1.0;
        return;
    }
    let ((_, mhi), eu) = exp_neg_split(t_lo);
    let ((mlo, _), ed) = exp_neg_split(t_hi);
    let (mut mu, mut eu) = (mhi, eu);
    let (mut md, mut ed) = (mlo, ed);
    const BIG: f64 = 1.2676506002282294e30; // 2^100
    const SMALL: f64 = 7.888609052210118e-31; // 2^-100
    for n in 0..len {
        if n > 0 {
            let nf = n as f64;
            mu = ru(ru(mu * t_hi) / nf);
            md = rd(rd(md * t_lo) / nf).max(0.0);
            if mu > BIG {
                mu *= SMALL;
                eu += 100;
            } else if mu < SMALL {
                mu *= BIG;
                eu -= 100;
            }
            if md > BIG {
                md *= SMALL;
                ed += 100;
            } else if md < SMALL && md > 0.0 {
                md *= BIG;
                ed -= 100;
            }
        }
        // mu * 2^eu with SMALL <= mu <= BIG (after the first step)
        up[n] = if eu < -1000 {
            FLOOR
        } else if eu > 900 {
            f64::INFINITY
        } else {
            let v = ru(mu * pow2(eu.max(-1074)));
            if v < FLOOR { FLOOR } else { v }
        };
        dn[n] = if md == 0.0 || !(-1000..=900).contains(&ed) {
            0.0
        } else {
            let v = rd(md * pow2(ed.max(-1074)));
            if v < FLOOR { 0.0 } else { v }
        };
    }
}

/// Upper bound of P(Pois(t) >= n) for t <= t_hi given an upper bound `p_n`
/// of P(Pois(t) = n): p_n / (1 - t_hi/(n+1)) when n + 1 > t_hi, else 1.
#[inline]
pub fn poisson_tail_up(p_n: f64, t_hi: f64, n: usize) -> f64 {
    let ratio = ru(t_hi / (n as f64 + 1.0));
    if ratio >= 1.0 {
        return 1.0;
    }
    ru(p_n / rd(1.0 - ratio)).min(1.0)
}

// ---------------------------------------------------------------------------
// vector kernels
// ---------------------------------------------------------------------------

/// First index with a nonzero entry (len if none).
#[inline]
pub fn first_nz(a: &[f64]) -> usize {
    a.iter().position(|&x| x != 0.0).unwrap_or(a.len())
}

/// Last index with a nonzero entry plus one (0 if none).
#[inline]
pub fn end_nz(a: &[f64]) -> usize {
    a.iter()
        .rposition(|&x| x != 0.0)
        .map(|i| i + 1)
        .unwrap_or(0)
}

/// RN dot product with eight accumulators.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0f64; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s = 0.0;
    for i in 0..ra.len() {
        s += ra[i] * rb[i];
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7])) + s
}

/// Truncated convolution `out[m] = sum_{j<=m} a[j] b[m-j]`, m < out.len(),
/// as a bound in direction `dir` (inputs are bounds in the same direction).
pub fn conv(a: &[f64], b: &[f64], out: &mut [f64], dir: Dir) {
    let n = out.len();
    out.fill(0.0);
    let a = &a[..n.min(a.len())];
    let b = &b[..n.min(b.len())];
    let (a0, a1) = (first_nz(a), end_nz(a));
    let (b0, b1) = (first_nz(b), end_nz(b));
    if a0 >= a1 || b0 >= b1 {
        return;
    }
    for j in a0..a1 {
        let aj = a[j];
        if aj == 0.0 {
            continue;
        }
        let start = j + b0;
        if start >= n {
            break;
        }
        let end = (j + b1).min(n);
        let dst = &mut out[start..end];
        let src = &b[b0..b0 + (end - start)];
        for (x, &y) in dst.iter_mut().zip(src) {
            *x += aj * y;
        }
    }
    let k = dir.k(n);
    for x in out.iter_mut() {
        *x = dir.p(*x * k);
    }
}

/// Convolution returning a new vector of length `n`.
pub fn conv_new(a: &[f64], b: &[f64], n: usize, dir: Dir) -> Vec<f64> {
    let mut out = vec![0.0; n];
    conv(a, b, &mut out, dir);
    out
}

/// Online triangular solver state: values f[0..len) are known and stored both
/// forward and backward (`rev[cap - 1 - i] = f[i]`) so that
/// `sum_{j=1}^{m} x[j] f[m-j]` (m = len) is a contiguous dot product.
pub struct Tri {
    pub f: Vec<f64>,
    rev: Vec<f64>,
    cap: usize,
    len: usize,
}

impl Tri {
    pub fn new(cap: usize) -> Tri {
        Tri {
            f: vec![0.0; cap],
            rev: vec![0.0; cap],
            cap,
            len: 0,
        }
    }
    #[inline]
    pub fn push(&mut self, v: f64) {
        self.f[self.len] = v;
        self.rev[self.cap - 1 - self.len] = v;
        self.len += 1;
    }
    /// RN value of `sum_{j=1}^{m} x[j] f[m-j]`, `m` = current length (x.len() > m).
    #[inline]
    pub fn cross(&self, x: &[f64]) -> f64 {
        let m = self.len;
        let lo = self.cap - m;
        dot(&x[1..=m], &self.rev[lo..self.cap])
    }
    /// RN value of `sum_{j=1}^{m-1} f[j] f[m-j]`, `m` = current length.
    #[inline]
    pub fn selfconv(&self) -> f64 {
        let m = self.len;
        if m < 2 {
            return 0.0;
        }
        let h = (m - 1) / 2;
        let lo = self.cap - m;
        let s = dot(&self.f[1..=h], &self.rev[lo..lo + h]);
        let mut t = 2.0 * s;
        if m.is_multiple_of(2) {
            let c = self.f[m / 2];
            t += c * c;
        }
        t
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn into_vec(self) -> Vec<f64> {
        self.f
    }
}
