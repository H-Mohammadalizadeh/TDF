//! Certified enclosure of the miss probability of one class under classical
//! cyclic exhaustive polling (visit classes 0, 1, ..., N-1 cyclically, serve each
//! until empty, switch even when empty, never idle).
//!
//! `m_i = 1 - E[u_i(I_i)] / E[I_i]`, `u_i(x) = W_i(min(x, d))`, with `I_i` the stationary
//! intervisit time of class i (Fuhrmann-Cooper decomposition, exponential service).
//! I_i is replaced by a lattice variable I_comp <=_icv I_i (truncation to K past
//! cycles, deletions, mean-preserving spreads); the upper pass computes upper bounds
//! of the lattice probabilities of I_comp, the lower pass lower bounds. The upper end
//! hi >= m_i holds for any lattice step, K, cap and jump cap; the lower end lo adds
//! the deleted means and the second-order spread term.
//!
//! Positions relative to the target class i: even p = 2r is the visit of class i + r,
//! odd p = 2r + 1 the reconfiguration from class i + r to i + r + 1. The intervisit
//! period of i is P = {1, ..., 2N-1}.
//!
//! Every operand that enters a lattice kernel without being the result of one is
//! admitted by the rules of [`super::fp`] (constants, stored values, divisors).

use super::fp::*;
use super::laws::*;
use crate::feasibility::rational::{
    Q, dec, exp_neg, exp_neg_powers, qi, qr, round_down, round_up, to_f64,
};
use crate::feasibility::special::sqrt_enclosure;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct System {
    pub lam: Vec<Q>,
    pub mu: Vec<Q>,
    /// `sw[j]`: reconfiguration from class j to class j+1 (mod N)
    pub sw: Vec<Law>,
}

impl System {
    pub fn symmetric(n: usize, rho: &Q, mu: &Q, law: &Law) -> System {
        let lam = rho * mu / qi(n as i64);
        System {
            lam: vec![lam; n],
            mu: vec![mu.clone(); n],
            sw: vec![law.clone(); n],
        }
    }
    pub fn n(&self) -> usize {
        self.lam.len()
    }
    pub fn rho(&self) -> Q {
        self.lam.iter().zip(&self.mu).map(|(l, m)| l / m).sum()
    }
    /// Enclosure of `E[I_i] = (1 - rho_i) s / (1 - rho)`.
    pub fn mean_intervisit(&self, i: usize) -> (Q, Q) {
        let rho = self.rho();
        let ri = &self.lam[i] / &self.mu[i];
        let (mut slo, mut shi) = (Q::zero(), Q::zero());
        for l in &self.sw {
            let m = l.mean();
            slo += m.0;
            shi += m.1;
        }
        let f = (Q::one() - &ri) / (Q::one() - &rho);
        (&slo * &f, &shi * &f)
    }
}

/// Numerical choices of the enclosure; they affect its width, never its validity.
#[derive(Clone, Debug)]
pub struct Opts {
    /// lattice step g
    pub g: Q,
    /// stop the truncation when `E[R_K] / E[I] <= tol_trunc`
    pub tol_trunc: f64,
    /// fewest and most past cycles K
    pub kmin: usize,
    pub kmax: usize,
    /// also compute the lower end lo
    pub lower: bool,
    /// skip a reconfiguration outside the period when its jump rate is at most eta_skip Lambda
    pub eta_skip: f64,
    /// target of the heuristic Chernoff choice of the jump cap
    pub chern_eps: f64,
    /// cap the jumps once the bound of P(N >= n) is at most qtail_eps
    pub qtail_eps: f64,
    /// partition step for reconfigurations outside the period (None: 2^floor(log2(mean/100)))
    pub h0: Option<Q>,
}

impl Default for Opts {
    fn default() -> Opts {
        Opts {
            g: qr(1, 200),
            tol_trunc: 1e-7,
            kmin: 1,
            kmax: 2000,
            lower: true,
            eta_skip: 1e-10,
            chern_eps: 1e-16,
            qtail_eps: 1e-14,
            h0: None,
        }
    }
}

/// The enclosure lo <= m_i <= hi of one class and its diagnostics.
#[derive(Clone, Debug, Default)]
pub struct Out {
    pub target: usize,
    /// lattice cells M = floor(d / g)
    pub m_cells: usize,
    /// past cycles K of the truncation
    pub k_cycles: usize,
    /// enclosure of `E[I_i]`
    pub e_i: (f64, f64),
    /// upper bound of the truncated mean `E[R_K]`
    pub e_rk: f64,
    /// upper bound of the deleted mean (skips, caps)
    pub r_del: f64,
    /// upper bound of the variance added by the spreads
    pub v_add: f64,
    /// hi and lo as decimals with 12 places, rounded outward
    pub hi: String,
    pub lo: String,
    /// hi and lo as f64, rounded outward
    pub hi_f: f64,
    pub lo_f: f64,
    /// lower bound of m(I_comp) (without the truncation/spread corrections)
    pub lo_comp_f: f64,
    /// upper bound of P(I_comp <= d)
    pub mass_le_d: f64,
    pub seconds: f64,
    pub convs: usize,
    pub nmax_max: usize,
    pub skipped: usize,
    /// operands changed by the admission rules (zero: the rules did not alter the computation)
    pub admitted: u64,
    /// upper and lower bounds of the lattice law of I_comp on 0, g, ..., M g
    pub law_up: Vec<f64>,
    pub law_dn: Vec<f64>,
}

fn md(a: i64, n: i64) -> i64 {
    a.rem_euclid(n)
}
fn visit_class(p: i64, i: usize, n: usize) -> usize {
    md(i as i64 + p.div_euclid(2), n as i64) as usize
}
fn switch_class(p: i64, i: usize, n: usize) -> usize {
    md(i as i64 + (p - 1).div_euclid(2), n as i64) as usize
}
fn next_visit(p: i64, k: usize, i: usize, n: usize) -> i64 {
    let nn = n as i64;
    if p.rem_euclid(2) == 0 {
        let c = visit_class(p, i, n);
        if k == c {
            p
        } else {
            p + 2 * md(k as i64 - c as i64, nn)
        }
    } else {
        let j = switch_class(p, i, n);
        p + 1 + 2 * md(k as i64 - j as i64 - 1, nn)
    }
}

/// Spread of Exp(rate) onto g Z: P(0) = 1 - (1 - r)/x, P(m) = C r^m (m >= 1),
/// r = e^{-x}, C = (1 - r)^2 / (r x), x = rate g. Returns f64 intervals of (P(0), C, r)
/// and an upper bound of the spread variance E[Var(X~|X)] = (g/rate)(1+r)/(1-r) - 2/rate^2.
#[derive(Clone, Debug)]
pub struct ExpSpread {
    pub p0: Iv,
    pub c: Iv,
    pub r: Iv,
    pub p0q: (Q, Q),
    pub var_hi: f64,
}

pub fn exp_spread(rate: &Q, g: &Q) -> ExpSpread {
    let x = rate * g;
    let (rlo, rhi) = exp_neg(&x, BITS + 32);
    let p0 = |r: &Q| Q::one() - (Q::one() - r) / &x;
    let cc = |r: &Q| (Q::one() - r) * (Q::one() - r) / (r * &x);
    let p0lo = p0(&rlo);
    let p0hi = p0(&rhi);
    let var = |r: &Q| (g / rate) * (Q::one() + r) / (Q::one() - r) - qi(2) / (rate * rate);
    let v = var(&rhi);
    let gg = q_up(&(g * g / qi(4)));
    ExpSpread {
        p0: (q_dn(&p0lo), q_up(&p0hi)),
        c: (q_dn(&cc(&rhi)), q_up(&cc(&rlo))),
        r: (q_dn(&rlo), q_up(&rhi)),
        p0q: (p0lo, p0hi),
        var_hi: q_up(&v).max(0.0).min(gg),
    }
}

/// Precomputed data of one reconfiguration law for a lattice (g, len).
#[derive(Debug)]
pub enum SwPre {
    Det {
        atoms: Vec<(usize, Iv)>,
        c: Iv,
        var_hi: f64,
    },
    Exp {
        rate: Iv,
        sp: ExpSpread,
    },
    Erlang {
        k: u32,
        rate: Iv,
        sp: ExpSpread,
    },
    Part {
        lat_lo: Vec<f64>,
        lat_hi: Vec<f64>,
        part: Arc<Partition>,
        var_hi: f64,
    },
}

fn cache_sw() -> &'static Mutex<HashMap<String, Arc<SwPre>>> {
    static C: OnceLock<Mutex<HashMap<String, Arc<SwPre>>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}
fn cache_part() -> &'static Mutex<HashMap<String, Arc<Partition>>> {
    static C: OnceLock<Mutex<HashMap<String, Arc<Partition>>>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Default lattice step for the threshold `d` and the mean reconfiguration time
/// `mean`: g = d / max(400, ceil(100 d / mean)), i.e. at most d/400 and about mean/100
/// (d / max(400, ceil(200 d)) at mean 1/2).
pub fn default_step(d: &Q, mean: &Q) -> Q {
    let m = (d * qi(100) / mean)
        .ceil()
        .to_integer()
        .to_i64()
        .unwrap_or(i64::MAX)
        .max(400);
    d / qi(m)
}

/// Default partition step: the largest power of two not above mean/100.
pub fn default_h0(law: &Law) -> Q {
    let m = to_f64(&law.mean().0) / 100.0;
    let e = m.log2().floor() as i32;
    if e >= 0 {
        Q::from_integer(num_bigint::BigInt::one() << e as u32)
    } else {
        Q::new(
            num_bigint::BigInt::one(),
            num_bigint::BigInt::one() << (-e) as u32,
        )
    }
}

pub fn partition_cached(law: &Law, h0: &Q) -> Arc<Partition> {
    let key = format!("{law:?}|{h0}");
    if let Some(p) = cache_part().lock().unwrap().get(&key) {
        return p.clone();
    }
    let mean = law.mean().1;
    let y0_end = (&mean * qi(20) / h0).ceil() * h0;
    let p = Arc::new(partition(law, h0, &y0_end));
    cache_part().lock().unwrap().insert(key, p.clone());
    p
}

/// Precomputed reconfiguration data, cached by (law, g); the lattice spread is
/// recomputed when a longer lattice is needed.
pub fn sw_pre(law: &Law, g: &Q, len: usize, h0: &Q) -> Arc<SwPre> {
    let key = format!("{law:?}|{g}|{h0}");
    if let Some(p) = cache_sw().lock().unwrap().get(&key) {
        let ok = match p.as_ref() {
            SwPre::Part { lat_hi, .. } => lat_hi.len() >= len,
            _ => true,
        };
        if ok {
            return p.clone();
        }
    }
    let gg = q_up(&(g * g / qi(4)));
    let pre = match law {
        Law::Det { c } => {
            let t = c / g;
            let k = t.floor();
            let f = &t - &k;
            let k = k.to_integer().to_usize().unwrap();
            let mut atoms = vec![(k, (q_dn(&(Q::one() - &f)), q_up(&(Q::one() - &f))))];
            if f.is_positive() {
                atoms.push((k + 1, (q_dn(&f), q_up(&f))));
            }
            let var = &f * (Q::one() - &f) * g * g;
            SwPre::Det {
                atoms,
                c: (q_dn(c), q_up(c)),
                var_hi: q_up(&var),
            }
        }
        Law::Exp { rate } => SwPre::Exp {
            rate: (q_dn(rate), q_up(rate)),
            sp: exp_spread(rate, g),
        },
        Law::Erlang { k, rate } => SwPre::Erlang {
            k: *k,
            rate: (q_dn(rate), q_up(rate)),
            sp: exp_spread(rate, g),
        },
        _ => {
            let (lo, hi) = spread_lattice(law, g, len);
            SwPre::Part {
                lat_lo: lo,
                lat_hi: hi,
                part: partition_cached(law, h0),
                var_hi: gg,
            }
        }
    };
    let pre = Arc::new(pre);
    cache_sw().lock().unwrap().insert(key, pre.clone());
    pre
}

/// Shift a vector right by k cells (truncated to len).
fn shift(v: &[f64], k: usize, len: usize) -> Vec<f64> {
    let mut out = vec![0.0; len];
    if k < len {
        let n = (len - k).min(v.len());
        out[k..k + n].copy_from_slice(&v[..n]);
    }
    out
}

/// e^{-lam_plus x} exp(x J~) (lattice law of a compound Poisson sum over time x
/// with jump-rate measure J~, J~[0] = 0), by the power-series exponential
/// F_m = (1/m) sum_j j x J~_j F_{m-j}; arguments x lam_plus above 200 are
/// halved and the result squared.
fn exp_series(
    jt: &[f64],
    x: Iv,
    lam_plus: Iv,
    len: usize,
    dir: Dir,
    convs: &mut usize,
) -> Vec<f64> {
    let t_hi = ru(lam_plus.1 * x.1);
    if t_hi > 200.0 {
        let half = (rd(x.0 * 0.5), ru(x.1 * 0.5));
        let h = exp_series(jt, half, lam_plus, len, dir, convs);
        *convs += 1;
        return conv_new(&h, &h, len, dir);
    }
    let xx = dir.constant(dir.pick(x));
    let mut h = vec![0.0; len];
    for j in 1..len.min(jt.len()) {
        if jt[j] != 0.0 {
            h[j] = dir.p(j as f64 * dir.p(xx * jt[j]));
        }
    }
    let f0 = match dir {
        Dir::Up => exp_neg_bounds(rd(lam_plus.0 * x.0).max(0.0)).1,
        Dir::Dn => exp_neg_bounds(t_hi).0,
    };
    let f0 = dir.p(f0);
    let mut tri = Tri::new(len);
    tri.push(f0);
    let kk = dir.k(len);
    for m in 1..len {
        let s = tri.cross(&h);
        let v = dir.p(dir.p(s * kk) / m as f64);
        tri.push(v);
    }
    *convs += 1;
    tri.into_vec()
}

/// T = t0 delta_0 + sum_{n>=1} coef^n X^{*n} t0 i.e. T_0 = t0, T_m = coef sum_j X_j T_{m-j}
/// (X[0] = 0): the geometric series t0 (delta_0 - coef X)^{-1}.
fn geo_inverse(x: &[f64], coef: f64, t0: f64, len: usize, dir: Dir, convs: &mut usize) -> Vec<f64> {
    let mut tri = Tri::new(len);
    tri.push(t0);
    let kk = dir.k(len);
    for _m in 1..len {
        let s = tri.cross(x);
        tri.push(dir.p(coef * dir.p(s * kk)));
    }
    *convs += 1;
    tri.into_vec()
}

/// p^k (delta_0 - X)^{-k}, X[0] = 0, by F_m = sum_j (1 + (k-1) j/m) X_j F_{m-j}.
fn nb_series(x: &[f64], k: u32, f0: f64, len: usize, dir: Dir, convs: &mut usize) -> Vec<f64> {
    let mut jx = vec![0.0; len];
    for j in 1..len.min(x.len()) {
        if x[j] != 0.0 {
            jx[j] = dir.p(j as f64 * x[j]);
        }
    }
    let km1 = (k - 1) as f64;
    let mut tri = Tri::new(len);
    tri.push(f0);
    let kk = dir.k(len);
    for m in 1..len {
        let a = dir.p(tri.cross(x) * kk);
        let b = if k > 1 {
            dir.p(dir.p(km1 * dir.p(tri.cross(&jx) * kk)) / m as f64)
        } else {
            0.0
        };
        tri.push(dir.p(a + b));
    }
    *convs += 2;
    tri.into_vec()
}

/// sum_{n < coef.len()} coef[n] X^{*n} on 0..len-1 by Paterson-Stockmeyer. With
/// `shift` (X[0] = 0 exactly) the terms n >= len vanish and are skipped.
fn ps_eval(
    coef: &[f64],
    x: &[f64],
    len: usize,
    shift_ok: bool,
    dir: Dir,
    convs: &mut usize,
) -> Vec<f64> {
    let mut dcount = coef.len();
    if shift_ok {
        dcount = dcount.min(len);
    }
    let mut out = vec![0.0; len];
    if dcount == 0 {
        return out;
    }
    if dcount == 1 {
        out[0] = coef[0];
        return out;
    }
    let k = ((dcount as f64).sqrt().ceil() as usize).max(2);
    let mut pw: Vec<Vec<f64>> = Vec::with_capacity(k + 1);
    let mut p0 = vec![0.0; len];
    p0[0] = 1.0;
    pw.push(p0);
    pw.push(x[..len].to_vec());
    for i in 2..=k {
        let v = conv_new(&pw[i - 1], x, len, dir);
        *convs += 1;
        pw.push(v);
    }
    let nb = dcount.div_ceil(k);
    let block = |j: usize| -> Vec<f64> {
        let mut b = vec![0.0; len];
        let kk = dir.k(k + 1);
        for i in 0..k {
            let n = j * k + i;
            if n >= dcount {
                break;
            }
            let c = coef[n];
            if c == 0.0 {
                continue;
            }
            for (bx, &px) in b.iter_mut().zip(&pw[i]) {
                *bx += c * px;
            }
        }
        for bx in b.iter_mut() {
            *bx = dir.p(*bx * kk);
        }
        b
    };
    let mut t = block(nb - 1);
    for j in (0..nb - 1).rev() {
        let mut nt = conv_new(&pw[k], &t, len, dir);
        *convs += 1;
        let b = block(j);
        for (x, y) in nt.iter_mut().zip(&b) {
            *x = dir.p(*x + *y);
        }
        t = nt;
    }
    t
}

/// Heuristic Chernoff choice of the jump cap: smallest n with
/// min_s e^{s d} (sum_m jhat_m e^{-s m g} + e^{-s len g})^n <= eps.
fn chernoff_n(jhat: &[f64], g: f64, len: usize, d: f64, eps: f64) -> (usize, f64) {
    let le = eps.ln();
    let mut best = (usize::MAX, 0.0);
    let mut s = 0.25 / d;
    while s * d < 600.0 {
        let e1 = (-s * g).exp();
        let mut em = 1.0;
        let mut phi = 0.0;
        for m in 1..len {
            em *= e1;
            phi += jhat[m] * em;
        }
        phi += em * e1;
        if phi < 1.0 && phi > 0.0 {
            let n = ((le - s * d) / phi.ln()).ceil();
            if n >= 0.0 && n < best.0 as f64 {
                best = (n as usize, s);
            }
        } else if phi == 0.0 {
            best = (1, s);
        }
        s *= 1.5;
    }
    best
}

/// Rigorous upper bound of e^{s d} (sum_m jhat_m e^{-s m g} + e^{-s len g})^n.
fn chernoff_up(jhat: &[f64], g: Iv, len: usize, d: f64, s: f64, n: usize) -> f64 {
    if s <= 0.0 || n == 0 {
        return 1.0;
    }
    let sd = ru(s * d);
    if sd > 700.0 {
        return 1.0;
    }
    let e1 = exp_neg_bounds(rd(s * g.0).max(0.0)).1; // >= e^{-s g}
    let mut em = 1.0f64;
    let mut phi = 0.0f64;
    for m in 1..len {
        em = ru(em * e1);
        phi += jhat[m] * em;
    }
    // em may become subnormal, and a product below 2^-1022 may lose up to 2^-1075.
    // The relative factor covers these losses once the sum is at least TINY; below
    // that, the absolute allowance len 2^-1074 is added.
    let mut phi = ru(phi * Dir::Up.k(len));
    if phi < TINY {
        note_admitted();
        phi = ru(phi + len as f64 * SUBNORMAL_MIN);
    }
    let phi = ru(phi + ru(em * e1));
    if phi >= 1.0 {
        return 1.0;
    }
    // phi^n by repeated squaring
    let mut base = phi;
    let mut acc = 1.0f64;
    let mut e = n;
    while e > 0 {
        if e & 1 == 1 {
            acc = ru(acc * base);
        }
        base = ru(base * base);
        e >>= 1;
    }
    ru(exp_pos_up(sd) * acc).min(1.0)
}

/// Mixed Poisson weights of the partition-spread law at the rate interval
/// lam_plus: q_up[n] >= sum_l w_l Pois(n; t) and q_dn[n] <= ... for every t in the
/// node's interval [lam_lo y_l, lam_hi y_l]; tail[n] >= P(N >= n); en[n] >= E[(N-n)^+].
struct MixedPois {
    q_up: Vec<f64>,
    q_dn: Vec<f64>,
    tail: Vec<f64>,
    en: Vec<f64>,
}

fn mixed_poisson(part: &Partition, lam_plus: Iv, nq: usize) -> MixedPois {
    let mut q_up = vec![0.0; nq];
    let mut q_dn = vec![0.0; nq];
    let mut tail = vec![0.0; nq];
    let mut en = vec![0.0; nq];
    let mut up = vec![0.0; nq];
    let mut dn = vec![0.0; nq];
    for l in 0..part.y.len() {
        if part.w_hi[l] == 0.0 {
            continue;
        }
        // the partition weights multiply stored Poisson bounds
        let (wl, wh) = (Dir::Dn.stored(part.w_lo[l]), Dir::Up.stored(part.w_hi[l]));
        let y = part.y[l];
        let t_lo = rd(lam_plus.0 * part.y_lo[l]).max(0.0);
        let t_hi = ru(lam_plus.1 * y);
        poisson_bounds(t_lo, t_hi, &mut up, &mut dn);
        for n in 0..nq {
            q_up[n] += wh * up[n];
            q_dn[n] += wl * dn[n];
            let tl = poisson_tail_up(up[n], t_hi, n);
            tail[n] += wh * tl;
            en[n] += wh * ru(t_hi * tl);
        }
    }
    let nn = part.y.len();
    let ku = Dir::Up.k(nn);
    let kd = Dir::Dn.k(nn);
    for n in 0..nq {
        q_up[n] = pu(q_up[n] * ku);
        q_dn[n] = pd(q_dn[n] * kd);
        tail[n] = ru(tail[n] * ku).min(1.0);
        en[n] = ru(en[n] * ku);
        // the factors ru(t_hi tl) may be subnormal (t_hi = ru(0) at y = 0): below TINY the
        // relative factor may not cover the underflow of the products, so add the allowance
        if en[n] < TINY {
            note_admitted();
            en[n] = ru(en[n] + 2.0 * nn as f64 * SUBNORMAL_MIN);
        }
    }
    MixedPois {
        q_up,
        q_dn,
        tail,
        en,
    }
}

struct VData {
    up: Vec<f64>,
    dn: Vec<f64>,
    z: Iv,
}

/// The enclosure for target class i and threshold d.
pub fn compute(sys: &System, i: usize, d: &Q, opts: &Opts) -> Out {
    let t0 = Instant::now();
    take_admitted();
    let n = sys.n();
    let nn = n as i64;
    let g = &opts.g;
    let gf: Iv = (q_dn(g), q_up(g));
    let lam_tot: Q = sys.lam.iter().sum();
    let nu: Vec<Q> = (0..n).map(|c| &sys.mu[c] + &lam_tot).collect();
    let lam_f: Vec<Iv> = sys.lam.iter().map(|x| (q_dn(x), q_up(x))).collect();
    let inv_nu_f: Vec<Iv> = nu
        .iter()
        .map(|x| {
            let r = x.recip();
            (q_dn(&r), q_up(&r))
        })
        .collect();
    let lamtot_f: Iv = (q_dn(&lam_tot), q_up(&lam_tot));
    let rho = sys.rho();
    assert!(rho < Q::one(), "unstable");
    let means: Vec<(Q, Q)> = sys.sw.iter().map(|l| l.mean()).collect();
    let s_f: Vec<Iv> = means.iter().map(|m| (q_dn(&m.0), q_up(&m.1))).collect();
    let ei_q = sys.mean_intervisit(i);
    let ei: Iv = (q_dn(&ei_q.0), q_up(&ei_q.1));
    let s_hi: Q = means.iter().map(|m| m.1.clone()).sum();
    let ec_hi = q_up(&(&s_hi / (Q::one() - &rho)));
    let theta = &sys.mu[i] - &sys.lam[i];
    let mcells = (d / g).floor().to_integer().to_usize().unwrap();
    let len = mcells + 1;
    let in_p = |p: i64| p >= 1 && p < 2 * nn;
    let vidx = |q: i64| ((2 * nn - 2 - q) / 2) as usize;
    let ridx = |p: i64| ((2 * nn - 1 - p) / 2) as usize;
    let mu_m_lam: Vec<Iv> = (0..n)
        .map(|c| {
            let x = &sys.mu[c] - &sys.lam[c];
            (q_dn(&x), q_up(&x))
        })
        .collect();

    // ---------------- pre-pass: means, kappa, truncation depth K ----------------
    let mut mbar: Vec<Iv> = Vec::new();
    let mut kappa: Vec<Iv> = Vec::new();
    let mut sum_sk: Iv = (0.0, 0.0);
    let mut k_cycles: usize;
    let mut p = 2 * nn - 1;
    loop {
        if p.rem_euclid(2) == 1 {
            let j = switch_class(p, i, n);
            let a = if in_p(p) { 1.0 } else { 0.0 };
            let mut acc: Iv = (a, a);
            for k in 0..n {
                let r = next_visit(p, k, i, n);
                if r < 2 * nn {
                    acc = iv_add(acc, iv_mul(lam_f[k], mbar[vidx(r)]));
                }
            }
            kappa.push(acc);
            sum_sk = iv_add(sum_sk, iv_mul(s_f[j], acc));
            if (p - 1).rem_euclid(2 * nn) == 0 && p <= 1 {
                k_cycles = ((1 - p) / (2 * nn)) as usize;
                let erk = ru(ei.1 - sum_sk.0);
                if (k_cycles >= opts.kmin && erk <= opts.tol_trunc * ei.0) || k_cycles >= opts.kmax
                {
                    break;
                }
            }
        } else {
            let c = visit_class(p, i, n);
            let a = if in_p(p) { 1.0 } else { 0.0 };
            let mut acc: Iv = (a, a);
            for k in 0..n {
                if k == c {
                    continue;
                }
                let r = next_visit(p, k, i, n);
                if r < 2 * nn {
                    acc = iv_add(acc, iv_mul(lam_f[k], mbar[vidx(r)]));
                }
            }
            mbar.push(iv_div(acc, mu_m_lam[c]));
        }
        p -= 1;
    }
    let p_min = 1 - 2 * nn * k_cycles as i64;
    let e_rk = ru(ei.1 - sum_sk.0).max(0.0);

    // ---------------- per-class and per-law constants ----------------
    let exp_sp: Vec<ExpSpread> = nu.iter().map(|x| exp_spread(x, g)).collect();
    let h0s: Vec<Q> = sys
        .sw
        .iter()
        .map(|l| opts.h0.clone().unwrap_or_else(|| default_h0(l)))
        .collect();
    let pres: Vec<Arc<SwPre>> = (0..n)
        .map(|j| sw_pre(&sys.sw[j], g, len, &h0s[j]))
        .collect();
    let mean_cap: Vec<f64> = (0..n).map(|j| s_f[j].1).collect();

    // chi(d - m g) and W(d)
    let bits = BITS + 32;
    let mq = qi(mcells as i64);
    let delta = d - &mq * g;
    let (e_lo_pows, e_hi_pows) = exp_neg_powers(&(&theta * g), mcells, bits);
    let (ed_lo, ed_hi) = exp_neg(&(&theta * &delta), bits);
    let mut chi_lo = vec![0.0; len];
    let mut chi_hi = vec![0.0; len];
    for m in 0..len {
        let jj = mcells - m;
        let v = &delta + qi(jj as i64) * g;
        let elo = &ed_lo * &e_lo_pows[jj];
        let ehi = &ed_hi * &e_hi_pows[jj];
        let clo = &v - (Q::one() - &elo) / &theta;
        let chi_ = &v - (Q::one() - &ehi) / &theta;
        // constants of the final dot products with the stored lattice laws
        chi_lo[m] = Dir::Dn.constant(q_dn(&clo).max(0.0));
        chi_hi[m] = Dir::Up.constant(q_up(&chi_).max(0.0));
    }
    let (edl, edh) = exp_neg(&(&theta * d), bits);
    let w_lo = d - (Q::one() - &edl) / &theta;
    let w_hi = d - (Q::one() - &edh) / &theta;

    // ---------------- main pass ----------------
    let dirs: Vec<Dir> = if opts.lower {
        vec![Dir::Up, Dir::Dn]
    } else {
        vec![Dir::Up]
    };
    let mut vis: Vec<Option<VData>> = Vec::new();
    let mut tot_up = vec![0.0; len];
    tot_up[0] = 1.0;
    let mut tot_dn = tot_up.clone();
    let mut convs = 0usize;
    let mut r_del = 0.0f64;
    let mut v_add = 0.0f64;
    let mut nmax_max = 0usize;
    let mut skipped = 0usize;
    let mut p = 2 * nn - 1;
    while p >= p_min {
        if p.rem_euclid(2) == 0 {
            // ---------------- visit ----------------
            let c = visit_class(p, i, n);
            let a1 = in_p(p);
            let mut zsum: Iv = (0.0, 0.0);
            let mut lam_mc = Q::zero();
            let mut refs: Vec<(usize, usize)> = Vec::new(); // (class, vidx)
            for k in 0..n {
                if k == c {
                    continue;
                }
                lam_mc += &sys.lam[k];
                let r = next_visit(p, k, i, n);
                if r < 2 * nn {
                    let v = vis[vidx(r)].as_ref().unwrap();
                    zsum = iv_add(zsum, iv_mul(lam_f[k], v.z));
                    refs.push((k, vidx(r)));
                }
            }
            let mu_q = &sys.mu[c];
            let lam_q = &sys.lam[c];
            let nu_q = &nu[c];
            // scalar part in rationals
            let zl = fq(zsum.0.max(0.0));
            let zh = fq(zsum.1);
            let mut l0 = [0.0f64; 2]; // [up, dn]
            let mut den = [0.0f64; 2]; // a=0: den bound for (up, dn); a=1: den1 bound
            let mut u0 = [0.0f64; 2];
            let mut alpha = [0.0f64; 2];
            let z: Iv;
            if !a1 {
                let disc = |zz: &Q| {
                    (mu_q - lam_q) * (mu_q - lam_q) + qi(2) * (mu_q + lam_q) * zz + zz * zz
                };
                let (sl_l, sh_l) = sqrt_enclosure(&disc(&zl), BITS);
                let (sl_h, sh_h) = sqrt_enclosure(&disc(&zh), BITS);
                let b_l = mu_q + lam_q + &zl;
                let b_h = mu_q + lam_q + &zh;
                l0[0] = q_up(&(qi(2) * mu_q / (&b_l + &sl_l)));
                den[0] = q_dn(&sl_l);
                l0[1] = q_dn(&(qi(2) * mu_q / (&b_h + &sh_h)));
                den[1] = q_up(&sh_h);
                let zlo = qi(2) * &zl / ((mu_q - lam_q + &zl) + &sh_l);
                let zhi = qi(2) * &zh / ((mu_q - lam_q + &zh) + &sl_h);
                z = (q_dn(&zlo).max(0.0), q_up(&zhi).min(1.0));
            } else {
                let sp = &exp_sp[c];
                let j0_hi = &lam_mc - &zl;
                let j0_lo = (&lam_mc - &zh).max(Q::zero());
                let solve = |e: &Q, j0: &Q, upper: bool| -> (Q, Q, Q, Q) {
                    let b = Q::one() - e * j0 / nu_q;
                    let dd = &b * &b - qi(4) * e * e * mu_q * lam_q / (nu_q * nu_q);
                    assert!(dd.is_positive(), "discriminant");
                    let (sl, sh) = sqrt_enclosure(&dd, BITS);
                    let s = if upper { sl } else { sh };
                    let lq = qi(2) * e * mu_q / nu_q / (&b + &s);
                    let l = if upper {
                        round_up(&lq, BITS + 32)
                    } else {
                        round_down(&lq, BITS + 32)
                    };
                    let u = (mu_q + j0 * &l + lam_q * &l * &l) / nu_q;
                    let al = (j0 + qi(2) * lam_q * &l) / nu_q;
                    let d1 = Q::one() - e * &al;
                    (l, u, al, d1)
                };
                let (lu, uu, au, du) = solve(&sp.p0q.1, &j0_hi, true);
                let (ld, ud, ad, dd1) = solve(&sp.p0q.0, &j0_lo, false);
                l0 = [q_up(&lu), q_dn(&ld)];
                u0 = [q_up(&uu), q_dn(&ud)];
                alpha = [q_up(&au), q_dn(&ad)];
                den = [q_dn(&du), q_up(&dd1)];
                assert!(den[0] > 0.0);
                z = (
                    q_dn(&(Q::one() - &lu)).max(0.0),
                    q_up(&(Q::one() - &ld)).min(1.0),
                );
            }
            let mut vd = VData {
                up: Vec::new(),
                dn: Vec::new(),
                z,
            };
            for &dir in &dirs {
                let di = if dir == Dir::Up { 0 } else { 1 };
                // J for m >= 1
                let mut jv = vec![0.0; len];
                for &(k, vi) in &refs {
                    let lk = dir.constant(dir.pick(lam_f[k]));
                    let v = vis[vi].as_ref().unwrap();
                    let src = if dir == Dir::Up { &v.up } else { &v.dn };
                    for m in 1..len {
                        jv[m] += lk * src[m];
                    }
                }
                let kk = dir.k(n);
                for m in 1..len {
                    jv[m] = dir.p(jv[m] * kk);
                }
                let kl = dir.k(len);
                let lam_c = dir.constant(dir.pick(lam_f[c]));
                let mut tri = Tri::new(len);
                tri.push(dir.stored(l0[di]));
                if !a1 {
                    let dn_ = dir.divisor(den[di]);
                    for _m in 1..len {
                        let s1 = dir.p(tri.cross(&jv) * kl);
                        let s2 = dir.p(tri.selfconv() * kl);
                        let num = dir.p(s1 + dir.p(lam_c * s2));
                        tri.push(dir.p(num / dn_));
                    }
                } else {
                    let sp = &exp_sp[c];
                    let (e0, cc, r) = (
                        dir.constant(dir.pick(sp.p0)),
                        dir.constant(dir.pick(sp.c)),
                        dir.constant(dir.pick(sp.r)),
                    );
                    let inv_nu = dir.constant(dir.pick(inv_nu_f[c]));
                    let (al, d1) = (dir.constant(alpha[di]), dir.divisor(den[di]));
                    let mut u_prev = u0[di];
                    let mut gm = 0.0f64;
                    for _m in 1..len {
                        gm = dir.p(r * dir.p(u_prev + gm));
                        let s1 = dir.p(tri.cross(&jv) * kl);
                        let s2 = dir.p(tri.selfconv() * kl);
                        let beta = dir.p(inv_nu * dir.p(s1 + dir.p(lam_c * s2)));
                        let lm = dir.p(dir.p(dir.p(e0 * beta) + dir.p(cc * gm)) / d1);
                        tri.push(lm);
                        u_prev = dir.p(dir.p(al * lm) + beta);
                    }
                }
                convs += 2;
                if dir == Dir::Up {
                    vd.up = tri.into_vec();
                } else {
                    vd.dn = tri.into_vec();
                }
            }
            debug_assert_eq!(vis.len(), vidx(p));
            vis.push(Some(vd));
            // free laws no longer referenced (references reach at most N-1 visits back)
            let cur = vis.len() - 1;
            if cur >= n {
                vis[cur - n] = None;
            }
        } else {
            // ---------------- reconfiguration ----------------
            let j = switch_class(p, i, n);
            let a1 = in_p(p);
            let kap = kappa[ridx(p)];
            let mut lp: Iv = (0.0, 0.0);
            let mut refs: Vec<(usize, usize)> = Vec::new();
            for k in 0..n {
                let r = next_visit(p, k, i, n);
                if r < 2 * nn {
                    let v = vis[vidx(r)].as_ref().unwrap();
                    lp = iv_add(lp, iv_mul(lam_f[k], v.z));
                    refs.push((k, vidx(r)));
                }
            }
            let lp = (lp.0.max(0.0), lp.1);
            let pre = pres[j].clone();
            let kap_ma = (
                rd(kap.0 - if a1 { 1.0 } else { 0.0 }).max(0.0),
                ru(kap.1 - if a1 { 1.0 } else { 0.0 }).max(0.0),
            );
            // skip rule (deletion) for reconfigurations outside the period
            let skip = !a1 && lp.0 <= opts.eta_skip * lamtot_f.0;
            if skip {
                skipped += 1;
                r_del += ru(kap.1 * s_f[j].1);
                p -= 1;
                continue;
            }
            // jump-rate measure J~ (m >= 1) per direction
            let mut jt: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
            for &dir in &dirs {
                let di = if dir == Dir::Up { 0 } else { 1 };
                let mut jv = vec![0.0; len];
                for &(k, vi) in &refs {
                    let lk = dir.constant(dir.pick(lam_f[k]));
                    let v = vis[vi].as_ref().unwrap();
                    let src = if dir == Dir::Up { &v.up } else { &v.dn };
                    for m in 1..len {
                        jv[m] += lk * src[m];
                    }
                }
                let kk = dir.k(n);
                for m in 1..len {
                    jv[m] = dir.p(jv[m] * kk);
                }
                jt[di] = jv;
            }
            // Partition laws outside the period: decide the jump cap once.
            let mut part_coef: Option<(Vec<f64>, Vec<f64>, usize)> = None; // (coef_up, coef_dn, nmax)
            let mut jhat: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
            if !a1 && let SwPre::Part { part, .. } = pre.as_ref() {
                // normalized jump laws
                for &dir in &dirs {
                    let di = if dir == Dir::Up { 0 } else { 1 };
                    let lpd = dir.pick_rev(lp); // Up: lam_plus_lo
                    let inv = dir.constant(match dir {
                        Dir::Up => ru(1.0 / lpd),
                        Dir::Dn => rd(1.0 / lpd),
                    });
                    jhat[di] = jt[di].iter().map(|&x| dir.p(x * inv)).collect();
                }
                let dff = q_up(d);
                let (nc, s_best) = chernoff_n(&jhat[0], gf.1, len, dff, opts.chern_eps);
                let nhi = nc.min(len);
                let mp = mixed_poisson(part, lp, nhi + 1);
                let mut nmax = nhi;
                for nn_ in 1..=nhi {
                    if mp.tail[nn_] <= opts.qtail_eps {
                        nmax = nn_;
                        break;
                    }
                }
                let mut cu: Vec<f64> = mp.q_up[..nmax].to_vec();
                let cd: Vec<f64> = mp.q_dn[..nmax].to_vec();
                if nmax < len {
                    cu.push(pu(mp.tail[nmax].min(1.0)));
                    // deleted jumps beyond the cap (lower end only)
                    let mean_j = ru(kap_ma.1 / lp.0.max(1e-300));
                    let b1 = ru(mean_j * mp.en[nmax]);
                    let ch = chernoff_up(&jhat[0], gf, len, dff, s_best, nmax);
                    let b2 = ru(ru(kap_ma.1 * mean_cap[j]) * ch);
                    r_del += b1.min(b2);
                }
                nmax_max = nmax_max.max(nmax);
                part_coef = Some((cu, cd, nmax));
                // cap of tau and spread onto the partition
                r_del += ru(kap.1 * part.tail_mean_hi);
                v_add += ru(ru(kap.1 * kap.1) * part.spread_var_hi);
            }
            if a1 {
                let v = match pre.as_ref() {
                    SwPre::Det { var_hi, .. } => *var_hi,
                    SwPre::Exp { sp, .. } => sp.var_hi,
                    SwPre::Erlang { k, sp, .. } => ru(*k as f64 * sp.var_hi),
                    SwPre::Part { var_hi, .. } => *var_hi,
                };
                v_add += ru(ru(kap.1 * kap.1) * v);
            }
            for &dir in &dirs {
                let di = if dir == Dir::Up { 0 } else { 1 };
                let jv = &jt[di];
                let tp: Vec<f64> = match (pre.as_ref(), a1) {
                    (SwPre::Det { atoms, .. }, true) => {
                        let mut acc = vec![0.0; len];
                        for &(kk_, piv) in atoms {
                            if kk_ >= len {
                                continue;
                            }
                            let x = (rd(kk_ as f64 * gf.0), ru(kk_ as f64 * gf.1));
                            let s = exp_series(jv, x, lp, len, dir, &mut convs);
                            let pi = dir.constant(dir.pick(piv));
                            for m in kk_..len {
                                acc[m] += pi * s[m - kk_];
                            }
                        }
                        let kk = dir.k(2);
                        acc.iter().map(|&x| dir.p(x * kk)).collect()
                    }
                    (SwPre::Det { c, .. }, false) => exp_series(jv, *c, lp, len, dir, &mut convs),
                    (SwPre::Exp { sp, .. }, true) | (SwPre::Erlang { sp, .. }, true) => {
                        let stages = if let SwPre::Erlang { k, .. } = pre.as_ref() {
                            *k
                        } else {
                            1
                        };
                        let eg = exp_series(jv, gf, lp, len, dir, &mut convs);
                        let a = shift(&eg, 1, len);
                        let h = geo_inverse(
                            &a,
                            dir.constant(dir.pick(sp.r)),
                            1.0,
                            len,
                            dir,
                            &mut convs,
                        );
                        let cc = dir.constant(dir.pick(sp.c));
                        let mut t1: Vec<f64> = h.iter().map(|&x| dir.p(cc * x)).collect();
                        t1[0] = dir.stored(dir.pick(sp.p0));
                        let mut t = t1.clone();
                        for _ in 1..stages {
                            t = conv_new(&t, &t1, len, dir);
                            convs += 1;
                        }
                        t
                    }
                    (SwPre::Exp { rate, .. }, false) | (SwPre::Erlang { rate, .. }, false) => {
                        let stages = if let SwPre::Erlang { k, .. } = pre.as_ref() {
                            *k
                        } else {
                            1
                        };
                        // p = rate/(rate + L+), coefficient 1/(rate + L+), at L+ = lam_plus bound
                        let (lpd, rt, rt2) = match dir {
                            Dir::Up => (lp.0, rate.1, rate.0),
                            Dir::Dn => (lp.1, rate.0, rate.1),
                        };
                        let (pp, coef) = match dir {
                            Dir::Up => (ru(rt / rd(rt2 + lpd)), ru(1.0 / rd(rt2 + lpd))),
                            Dir::Dn => (rd(rt / ru(rt2 + lpd)), rd(1.0 / ru(rt2 + lpd))),
                        };
                        let (pp, coef) = (dir.constant(pp), dir.constant(coef));
                        let x: Vec<f64> = jv.iter().map(|&v| dir.p(v * coef)).collect();
                        let mut f0 = 1.0f64;
                        for _ in 0..stages {
                            f0 = dir.p(f0 * pp);
                        }
                        if stages == 1 {
                            geo_inverse(&x, 1.0, f0, len, dir, &mut convs)
                        } else {
                            nb_series(&x, stages, f0, len, dir, &mut convs)
                        }
                    }
                    (SwPre::Part { lat_lo, lat_hi, .. }, true) if lp.1 == 0.0 => {
                        // no jumps can occur: D(tau~) = tau~
                        let src: &[f64] = if dir == Dir::Up {
                            &lat_hi[..len]
                        } else {
                            &lat_lo[..len]
                        };
                        dir.stored_vec(src)
                    }
                    (SwPre::Part { lat_lo, lat_hi, .. }, true) => {
                        let eg = exp_series(jv, gf, lp, len, dir, &mut convs);
                        let a = shift(&eg, 1, len);
                        let src: &[f64] = if dir == Dir::Up {
                            &lat_hi[..len]
                        } else {
                            &lat_lo[..len]
                        };
                        ps_eval(&dir.stored_vec(src), &a, len, true, dir, &mut convs)
                    }
                    (SwPre::Part { .. }, false) => {
                        let (cu, cd, _nmax) = part_coef.as_ref().unwrap();
                        let coef = if dir == Dir::Up { cu } else { cd };
                        ps_eval(coef, &jhat[di], len, true, dir, &mut convs)
                    }
                };
                let tot = if dir == Dir::Up {
                    &mut tot_up
                } else {
                    &mut tot_dn
                };
                let newt = conv_new(tot, &tp, len, dir);
                convs += 1;
                *tot = newt;
            }
        }
        p -= 1;
    }

    // ---------------- variance added by the service spreads ----------------
    for c in 0..n {
        if c == i {
            continue;
        }
        let ev = ru(ru(lam_f[c].1 * ec_hi) * ru(1.0 + ru(lamtot_f.1 / q_dn(&sys.mu[c]))));
        v_add += ru(ev * exp_sp[c].var_hi);
    }

    // ---------------- assembly ----------------
    let su = pu(dot(&tot_up, &chi_hi) * Dir::Up.k(len));
    let l_lo = &w_lo - fq(su);
    let hi_q = if l_lo.is_positive() {
        Q::one() - &l_lo / &ei_q.1
    } else {
        Q::one()
    };
    let mass: f64 = tot_up.iter().sum();
    let (lo_q, lo_comp) = if opts.lower {
        let sd = pd(dot(&tot_dn, &chi_lo) * Dir::Dn.k(len));
        let l_comp = &w_hi - fq(sd);
        let corr = fq(e_rk) + fq(r_del) + &theta / qi(2) * fq(v_add);
        let l_hi = &l_comp + &corr;
        let lo = (Q::one() - &l_hi / &ei_q.0).max(Q::zero());
        let lc = (Q::one() - &l_comp / &ei_q.0).max(Q::zero());
        (lo, q_dn(&lc))
    } else {
        (Q::zero(), 0.0)
    };
    Out {
        target: i,
        m_cells: mcells,
        k_cycles,
        e_i: ei,
        e_rk,
        r_del,
        v_add,
        hi: dec(&hi_q, 12, true),
        lo: dec(&lo_q, 12, false),
        hi_f: q_up(&hi_q),
        lo_f: q_dn(&lo_q),
        lo_comp_f: lo_comp,
        mass_le_d: mass,
        seconds: t0.elapsed().as_secs_f64(),
        convs,
        nmax_max,
        skipped,
        admitted: take_admitted(),
        law_up: tot_up,
        law_dn: tot_dn,
    }
}
