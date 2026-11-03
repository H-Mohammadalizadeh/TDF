//! Certified phase coefficients for continuous reconfiguration-time
//! distributions (laws, in identifiers) on a uniform time grid whose step
//! divides every deadline. The maximum completion probability `H_ji(t)` is
//! enclosed by two recursions evaluated in doubles with outward rounding
//! (see `target_result`), starting from rigorous enclosures of the cell and
//! survival probabilities; the coefficients are converted exactly to
//! rationals and can be cached on disk ([`grid_coefficients_cached`]).

use super::coefficients::{Mode, PhaseCoefficients};
use super::distributions::{Distribution, Model};
use super::rational::{Q, exp_neg, exp_neg_powers, parse_q, qi, rational_gcd, to_f64};
use num_traits::{One, ToPrimitive, Zero};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// Numerical settings of the grid construction.
#[derive(Clone, Debug)]
pub struct GridSettings {
    /// Requested grid step, refined so that every deadline and finite
    /// support point is a multiple.
    pub step: Q,
    /// Fractional bits of the rational enclosures.
    pub bits: u32,
    /// Maximum number of sweeps of the upper bound (at least one).
    pub sweeps: usize,
    /// Use the symmetry of models whose pairs all carry the same distribution
    /// (one row per target instead of `N - 1`); `false` forces the full computation.
    pub shortcut: bool,
    /// Window constraints for every subset of at least two active classes.
    pub all_subsets: bool,
    /// The visit-gap constraints in the full test (they do not change the
    /// coefficients).
    pub gap_rows: bool,
}

impl GridSettings {
    /// The settings for `n` classes: step 1/2000 up to four classes, 1/1000
    /// up to eight and 1/500 beyond; 128 bits, three sweeps, the symmetry
    /// shortcut, the declared window subsets and no visit-gap constraints.
    pub fn default_for(n: usize) -> GridSettings {
        let step = if n <= 4 {
            "1/2000"
        } else if n <= 8 {
            "1/1000"
        } else {
            "1/500"
        };
        GridSettings {
            step: parse_q(step).unwrap(),
            bits: 128,
            sweeps: 3,
            shortcut: true,
            all_subsets: false,
            gap_rows: false,
        }
    }

    /// The settings as JSON; `gap_rows` appears only when set, so that the
    /// records of the test without the visit-gap constraints are unchanged.
    pub fn to_json(&self) -> Value {
        let mut v = json!({"step": self.step.to_string(), "bits": self.bits, "sweeps": self.sweeps, "symmetric_shortcut": self.shortcut, "all_subsets": self.all_subsets});
        if self.gap_rows {
            v["gap_rows"] = json!(true);
        }
        v
    }
}

/// The smallest double not below `q`.
pub fn q_to_f64_up(q: &Q) -> f64 {
    let mut f = to_f64(q);
    if !f.is_finite() {
        return f;
    }
    while Q::from_float(f).is_none_or(|r| r < *q) {
        f = f.next_up();
    }
    f
}

/// The largest double not above `q`.
pub fn q_to_f64_down(q: &Q) -> f64 {
    let mut f = to_f64(q);
    if !f.is_finite() {
        return f;
    }
    while Q::from_float(f).is_none_or(|r| r > *q) {
        f = f.next_down();
    }
    f
}

/// An upper bound on `a * b`: the rounded product raised by one ulp.
#[inline]
fn mul_up(a: f64, b: f64) -> f64 {
    (a * b).next_up()
}

/// A lower bound on `a * b >= 0`: the rounded product lowered by one ulp,
/// clamped at zero.
#[inline]
fn mul_dn(a: f64, b: f64) -> f64 {
    (a * b).next_down().max(0.0)
}

/// An upper bound on `a + b`: the rounded sum raised by one ulp.
#[inline]
fn add_up(a: f64, b: f64) -> f64 {
    (a + b).next_up()
}

/// A lower bound on `a + b >= 0`: the rounded sum lowered by one ulp,
/// clamped at zero.
#[inline]
fn add_dn(a: f64, b: f64) -> f64 {
    (a + b).next_down().max(0.0)
}

/// Outward-rounded doubles of the cell probabilities and survival
/// probabilities of one distribution on the grid.
struct LawTable {
    /// `P(tau in [m g, (m + 1) g))`, `m < cells`.
    p_lo: Vec<f64>,
    p_hi: Vec<f64>,
    /// `P(tau > m g)`, `m <= cells`.
    s_lo: Vec<f64>,
    s_hi: Vec<f64>,
    /// Enclosure of the mean.
    mean: (Q, Q),
    /// Largest width of a rational cell-probability enclosure (diagnostics).
    max_cell_width: f64,
    /// Largest width of a rational survival enclosure (diagnostics).
    max_survival_width: f64,
}

/// Tables computed in this process, shared across models with the same
/// distribution, step, cell count and precision (a distribution recurs for
/// every `N` and load); the cache is cleared when it exceeds 512 entries.
fn table_cache() -> &'static Mutex<HashMap<String, Arc<LawTable>>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<LawTable>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The table of `law` on `cells` cells, from the process cache when present.
fn law_table_cached(law: &Distribution, step: &Q, cells: usize, bits: u32) -> Arc<LawTable> {
    let key = format!("{}|{}|{}|{}", law.id(), step, cells, bits);
    if let Some(t) = table_cache().lock().unwrap().get(&key) {
        return t.clone();
    }
    let t = Arc::new(law_table(law, step, cells, bits));
    let mut cache = table_cache().lock().unwrap();
    if cache.len() > 512 {
        cache.clear();
    }
    cache.insert(key, t.clone());
    t
}

/// The cell and survival probabilities of `law` on `cells` grid cells: the
/// rational enclosures at the grid points, differenced outward for the cells
/// and converted outward to doubles.
fn law_table(law: &Distribution, step: &Q, cells: usize, bits: u32) -> LawTable {
    let prepared = law.prepare(bits);
    let points: Vec<((Q, Q), (Q, Q))> = (0..=cells)
        .into_par_iter()
        .map(|m| {
            let x = step * qi(m as i64);
            prepared.both(&x)
        })
        .collect();
    let mut p_lo = Vec::with_capacity(cells);
    let mut p_hi = Vec::with_capacity(cells);
    let mut max_cell_width: f64 = 0.0;
    for m in 0..cells {
        let lo = (&points[m + 1].0.0 - &points[m].0.1).max(Q::zero());
        let hi = (&points[m + 1].0.1 - &points[m].0.0).min(Q::one());
        max_cell_width = max_cell_width.max(to_f64(&(&hi - &lo)));
        p_lo.push(q_to_f64_down(&lo));
        p_hi.push(q_to_f64_up(&hi));
    }
    let mut max_survival_width: f64 = 0.0;
    let s_lo: Vec<f64> = points.iter().map(|p| q_to_f64_down(&p.1.0)).collect();
    let s_hi: Vec<f64> = points.iter().map(|p| q_to_f64_up(&p.1.1)).collect();
    for p in &points {
        max_survival_width = max_survival_width.max(to_f64(&(&p.1.1 - &p.1.0)));
    }
    LawTable {
        p_lo,
        p_hi,
        s_lo,
        s_hi,
        mean: law.mean(bits),
        max_cell_width,
        max_survival_width,
    }
}

/// Result for one target class `i`: lower and upper enclosures of `A_.i`,
/// `B_.i`, `C_..i` and the service-only constant, with diagnostics.
struct TargetResult {
    a_lo: Vec<Q>,
    a_hi: Vec<Q>,
    b_lo: Vec<Q>,
    b_hi: Vec<Q>,
    c_lo: Vec<Vec<Q>>,
    c_hi: Vec<Vec<Q>>,
    service_lo: Q,
    service_hi: Q,
    diag: Value,
}

/// `max(0, 1 - f)` as an exact rational.
fn one_minus(f: f64) -> Q {
    (Q::one() - Q::from_float(f).expect("finite double")).max(Q::zero())
}

/// The enclosure of `H_ji` for the target class `i` on the grid `t = n g`,
/// and the coefficients derived from it. `H` is enclosed by two recursions:
///
/// * an upper bound `U`, started at one and swept at most `sweeps` times
///   (stopping early when a sweep lowers no value). A reconfiguration whose
///   duration lies in the cell `[m g, (m + 1) g)` is charged the left end
///   `m g`; the cell `[0, g)`, which does not reduce the remaining time,
///   uses the previous sweep's value. The discretized operator is monotone
///   and dominates the exact one, of which `H` is a fixed point, so every
///   sweep stays above `H`;
/// * a lower bound `L` in one pass, charging every cell its right end.
///
/// Every double operation is rounded outward. Nothing is truncated: both
/// enclosures of `C` integrate over `[0, d_i]` only and take the part beyond
/// the deadline from the full mean, since no completion is possible after it.
#[allow(clippy::too_many_arguments)]
fn target_result(
    model: &Model,
    tables: &[Arc<LawTable>],
    law_of: &[Vec<usize>],
    symmetric: bool,
    step: &Q,
    cells_of: &[usize],
    settings: &GridSettings,
    i: usize,
) -> TargetResult {
    let n = model.n;
    let mi = cells_of[i];
    let bits = settings.bits;
    // H_ii(n g) = 1 - exp(-mu_i n g)
    let (lo_pow, hi_pow) = exp_neg_powers(&(&model.mu[i] * step), mi, bits);
    let hii_up: Vec<f64> = (0..=mi)
        .map(|k| q_to_f64_up(&(Q::one() - &lo_pow[k])).min(1.0))
        .collect();
    let hii_dn: Vec<f64> = (0..=mi)
        .map(|k| q_to_f64_down(&(Q::one() - &hi_pow[k])).max(0.0))
        .collect();
    // Representatives: with identical distributions on every pair, H_ji is
    // the same for every j != i, so one row is computed and broadcast.
    let rows: Vec<usize> = if symmetric {
        vec![(i + 1) % n]
    } else {
        (0..n).filter(|j| *j != i).collect()
    };
    let hops = |j: usize| -> Vec<usize> {
        if symmetric {
            // a reconfiguration directly to i and, for N >= 3, one to another
            // non-target configuration, whose row equals the representative's own row
            let mut h = vec![i];
            if n >= 3 {
                h.push(j);
            }
            h
        } else {
            (0..n).filter(|k| *k != j).collect()
        }
    };
    let table_index = |j: usize, k: usize| -> usize {
        if symmetric {
            law_of[j][if k == j { i } else { k }]
        } else {
            law_of[j][k]
        }
    };
    // u[k][b] and l[k][b] bound H_ki(b g); in the symmetric case the
    // representative's row stands in for every non-target row
    let mut u: Vec<Vec<f64>> = vec![vec![1.0; mi + 1]; n];
    let mut l: Vec<Vec<f64>> = vec![vec![0.0; mi + 1]; n];
    u[i] = hii_up.clone();
    l[i] = hii_dn.clone();
    for &j in &rows {
        u[j][0] = 0.0;
    }
    // Lower bound: one pass, right cell ends.
    for b in 1..=mi {
        for &j in &rows {
            let mut best = 0.0f64;
            for k in hops(j) {
                let t = &tables[table_index(j, k)];
                let src = if symmetric && k == j { j } else { k };
                let mut acc = 0.0f64;
                for m in 0..b.saturating_sub(1) {
                    acc = add_dn(acc, mul_dn(t.p_lo[m], l[src][b - m - 1]));
                }
                best = best.max(acc);
            }
            l[j][b] = best.min(1.0);
        }
    }
    // Upper bound: sweeps from one, left cell ends, cell [0, g) from the previous sweep.
    let mut sweeps_done = 0;
    let mut last_change = f64::INFINITY;
    for _ in 0..settings.sweeps.max(1) {
        let prev = u.clone();
        let mut change = 0.0f64;
        for b in 1..=mi {
            for &j in &rows {
                let mut best = 0.0f64;
                for k in hops(j) {
                    let t = &tables[table_index(j, k)];
                    let src = if symmetric && k == j { j } else { k };
                    let mut acc = mul_up(t.p_hi[0], prev[src][b]);
                    for m in 1..b {
                        acc = add_up(acc, mul_up(t.p_hi[m], u[src][b - m]));
                    }
                    best = best.max(acc);
                }
                let v = best.min(1.0);
                change = change.max(prev[j][b] - v);
                u[j][b] = v;
            }
        }
        sweeps_done += 1;
        last_change = change;
        if change <= 0.0 {
            break;
        }
    }
    if symmetric {
        let r = rows[0];
        for j in 0..n {
            if j != i && j != r {
                u[j] = u[r].clone();
                l[j] = l[r].clone();
            }
        }
    }
    let mut max_h_width: f64 = 0.0;
    for j in 0..n {
        for b in 0..=mi {
            debug_assert!(l[j][b] <= u[j][b] + 1e-12, "H enclosure order");
            max_h_width = max_h_width.max(u[j][b] - l[j][b]);
        }
    }
    // Coefficients.
    let mut a_lo = vec![Q::zero(); n];
    let mut a_hi = vec![Q::zero(); n];
    let mut b_lo = vec![Q::zero(); n];
    let mut b_hi = vec![Q::zero(); n];
    for j in 0..n {
        let (lo_p, hi_p) = exp_neg_powers(&(&model.mu[j] * step), mi + 1, bits);
        let mut sum_up = 0.0f64;
        let mut sum_dn = 0.0f64;
        for m in 0..mi {
            let r_hi = q_to_f64_up(&(&hi_p[m] - &lo_p[m + 1]).max(Q::zero()));
            let r_lo = q_to_f64_down(&(&lo_p[m] - &hi_p[m + 1]).max(Q::zero()));
            sum_up = add_up(sum_up, mul_up(r_hi, u[j][mi - m]));
            sum_dn = add_dn(sum_dn, mul_dn(r_lo, l[j][mi - m - 1]));
        }
        a_lo[j] = one_minus(sum_up);
        a_hi[j] = one_minus(sum_dn);
        b_lo[j] = one_minus(u[j][mi]);
        b_hi[j] = one_minus(l[j][mi]);
    }
    // C = int_0^inf P(tau > r)(1 - H(d - r)) dr has two enclosures, and each end keeps the
    // tighter one. The full form subtracts the integral of P(tau > r) H(d - r) over [0, d] from
    // the mean: it is tight when the support points lie on the grid, but for a continuous
    // distribution its cell-end error, of order g, can exceed C itself when reconfiguration is
    // short. The cell form integrates over [0, d] cell by cell with nonnegative terms and adds
    // the part beyond the deadline, E[(tau - d)^+], enclosed as the mean minus the integral of
    // P(tau > r) over [0, d].
    let mut c_lo = vec![vec![Q::zero(); n]; n];
    let mut c_hi = vec![vec![Q::zero(); n]; n];
    for (uu, v) in model.edges() {
        let t = &tables[law_of[uu][v]];
        let (mut sum_up, mut sum_dn) = (0.0f64, 0.0f64);
        let (mut cell_dn, mut cell_up) = (0.0f64, 0.0f64);
        let (mut head_up, mut head_dn) = (0.0f64, 0.0f64);
        for m in 0..mi {
            sum_up = add_up(sum_up, mul_up(t.s_hi[m], u[v][mi - m]));
            sum_dn = add_dn(sum_dn, mul_dn(t.s_lo[m + 1], l[v][mi - m - 1]));
            // on [m g, (m + 1) g): P(tau > r) in [s(m + 1), s(m)], 1 - H(d - r) in [1 - U, 1 - L]
            let miss_dn = (1.0 - u[v][mi - m]).next_down().max(0.0);
            let miss_up = (1.0 - l[v][mi - m - 1]).next_up().min(1.0);
            cell_dn = add_dn(cell_dn, mul_dn(t.s_lo[m + 1], miss_dn));
            cell_up = add_up(cell_up, mul_up(t.s_hi[m], miss_up));
            head_up = add_up(head_up, t.s_hi[m]);
            head_dn = add_dn(head_dn, t.s_lo[m + 1]);
        }
        let times_step = |x: f64| step * Q::from_float(x).expect("finite");
        let full_lo = &t.mean.0 - times_step(sum_up);
        let full_hi = &t.mean.1 - times_step(sum_dn);
        let tail_lo = (&t.mean.0 - times_step(head_up)).max(Q::zero());
        let tail_hi = (&t.mean.1 - times_step(head_dn)).max(Q::zero());
        let cell_lo = times_step(cell_dn) + tail_lo;
        let cell_hi = times_step(cell_up) + tail_hi;
        c_lo[uu][v] = full_lo.max(cell_lo).max(Q::zero());
        c_hi[uu][v] = full_hi.min(cell_hi).max(Q::zero());
    }
    let (service_lo, service_hi) = exp_neg(&(&model.mu[i] * &model.d[i]), bits);
    let width = |lo: &Q, hi: &Q| to_f64(&(hi - lo));
    let max_a = (0..n)
        .map(|j| width(&a_lo[j], &a_hi[j]))
        .fold(0.0, f64::max);
    let max_b = (0..n)
        .map(|j| width(&b_lo[j], &b_hi[j]))
        .fold(0.0, f64::max);
    let max_c = model
        .edges()
        .iter()
        .map(|(uu, v)| width(&c_lo[*uu][*v], &c_hi[*uu][*v]))
        .fold(0.0, f64::max);
    let diag = json!({
        "target": i, "cells": mi, "sweeps": sweeps_done, "last_sweep_change": last_change,
        "max_h_width": max_h_width, "max_a_width": max_a, "max_b_width": max_b, "max_c_width": max_c,
        "symmetric_shortcut": symmetric
    });
    TargetResult {
        a_lo,
        a_hi,
        b_lo,
        b_hi,
        c_lo,
        c_hi,
        service_lo,
        service_hi,
        diag,
    }
}

/// The grid step `g0 / ceil(g0 / target)`, where `g0` is the rational gcd of
/// the deadlines and the finite support points: the largest step of the form
/// `g0 / m` not above `target`.
pub fn grid_step(model: &Model, target: &Q) -> Result<Q, String> {
    let mut values: Vec<Q> = model.d.clone();
    for (u, v) in model.edges() {
        values.extend(model.law(u, v).support_points());
    }
    let g0 = rational_gcd(&values);
    if g0.is_zero() {
        return Err("empty grid".into());
    }
    let m = (&g0 / target).ceil().to_integer();
    let m: i64 = m.to_i64().ok_or("grid refinement too large")?;
    Ok(&g0 / qi(m.max(1)))
}

/// Both coefficient directions (`Mode::Upper`: lower coefficients, the
/// certificate direction; `Mode::Lower`: upper coefficients) and the
/// diagnostics of the construction. Means that are only enclosed keep both
/// ends, so that the relaxation can replace the time-budget equality by two
/// inequalities.
pub fn grid_coefficients(
    model: &Model,
    settings: &GridSettings,
) -> Result<(PhaseCoefficients, PhaseCoefficients, Value), String> {
    let t0 = Instant::now();
    let n = model.n;
    let step = grid_step(model, &settings.step)?;
    let cells_of: Vec<usize> = model
        .d
        .iter()
        .map(|d| {
            let r = d / &step;
            r.to_integer().to_usize().expect("cells")
        })
        .collect();
    let cells = *cells_of.iter().max().unwrap();
    // distinct distributions
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut laws: Vec<&Distribution> = Vec::new();
    let mut law_of = vec![vec![usize::MAX; n]; n];
    for (u, v) in model.edges() {
        let law = model.law(u, v);
        let id = law.id();
        let k = *index.entry(id).or_insert_with(|| {
            laws.push(law);
            laws.len() - 1
        });
        law_of[u][v] = k;
    }
    let symmetric = settings.shortcut && laws.len() == 1;
    let tables: Vec<Arc<LawTable>> = laws
        .iter()
        .map(|law| law_table_cached(law, &step, cells, settings.bits))
        .collect();
    let table_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = Instant::now();
    let results: Vec<TargetResult> = (0..n)
        .into_par_iter()
        .map(|i| {
            target_result(
                model, &tables, &law_of, symmetric, &step, &cells_of, settings, i,
            )
        })
        .collect();
    let sweep_ms = t1.elapsed().as_secs_f64() * 1e3;
    let mut mean_lo = vec![vec![Q::zero(); n]; n];
    let mut mean_hi = vec![vec![Q::zero(); n]; n];
    for (u, v) in model.edges() {
        let t = &tables[law_of[u][v]];
        mean_lo[u][v] = t.mean.0.clone();
        mean_hi[u][v] = t.mean.1.clone();
    }
    let assemble = |mode: Mode| -> PhaseCoefficients {
        let mut a = vec![vec![Q::zero(); n]; n];
        let mut b = vec![vec![Q::zero(); n]; n];
        let mut c = vec![vec![vec![Q::zero(); n]; n]; n];
        let mut service_only = vec![Q::zero(); n];
        for (i, r) in results.iter().enumerate() {
            for j in 0..n {
                a[j][i] = if mode == Mode::Upper {
                    r.a_lo[j].clone()
                } else {
                    r.a_hi[j].clone()
                };
                b[j][i] = if mode == Mode::Upper {
                    r.b_lo[j].clone()
                } else {
                    r.b_hi[j].clone()
                };
                for v in 0..n {
                    c[j][v][i] = if mode == Mode::Upper {
                        r.c_lo[j][v].clone()
                    } else {
                        r.c_hi[j][v].clone()
                    };
                }
            }
            service_only[i] = if mode == Mode::Upper {
                r.service_lo.clone()
            } else {
                r.service_hi.clone()
            };
        }
        PhaseCoefficients {
            mode,
            bits: settings.bits,
            g: step.clone(),
            a,
            b,
            c,
            service_only,
            mean_tau: mean_lo.clone(),
            mean_tau_hi: mean_hi.clone(),
        }
    };
    let lower = assemble(Mode::Upper);
    let upper = assemble(Mode::Lower);
    let diagnostics = json!({
        "step": step.to_string(), "cells": cells, "bits": settings.bits, "sweeps_requested": settings.sweeps,
        "distinct_laws": laws.iter().map(|l| l.id()).collect::<Vec<_>>(), "symmetric_shortcut": symmetric,
        "law_tables": tables.iter().zip(laws.iter()).map(|(t, l)| json!({"law": l.id(), "max_cell_width": t.max_cell_width, "max_survival_width": t.max_survival_width, "mean": [to_f64(&t.mean.0), to_f64(&t.mean.1)], "mean_width": to_f64(&(&t.mean.1 - &t.mean.0))})).collect::<Vec<_>>(),
        "targets": results.iter().map(|r| r.diag.clone()).collect::<Vec<_>>(),
        "timing_ms": {"law_tables": table_ms, "sweeps_and_coefficients": sweep_ms, "total": t0.elapsed().as_secs_f64() * 1e3}
    });
    Ok((lower, upper, diagnostics))
}

/// Cache key of the grid coefficients: everything they depend on (service
/// rates, deadlines, distributions, step, bits and sweeps, but not the
/// arrival rates).
pub fn grid_coefficient_key(model: &Model, settings: &GridSettings) -> String {
    let mut edges = serde_json::Map::new();
    for (u, v) in model.edges() {
        edges.insert(format!("{u},{v}"), model.law(u, v).to_json());
    }
    let v = json!({
        "version": 2, "grid": true,
        "mu": model.mu.iter().map(|x| x.to_string()).collect::<Vec<_>>(),
        "d": model.d.iter().map(|x| x.to_string()).collect::<Vec<_>>(),
        "edges": edges, "step": settings.step.to_string(), "bits": settings.bits, "sweeps": settings.sweeps
    });
    crate::digest::digest(&v)
}

/// Grid coefficients from the cache directory when present, otherwise
/// computed and stored. A cached file is used only if both directions parse,
/// the lower coefficients are in the certificate direction and the precision
/// matches. Returns lower, upper, diagnostics and the cache hit flag.
pub fn grid_coefficients_cached(
    model: &Model,
    settings: &GridSettings,
    cache: Option<&std::path::Path>,
) -> Result<(PhaseCoefficients, PhaseCoefficients, Value, bool), String> {
    let key = grid_coefficient_key(model, settings);
    if let Some(dir) = cache {
        let path = dir.join(format!("{key}.json"));
        if let Ok(text) = std::fs::read_to_string(&path)
            && let Ok(v) = serde_json::from_str::<Value>(&text)
            && let (Ok(lo), Ok(hi)) = (
                PhaseCoefficients::from_exact_json(&v["lower"]),
                PhaseCoefficients::from_exact_json(&v["upper"]),
            )
            && lo.mode == Mode::Upper
            && lo.bits == settings.bits
        {
            return Ok((lo, hi, v["diagnostics"].clone(), true));
        }
    }
    let (lo, hi, diag) = grid_coefficients(model, settings)?;
    if let Some(dir) = cache {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let v = json!({"key": key, "model_name": model.name, "settings": settings.to_json(), "laws": model.describe_laws(settings.bits),
                       "lower": lo.to_exact_json(), "upper": hi.to_exact_json(), "diagnostics": diag});
        std::fs::write(
            dir.join(format!("{key}.json")),
            serde_json::to_string(&v).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
    }
    Ok((lo, hi, diag, false))
}

#[cfg(test)]
mod tests {
    use super::super::distributions::profile;
    use super::super::rational::qr;
    use super::*;

    #[test]
    fn directed_double_conversion_brackets_the_rational() {
        let q = qr(1, 3);
        let up = q_to_f64_up(&q);
        let dn = q_to_f64_down(&q);
        assert!(Q::from_float(dn).unwrap() <= q && q <= Q::from_float(up).unwrap());
        assert!(up - dn <= 2.0 * f64::EPSILON);
    }

    #[test]
    fn symmetric_shortcut_matches_the_full_computation() {
        let law = profile("gamma", &qr(1, 2), &qr(1, 3)).unwrap();
        let m = Model::balanced("sym", &law, 3, &qr(1, 2), &qi(2), &qr(1, 100));
        let fast = GridSettings {
            step: parse_q("1/100").unwrap(),
            bits: 64,
            sweeps: 2,
            shortcut: true,
            all_subsets: false,
            gap_rows: false,
        };
        let full = GridSettings {
            shortcut: false,
            ..fast.clone()
        };
        let (lo1, hi1, d1) = grid_coefficients(&m, &fast).unwrap();
        let (lo2, hi2, d2) = grid_coefficients(&m, &full).unwrap();
        assert!(d1["symmetric_shortcut"].as_bool().unwrap());
        assert!(!d2["symmetric_shortcut"].as_bool().unwrap());
        assert_eq!(lo1.a, lo2.a);
        assert_eq!(lo1.b, lo2.b);
        assert_eq!(lo1.c, lo2.c);
        assert_eq!(hi1.a, hi2.a);
        assert_eq!(hi1.c, hi2.c);
    }
}
