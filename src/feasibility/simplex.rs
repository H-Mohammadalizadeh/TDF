//! Linear programs with exact certificates. A dense two-phase tableau
//! simplex, generic over the scalar, proposes a basis in double precision
//! and serves as the exact fallback in rationals. Every result is verified
//! against the original rational data: a Farkas witness for an empty
//! feasible set, a dual-feasible multiplier vector whose objective
//! lower-bounds the minimum, and a primal point checked row by row.
//! Floating-point error can leave a program numerically inconclusive but
//! cannot produce a wrong certificate.

use super::rational::{Q, dec, to_f64};
use num_traits::{One, Signed, Zero};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::time::Instant;

/// The sense of a row: `=`, `>=` or `<=`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sense {
    Eq,
    Ge,
    Le,
}

/// One row `sum_j coef_j y_j (sense) rhs`, with a label for reports.
#[derive(Clone, Debug)]
pub struct Row {
    pub coef: Vec<(usize, Q)>,
    pub sense: Sense,
    pub rhs: Q,
    pub label: String,
}

/// `min c'y` subject to the rows and `y >= 0`.
#[derive(Clone, Debug, Default)]
pub struct Lp {
    pub nvars: usize,
    pub names: Vec<String>,
    pub rows: Vec<Row>,
    pub objective: Vec<(usize, Q)>,
}

impl Lp {
    /// Add a variable and return its column index.
    pub fn var(&mut self, name: &str) -> usize {
        self.names.push(name.into());
        self.nvars += 1;
        self.nvars - 1
    }

    /// Add a row; repeated columns are summed and zero coefficients dropped.
    pub fn row(&mut self, coef: Vec<(usize, Q)>, sense: Sense, rhs: Q, label: impl Into<String>) {
        let mut merged: HashMap<usize, Q> = HashMap::new();
        for (j, c) in coef {
            *merged.entry(j).or_insert_with(Q::zero) += c;
        }
        let mut coef: Vec<(usize, Q)> = merged.into_iter().filter(|(_, c)| !c.is_zero()).collect();
        coef.sort_by_key(|(j, _)| *j);
        self.rows.push(Row {
            coef,
            sense,
            rhs,
            label: label.into(),
        });
    }

    /// The objective coefficient of column `j`.
    pub fn cost(&self, j: usize) -> Q {
        self.objective
            .iter()
            .find(|(k, _)| *k == j)
            .map(|(_, c)| c.clone())
            .unwrap_or_else(Q::zero)
    }
}

/// Arithmetic shared by the floating and the exact tableau.
pub trait Scalar: Clone + PartialOrd + std::fmt::Debug {
    fn s_zero() -> Self;
    fn s_one() -> Self;
    fn from_q(q: &Q) -> Self;
    fn s_add(&self, o: &Self) -> Self;
    fn s_sub(&self, o: &Self) -> Self;
    fn s_mul(&self, o: &Self) -> Self;
    fn s_div(&self, o: &Self) -> Self;
    fn s_neg(&self) -> Self;
    fn s_is_zero(&self) -> bool;
    fn s_is_neg(&self) -> bool;
    fn s_is_pos(&self) -> bool;
}

/// Zero tolerance of the floating tableau.
const F64_TOL: f64 = 1e-9;

impl Scalar for f64 {
    fn s_zero() -> Self {
        0.0
    }
    fn s_one() -> Self {
        1.0
    }
    fn from_q(q: &Q) -> Self {
        to_f64(q)
    }
    fn s_add(&self, o: &Self) -> Self {
        self + o
    }
    fn s_sub(&self, o: &Self) -> Self {
        self - o
    }
    fn s_mul(&self, o: &Self) -> Self {
        self * o
    }
    fn s_div(&self, o: &Self) -> Self {
        self / o
    }
    fn s_neg(&self) -> Self {
        -self
    }
    fn s_is_zero(&self) -> bool {
        self.abs() <= F64_TOL
    }
    fn s_is_neg(&self) -> bool {
        *self < -F64_TOL
    }
    fn s_is_pos(&self) -> bool {
        *self > F64_TOL
    }
}

impl Scalar for Q {
    fn s_zero() -> Self {
        Zero::zero()
    }
    fn s_one() -> Self {
        One::one()
    }
    fn from_q(q: &Q) -> Self {
        q.clone()
    }
    fn s_add(&self, o: &Self) -> Self {
        self + o
    }
    fn s_sub(&self, o: &Self) -> Self {
        self - o
    }
    fn s_mul(&self, o: &Self) -> Self {
        self * o
    }
    fn s_div(&self, o: &Self) -> Self {
        self / o
    }
    fn s_neg(&self) -> Self {
        -self
    }
    fn s_is_zero(&self) -> bool {
        Zero::is_zero(self)
    }
    fn s_is_neg(&self) -> bool {
        self.is_negative()
    }
    fn s_is_pos(&self) -> bool {
        self.is_positive()
    }
}

/// Termination status of a simplex run.
#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Optimal,
    Infeasible,
    Unbounded,
    PivotLimit,
}

/// The final state of a simplex run.
#[derive(Clone, Debug)]
pub struct Solved<S: Scalar> {
    pub status: Status,
    /// Objective value (phase-1 infeasibility measure when infeasible).
    pub objective: S,
    /// Structural variables of the final basic solution.
    pub y: Vec<S>,
    /// Row multipliers of the final tableau (phase 1 when infeasible), in the
    /// original row orientation (`w_r = sigma_r pi_r`).
    pub duals: Vec<S>,
    pub pivots: usize,
    /// Basic column of every row.
    pub basis: Vec<usize>,
    /// Row orientation: `-1` for a row negated to a nonnegative right-hand
    /// side, `1` otherwise.
    pub sigma: Vec<i8>,
    /// Number of slack columns.
    pub nslack: usize,
    /// Structural part of an improving ray when unbounded.
    pub ray: Option<Vec<S>>,
}

/// A dense tableau: `m` rows over `ncols` columns plus the right-hand side,
/// the reduced-cost row (its last entry is the negated objective value), the
/// basic column of every row and the artificial column of every row.
struct Tableau<S: Scalar> {
    m: usize,
    ncols: usize,
    t: Vec<Vec<S>>,
    obj: Vec<S>,
    basis: Vec<usize>,
    art: Vec<usize>,
    pivots: usize,
}

impl<S: Scalar> Tableau<S> {
    /// Pivot on row `r` and column `j`.
    fn pivot(&mut self, r: usize, j: usize) {
        let p = self.t[r][j].clone();
        let row: Vec<S> = self.t[r].iter().map(|v| v.s_div(&p)).collect();
        self.t[r] = row.clone();
        for i in 0..self.m {
            if i != r && !self.t[i][j].s_is_zero() {
                let f = self.t[i][j].clone();
                for k in 0..=self.ncols {
                    if !row[k].s_is_zero() {
                        self.t[i][k] = self.t[i][k].s_sub(&f.s_mul(&row[k]));
                    }
                }
                self.t[i][j] = S::s_zero();
            }
        }
        if !self.obj[j].s_is_zero() {
            let f = self.obj[j].clone();
            for k in 0..=self.ncols {
                if !row[k].s_is_zero() {
                    self.obj[k] = self.obj[k].s_sub(&f.s_mul(&row[k]));
                }
            }
            self.obj[j] = S::s_zero();
        }
        self.basis[r] = j;
        self.pivots += 1;
    }

    /// Primal simplex iterations until optimality, unboundedness or the pivot
    /// limit. Entering column: the most negative reduced cost (Dantzig), or
    /// the lowest index with a negative reduced cost (Bland) after more than
    /// 25 consecutive degenerate pivots. Leaving row: the minimum ratio, ties
    /// broken by the lowest basic index. `allowed(j)` restricts the entering
    /// columns.
    fn run(
        &mut self,
        allowed: impl Fn(usize) -> bool,
        max_pivots: usize,
    ) -> (Status, Option<usize>) {
        let mut degenerate_streak = 0usize;
        loop {
            if self.pivots > max_pivots {
                return (Status::PivotLimit, None);
            }
            // Bland's rule cannot cycle, so every run of degenerate pivots ends;
            // every nondegenerate pivot strictly improves the objective, so in
            // exact arithmetic the iteration terminates.
            let entering = if degenerate_streak > 25 {
                (0..self.ncols).find(|j| allowed(*j) && self.obj[*j].s_is_neg())
            } else {
                let mut best: Option<usize> = None;
                for j in 0..self.ncols {
                    if allowed(j)
                        && self.obj[j].s_is_neg()
                        && best.is_none_or(|b| self.obj[j] < self.obj[b])
                    {
                        best = Some(j);
                    }
                }
                best
            };
            let Some(j) = entering else {
                return (Status::Optimal, None);
            };
            let mut leave: Option<(usize, S)> = None;
            for r in 0..self.m {
                if self.t[r][j].s_is_pos() {
                    let ratio = self.t[r][self.ncols].s_div(&self.t[r][j]);
                    let better = match &leave {
                        None => true,
                        Some((lr, lratio)) => {
                            ratio < *lratio || (ratio == *lratio && self.basis[r] < self.basis[*lr])
                        }
                    };
                    if better {
                        leave = Some((r, ratio));
                    }
                }
            }
            let Some((r, ratio)) = leave else {
                return (Status::Unbounded, Some(j));
            };
            if ratio.s_is_zero() {
                degenerate_streak += 1;
            } else {
                degenerate_streak = 0;
            }
            self.pivot(r, j);
        }
    }

    /// The improving ray of an unbounded tableau: the entering column at one,
    /// the basic columns at minus their tableau entries.
    fn ray(&self, entering: usize) -> Vec<S> {
        let mut d = vec![S::s_zero(); self.ncols];
        d[entering] = S::s_one();
        for r in 0..self.m {
            d[self.basis[r]] = self.t[r][entering].s_neg();
        }
        d
    }
}

/// Two-phase simplex on the standard form of `lp`: a slack column per
/// inequality row, rows negated to a nonnegative right-hand side, and one
/// artificial column per row. Phase 1 minimizes the sum of the artificials;
/// phase 2 minimizes the objective with the artificials barred from entering.
pub fn solve<S: Scalar>(lp: &Lp, max_pivots: usize) -> Solved<S> {
    let m = lp.rows.len();
    let n = lp.nvars;
    let mut slack: Vec<Option<usize>> = vec![None; m];
    let mut ncols = n;
    for (r, row) in lp.rows.iter().enumerate() {
        if row.sense != Sense::Eq {
            slack[r] = Some(ncols);
            ncols += 1;
        }
    }
    let nslack = ncols - n;
    let art: Vec<usize> = (0..m).map(|r| ncols + r).collect();
    ncols += m;
    let mut t: Vec<Vec<S>> = vec![vec![S::s_zero(); ncols + 1]; m];
    let mut sigma = vec![1i8; m];
    for (r, row) in lp.rows.iter().enumerate() {
        for (j, c) in &row.coef {
            t[r][*j] = t[r][*j].s_add(&S::from_q(c));
        }
        if let Some(s) = slack[r] {
            t[r][s] = match row.sense {
                Sense::Ge => S::s_one().s_neg(),
                Sense::Le => S::s_one(),
                Sense::Eq => S::s_zero(),
            };
        }
        t[r][ncols] = S::from_q(&row.rhs);
        if t[r][ncols].s_is_neg() {
            sigma[r] = -1;
            for v in t[r].iter_mut() {
                *v = v.s_neg();
            }
        }
        t[r][art[r]] = S::s_one();
    }
    let mut obj = vec![S::s_zero(); ncols + 1];
    for j in 0..=ncols {
        if j >= ncols - m && j < ncols {
            continue;
        }
        let mut s = S::s_zero();
        for row in t.iter().take(m) {
            s = s.s_add(&row[j]);
        }
        obj[j] = s.s_neg();
    }
    let mut tab = Tableau {
        m,
        ncols,
        t,
        obj,
        basis: art.clone(),
        art: art.clone(),
        pivots: 0,
    };
    let (status, _) = tab.run(|_| true, max_pivots);
    let infeasibility = tab.obj[ncols].s_neg();
    let duals_of = |tab: &Tableau<S>, art_cost: &S| -> Vec<S> {
        (0..m)
            .map(|r| {
                let pi = art_cost.s_sub(&tab.obj[tab.art[r]]);
                if sigma[r] < 0 { pi.s_neg() } else { pi }
            })
            .collect()
    };
    let solution = |tab: &Tableau<S>| -> Vec<S> {
        let mut y = vec![S::s_zero(); n];
        for r in 0..m {
            if tab.basis[r] < n {
                y[tab.basis[r]] = tab.t[r][ncols].clone();
            }
        }
        y
    };
    if status == Status::PivotLimit {
        return Solved {
            status,
            objective: infeasibility,
            y: solution(&tab),
            duals: duals_of(&tab, &S::s_one()),
            pivots: tab.pivots,
            basis: tab.basis.clone(),
            sigma,
            nslack,
            ray: None,
        };
    }
    if infeasibility.s_is_pos() {
        return Solved {
            status: Status::Infeasible,
            objective: infeasibility,
            y: solution(&tab),
            duals: duals_of(&tab, &S::s_one()),
            pivots: tab.pivots,
            basis: tab.basis.clone(),
            sigma,
            nslack,
            ray: None,
        };
    }
    // Drive artificial columns out of the basis where possible.
    for r in 0..m {
        if tab.basis[r] >= ncols - m
            && let Some(j) = (0..ncols - m).find(|j| !tab.t[r][*j].s_is_zero())
        {
            tab.pivot(r, j);
        }
    }
    // Phase 2 objective row.
    let cost: Vec<S> = (0..ncols)
        .map(|j| {
            if j < n {
                S::from_q(&lp.cost(j))
            } else {
                S::s_zero()
            }
        })
        .collect();
    let mut obj = vec![S::s_zero(); ncols + 1];
    for j in 0..=ncols {
        let mut s = if j < ncols {
            cost[j].clone()
        } else {
            S::s_zero()
        };
        for r in 0..m {
            let cb = &cost[tab.basis[r]];
            if !cb.s_is_zero() {
                s = s.s_sub(&cb.s_mul(&tab.t[r][j]));
            }
        }
        obj[j] = s;
    }
    tab.obj = obj;
    let (status, entering) = tab.run(|j| j < ncols - m, max_pivots);
    let objective = tab.obj[ncols].s_neg();
    let duals = duals_of(&tab, &S::s_zero());
    let ray = entering.map(|j| tab.ray(j)[..n].to_vec());
    Solved {
        status,
        objective,
        y: solution(&tab),
        duals,
        pivots: tab.pivots,
        basis: tab.basis.clone(),
        sigma,
        nslack,
        ray,
    }
}

/// The dual of `min c'y, A y (sense) b, y >= 0` as a minimization over
/// nonnegative variables: one variable per `>=` row (`w_r >= 0`), one per
/// `<=` row (`w_r = -u_r`), two per `=` row (`w_r = u+ - u-`); one `<=`
/// constraint per primal column (`A_j' w <= c_j`); objective `-b'w`.
/// Returns the LP and, per dual variable, `(primal row, sign)`.
pub fn dual_of(lp: &Lp) -> (Lp, Vec<(usize, i8)>) {
    let mut d = Lp::default();
    let mut map: Vec<(usize, i8)> = Vec::new();
    for (r, row) in lp.rows.iter().enumerate() {
        match row.sense {
            Sense::Ge => {
                d.var(&format!("w{r}"));
                map.push((r, 1));
            }
            Sense::Le => {
                d.var(&format!("w{r}-"));
                map.push((r, -1));
            }
            Sense::Eq => {
                d.var(&format!("w{r}+"));
                map.push((r, 1));
                d.var(&format!("w{r}-"));
                map.push((r, -1));
            }
        }
    }
    let mut cols: Vec<Vec<(usize, Q)>> = vec![Vec::new(); lp.nvars];
    for (k, &(r, sign)) in map.iter().enumerate() {
        for (j, c) in &lp.rows[r].coef {
            cols[*j].push((k, if sign < 0 { -c } else { c.clone() }));
        }
    }
    for j in 0..lp.nvars {
        d.row(
            cols[j].clone(),
            Sense::Le,
            lp.cost(j),
            format!("dual[{}]", lp.names[j]),
        );
    }
    d.objective = map
        .iter()
        .enumerate()
        .map(|(k, &(r, sign))| {
            (
                k,
                if sign < 0 {
                    lp.rows[r].rhs.clone()
                } else {
                    -lp.rows[r].rhs.clone()
                },
            )
        })
        .collect();
    (d, map)
}

/// Primal row multipliers from a dual solution vector.
fn multipliers(lp: &Lp, map: &[(usize, i8)], u: &[Q]) -> Vec<Q> {
    let mut w = vec![Q::zero(); lp.rows.len()];
    for (k, &(r, sign)) in map.iter().enumerate() {
        if sign < 0 {
            w[r] = &w[r] - &u[k];
        } else {
            w[r] = &w[r] + &u[k];
        }
    }
    w
}

/// Exact Gauss-Jordan elimination for the square system `a x = b`; `None`
/// when `a` is singular.
pub fn solve_exact(mut a: Vec<Vec<Q>>, mut b: Vec<Q>) -> Option<Vec<Q>> {
    let n = b.len();
    if n == 0 {
        return Some(Vec::new());
    }
    for col in 0..n {
        let pivot = (col..n).find(|r| !a[*r][col].is_zero())?;
        a.swap(col, pivot);
        b.swap(col, pivot);
        let p = a[col][col].clone();
        for k in 0..n {
            a[col][k] = &a[col][k] / &p;
        }
        b[col] = &b[col] / &p;
        for r in 0..n {
            if r != col && !a[r][col].is_zero() {
                let f = a[r][col].clone();
                for k in 0..n {
                    let v = &a[col][k] * &f;
                    a[r][k] = &a[r][k] - v;
                }
                let v = &b[col] * &f;
                b[r] = &b[r] - v;
            }
        }
    }
    Some(b)
}

/// Each row divided by the power of two nearest its largest coefficient
/// magnitude: the feasible set, the basis structure and the variables are
/// unchanged, and the floating simplex sees rows of comparable scale.
pub fn row_scaled(lp: &Lp) -> Lp {
    let mut out = lp.clone();
    for row in out.rows.iter_mut() {
        let max = row
            .coef
            .iter()
            .map(|(_, c)| to_f64(c).abs())
            .fold(0.0, f64::max);
        if max > 0.0 && max.is_finite() {
            let k = max.log2().round() as i32;
            let factor = if k >= 0 {
                Q::new(
                    num_bigint::BigInt::from(1),
                    num_bigint::BigInt::from(1) << (k as u32),
                )
            } else {
                Q::from_integer(num_bigint::BigInt::from(1) << ((-k) as u32))
            };
            for (_, c) in row.coef.iter_mut() {
                *c = &*c * &factor;
            }
            row.rhs = &row.rhs * &factor;
        }
    }
    out
}

/// The inequality rows, in the order of their slack columns.
fn slack_rows(lp: &Lp) -> Vec<usize> {
    lp.rows
        .iter()
        .enumerate()
        .filter(|(_, r)| r.sense != Sense::Eq)
        .map(|(r, _)| r)
        .collect()
}

/// Exact optimal certificate from the final basis of a floating solve: the
/// tight rows and the basic structural columns form a square system; its
/// exact primal solution must satisfy every row and `y >= 0`, its exact dual
/// solution must be dual feasible, and the two objectives must agree.
pub fn exact_optimum_from_basis(lp: &Lp, f: &Solved<f64>) -> Option<(Vec<Q>, Vec<Q>, Q)> {
    let m = lp.rows.len();
    let n = lp.nvars;
    let art_start = n + f.nslack;
    let slack_row = slack_rows(lp);
    let mut tight = vec![true; m];
    let mut structural = Vec::new();
    for &b in &f.basis {
        if b < n {
            structural.push(b);
        } else if b < art_start {
            tight[slack_row[b - n]] = false;
        } else {
            // a basic artificial at zero level marks a redundant row: it is
            // left out of the square system and gets a zero multiplier; the
            // exact check of every row below covers it as well
            tight[b - art_start] = false;
        }
    }
    let t_rows: Vec<usize> = (0..m).filter(|r| tight[*r]).collect();
    let k = structural.len();
    if t_rows.len() != k {
        return None;
    }
    let mut a = vec![vec![Q::zero(); k]; k];
    let mut b = vec![Q::zero(); k];
    for (e, &r) in t_rows.iter().enumerate() {
        b[e] = lp.rows[r].rhs.clone();
        for (j, c) in &lp.rows[r].coef {
            if let Some(pos) = structural.iter().position(|x| x == j) {
                a[e][pos] = c.clone();
            }
        }
    }
    let yj = solve_exact(a.clone(), b)?;
    let mut y = vec![Q::zero(); n];
    for (pos, &j) in structural.iter().enumerate() {
        if yj[pos].is_negative() {
            return None;
        }
        y[j] = yj[pos].clone();
    }
    let primal = verify_primal(lp, &y).ok()?;
    let at: Vec<Vec<Q>> = (0..k)
        .map(|i| (0..k).map(|e| a[e][i].clone()).collect())
        .collect();
    let c: Vec<Q> = structural.iter().map(|&j| lp.cost(j)).collect();
    let pit = solve_exact(at, c)?;
    let mut w = vec![Q::zero(); m];
    for (e, &r) in t_rows.iter().enumerate() {
        w[r] = pit[e].clone();
    }
    let value = verify_dual_bound(lp, &w).ok()?;
    if primal != value {
        return None;
    }
    Some((w, y, value))
}

/// Exact Farkas witness from the final phase-1 basis of a floating solve.
/// The phase-1 multipliers are zero on rows with a basic slack and one on
/// rows with a basic artificial (in the row orientation of the phase-1
/// tableau); every basic structural column has zero reduced cost, which
/// gives the remaining equations. The witness is returned only if it passes
/// [`verify_farkas`].
pub fn exact_farkas_from_basis(lp: &Lp, f: &Solved<f64>) -> Option<Vec<Q>> {
    let m = lp.rows.len();
    let n = lp.nvars;
    let art_start = n + f.nslack;
    let slack_row = slack_rows(lp);
    let mut pi: Vec<Option<Q>> = vec![None; m];
    let mut structural = Vec::new();
    for &b in &f.basis {
        if b < n {
            structural.push(b);
        } else if b < art_start {
            let r = slack_row[b - n];
            if pi[r].is_some() {
                return None;
            }
            pi[r] = Some(Q::zero());
        } else {
            let r = b - art_start;
            if pi[r].is_some() {
                return None;
            }
            pi[r] = Some(Q::one());
        }
    }
    let unknown: Vec<usize> = (0..m).filter(|r| pi[*r].is_none()).collect();
    let k = unknown.len();
    if k != structural.len() {
        return None;
    }
    let signed = |r: usize, c: &Q| -> Q { if f.sigma[r] < 0 { -c } else { c.clone() } };
    let mut a = vec![vec![Q::zero(); k]; k];
    let mut rhs = vec![Q::zero(); k];
    for (e, &j) in structural.iter().enumerate() {
        for (r, row) in lp.rows.iter().enumerate() {
            let Some((_, c)) = row.coef.iter().find(|(jj, _)| *jj == j) else {
                continue;
            };
            let c = signed(r, c);
            match &pi[r] {
                Some(v) => rhs[e] = &rhs[e] - v * c,
                None => {
                    let pos = unknown.iter().position(|x| *x == r).unwrap();
                    a[e][pos] = &a[e][pos] + c;
                }
            }
        }
    }
    let sol = solve_exact(a, rhs)?;
    for (pos, &r) in unknown.iter().enumerate() {
        pi[r] = Some(sol[pos].clone());
    }
    let w: Vec<Q> = (0..m)
        .map(|r| {
            let v = pi[r].clone().unwrap();
            if f.sigma[r] < 0 { -v } else { v }
        })
        .collect();
    verify_farkas(lp, &w).ok()?;
    Some(w)
}

/// Check `y >= 0` and every original rational row (target rows included)
/// exactly, and return the primal objective. A feasible point shows
/// membership in the relaxation only; it does not certify a scheduling policy.
pub fn verify_primal(lp: &Lp, y: &[Q]) -> Result<Q, String> {
    if y.len() != lp.nvars {
        return Err("primal point length".into());
    }
    for (j, value) in y.iter().enumerate() {
        if value.is_negative() {
            return Err(format!("variable {} is negative", lp.names[j]));
        }
    }
    for row in &lp.rows {
        let lhs: Q = row.coef.iter().map(|(j, c)| c * &y[*j]).sum();
        let valid = match row.sense {
            Sense::Eq => lhs == row.rhs,
            Sense::Ge => lhs >= row.rhs,
            Sense::Le => lhs <= row.rhs,
        };
        if !valid {
            return Err(format!("primal point violates row {}", row.label));
        }
    }
    Ok((0..lp.nvars).map(|j| lp.cost(j) * &y[j]).sum())
}

/// The dual LP has one `<=` row per primal variable, so its row multipliers
/// propose the primal point `y = -dual_rows`; the candidate is accepted only
/// if it passes [`verify_primal`].
fn checked_primal_from_dual_rows(lp: &Lp, dual_rows: &[Q]) -> Result<Vec<Q>, String> {
    let y: Vec<Q> = dual_rows.iter().map(|v| -v).collect();
    verify_primal(lp, &y)?;
    Ok(y)
}

/// Check a Farkas witness against the original rows: `w_r >= 0` on `>=`
/// rows, `w_r <= 0` on `<=` rows, `w'A <= 0` for every column and `w'b > 0`,
/// and return `w'b`. Every feasible `y` would then satisfy
/// `0 >= w'A y >= w'b > 0`, so none exists.
pub fn verify_farkas(lp: &Lp, w: &[Q]) -> Result<Q, String> {
    if w.len() != lp.rows.len() {
        return Err("witness length".into());
    }
    let mut col = vec![Q::zero(); lp.nvars];
    let mut wb = Q::zero();
    for (r, row) in lp.rows.iter().enumerate() {
        match row.sense {
            Sense::Ge if w[r].is_negative() => {
                return Err(format!("row {} needs w >= 0", row.label));
            }
            Sense::Le if w[r].is_positive() => {
                return Err(format!("row {} needs w <= 0", row.label));
            }
            _ => {}
        }
        if w[r].is_zero() {
            continue;
        }
        for (j, c) in &row.coef {
            col[*j] = &col[*j] + c * &w[r];
        }
        wb += &row.rhs * &w[r];
    }
    for (j, v) in col.iter().enumerate() {
        if v.is_positive() {
            return Err(format!(
                "column {} has w'A = {} > 0",
                lp.names[j],
                to_f64(v)
            ));
        }
    }
    if !wb.is_positive() {
        return Err(format!("w'b = {} is not positive", to_f64(&wb)));
    }
    Ok(wb)
}

/// Check dual feasibility of `w` for `min c'y` and return the dual objective
/// `w'b`, a lower bound on the optimum by weak duality.
pub fn verify_dual_bound(lp: &Lp, w: &[Q]) -> Result<Q, String> {
    if w.len() != lp.rows.len() {
        return Err("witness length".into());
    }
    let mut col = vec![Q::zero(); lp.nvars];
    let mut wb = Q::zero();
    for (r, row) in lp.rows.iter().enumerate() {
        match row.sense {
            Sense::Ge if w[r].is_negative() => {
                return Err(format!("row {} needs w >= 0", row.label));
            }
            Sense::Le if w[r].is_positive() => {
                return Err(format!("row {} needs w <= 0", row.label));
            }
            _ => {}
        }
        if w[r].is_zero() {
            continue;
        }
        for (j, c) in &row.coef {
            col[*j] = &col[*j] + c * &w[r];
        }
        wb += &row.rhs * &w[r];
    }
    for j in 0..lp.nvars {
        if col[j] > lp.cost(j) {
            return Err(format!(
                "column {} has w'A = {} > c = {}",
                lp.names[j],
                to_f64(&col[j]),
                to_f64(&lp.cost(j))
            ));
        }
    }
    Ok(wb)
}

/// The outcome of one LP with its exact certificate.
#[derive(Clone, Debug)]
pub struct Certified {
    /// `certified_infeasible`, `relaxation_feasible`, `certified_lower_bound`
    /// or `numerically_inconclusive`.
    pub status: String,
    /// Objective of the floating solve, when its status is optimal.
    pub float_objective: Option<f64>,
    /// The verified `w'b`: the lower bound on the minimum, or the positive
    /// right-hand side of a Farkas witness.
    pub certified_value: Option<Q>,
    /// Nonzero row multipliers of the certificate, by row label.
    pub witness: Option<Vec<(String, Q)>>,
    /// The method that produced the certificate, or the reason for its absence.
    pub note: String,
    pub pivots: usize,
    pub rows: usize,
    pub cols: usize,
    pub float_ms: f64,
    pub exact_ms: f64,
    /// Time of the independent certificate checks against the rational rows.
    pub verify_ms: f64,
    /// Whether the result comes from the exact simplex fallback.
    pub used_exact_simplex: bool,
    /// Set only when a verified primal point attains a verified dual bound:
    /// the certified value is then the exact optimum of the rational
    /// program, not only a lower bound on it.
    pub exact_optimum: bool,
    /// The solution as doubles: the verified exact point when there is one,
    /// otherwise the floating solution.
    pub y: Vec<(String, f64)>,
}

impl Certified {
    pub fn to_json(&self) -> Value {
        json!({
            "status": self.status, "float_objective": self.float_objective, "exact_optimum": self.exact_optimum,
            "certified_value": self.certified_value.as_ref().map(|v| dec(v, 15, false)),
            "certified_value_exact": self.certified_value.as_ref().map(|v| v.to_string()),
            "witness": self.witness.as_ref().map(|w| w.iter().map(|(l, v)| json!({"row": l, "weight": v.to_string(), "weight_decimal": to_f64(v)})).collect::<Vec<_>>()),
            "note": self.note, "pivots": self.pivots, "rows": self.rows, "cols": self.cols,
            "float_ms": self.float_ms, "exact_ms": self.exact_ms, "verify_ms": self.verify_ms, "used_exact_simplex": self.used_exact_simplex,
            "solution": self.y.iter().filter(|(_, v)| v.abs() > 1e-12).map(|(k, v)| json!({"var": k, "value": v})).collect::<Vec<_>>()
        })
    }
}

fn witness_rows(lp: &Lp, w: &[Q]) -> Vec<(String, Q)> {
    lp.rows
        .iter()
        .zip(w)
        .filter(|(_, v)| !v.is_zero())
        .map(|(r, v)| (r.label.clone(), v.clone()))
        .collect()
}

/// Certify `lp`. The floating simplex solves the row-scaled LP, and its
/// final basis is re-solved exactly: a Farkas witness when phase 1 ends
/// infeasible, a primal-dual pair with equal objectives when optimal. If
/// that fails, the exact simplex on the dual LP (as many rows as primal
/// variables) returns a dual solution, whose objective is a certified lower
/// bound, or an improving ray, which is a Farkas witness of primal
/// infeasibility; an infeasible dual passes the LP to the exact primal
/// phase 1. Every certificate is verified against the original rational
/// rows, and a failed verification or the pivot limit makes the result
/// `numerically_inconclusive`. With `feasibility_only`, `relaxation_feasible`
/// requires a verified primal point.
fn certify_exact(lp: &Lp, feasibility_only: bool) -> Certified {
    let t0 = Instant::now();
    let f = solve::<f64>(&row_scaled(lp), 200_000);
    let float_ms = t0.elapsed().as_secs_f64() * 1e3;
    let cols = lp.nvars;
    let rows = lp.rows.len();
    let y: Vec<(String, f64)> = lp.names.iter().cloned().zip(f.y.iter().cloned()).collect();
    let float_objective = (f.status == Status::Optimal).then_some(f.objective);
    let t1 = Instant::now();
    let done = |status: &str,
                value: Option<Q>,
                witness: Option<Vec<(String, Q)>>,
                note: String,
                pivots: usize,
                exact: bool,
                y: Vec<(String, f64)>,
                verify_ms: f64| Certified {
        status: status.into(),
        float_objective,
        certified_value: value,
        witness,
        note,
        pivots,
        rows,
        cols,
        float_ms,
        exact_ms: t1.elapsed().as_secs_f64() * 1e3,
        verify_ms,
        used_exact_simplex: exact,
        exact_optimum: false,
        y,
    };
    // Fast path: the floating basis re-solved exactly.
    match f.status {
        Status::Infeasible => {
            if let Some(w) = exact_farkas_from_basis(lp, &f) {
                let tv = Instant::now();
                let wb = verify_farkas(lp, &w).expect("verified above");
                let verify_ms = tv.elapsed().as_secs_f64() * 1e3;
                return done("certified_infeasible", Some(wb), Some(witness_rows(lp, &w)), "Farkas witness from the floating phase-1 basis, re-solved and verified in exact arithmetic".into(), f.pivots, false, y, verify_ms);
            }
        }
        Status::Optimal => {
            if let Some((w, ye, value)) = exact_optimum_from_basis(lp, &f) {
                let tv = Instant::now();
                let recheck = verify_dual_bound(lp, &w).expect("verified above");
                let verify_ms = tv.elapsed().as_secs_f64() * 1e3;
                debug_assert_eq!(recheck, value);
                let y_exact: Vec<(String, f64)> = lp
                    .names
                    .iter()
                    .cloned()
                    .zip(ye.iter().map(to_f64))
                    .collect();
                if feasibility_only {
                    return done("relaxation_feasible", None, None, "exact primal solution of the floating basis satisfies every row (a point of the relaxation, not a schedule)".into(), f.pivots, false, y_exact, verify_ms);
                }
                let mut c = done("certified_lower_bound", Some(value), Some(witness_rows(lp, &w)), "floating basis re-solved exactly: primal feasible, dual feasible, objectives agree (exact optimum)".into(), f.pivots, false, y_exact, verify_ms);
                c.exact_optimum = true;
                return c;
            }
        }
        _ => {}
    }
    // Fallback: the exact simplex on the dual.
    let (dual, map) = dual_of(lp);
    let e = solve::<Q>(&dual, 60_000);
    match e.status {
        Status::Unbounded => {
            let ray = e.ray.clone().unwrap_or_default();
            let w = multipliers(lp, &map, &ray);
            let tv = Instant::now();
            let check = verify_farkas(lp, &w);
            let verify_ms = tv.elapsed().as_secs_f64() * 1e3;
            match check {
                Ok(wb) => done("certified_infeasible", Some(wb), Some(witness_rows(lp, &w)), "Farkas witness (dual ray of the exact simplex) verified against the primal rows".into(), e.pivots, true, y, verify_ms),
                Err(err) => done("numerically_inconclusive", None, None, format!("dual ray failed verification: {err}"), e.pivots, true, y, verify_ms),
            }
        }
        Status::Optimal => {
            let w = multipliers(lp, &map, &e.y);
            let tv = Instant::now();
            let check = verify_dual_bound(lp, &w);
            let primal = feasibility_only.then(|| checked_primal_from_dual_rows(lp, &e.duals));
            let verify_ms = tv.elapsed().as_secs_f64() * 1e3;
            match check {
                Ok(wb) => {
                    if feasibility_only {
                        match primal.expect("primal check requested") {
                            Ok(point) => {
                                let y_checked = lp
                                    .names
                                    .iter()
                                    .cloned()
                                    .zip(point.iter().map(to_f64))
                                    .collect();
                                done("relaxation_feasible", None, None, "exact primal candidate from the dual row multipliers satisfies every original row and nonnegativity bound (a point of the relaxation, not a schedule)".into(), e.pivots, true, y_checked, verify_ms)
                            }
                            Err(err) => done(
                                "numerically_inconclusive",
                                None,
                                None,
                                format!(
                                    "exact dual bound verified, but primal candidate failed verification: {err}"
                                ),
                                e.pivots,
                                true,
                                y,
                                verify_ms,
                            ),
                        }
                    } else {
                        // A primal point reconstructed from the dual rows that passes every
                        // row check and attains the dual bound upgrades the bound to a
                        // verified minimum; otherwise the value stays a certified lower bound.
                        let attained = checked_primal_from_dual_rows(lp, &e.duals)
                            .ok()
                            .filter(|p| (0..lp.nvars).map(|j| lp.cost(j) * &p[j]).sum::<Q>() == wb);
                        match attained {
                            Some(point) => {
                                let y_checked: Vec<(String, f64)> = lp
                                    .names
                                    .iter()
                                    .cloned()
                                    .zip(point.iter().map(to_f64))
                                    .collect();
                                let mut c = done("certified_lower_bound", Some(wb), Some(witness_rows(lp, &w)), "dual solution of the exact simplex verified against the primal rows, and a verified primal point attains it (exact optimum)".into(), e.pivots, true, y_checked, verify_ms);
                                c.exact_optimum = true;
                                c
                            }
                            None => done("certified_lower_bound", Some(wb), Some(witness_rows(lp, &w)), "dual solution of the exact simplex verified feasible against the primal rows; its objective is the certified lower bound (no primal point verified)".into(), e.pivots, true, y, verify_ms),
                        }
                    }
                }
                Err(err) => done(
                    "numerically_inconclusive",
                    None,
                    None,
                    format!("exact dual solution failed verification: {err}"),
                    e.pivots,
                    true,
                    y,
                    verify_ms,
                ),
            }
        }
        Status::Infeasible => {
            let p = solve::<Q>(lp, 60_000);
            if p.status == Status::Infeasible {
                let tv = Instant::now();
                let check = verify_farkas(lp, &p.duals);
                let verify_ms = tv.elapsed().as_secs_f64() * 1e3;
                match check {
                    Ok(wb) => done(
                        "certified_infeasible",
                        Some(wb),
                        Some(witness_rows(lp, &p.duals)),
                        "Farkas witness from the exact primal phase 1 verified".into(),
                        p.pivots,
                        true,
                        y,
                        verify_ms,
                    ),
                    Err(err) => done(
                        "numerically_inconclusive",
                        None,
                        None,
                        format!("primal witness failed verification: {err}"),
                        p.pivots,
                        true,
                        y,
                        verify_ms,
                    ),
                }
            } else {
                done(
                    "numerically_inconclusive",
                    None,
                    None,
                    format!("dual infeasible, exact primal {:?}", p.status),
                    p.pivots,
                    true,
                    y,
                    0.0,
                )
            }
        }
        Status::PivotLimit => done(
            "numerically_inconclusive",
            None,
            None,
            "exact dual simplex reached the pivot limit".into(),
            e.pivots,
            true,
            y,
            0.0,
        ),
    }
}

/// Feasibility of `lp`: `certified_infeasible` with a verified Farkas
/// witness, or `relaxation_feasible` with a verified primal point.
pub fn certify_feasibility(lp: &Lp) -> Certified {
    certify_exact(lp, true)
}

/// Minimization of `lp`: a verified dual bound (`certified_lower_bound`,
/// with `exact_optimum` when a verified primal point attains it) or a
/// verified infeasibility witness.
pub fn certify_minimum(lp: &Lp) -> Certified {
    certify_exact(lp, false)
}

#[cfg(test)]
mod tests {
    use super::super::rational::{qi, qr};
    use super::*;

    fn tiny() -> Lp {
        // min x + y  s.t. x + 2y >= 2, x <= 3, x - y = 0.5
        let mut lp = Lp::default();
        let x = lp.var("x");
        let y = lp.var("y");
        lp.row(vec![(x, qi(1)), (y, qi(2))], Sense::Ge, qi(2), "r1");
        lp.row(vec![(x, qi(1))], Sense::Le, qi(3), "r2");
        lp.row(vec![(x, qi(1)), (y, qi(-1))], Sense::Eq, qr(1, 2), "r3");
        lp.objective = vec![(x, qi(1)), (y, qi(1))];
        lp
    }

    #[test]
    fn float_and_exact_simplex_agree_on_a_small_lp() {
        let lp = tiny();
        let f = solve::<f64>(&lp, 1000);
        let e = solve::<Q>(&lp, 1000);
        assert_eq!(f.status, Status::Optimal);
        assert_eq!(e.status, Status::Optimal);
        // y = 0.5, x = 1 -> objective 1.5
        assert!((f.objective - 1.5).abs() < 1e-9);
        assert_eq!(e.objective, qr(3, 2));
        let c = certify_minimum(&lp);
        assert_eq!(c.status, "certified_lower_bound");
        assert_eq!(c.certified_value.unwrap(), qr(3, 2));
    }

    #[test]
    fn infeasible_lp_gets_a_verified_farkas_witness() {
        let mut lp = Lp::default();
        let x = lp.var("x");
        lp.row(vec![(x, qi(1))], Sense::Ge, qi(2), "x >= 2");
        lp.row(vec![(x, qi(1))], Sense::Le, qi(1), "x <= 1");
        let c = certify_feasibility(&lp);
        assert_eq!(c.status, "certified_infeasible", "{}", c.note);
        let w: Vec<Q> = c.witness.unwrap().into_iter().map(|(_, v)| v).collect();
        assert!(w[0].is_positive() && w[1].is_negative());
        // tampering with the witness fails verification
        assert!(verify_farkas(&lp, &[qi(1), qi(1)]).is_err());
        assert!(verify_farkas(&lp, &[qi(1), qi(0)]).is_err());
    }

    #[test]
    fn primal_checks_include_row_senses_targets_and_nonnegativity() {
        let lp = tiny();
        assert_eq!(verify_primal(&lp, &[qi(1), qr(1, 2)]), Ok(qr(3, 2)));
        assert!(verify_primal(&lp, &[qi(0), qr(1, 2)]).is_err());
        assert!(verify_primal(&lp, &[qi(1), qi(1)]).is_err());
        assert!(verify_primal(&lp, &[qi(4), qr(7, 2)]).is_err());
        assert!(verify_primal(&lp, &[qi(1)]).is_err());

        let mut target = Lp::default();
        let x = target.var("x");
        target.row(vec![(x, qi(1))], Sense::Le, qr(1, 100), "target");
        assert!(verify_primal(&target, &[qr(1, 100)]).is_ok());
        assert!(verify_primal(&target, &[qr(1, 99)]).is_err());
        assert!(verify_primal(&target, &[qr(-1, 100)]).is_err());
    }

    #[test]
    fn exact_dual_rows_must_supply_a_checked_primal_point() {
        let lp = tiny();
        let (dual, _) = dual_of(&lp);
        let solved = solve::<Q>(&dual, 1000);
        assert_eq!(solved.status, Status::Optimal);
        assert_eq!(
            checked_primal_from_dual_rows(&lp, &solved.duals).unwrap(),
            vec![qi(1), qr(1, 2)]
        );
        // Even an optimal status cannot substitute for missing or invalid evidence.
        assert!(checked_primal_from_dual_rows(&lp, &[]).is_err());
        assert!(checked_primal_from_dual_rows(&lp, &[qi(0), qi(0)]).is_err());
        assert!(checked_primal_from_dual_rows(&lp, &[qi(1), qr(-1, 2)]).is_err());
    }

    #[test]
    fn fallback_feasibility_uses_the_checked_exact_primal() {
        let mut lp = Lp::default();
        let x = lp.var("x");
        lp.row(vec![(x, qi(1))], Sense::Le, qi(1), "target");
        // The floating tolerance ignores this improving direction. Exact
        // reconstruction rejects that basis and triggers the dual fallback.
        lp.objective = vec![(
            x,
            super::super::rational::parse_q("1/1000000000000").unwrap(),
        )];
        let c = certify_feasibility(&lp);
        assert!(c.used_exact_simplex);
        assert_eq!(c.status, "relaxation_feasible", "{}", c.note);
        assert_eq!(c.y, vec![("x".into(), 0.0)]);
        assert!(c.note.contains("satisfies every original row"));
    }

    #[test]
    fn checked_dual_bound_in_the_fallback_is_an_exact_optimum_only_with_a_verified_primal_point() {
        // The floating basis cannot be re-solved here (the tiny cost defeats the
        // fast path), so the exact dual simplex supplies the bound. The primal
        // point x = 0 reconstructed from the dual rows passes every row check
        // and attains the bound, which makes the value a verified minimum; the
        // flag, not the status, records that.
        let mut lp = Lp::default();
        let x = lp.var("x");
        lp.row(vec![(x, qi(1))], Sense::Le, qi(1), "target");
        let cost = super::super::rational::parse_q("1/1000000000000").unwrap();
        lp.objective = vec![(x, cost.clone())];
        let c = certify_minimum(&lp);
        assert!(c.used_exact_simplex);
        assert_eq!(c.status, "certified_lower_bound");
        assert_eq!(c.certified_value, Some(qi(0)));
        assert!(c.exact_optimum, "{}", c.note);
        assert!(c.note.contains("exact optimum"));
        // the fast path marks its verified optimum the same way
        let t = tiny();
        let ct = certify_minimum(&t);
        assert!(!ct.used_exact_simplex);
        assert!(ct.exact_optimum);
        // a feasibility certificate never claims an optimum
        let cf = certify_feasibility(&lp);
        assert_eq!(cf.status, "relaxation_feasible");
        assert!(!cf.exact_optimum);
    }

    #[test]
    fn unbounded_directions_are_reported() {
        let mut lp = Lp::default();
        let x = lp.var("x");
        lp.row(vec![(x, qi(1))], Sense::Ge, qi(1), "x >= 1");
        lp.objective = vec![(x, qi(-1))];
        assert_eq!(solve::<f64>(&lp, 100).status, Status::Unbounded);
    }
}
