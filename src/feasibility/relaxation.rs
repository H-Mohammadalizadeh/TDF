//! The linear relaxation of the necessary conditions, built at a level (the
//! constraint groups imposed) for an objective. With the target rows it tests
//! feasibility; without them it lower-bounds the best achievable miss
//! probabilities and yields `kappa_L`; with them and a rate objective it
//! bounds the reconfiguration rates of any policy that meets the targets.
//! [`build`] lists the variables and the constraint groups.

use super::coefficients::PhaseCoefficients;
use super::gaps::GapRow;
use super::model::DiscreteModel;
use super::rational::{Q, exp_neg};
use super::simplex::{Lp, Sense};
use super::windows::WindowRow;
use num_traits::{One, Zero};

/// The constraint levels. Every level also carries the target rows, which
/// the objective keeps or drops.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Level {
    /// Variable bounds, the service-only and single-class rows, the time
    /// budget and flow balance.
    Base,
    /// The base level plus the window constraints.
    BaseWindows,
    /// The base level plus the phase constraints.
    BasePhases,
    /// The base level plus the window and phase constraints: the feasibility
    /// test.
    Full,
}

impl Level {
    /// The four levels, from the base level to the full test.
    pub fn all() -> [Level; 4] {
        [
            Level::Base,
            Level::BaseWindows,
            Level::BasePhases,
            Level::Full,
        ]
    }

    /// The name used in output records.
    pub fn name(&self) -> &'static str {
        match self {
            Level::Base => "base",
            Level::BaseWindows => "base_windows",
            Level::BasePhases => "base_phases",
            Level::Full => "full",
        }
    }

    /// Constraint groups of the level.
    pub fn groups(&self) -> Vec<&'static str> {
        self.groups_with(false)
    }

    /// Constraint groups of the level; with `gap_rows`, the full test also
    /// carries the visit-gap groups `gap_window` and `gap` (`gaps.rs`).
    pub fn groups_with(&self, gap_rows: bool) -> Vec<&'static str> {
        match self {
            Level::Full if gap_rows => vec![
                "bounds",
                "service_only",
                "mm1",
                "time",
                "flow",
                "window",
                "phase",
                "gap_window",
                "gap",
                "target",
            ],
            Level::Full => vec![
                "bounds",
                "service_only",
                "mm1",
                "time",
                "flow",
                "window",
                "phase",
                "target",
            ],
            Level::Base => vec!["bounds", "service_only", "mm1", "time", "flow", "target"],
            Level::BaseWindows => vec![
                "bounds",
                "service_only",
                "mm1",
                "time",
                "flow",
                "window",
                "target",
            ],
            Level::BasePhases => vec![
                "bounds",
                "service_only",
                "mm1",
                "time",
                "flow",
                "phase",
                "target",
            ],
        }
    }
}

/// The objective of one LP over the relaxation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Objective {
    /// Target rows kept, no objective.
    Feasibility,
    /// `min t` with `x_i <= t` (the worst-class miss probability), targets
    /// dropped.
    MinMax,
    /// `min sum lambda_i x_i / sum lambda_i` (the arrival-weighted miss
    /// probability), targets dropped.
    MinArrivalWeighted,
    /// The total reconfiguration rate `sum_{u != v} nu_uv`, minimized or (with
    /// `maximize`, through the negated objective) maximized over the region
    /// with every target row retained: a necessary operating range.
    TotalRate { maximize: bool },
    /// The entry rate `eta_i = sum_{u != i} nu_ui` into configuration `class`,
    /// minimized or maximized like `TotalRate`, targets retained.
    EntryRate { class: usize, maximize: bool },
}

/// Everything the relaxation needs for one model. `gaps` is `None` unless the
/// visit-gap constraints are requested; the full test then carries them.
pub struct Ingredients<'a> {
    pub model: &'a DiscreteModel,
    pub coef: &'a PhaseCoefficients,
    pub windows: &'a [WindowRow],
    pub gaps: Option<&'a [GapRow]>,
}

/// The LP of one level and objective, with the column indices of `nu_uv`
/// (`None` on the diagonal), `z_j` and `x_i`.
pub struct Relaxation {
    pub lp: Lp,
    pub nu: Vec<Vec<Option<usize>>>,
    pub z: Vec<usize>,
    pub x: Vec<usize>,
}

/// Build the relaxation at one level for one objective. Variables: the
/// reconfiguration rate `nu_uv >= 0` of every ordered pair, the idle
/// fraction `z_j >= 0` of every configuration, a candidate miss probability
/// `x_i` of every class (`x_i <= 1` for active classes) and, for `MinMax`,
/// `t >= 0`. Constraint groups:
///
/// * `service_only`: `x_i >= exp(-mu_i d_i)` (a job misses when its own
///   service exceeds the deadline);
/// * `mm1`: the single-class baseline `x_i >= exp(-(mu_i - lambda_i) d_i)`
///   for `lambda_i < mu_i`;
/// * `time`: the time budget `sum rho + sum nu E[tau] + sum z = 1` (two
///   inequalities when the mean reconfiguration time is only enclosed);
/// * `flow`: flow balance, as many reconfigurations enter every
///   configuration as leave it;
/// * `window`: the minimum reconfiguration rate (`windows.rs`);
/// * `phase`: the misses caused by service, idling and reconfiguration,
///   `x_k >= sum_j rho_j A_jk + sum_j z_j B_jk + sum_{u != v} nu_uv C_uv,k`;
/// * `gap_window`, `gap`: the visit-gap constraints of `ing.gaps`
///   (`gaps.rs`), in the full test only when they are given;
/// * `target`: `x_i <= eps_i`, kept for the feasibility and rate objectives.
///
/// Per-class rows exist only for active classes. With coefficients in the
/// certificate direction, every approximated constant weakens its row, so
/// the relaxation contains the exact region.
pub fn build(ing: &Ingredients, level: Level, objective: Objective) -> Relaxation {
    build_groups(ing, &level.groups_with(ing.gaps.is_some()), objective)
}

/// The relaxation with the constraint groups `groups` (any of those of
/// [`build`]) for one objective.
pub fn build_groups(ing: &Ingredients, groups: &[&str], objective: Objective) -> Relaxation {
    let model = ing.model;
    let n = model.n;
    let has = |g: &str| groups.contains(&g);
    let mut lp = Lp::default();
    let mut nu = vec![vec![None; n]; n];
    for (u, v) in model.edges() {
        nu[u][v] = Some(lp.var(&format!("nu_{u}_{v}")));
    }
    let z: Vec<usize> = (0..n).map(|j| lp.var(&format!("z_{j}"))).collect();
    let x: Vec<usize> = (0..n).map(|i| lp.var(&format!("x_{i}"))).collect();
    let t = matches!(objective, Objective::MinMax).then(|| lp.var("t"));
    let active: Vec<usize> = (0..n).filter(|i| model.active(*i)).collect();
    for &i in &active {
        lp.row(
            vec![(x[i], Q::one())],
            Sense::Le,
            Q::one(),
            format!("x_{i} <= 1"),
        );
    }
    if has("service_only") {
        for &i in &active {
            lp.row(
                vec![(x[i], Q::one())],
                Sense::Ge,
                ing.coef.service_only[i].clone(),
                format!("service_only[{i}]"),
            );
        }
    }
    if has("mm1") {
        // Single-class baseline: under FCFS within the class, a class-i job
        // completes no earlier than in an M/M/1 queue with the same arrivals
        // and service times, so x_i >= exp(-(mu_i - lambda_i) d_i) when
        // lambda_i < mu_i (lower enclosure of the exponential).
        for &i in &active {
            if model.lam[i] < model.mu[i] {
                let arg = (&model.mu[i] - &model.lam[i]) * &model.d[i];
                let (lo, _) = exp_neg(&arg, ing.coef.bits);
                lp.row(
                    vec![(x[i], Q::one())],
                    Sense::Ge,
                    lo,
                    format!("mm1_dedicated[{i}]"),
                );
            }
        }
    }
    if has("time") {
        // Time budget: sum rho + sum nu E[tau] + sum z = 1. With an enclosed
        // mean E[tau] in [lo, hi] the equality becomes the two necessary
        // inequalities sum nu lo + sum z <= 1 - sum rho <= sum nu hi + sum z.
        let exact = model
            .edges()
            .iter()
            .all(|(u, v)| ing.coef.mean_tau_hi[*u][*v] == ing.coef.mean_tau[*u][*v]);
        let rhs = Q::one() - model.total_rho();
        let z_terms: Vec<(usize, Q)> = (0..n).map(|j| (z[j], Q::one())).collect();
        if exact {
            let mut coef: Vec<(usize, Q)> = Vec::new();
            for (u, v) in model.edges() {
                coef.push((nu[u][v].unwrap(), ing.coef.mean_tau[u][v].clone()));
            }
            coef.extend(z_terms);
            lp.row(coef, Sense::Eq, rhs, "time_accounting");
        } else {
            let mut lower: Vec<(usize, Q)> = Vec::new();
            let mut upper: Vec<(usize, Q)> = Vec::new();
            for (u, v) in model.edges() {
                lower.push((nu[u][v].unwrap(), ing.coef.mean_tau[u][v].clone()));
                upper.push((nu[u][v].unwrap(), ing.coef.mean_tau_hi[u][v].clone()));
            }
            lower.extend(z_terms.clone());
            upper.extend(z_terms);
            lp.row(lower, Sense::Le, rhs.clone(), "time_accounting_lower_mean");
            lp.row(upper, Sense::Ge, rhs, "time_accounting_upper_mean");
        }
    }
    if has("flow") {
        // The N flow rows sum to zero identically; the last one is dropped.
        for j in 0..n.saturating_sub(1) {
            let mut coef = Vec::new();
            for v in 0..n {
                if v != j {
                    coef.push((nu[j][v].unwrap(), Q::one()));
                    coef.push((nu[v][j].unwrap(), -Q::one()));
                }
            }
            lp.row(coef, Sense::Eq, Q::zero(), format!("flow[{j}]"));
        }
    }
    if has("phase") {
        for &i in &active {
            let mut coef = vec![(x[i], Q::one())];
            let mut rhs = Q::zero();
            for j in 0..n {
                rhs += model.rho(j) * &ing.coef.a[j][i];
                coef.push((z[j], -ing.coef.b[j][i].clone()));
            }
            for (u, v) in model.edges() {
                coef.push((nu[u][v].unwrap(), -ing.coef.c[u][v][i].clone()));
            }
            lp.row(coef, Sense::Ge, rhs, format!("phase[{i}]"));
        }
    }
    if has("window") {
        for (k, w) in ing.windows.iter().enumerate() {
            let mut coef = Vec::new();
            for (u, v) in model.edges() {
                coef.push((nu[u][v].unwrap(), w.h.clone()));
            }
            for &i in &w.subset {
                if !w.lam_a[i].is_zero() {
                    coef.push((x[i], w.lam_a[i].clone()));
                }
            }
            let subset: Vec<String> = w.subset.iter().map(|i| i.to_string()).collect();
            lp.row(
                coef,
                Sense::Ge,
                w.rhs.clone(),
                format!("window[{k}] h={} S={{{}}}", w.h, subset.join(",")),
            );
        }
    }
    for g in ing.gaps.unwrap_or_default() {
        if !has(g.group) {
            continue;
        }
        let i = g.class;
        let mut coef = vec![(x[i], g.x.clone()), (z[i], g.z.clone())];
        for u in (0..n).filter(|u| *u != i) {
            coef.push((nu[u][i].unwrap(), g.entry[u].clone()));
        }
        lp.row(coef, Sense::Ge, g.rhs.clone(), g.label.clone());
    }
    // The target rows stay in the feasibility test and in the rate objectives
    // (operating ranges of a policy that meets every target); the
    // performance bounds drop them.
    let keep_targets = matches!(
        objective,
        Objective::Feasibility | Objective::TotalRate { .. } | Objective::EntryRate { .. }
    );
    if keep_targets && has("target") {
        for &i in &active {
            lp.row(
                vec![(x[i], Q::one())],
                Sense::Le,
                model.eps[i].clone(),
                format!("target[{i}]"),
            );
        }
    }
    match objective {
        Objective::Feasibility => {}
        Objective::MinMax => {
            let t = t.unwrap();
            for &i in &active {
                lp.row(
                    vec![(t, Q::one()), (x[i], -Q::one())],
                    Sense::Ge,
                    Q::zero(),
                    format!("t >= x_{i}"),
                );
            }
            lp.objective = vec![(t, Q::one())];
        }
        Objective::MinArrivalWeighted => {
            let total: Q = active.iter().map(|i| model.lam[*i].clone()).sum();
            lp.objective = active
                .iter()
                .map(|&i| (x[i], &model.lam[i] / &total))
                .collect();
        }
        Objective::TotalRate { maximize } => {
            let s = if maximize { -Q::one() } else { Q::one() };
            lp.objective = model
                .edges()
                .iter()
                .map(|(u, v)| (nu[*u][*v].unwrap(), s.clone()))
                .collect();
        }
        Objective::EntryRate { class, maximize } => {
            let s = if maximize { -Q::one() } else { Q::one() };
            lp.objective = (0..n)
                .filter(|u| *u != class)
                .map(|u| (nu[u][class].unwrap(), s.clone()))
                .collect();
        }
    }
    Relaxation { lp, nu, z, x }
}
