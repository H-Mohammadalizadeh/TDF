//! The feasibility checker: policy-independent necessary conditions for
//! tail-delay targets, assembled into a linear relaxation ([`relaxation`],
//! [`windows`], [`gaps`]) whose phase coefficients are enclosed on a lattice
//! or a grid ([`coefficients`], [`grid`]) and which is decided with exact
//! certificates ([`simplex`]). [`rational`] and [`special`] provide the enclosures,
//! [`model`] and [`distributions`] the models, and [`reference`](mod@reference) the
//! certified interval of the symmetric case. This module evaluates a model
//! at a set of levels and computes operating ranges.

pub mod coefficients;
pub mod distributions;
pub mod gaps;
pub mod grid;
pub mod model;
pub mod rational;
pub mod reference;
pub mod relaxation;
pub mod simplex;
pub mod special;
pub mod windows;

use coefficients::{Lattice, Mode, PhaseCoefficients, lattice, phase_coefficients};
use distributions::Model;
use gaps::{GapRow, gap_rows};
use grid::GridSettings;
use model::DiscreteModel;
use num_traits::Zero;
use rational::{Q, to_f64};
use relaxation::{Ingredients, Level, Objective, build};
use serde_json::{Value, json};
use simplex::{Certified, certify_feasibility, certify_minimum};
use std::time::Instant;
use windows::{WindowRow, all_subsets, default_subsets, default_windows, window_rows};

/// Numerical settings of the relaxation.
#[derive(Clone, Debug)]
pub struct Options {
    /// Requested lattice or grid step of the phase coefficients.
    pub step: Q,
    /// Fractional bits of every rounded rational.
    pub bits: u32,
    /// Window constraints for every subset of at least two active classes
    /// instead of the declared subsets.
    pub all_subsets: bool,
    /// The visit-gap constraints in the full test.
    pub gap_rows: bool,
}

impl Options {
    /// The options matching grid settings.
    pub fn from_grid(settings: &GridSettings) -> Options {
        Options {
            step: settings.step.clone(),
            bits: settings.bits,
            all_subsets: settings.all_subsets,
            gap_rows: settings.gap_rows,
        }
    }
}

/// The results of one level: feasibility of the targets and the certified
/// lower bounds on the worst-class and the arrival-weighted miss probability.
#[derive(Clone, Debug)]
pub struct LevelResult {
    pub level: Level,
    /// The constraint groups imposed.
    pub groups: Vec<&'static str>,
    pub feasibility: Certified,
    pub min_max: Certified,
    pub min_avg: Certified,
    /// Rows of the feasibility LP.
    pub rows: usize,
}

impl LevelResult {
    /// The verdict of the feasibility LP: `certified_infeasible`,
    /// `undetermined` (a nonempty relaxation does not prove feasibility) or
    /// `numerically_inconclusive`.
    pub fn category(&self) -> &'static str {
        match self.feasibility.status.as_str() {
            "certified_infeasible" => "certified_infeasible",
            "relaxation_feasible" => "undetermined",
            _ => "numerically_inconclusive",
        }
    }

    /// Certified lower bound on the worst-class miss probability.
    pub fn lower_bound_max(&self) -> Option<f64> {
        self.min_max.certified_value.as_ref().map(to_f64)
    }

    /// Certified lower bound on the arrival-weighted miss probability.
    pub fn lower_bound_avg(&self) -> Option<f64> {
        self.min_avg.certified_value.as_ref().map(to_f64)
    }

    pub fn to_json(&self) -> Value {
        json!({
            "level": self.level.name(), "groups": self.groups, "category": self.category(), "rows": self.rows,
            "feasibility": self.feasibility.to_json(), "min_max": self.min_max.to_json(), "min_arrival_weighted": self.min_avg.to_json()
        })
    }
}

/// The evaluation of one model at a set of levels, with timings in
/// milliseconds.
#[derive(Clone, Debug)]
pub struct PointEvaluation {
    pub model: DiscreteModel,
    pub levels: Vec<LevelResult>,
    pub coefficient_ms: f64,
    pub window_ms: f64,
    pub lp_ms: f64,
    pub certificate_ms: f64,
    pub verify_ms: f64,
    /// Step of the lattice or grid of the coefficients.
    pub lattice_step: Q,
    /// Number of window constraints.
    pub windows: usize,
    /// Number of visit-gap constraints and the time to generate them, when
    /// they are requested.
    pub gaps: Option<(usize, f64)>,
}

impl PointEvaluation {
    /// The result at `level`, which must have been evaluated.
    pub fn level(&self, level: Level) -> &LevelResult {
        self.levels
            .iter()
            .find(|l| l.level == level)
            .expect("level evaluated")
    }

    pub fn to_json(&self) -> Value {
        let mut v = json!({
            "model": self.model.to_json(),
            "lattice_step": self.lattice_step.to_string(), "window_rows": self.windows,
            "timing_ms": {"coefficients": self.coefficient_ms, "windows": self.window_ms, "lp": self.lp_ms, "certificates": self.certificate_ms, "verification": self.verify_ms},
            "levels": self.levels.iter().map(|l| l.to_json()).collect::<Vec<_>>()
        });
        // Recorded only with the visit-gap constraints, so that the records of
        // the test without them are unchanged.
        if let Some((rows, ms)) = self.gaps {
            v["gap_rows"] = json!(rows);
            v["timing_ms"]["gaps"] = json!(ms);
        }
        v
    }
}

/// Phase coefficients of a finite-support model on its lattice, in the
/// certificate direction (independent of the arrival rates).
pub fn lattice_coefficients(
    model: &DiscreteModel,
    opts: &Options,
) -> Result<(Lattice, PhaseCoefficients), String> {
    let lat = lattice(model, &opts.step)?;
    let coef = phase_coefficients(model, &lat, Mode::Upper, opts.bits);
    Ok((lat, coef))
}

/// The window constraints of the declared window lengths and of the
/// declared (or, with `all_subsets`, all) class subsets.
fn window_rows_for(model: &DiscreteModel, opts: &Options) -> Vec<WindowRow> {
    let subsets = if opts.all_subsets {
        all_subsets(model)
    } else {
        default_subsets(model)
    };
    window_rows(model, &default_windows(model), &subsets, opts.bits)
}

/// The visit-gap constraints of every active class at the declared window
/// lengths and tangent points ([`gap_rows`]), when `opts.gap_rows` requests
/// them, with coefficients rounded to [`gaps::ROW_BITS`] fractional bits.
fn gap_rows_for(
    model: &DiscreteModel,
    coef: &PhaseCoefficients,
    opts: &Options,
) -> Option<Vec<GapRow>> {
    let bits = opts.bits.min(gaps::ROW_BITS);
    opts.gap_rows
        .then(|| gap_rows(model, &coef.mean_tau, &default_windows(model), bits))
}

/// Evaluate the relaxation of `model` with the coefficients `coef` at
/// every level of `levels`: feasibility of the targets and the certified
/// lower bounds on the worst-class and the arrival-weighted miss probability.
pub fn evaluate_with(
    model: &DiscreteModel,
    coef: &PhaseCoefficients,
    opts: &Options,
    levels: &[Level],
    coefficient_ms: f64,
) -> PointEvaluation {
    let t0 = Instant::now();
    let windows = window_rows_for(model, opts);
    let window_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = Instant::now();
    let gaps = gap_rows_for(model, coef, opts);
    let gap_ms = t1.elapsed().as_secs_f64() * 1e3;
    let ing = Ingredients {
        model,
        coef,
        windows: &windows,
        gaps: gaps.as_deref(),
    };
    let mut out = Vec::new();
    let (mut lp_ms, mut cert_ms, mut verify_ms) = (0.0, 0.0, 0.0);
    for &level in levels {
        let f = build(&ing, level, Objective::Feasibility);
        let feasibility = certify_feasibility(&f.lp);
        let mm = build(&ing, level, Objective::MinMax);
        let min_max = certify_minimum(&mm.lp);
        let ma = build(&ing, level, Objective::MinArrivalWeighted);
        let min_avg = certify_minimum(&ma.lp);
        lp_ms += feasibility.float_ms + min_max.float_ms + min_avg.float_ms;
        cert_ms += feasibility.exact_ms + min_max.exact_ms + min_avg.exact_ms;
        verify_ms += feasibility.verify_ms + min_max.verify_ms + min_avg.verify_ms;
        out.push(LevelResult {
            level,
            groups: level.groups_with(gaps.is_some()),
            rows: f.lp.rows.len(),
            feasibility,
            min_max,
            min_avg,
        });
    }
    PointEvaluation {
        model: model.clone(),
        levels: out,
        coefficient_ms,
        window_ms,
        lp_ms,
        certificate_ms: cert_ms,
        verify_ms,
        lattice_step: coef.g.clone(),
        windows: windows.len(),
        gaps: gaps.as_ref().map(|g| (g.len(), gap_ms)),
    }
}

/// Evaluate a finite-support model at the four levels through its lattice
/// coefficients.
pub fn evaluate_discrete(model: &DiscreteModel, opts: &Options) -> Result<PointEvaluation, String> {
    let t0 = Instant::now();
    let (_, coef) = lattice_coefficients(model, opts)?;
    let coefficient_ms = t0.elapsed().as_secs_f64() * 1e3;
    Ok(evaluate_with(
        model,
        &coef,
        opts,
        &Level::all(),
        coefficient_ms,
    ))
}

/// The evaluation of a model with continuous distributions: the levels,
/// both coefficient directions and the diagnostics of the grid.
#[derive(Clone, Debug)]
pub struct Evaluation {
    pub evaluation: PointEvaluation,
    /// Lower coefficients (the certificate direction).
    pub lower: PhaseCoefficients,
    /// Upper coefficients.
    pub upper: PhaseCoefficients,
    pub diagnostics: Value,
    pub cache_hit: bool,
}

/// Evaluate a model: grid coefficients (read from or written to the cache
/// directory when one is given), then the relaxation at every level of
/// `levels`, assembled on the model's LP shell ([`Model::lp_shell`]).
pub fn evaluate(
    model: &Model,
    settings: &GridSettings,
    levels: &[Level],
    cache: Option<&std::path::Path>,
) -> Result<Evaluation, String> {
    let t0 = Instant::now();
    let (lower, upper, diagnostics, cache_hit) =
        grid::grid_coefficients_cached(model, settings, cache)?;
    let coefficient_ms = t0.elapsed().as_secs_f64() * 1e3;
    let shell = model.lp_shell(settings.bits);
    let opts = Options::from_grid(settings);
    let evaluation = evaluate_with(&shell, &lower, &opts, levels, coefficient_ms);
    Ok(Evaluation {
        evaluation,
        lower,
        upper,
        diagnostics,
        cache_hit,
    })
}

/// Whether the long-run vector `(nu_uv = nu_pair for every pair, z_j = z,
/// x_i = x)` satisfies every row of the relaxation with the target rows at
/// `level`, checked exactly. Membership is necessary, not sufficient: it
/// does not show that any policy attains these long-run quantities.
pub fn point_in_relaxation(
    shell: &DiscreteModel,
    coef: &PhaseCoefficients,
    opts: &Options,
    level: Level,
    nu_pair: &Q,
    z: &Q,
    x: &Q,
) -> Result<(), String> {
    let windows = window_rows_for(shell, opts);
    let gaps = gap_rows_for(shell, coef, opts);
    let ing = Ingredients {
        model: shell,
        coef,
        windows: &windows,
        gaps: gaps.as_deref(),
    };
    let rel = build(&ing, level, Objective::Feasibility);
    let mut y = vec![Q::zero(); rel.lp.nvars];
    for row in &rel.nu {
        for k in row.iter().flatten() {
            y[*k] = nu_pair.clone();
        }
    }
    for &k in &rel.z {
        y[k] = z.clone();
    }
    for &k in &rel.x {
        y[k] = x.clone();
    }
    simplex::verify_primal(&rel.lp, &y).map(|_| ())
}

/// A necessary operating range of one long-run rate (the total
/// reconfiguration rate, or the entry rate into one configuration) over the
/// relaxation with every target row retained. The true minimum over the
/// region is at least `lo()`, and the true maximum is at most `hi()`, the
/// negation of the certified lower bound on the minimum of the negated rate.
/// An endpoint equals the LP extremum only when the corresponding
/// `exact_optimum` flag is set; otherwise it is a conservative certified
/// bound. The range is necessary only: it does not construct a policy, the
/// points between the endpoints need not be attainable, and the separate
/// ranges do not describe the joint region.
#[derive(Clone, Debug)]
pub struct RateInterval {
    pub name: String,
    pub lower: Certified,
    pub upper: Certified,
}

impl RateInterval {
    /// The certified lower end, when the minimum is certified.
    pub fn lo(&self) -> Option<Q> {
        if self.lower.status == "certified_lower_bound" {
            self.lower.certified_value.clone()
        } else {
            None
        }
    }

    /// The certified upper end, when the minimum of the negated rate is
    /// certified.
    pub fn hi(&self) -> Option<Q> {
        if self.upper.status == "certified_lower_bound" {
            self.upper.certified_value.as_ref().map(|v| -v)
        } else {
            None
        }
    }

    /// Whether either LP is certified infeasible, i.e. the region with the
    /// targets is empty.
    pub fn empty(&self) -> bool {
        self.lower.status == "certified_infeasible" || self.upper.status == "certified_infeasible"
    }

    pub fn to_json(&self) -> Value {
        let q = |v: Option<Q>| {
            v.map(|v| json!({"exact": v.to_string(), "f64": to_f64(&v)}))
                .unwrap_or(Value::Null)
        };
        json!({
            "name": self.name, "empty": self.empty(), "lo": q(self.lo()), "hi": q(self.hi()),
            "lo_is_lp_extremum": self.lower.exact_optimum, "hi_is_lp_extremum": self.upper.exact_optimum,
            "lower": self.lower.to_json(), "upper": self.upper.to_json()
        })
    }
}

/// Certified operating ranges at one level: the total reconfiguration rate,
/// then the entry rate of every configuration.
pub fn rate_intervals(
    shell: &DiscreteModel,
    coef: &PhaseCoefficients,
    opts: &Options,
    level: Level,
) -> Vec<RateInterval> {
    let windows = window_rows_for(shell, opts);
    let gaps = gap_rows_for(shell, coef, opts);
    let ing = Ingredients {
        model: shell,
        coef,
        windows: &windows,
        gaps: gaps.as_deref(),
    };
    let pair = |name: String, min_obj: Objective, max_obj: Objective| RateInterval {
        name,
        lower: certify_minimum(&build(&ing, level, min_obj).lp),
        upper: certify_minimum(&build(&ing, level, max_obj).lp),
    };
    let mut out = vec![pair(
        "total".into(),
        Objective::TotalRate { maximize: false },
        Objective::TotalRate { maximize: true },
    )];
    for i in 0..shell.n {
        out.push(pair(
            format!("entry[{i}]"),
            Objective::EntryRate {
                class: i,
                maximize: false,
            },
            Objective::EntryRate {
                class: i,
                maximize: true,
            },
        ));
    }
    out
}
