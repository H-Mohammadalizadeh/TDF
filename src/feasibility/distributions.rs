//! Reconfiguration-time distributions with exact rational parameters (laws,
//! in identifiers) and rigorous enclosures of `P(tau < x)`, `P(tau > x)`, the
//! mean and the variance. [`profile`] builds the six named distributions
//! (profiles) from a mean and a squared coefficient of variation. [`Model`]
//! carries the classes and one distribution per ordered configuration pair
//! (an edge); the checker reads it through [`Model::lp_shell`] and the grid
//! coefficients, the simulator through [`Model::system_json`], whose `switch`
//! entry holds the reconfiguration-time distributions.

use super::model::{DiscreteLaw, DiscreteModel};
use super::rational::{Q, exp_neg, q_from_json, qi, qr, to_f64};
use super::special::{
    gamma_lower_regularized_enclosure, idiv, imul, isub, ln_enclosure, norm_cdf_enclosure,
    norm_pdf_enclosure, point, pow_rational_enclosure, sqrt_enclosure,
};
use num_traits::{One, Signed, ToPrimitive, Zero};
use serde_json::{Map, Value, json};

/// A reconfiguration-time distribution.
#[derive(Clone, Debug, PartialEq)]
pub enum Distribution {
    /// A point mass.
    Deterministic { value: Q },
    /// A finite-support distribution with exact atoms (for cross-checks of the
    /// grid construction against the lattice construction).
    Discrete(DiscreteLaw),
    /// Mean `1 / rate`.
    Exponential { rate: Q },
    /// Shape `k` (any positive rational; Erlang when integer), rate `r`:
    /// mean `k / r`, variance `k / r^2`.
    Gamma { shape: Q, rate: Q },
    /// The normal distribution `N(mu0, sigma0^2)` conditioned on a positive
    /// outcome.
    TruncNormal { mu0: Q, sigma0: Q },
    /// Parameterized by its exact mean and squared coefficient of variation.
    Lognormal { mean: Q, cv2: Q },
    /// Type I: `P(tau > x) = (xm / x)^alpha` for `x >= xm`; mean
    /// `alpha xm / (alpha - 1)` (needs `alpha > 1`).
    Pareto { alpha: Q, xm: Q },
}

/// Both ends of an enclosure clamped to `[0, 1]`.
fn clip01(e: (Q, Q)) -> (Q, Q) {
    (
        e.0.max(Q::zero()).min(Q::one()),
        e.1.max(Q::zero()).min(Q::one()),
    )
}

/// The Erlang survival function `P(tau > x) = exp(-r x) sum_{j<k} (r x)^j / j!`
/// for shape `k` and rate `r`, with directed rounding.
fn erlang_survival(k: u32, rate: &Q, x: &Q, bits: u32) -> (Q, Q) {
    let rx = rate * x;
    let mut poly = Q::zero();
    let mut term = Q::one();
    for j in 0..k {
        if j > 0 {
            term = term * &rx / qi(j as i64);
        }
        poly += &term;
    }
    let (lo, hi) = exp_neg(&rx, bits + 8);
    (
        super::rational::round_down(&(lo * &poly), bits),
        super::rational::round_up(&(hi * &poly), bits),
    )
}

/// The inverse Mills ratio `lambda(alpha) = phi(alpha) / (1 - Phi(alpha))` of
/// the standard normal distribution at rational `alpha`.
fn mills(alpha: &Q, bits: u32) -> (Q, Q) {
    let work = bits + 32;
    let pdf = norm_pdf_enclosure(alpha, work);
    let cdf = norm_cdf_enclosure(alpha, work);
    let tail = (Q::one() - &cdf.1, Q::one() - &cdf.0);
    idiv(&pdf, &tail, bits)
}

impl Distribution {
    /// The family name used in JSON.
    pub fn family(&self) -> &'static str {
        match self {
            Distribution::Deterministic { .. } => "deterministic",
            Distribution::Discrete(_) => "discrete",
            Distribution::Exponential { .. } => "exponential",
            Distribution::Gamma { .. } => "gamma",
            Distribution::TruncNormal { .. } => "truncated_normal",
            Distribution::Lognormal { .. } => "lognormal",
            Distribution::Pareto { .. } => "pareto",
        }
    }

    /// A short exact identifier: it recognizes equal distributions (the grid's
    /// probability tables, symmetry checks) and labels output records.
    pub fn id(&self) -> String {
        match self {
            Distribution::Deterministic { value } => format!("det[{value}]"),
            Distribution::Discrete(l) => format!(
                "discrete[{}]",
                l.atoms
                    .iter()
                    .map(|(s, p)| format!("{s}:{p}"))
                    .collect::<Vec<_>>()
                    .join(";")
            ),
            Distribution::Exponential { rate } => format!("exp[{rate}]"),
            Distribution::Gamma { shape, rate } => format!("gamma[{shape},{rate}]"),
            Distribution::TruncNormal { mu0, sigma0 } => format!("tnorm[{mu0},{sigma0}]"),
            Distribution::Lognormal { mean, cv2 } => format!("lognormal[{mean},{cv2}]"),
            Distribution::Pareto { alpha, xm } => format!("pareto[{alpha},{xm}]"),
        }
    }

    /// Whether the distribution has finite support (the lattice construction
    /// applies).
    pub fn is_finite_support(&self) -> bool {
        matches!(
            self,
            Distribution::Deterministic { .. } | Distribution::Discrete(_)
        )
    }

    /// The lower end of the support.
    pub fn min_support(&self) -> Q {
        match self {
            Distribution::Deterministic { value } => value.clone(),
            Distribution::Discrete(l) => l.min_support(),
            Distribution::Pareto { xm, .. } => xm.clone(),
            _ => Q::zero(),
        }
    }

    /// Support points of a finite-support distribution (empty otherwise).
    pub fn support_points(&self) -> Vec<Q> {
        match self {
            Distribution::Deterministic { value } => vec![value.clone()],
            Distribution::Discrete(l) => l.atoms.iter().map(|(s, _)| s.clone()).collect(),
            _ => Vec::new(),
        }
    }

    /// The distribution with the constants of its distribution function that
    /// do not depend on `x` enclosed once (the truncation mass of the
    /// truncated normal, the lognormal parameters), for repeated evaluations
    /// on a grid.
    pub fn prepare(&self, bits: u32) -> Prepared<'_> {
        let work = bits + 32;
        let tn = match self {
            Distribution::TruncNormal { mu0, sigma0 } => {
                let alpha = -mu0 / sigma0;
                let phi_a = norm_cdf_enclosure(&alpha, work);
                let den = (Q::one() - &phi_a.1, Q::one() - &phi_a.0);
                Some((phi_a, den))
            }
            _ => None,
        };
        let ln = match self {
            Distribution::Lognormal { mean, cv2 } => {
                let sigma2 = ln_enclosure(&(Q::one() + cv2), work);
                let sigma = (
                    sqrt_enclosure(&sigma2.0, work).0,
                    sqrt_enclosure(&sigma2.1, work).1,
                );
                let mu = isub(
                    &ln_enclosure(mean, work),
                    &imul(&sigma2, &point(&qr(1, 2)), work),
                    work,
                );
                Some((sigma, mu))
            }
            _ => None,
        };
        Prepared {
            law: self,
            bits,
            tn,
            ln,
        }
    }

    /// `lo <= P(tau < x) <= hi`. A gamma distribution of integer shape up to
    /// 64 uses the Erlang closed form, any other shape the incomplete gamma
    /// series.
    pub fn below(&self, x: &Q, bits: u32) -> (Q, Q) {
        if !x.is_positive() {
            return (Q::zero(), Q::zero());
        }
        match self {
            Distribution::Deterministic { value } => {
                if value < x {
                    (Q::one(), Q::one())
                } else {
                    (Q::zero(), Q::zero())
                }
            }
            Distribution::Discrete(l) => {
                let p: Q = l
                    .atoms
                    .iter()
                    .filter(|(s, _)| s < x)
                    .map(|(_, p)| p.clone())
                    .sum();
                (p.clone(), p)
            }
            Distribution::Exponential { rate } => {
                let (lo, hi) = exp_neg(&(rate * x), bits);
                (Q::one() - hi, Q::one() - lo)
            }
            Distribution::Gamma { shape, rate } => {
                if shape.is_integer() && *shape <= qi(64) {
                    let k = shape.to_integer().to_u32().unwrap();
                    let (lo, hi) = erlang_survival(k, rate, x, bits);
                    (Q::one() - hi, Q::one() - lo)
                } else {
                    gamma_lower_regularized_enclosure(shape, &(rate * x), bits)
                }
            }
            Distribution::TruncNormal { .. } | Distribution::Lognormal { .. } => {
                self.prepare(bits).below(x)
            }
            Distribution::Pareto { alpha, xm } => {
                if x <= xm {
                    return (Q::zero(), Q::zero());
                }
                let (lo, hi) = pow_rational_enclosure(&(xm / x), alpha, bits);
                clip01((Q::one() - hi, Q::one() - lo))
            }
        }
    }

    /// `lo <= P(tau > x) <= hi`.
    pub fn survival(&self, x: &Q, bits: u32) -> (Q, Q) {
        match self {
            Distribution::Deterministic { value } => {
                if value > x {
                    (Q::one(), Q::one())
                } else {
                    (Q::zero(), Q::zero())
                }
            }
            Distribution::Discrete(l) => {
                let p = l.survival(x);
                (p.clone(), p)
            }
            Distribution::Exponential { rate } => {
                if !x.is_positive() {
                    return (Q::one(), Q::one());
                }
                exp_neg(&(rate * x), bits)
            }
            _ => {
                // continuous: P(tau = x) = 0
                let (lo, hi) = self.below(x, bits);
                (Q::one() - hi, Q::one() - lo)
            }
        }
    }

    /// `lo <= E[tau] <= hi` (exact for every family except the truncated normal).
    pub fn mean(&self, bits: u32) -> (Q, Q) {
        match self {
            Distribution::Deterministic { value } => point(value),
            Distribution::Discrete(l) => point(&l.mean()),
            Distribution::Exponential { rate } => point(&rate.recip()),
            Distribution::Gamma { shape, rate } => point(&(shape / rate)),
            Distribution::Lognormal { mean, .. } => point(mean),
            Distribution::Pareto { alpha, xm } => point(&(alpha * xm / (alpha - Q::one()))),
            Distribution::TruncNormal { mu0, sigma0 } => {
                // mu0 + sigma0 lam with the inverse Mills ratio at alpha = -mu0 / sigma0
                let work = bits + 32;
                let alpha = -mu0 / sigma0;
                let lam = mills(&alpha, work);
                let e = imul(&lam, &point(sigma0), work);
                super::special::iadd(&point(mu0), &e, bits)
            }
        }
    }

    /// `lo <= Var[tau] <= hi`, or `None` when the variance is infinite.
    pub fn variance(&self, bits: u32) -> Option<(Q, Q)> {
        Some(match self {
            Distribution::Deterministic { .. } => point(&Q::zero()),
            Distribution::Discrete(l) => point(&l.variance()),
            Distribution::Exponential { rate } => point(&(rate * rate).recip()),
            Distribution::Gamma { shape, rate } => point(&(shape / (rate * rate))),
            Distribution::Lognormal { mean, cv2 } => point(&(mean * mean * cv2)),
            Distribution::Pareto { alpha, xm } => {
                if *alpha <= qi(2) {
                    return None;
                }
                let am1 = alpha - Q::one();
                point(&(alpha * xm * xm / (&am1 * &am1 * (alpha - qi(2)))))
            }
            Distribution::TruncNormal { mu0, sigma0 } => {
                let work = bits + 32;
                let alpha = -mu0 / sigma0;
                let lam = mills(&alpha, work);
                // sigma0^2 (1 + alpha lam - lam^2)
                let al = imul(&point(&alpha), &lam, work);
                let l2 = imul(&lam, &lam, work);
                let inner = isub(
                    &super::special::iadd(&point(&Q::one()), &al, work),
                    &l2,
                    work,
                );
                imul(&inner, &point(&(sigma0 * sigma0)), bits)
            }
        })
    }

    /// Enclosure of the squared coefficient of variation (`None` when the
    /// variance is infinite).
    pub fn cv2(&self, bits: u32) -> Option<(Q, Q)> {
        let v = self.variance(bits)?;
        let m = self.mean(bits);
        let m2 = imul(&m, &m, bits + 32);
        Some(idiv(&v, &m2, bits))
    }

    /// Every duration multiplied by `k > 0`.
    pub fn scaled(&self, k: &Q) -> Distribution {
        match self {
            Distribution::Deterministic { value } => {
                Distribution::Deterministic { value: value * k }
            }
            Distribution::Discrete(l) => Distribution::Discrete(l.scaled(k)),
            Distribution::Exponential { rate } => Distribution::Exponential { rate: rate / k },
            Distribution::Gamma { shape, rate } => Distribution::Gamma {
                shape: shape.clone(),
                rate: rate / k,
            },
            Distribution::TruncNormal { mu0, sigma0 } => Distribution::TruncNormal {
                mu0: mu0 * k,
                sigma0: sigma0 * k,
            },
            Distribution::Lognormal { mean, cv2 } => Distribution::Lognormal {
                mean: mean * k,
                cv2: cv2.clone(),
            },
            Distribution::Pareto { alpha, xm } => Distribution::Pareto {
                alpha: alpha.clone(),
                xm: xm * k,
            },
        }
    }

    /// Exact JSON (rationals as strings).
    pub fn to_json(&self) -> Value {
        match self {
            Distribution::Deterministic { value } => {
                json!({"family": "deterministic", "value": value.to_string()})
            }
            Distribution::Discrete(l) => {
                let mut v = l.to_json();
                v["family"] = json!("discrete");
                v
            }
            Distribution::Exponential { rate } => {
                json!({"family": "exponential", "rate": rate.to_string()})
            }
            Distribution::Gamma { shape, rate } => {
                json!({"family": "gamma", "shape": shape.to_string(), "rate": rate.to_string()})
            }
            Distribution::TruncNormal { mu0, sigma0 } => {
                json!({"family": "truncated_normal", "mu0": mu0.to_string(), "sigma0": sigma0.to_string()})
            }
            Distribution::Lognormal { mean, cv2 } => {
                json!({"family": "lognormal", "mean": mean.to_string(), "cv2": cv2.to_string()})
            }
            Distribution::Pareto { alpha, xm } => {
                json!({"family": "pareto", "alpha": alpha.to_string(), "xm": xm.to_string()})
            }
        }
    }

    /// A distribution from the exact JSON of [`Distribution::to_json`].
    pub fn from_json(v: &Value) -> Result<Distribution, String> {
        let q = |k: &str| -> Result<Q, String> {
            q_from_json(v.get(k).ok_or_else(|| format!("distribution needs {k}"))?)
        };
        match v["family"].as_str().unwrap_or("") {
            "deterministic" => Ok(Distribution::Deterministic { value: q("value")? }),
            "discrete" => Ok(Distribution::Discrete(DiscreteLaw::from_json(v)?)),
            "exponential" => Ok(Distribution::Exponential { rate: q("rate")? }),
            "gamma" => Ok(Distribution::Gamma {
                shape: q("shape")?,
                rate: q("rate")?,
            }),
            "truncated_normal" => Ok(Distribution::TruncNormal {
                mu0: q("mu0")?,
                sigma0: q("sigma0")?,
            }),
            "lognormal" => Ok(Distribution::Lognormal {
                mean: q("mean")?,
                cv2: q("cv2")?,
            }),
            "pareto" => Ok(Distribution::Pareto {
                alpha: q("alpha")?,
                xm: q("xm")?,
            }),
            other => Err(format!("unknown distribution family {other:?}")),
        }
    }

    /// The distribution in the simulator's JSON format (doubles), as parsed by
    /// `sim::laws`.
    pub fn to_engine_json(&self) -> Value {
        match self {
            Distribution::Deterministic { value } => {
                json!({"family": "deterministic", "value": to_f64(value)})
            }
            Distribution::Discrete(l) => l.to_engine_json(),
            Distribution::Exponential { rate } => {
                json!({"family": "exponential", "mean": to_f64(&rate.recip())})
            }
            Distribution::Gamma { shape, rate } => {
                json!({"family": "gamma", "shape": to_f64(shape), "scale": to_f64(&rate.recip())})
            }
            Distribution::TruncNormal { mu0, sigma0 } => {
                json!({"family": "truncated_normal", "mu0": to_f64(mu0), "sigma0": to_f64(sigma0)})
            }
            Distribution::Lognormal { mean, cv2 } => {
                json!({"family": "lognormal", "mean": to_f64(mean), "cv": to_f64(cv2).sqrt()})
            }
            Distribution::Pareto { alpha, xm } => {
                json!({"family": "pareto", "shape": to_f64(alpha), "min": to_f64(xm)})
            }
        }
    }

    /// Human-readable parameters with double approximations of the moments.
    pub fn describe(&self, bits: u32) -> Value {
        let m = self.mean(bits);
        let v = self.variance(bits);
        json!({
            "family": self.family(), "id": self.id(), "exact": self.to_json(),
            "mean": [to_f64(&m.0), to_f64(&m.1)],
            "variance": v.as_ref().map(|v| json!([to_f64(&v.0), to_f64(&v.1)])).unwrap_or(Value::Null),
            "cv2": self.cv2(bits).map(|c| json!([to_f64(&c.0), to_f64(&c.1)])).unwrap_or(Value::Null),
            "min_support": to_f64(&self.min_support())
        })
    }
}

/// A distribution with its `x`-independent constants enclosed once.
pub struct Prepared<'a> {
    pub law: &'a Distribution,
    pub bits: u32,
    /// Truncated normal: `Phi(alpha)` and `1 - Phi(alpha)`.
    tn: Option<((Q, Q), (Q, Q))>,
    /// Lognormal: `sigma` and `mu`.
    ln: Option<((Q, Q), (Q, Q))>,
}

impl Prepared<'_> {
    /// `lo <= P(tau < x) <= hi`, using the prepared constants.
    pub fn below(&self, x: &Q) -> (Q, Q) {
        if !x.is_positive() {
            return (Q::zero(), Q::zero());
        }
        let bits = self.bits;
        let work = bits + 32;
        match self.law {
            Distribution::TruncNormal { mu0, sigma0 } => {
                // (Phi(beta) - Phi(alpha)) / (1 - Phi(alpha)) with beta = (x - mu0) / sigma0
                let (phi_a, den) = self.tn.as_ref().expect("prepared");
                let beta = (x - mu0) / sigma0;
                let phi_b = norm_cdf_enclosure(&beta, work);
                let num = isub(&phi_b, phi_a, work);
                clip01(idiv(&num, den, bits))
            }
            Distribution::Lognormal { .. } => {
                let (sigma, mu) = self.ln.as_ref().expect("prepared");
                let arg = idiv(&isub(&ln_enclosure(x, work), mu, work), sigma, work);
                // Phi((ln x - mu) / sigma). Phi is increasing with slope at most
                // 1/sqrt(2 pi) < 2/5, so one evaluation at the lower end of the
                // argument enclosure suffices.
                let base = norm_cdf_enclosure(&arg.0, work);
                let slack = super::rational::round_up(&(qr(2, 5) * (&arg.1 - &arg.0)), work);
                clip01((base.0, base.1 + slack))
            }
            other => other.below(x, bits),
        }
    }

    /// `(below, survival)` at `x`. Finite-support and exponential
    /// distributions evaluate both directly; for the others the survival
    /// enclosure is the complement of the single evaluation of `below`.
    pub fn both(&self, x: &Q) -> ((Q, Q), (Q, Q)) {
        if self.law.is_finite_support() || matches!(self.law, Distribution::Exponential { .. }) {
            return (self.below(x), self.law.survival(x, self.bits));
        }
        let (lo, hi) = self.below(x);
        ((lo.clone(), hi.clone()), (Q::one() - hi, Q::one() - lo))
    }
}

/// Rational square root when it exists.
fn rational_sqrt(x: &Q) -> Option<Q> {
    if x.is_negative() {
        return None;
    }
    let n = x.numer().sqrt();
    let d = x.denom().sqrt();
    if &n * &n == *x.numer() && &d * &d == *x.denom() {
        Some(Q::new(n, d))
    } else {
        None
    }
}

/// The named distribution (profile) `name` with the given mean: `det`, `exp`,
/// `gamma`, `tnorm`, `lognormal` or `pareto` (also accepted: `deterministic`,
/// `exponential`, `erlang`, `truncated_normal`). All but `det` and `exp` also
/// match the squared coefficient of variation `cv2`; `pareto` needs
/// `1 + 1/cv2` to be a rational square. The truncated normal takes the parent
/// parameters fitted in doubles by `sim::laws::truncnormal_parent`, so it
/// matches the mean and `cv2` only approximately.
pub fn profile(name: &str, mean: &Q, cv2: &Q) -> Result<Distribution, String> {
    if !mean.is_positive() {
        return Err("the mean must be positive".into());
    }
    match name {
        "det" | "deterministic" => Ok(Distribution::Deterministic {
            value: mean.clone(),
        }),
        "exp" | "exponential" => Ok(Distribution::Exponential { rate: mean.recip() }),
        "gamma" | "erlang" => {
            if !cv2.is_positive() {
                return Err("gamma needs a positive squared coefficient of variation".into());
            }
            let shape = cv2.recip();
            Ok(Distribution::Gamma {
                rate: &shape / mean,
                shape,
            })
        }
        "tnorm" | "truncated_normal" => {
            let (mu0, sigma0) =
                crate::sim::laws::truncnormal_parent(to_f64(mean), to_f64(cv2).sqrt())?;
            Ok(Distribution::TruncNormal {
                mu0: Q::from_float(mu0).ok_or("mu0 not finite")?,
                sigma0: Q::from_float(sigma0).ok_or("sigma0 not finite")?,
            })
        }
        "lognormal" => {
            if !cv2.is_positive() {
                return Err("lognormal needs a positive squared coefficient of variation".into());
            }
            Ok(Distribution::Lognormal {
                mean: mean.clone(),
                cv2: cv2.clone(),
            })
        }
        "pareto" => {
            if !cv2.is_positive() {
                return Err("pareto needs a positive squared coefficient of variation".into());
            }
            // cv2 = 1 / (alpha (alpha - 2))  =>  alpha = 1 + sqrt(1 + 1 / cv2)
            let s = rational_sqrt(&(Q::one() + cv2.recip())).ok_or_else(|| {
                format!(
                    "pareto: 1 + 1/cv2 = {} is not a rational square",
                    Q::one() + cv2.recip()
                )
            })?;
            let alpha = Q::one() + s;
            Ok(pareto_with_alpha(mean, &alpha))
        }
        other => Err(format!("unknown profile {other:?}")),
    }
}

/// Pareto with a prescribed shape (`alpha > 1`) and the given mean.
pub fn pareto_with_alpha(mean: &Q, alpha: &Q) -> Distribution {
    assert!(*alpha > Q::one());
    Distribution::Pareto {
        alpha: alpha.clone(),
        xm: mean * (alpha - Q::one()) / alpha,
    }
}

/// The short names of the six profiles.
pub fn profile_names() -> [&'static str; 6] {
    ["det", "exp", "gamma", "tnorm", "lognormal", "pareto"]
}

/// The model: `N` classes with Poisson arrivals, exponential service,
/// deadlines and targets, and one reconfiguration-time distribution per
/// ordered configuration pair.
#[derive(Clone, Debug)]
pub struct Model {
    pub name: String,
    pub n: usize,
    pub lam: Vec<Q>,
    pub mu: Vec<Q>,
    pub d: Vec<Q>,
    pub eps: Vec<Q>,
    pub laws: Vec<Vec<Option<Distribution>>>,
    /// Service-time distribution of the simulated network: `None` or
    /// `"exponential"` (as in the model), `"deterministic"` or
    /// `"lognormal:<cv>"`. The last two lie outside the model, and the
    /// certificates then serve as reference predictions only.
    pub service: Option<String>,
    /// Mean geometric batch size of the simulated arrivals (outside the
    /// model); `None` gives the model's Poisson arrivals of single jobs.
    pub batch_mean: Option<Q>,
}

impl Model {
    /// A model with validated dimensions and parameters, exponential service
    /// and single arrivals.
    pub fn new(
        name: &str,
        lam: Vec<Q>,
        mu: Vec<Q>,
        d: Vec<Q>,
        eps: Vec<Q>,
        laws: Vec<Vec<Option<Distribution>>>,
    ) -> Result<Model, String> {
        let n = mu.len();
        if n == 0 || lam.len() != n || d.len() != n || eps.len() != n || laws.len() != n {
            return Err("inconsistent dimensions".into());
        }
        for i in 0..n {
            if !mu[i].is_positive() || lam[i].is_negative() || !d[i].is_positive() {
                return Err(
                    "rates must be positive, arrival rates nonnegative, deadlines positive".into(),
                );
            }
            if !eps[i].is_positive() || eps[i] >= Q::one() {
                return Err("targets must lie strictly between zero and one".into());
            }
            if laws[i].len() != n {
                return Err("laws must be N x N".into());
            }
            for j in 0..n {
                if (i == j) != laws[i][j].is_none() {
                    return Err("laws: None exactly on the diagonal".into());
                }
            }
        }
        Ok(Model {
            name: name.into(),
            n,
            lam,
            mu,
            d,
            eps,
            laws,
            service: None,
            batch_mean: None,
        })
    }

    /// The simulator's service-time distribution of every class, or `None`
    /// for exponential service.
    fn service_json(&self) -> Option<Value> {
        let spec = self.service.as_deref()?;
        if spec == "exponential" {
            return None;
        }
        let laws: Vec<Value> = (0..self.n)
            .map(|i| {
                let m = to_f64(&self.mu[i].recip());
                if let Some(cv) = spec.strip_prefix("lognormal:") {
                    json!({"family": "lognormal", "mean": m, "cv": cv.parse::<f64>().unwrap_or(f64::NAN)})
                } else {
                    json!({"family": spec, "mean": m})
                }
            })
            .collect();
        Some(Value::Array(laws))
    }

    /// `N` identical classes with `mu_i = 4`, `lambda_i = 4 rho / N`, a common
    /// deadline and target, and one distribution on every configuration pair.
    pub fn balanced(name: &str, law: &Distribution, n: usize, rho: &Q, d: &Q, eps: &Q) -> Model {
        let mu = vec![qi(4); n];
        let lam = vec![rho * qi(4) / qi(n as i64); n];
        let laws = (0..n)
            .map(|u| (0..n).map(|v| (u != v).then(|| law.clone())).collect())
            .collect();
        Model::new(
            name,
            lam,
            mu,
            vec![d.clone(); n],
            vec![eps.clone(); n],
            laws,
        )
        .unwrap()
    }

    /// The ordered configuration pairs `(u, v)`, `u != v`, row by row.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        (0..self.n)
            .flat_map(|u| (0..self.n).filter(move |v| *v != u).map(move |v| (u, v)))
            .collect()
    }

    /// The reconfiguration-time distribution of the pair `u -> v`.
    pub fn law(&self, u: usize, v: usize) -> &Distribution {
        self.laws[u][v].as_ref().expect("off-diagonal pair")
    }

    /// The load `lambda_i / mu_i` of class `i`.
    pub fn rho(&self, i: usize) -> Q {
        &self.lam[i] / &self.mu[i]
    }

    /// The total offered load.
    pub fn total_rho(&self) -> Q {
        (0..self.n).map(|i| self.rho(i)).sum()
    }

    /// The same rates, deadlines and targets as a [`DiscreteModel`], with a
    /// point mass at the lower end of the mean enclosure (at least `10^-6`) on
    /// every pair. The relaxation and the window constraints never read the
    /// distributions, so this shell lets the shared code assemble them;
    /// coefficients must never be computed from it.
    pub fn lp_shell(&self, bits: u32) -> DiscreteModel {
        let laws = (0..self.n)
            .map(|u| {
                (0..self.n)
                    .map(|v| {
                        (u != v).then(|| {
                            DiscreteLaw::point(self.law(u, v).mean(bits).0.max(qr(1, 1_000_000)))
                        })
                    })
                    .collect()
            })
            .collect();
        DiscreteModel::new(
            &format!("{}|lp-shell", self.name),
            self.lam.clone(),
            self.mu.clone(),
            self.d.clone(),
            self.eps.clone(),
            laws,
        )
        .unwrap()
    }

    /// The simulator's system JSON with the given warm-up, measurement and
    /// follow-up durations; `records` raises the completion-trace limit from
    /// 64 to `10^9` entries.
    pub fn system_json(&self, warmup: f64, measure: f64, followup: f64, records: bool) -> Value {
        let mut edges = Map::new();
        for (u, v) in self.edges() {
            edges.insert(format!("{u},{v}"), self.law(u, v).to_engine_json());
        }
        let mut v = json!({
            "mu": self.mu.iter().map(to_f64).collect::<Vec<_>>(),
            "lambda": self.lam.iter().map(to_f64).collect::<Vec<_>>(),
            "deadlines": self.d.iter().map(to_f64).collect::<Vec<_>>(),
            "epsilon": self.eps.iter().map(to_f64).collect::<Vec<_>>(),
            "switch": {"default": if self.n > 1 { self.law(0, 1).to_engine_json() } else { json!({"family": "deterministic", "value": 0.0}) }, "edges": edges},
            "warmup": warmup, "measure": measure, "followup": followup,
            "backlog_samples": 201, "trace_limit": if records { 1_000_000_000u64 } else { 64 },
            "max_events": Value::Null, "max_jobs": Value::Null, "max_wall_seconds": Value::Null
        });
        if let Some(service) = self.service_json() {
            v["service"] = service;
        }
        if let Some(b) = &self.batch_mean {
            v["batch_mean"] = json!(to_f64(b));
        }
        v
    }

    pub fn to_json(&self) -> Value {
        let mut edges = Map::new();
        for (u, v) in self.edges() {
            edges.insert(format!("{u},{v}"), self.law(u, v).to_json());
        }
        json!({
            "name": self.name, "n": self.n,
            "lambda": self.lam.iter().map(|x| x.to_string()).collect::<Vec<_>>(),
            "mu": self.mu.iter().map(|x| x.to_string()).collect::<Vec<_>>(),
            "deadlines": self.d.iter().map(|x| x.to_string()).collect::<Vec<_>>(),
            "epsilon": self.eps.iter().map(|x| x.to_string()).collect::<Vec<_>>(),
            "rho": self.total_rho().to_string(),
            "edges": edges,
            "service": self.service,
            "batch_mean": self.batch_mean.as_ref().map(|b| b.to_string())
        })
    }

    /// A model from JSON with the fields of [`Model::to_json`]. `epsilon` may
    /// be a list or a single value (default 1/100), and pairs missing from
    /// `edges` take the distribution under `default`.
    pub fn from_json(v: &Value) -> Result<Model, String> {
        let list = |key: &str| -> Result<Vec<Q>, String> {
            v[key]
                .as_array()
                .ok_or_else(|| format!("{key} must be a list"))?
                .iter()
                .map(q_from_json)
                .collect()
        };
        let mu = list("mu")?;
        let n = mu.len();
        let lam = list("lambda")?;
        let d = list("deadlines")?;
        let eps = match v.get("epsilon") {
            Some(Value::Array(_)) => list("epsilon")?,
            Some(e) => vec![q_from_json(e)?; n],
            None => vec![qr(1, 100); n],
        };
        let default = v.get("default").map(Distribution::from_json).transpose()?;
        let edges = v.get("edges").and_then(Value::as_object);
        let mut laws: Vec<Vec<Option<Distribution>>> = vec![vec![None; n]; n];
        for u in 0..n {
            for w in 0..n {
                if u == w {
                    continue;
                }
                let law = match edges.and_then(|e| e.get(&format!("{u},{w}"))) {
                    Some(spec) => Distribution::from_json(spec)?,
                    None => default.clone().ok_or_else(|| {
                        format!("pair {u},{w} has no distribution and no default")
                    })?,
                };
                laws[u][w] = Some(law);
            }
        }
        let mut m = Model::new(v["name"].as_str().unwrap_or("model"), lam, mu, d, eps, laws)?;
        m.service = v.get("service").and_then(Value::as_str).map(String::from);
        m.batch_mean = v
            .get("batch_mean")
            .filter(|b| !b.is_null())
            .map(q_from_json)
            .transpose()?;
        Ok(m)
    }

    /// The same model with its finite-support distributions as
    /// [`Distribution::Discrete`] (to cross-check the grid construction
    /// against the lattice construction).
    pub fn from_discrete(m: &DiscreteModel) -> Model {
        let laws = (0..m.n)
            .map(|u| {
                (0..m.n)
                    .map(|v| (u != v).then(|| Distribution::Discrete(m.law(u, v).clone())))
                    .collect()
            })
            .collect();
        Model::new(
            &m.name,
            m.lam.clone(),
            m.mu.clone(),
            m.d.clone(),
            m.eps.clone(),
            laws,
        )
        .unwrap()
    }

    /// Every reconfiguration-time distribution scaled by `k` (mean times `k`).
    pub fn with_reconfiguration_scaled(&self, k: &Q) -> Model {
        let mut m = self.clone();
        m.laws = self
            .laws
            .iter()
            .map(|row| {
                row.iter()
                    .map(|l| l.as_ref().map(|l| l.scaled(k)))
                    .collect()
            })
            .collect();
        m.name = format!("{}|reconfiguration x {}", self.name, k);
        m
    }

    /// Every arrival rate multiplied by `k`. The phase coefficients do not depend on the arrival
    /// rates, so they are those of the unscaled model.
    pub fn with_arrivals_scaled(&self, k: &Q) -> Model {
        let mut m = self.clone();
        m.lam = self.lam.iter().map(|l| l * k).collect();
        m.name = format!("{}|arrivals x {}", self.name, k);
        m
    }

    /// Every deadline multiplied by `k > 0`, targets unchanged. What depends on the deadlines
    /// follows them: the grid step is refined so that it divides every new deadline
    /// ([`super::grid::grid_step`]), the coefficients are computed and cached under the new
    /// deadlines, and the declared window lengths ([`super::windows::default_windows`]) are
    /// offsets from the new deadlines.
    pub fn with_deadlines_scaled(&self, k: &Q) -> Model {
        let mut m = self.clone();
        m.d = self.d.iter().map(|x| x * k).collect();
        m.name = format!("{}|deadlines x {}", self.name, k);
        m
    }

    /// Descriptions of the distinct reconfiguration-time distributions, in
    /// pair order.
    pub fn describe_laws(&self, bits: u32) -> Value {
        let mut seen: Vec<String> = Vec::new();
        let mut out = Vec::new();
        for (u, v) in self.edges() {
            let id = self.law(u, v).id();
            if !seen.contains(&id) {
                seen.push(id);
                out.push(self.law(u, v).describe(bits));
            }
        }
        Value::Array(out)
    }
}

#[cfg(test)]
mod tests {
    use super::super::rational::parse_q;
    use super::*;

    fn f(e: &(Q, Q)) -> (f64, f64) {
        (to_f64(&e.0), to_f64(&e.1))
    }

    #[test]
    fn profiles_have_the_prescribed_moments() {
        let mean = qr(1, 2);
        let cv2 = qr(1, 3);
        for name in profile_names() {
            let law = profile(name, &mean, &cv2).unwrap();
            let m = f(&law.mean(128));
            assert!(m.0 <= 0.5 + 1e-9 && 0.5 - 1e-9 <= m.1, "{name} mean {m:?}");
            let expected_cv2 = match name {
                "det" => 0.0,
                "exp" => 1.0,
                _ => 1.0 / 3.0,
            };
            let c = f(&law.cv2(128).unwrap());
            assert!(
                c.0 <= expected_cv2 + 1e-8 && expected_cv2 - 1e-8 <= c.1,
                "{name} cv2 {c:?}"
            );
        }
        let heavy = pareto_with_alpha(&mean, &qr(3, 2));
        assert_eq!(heavy.mean(64).0, mean);
        assert!(heavy.variance(64).is_none());
    }

    #[test]
    fn distribution_functions_are_consistent_and_tight() {
        let mean = qr(1, 2);
        let cv2 = qr(1, 3);
        for name in profile_names() {
            let law = profile(name, &mean, &cv2).unwrap();
            let mut last = 0.0;
            for x in ["0", "1/10", "1/3", "1/2", "1", "2", "3", "6", "12"] {
                let xq = parse_q(x).unwrap();
                let b = law.below(&xq, 128);
                let s = law.survival(&xq, 128);
                let (blo, bhi) = f(&b);
                let (slo, shi) = f(&s);
                assert!(
                    blo <= bhi + 1e-30 && 0.0 <= blo && bhi <= 1.0,
                    "{name} below {x}"
                );
                assert!(
                    slo <= shi + 1e-30 && 0.0 <= slo && shi <= 1.0,
                    "{name} survival {x}"
                );
                assert!(blo + 1e-12 >= last, "{name} monotone at {x}");
                last = blo;
                if !law.is_finite_support() {
                    assert!(bhi - blo < 1e-12, "{name} width at {x}: {}", bhi - blo);
                    assert!(
                        (blo + shi - 1.0).abs() < 1e-12,
                        "{name} below + survival at {x}"
                    );
                }
            }
        }
    }

    #[test]
    fn gamma_shape_one_agrees_with_the_exponential_profile() {
        let e = profile("exp", &qr(1, 2), &qi(1)).unwrap();
        let g = profile("gamma", &qr(1, 2), &qi(1)).unwrap();
        // shape one is an integer, so the Erlang closed form is used; the
        // series for non-integer shapes is tested in `special`.
        for x in ["1/10", "1/2", "3", "12"] {
            let xq = parse_q(x).unwrap();
            let a = f(&e.below(&xq, 128));
            let b = f(&g.below(&xq, 128));
            assert!(
                a.0 <= b.1 + 1e-30 && b.0 <= a.1 + 1e-30,
                "{x}: {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn json_round_trip_and_scaling() {
        let m = Model::balanced(
            "t",
            &profile("gamma", &qr(1, 2), &qr(1, 3)).unwrap(),
            3,
            &qr(1, 2),
            &qi(4),
            &qr(1, 100),
        );
        let back = Model::from_json(&m.to_json()).unwrap();
        assert_eq!(back.law(0, 1), m.law(0, 1));
        let s = m.with_reconfiguration_scaled(&qr(3, 4));
        assert_eq!(s.law(1, 2).mean(64).0, qr(3, 8));
        assert_eq!(m.with_arrivals_scaled(&qr(1, 2)).lam, vec![qr(1, 3); 3]);
        let longer = m.with_deadlines_scaled(&qr(7, 4));
        assert_eq!(longer.d, vec![qi(7); 3]);
        assert_eq!(longer.eps, m.eps);
        let shell = m.lp_shell(64);
        assert_eq!(shell.n, 3);
    }
}
