//! Service-time and reconfiguration-time distributions of the simulator (a [`Law`] is a
//! probability distribution) in double precision: samplers, survival functions, moments and
//! integrated tails. The families are the point mass, the exponential, the lognormal, finite
//! discrete distributions, the gamma (Marsaglia–Tsang sampler), the normal conditioned on a
//! positive outcome (rejection from the parent normal) and the Pareto type I (inverse transform).
//! The truncated normal is calibrated here to a prescribed mean and coefficient of variation; the
//! exact enclosures of the feasibility test take the same parent parameters, so the certificates
//! and the simulator use the same distribution.

use super::rng::{Rng, fsum};
use serde_json::{Value, json};
use statrs::function::gamma::gamma_ur;

/// A probability distribution on the nonnegative reals.
#[derive(Debug, Clone)]
pub enum Law {
    Deterministic(f64),
    /// Given by its mean.
    Exponential(f64),
    /// Given by its mean and coefficient of variation; `loc` and `sigma2` are the mean and the
    /// variance of the logarithm.
    Lognormal {
        mean: f64,
        cv: f64,
        sigma2: f64,
        loc: f64,
    },
    Discrete {
        values: Vec<f64>,
        probs: Vec<f64>,
    },
    /// Shape `k`, scale `theta`: mean `k theta`, variance `k theta^2`.
    Gamma {
        shape: f64,
        scale: f64,
    },
    /// The normal distribution `N(mu0, sigma0^2)` conditioned on a positive outcome.
    TruncNormal {
        mu0: f64,
        sigma0: f64,
    },
    /// Type I: `P(T > x) = (xm / x)^alpha` for `x >= xm`.
    Pareto {
        alpha: f64,
        xm: f64,
    },
}

fn finite(v: f64, name: &str, positive: bool) -> Result<f64, String> {
    if !v.is_finite() || (if positive { v <= 0.0 } else { v < 0.0 }) {
        return Err(format!(
            "{name} must be finite and {}",
            if positive { "positive" } else { "nonnegative" }
        ));
    }
    Ok(v)
}

unsafe extern "C" {
    #[link_name = "erfc"]
    fn libm_erfc(x: f64) -> f64;
}

/// The complementary error function of the platform C library.
fn erfc(x: f64) -> f64 {
    unsafe { libm_erfc(x) }
}

/// Standard normal distribution function.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Standard normal density.
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Inverse Mills ratio `phi(a) / (1 - Phi(a))`.
fn mills(a: f64) -> f64 {
    norm_pdf(a) / (1.0 - norm_cdf(a))
}

/// Squared coefficient of variation of the positive-truncated normal as a
/// function of `a = -mu0 / sigma0` (increasing in `a`).
fn truncated_cv2(a: f64) -> f64 {
    let l = mills(a);
    (1.0 + a * l - l * l) / ((l - a) * (l - a))
}

/// Parent parameters `(mu0, sigma0)` of the normal distribution whose
/// positive truncation has the given mean and coefficient of variation.
pub fn truncnormal_parent(mean: f64, cv: f64) -> Result<(f64, f64), String> {
    let mean = finite(mean, "mean", true)?;
    let cv = finite(cv, "cv", true)?;
    let target = cv * cv;
    // The coefficient of variation of the truncated normal is below 1 (the
    // distribution tends to the exponential as `a` grows); beyond `a = 5` the
    // closed form cancels catastrophically in doubles, so the calibration
    // stops there.
    let (mut lo, mut hi) = (-30.0f64, 5.0f64);
    if cv >= 1.0 || !(truncated_cv2(lo) < target && target < truncated_cv2(hi)) {
        return Err(format!(
            "truncated normal: coefficient of variation {cv} outside the supported range ({:.4}, {:.4})",
            truncated_cv2(lo).sqrt(),
            truncated_cv2(hi).sqrt()
        ));
    }
    for _ in 0..300 {
        let mid = 0.5 * (lo + hi);
        if truncated_cv2(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let a = 0.5 * (lo + hi);
    let l = mills(a);
    let sigma0 = mean / (l - a);
    let mu0 = -a * sigma0;
    Ok((mu0, sigma0))
}

/// `Gamma(shape, 1)` by Marsaglia and Tsang, boosted below shape one.
fn gamma_variate(rng: &mut Rng, shape: f64) -> f64 {
    if shape < 1.0 {
        let boost = gamma_variate(rng, shape + 1.0);
        let u = 1.0 - rng.random();
        return boost * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = rng.normalvariate(0.0, 1.0);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = rng.random();
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v;
        }
        if u > 0.0 && u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

impl Law {
    pub fn deterministic(v: f64) -> Result<Law, String> {
        Ok(Law::Deterministic(finite(v, "value", false)?))
    }

    pub fn exponential(mean: f64) -> Result<Law, String> {
        Ok(Law::Exponential(finite(mean, "mean", true)?))
    }

    /// Lognormal with the given mean and coefficient of variation.
    pub fn lognormal(mean: f64, cv: f64) -> Result<Law, String> {
        let mean = finite(mean, "mean", true)?;
        let cv = finite(cv, "cv", false)?;
        let sigma2 = (cv * cv).ln_1p();
        let loc = mean.ln() - sigma2 / 2.0;
        Ok(Law::Lognormal {
            mean,
            cv,
            sigma2,
            loc,
        })
    }

    pub fn discrete(values: Vec<f64>, probabilities: Vec<f64>) -> Result<Law, String> {
        if values.is_empty() || values.len() != probabilities.len() {
            return Err("values and probabilities need equal positive length".into());
        }
        for v in &values {
            finite(*v, "discrete value", true)?;
        }
        for p in &probabilities {
            finite(*p, "probability", false)?;
        }
        let total = fsum(&probabilities);
        if (total - 1.0).abs() > 1e-12 {
            return Err("probabilities must sum to one (tolerance 1e-12)".into());
        }
        Ok(Law::Discrete {
            values,
            probs: probabilities.iter().map(|p| p / total).collect(),
        })
    }

    pub fn gamma(shape: f64, scale: f64) -> Result<Law, String> {
        Ok(Law::Gamma {
            shape: finite(shape, "shape", true)?,
            scale: finite(scale, "scale", true)?,
        })
    }

    /// The positive-truncated normal; refused when less than 2 % of the
    /// parent mass is positive (rejection sampling would be too slow).
    pub fn truncnormal(mu0: f64, sigma0: f64) -> Result<Law, String> {
        if !mu0.is_finite() {
            return Err("mu0 must be finite".to_string());
        }
        let sigma0 = finite(sigma0, "sigma0", true)?;
        let acceptance = 1.0 - norm_cdf(-mu0 / sigma0);
        if acceptance < 0.02 {
            return Err(format!(
                "truncated normal: only {acceptance:.3} of the parent mass is positive; rejection sampling refused"
            ));
        }
        Ok(Law::TruncNormal { mu0, sigma0 })
    }

    pub fn pareto(alpha: f64, xm: f64) -> Result<Law, String> {
        let alpha = finite(alpha, "shape", true)?;
        if alpha <= 1.0 {
            return Err("pareto needs shape > 1 for a finite mean".into());
        }
        Ok(Law::Pareto {
            alpha,
            xm: finite(xm, "min", true)?,
        })
    }

    /// A distribution from its JSON form (see `to_json`); a point mass may
    /// also be given by its `mean`.
    pub fn from_json(spec: &Value) -> Result<Law, String> {
        let family = spec.get("family").and_then(Value::as_str).unwrap_or("");
        let num = |key: &str| {
            spec.get(key)
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("invalid {family:?} distribution: missing {key}"))
        };
        let list = |key: &str| -> Result<Vec<f64>, String> {
            spec.get(key)
                .and_then(Value::as_array)
                .ok_or_else(|| format!("invalid {family:?} distribution: missing {key}"))?
                .iter()
                .map(|x| {
                    x.as_f64().ok_or_else(|| {
                        format!("invalid {family:?} distribution: non-numeric {key}")
                    })
                })
                .collect()
        };
        match family {
            "deterministic" => {
                if spec.get("value").is_some() && spec.get("mean").is_some() {
                    return Err("deterministic distribution accepts value or mean, not both".into());
                }
                let v = if spec.get("value").is_some() {
                    num("value")?
                } else {
                    num("mean")?
                };
                Law::deterministic(v)
            }
            "exponential" => Law::exponential(num("mean")?),
            "lognormal" => Law::lognormal(num("mean")?, num("cv")?),
            "discrete" => Law::discrete(list("values")?, list("probabilities")?),
            "gamma" => Law::gamma(num("shape")?, num("scale")?),
            "truncated_normal" => Law::truncnormal(num("mu0")?, num("sigma0")?),
            "pareto" => Law::pareto(num("shape")?, num("min")?),
            other => Err(format!("unknown distribution family {other:?}")),
        }
    }

    /// The JSON form of the distribution.
    pub fn to_json(&self) -> Value {
        match self {
            Law::Deterministic(v) => json!({"family": "deterministic", "value": v}),
            Law::Exponential(m) => json!({"family": "exponential", "mean": m}),
            Law::Lognormal { mean, cv, .. } => {
                json!({"family": "lognormal", "mean": mean, "cv": cv})
            }
            Law::Discrete { values, probs } => {
                json!({"family": "discrete", "values": values, "probabilities": probs})
            }
            Law::Gamma { shape, scale } => {
                json!({"family": "gamma", "shape": shape, "scale": scale})
            }
            Law::TruncNormal { mu0, sigma0 } => {
                json!({"family": "truncated_normal", "mu0": mu0, "sigma0": sigma0})
            }
            Law::Pareto { alpha, xm } => json!({"family": "pareto", "shape": alpha, "min": xm}),
        }
    }

    /// One draw from `rng`.
    pub fn sample(&self, rng: &mut Rng) -> f64 {
        match self {
            Law::Deterministic(v) => *v,
            Law::Exponential(m) => rng.expovariate(1.0 / m),
            Law::Lognormal {
                mean,
                cv,
                sigma2,
                loc,
            } => {
                if *cv != 0.0 {
                    rng.lognormvariate(*loc, sigma2.sqrt())
                } else {
                    *mean
                }
            }
            Law::Discrete { values, probs } => {
                let u = rng.random();
                let mut cumulative = 0.0;
                for (v, p) in values.iter().zip(probs) {
                    cumulative += p;
                    if u < cumulative {
                        return *v;
                    }
                }
                let last = probs.iter().rposition(|p| *p > 0.0).unwrap_or(0);
                values[last]
            }
            Law::Gamma { shape, scale } => gamma_variate(rng, *shape) * scale,
            Law::TruncNormal { mu0, sigma0 } => loop {
                let x = rng.normalvariate(*mu0, *sigma0);
                if x > 0.0 {
                    return x;
                }
            },
            Law::Pareto { alpha, xm } => xm * (1.0 - rng.random()).powf(-1.0 / alpha),
        }
    }

    /// Strict survival `P(T > x)`.
    pub fn survival(&self, x: f64) -> f64 {
        match self {
            Law::Deterministic(v) => (*v > x) as i32 as f64,
            Law::Exponential(m) => {
                if x < 0.0 {
                    1.0
                } else {
                    (-x / m).exp()
                }
            }
            Law::Lognormal {
                mean,
                cv,
                sigma2,
                loc,
            } => {
                if *cv == 0.0 {
                    return (*mean > x) as i32 as f64;
                }
                if x <= 0.0 {
                    return 1.0;
                }
                0.5 * erfc((x.ln() - loc) / (2.0 * sigma2).sqrt())
            }
            Law::Discrete { values, probs } => {
                let terms: Vec<f64> = values
                    .iter()
                    .zip(probs)
                    .filter(|(v, _)| **v > x)
                    .map(|(_, p)| *p)
                    .collect();
                fsum(&terms)
            }
            Law::Gamma { shape, scale } => {
                if x <= 0.0 {
                    1.0
                } else {
                    gamma_ur(*shape, x / scale)
                }
            }
            Law::TruncNormal { mu0, sigma0 } => {
                if x <= 0.0 {
                    return 1.0;
                }
                let tail_x = 1.0 - norm_cdf((x - mu0) / sigma0);
                let tail_0 = 1.0 - norm_cdf(-mu0 / sigma0);
                (tail_x / tail_0).clamp(0.0, 1.0)
            }
            Law::Pareto { alpha, xm } => {
                if x < *xm {
                    1.0
                } else {
                    (xm / x).powf(*alpha)
                }
            }
        }
    }

    pub fn mean(&self) -> f64 {
        match self {
            Law::Deterministic(v) => *v,
            Law::Exponential(m) => *m,
            Law::Lognormal { mean, .. } => *mean,
            Law::Discrete { values, probs } => fsum(
                &values
                    .iter()
                    .zip(probs)
                    .map(|(v, p)| v * p)
                    .collect::<Vec<_>>(),
            ),
            Law::Gamma { shape, scale } => shape * scale,
            Law::TruncNormal { mu0, sigma0 } => {
                let a = -mu0 / sigma0;
                mu0 + sigma0 * mills(a)
            }
            Law::Pareto { alpha, xm } => alpha * xm / (alpha - 1.0),
        }
    }

    pub fn second_moment(&self) -> f64 {
        match self {
            Law::Deterministic(v) => v * v,
            Law::Exponential(m) => 2.0 * m * m,
            Law::Lognormal { mean, cv, .. } => mean * mean * (1.0 + cv * cv),
            Law::Discrete { values, probs } => fsum(
                &values
                    .iter()
                    .zip(probs)
                    .map(|(v, p)| v * v * p)
                    .collect::<Vec<_>>(),
            ),
            Law::Gamma { shape, scale } => shape * (shape + 1.0) * scale * scale,
            Law::TruncNormal { .. } => {
                let m = self.mean();
                self.variance() + m * m
            }
            Law::Pareto { alpha, xm } => {
                if *alpha <= 2.0 {
                    f64::INFINITY
                } else {
                    alpha * xm * xm / (alpha - 2.0)
                }
            }
        }
    }

    pub fn variance(&self) -> f64 {
        match self {
            Law::Deterministic(_) => 0.0,
            Law::Exponential(m) => m * m,
            Law::Lognormal { mean, cv, .. } => (mean * cv) * (mean * cv),
            Law::Gamma { shape, scale } => shape * scale * scale,
            Law::TruncNormal { mu0, sigma0 } => {
                let a = -mu0 / sigma0;
                let l = mills(a);
                sigma0 * sigma0 * (1.0 + a * l - l * l)
            }
            Law::Pareto { alpha, xm } => {
                if *alpha <= 2.0 {
                    f64::INFINITY
                } else {
                    alpha * xm * xm / ((alpha - 1.0) * (alpha - 1.0) * (alpha - 2.0))
                }
            }
            _ => {
                let second = self.second_moment();
                let m = self.mean();
                (second - m * m).max(0.0)
            }
        }
    }

    /// The integrated tail `E[(T - x)^+]`.
    pub fn stop_loss(&self, x: f64) -> f64 {
        match self {
            Law::Deterministic(v) => (v - x).max(0.0),
            Law::Exponential(m) => {
                if x < 0.0 {
                    m - x
                } else {
                    m * self.survival(x)
                }
            }
            Law::Lognormal {
                mean,
                cv,
                sigma2,
                loc,
            } => {
                if *cv == 0.0 {
                    return (mean - x).max(0.0);
                }
                if x <= 0.0 {
                    return mean - x;
                }
                if x.is_infinite() {
                    return 0.0;
                }
                let partial = mean * 0.5 * erfc((x.ln() - loc - sigma2) / (2.0 * sigma2).sqrt());
                (partial - x * self.survival(x)).max(0.0)
            }
            Law::Discrete { values, probs } => fsum(
                &values
                    .iter()
                    .zip(probs)
                    .map(|(v, p)| p * (v - x).max(0.0))
                    .collect::<Vec<_>>(),
            ),
            Law::Gamma { shape, scale } => {
                if x <= 0.0 {
                    return self.mean() - x;
                }
                if x.is_infinite() {
                    return 0.0;
                }
                let y = x / scale;
                (shape * scale * gamma_ur(shape + 1.0, y) - x * gamma_ur(*shape, y)).max(0.0)
            }
            Law::TruncNormal { mu0, sigma0 } => {
                if x <= 0.0 {
                    return self.mean() - x;
                }
                if x.is_infinite() {
                    return 0.0;
                }
                let a = -mu0 / sigma0;
                let b = (x - mu0) / sigma0;
                let tail_0 = 1.0 - norm_cdf(a);
                (sigma0 * (norm_pdf(b) - b * (1.0 - norm_cdf(b))) / tail_0).max(0.0)
            }
            Law::Pareto { alpha, xm } => {
                if x < *xm {
                    return self.mean() - x;
                }
                if x.is_infinite() {
                    return 0.0;
                }
                x * (xm / x).powf(*alpha) / (alpha - 1.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_point_laws_have_the_declared_moments_and_stop_loss() {
        let a = Law::discrete(vec![0.1, 0.9], vec![0.5, 0.5]).unwrap();
        let b = Law::discrete(vec![0.45, 3.7], vec![64.0 / 65.0, 1.0 / 65.0]).unwrap();
        assert_eq!(a.mean(), 0.5);
        assert!((b.mean() - 0.5).abs() < 1e-15);
        assert!((a.variance() - 0.16).abs() < 1e-15);
        assert!((b.variance() - 0.16).abs() < 1e-15);
        assert_eq!(a.stop_loss(3.0), 0.0);
        assert!((b.stop_loss(3.0) - 0.7 / 65.0).abs() < 1e-15);
    }

    /// Mean 1/2 and squared coefficient of variation 1/3 in every family.
    fn calibrated() -> Vec<(&'static str, Law)> {
        let cv = (1.0f64 / 3.0).sqrt();
        let (mu0, sigma0) = truncnormal_parent(0.5, cv).unwrap();
        vec![
            ("gamma", Law::gamma(3.0, 1.0 / 6.0).unwrap()),
            ("tnorm", Law::truncnormal(mu0, sigma0).unwrap()),
            ("pareto", Law::pareto(3.0, 1.0 / 3.0).unwrap()),
            ("lognormal", Law::lognormal(0.5, cv).unwrap()),
        ]
    }

    #[test]
    fn calibrated_laws_have_the_prescribed_moments_and_consistent_tails() {
        for (name, law) in calibrated() {
            assert!(
                (law.mean() - 0.5).abs() < 1e-9,
                "{name} mean {}",
                law.mean()
            );
            assert!(
                (law.variance() - 1.0 / 12.0).abs() < 1e-9,
                "{name} var {}",
                law.variance()
            );
            // The integrated tail at zero is the mean, and the survival function is nonincreasing.
            assert!((law.stop_loss(0.0) - 0.5).abs() < 1e-9);
            let mut last = 1.0;
            for x in [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0] {
                let s = law.survival(x);
                assert!(s <= last + 1e-12 && (0.0..=1.0).contains(&s));
                last = s;
            }
        }
        let heavy = Law::pareto(1.5, 0.5 / 3.0).unwrap();
        assert!((heavy.mean() - 0.5).abs() < 1e-12 && heavy.variance().is_infinite());
    }

    #[test]
    fn samples_match_the_moments() {
        let (mu0, sigma0) = truncnormal_parent(0.5, 0.25).unwrap();
        let mut cases: Vec<(&str, Law, bool)> = calibrated()
            .into_iter()
            .filter(|(name, _)| *name != "lognormal")
            .map(|(name, law)| (name, law, name != "pareto"))
            .collect();
        cases.push(("gamma_quarter", Law::gamma(0.25, 2.0).unwrap(), true));
        cases.push(("tnorm_low", Law::truncnormal(mu0, sigma0).unwrap(), true));
        for (name, law, check_variance) in cases {
            let mut rng = Rng::xoshiro(20260922);
            let n = 400_000;
            let (mut s, mut s2) = (0.0f64, 0.0f64);
            for _ in 0..n {
                let x = law.sample(&mut rng);
                assert!(x > 0.0, "{name}: nonpositive sample");
                s += x;
                s2 += x * x;
            }
            let mean = s / n as f64;
            let var = s2 / n as f64 - mean * mean;
            assert!(
                (mean - law.mean()).abs() < 0.01 * law.mean(),
                "{name}: sample mean {mean} vs {}",
                law.mean()
            );
            if check_variance {
                assert!(
                    (var - law.variance()).abs() < 0.05 * law.variance(),
                    "{name}: sample var {var} vs {}",
                    law.variance()
                );
            }
        }
    }

    #[test]
    fn truncated_normal_calibration_and_json_round_trip() {
        let (mu0, sigma0) = truncnormal_parent(0.5, 0.25).unwrap();
        let law = Law::truncnormal(mu0, sigma0).unwrap();
        assert!((law.mean() - 0.5).abs() < 1e-10 && (law.variance() - 0.015625).abs() < 1e-10);
        assert!(truncnormal_parent(0.5, 1.5).is_err());
        for spec in [
            json!({"family": "gamma", "shape": 4.0, "scale": 0.125}),
            json!({"family": "truncated_normal", "mu0": mu0, "sigma0": sigma0}),
            json!({"family": "pareto", "shape": 1.5, "min": 0.5 / 3.0}),
            json!({"family": "lognormal", "mean": 0.5, "cv": 0.3}),
            json!({"family": "deterministic", "mean": 0.25}),
        ] {
            let law = Law::from_json(&spec).unwrap();
            let back = Law::from_json(&law.to_json()).unwrap();
            assert!((law.mean() - back.mean()).abs() < 1e-12);
        }
        assert!(Law::from_json(&json!({"family": "capped"})).is_err());
    }
}
