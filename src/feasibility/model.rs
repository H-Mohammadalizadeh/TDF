//! Models with finite-support reconfiguration-time distributions (laws, in
//! identifiers), in exact rationals. [`DiscreteModel`] carries `N` classes
//! (arrival and service rates, deadlines, targets) and one distribution per
//! ordered configuration pair (an edge). Its distributions feed the lattice
//! construction of the phase coefficients (`coefficients.rs`); the relaxation
//! and the window constraints read only its rates, deadlines and targets, so
//! a model with continuous distributions enters them through `Model::lp_shell`.

use super::rational::{Q, q_from_json, qi, qr, to_f64};
use num_traits::{One, Zero};
use serde_json::{Map, Value, json};

/// A reconfiguration-time distribution with finite positive support.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscreteLaw {
    /// `(support point, probability)`, sorted by support point.
    pub atoms: Vec<(Q, Q)>,
}

impl DiscreteLaw {
    /// The distribution with the given atoms, sorted; the support points must
    /// be distinct and positive, the probabilities positive and summing to one.
    pub fn new(mut atoms: Vec<(Q, Q)>) -> Result<Self, String> {
        if atoms.is_empty() {
            return Err("a reconfiguration-time distribution needs at least one atom".into());
        }
        atoms.sort_by(|a, b| a.0.cmp(&b.0));
        for w in atoms.windows(2) {
            if w[0].0 == w[1].0 {
                return Err("duplicate support point".into());
            }
        }
        let mut total = Q::zero();
        for (s, p) in &atoms {
            if *s <= Q::zero() {
                return Err("support points must be positive".into());
            }
            if *p <= Q::zero() {
                return Err("probabilities must be positive".into());
            }
            total += p;
        }
        if total != Q::one() {
            return Err(format!("probabilities sum to {total}, not one"));
        }
        Ok(DiscreteLaw { atoms })
    }

    /// A point mass at `s`.
    pub fn point(s: Q) -> Self {
        DiscreteLaw {
            atoms: vec![(s, Q::one())],
        }
    }

    pub fn mean(&self) -> Q {
        self.atoms.iter().map(|(s, p)| s * p).sum()
    }

    pub fn variance(&self) -> Q {
        let m = self.mean();
        self.atoms
            .iter()
            .map(|(s, p)| (s - &m) * (s - &m) * p)
            .sum()
    }

    /// `Pr(tau > r)`.
    pub fn survival(&self, r: &Q) -> Q {
        self.atoms
            .iter()
            .filter(|(s, _)| s > r)
            .map(|(_, p)| p.clone())
            .sum()
    }

    /// Every support point multiplied by `k`.
    pub fn scaled(&self, k: &Q) -> Self {
        DiscreteLaw {
            atoms: self.atoms.iter().map(|(s, p)| (s * k, p.clone())).collect(),
        }
    }

    pub fn min_support(&self) -> Q {
        self.atoms[0].0.clone()
    }

    pub fn max_support(&self) -> Q {
        self.atoms.last().unwrap().0.clone()
    }

    /// The distribution in the simulator's JSON format (doubles).
    pub fn to_engine_json(&self) -> Value {
        if self.atoms.len() == 1 {
            json!({"family": "deterministic", "value": to_f64(&self.atoms[0].0)})
        } else {
            json!({
                "family": "discrete",
                "values": self.atoms.iter().map(|(s, _)| to_f64(s)).collect::<Vec<_>>(),
                "probabilities": self.atoms.iter().map(|(_, p)| to_f64(p)).collect::<Vec<_>>()
            })
        }
    }

    /// Exact JSON (rationals as strings).
    pub fn to_json(&self) -> Value {
        json!({
            "values": self.atoms.iter().map(|(s, _)| s.to_string()).collect::<Vec<_>>(),
            "probabilities": self.atoms.iter().map(|(_, p)| p.to_string()).collect::<Vec<_>>()
        })
    }

    /// A distribution from `{"value": v}` (a point mass) or
    /// `{"values": [...], "probabilities": [...]}`.
    pub fn from_json(v: &Value) -> Result<Self, String> {
        if let Some(val) = v.get("value") {
            return Ok(Self::point(q_from_json(val)?));
        }
        let values = v["values"].as_array().ok_or("distribution needs values")?;
        let probs = v["probabilities"]
            .as_array()
            .ok_or("distribution needs probabilities")?;
        if values.len() != probs.len() {
            return Err("values and probabilities differ in length".into());
        }
        Self::new(
            values
                .iter()
                .zip(probs)
                .map(|(s, p)| Ok((q_from_json(s)?, q_from_json(p)?)))
                .collect::<Result<Vec<_>, String>>()?,
        )
    }
}

/// A model with finite-support reconfiguration-time distributions.
#[derive(Clone, Debug)]
pub struct DiscreteModel {
    pub name: String,
    pub n: usize,
    pub lam: Vec<Q>,
    pub mu: Vec<Q>,
    pub d: Vec<Q>,
    pub eps: Vec<Q>,
    /// `laws[u][v]` for `u != v`; `None` on the diagonal.
    pub laws: Vec<Vec<Option<DiscreteLaw>>>,
}

impl DiscreteModel {
    /// A model with validated dimensions and parameters; `laws` must be
    /// `None` exactly on the diagonal.
    pub fn new(
        name: &str,
        lam: Vec<Q>,
        mu: Vec<Q>,
        d: Vec<Q>,
        eps: Vec<Q>,
        laws: Vec<Vec<Option<DiscreteLaw>>>,
    ) -> Result<Self, String> {
        let n = mu.len();
        if n == 0 {
            return Err("at least one class".into());
        }
        if lam.len() != n || d.len() != n || eps.len() != n || laws.len() != n {
            return Err("lam, mu, d, eps and laws must all have N entries".into());
        }
        for i in 0..n {
            if mu[i] <= Q::zero() {
                return Err("service rates must be positive".into());
            }
            if lam[i] < Q::zero() {
                return Err("arrival rates must be nonnegative".into());
            }
            if d[i] <= Q::zero() {
                return Err("deadlines must be positive".into());
            }
            if eps[i] <= Q::zero() || eps[i] >= Q::one() {
                return Err("targets must lie strictly between zero and one".into());
            }
            if laws[i].len() != n {
                return Err("laws must be an N x N table".into());
            }
            for j in 0..n {
                if (i == j) != laws[i][j].is_none() {
                    return Err("laws: None exactly on the diagonal".into());
                }
            }
        }
        Ok(DiscreteModel {
            name: name.into(),
            n,
            lam,
            mu,
            d,
            eps,
            laws,
        })
    }

    /// The load `lambda_i / mu_i` of class `i`.
    pub fn rho(&self, i: usize) -> Q {
        &self.lam[i] / &self.mu[i]
    }

    /// The total offered load.
    pub fn total_rho(&self) -> Q {
        (0..self.n).map(|i| self.rho(i)).sum()
    }

    /// Whether class `i` has a positive arrival rate.
    pub fn active(&self, i: usize) -> bool {
        self.lam[i] > Q::zero()
    }

    /// The ordered configuration pairs `(u, v)`, `u != v`, row by row.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        (0..self.n)
            .flat_map(|u| (0..self.n).filter(move |v| *v != u).map(move |v| (u, v)))
            .collect()
    }

    /// The reconfiguration-time distribution of the pair `u -> v`.
    pub fn law(&self, u: usize, v: usize) -> &DiscreteLaw {
        self.laws[u][v].as_ref().expect("off-diagonal edge")
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
            "edges": edges
        })
    }
}

/// The two-class example: `lambda_i = 1`, `mu_i = 4`, `d_i = 3`, targets
/// 1/100, and on both pairs the reconfiguration time 0.45 with probability
/// 64/65 and 3.7 with probability 1/65 (mean 0.5, variance 0.16). Neither
/// the window constraints nor the phase constraints exclude its targets
/// alone; together they do.
pub fn two_class_example() -> DiscreteModel {
    let law = DiscreteLaw::new(vec![(qr(9, 20), qr(64, 65)), (qr(37, 10), qr(1, 65))]).unwrap();
    let laws = (0..2)
        .map(|u| (0..2).map(|v| (u != v).then(|| law.clone())).collect())
        .collect();
    DiscreteModel::new(
        "two_class_example",
        vec![qi(1); 2],
        vec![qi(4); 2],
        vec![qi(3); 2],
        vec![qr(1, 100); 2],
        laws,
    )
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_class_example_has_the_declared_parameters() {
        let m = two_class_example();
        assert_eq!(m.lam, vec![Q::one(), Q::one()]);
        assert_eq!(m.total_rho(), qr(1, 2));
        let law = m.law(0, 1);
        assert_eq!(law, m.law(1, 0));
        assert_eq!(law.mean(), qr(1, 2));
        assert_eq!(law.variance(), qr(4, 25));
        assert_eq!(law.survival(&qi(3)), qr(1, 65));
        assert_eq!(law.survival(&qr(9, 20)), qr(1, 65));
        assert_eq!(law.survival(&qr(2, 5)), Q::one());
    }

    #[test]
    fn laws_round_trip_through_json_and_scale() {
        let law = two_class_example().law(0, 1).clone();
        assert_eq!(DiscreteLaw::from_json(&law.to_json()).unwrap(), law);
        assert_eq!(law.scaled(&qi(2)).mean(), qi(1));
        assert_eq!(law.max_support(), qr(37, 10));
        assert!(DiscreteLaw::new(vec![(qi(1), qr(1, 2))]).is_err());
    }
}
