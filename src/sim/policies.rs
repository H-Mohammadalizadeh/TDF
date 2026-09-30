//! The scheduling policies of the simulator: exhaustive cyclic service (EC, `ec`), head-of-line
//! biased max-weight (HOL-BMW, `hol_bmw`) and oldest job first (OJF, `ojf`); each is described at
//! its type. A `queue` is the queue of waiting jobs of a class, and `switch_reference` the
//! reference reconfiguration time of HOL-BMW. None of the policies reads the deadlines or
//! targets, and none draws random numbers.

use super::engine::{Policy, View};
use super::laws::Law;
use super::rng::compensated_sum;
use serde_json::{Map, Value};

fn num(params: &Map<String, Value>, key: &str) -> Result<Option<f64>, String> {
    match params.get(key) {
        None => Ok(None),
        Some(v) => v
            .as_f64()
            .map(Some)
            .ok_or_else(|| format!("{key} must be numeric")),
    }
}

fn positive(v: f64, label: &str, zero: bool) -> Result<f64, String> {
    if !v.is_finite() || (if zero { v < 0.0 } else { v <= 0.0 }) {
        return Err(format!(
            "{label} must be finite and {}",
            if zero { "nonnegative" } else { "positive" }
        ));
    }
    Ok(v)
}

fn no_parameters(name: &str, params: &Map<String, Value>) -> Result<(), String> {
    match params.keys().next() {
        Some(k) => Err(format!("unknown parameters for {name}: {k}")),
        None => Ok(()),
    }
}

/// Exhaustive cyclic service (EC): serve the current class until its queue is empty, then
/// reconfigure to the next class with waiting jobs in cyclic order.
pub struct Exhaustive;

impl Policy for Exhaustive {
    fn decide(&mut self, view: &View) -> i64 {
        let lengths: Vec<usize> = view.arrivals.iter().map(|q| q.len()).collect();
        let at = view.at;
        if lengths.iter().all(|l| *l == 0) {
            return -1;
        }
        if lengths[at] > 0 {
            return at as i64;
        }
        (1..lengths.len())
            .map(|offset| (at + offset) % lengths.len())
            .find(|j| lengths[*j] > 0)
            .unwrap() as i64
    }
}

/// Head-of-line biased max-weight (HOL-BMW). The weight `w_i` of class `i` is the age of its
/// oldest waiting job in units of `time_unit` (zero for an empty queue). In configuration `j`, the
/// network stays if only class `j` has waiting jobs or `w_j` is maximal. Otherwise, with `k` the
/// class of largest weight (lowest index on ties), it reconfigures to `k` if queue `j` is empty
/// or `w_k >= (1 + T / G) w_j`, and stays if not. `T` is a reference reconfiguration time in units
/// of `time_unit` (by default the mean over all ordered pairs), and `G = max(1, W^alpha)` is set
/// from the total weight `W` at the first decision and whenever a reconfiguration starts.
pub struct HolBmw {
    alpha: f64,
    time_unit: f64,
    /// The reference reconfiguration time `T`.
    switch_reference: f64,
    /// The denominator `G`, unset before the first decision.
    interval_denominator: Option<f64>,
}

impl HolBmw {
    /// Parameters `alpha` in `(0, 1)` (default 0.1), `time_unit > 0`
    /// (default 1) and `switch_reference >= 0` (default: the average of the
    /// mean reconfiguration times of `laws`).
    pub fn new(
        params: &Map<String, Value>,
        laws: &[((usize, usize), Law)],
    ) -> Result<HolBmw, String> {
        for k in params.keys() {
            if !["alpha", "switch_reference", "time_unit"].contains(&k.as_str()) {
                return Err(format!("unknown parameters for hol_bmw: {k}"));
            }
        }
        let default_reference = if laws.is_empty() {
            0.0
        } else {
            compensated_sum(&laws.iter().map(|(_, l)| l.mean()).collect::<Vec<_>>())
                / laws.len() as f64
        };
        let mut policy = HolBmw {
            alpha: 0.1,
            time_unit: 1.0,
            switch_reference: default_reference,
            interval_denominator: None,
        };
        if let Some(v) = num(params, "alpha")? {
            let v = positive(v, "alpha", false)?;
            if v >= 1.0 {
                return Err("alpha must lie strictly between 0 and 1".into());
            }
            policy.alpha = v;
        }
        if let Some(v) = num(params, "time_unit")? {
            policy.time_unit = positive(v, "time_unit", false)?;
        }
        if let Some(v) = num(params, "switch_reference")? {
            policy.switch_reference = positive(v, "switch_reference", true)?;
        }
        Ok(policy)
    }

    /// Set `G = max(1, W^alpha)` from the total `W` of the current weights.
    fn freeze_interval(&mut self, weights: &[f64]) {
        self.interval_denominator = Some(compensated_sum(weights).powf(self.alpha).max(1.0));
    }

    /// The class with waiting jobs and the largest weight, the lowest index on ties.
    fn best(indices: &[usize], weights: &[f64]) -> usize {
        let mut best = indices[0];
        for &i in &indices[1..] {
            if weights[i] > weights[best] || (weights[i] == weights[best] && i < best) {
                best = i;
            }
        }
        best
    }
}

impl Policy for HolBmw {
    fn decide(&mut self, view: &View) -> i64 {
        let at = view.at;
        let nonempty: Vec<usize> = (0..view.n())
            .filter(|&i| !view.arrivals[i].is_empty())
            .collect();
        let weights: Vec<f64> = view
            .arrivals
            .iter()
            .map(|row| {
                if row.is_empty() {
                    0.0
                } else {
                    (view.now - row[0]) / self.time_unit
                }
            })
            .collect();
        if self.interval_denominator.is_none() {
            self.freeze_interval(&weights);
        }
        if nonempty.is_empty() {
            return -1;
        }
        if nonempty.iter().all(|&i| i == at) {
            return at as i64;
        }
        let j = Self::best(&nonempty, &weights);
        let factor =
            1.0 + self.switch_reference / self.time_unit / self.interval_denominator.unwrap();
        let target = if view.arrivals[at].is_empty() {
            j
        } else {
            let max_w = nonempty
                .iter()
                .map(|&i| weights[i])
                .fold(f64::NEG_INFINITY, f64::max);
            if weights[at] == max_w {
                at
            } else if factor * weights[at] <= weights[j] {
                j
            } else {
                at
            }
        };
        if target != at {
            self.freeze_interval(&weights);
        }
        target as i64
    }
}

/// Oldest job first (OJF), global FCFS across classes: serve the oldest waiting job and
/// reconfigure directly to its class when needed. After a reconfiguration the engine serves the
/// head of the destination queue without consulting the policy; since arrivals only append, that
/// head is the selected job, so the selection holds through the reconfiguration. OJF is the
/// reference policy analyzed in `feasibility::reference`.
pub struct Ojf;

impl Policy for Ojf {
    fn decide(&mut self, view: &View) -> i64 {
        let mut oldest: Option<(f64, usize)> = None;
        for (q, row) in view.arrivals.iter().enumerate() {
            if let Some(&a) = row.first()
                && oldest.is_none_or(|(b, _)| a < b)
            {
                oldest = Some((a, q));
            }
        }
        match oldest {
            None => -1,
            Some((_, q)) => q as i64,
        }
    }
}

/// The policy named `name` (`ec`, `hol_bmw` or `ojf`) with its parameters;
/// `laws` are the reconfiguration-time distributions of the ordered pairs.
pub fn make_policy(
    name: &str,
    params: &Map<String, Value>,
    laws: &[((usize, usize), Law)],
) -> Result<Box<dyn Policy>, String> {
    match name {
        "ec" => {
            no_parameters(name, params)?;
            Ok(Box::new(Exhaustive))
        }
        "hol_bmw" => Ok(Box::new(HolBmw::new(params, laws)?)),
        "ojf" => {
            no_parameters(name, params)?;
            Ok(Box::new(Ojf))
        }
        other => Err(format!("unknown policy {other:?}")),
    }
}
