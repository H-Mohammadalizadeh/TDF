//! The scheduling policies of the simulator, the two policies of the paper: exhaustive cyclic
//! polling (EC, `ec`) and oldest job first (OJF, `ojf`); each is described at its type. Neither
//! reads the deadlines or the targets, and neither draws random numbers.

use super::engine::{Policy, View};
use super::laws::Law;
use serde_json::{Map, Value};

fn no_parameters(name: &str, params: &Map<String, Value>) -> Result<(), String> {
    match params.keys().next() {
        Some(k) => Err(format!("unknown parameters for {name}: {k}")),
        None => Ok(()),
    }
}

/// Exhaustive cyclic polling (EC), the classical polling discipline and the certified policy of
/// the paper at high load: visit the configurations in the fixed cyclic order
/// `0, 1, ..., N - 1, 0, ...`, serve each class until its queue is empty, then reconfigure to the
/// next configuration even if its class is empty. The network never idles: with every queue empty
/// it keeps cycling (with one class it idles, since there is nothing to cycle through).
pub struct CyclicPolling;

impl Policy for CyclicPolling {
    fn decide(&mut self, view: &View) -> i64 {
        if view.arrivals[view.at].is_empty() {
            ((view.at + 1) % view.n()) as i64
        } else {
            view.at as i64
        }
    }

    fn visits_empty(&self) -> bool {
        true
    }
}

/// Oldest job first (OJF), global FCFS across classes: serve the oldest waiting job and
/// reconfigure directly to its class when needed. After a reconfiguration the engine serves the
/// head of the destination queue without consulting the policy; since arrivals only append, that
/// head is the selected job, so the selection holds through the reconfiguration. OJF is the
/// certified policy of the paper at light load, analyzed in `feasibility::reference`.
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

/// The policy named `name` (`ec` or `ojf`); neither takes parameters. `laws` are the
/// reconfiguration-time distributions of the ordered pairs and `deadlines` the deadlines of the
/// classes. EC needs a positive mean reconfiguration time around its cycle, since with every queue
/// empty it would otherwise cycle without advancing the clock.
pub fn make_policy(
    name: &str,
    params: &Map<String, Value>,
    laws: &[((usize, usize), Law)],
    deadlines: &[f64],
) -> Result<Box<dyn Policy>, String> {
    match name {
        "ec" => {
            no_parameters(name, params)?;
            let n = deadlines.len();
            let cycle: f64 = laws
                .iter()
                .filter(|((u, v), _)| *v == (*u + 1) % n)
                .map(|(_, law)| law.mean())
                .sum();
            if n > 1 && (cycle <= 0.0 || cycle.is_nan()) {
                return Err(
                    "ec needs a positive mean reconfiguration time around the cycle".into(),
                );
            }
            Ok(Box::new(CyclicPolling))
        }
        "ojf" => {
            no_parameters(name, params)?;
            Ok(Box::new(Ojf))
        }
        other => Err(format!("unknown policy {other:?}")),
    }
}
