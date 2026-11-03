//! Running the simulator on a model: run specifications, observation windows and the per-run
//! quantities the experiments use. A run specification (`job` in identifiers, not a job of the
//! network) is a JSON object `{"system", "policy": {"name", "params"}, "seed", "compact"}`;
//! [`run_job`] simulates it and returns the engine's record, and [`run`] condenses the record of
//! one policy on one [`Model`] into a [`Run`]. As in the engine, `measure` is the length of the
//! observation window and `switch` refers to reconfigurations.

use super::engine::{System, simulate};
use super::laws::Law;
use super::policies::make_policy;
use crate::feasibility::distributions::Model;
use crate::feasibility::rational::to_f64;
use num_traits::Signed;
use serde_json::{Map, Value, json};
use std::time::Instant;

/// Simulate one run specification; its `seed`, when present, replaces the seed of the system. An
/// invalid specification yields a result with `status = "aborted"` and the reason in
/// `abort_reason`. With `"compact": true` the reconfiguration and visit records are left out.
/// Every result gets the elapsed `wall_seconds`.
pub fn run_job(job: &Value) -> Value {
    let started = Instant::now();
    let outcome = (|| -> Result<Value, String> {
        let mut system = System::from_json(job.get("system").ok_or("job needs a system")?)?;
        if let Some(seed) = job.get("seed") {
            system.seed = seed.as_u64().ok_or("seed must be a nonnegative integer")?;
        }
        let policy_json = job.get("policy").ok_or("job needs a policy")?;
        let name = policy_json
            .get("name")
            .and_then(Value::as_str)
            .ok_or("policy needs a name")?;
        let empty = Map::new();
        let params = policy_json
            .get("params")
            .and_then(Value::as_object)
            .unwrap_or(&empty);
        let n = system.n();
        let laws: Vec<((usize, usize), Law)> = (0..n)
            .flat_map(|i| (0..n).filter(move |j| *j != i).map(move |j| (i, j)))
            .map(|e| (e, system.switch[&e].clone()))
            .collect();
        let mut policy = make_policy(name, params, &laws, &system.deadlines)?;
        Ok(simulate(&system, policy.as_mut()))
    })();
    let mut result = match outcome {
        Ok(r) => r,
        Err(e) => json!({"status": "aborted", "abort_reason": e}),
    };
    if job.get("compact").and_then(Value::as_bool).unwrap_or(false)
        && let Some(obj) = result.as_object_mut()
    {
        obj.remove("switches");
        obj.remove("visits");
    }
    result["wall_seconds"] = json!(started.elapsed().as_secs_f64());
    result
}

/// The grid of sojourn-time thresholds `x d` in multiples `x` of the deadline `d`, for the
/// exceedance events `D > x d` of the sojourn time `D`: sixty-one log-spaced points from `1/10`
/// to `8` and the deadline itself (`x = 1`), in increasing order.
pub fn x_grid() -> Vec<f64> {
    let mut xs: Vec<f64> = (0..=60)
        .map(|k| 0.1 * 80f64.powf(k as f64 / 60.0))
        .collect();
    xs.push(1.0);
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    xs
}

/// Warm-up, observation-window (`measure`) and follow-up durations of a run, and an optional
/// wall-clock limit in seconds.
#[derive(Clone, Debug)]
pub struct Window {
    pub warmup: f64,
    pub measure: f64,
    pub followup: f64,
    pub max_wall_seconds: Option<f64>,
}

impl Window {
    /// The windows of the experiments: the given warm-up, an observation window of length
    /// `max(20000, ceil(5000 / lambda_min)) * factor` for the smallest positive arrival rate
    /// `lambda_min` (at least 5000 expected arrivals of every active class for `factor = 1`),
    /// and a follow-up `max(100, 10 d_max)`, which decides every exceedance of `x d_i`,
    /// `x <= 8`, for every job of the observation window.
    pub fn for_model(
        model: &Model,
        warmup: f64,
        measure_factor: f64,
        max_wall_seconds: Option<f64>,
    ) -> Window {
        let lam_min = model
            .lam
            .iter()
            .filter(|l| l.is_positive())
            .map(to_f64)
            .fold(f64::INFINITY, f64::min);
        let measure = (20_000f64).max((5_000.0 / lam_min).ceil()) * measure_factor;
        let dmax = model.d.iter().map(to_f64).fold(0.0, f64::max);
        Window {
            warmup,
            measure,
            followup: (100f64).max(10.0 * dmax),
            max_wall_seconds,
        }
    }

    pub fn to_json(&self) -> Value {
        json!({"warmup": self.warmup, "measure": self.measure, "followup": self.followup, "max_wall_seconds": self.max_wall_seconds})
    }
}

/// Cohort counts of one class in one run.
#[derive(Clone, Debug)]
pub struct ClassCounts {
    pub arrivals: u64,
    /// Jobs whose deadline passed before completion.
    pub misses: u64,
    /// Censored jobs whose deadline had not passed at the end of the run.
    pub unresolved: u64,
    /// Jobs not completed at the end of the run.
    pub censored: u64,
    /// Number of jobs of the observation window with `D > x d`, per grid point `x`; a censored
    /// job counts where its observed lower bound on `D` exceeds `x d`.
    pub exceed: Vec<u64>,
    /// Sojourn-time quantiles at the rounded rank `(n - 1) p` of the `n` completed jobs; `None`
    /// without completed jobs or with a censored job.
    pub p50: Option<f64>,
    pub p99: Option<f64>,
    /// Time average of the age of the class's oldest job in the network (zero while there is
    /// none) over the observation window.
    pub mean_oldest_age: Option<f64>,
}

/// Phase occupations and reconfiguration-start rates over the observation window, rebuilt from
/// the complete event records of a run.
#[derive(Clone, Debug)]
pub struct Reconstruction {
    /// Service occupation per class.
    pub rho: Vec<f64>,
    /// Idle occupation per configuration.
    pub z: Vec<f64>,
    /// Reconfiguration starts per unit time per ordered pair.
    pub nu: Vec<Vec<f64>>,
    /// Fraction of the observation window spent reconfiguring.
    pub switch_time_fraction: f64,
    /// Length of the observation window.
    pub window: f64,
}

/// The rebuilt occupations and rates, or `None` for a compact result.
pub fn reconstruct(result: &Value, n: usize) -> Option<Reconstruction> {
    let ws = result["warmup"].as_f64()?;
    let measure = result["measure"].as_f64()?;
    let we = ws + measure;
    let overlap = |a: f64, b: f64| (b.min(we) - a.max(ws)).max(0.0);
    let mut service = vec![0.0; n];
    for ev in result["completion_trace"].as_array()? {
        let t = ev["time"].as_f64()?;
        let s = ev["service_duration"].as_f64()?;
        let q = ev["queue"].as_u64()? as usize;
        service[q] += overlap(t - s, t);
    }
    let mut nu = vec![vec![0.0; n]; n];
    let mut switch_time = 0.0;
    for sw in result["switches"].as_array()? {
        let start = sw["start"].as_f64()?;
        let end = start + sw["elapsed"].as_f64()?;
        switch_time += overlap(start, end);
        if ws <= start && start < we {
            nu[sw["from"].as_u64()? as usize][sw["to"].as_u64()? as usize] += 1.0;
        }
    }
    let mut visit = vec![0.0; n];
    for v in result["visits"].as_array()? {
        let q = v["queue"].as_u64()? as usize;
        visit[q] += overlap(v["start"].as_f64()?, v["end"].as_f64()?);
    }
    Some(Reconstruction {
        rho: service.iter().map(|s| s / measure).collect(),
        z: (0..n)
            .map(|j| (visit[j] - service[j]).max(0.0) / measure)
            .collect(),
        nu: nu
            .iter()
            .map(|r| r.iter().map(|c| c / measure).collect())
            .collect(),
        switch_time_fraction: switch_time / measure,
        window: measure,
    })
}

/// One simulated run, condensed.
#[derive(Clone, Debug)]
pub struct Run {
    pub seed: u64,
    /// `completed`, `censored`, `deadline_unresolved` or `aborted`.
    pub status: String,
    pub abort_reason: String,
    /// The engine's growing-backlog flag.
    pub growth: bool,
    /// Service, reconfiguration and idle fractions of the observation window.
    pub phase: (f64, f64, f64),
    /// Reconfigurations started per unit time in the observation window, over all pairs.
    pub switch_rate: f64,
    pub reconstruction: Option<Reconstruction>,
    pub classes: Vec<ClassCounts>,
    pub decision_seconds: f64,
    pub decisions: u64,
    pub runtime_seconds: f64,
    pub events: u64,
    pub max_backlog: Option<f64>,
    /// Completed sojourn times per class, kept for runs with records.
    pub sojourns: Option<Vec<Vec<f64>>>,
}

/// One run of `policy` (`{"name", "params"}`) on `model`, condensed: per-class cohort counts,
/// exceedance counts of the sojourn times on the grid `xs` of multiples of the deadline, phase
/// fractions and the reconfiguration rate. With `records` the run keeps its complete event
/// records, from which the phase occupations and reconfiguration rates are rebuilt, and its
/// sojourn times.
pub fn run(
    model: &Model,
    policy: &Value,
    seed: u64,
    records: bool,
    window: &Window,
    xs: &[f64],
) -> Run {
    let mut system = model.system_json(window.warmup, window.measure, window.followup, records);
    if let Some(w) = window.max_wall_seconds {
        system["max_wall_seconds"] = json!(w);
    }
    let r =
        run_job(&json!({"system": system, "policy": policy, "seed": seed, "compact": !records}));
    let pf = &r["phase_fractions"];
    let switch_rate: f64 = r["edge_metrics"]
        .as_object()
        .map(|m| {
            m.values()
                .map(|e| e["rate_per_wall_time"].as_f64().unwrap_or(0.0))
                .sum()
        })
        .unwrap_or(0.0);
    let mut classes = Vec::with_capacity(model.n);
    let mut sojourns = Vec::with_capacity(model.n);
    for q in 0..model.n {
        let pq = &r["per_queue"][q];
        let d = to_f64(&model.d[q]);
        let completed: Vec<f64> = pq["completed_sojourns"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_f64).collect())
            .unwrap_or_default();
        let censored_lb: Vec<f64> = pq["censored_lower_bounds"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_f64).collect())
            .unwrap_or_default();
        let exceed: Vec<u64> = xs
            .iter()
            .map(|x| {
                let t = x * d;
                completed.iter().filter(|s| **s > t).count() as u64
                    + censored_lb.iter().filter(|s| **s > t).count() as u64
            })
            .collect();
        let mut sorted = completed.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let quant = |p: f64| {
            if sorted.is_empty() || !censored_lb.is_empty() {
                None
            } else {
                let k = ((sorted.len() as f64 - 1.0) * p).round() as usize;
                Some(sorted[k.min(sorted.len() - 1)])
            }
        };
        classes.push(ClassCounts {
            arrivals: pq["arrivals"].as_u64().unwrap_or(0),
            misses: pq["deadline_violations"].as_u64().unwrap_or(0),
            unresolved: pq["deadline_unresolved"].as_u64().unwrap_or(0),
            censored: pq["censored"].as_u64().unwrap_or(0),
            exceed,
            p50: quant(0.5),
            p99: quant(0.99),
            mean_oldest_age: pq["mean_oldest_age"].as_f64(),
        });
        sojourns.push(completed);
    }
    Run {
        seed,
        status: r["status"].as_str().unwrap_or("?").into(),
        abort_reason: r["abort_reason"].as_str().unwrap_or("").to_string(),
        growth: r["backlog_diagnostic"] == "growing_backlog",
        phase: (
            pf["service"].as_f64().unwrap_or(0.0),
            pf["switch"].as_f64().unwrap_or(0.0),
            pf["idle"].as_f64().unwrap_or(0.0),
        ),
        switch_rate,
        reconstruction: reconstruct(&r, model.n),
        classes,
        decision_seconds: r["decision_seconds"]["total"].as_f64().unwrap_or(0.0),
        decisions: r["decision_seconds"]["count"].as_u64().unwrap_or(0),
        runtime_seconds: r["runtime_seconds"].as_f64().unwrap_or(0.0),
        events: r["events"].as_u64().unwrap_or(0),
        max_backlog: r["max_backlog"].as_f64(),
        sojourns: if records { Some(sojourns) } else { None },
    }
}
