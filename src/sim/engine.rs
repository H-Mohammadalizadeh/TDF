//! The discrete-event engine: one run of a [`System`] under a [`Policy`], see [`simulate`].
//!
//! The network serves `N` classes, one configuration per class; the jobs of a class wait in a
//! FCFS queue and are served nonpreemptively. In identifiers and output keys, `queue` denotes a
//! class (its index or its queue of waiting jobs), `switch` a reconfiguration, `edge` an ordered
//! pair of configurations, `measure` the length of the observation window, and `law` a probability
//! distribution.

use super::laws::Law;
use super::rng::{Label, Rng, compensated_sum, stream_seed, tuple_seed};
use serde_json::{Map, Value, json};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::time::Instant;

/// What a policy observes at a decision epoch.
#[derive(Debug, Clone)]
pub struct View {
    /// Time of the decision.
    pub now: f64,
    /// The current configuration.
    pub at: usize,
    /// Arrival times of the waiting jobs per class, oldest first (no job is
    /// in service at a decision epoch).
    pub arrivals: Vec<Vec<f64>>,
}

impl View {
    pub fn n(&self) -> usize {
        self.arrivals.len()
    }
}

/// A scheduling policy. `decide` returns the current configuration (serve
/// its oldest job), another configuration with waiting jobs (reconfigure;
/// any other configuration if the policy visits empty configurations), or
/// `-1` (idle until the next arrival).
pub trait Policy {
    fn decide(&mut self, view: &View) -> i64;

    /// Whether the policy also passes through configurations whose class is empty, as cyclic
    /// polling does: it is then consulted when every queue is empty, and it may reconfigure
    /// towards an empty queue. `false` by default.
    fn visits_empty(&self) -> bool {
        false
    }
}

/// The simulated system: the parameters of one run.
#[derive(Debug, Clone)]
pub struct System {
    /// Service rate per class.
    pub mu: Vec<f64>,
    /// Arrival rate of jobs per class.
    pub lam: Vec<f64>,
    /// Mean geometric batch size per class; 1 means single Poisson arrivals.
    pub batch_mean: Vec<f64>,
    /// Deadline per class.
    pub deadlines: Vec<f64>,
    /// Target per class; reported in the record, not used by the engine.
    pub epsilon: Vec<f64>,
    /// Warm-up before the observation window.
    pub warmup: f64,
    /// Length of the observation window.
    pub measure: f64,
    /// Time simulated after the observation window to resolve the deadline outcomes.
    pub followup: f64,
    pub seed: u64,
    /// Service-time distribution per class.
    pub service: Vec<Law>,
    /// Reconfiguration-time distributions keyed by the ordered pair `(u, v)`, `u != v`.
    pub switch: HashMap<(usize, usize), Law>,
    /// Limits on processed events, generated jobs and wall-clock seconds; a run that reaches
    /// one is aborted (`None`: no limit).
    pub max_events: Option<u64>,
    pub max_jobs: Option<u64>,
    pub max_wall_seconds: Option<f64>,
    /// Number of equally spaced backlog samples over the observation window.
    pub backlog_samples: usize,
    /// Maximum length of the completion trace.
    pub trace_limit: usize,
}

fn f(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}

fn vec_f(v: &Value, key: &str) -> Result<Vec<f64>, String> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{key} must be a vector"))?
        .iter()
        .map(|x| x.as_f64().ok_or_else(|| format!("{key} must be numeric")))
        .collect()
}

impl System {
    /// Parse the `system` object of a run specification (see [`super::runner::run_job`]):
    ///
    /// * `mu`, `lambda`, `deadlines` (one entry per class), `epsilon`
    ///   (targets, default 0.01), `batch_mean` (a number or one per class,
    ///   default 1);
    /// * `service`: `"exponential"` (default), `"deterministic"`, or a distribution
    ///   (or one entry per class) with mean `1 / mu_i`;
    /// * `switch`: `{"default": law, "edges": {"u,v": law}}` with zero-based
    ///   configurations; the default distribution is a point mass at zero;
    /// * `warmup` (default 0), `measure` (default 100), `followup` (default
    ///   the largest deadline), `seed` (default 1);
    /// * limits `max_events` (default 2,000,000), `max_jobs` (default
    ///   1,000,000), `max_wall_seconds` (default 120); `null` removes a limit;
    /// * `backlog_samples` (default 101) and `trace_limit` (default 10,000).
    pub fn from_json(cfg: &Value) -> Result<System, String> {
        let obj = cfg
            .as_object()
            .ok_or("system configuration must be an object")?;
        const ALLOWED: &[&str] = &[
            "mu",
            "lambda",
            "batch_mean",
            "deadlines",
            "epsilon",
            "warmup",
            "measure",
            "followup",
            "seed",
            "switch",
            "service",
            "max_events",
            "max_jobs",
            "max_wall_seconds",
            "backlog_samples",
            "trace_limit",
        ];
        for key in obj.keys() {
            if !ALLOWED.contains(&key.as_str()) {
                return Err(format!("unknown system key: {key}"));
            }
        }
        let mu = vec_f(cfg, "mu")?;
        let n = mu.len();
        if n == 0 || mu.iter().any(|m| !m.is_finite() || *m <= 0.0) {
            return Err("mu must be strictly positive".into());
        }
        let deadlines = vec_f(cfg, "deadlines")?;
        if deadlines.len() != n || deadlines.iter().any(|d| !d.is_finite() || *d <= 0.0) {
            return Err("deadlines must be strictly positive with N entries".into());
        }
        let epsilon = match obj.get("epsilon") {
            Some(_) => vec_f(cfg, "epsilon")?,
            None => vec![0.01; n],
        };
        if epsilon.len() != n || epsilon.iter().any(|e| !(*e > 0.0 && *e < 1.0)) {
            return Err("0<epsilon<1 with N entries".into());
        }
        let lam = vec_f(cfg, "lambda")?;
        if lam.len() != n || lam.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err("lambda must be nonnegative with N entries".into());
        }
        let batch_mean: Vec<f64> = match obj.get("batch_mean") {
            None | Some(Value::Null) => vec![1.0; n],
            Some(Value::Array(_)) => vec_f(cfg, "batch_mean")?,
            Some(v) => vec![v.as_f64().ok_or("batch_mean must be numeric")?; n],
        };
        if batch_mean.len() != n || batch_mean.iter().any(|b| !b.is_finite() || *b < 1.0) {
            return Err("batch_mean must be >= 1 with N entries".into());
        }
        let warmup = f(cfg, "warmup").unwrap_or(0.0);
        let measure = f(cfg, "measure").unwrap_or(100.0);
        let max_deadline = deadlines.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let followup = f(cfg, "followup").unwrap_or(max_deadline);
        if warmup < 0.0
            || measure <= 0.0
            || followup < 0.0
            || !(warmup + measure + followup).is_finite()
        {
            return Err("invalid observation window".into());
        }
        let seed = match obj.get("seed") {
            None => 1,
            Some(v) => v.as_u64().ok_or("seed must be a nonnegative integer")?,
        };
        // Service-time distributions.
        let service_spec = obj
            .get("service")
            .cloned()
            .unwrap_or(Value::String("exponential".into()));
        let specs: Vec<Value> = match &service_spec {
            Value::Array(a) => a.clone(),
            other => vec![other.clone(); n],
        };
        if specs.len() != n {
            return Err("service-time distribution list must have N entries".into());
        }
        let mut service = Vec::new();
        for (i, spec) in specs.iter().enumerate() {
            let law = match spec {
                Value::String(s) if s == "exponential" => Law::exponential(1.0 / mu[i])?,
                Value::String(s) if s == "deterministic" => Law::deterministic(1.0 / mu[i])?,
                Value::String(_) => {
                    return Err("service string must be exponential or deterministic".into());
                }
                other => Law::from_json(other)?,
            };
            if (law.mean() - 1.0 / mu[i]).abs() > 1e-10 * (1.0 / mu[i]) + 1e-12 {
                return Err("service-time distribution mean must equal 1/mu".into());
            }
            service.push(law);
        }
        // Reconfiguration-time distributions.
        let default_json = json!({"family": "deterministic", "value": 0.0});
        let switch_spec = obj
            .get("switch")
            .cloned()
            .unwrap_or(json!({"default": default_json.clone()}));
        let sw = switch_spec.as_object().ok_or("switch must be an object")?;
        for key in sw.keys() {
            if key != "default" && key != "edges" {
                return Err("switch must contain only default and edges".into());
            }
        }
        let default_law = sw.get("default").cloned().unwrap_or(default_json);
        Law::from_json(&default_law)?;
        let edges = sw.get("edges").cloned().unwrap_or(json!({}));
        let edges = edges.as_object().ok_or("switch edges must be an object")?;
        let mut switch = HashMap::new();
        for (key, spec) in edges {
            let parts: Vec<&str> = key.split(',').collect();
            if parts.len() != 2 {
                return Err("switch edge keys must be zero-based 'u,v'".into());
            }
            let u: usize = parts[0]
                .parse()
                .map_err(|_| "switch edge keys must be zero-based 'u,v'")?;
            let v: usize = parts[1]
                .parse()
                .map_err(|_| "switch edge keys must be zero-based 'u,v'")?;
            if u >= n || v >= n || u == v || *key != format!("{u},{v}") {
                return Err("switch edge keys must be distinct canonical indices".into());
            }
            Law::from_json(spec)?;
        }
        for u in 0..n {
            for v in 0..n {
                if u == v {
                    continue;
                }
                let spec = edges
                    .get(&format!("{u},{v}"))
                    .cloned()
                    .unwrap_or(default_law.clone());
                switch.insert((u, v), Law::from_json(&spec)?);
            }
        }
        let cap_u64 = |key: &str, default: u64| -> Result<Option<u64>, String> {
            match obj.get(key) {
                None => Ok(Some(default)),
                Some(Value::Null) => Ok(None),
                Some(v) => v
                    .as_u64()
                    .filter(|x| *x >= 1)
                    .map(Some)
                    .ok_or_else(|| format!("{key} must be an integer >= 1")),
            }
        };
        let max_events = cap_u64("max_events", 2_000_000)?;
        let max_jobs = cap_u64("max_jobs", 1_000_000)?;
        let max_wall_seconds = match obj.get("max_wall_seconds") {
            None => Some(120.0),
            Some(Value::Null) => None,
            Some(v) => Some(
                v.as_f64()
                    .filter(|x| *x > 0.0)
                    .ok_or("max_wall_seconds must be positive")?,
            ),
        };
        let backlog_samples = obj
            .get("backlog_samples")
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .unwrap_or(101);
        if backlog_samples < 2 {
            return Err("backlog_samples must be an integer >= 2".into());
        }
        let trace_limit = obj
            .get("trace_limit")
            .map(|v| v.as_u64().unwrap_or(u64::MAX) as usize)
            .unwrap_or(10000);
        Ok(System {
            mu,
            lam,
            batch_mean,
            deadlines,
            epsilon,
            warmup,
            measure,
            followup,
            seed,
            service,
            switch,
            max_events,
            max_jobs,
            max_wall_seconds,
            backlog_samples,
            trace_limit,
        })
    }

    pub fn n(&self) -> usize {
        self.mu.len()
    }
}

#[derive(Debug, Clone)]
struct Job {
    arrival: f64,
    cohort: bool,
    completion: Option<f64>,
    late: bool,
}

#[derive(Debug)]
struct Event {
    t: f64,
    priority: u8,
    seq: u64,
    kind: u8, // 0 service end, 1 reconfiguration end, 2 arrival, 3 initial decision
    queue: usize,
    data: f64,
    index: usize,
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Event {}
impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Event {
    // Reversed because `BinaryHeap` is a max-heap: the earliest event comes first, then the lower
    // priority value (service end, reconfiguration end, arrival, initial decision), then the
    // earlier insertion.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .t
            .partial_cmp(&self.t)
            .unwrap_or(Ordering::Equal)
            .then(other.priority.cmp(&self.priority))
            .then(other.seq.cmp(&self.seq))
    }
}

/// The deadline of a job, ordered so that the heap yields the earliest deadline first (ties by
/// class, then job index).
#[derive(Debug, Clone, PartialEq)]
struct Expiry {
    deadline: f64,
    queue: usize,
    index: usize,
}
impl Eq for Expiry {}
impl PartialOrd for Expiry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Expiry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .deadline
            .partial_cmp(&self.deadline)
            .unwrap_or(Ordering::Equal)
            .then(other.queue.cmp(&self.queue))
            .then(other.index.cmp(&self.index))
    }
}

/// One reconfiguration of the run.
#[derive(Debug, Clone)]
struct Switch {
    from: usize,
    to: usize,
    start: f64,
    end: Option<f64>,
    duration: Option<f64>,
    elapsed: f64,
    completed: bool,
    started_in_window: bool,
    planned_end: f64,
}

/// Geometric batch size on `{1, 2, ...}` with the given mean, drawn from the
/// class's arrival stream. A mean of one makes no draw, so single Poisson
/// arrivals consume exactly one draw (the interarrival time) per arrival.
fn geometric_batch(mean: f64, rng: &mut Rng) -> usize {
    if mean <= 1.0 {
        return 1;
    }
    let p = 1.0 / mean;
    let u = 1.0 - rng.random(); // (0, 1]
    1 + (u.ln() / (1.0 - p).ln()).floor().max(0.0) as usize
}

/// Quantile of `values` with linear interpolation between order statistics.
fn quantile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = (v.len() - 1) as f64 * p;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    Some(v[lo] + (v[hi] - v[lo]) * (pos - lo as f64))
}

fn mean_or_null(values: &[f64]) -> Value {
    if values.is_empty() {
        Value::Null
    } else {
        json!(compensated_sum(values) / values.len() as f64)
    }
}

/// Simulate one run of `sys` under `policy` and return its record.
///
/// Jobs of class `i` arrive in a Poisson stream of rate `lambda_i`, or in geometric batches with
/// the same job rate. A decision is due whenever a service or a reconfiguration ends or an
/// arrival finds the network idle. If every queue is empty, the network idles in place; if a
/// reconfiguration has just ended at a nonempty queue, its head is served; otherwise the policy
/// chooses: serve the head of the current queue, reconfigure towards a nonempty queue (the
/// reconfiguration time is drawn from the distribution of the pair), or idle until the next
/// arrival. A policy that visits empty configurations ([`Policy::visits_empty`]) is consulted
/// also when every queue is empty, and it may reconfigure towards an empty queue.
///
/// Every job that arrives in the observation window `[warmup, warmup + measure)` belongs to the
/// cohort. The run continues for a follow-up period so that the deadline outcomes of the cohort
/// are resolved; a cohort job not completed at the end of the run is censored, and it is
/// unresolved when its deadline has not passed either. Phase occupations (service,
/// reconfiguration, idle), reconfiguration rates and backlog statistics are measured over the
/// observation window.
///
/// Random draws come from separate streams per class and per ordered pair (see
/// [`super::rng`]), so the draws of one stream do not depend on how the policy uses the others:
/// with the same seed, the arrival times and the service time of every job are the same under
/// every policy.
///
/// The record holds the status, per-class cohort statistics (`per_queue`), phase times and
/// fractions, conservation checks, reconfiguration counts and rates per pair (`edge_metrics`),
/// the reconfiguration and visit records, the completion trace, the backlog samples with the
/// growth diagnostic, and the decision count and time. An invalid policy decision aborts the
/// run with a record of the status and the reason only.
pub fn simulate(sys: &System, policy: &mut dyn Policy) -> Value {
    let started = Instant::now();
    let n = sys.n();
    let seed = sys.seed;
    // One service stream per class, drawn in job order (FCFS).
    let mut service_stream: Vec<Rng> = (0..n)
        .map(|q| {
            Rng::xoshiro(tuple_seed(
                seed,
                &[Label::Str("service_stream"), Label::Int(q as u64)],
            ))
        })
        .collect();
    let ws = sys.warmup;
    let we = ws + sys.measure;
    let limit = we + sys.followup;

    // One arrival stream per class and one reconfiguration stream per ordered pair.
    let mut arrival_rng: Vec<Rng> = (0..n)
        .map(|i| {
            Rng::mersenne(stream_seed(
                seed,
                &[Label::Str("arrivals"), Label::Int(i as u64)],
            ))
        })
        .collect();
    let mut switch_rng: HashMap<(usize, usize), Rng> = HashMap::new();
    for u in 0..n {
        for v in 0..n {
            if u != v {
                switch_rng.insert(
                    (u, v),
                    Rng::mersenne(stream_seed(
                        seed,
                        &[
                            Label::Str("switch"),
                            Label::Int(u as u64),
                            Label::Int(v as u64),
                        ],
                    )),
                );
            }
        }
    }
    let next_arrival = |q: usize, rng: &mut Rng, now: f64| -> f64 {
        let hazard = rng.expovariate(1.0);
        let rate = sys.lam[q] / sys.batch_mean[q];
        if rate != 0.0 {
            now + hazard / rate
        } else {
            f64::INFINITY
        }
    };

    let mut queues: Vec<VecDeque<usize>> = vec![VecDeque::new(); n];
    let mut jobs: Vec<Vec<Job>> = vec![Vec::new(); n];
    let mut expiry: BinaryHeap<Expiry> = BinaryHeap::new();
    let mut events: BinaryHeap<Event> = BinaryHeap::new();
    let mut seq: u64 = 0;
    let mut now = 0.0f64;
    let mut at = 0usize;
    let mut phase: u8 = 2; // 0 service, 1 reconfiguration, 2 idle
    let mut in_service: Option<(usize, usize)> = None; // (queue, index)
    let mut served: usize = 0;
    let mut visit_start = 0.0f64;
    let mut visits: Vec<Value> = Vec::new();
    let mut all_arrivals = vec![0u64; n];
    let mut all_completions = vec![0u64; n];
    let mut crossing_counts = vec![0u64; n];
    let mut cohort_crossing_counts = vec![0u64; n];
    let mut completion_window = vec![0u64; n];
    let mut phase_times = [0.0f64; 3];
    let mut queue_area = vec![0.0f64; n];
    let mut unfinished_area = vec![0.0f64; n];
    let mut age_area = vec![0.0f64; n];
    let mut completion_trace: Vec<Value> = Vec::new();
    let mut switches: Vec<Switch> = Vec::new();
    let mut decision_times: Vec<f64> = Vec::new();
    let mut backlog_series: Vec<Value> = Vec::new();
    let samples = sys.backlog_samples.max(2);
    let mut next_sample: usize = 0;
    let mut event_count: u64 = 0;
    let mut status = "completed";
    let mut abort_reason: Option<&str> = None;
    let mut max_backlog: usize = 0;
    let trace_limit = sys.trace_limit;
    let mut traces_truncated = false;

    macro_rules! push_event {
        ($t:expr, $priority:expr, $kind:expr, $queue:expr, $data:expr, $index:expr) => {
            if $t.is_finite() && $t <= limit {
                seq += 1;
                events.push(Event {
                    t: $t,
                    priority: $priority,
                    seq,
                    kind: $kind,
                    queue: $queue,
                    data: $data,
                    index: $index,
                });
            }
        };
    }

    let enqueue = |q: usize,
                   t: f64,
                   jobs: &mut Vec<Vec<Job>>,
                   queues: &mut Vec<VecDeque<usize>>,
                   expiry: &mut BinaryHeap<Expiry>,
                   all_arrivals: &mut Vec<u64>,
                   in_service: &Option<(usize, usize)>,
                   max_backlog: &mut usize| {
        let cohort = ws <= t && t < we;
        let index = jobs[q].len();
        jobs[q].push(Job {
            arrival: t,
            cohort,
            completion: None,
            late: false,
        });
        queues[q].push_back(index);
        expiry.push(Expiry {
            deadline: t + sys.deadlines[q],
            queue: q,
            index,
        });
        all_arrivals[q] += 1;
        let backlog: usize =
            queues.iter().map(|q| q.len()).sum::<usize>() + in_service.is_some() as usize;
        if backlog > *max_backlog {
            *max_backlog = backlog;
        }
    };

    for q in 0..n {
        let t = next_arrival(q, &mut arrival_rng[q], 0.0);
        push_event!(t, 2, 2, q, 0.0, 0);
    }

    let make_view =
        |now: f64, at: usize, queues: &Vec<VecDeque<usize>>, jobs: &Vec<Vec<Job>>| View {
            now,
            at,
            arrivals: (0..n)
                .map(|q| queues[q].iter().map(|&i| jobs[q][i].arrival).collect())
                .collect(),
        };

    // Advance the clock to `t`: accumulate the phase, backlog and head-of-line age integrals over
    // the part of `[now, t)` inside the observation window, take the backlog samples that fall
    // in it, and mark the uncompleted jobs whose deadline passed before `t` as late.
    macro_rules! account {
        ($t:expr) => {{
            let t: f64 = $t;
            let lo = now.max(ws);
            let hi = t.min(we);
            let dt = (hi - lo).max(0.0);
            phase_times[phase as usize] += dt;
            for q in 0..n {
                queue_area[q] += dt * queues[q].len() as f64;
                let extra = match in_service { Some((sq, _)) if sq == q => 1usize, _ => 0 };
                unfinished_area[q] += dt * (queues[q].len() + extra) as f64;
                let mut head: Option<f64> = queues[q].front().map(|&i| jobs[q][i].arrival);
                if extra == 1 {
                    let sa = jobs[q][in_service.unwrap().1].arrival;
                    head = Some(match head { Some(h) => h.min(sa), None => sa });
                }
                if let Some(h) = head {
                    if dt != 0.0 {
                        age_area[q] += dt * ((lo + hi) / 2.0 - h);
                    }
                }
            }
            while next_sample < samples {
                let when = ws + next_sample as f64 * sys.measure / (samples - 1) as f64;
                if when > t.min(we) {
                    break;
                }
                if when >= now {
                    let backlog: Vec<usize> = (0..n)
                        .map(|i| queues[i].len() + match in_service { Some((sq, _)) if sq == i => 1, _ => 0 })
                        .collect();
                    let mut ages: Vec<f64> = (0..n).map(|i| queues[i].front().map(|&j| when - jobs[i][j].arrival).unwrap_or(0.0)).collect();
                    if let Some((sq, si)) = in_service {
                        ages[sq] = ages[sq].max(when - jobs[sq][si].arrival);
                    }
                    backlog_series.push(json!({"time": when, "backlog": backlog, "oldest_age": ages}));
                }
                next_sample += 1;
            }
            while let Some(top) = expiry.peek() {
                if top.deadline < t {
                    let e = expiry.pop().unwrap();
                    let job = &mut jobs[e.queue][e.index];
                    if job.completion.is_none() && !job.late {
                        job.late = true;
                        crossing_counts[e.queue] += 1;
                        cohort_crossing_counts[e.queue] += job.cohort as u64;
                    }
                } else {
                    break;
                }
            }
            now = t;
        }};
    }

    macro_rules! start_service {
        () => {{
            if let Some(index) = queues[at].pop_front() {
                in_service = Some((at, index));
                phase = 0;
                let duration = sys.service[at].sample(&mut service_stream[at]);
                assert!(
                    duration.is_finite() && duration > 0.0,
                    "service draws must be finite and strictly positive"
                );
                push_event!(now + duration, 0, 0, at, duration, index);
            }
        }};
    }

    push_event!(0.0_f64, 3, 3, 0, 0.0, 0);
    let mut policy_error: Option<String> = None;
    let visits_empty = policy.visits_empty();
    while let Some(top) = events.peek() {
        let events_exhausted = sys.max_events.is_some_and(|m| event_count >= m);
        let jobs_exhausted = sys
            .max_jobs
            .is_some_and(|m| jobs.iter().map(|j| j.len() as u64).sum::<u64>() >= m);
        let wall_exhausted = sys
            .max_wall_seconds
            .is_some_and(|w| started.elapsed().as_secs_f64() >= w);
        if events_exhausted || jobs_exhausted || wall_exhausted {
            status = "aborted";
            abort_reason = Some(if events_exhausted {
                "max_events"
            } else if jobs_exhausted {
                "max_jobs"
            } else {
                "max_wall_seconds"
            });
            break;
        }
        let t = top.t;
        account!(t);
        let mut must_decide = false;
        let mut switched = false;
        while events.peek().is_some_and(|e| e.t == t) {
            let events_exhausted = sys.max_events.is_some_and(|m| event_count >= m);
            let jobs_exhausted = sys
                .max_jobs
                .is_some_and(|m| jobs.iter().map(|j| j.len() as u64).sum::<u64>() >= m);
            if events_exhausted || jobs_exhausted {
                status = "aborted";
                abort_reason = Some(if events_exhausted {
                    "max_events"
                } else {
                    "max_jobs"
                });
                break;
            }
            let ev = events.pop().unwrap();
            event_count += 1;
            match ev.kind {
                2 => {
                    let q = ev.queue;
                    let batch = geometric_batch(sys.batch_mean[q], &mut arrival_rng[q]);
                    for _ in 0..batch {
                        enqueue(
                            q,
                            now,
                            &mut jobs,
                            &mut queues,
                            &mut expiry,
                            &mut all_arrivals,
                            &in_service,
                            &mut max_backlog,
                        );
                    }
                    let nt = next_arrival(q, &mut arrival_rng[q], now);
                    push_event!(nt, 2, 2, q, 0.0, 0);
                    must_decide |= phase == 2;
                }
                0 => {
                    let (sq, si) = in_service.expect("service end without job");
                    let job = &mut jobs[sq][si];
                    job.completion = Some(now);
                    all_completions[at] += 1;
                    completion_window[at] += (ws <= now && now < we) as u64;
                    served += 1;
                    let late = now > job.arrival + sys.deadlines[at];
                    if completion_trace.len() < trace_limit {
                        completion_trace.push(json!({"queue": at, "index": si, "arrival": job.arrival, "time": now, "sojourn": now - job.arrival, "service_duration": ev.data, "cohort": job.cohort, "late": late}));
                    } else {
                        traces_truncated = true;
                    }
                    in_service = None;
                    phase = 2;
                    must_decide = true;
                }
                1 => {
                    let sw = &mut switches[ev.index];
                    sw.end = Some(now);
                    sw.duration = Some(now - sw.start);
                    sw.elapsed = now - sw.start;
                    sw.completed = true;
                    at = sw.to;
                    served = 0;
                    visit_start = now;
                    phase = 2;
                    switched = true;
                    must_decide = true;
                }
                _ => {
                    must_decide = true;
                }
            }
        }
        if status == "aborted" {
            break;
        }
        if switched && !queues[at].is_empty() {
            start_service!();
        } else if must_decide && now < limit {
            if !visits_empty && queues.iter().all(|q| q.is_empty()) {
                phase = 2;
            } else {
                let tick = Instant::now();
                let action: i64 = policy.decide(&make_view(now, at, &queues, &jobs));
                decision_times.push(tick.elapsed().as_secs_f64());
                if action < -1 || action >= n as i64 {
                    policy_error = Some(format!("invalid policy action {action}"));
                    break;
                }
                if !visits_empty
                    && action >= 0
                    && action as usize != at
                    && queues[action as usize].is_empty()
                {
                    policy_error = Some("switch destination must be nonempty".into());
                    break;
                }
                if action == -1 {
                    phase = 2;
                } else if action as usize == at {
                    if !queues[at].is_empty() {
                        start_service!();
                    } else {
                        phase = 2;
                    }
                } else {
                    let dest = action as usize;
                    let edge = (at, dest);
                    let law = &sys.switch[&edge];
                    let duration = law.sample(switch_rng.get_mut(&edge).unwrap());
                    assert!(
                        duration.is_finite() && duration >= 0.0,
                        "switch draws must be finite and nonnegative"
                    );
                    visits.push(
                        json!({"queue": at, "start": visit_start, "end": now, "completed": served}),
                    );
                    switches.push(Switch {
                        from: at,
                        to: dest,
                        start: now,
                        end: None,
                        duration: None,
                        elapsed: 0.0,
                        completed: false,
                        started_in_window: ws <= now && now < we,
                        planned_end: now + duration,
                    });
                    phase = 1;
                    push_event!(now + duration, 1, 1, dest, duration, switches.len() - 1);
                }
            }
        }
        if now >= limit {
            break;
        }
    }
    if let Some(err) = policy_error {
        return json!({"status": "aborted", "abort_reason": format!("policy error: {err}")});
    }
    if status != "aborted" {
        account!(limit);
    }
    let observed_duration = (now.min(we) - ws).max(0.0);
    visits.push(json!({"queue": at, "start": visit_start, "end": now, "completed": served, "right_censored": true}));

    // Reconfiguration counts, durations and rates per ordered pair.
    let mut edge_order: Vec<String> = Vec::new();
    let mut edge_metrics: Map<String, Value> = Map::new();
    let mut switch_rows: Vec<Value> = Vec::new();
    for sw in switches.iter_mut() {
        let edge = format!("{},{}", sw.from, sw.to);
        if !edge_metrics.contains_key(&edge) {
            edge_order.push(edge.clone());
            edge_metrics.insert(
                edge.clone(),
                json!({"count_started_window": 0, "count_all": 0, "completed_durations": [], "censored_elapsed": []}),
            );
        }
        let row = edge_metrics.get_mut(&edge).unwrap();
        row["count_all"] = json!(row["count_all"].as_u64().unwrap() + 1);
        row["count_started_window"] =
            json!(row["count_started_window"].as_u64().unwrap() + sw.started_in_window as u64);
        // A reconfiguration in progress at the end of the run is censored at `now`.
        let actual_end = sw.planned_end.min(now);
        sw.elapsed = actual_end - sw.start;
        let key = if sw.completed {
            "completed_durations"
        } else {
            "censored_elapsed"
        };
        row[key].as_array_mut().unwrap().push(json!(sw.elapsed));
        switch_rows.push(json!({"from": sw.from, "to": sw.to, "start": sw.start, "end": sw.end, "duration": sw.duration, "elapsed": sw.elapsed, "completed": sw.completed, "started_in_window": sw.started_in_window}));
    }
    for edge in &edge_order {
        let row = edge_metrics.get_mut(edge).unwrap();
        let count = row["count_started_window"].as_u64().unwrap() as f64;
        row["rate_per_wall_time"] = if observed_duration != 0.0 {
            json!(count / observed_duration)
        } else {
            Value::Null
        };
    }

    // Per-class cohort statistics.
    let mut per_queue: Vec<Value> = Vec::new();
    let mut any_unresolved = false;
    let mut any_censored = false;
    for q in 0..n {
        let cohort: Vec<&Job> = jobs[q].iter().filter(|j| j.cohort).collect();
        let completed: Vec<&Job> = cohort
            .iter()
            .cloned()
            .filter(|j| j.completion.is_some())
            .collect();
        let censored: Vec<&Job> = cohort
            .iter()
            .cloned()
            .filter(|j| j.completion.is_none())
            .collect();
        let sojourns: Vec<f64> = completed
            .iter()
            .map(|j| j.completion.unwrap() - j.arrival)
            .collect();
        let unresolved = censored
            .iter()
            .filter(|j| now <= j.arrival + sys.deadlines[q])
            .count();
        let late = cohort.iter().filter(|j| j.late).count();
        let fully_observed = !cohort.is_empty() && censored.is_empty();
        let nc = cohort.len() as f64;
        any_unresolved |= unresolved > 0;
        any_censored |= !censored.is_empty();
        let mean_sojourn = if fully_observed {
            json!(compensated_sum(&sojourns) / sojourns.len() as f64)
        } else {
            Value::Null
        };
        let quant = |p: f64| {
            if fully_observed {
                quantile(&sojourns, p)
                    .map(|v| json!(v))
                    .unwrap_or(Value::Null)
            } else {
                Value::Null
            }
        };
        per_queue.push(json!({
            "queue": q, "active": sys.lam[q] > 0.0, "lambda": sys.lam[q], "mu": sys.mu[q], "deadline": sys.deadlines[q], "epsilon": sys.epsilon[q],
            "arrivals": cohort.len(), "completed": completed.len(), "unfinished": censored.len(), "censored": censored.len(),
            "deadline_violations": late, "deadline_unresolved": unresolved,
            "violation_rate": if !cohort.is_empty() && unresolved == 0 { json!(late as f64 / nc) } else { Value::Null },
            "violation_rate_lower": if !cohort.is_empty() { json!(late as f64 / nc) } else { Value::Null },
            "violation_rate_upper": if !cohort.is_empty() { json!((late + unresolved) as f64 / nc) } else { Value::Null },
            "mean_sojourn": mean_sojourn, "p50_sojourn": quant(0.5), "p95_sojourn": quant(0.95), "p99_sojourn": quant(0.99),
            "completed_sojourns": sojourns,
            "censored_lower_bounds": censored.iter().map(|j| now - j.arrival).collect::<Vec<_>>(),
            "completed_only_mean": mean_or_null(&sojourns),
            "throughput": if observed_duration != 0.0 { json!(completion_window[q] as f64 / observed_duration) } else { Value::Null },
            "completions_window": completion_window[q],
            "mean_waiting_backlog": if observed_duration != 0.0 { json!(queue_area[q] / observed_duration) } else { Value::Null },
            "mean_unfinished_backlog": if observed_duration != 0.0 { json!(unfinished_area[q] / observed_duration) } else { Value::Null },
            "mean_oldest_age": if observed_duration != 0.0 { json!(age_area[q] / observed_duration) } else { Value::Null },
            "final_unfinished": jobs[q].iter().filter(|j| j.completion.is_none()).count(),
            "all_arrivals": all_arrivals[q], "all_completions": all_completions[q], "all_deadline_crossings": crossing_counts[q]
        }));
    }
    let total_unfinished: u64 = per_queue
        .iter()
        .map(|p| p["final_unfinished"].as_u64().unwrap())
        .sum();
    let phase_total = compensated_sum(&phase_times);
    let conservation = json!({
        "arrivals": all_arrivals.iter().sum::<u64>(),
        "completions": all_completions.iter().sum::<u64>(), "unfinished": total_unfinished,
        "arrival_balance": all_arrivals.iter().sum::<u64>() as i64 - all_completions.iter().sum::<u64>() as i64 - total_unfinished as i64,
        "observed_window": observed_duration, "phase_total": phase_total, "time_error": phase_total - observed_duration,
        "deadline_count_consistent": (0..n).all(|i| cohort_crossing_counts[i] == per_queue[i]["deadline_violations"].as_u64().unwrap())
    });
    // Growth diagnostic: least-squares slope of the total backlog over the samples of the
    // observation window, flagged when it exceeds 1 % of the total arrival rate.
    let times: Vec<f64> = backlog_series
        .iter()
        .map(|r| r["time"].as_f64().unwrap())
        .collect();
    let totals: Vec<f64> = backlog_series
        .iter()
        .map(|r| {
            r["backlog"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .sum::<u64>() as f64
        })
        .collect();
    let mean_t = if times.is_empty() {
        0.0
    } else {
        compensated_sum(&times) / times.len() as f64
    };
    let mean_b = if totals.is_empty() {
        0.0
    } else {
        compensated_sum(&totals) / totals.len() as f64
    };
    let variance_t: f64 = compensated_sum(
        &times
            .iter()
            .map(|t| (t - mean_t).powf(2.0))
            .collect::<Vec<_>>(),
    );
    let slope: Option<f64> = if variance_t != 0.0 {
        Some(
            compensated_sum(
                &times
                    .iter()
                    .zip(&totals)
                    .map(|(t, b)| (t - mean_t) * (b - mean_b))
                    .collect::<Vec<_>>(),
            ) / variance_t,
        )
    } else {
        None
    };
    let lam_sum: f64 = compensated_sum(&sys.lam);
    let growing = slope.is_some_and(|s| s > 0.01 * lam_sum.max(0.01));
    if status != "aborted" {
        status = if any_unresolved {
            "deadline_unresolved"
        } else if any_censored {
            "censored"
        } else {
            "completed"
        };
    }
    json!({
        "seed": seed, "status": status, "abort_reason": abort_reason,
        "end_time": now, "warmup": ws, "measure": sys.measure, "followup": limit - we,
        "per_queue": per_queue,
        "phase_times": {"service": phase_times[0], "switch": phase_times[1], "idle": phase_times[2]},
        "phase_fractions": {
            "service": if observed_duration != 0.0 { json!(phase_times[0] / observed_duration) } else { Value::Null },
            "switch": if observed_duration != 0.0 { json!(phase_times[1] / observed_duration) } else { Value::Null },
            "idle": if observed_duration != 0.0 { json!(phase_times[2] / observed_duration) } else { Value::Null }
        },
        "conservation": conservation, "edge_metrics": edge_metrics, "switches": switch_rows, "visits": visits,
        "completion_trace": completion_trace,
        "decision_seconds": {"count": decision_times.len(), "total": compensated_sum(&decision_times)},
        "backlog_series": backlog_series, "backlog_slope": slope,
        "backlog_diagnostic": if growing { "growing_backlog" } else { "insufficient_evidence_of_stability" },
        "max_backlog": max_backlog,
        "runtime_seconds": started.elapsed().as_secs_f64(), "events": event_count, "traces_truncated": traces_truncated,
        "batch_mean": sys.batch_mean
    })
}
