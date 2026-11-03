//! Tests of exhaustive cyclic polling (EC, `ec`): its decisions on given views; in complete run
//! records it follows its cycle, every visit is exhaustive and it never idles; it matches the mean
//! waiting time of symmetric exhaustive polling and reconfigures a fraction `1 - rho` of the time;
//! and its miss probability at N = 8, load 0.7 and deadline 6 matches an independent simulation
//! (52.9 % with deterministic reconfiguration times).

use serde_json::{Map, Value, json};
use tdf::sim::engine::View;
use tdf::sim::laws::Law;
use tdf::sim::policies::make_policy;
use tdf::sim::runner::run_job;

/// Runs in the comparison of miss probabilities.
const SEEDS: u64 = 8;
/// Tolerance on a class-averaged miss probability in percentage points: at least four standard
/// errors of its difference from the reference (0.14 to 0.17 points for the mean of `SEEDS` runs
/// of 2·10^4 s, at most 0.11 points for the reference).
const TOL_MISS: f64 = 0.8;
/// Relative tolerance on the mean sojourn time and absolute tolerance on the reconfiguration
/// fraction, both at least four standard errors of the mean of eight runs of 4·10^4 s (about
/// 0.35 % and 0.001).
const TOL_SOJOURN: f64 = 0.015;
const TOL_SWITCH: f64 = 0.005;

/// Deterministic reconfiguration times `tau(u, v)` for the ordered pairs of `n` configurations.
fn pair_laws(n: usize, tau: impl Fn(usize, usize) -> f64) -> Vec<((usize, usize), Law)> {
    (0..n)
        .flat_map(|u| (0..n).filter(move |v| *v != u).map(move |v| (u, v)))
        .map(|(u, v)| ((u, v), Law::deterministic(tau(u, v)).unwrap()))
        .collect()
}

fn view(now: f64, at: usize, arrivals: Vec<Vec<f64>>) -> View {
    View { now, at, arrivals }
}

#[test]
fn ec_serves_exhaustively_and_moves_on_in_fixed_cyclic_order() {
    let laws = pair_laws(3, |_, _| 0.5);
    let mut ec = make_policy("ec", &Map::new(), &laws, &[6.0; 3]).unwrap();
    assert!(ec.visits_empty());
    let a = vec![vec![0.5], vec![0.2], vec![]];
    assert_eq!(ec.decide(&view(1.0, 1, a)), 1);
    // The current class is empty: the next configuration, although its class is empty too.
    let a = vec![vec![0.5], vec![], vec![]];
    assert_eq!(ec.decide(&view(1.0, 1, a)), 2);
    // Every class is empty: keep cycling.
    assert_eq!(ec.decide(&view(1.0, 2, vec![vec![]; 3])), 0);
    // One class: nothing to cycle through, so stay (the engine idles).
    let mut single = make_policy("ec", &Map::new(), &[], &[6.0]).unwrap();
    assert_eq!(single.decide(&view(1.0, 0, vec![vec![]])), 0);
}

#[test]
fn invalid_ec_jobs_are_reported_as_aborted_runs() {
    let aborted = |policy: Value, switch: Value| {
        let system = json!({"mu": [4.0, 4.0], "lambda": [1.0, 1.0], "deadlines": [3.0, 3.0],
                            "switch": switch, "measure": 100.0});
        let r = run_job(&json!({"system": system, "policy": policy, "seed": 1}));
        assert_eq!(r["status"], "aborted", "{policy}");
        r["abort_reason"].as_str().unwrap().to_string()
    };
    let half = json!({"default": {"family": "deterministic", "value": 0.5}});
    let reason = aborted(json!({"name": "ec", "params": {"window": 1}}), half);
    assert!(reason.contains("unknown parameters"), "{reason}");
    // Without reconfiguration time, an empty network would cycle forever at one instant.
    let reason = aborted(json!({"name": "ec", "params": {}}), json!({}));
    assert!(
        reason.contains("positive mean reconfiguration time"),
        "{reason}"
    );
}

/// A symmetric network: `n` classes with service rate 4, total load `rho`, deadline `d` and the
/// reconfiguration-time distribution `law` for every pair; complete records, no limits.
fn symmetric(n: usize, rho: f64, d: f64, law: Value, window: [f64; 3]) -> Value {
    json!({"mu": vec![4.0; n], "lambda": vec![4.0 * rho / n as f64; n], "deadlines": vec![d; n],
           "switch": {"default": law}, "warmup": window[0], "measure": window[1],
           "followup": window[2], "trace_limit": Value::Null, "max_events": Value::Null,
           "max_jobs": Value::Null, "max_wall_seconds": Value::Null})
}

fn det(value: f64) -> Value {
    json!({"family": "deterministic", "value": value})
}

fn exp(mean: f64) -> Value {
    json!({"family": "exponential", "mean": mean})
}

/// One completed run of `policy`, with conserved jobs and consistent deadline counts.
fn simulate(system: &Value, policy: &str, seed: u64) -> Value {
    let job = json!({"system": system, "policy": {"name": policy, "params": {}}, "seed": seed});
    let r = run_job(&job);
    assert_eq!(r["status"], "completed", "{policy}: {}", r["abort_reason"]);
    assert_eq!(r["conservation"]["arrival_balance"], 0, "{policy}");
    assert_eq!(
        r["conservation"]["deadline_count_consistent"], true,
        "{policy}"
    );
    r
}

fn index(v: &Value, key: &str) -> usize {
    v[key].as_u64().unwrap() as usize
}

/// Arrival and service-start times of the completed jobs per class, from the completion trace.
fn served(r: &Value, n: usize) -> Vec<Vec<(f64, f64)>> {
    let mut jobs = vec![Vec::new(); n];
    for c in r["completion_trace"].as_array().unwrap() {
        let start = c["time"].as_f64().unwrap() - c["service_duration"].as_f64().unwrap();
        jobs[index(c, "queue")].push((c["arrival"].as_f64().unwrap(), start));
    }
    jobs
}

/// Arrival times of the jobs of one class that wait at a decision epoch `s`, including a job that
/// arrived at `s` and found the network idle. Complete for `s` before the end of the observation
/// window of a completed run: every job that arrived by then was served.
fn waiting(jobs: &[(f64, f64)], s: f64) -> Vec<f64> {
    jobs.iter()
        .filter(|(a, start)| *a <= s && *start > s)
        .map(|(a, _)| *a)
        .collect()
}

#[test]
fn ec_follows_its_cycle_serves_exhaustively_and_never_idles() {
    let n = 4;
    let system = symmetric(n, 0.5, 4.0, exp(0.5), [200.0, 5000.0, 200.0]);
    let r = simulate(&system, "ec", 3);
    let jobs = served(&r, n);
    let mut checked = 0;
    for sw in r["switches"].as_array().unwrap() {
        let (from, to, s) = (
            index(sw, "from"),
            index(sw, "to"),
            sw["start"].as_f64().unwrap(),
        );
        assert_eq!(to, (from + 1) % n, "{sw}");
        if s < 5200.0 {
            assert!(
                waiting(&jobs[from], s).is_empty(),
                "class {from} left nonempty at {s}"
            );
            checked += 1;
        }
    }
    assert!(checked > 1000, "{checked} reconfigurations");
    // It passes through empty configurations and never idles.
    let visits = r["visits"].as_array().unwrap();
    let empty = visits.iter().filter(|v| v["completed"] == 0).count();
    assert!(
        empty * 10 > visits.len(),
        "{empty} of {} visits empty",
        visits.len()
    );
    assert_eq!(r["phase_times"]["idle"], 0.0);
}

/// Completed-job mean sojourn time over all classes and the reconfiguration fraction.
fn sojourn_and_switching(r: &Value) -> (f64, f64) {
    let (mut sum, mut count) = (0.0, 0.0);
    for q in r["per_queue"].as_array().unwrap() {
        let c = q["completed"].as_f64().unwrap();
        sum += q["mean_sojourn"].as_f64().unwrap() * c;
        count += c;
    }
    (
        sum / count,
        r["phase_fractions"]["switch"].as_f64().unwrap(),
    )
}

#[test]
fn ec_matches_symmetric_exhaustive_polling() {
    // Eight classes, service rate 4, total load 0.7, reconfiguration mean 0.5 s. Symmetric
    // exhaustive polling has the mean waiting time (Takagi, Analysis of Polling Systems, 1986)
    // E[W] = delta2 / (2 r) + (N lambda b2 + r (1 - lambda b)) / (2 (1 - rho)), with lambda the
    // arrival rate of a class, r the mean and delta2 the variance of the total reconfiguration
    // time of a cycle, and b, b2 the first two moments of the service time: E[W] = 20/3 s for
    // deterministic and 83/12 s for exponential reconfiguration times. Since the network never
    // idles, it reconfigures a fraction 1 - rho of the time.
    let (n, rho, tau, b) = (8, 0.7, 0.5, 0.25);
    let (nf, lam) = (n as f64, 4.0 * rho / n as f64);
    for (law, variance) in [(det(tau), 0.0), (exp(tau), tau * tau)] {
        let r = nf * tau;
        let wait = nf * variance / (2.0 * r)
            + (nf * lam * 2.0 * b * b + r * (1.0 - lam * b)) / (2.0 * (1.0 - rho));
        let system = symmetric(n, rho, 6.0, law.clone(), [1000.0, 40000.0, 100.0]);
        let runs: Vec<(f64, f64)> = (1..=8u64)
            .map(|seed| sojourn_and_switching(&simulate(&system, "ec", seed)))
            .collect();
        let sojourn = runs.iter().map(|x| x.0).sum::<f64>() / runs.len() as f64;
        let switching = runs.iter().map(|x| x.1).sum::<f64>() / runs.len() as f64;
        assert!(
            (sojourn / (wait + b) - 1.0).abs() < TOL_SOJOURN,
            "{law}: mean sojourn {sojourn} s, exhaustive polling {} s",
            wait + b
        );
        assert!(
            (switching - (1.0 - rho)).abs() < TOL_SWITCH,
            "{law}: reconfiguration fraction {switching}, expected {}",
            1.0 - rho
        );
    }
}

#[test]
fn the_miss_probability_at_n8_load_07_deadline_6_matches_an_independent_simulation() {
    // Class-averaged miss probability with deterministic reconfiguration times of 0.5 s. The
    // reference comes from a simulator written independently of this engine; it pools three runs
    // of 2·10^5 s (standard error 0.11 points).
    let system = symmetric(8, 0.7, 6.0, det(0.5), [2000.0, 20000.0, 100.0]);
    let reference = 52.93;
    let mut miss = 0.0;
    for seed in 1..=SEEDS {
        let r = simulate(&system, "ec", seed);
        for q in r["per_queue"].as_array().unwrap() {
            let late = q["deadline_violations"].as_f64().unwrap();
            miss += late / q["arrivals"].as_f64().unwrap() / (8 * SEEDS) as f64;
        }
    }
    assert!(
        (100.0 * miss - reference).abs() < TOL_MISS,
        "{} % against {reference} %",
        100.0 * miss
    );
}
