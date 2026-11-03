//! Tests of the simulation engine: a single loaded class against the M/M/1 mean sojourn time
//! `1 / (mu - lambda)`, conservation of jobs and time in the run record under EC and OJF, and
//! the rejection of invalid run specifications (`job` in identifiers) as aborted runs.

use serde_json::{Value, json};
use tdf::sim::runner::run_job;

fn system(lambda: [f64; 2]) -> Value {
    json!({"mu": [4.0, 4.0], "lambda": lambda, "deadlines": [100.0, 100.0], "epsilon": [0.01, 0.01],
           "switch": {"default": {"family": "deterministic", "value": 1.0}}, "warmup": 500.0, "measure": 5000.0, "followup": 200.0,
           "max_events": Value::Null, "max_jobs": Value::Null, "max_wall_seconds": Value::Null})
}

#[test]
fn single_loaded_class_is_an_m_m_1_queue_and_never_reconfigures() {
    let mut means = Vec::new();
    for seed in 1..=3u64 {
        let job = json!({"system": system([2.0, 0.0]), "policy": {"name": "ojf", "params": {}}, "seed": seed});
        let result = run_job(&job);
        assert_eq!(result["status"], "completed", "{}", result["abort_reason"]);
        assert_eq!(result["conservation"]["arrival_balance"], 0);
        assert_eq!(result["conservation"]["deadline_count_consistent"], true);
        assert_eq!(result["switches"].as_array().unwrap().len(), 0);
        assert_eq!(result["per_queue"][1]["active"], false);
        means.push(result["per_queue"][0]["mean_sojourn"].as_f64().unwrap());
    }
    let mean = means.iter().sum::<f64>() / means.len() as f64;
    assert!(
        (mean - 0.5).abs() < 0.04,
        "M/M/1 mean sojourn 1/(mu - lambda) = 0.5, observed {mean}"
    );
}

#[test]
fn every_policy_conserves_jobs_and_time() {
    for policy in [
        json!({"name": "ec", "params": {}}),
        json!({"name": "ojf", "params": {}}),
    ] {
        // Total arrival rate 0.5: stable even if every job needs a reconfiguration of length 1.
        let job = json!({"system": system([0.3, 0.2]), "policy": policy, "seed": 11});
        let r = run_job(&job);
        assert_eq!(r["status"], "completed", "{policy}: {}", r["abort_reason"]);
        let c = &r["conservation"];
        assert_eq!(c["arrival_balance"], 0, "{policy}");
        assert_eq!(c["deadline_count_consistent"], true, "{policy}");
        assert!(
            c["time_error"].as_f64().unwrap().abs() < 1e-6,
            "{policy}: {c}"
        );
        let phases = &r["phase_fractions"];
        let total: f64 = ["service", "switch", "idle"]
            .iter()
            .map(|k| phases[*k].as_f64().unwrap())
            .sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "{policy}: phase fractions sum to {total}"
        );
        // Both classes are served, so the network reconfigures.
        assert!(!r["switches"].as_array().unwrap().is_empty(), "{policy}");
    }
}

#[test]
fn invalid_jobs_are_reported_as_aborted_runs() {
    let aborted = |job: Value| {
        let r = run_job(&job);
        assert_eq!(r["status"], "aborted", "{job}");
        r["abort_reason"].as_str().unwrap().to_string()
    };
    assert!(
        aborted(json!({"system": system([1.0, 1.0]), "policy": {"name": "nonsense"}, "seed": 1}))
            .contains("unknown policy")
    );
    assert!(aborted(json!({"system": system([1.0, 1.0]), "policy": {"name": "ec", "params": {"k": 2}}, "seed": 1})).contains("unknown parameters"));
    let mut bad = system([1.0, 1.0]);
    bad["initial"] = json!([[], []]);
    assert!(
        aborted(json!({"system": bad, "policy": {"name": "ec", "params": {}}, "seed": 1}))
            .contains("unknown system key")
    );
}
