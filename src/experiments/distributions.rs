//! `tdf distributions`: the named reconfiguration-time distributions (profiles) as the simulator
//! samples them. For every profile and for the heavy-tailed Pareto distribution of shape 3/2
//! (`pareto_heavy`), the distribution function, the survival function and the integrated tail
//! `E[(tau - t)^+]` on a grid of `t / mean` in `[0, 8]` go to `profiles.csv`; the exact
//! parameters, the exact enclosures of the moments and the simulator's moments go to
//! `profiles.json`.

use super::common::{parse_rational, write_json, write_lines};
use crate::feasibility::distributions::{Distribution, pareto_with_alpha, profile, profile_names};
use crate::feasibility::rational::{Q, parse_q, to_f64};
use crate::sim::laws::Law;
use serde_json::{Value, json};
use std::fs;
use std::path::PathBuf;

/// Tabulate the named reconfiguration-time distributions (profiles).
#[derive(clap::Args, Debug)]
pub struct Args {
    /// Output directory.
    #[arg(long)]
    pub out: PathBuf,
    /// Mean reconfiguration time in seconds.
    #[arg(long, default_value = "1/2", value_parser = parse_rational)]
    pub mean: Q,
    /// Squared coefficient of variation of the reconfiguration time.
    #[arg(long, default_value = "1/3", value_parser = parse_rational)]
    pub cv2: Q,
    /// Number of grid intervals on [0, 8] in units of the mean.
    #[arg(long, default_value_t = 800)]
    pub points: usize,
}

/// Write the tables and return the moments of every profile.
pub fn run(args: &Args) -> Result<Value, String> {
    let (out, mean, cv2, points) = (&args.out, &args.mean, &args.cv2, args.points);
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let mut names: Vec<(String, Distribution)> = profile_names()
        .iter()
        .map(|p| (p.to_string(), profile(p, mean, cv2).unwrap()))
        .collect();
    names.push((
        "pareto_heavy".into(),
        pareto_with_alpha(mean, &parse_q("3/2")?),
    ));
    let m = to_f64(mean);
    let mut lines = vec!["profile,u,t,cdf,ccdf,integrated_tail".to_string()];
    let mut moments = Vec::new();
    for (name, law) in &names {
        let engine = Law::from_json(&law.to_engine_json())?;
        for k in 0..=points {
            let u = 8.0 * k as f64 / points as f64;
            let t = u * m;
            let s = engine.survival(t);
            lines.push(format!(
                "{name},{u},{t},{},{s},{}",
                1.0 - s,
                engine.stop_loss(t)
            ));
        }
        moments.push(json!({"profile": name, "law": law.describe(128), "engine": engine.to_json(), "engine_mean": engine.mean(), "engine_variance": engine.variance(),
                            "cv": engine.variance().sqrt() / engine.mean()}));
    }
    write_lines(&out.join("profiles.csv"), &lines)?;
    let v = json!({"mean": mean.to_string(), "cv2": cv2.to_string(), "profiles": moments});
    write_json(&out.join("profiles.json"), &v)?;
    Ok(v)
}
