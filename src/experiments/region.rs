//! `tdf region`: the operating region of one symmetric setting in the target-rate plane. For
//! every common target of a fine grid: whether the target is excluded and, if not, the certified
//! range `[nu_min, nu_max]` that contains the total reconfiguration rate of any policy meeting
//! it. Also written: the certified interval of the best achievable miss probability at the
//! smallest target and OJF's rates. Output: `manifest.json`, `region.json`, `region.csv`.

use super::common::{
    grid_settings, manifest, parse_rational, point_id, profile_law, write_csv, write_json,
};
use crate::feasibility::distributions::Model;
use crate::feasibility::rational::{Q, to_f64};
use crate::feasibility::reference::{Homogeneous, bracket};
use crate::feasibility::relaxation::Level;
use crate::feasibility::{Options, evaluate, rate_intervals};
use serde_json::{Value, json};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Certify the range of the total reconfiguration rate over a fine grid of targets.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// Output directory.
    #[arg(long)]
    pub out: PathBuf,
    /// Number of classes.
    #[arg(long, default_value_t = 2)]
    pub n: usize,
    /// Reconfiguration-time profile (det, exp, gamma, tnorm, lognormal, pareto, pareto_heavy).
    #[arg(long, default_value = "exp")]
    pub profile: String,
    /// Total load.
    #[arg(long, default_value = "1/20", value_parser = parse_rational)]
    pub rho: Q,
    /// Deadline in seconds.
    #[arg(long, default_value = "3", value_parser = parse_rational)]
    pub d: Q,
    /// Common targets (evaluated in increasing order).
    #[arg(long, value_delimiter = ',', default_value = "1/500,1/400,7/2500,3/1000,13/4000,7/2000,1/250,9/2000,1/200,11/2000,3/500,13/2000,7/1000,3/400,1/125,9/1000,1/100,3/250,7/500,2/125,1/50,1/40,3/100,1/25,1/20,3/40,1/10", value_parser = parse_rational)]
    pub epsilons: Vec<Q>,
    /// Mean reconfiguration time in seconds.
    #[arg(long, default_value = "1/2", value_parser = parse_rational)]
    pub mean: Q,
    /// Squared coefficient of variation of the reconfiguration time.
    #[arg(long, default_value = "1/3", value_parser = parse_rational)]
    pub cv2: Q,
    /// Grid step in seconds (default 1/2000 up to 4 classes, 1/1000 up to 8, 1/500 beyond).
    #[arg(long)]
    pub step: Option<String>,
    /// Refinement sweeps of the upper bound on the completion probability (default 3).
    #[arg(long)]
    pub sweeps: Option<usize>,
    /// Fractional bits of the rational enclosures (default 128).
    #[arg(long)]
    pub bits: Option<u32>,
    /// Directory of cached phase coefficients, read when present and written otherwise
    /// (default: no cache).
    #[arg(long)]
    pub cache: Option<PathBuf>,
    /// Worker threads (default: all cores).
    #[arg(long)]
    pub threads: Option<usize>,
}

/// Evaluate every target and write the tables; returns a summary.
pub fn run(args: &Args) -> Result<Value, String> {
    let out = &args.out;
    let (n, profile_name, rho, d) = (args.n, args.profile.as_str(), &args.rho, &args.d);
    let cache = args.cache.as_deref();
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    let law = profile_law(profile_name, &args.mean, &args.cv2)?;
    let settings = grid_settings(n, args.step.as_deref(), args.sweeps, args.bits, false)?;
    write_json(
        &out.join("manifest.json"),
        &manifest(
            "region",
            json!({"n": n, "profile": profile_name, "rho": rho.to_string(), "d": d.to_string(),
                   "epsilons": args.epsilons.iter().map(|q| q.to_string()).collect::<Vec<_>>(),
                   "mean": args.mean.to_string(), "cv2": args.cv2.to_string()}),
            Some(&settings),
        ),
    )?;
    let mut sorted: Vec<Q> = args.epsilons.to_vec();
    sorted.sort();
    let mut region = Vec::new();
    let mut h_opt: Option<Homogeneous> = None;
    let mut bracket_json = Value::Null;
    for eps in &sorted {
        let id = point_id(profile_name, n, rho, d, eps);
        let model = Model::balanced(&id, &law, n, rho, d, eps);
        let ev = evaluate(&model, &settings, &[Level::Full], cache)?;
        let full = ev.evaluation.level(Level::Full);
        let h = Homogeneous::from_model(&model)?;
        let excluded = full.category() == "certified_infeasible";
        if h_opt.is_none() {
            let br = bracket(
                &h,
                full.min_max.certified_value.as_ref(),
                &full.min_max.status,
                excluded,
                &settings,
            )?;
            bracket_json = br.to_json();
            h_opt = Some(h.clone());
        }
        let (lo, hi, exact) = if excluded {
            (None, None, false)
        } else {
            let shell = model.lp_shell(settings.bits);
            let ivs = rate_intervals(
                &shell,
                &ev.lower,
                &Options::from_grid(&settings),
                Level::Full,
            );
            let iv = &ivs[0];
            (
                iv.lo(),
                iv.hi(),
                iv.lower.exact_optimum && iv.upper.exact_optimum,
            )
        };
        region.push(json!({
            "eps": eps.to_string(), "eps_f64": to_f64(eps), "excluded": excluded,
            "nu_lo": lo.as_ref().map(to_f64), "nu_hi": hi.as_ref().map(to_f64), "endpoints_exact": exact,
            "nu_lo_exact": lo.map(|v| v.to_string()), "nu_hi_exact": hi.map(|v| v.to_string())
        }));
    }
    let h = h_opt.ok_or("no target")?;
    let rates = h.reference_rates(settings.bits);
    let region_csv: Vec<Vec<String>> = region
        .iter()
        .map(|r| {
            vec![
                r["eps"].as_str().unwrap().into(),
                r["eps_f64"].to_string(),
                r["excluded"].to_string(),
                r["nu_lo"].to_string(),
                r["nu_hi"].to_string(),
                r["endpoints_exact"].to_string(),
            ]
        })
        .collect();
    write_csv(
        &out.join("region.csv"),
        &[
            "eps",
            "eps_f64",
            "excluded",
            "nu_lo",
            "nu_hi",
            "endpoints_exact",
        ],
        &region_csv,
    )?;
    let summary = json!({"targets": region.len(), "total_seconds": t0.elapsed().as_secs_f64()});
    write_json(
        &out.join("region.json"),
        &json!({"summary": summary, "setting": h.to_json(), "bracket": bracket_json,
        "reference_rates": rates.to_json(), "region": region}),
    )?;
    Ok(summary)
}
