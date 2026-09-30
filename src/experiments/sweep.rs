//! `tdf sweep`: a sweep of the common target at one symmetric setting. For every target: the
//! verdict and the certified interval of the best achievable miss probability and, when the
//! target is not excluded, the certified ranges of the total reconfiguration rate and of the rate
//! of reconfigurations into each configuration (entry rate), with OJF's rates as reference and
//! whether OJF's exact long-run vector lies in the relaxation. A target that OJF provably meets
//! while its rates fall outside a range is a consistency failure. Output: `manifest.json`,
//! `sweep.json`, `sweep.csv`.

use super::common::{
    grid_settings, interval_json, level_summary, manifest, opt_f, parse_rational, point_id,
    profile_law, write_csv, write_json,
};
use crate::feasibility::distributions::Model;
use crate::feasibility::rational::{Q, to_f64};
use crate::feasibility::reference::{Homogeneous, bracket};
use crate::feasibility::relaxation::Level;
use crate::feasibility::{Options, evaluate, point_in_relaxation, rate_intervals};
use serde_json::{Value, json};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Sweep the common target.
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
    /// Common targets.
    #[arg(long, value_delimiter = ',', default_value = "1/500,1/400,3/1000,1/250,9/2000,1/200,3/400,1/100,1/50,1/20", value_parser = parse_rational)]
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
            "sweep",
            json!({"n": n, "profile": profile_name, "rho": rho.to_string(), "d": d.to_string(),
                   "epsilons": args.epsilons.iter().map(|q| q.to_string()).collect::<Vec<_>>(), "mean": args.mean.to_string(), "cv2": args.cv2.to_string()}),
            Some(&settings),
        ),
    )?;
    let mut rows = Vec::new();
    let mut csv_rows = Vec::new();
    for eps in &args.epsilons {
        let t = Instant::now();
        let id = point_id(profile_name, n, rho, d, eps);
        let model = Model::balanced(&id, &law, n, rho, d, eps);
        let ev = evaluate(&model, &settings, &[Level::Full], cache)?;
        let full = ev.evaluation.level(Level::Full);
        let h = Homogeneous::from_model(&model)?;
        let excluded = full.category() == "certified_infeasible";
        let br = bracket(
            &h,
            full.min_max.certified_value.as_ref(),
            &full.min_max.status,
            excluded,
            &settings,
        )?;
        let rates = h.reference_rates(settings.bits);
        let reference_meets_target = br.u_plus.as_ref().is_some_and(|u| u <= eps);
        // Membership of OJF's exact long-run vector, with `x_i = eps`, in the relaxation; checked
        // only when the idle fraction is exact.
        let fcfs_point = if rates.z.0 == rates.z.1 {
            let shell = model.lp_shell(settings.bits);
            Some(
                point_in_relaxation(
                    &shell,
                    &ev.lower,
                    &Options::from_grid(&settings),
                    Level::Full,
                    &rates.nu_pair,
                    &rates.z.0,
                    eps,
                )
                .is_ok(),
            )
        } else {
            None
        };
        let intervals: Option<Vec<Value>> = if excluded {
            None
        } else {
            let shell = model.lp_shell(settings.bits);
            let opts = Options::from_grid(&settings);
            let ivs = rate_intervals(&shell, &ev.lower, &opts, Level::Full);
            Some(
                ivs.iter()
                    .enumerate()
                    .map(|(k, iv)| {
                        if k == 0 {
                            interval_json(iv, Some(&rates.nu_total), None)
                        } else {
                            interval_json(iv, Some(&rates.eta), Some(&h.lambda_class()))
                        }
                    })
                    .collect(),
            )
        };
        let reference_inside_all = intervals
            .as_ref()
            .map(|v| v.iter().all(|i| i["contains_reference"] == true));
        let consistency_failure = reference_meets_target && reference_inside_all == Some(false);
        let verdict = match &br.verdict {
            Ok(v) => v.name().to_string(),
            Err(_) => "correctness_failure".into(),
        };
        let total = intervals.as_ref().and_then(|v| v.first().cloned());
        csv_rows.push(vec![
            id.clone(),
            eps.to_string(),
            to_f64(eps).to_string(),
            full.category().to_string(),
            verdict.clone(),
            opt_f(&br.lp_lower),
            to_f64(&br.l_safe).to_string(),
            opt_f(&br.u_plus),
            total
                .as_ref()
                .map(|t| t["lo"]["f64"].to_string())
                .unwrap_or_default(),
            total
                .as_ref()
                .map(|t| t["hi"]["f64"].to_string())
                .unwrap_or_default(),
            total
                .as_ref()
                .map(|t| t["lo_is_lp_extremum"].to_string())
                .unwrap_or_default(),
            total
                .as_ref()
                .map(|t| t["hi_is_lp_extremum"].to_string())
                .unwrap_or_default(),
            to_f64(&rates.nu_total).to_string(),
            to_f64(&rates.eta).to_string(),
            reference_meets_target.to_string(),
            reference_inside_all
                .map(|b| b.to_string())
                .unwrap_or_default(),
        ]);
        rows.push(json!({
            "id": id, "eps": eps.to_string(), "eps_f64": to_f64(eps), "full": level_summary(full), "excluded": excluded, "verdict": verdict,
            "bracket": br.to_json(), "reference_rates": rates.to_json(), "reference_meets_target": reference_meets_target,
            "fcfs_rates_point_in_relaxation": fcfs_point,
            "intervals": intervals, "reference_inside_all_intervals": reference_inside_all, "consistency_failure": consistency_failure,
            "seconds": t.elapsed().as_secs_f64()
        }));
    }
    write_csv(
        &out.join("sweep.csv"),
        &[
            "id",
            "eps",
            "eps_f64",
            "full_category",
            "verdict",
            "l_lp",
            "l_safe",
            "u_plus",
            "nu_lo",
            "nu_hi",
            "nu_lo_exact",
            "nu_hi_exact",
            "nu_reference",
            "eta_reference",
            "reference_meets_target",
            "reference_inside_all",
        ],
        &csv_rows,
    )?;
    let failures = rows
        .iter()
        .filter(|r| r["consistency_failure"] == true || r["verdict"] == "correctness_failure")
        .count();
    let summary = json!({"targets": rows.len(), "excluded": rows.iter().filter(|r| r["excluded"] == true).count(),
        "certified_feasible": rows.iter().filter(|r| r["verdict"] == "certified_feasible").count(), "failures": failures,
        "total_seconds": t0.elapsed().as_secs_f64()});
    write_json(
        &out.join("sweep.json"),
        &json!({"summary": summary, "rows": rows}),
    )?;
    Ok(summary)
}
