//! `tdf bracket`: the certified interval `L_safe <= m* <= U+` of the best achievable miss
//! probability `m*` in symmetric settings (profiles x numbers of classes x loads x deadlines, one
//! common target), with the verdict at the target. `L_safe` is the larger of the relaxation's
//! bound `L_LP` and the no-waiting miss probability `m_0(d)` (`floor`, columns `b_lo`, `b_hi`);
//! `U+` is the smallest certified upper bound on the miss probability of OJF among the baseline
//! bound `U_0` (`u_simple`), the exact value of the exponential benchmark where it applies, and
//! the workload bound. Output: `manifest.json`, `bracket.json`, `bracket.csv`.

use super::common::{
    grid_settings, level_summary, manifest, opt_f, parse_rational, point_id, profile_law,
    write_csv, write_json,
};
use crate::feasibility::distributions::Model;
use crate::feasibility::evaluate;
use crate::feasibility::rational::{Q, to_f64};
use crate::feasibility::reference::{Bracket, Homogeneous, bracket};
use crate::feasibility::relaxation::Level;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Certify the interval of the best achievable miss probability in symmetric settings.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// Output directory.
    #[arg(long)]
    pub out: PathBuf,
    /// Reconfiguration-time profiles (det, exp, gamma, tnorm, lognormal,
    /// pareto, pareto_heavy).
    #[arg(long, value_delimiter = ',', default_value = "exp")]
    pub profiles: Vec<String>,
    /// Numbers of classes.
    #[arg(long, value_delimiter = ',', default_value = "2")]
    pub ns: Vec<usize>,
    /// Total loads.
    #[arg(long, value_delimiter = ',', default_value = "1/10000,1/1000,1/100,3/100,1/20,1/10,1/5,3/10,2/5,9/20", value_parser = parse_rational)]
    pub rhos: Vec<Q>,
    /// Deadlines in seconds.
    #[arg(long, value_delimiter = ',', default_value = "2,3", value_parser = parse_rational)]
    pub ds: Vec<Q>,
    /// Common target miss probability.
    #[arg(long, default_value = "1/100", value_parser = parse_rational)]
    pub eps: Q,
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

fn bracket_csv_fields(br: &Bracket) -> Vec<String> {
    let iv = |e: &(Q, Q)| vec![to_f64(&e.0).to_string(), to_f64(&e.1).to_string()];
    let mut f = vec![
        opt_f(&br.lp_lower),
        br.lp_status.clone(),
        br.lp_infeasible_at_target.to_string(),
    ];
    f.extend(iv(&br.floor));
    f.extend(
        br.tail
            .closed_form
            .as_ref()
            .map(&iv)
            .unwrap_or_else(|| vec![String::new(), String::new()]),
    );
    f.extend(iv(&br.tail.grid));
    f.extend(iv(&br.rho_f));
    f.push(opt_f(&br.u_simple));
    f.extend(
        br.benchmark
            .as_ref()
            .map(|b| iv(&b.miss))
            .unwrap_or_else(|| vec![String::new(), String::new()]),
    );
    f.push(
        br.workload
            .as_ref()
            .map(|w| to_f64(&w.u_k_plus).to_string())
            .unwrap_or_default(),
    );
    f.push(
        br.workload
            .as_ref()
            .map(|w| w.k.to_string())
            .unwrap_or_default(),
    );
    f.push(
        br.workload
            .as_ref()
            .map(|w| to_f64(&w.truncation).to_string())
            .unwrap_or_default(),
    );
    f.push(opt_f(&br.u_plus));
    f.push(to_f64(&br.l_safe).to_string());
    f.push(opt_f(&br.gap_safe));
    f.push(opt_f(&br.gap_raw));
    f.push(match &br.verdict {
        Ok(v) => v.name().to_string(),
        Err(_) => "correctness_failure".into(),
    });
    f
}

const BRACKET_CSV: [&str; 28] = [
    "l_lp",
    "l_lp_status",
    "lp_infeasible_at_target",
    "b_lo",
    "b_hi",
    "tail_closed_lo",
    "tail_closed_hi",
    "tail_grid_lo",
    "tail_grid_hi",
    "rho_f_lo",
    "rho_f_hi",
    "u_simple",
    "u_exact_lo",
    "u_exact_hi",
    "u_workload",
    "workload_k",
    "workload_truncation",
    "u_plus",
    "l_safe",
    "gap_safe",
    "gap_raw",
    "verdict",
    "seconds",
    "max_b_width",
    "cells",
    "workload_lower",
    "workload_numeric_width",
    "workload_y_cdf_width",
];

/// Compute every interval (in parallel over the settings) and write the
/// tables; returns the counts of the verdicts.
pub fn run(args: &Args) -> Result<Value, String> {
    let out = &args.out;
    let (eps, mean, cv2) = (&args.eps, &args.mean, &args.cv2);
    let (step, sweeps, bits, cache) = (
        args.step.as_deref(),
        args.sweeps,
        args.bits,
        args.cache.as_deref(),
    );
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    let mut cases = Vec::new();
    for p in &args.profiles {
        for &n in &args.ns {
            for rho in &args.rhos {
                for d in &args.ds {
                    cases.push((p.clone(), n, rho.clone(), d.clone()));
                }
            }
        }
    }
    write_json(
        &out.join("manifest.json"),
        &manifest(
            "bracket",
            json!({"profiles": args.profiles, "ns": args.ns, "rhos": args.rhos.iter().map(|q| q.to_string()).collect::<Vec<_>>(),
                   "ds": args.ds.iter().map(|q| q.to_string()).collect::<Vec<_>>(), "eps": eps.to_string(), "mean": mean.to_string(), "cv2": cv2.to_string(),
                   "step": step, "sweeps": sweeps, "bits": bits, "cache": cache.map(|c| c.display().to_string()), "cases": cases.len()}),
            None,
        ),
    )?;
    let rows: Vec<Result<Value, String>> = cases
        .par_iter()
        .map(|(p, n, rho, d)| {
            let t = Instant::now();
            let law = profile_law(p, mean, cv2)?;
            let id = point_id(p, *n, rho, d, eps);
            let model = Model::balanced(&id, &law, *n, rho, d, eps);
            let settings = grid_settings(*n, step, sweeps, bits, false)?;
            let ev = evaluate(&model, &settings, &Level::all(), cache)?;
            let full = ev.evaluation.level(Level::Full);
            let h = Homogeneous::from_model(&model)?;
            let br = bracket(
                &h,
                full.min_max.certified_value.as_ref(),
                &full.min_max.status,
                full.category() == "certified_infeasible",
                &settings,
            )?;
            let unstable = h.exact_exponential_benchmark(settings.bits).err();
            let targets = ev.diagnostics["targets"].as_array().cloned().unwrap_or_default();
            let maxf = |key: &str| targets.iter().filter_map(|t| t[key].as_f64()).fold(0.0, f64::max);
            let seconds = t.elapsed().as_secs_f64();
            let mut csv = vec![
                id.clone(),
                p.clone(),
                n.to_string(),
                rho.to_string(),
                to_f64(rho).to_string(),
                d.to_string(),
                eps.to_string(),
                full.category().to_string(),
            ];
            csv.extend(bracket_csv_fields(&br));
            csv.push(seconds.to_string());
            csv.push(maxf("max_b_width").to_string());
            csv.push(ev.diagnostics["cells"].to_string());
            // The workload enclosure of the miss probability of OJF: its certified lower end
            // `U_K^-`, its numerical width `C_K^+ - C_K^-` and the largest width of the enclosed
            // distribution function of the effective service time.
            csv.push(br.workload.as_ref().map(|w| to_f64(&w.u_k_minus).to_string()).unwrap_or_default());
            csv.push(br.workload.as_ref().map(|w| to_f64(&w.numeric_width).to_string()).unwrap_or_default());
            csv.push(br.workload.as_ref().map(|w| w.y_cdf_width.to_string()).unwrap_or_default());
            Ok(json!({
                "id": id, "profile": p, "n": n, "rho": rho.to_string(), "rho_f64": to_f64(rho), "d": d.to_string(), "d_f64": to_f64(d),
                "eps": eps.to_string(), "law": law.id(), "settings": settings.to_json(), "cache_hit": ev.cache_hit,
                "levels": ev.evaluation.levels.iter().map(level_summary).collect::<Vec<_>>(),
                "bracket": br.to_json(), "benchmark_unstable": unstable,
                "grid": {"step": ev.diagnostics["step"], "cells": ev.diagnostics["cells"], "max_h_width": maxf("max_h_width"),
                         "max_a_width": maxf("max_a_width"), "max_b_width": maxf("max_b_width"), "max_c_width": maxf("max_c_width")},
                "seconds": seconds, "csv": csv
            }))
        })
        .collect();
    let mut ok = Vec::new();
    let mut failed = Vec::new();
    for (case, r) in cases.iter().zip(rows) {
        match r {
            Ok(v) => ok.push(v),
            Err(e) => failed.push(json!({"case": format!("{case:?}"), "error": e})),
        }
    }
    let mut header: Vec<&str> = vec![
        "id",
        "profile",
        "n",
        "rho",
        "rho_f64",
        "d",
        "eps",
        "full_category",
    ];
    header.extend(BRACKET_CSV.iter());
    let csv_rows: Vec<Vec<String>> = ok
        .iter()
        .map(|r| {
            r["csv"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect()
        })
        .collect();
    write_csv(&out.join("bracket.csv"), &header, &csv_rows)?;
    let verdicts = |name: &str| {
        ok.iter()
            .filter(|r| r["bracket"]["verdict"] == name)
            .count()
    };
    let failures = ok
        .iter()
        .filter(|r| r["bracket"]["verdict"].get("error").is_some())
        .count();
    let summary = json!({
        "cases": cases.len(), "computed": ok.len(), "errors": failed.len(),
        "certified_infeasible": verdicts("certified_infeasible"), "certified_feasible": verdicts("certified_feasible"),
        "undetermined": verdicts("undetermined"), "correctness_failures": failures,
        "total_seconds": t0.elapsed().as_secs_f64()
    });
    write_json(
        &out.join("bracket.json"),
        &json!({"summary": summary, "rows": ok, "failed": failed}),
    )?;
    Ok(summary)
}
