//! `tdf directional`: three classes with `mu = (4, 5, 6)`, load shares `a = (1/2, 1/3, 1/6)`,
//! arrival rates `lambda_i = rho a_i mu_i`, deadlines `h (2, 5/2, 3)`, targets
//! `(1/200, 1/100, 1/50)`, and gamma reconfiguration times of shape 3 whose means form the matrix
//! `[[-, 1/4, 1], [1, -, 1/4], [1/4, 1, -]]` (rows: source) or its transpose. Per orientation, load
//! `rho` and deadline scale `h`: the verdict, constraint levels, certificate families, class
//! bounds, target scale bound `kappa_L` and rate ranges; the excluded case `rho = 1/10`, `h = 3/4`
//! again with a faster first class and with shorter reconfiguration times (`interventions.csv`).
//! Further output: `manifest.json`, `directional.json`, `directional.csv`.

use super::common::{
    grid_settings, interval_json, level_summary, manifest, parse_rational, witness_families,
    write_csv, write_json,
};
use crate::feasibility::distributions::{Distribution, Model};
use crate::feasibility::rational::{Q, parse_q, qi, qr, to_f64};
use crate::feasibility::relaxation::Level;
use crate::feasibility::{Options, class_bounds, evaluate, rate_intervals, scale_bound};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Evaluate the three-class case with direction-dependent reconfiguration times.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// Output directory.
    #[arg(long)]
    pub out: PathBuf,
    /// Total loads `rho` (arrival rates `rho a_i mu_i`).
    #[arg(long, value_delimiter = ',', default_value = "1/20,1/10,1/5", value_parser = parse_rational)]
    pub rhos: Vec<Q>,
    /// Deadline scales `h` (deadlines `h (2, 5/2, 3)` seconds).
    #[arg(long, value_delimiter = ',', default_value = "1/2,3/4,1,5/4,3/2,2", value_parser = parse_rational)]
    pub scales: Vec<Q>,
    /// Grid step in seconds (default 1/2000).
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

/// Evaluate every case and the changes of the excluded case, and write the
/// tables; returns a summary.
pub fn run(args: &Args) -> Result<Value, String> {
    let out = &args.out;
    let (rhos, scales) = (&args.rhos, &args.scales);
    let (step, sweeps, bits, cache) = (
        args.step.as_deref(),
        args.sweeps,
        args.bits,
        args.cache.as_deref(),
    );
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    let n = 3usize;
    let mu: Vec<Q> = [4, 5, 6].iter().map(|m| qi(*m)).collect();
    let shares: Vec<Q> = vec![qr(1, 2), qr(1, 3), qr(1, 6)];
    let d0: Vec<Q> = vec![qi(2), qr(5, 2), qi(3)];
    let eps: Vec<Q> = vec![qr(1, 200), qr(1, 100), qr(1, 50)];
    // Mean reconfiguration times: rows are the source, columns the destination configuration.
    let m: Vec<Vec<Option<Q>>> = vec![
        vec![None, Some(qr(1, 4)), Some(qi(1))],
        vec![Some(qi(1)), None, Some(qr(1, 4))],
        vec![Some(qr(1, 4)), Some(qi(1)), None],
    ];
    let transpose: Vec<Vec<Option<Q>>> = (0..n)
        .map(|i| (0..n).map(|j| m[j][i].clone()).collect())
        .collect();
    let settings = grid_settings(n, step, sweeps, bits, false)?;
    write_json(
        &out.join("manifest.json"),
        &manifest(
            "directional",
            json!({"n": n, "mu": ["4","5","6"], "shares": ["1/2","1/3","1/6"], "base_deadlines": ["2","5/2","3"], "eps": ["1/200","1/100","1/50"],
                   "law": "gamma shape 3, rate 3/m_ij", "mean_matrix_rows_from": [[null,"1/4","1"],["1",null,"1/4"],["1/4","1",null]],
                   "rhos": rhos.iter().map(|q| q.to_string()).collect::<Vec<_>>(), "scales": scales.iter().map(|q| q.to_string()).collect::<Vec<_>>()}),
            Some(&settings),
        ),
    )?;
    let mut cases: Vec<(String, Q, Q)> = Vec::new();
    for orientation in ["matrix", "transpose"] {
        for rho in rhos {
            for h in scales {
                cases.push((orientation.into(), rho.clone(), h.clone()));
            }
        }
    }
    let rows: Vec<Result<Value, String>> = cases
        .par_iter()
        .map(|(orientation, rho, h)| {
            let t = Instant::now();
            let means = if orientation == "matrix" { &m } else { &transpose };
            let lam: Vec<Q> = (0..n).map(|i| rho * &shares[i] * &mu[i]).collect();
            let d: Vec<Q> = d0.iter().map(|x| x * h).collect();
            let laws: Vec<Vec<Option<Distribution>>> = (0..n)
                .map(|u| {
                    (0..n)
                        .map(|v| {
                            means[u][v].as_ref().map(|mean| Distribution::Gamma {
                                shape: qi(3),
                                rate: qi(3) / mean,
                            })
                        })
                        .collect()
                })
                .collect();
            let id = format!("directional|{orientation}|rho={rho}|h={h}");
            let model = Model::new(&id, lam.clone(), mu.clone(), d.clone(), eps.clone(), laws)?;
            let ev = evaluate(&model, &settings, &Level::all(), cache)?;
            let full = ev.evaluation.level(Level::Full);
            let shell = model.lp_shell(settings.bits);
            let opts = Options::from_grid(&settings);
            let class = class_bounds(&shell, &ev.lower, &opts, Level::Full);
            let kappa = scale_bound(&shell, &ev.lower, &opts, Level::Full);
            let excluded = full.category() == "certified_infeasible";
            let intervals: Option<Vec<Value>> = (!excluded).then(|| {
                rate_intervals(&shell, &ev.lower, &opts, Level::Full)
                    .iter()
                    .enumerate()
                    .map(|(k, iv)| interval_json(iv, None, (k > 0).then(|| &lam[k - 1])))
                    .collect()
            });
            let class_excluded: Vec<bool> = class.iter().zip(&eps).map(|(b, e)| b.is_some_and(|b| b > to_f64(e))).collect();
            Ok(json!({
                "id": id, "orientation": orientation, "rho": rho.to_string(), "rho_f64": to_f64(rho), "h": h.to_string(), "h_f64": to_f64(h),
                "lambda": lam.iter().map(|q| q.to_string()).collect::<Vec<_>>(), "deadlines": d.iter().map(|q| q.to_string()).collect::<Vec<_>>(),
                "eps": eps.iter().map(|q| q.to_string()).collect::<Vec<_>>(),
                "levels": ev.evaluation.levels.iter().map(level_summary).collect::<Vec<_>>(),
                "full_category": full.category(), "lb_max": full.lower_bound_max(), "witness_families": witness_families(&full.feasibility),
                "class_bounds": class, "class_bound_exceeds_target": class_excluded,
                "kappa_lower": kappa.certified_value.as_ref().map(to_f64), "kappa_status": kappa.status, "kappa_exact": kappa.exact_optimum,
                "intervals": intervals, "grid": {"step": ev.diagnostics["step"], "cells": ev.diagnostics["cells"]}, "seconds": t.elapsed().as_secs_f64()
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
    // Changes of the excluded case rho = 1/10, h = 3/4 in both orientations, evaluated when that
    // case is part of the run: the service rate of the first class times 5/4, 3/2 and 2 at fixed
    // arrival rates, and every reconfiguration time times 4/5, 3/5 and 2/5.
    let base_rho = qr(1, 10);
    let base_h = qr(3, 4);
    let mut interventions: Vec<Value> = Vec::new();
    if rhos.contains(&base_rho) && scales.contains(&base_h) {
        let mut variants: Vec<(String, String, Model)> = Vec::new();
        for orientation in ["matrix", "transpose"] {
            let means = if orientation == "matrix" {
                &m
            } else {
                &transpose
            };
            let lam: Vec<Q> = (0..n).map(|i| &base_rho * &shares[i] * &mu[i]).collect();
            let d: Vec<Q> = d0.iter().map(|x| x * &base_h).collect();
            let laws: Vec<Vec<Option<Distribution>>> = (0..n)
                .map(|u| {
                    (0..n)
                        .map(|v| {
                            means[u][v].as_ref().map(|mean| Distribution::Gamma {
                                shape: qi(3),
                                rate: qi(3) / mean,
                            })
                        })
                        .collect()
                })
                .collect();
            let base = Model::new(
                &format!("directional|{orientation}|rho=1/10|h=3/4"),
                lam,
                mu.clone(),
                d,
                eps.clone(),
                laws,
            )?;
            variants.push((orientation.into(), "base".into(), base.clone()));
            for k in ["5/4", "3/2", "2"] {
                let kq = parse_q(k)?;
                let mut v = base.clone();
                v.mu[0] = &base.mu[0] * &kq;
                v.name = format!("{}|service[1] x {k}", base.name);
                variants.push((orientation.into(), format!("service_class1_x{k}"), v));
            }
            for k in ["4/5", "3/5", "2/5"] {
                let kq = parse_q(k)?;
                variants.push((
                    orientation.into(),
                    format!("reconfiguration_x{k}"),
                    base.with_reconfiguration_scaled(&kq),
                ));
            }
        }
        let results: Vec<Result<Value, String>> = variants
            .par_iter()
            .map(|(orientation, variant, model)| {
                let t = Instant::now();
                let ev = evaluate(model, &settings, &[Level::Full], cache)?;
                let full = ev.evaluation.level(Level::Full);
                let shell = model.lp_shell(settings.bits);
                let opts = Options::from_grid(&settings);
                let class = class_bounds(&shell, &ev.lower, &opts, Level::Full);
                let kappa = scale_bound(&shell, &ev.lower, &opts, Level::Full);
                let excluded = full.category() == "certified_infeasible";
                let intervals: Option<Vec<Value>> = (!excluded).then(|| {
                    rate_intervals(&shell, &ev.lower, &opts, Level::Full)
                        .iter()
                        .enumerate()
                        .map(|(k, iv)| interval_json(iv, None, (k > 0).then(|| &model.lam[k - 1])))
                        .collect()
                });
                Ok(json!({
                    "orientation": orientation, "variant": variant, "model": model.name,
                    "mu": model.mu.iter().map(|q| q.to_string()).collect::<Vec<_>>(),
                    "mean_matrix": (0..n).map(|u| (0..n).map(|v| if u == v { Value::Null } else { json!(to_f64(&model.law(u, v).mean(settings.bits).0)) }).collect::<Vec<_>>()).collect::<Vec<_>>(),
                    "full_category": full.category(), "lb_max": full.lower_bound_max(), "witness_families": witness_families(&full.feasibility),
                    "class_bounds": class, "kappa_lower": kappa.certified_value.as_ref().map(to_f64), "kappa_exact": kappa.exact_optimum,
                    "intervals": intervals, "seconds": t.elapsed().as_secs_f64()
                }))
            })
            .collect();
        for r in results {
            match r {
                Ok(v) => interventions.push(v),
                Err(e) => failed.push(json!({"case": "intervention", "error": e})),
            }
        }
    }
    // Verdicts of the two orientations per (rho, h).
    let mut comparison = Vec::new();
    for rho in rhos {
        for h in scales {
            let find = |o: &str| {
                ok.iter().find(|r| {
                    r["orientation"] == o && r["rho"] == rho.to_string() && r["h"] == h.to_string()
                })
            };
            if let (Some(a), Some(b)) = (find("matrix"), find("transpose")) {
                comparison.push(json!({"rho": rho.to_string(), "h": h.to_string(), "matrix": a["full_category"], "transpose": b["full_category"],
                    "verdicts_differ": a["full_category"] != b["full_category"], "lb_max": [a["lb_max"], b["lb_max"]]}));
            }
        }
    }
    let csv_rows: Vec<Vec<String>> = ok
        .iter()
        .map(|r| {
            let cb = r["class_bounds"].as_array().unwrap();
            vec![
                r["id"].as_str().unwrap().into(),
                r["orientation"].as_str().unwrap().into(),
                r["rho"].as_str().unwrap().into(),
                r["h"].as_str().unwrap().into(),
                r["full_category"].as_str().unwrap().into(),
                r["lb_max"].to_string(),
                r["kappa_lower"].to_string(),
                r["witness_families"].to_string().replace(',', ";"),
                cb[0].to_string(),
                cb[1].to_string(),
                cb[2].to_string(),
                r["intervals"]
                    .as_array()
                    .and_then(|v| v.first())
                    .map(|t| t["lo"]["f64"].to_string())
                    .unwrap_or_default(),
                r["intervals"]
                    .as_array()
                    .and_then(|v| v.first())
                    .map(|t| t["hi"]["f64"].to_string())
                    .unwrap_or_default(),
            ]
        })
        .collect();
    write_csv(
        &out.join("directional.csv"),
        &[
            "id",
            "orientation",
            "rho",
            "h",
            "full_category",
            "lb_max",
            "kappa_lower",
            "witness_families",
            "class_bound_0",
            "class_bound_1",
            "class_bound_2",
            "nu_lo",
            "nu_hi",
        ],
        &csv_rows,
    )?;
    let iv_rows: Vec<Vec<String>> = interventions
        .iter()
        .map(|r| {
            let cb = r["class_bounds"].as_array().unwrap();
            vec![
                r["orientation"].as_str().unwrap().into(),
                r["variant"].as_str().unwrap().into(),
                r["full_category"].as_str().unwrap().into(),
                r["lb_max"].to_string(),
                r["kappa_lower"].to_string(),
                r["kappa_exact"].to_string(),
                cb[0].to_string(),
                cb[1].to_string(),
                cb[2].to_string(),
            ]
        })
        .collect();
    write_csv(
        &out.join("interventions.csv"),
        &[
            "orientation",
            "variant",
            "full_category",
            "lb_max",
            "kappa_lower",
            "kappa_exact",
            "class_bound_0",
            "class_bound_1",
            "class_bound_2",
        ],
        &iv_rows,
    )?;
    let differing = comparison
        .iter()
        .filter(|c| c["verdicts_differ"] == true)
        .count();
    let summary = json!({"cases": cases.len(), "computed": ok.len(), "errors": failed.len(),
        "excluded": ok.iter().filter(|r| r["full_category"] == "certified_infeasible").count(),
        "orientation_pairs": comparison.len(), "pairs_with_different_verdicts": differing, "total_seconds": t0.elapsed().as_secs_f64()});
    write_json(
        &out.join("directional.json"),
        &json!({"summary": summary, "rows": ok, "failed": failed, "comparison": comparison, "interventions": interventions}),
    )?;
    Ok(summary)
}
