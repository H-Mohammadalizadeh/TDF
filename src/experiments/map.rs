//! `tdf map`: the evaluation grid of symmetric settings (profiles x mean reconfiguration times x
//! numbers of classes x loads x deadlines x targets), each certified at the four constraint
//! levels. For the fixed service rate, the phase coefficients depend only on the profile, the
//! mean, the number of classes and the deadline, so the settings are processed in groups of
//! these, in parallel over the groups; with `--cache`, a group's coefficients are computed once
//! and read back for its other loads and targets. Deadlines given as multiples of the mean
//! intervisit time (`--d-over-i`) depend on the load, so such a group shares coefficients only
//! across targets. Every setting's certificate goes to `certificates/`, the table to `map.csv`
//! and `map.json`, the counts to `summary.json`. With the defaults (one mean, 1/2) the
//! identifiers, tables and records are those of the grid without these options; the columns
//! `mean` and `d_over_i` are added only with other means or `--d-over-i`.

use super::common::{
    file_stem, grid_settings, parse_rational, point_id, profile_law, write_json, write_lines,
};
use crate::feasibility::Evaluation;
use crate::feasibility::distributions::{Distribution, Model};
use crate::feasibility::evaluate;
use crate::feasibility::rational::{Q, qi, qr, to_f64};
use crate::feasibility::relaxation::Level;
use num_traits::One;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Certify every setting of the grid.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// Output directory.
    #[arg(long)]
    pub out: PathBuf,
    /// Reconfiguration-time profiles.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "det,exp,gamma,tnorm,lognormal,pareto"
    )]
    pub profiles: Vec<String>,
    /// Numbers of classes.
    #[arg(long, value_delimiter = ',', default_value = "2,3,4,6,8")]
    pub ns: Vec<usize>,
    /// Total loads.
    #[arg(long, value_delimiter = ',', default_value = "1/20,1/10,1/5,3/10,1/2,7/10,17/20,9/10,19/20", value_parser = parse_rational)]
    pub rhos: Vec<Q>,
    /// Deadlines in seconds.
    #[arg(long, value_delimiter = ',', default_value = "1,2,3,4,6,8,12", value_parser = parse_rational)]
    pub ds: Vec<Q>,
    /// Deadlines as multiples of `I = (1 - rho/N) N taubar / (1 - rho)`, the mean time between
    /// visits to a configuration under cyclic exhaustive service at mean reconfiguration time
    /// `taubar`, each rounded to the nearest multiple of the grid step, in place of `--ds` (for
    /// example 0.5,0.75,0.9,1,1.25,1.5,2). The grid cells grow with the deadline, and the cost
    /// of the coefficients with their square.
    #[arg(long, value_delimiter = ',', value_parser = parse_rational)]
    pub d_over_i: Vec<Q>,
    /// Target miss probabilities.
    #[arg(long, value_delimiter = ',', default_value = "1/1000,1/100,1/20", value_parser = parse_rational)]
    pub epsilons: Vec<Q>,
    /// Mean reconfiguration times in seconds; the classes and the distribution shapes stay the
    /// same, and every setting is evaluated at each mean.
    #[arg(long = "means", alias = "mean", value_delimiter = ',', default_value = "1/2", value_parser = parse_rational)]
    pub means: Vec<Q>,
    /// Squared coefficient of variation of the reconfiguration time.
    #[arg(long, default_value = "1/3", value_parser = parse_rational)]
    pub cv2: Q,
    /// Grid step in seconds (default 1/2000 up to 4 classes, 1/1000 up to 8, 1/500 beyond,
    /// multiplied by the mean reconfiguration time over 1/2).
    #[arg(long)]
    pub step: Option<String>,
    /// Refinement sweeps of the upper bound on the completion probability (default 3).
    #[arg(long)]
    pub sweeps: Option<usize>,
    /// Fractional bits of the rational enclosures (default 128).
    #[arg(long)]
    pub bits: Option<u32>,
    /// Impose the window constraint on every subset of at least two classes (default: all
    /// classes, every pair and, with more than three classes, every leave-one-out set).
    #[arg(long)]
    pub all_subsets: bool,
    /// Add the visit-gap constraints (groups `gap_window` and `gap`) to the full test.
    #[arg(long)]
    pub gap_rows: bool,
    /// Directory of cached phase coefficients, read when present and written otherwise
    /// (default: no cache).
    #[arg(long)]
    pub cache: Option<PathBuf>,
    /// Worker threads (default: all cores).
    #[arg(long)]
    pub threads: Option<usize>,
}

/// The deadline of a group of settings: in seconds, or a multiple of the mean intervisit time,
/// which depends on the load.
#[derive(Clone, Debug)]
enum Deadline {
    Seconds(Q),
    OverI(Q),
}

/// The deadline `r I` rounded to the nearest multiple of `step` (at least one step), with
/// `I = (1 - rho/N) N mean / (1 - rho)`.
fn deadline_over_i(n: usize, rho: &Q, mean: &Q, r: &Q, step: &Q) -> Result<Q, String> {
    if *rho >= Q::one() {
        return Err(format!("--d-over-i needs a load below one, got {rho}"));
    }
    let nq = qi(n as i64);
    let intervisit = (Q::one() - rho / &nq) * &nq * mean / (Q::one() - rho);
    Ok((r * intervisit / step).round().max(Q::one()) * step)
}

/// The identifier of a setting ([`point_id`]), followed by the mean reconfiguration time when it
/// is not 1/2 and by the multiple of `I` when the deadline is given as one.
fn setting_id(p: &str, n: usize, rho: &Q, d: &Q, eps: &Q, mean: &Q, spec: &Deadline) -> String {
    let mut id = point_id(p, n, rho, d, eps);
    if *mean != qr(1, 2) {
        id.push_str(&format!("|tau={mean}"));
    }
    if let Deadline::OverI(r) = spec {
        id.push_str(&format!("|d/I={r}"));
    }
    id
}

/// Peak resident memory of the process in kB, where the platform reports it.
fn peak_memory_kb() -> Option<u64> {
    let s = fs::read_to_string("/proc/self/status").ok()?;
    s.lines()
        .find(|l| l.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

/// One row of the table.
#[allow(clippy::too_many_arguments)]
fn row(
    id: &str,
    profile_name: &str,
    n: usize,
    rho: &Q,
    d: &Q,
    eps: &Q,
    law: &Distribution,
    ev: &Evaluation,
    bits: u32,
) -> Value {
    let mut levels = serde_json::Map::new();
    for l in &ev.evaluation.levels {
        levels.insert(l.level.name().into(), json!({
            "category": l.category(), "lb_max": l.lower_bound_max(), "lb_avg": l.lower_bound_avg(), "rows": l.rows,
            "feasibility_status": l.feasibility.status, "feasibility_exact_simplex": l.feasibility.used_exact_simplex,
            "min_max_status": l.min_max.status, "min_max_exact_simplex": l.min_max.used_exact_simplex
        }));
    }
    let targets = ev.diagnostics["targets"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let maxf = |key: &str| {
        targets
            .iter()
            .filter_map(|t| t[key].as_f64())
            .fold(0.0, f64::max)
    };
    json!({
        "id": id, "profile": profile_name, "n": n, "rho": rho.to_string(), "rho_f64": to_f64(rho), "d": d.to_string(), "d_f64": to_f64(d),
        "eps": eps.to_string(), "eps_f64": to_f64(eps), "law": law.id(), "law_mean": law.mean(bits).0.to_string(),
        "levels": levels,
        "grid": {"step": ev.diagnostics["step"], "cells": ev.diagnostics["cells"], "sweeps": maxf("sweeps"), "last_sweep_change": maxf("last_sweep_change"),
                 "max_h_width": maxf("max_h_width"), "max_a_width": maxf("max_a_width"), "max_b_width": maxf("max_b_width"), "max_c_width": maxf("max_c_width"),
                 "mean_width": ev.diagnostics["law_tables"][0]["mean_width"], "cache_hit": ev.cache_hit},
        "timing_ms": {"coefficients": ev.evaluation.coefficient_ms, "windows": ev.evaluation.window_ms, "lp": ev.evaluation.lp_ms,
                      "certificates": ev.evaluation.certificate_ms, "verification": ev.evaluation.verify_ms}
    })
}

fn csv_line(row: &Value) -> String {
    let lv = |l: &str, k: &str| -> String {
        match &row["levels"][l][k] {
            Value::Null => String::new(),
            Value::String(s) => s.clone(),
            other => other.to_string(),
        }
    };
    let g = |k: &str| row["grid"][k].to_string().trim_matches('"').to_string();
    let t = |k: &str| row["timing_ms"][k].to_string();
    let mut fields = vec![
        row["id"].as_str().unwrap().to_string(),
        row["profile"].as_str().unwrap().to_string(),
        row["n"].to_string(),
        row["rho"].as_str().unwrap().to_string(),
        row["rho_f64"].to_string(),
        row["d"].as_str().unwrap().to_string(),
        row["eps"].as_str().unwrap().to_string(),
        row["eps_f64"].to_string(),
        row["law"].as_str().unwrap().to_string().replace(',', ";"),
    ];
    for l in ["base", "base_windows", "base_phases", "full"] {
        fields.push(lv(l, "category"));
        fields.push(lv(l, "lb_max"));
        fields.push(lv(l, "lb_avg"));
        fields.push(lv(l, "rows"));
        fields.push(lv(l, "feasibility_exact_simplex"));
    }
    for k in [
        "step",
        "cells",
        "sweeps",
        "last_sweep_change",
        "max_h_width",
        "max_a_width",
        "max_b_width",
        "max_c_width",
        "mean_width",
        "cache_hit",
    ] {
        fields.push(g(k));
    }
    for k in [
        "coefficients",
        "windows",
        "lp",
        "certificates",
        "verification",
    ] {
        fields.push(t(k));
    }
    fields.join(",")
}

fn csv_header() -> String {
    let mut h = vec![
        "id", "profile", "n", "rho", "rho_f64", "d", "eps", "eps_f64", "law",
    ]
    .into_iter()
    .map(String::from)
    .collect::<Vec<_>>();
    for l in ["base", "base_windows", "base_phases", "full"] {
        for k in ["category", "lb_max", "lb_avg", "rows", "exact_simplex"] {
            h.push(format!("{l}_{k}"));
        }
    }
    for k in [
        "step",
        "cells",
        "sweeps",
        "last_sweep_change",
        "max_h_width",
        "max_a_width",
        "max_b_width",
        "max_c_width",
        "mean_width",
        "cache_hit",
    ] {
        h.push(format!("grid_{k}"));
    }
    for k in [
        "coefficients",
        "windows",
        "lp",
        "certificates",
        "verification",
    ] {
        h.push(format!("ms_{k}"));
    }
    h.join(",")
}

/// Certify the grid and write the tables; returns the summary.
pub fn run(args: &Args) -> Result<Value, String> {
    let out = &args.out;
    let cache = args.cache.as_deref();
    fs::create_dir_all(out.join("certificates")).map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    // The columns `mean` and `d_over_i` appear only when the grid varies them.
    let axis = args.means != [qr(1, 2)] || !args.d_over_i.is_empty();
    let deadlines: Vec<Deadline> = if args.d_over_i.is_empty() {
        args.ds.iter().cloned().map(Deadline::Seconds).collect()
    } else {
        args.d_over_i.iter().cloned().map(Deadline::OverI).collect()
    };
    let mut groups = Vec::new();
    for p in &args.profiles {
        for mean in &args.means {
            let law = profile_law(p, mean, &args.cv2)?;
            for &n in &args.ns {
                for d in &deadlines {
                    groups.push((p.clone(), law.clone(), mean.clone(), n, d.clone()));
                }
            }
        }
    }
    let total_points = groups.len() * args.rhos.len() * args.epsilons.len();
    eprintln!(
        "map: {} coefficient groups, {} points",
        groups.len(),
        total_points
    );
    let done = AtomicUsize::new(0);
    let rows: Vec<Result<Vec<Value>, String>> = groups
        .par_iter()
        .map(|(p, law, mean, n, spec)| {
            let mut settings = grid_settings(*n, args.step.as_deref(), args.sweeps, args.bits, args.all_subsets)?;
            if args.step.is_none() {
                settings.step = &settings.step * mean * qi(2);
            }
            settings.gap_rows = args.gap_rows;
            let mut out_rows = Vec::new();
            let t1 = Instant::now();
            for rho in &args.rhos {
                let d = match spec {
                    Deadline::Seconds(d) => d.clone(),
                    Deadline::OverI(r) => deadline_over_i(*n, rho, mean, r, &settings.step)?,
                };
                for eps in &args.epsilons {
                    let id = setting_id(p, *n, rho, &d, eps, mean, spec);
                    let model = Model::balanced(&id, law, *n, rho, &d, eps);
                    let ev = evaluate(&model, &settings, &Level::all(), cache)?;
                    let mut r = row(&id, p, *n, rho, &d, eps, law, &ev, settings.bits);
                    if axis {
                        r["mean"] = json!(mean.to_string());
                        r["d_over_i"] = match spec {
                            Deadline::OverI(x) => json!(x.to_string()),
                            Deadline::Seconds(_) => Value::Null,
                        };
                    }
                    if args.gap_rows {
                        for l in &ev.evaluation.levels {
                            r["levels"][l.level.name()]["groups"] = json!(l.groups);
                        }
                    }
                    let full = ev.evaluation.level(Level::Full);
                    write_json(
                        &out.join("certificates").join(format!("{}.json", file_stem(&id))),
                        &json!({"id": id, "model": model.to_json(), "laws": model.describe_laws(settings.bits), "settings": settings.to_json(),
                                "full_level": full.to_json(), "levels": ev.evaluation.levels.iter().map(|l| json!({"level": l.level.name(), "category": l.category(), "lb_max": l.lower_bound_max(), "lb_avg": l.lower_bound_avg()})).collect::<Vec<_>>(),
                                "diagnostics": ev.diagnostics}),
                    )?;
                    out_rows.push(r);
                }
            }
            let k = done.fetch_add(1, Ordering::SeqCst) + 1;
            let d = match spec {
                Deadline::Seconds(d) => format!("d={d}"),
                Deadline::OverI(r) => format!("d/I={r}"),
            };
            let tau = if *mean != qr(1, 2) { format!(" tau={mean}") } else { String::new() };
            eprintln!("map: group {k}/{} {p}{tau} N={n} {d} done in {:.1} s (elapsed {:.0} s)", groups.len(), t1.elapsed().as_secs_f64(), t0.elapsed().as_secs_f64());
            Ok(out_rows)
        })
        .collect();
    let mut all = Vec::new();
    for r in rows {
        all.extend(r?);
    }
    let field = |v: &Value| v.as_str().unwrap_or_default().to_string();
    let mut csv = vec![csv_header() + if axis { ",mean,d_over_i" } else { "" }];
    csv.extend(all.iter().map(|r| {
        let line = csv_line(r);
        if axis {
            format!("{line},{},{}", field(&r["mean"]), field(&r["d_over_i"]))
        } else {
            line
        }
    }));
    write_lines(&out.join("map.csv"), &csv)?;
    let mut summary = json!({
        "points": all.len(), "groups": groups.len(), "total_seconds": t0.elapsed().as_secs_f64(),
        "profiles": args.profiles, "ns": args.ns, "rhos": args.rhos.iter().map(|q| q.to_string()).collect::<Vec<_>>(),
        "ds": args.ds.iter().map(|q| q.to_string()).collect::<Vec<_>>(), "epsilons": args.epsilons.iter().map(|q| q.to_string()).collect::<Vec<_>>(),
        "mean": args.means[0].to_string(), "cv2": args.cv2.to_string(),
        "full_infeasible": all.iter().filter(|r| r["levels"]["full"]["category"] == "certified_infeasible").count(),
        "base_infeasible": all.iter().filter(|r| r["levels"]["base"]["category"] == "certified_infeasible").count(),
        "base_windows_infeasible": all.iter().filter(|r| r["levels"]["base_windows"]["category"] == "certified_infeasible").count(),
        "base_phases_infeasible": all.iter().filter(|r| r["levels"]["base_phases"]["category"] == "certified_infeasible").count(),
        "inconclusive": all.iter().filter(|r| r["levels"]["full"]["category"] == "numerically_inconclusive").count(),
        "peak_memory_kb": peak_memory_kb()
    });
    // Recorded only when used, so that the summary of the default grid is unchanged.
    if args.means.len() > 1 {
        summary["means"] = json!(args.means.iter().map(|q| q.to_string()).collect::<Vec<_>>());
        summary.as_object_mut().unwrap().remove("mean");
    }
    if !args.d_over_i.is_empty() {
        summary["d_over_i"] = json!(
            args.d_over_i
                .iter()
                .map(|q| q.to_string())
                .collect::<Vec<_>>()
        );
        summary.as_object_mut().unwrap().remove("ds");
    }
    if args.gap_rows {
        summary["gap_rows"] = json!(true);
        summary["full_groups"] = json!(Level::Full.groups_with(true));
    }
    write_json(
        &out.join("map.json"),
        &json!({"summary": summary, "rows": all}),
    )?;
    write_json(&out.join("summary.json"), &summary)?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feasibility::rational::parse_q;

    #[test]
    fn deadlines_over_the_intervisit_time_are_rounded_to_the_step() {
        // N = 4, rho = 7/10, mean 50: I = (1 - 7/40) 4 50 / (3/10) = 550.
        let d = deadline_over_i(4, &qr(7, 10), &qi(50), &qr(1, 2), &qr(1, 20)).unwrap();
        assert_eq!(d, qi(275));
        // N = 4, rho = 9/10, mean 1/2: I = 31/2, and 3/4 I is a multiple of 1/2000.
        let d = deadline_over_i(4, &qr(9, 10), &qr(1, 2), &qr(3, 4), &qr(1, 2000)).unwrap();
        assert_eq!(d, parse_q("11.625").unwrap());
        // N = 2, rho = 1/20: I = 39/38; I/3 = 0.342 rounds to 3/10, and I/100 to one step.
        let d = deadline_over_i(2, &qr(1, 20), &qr(1, 2), &qr(1, 3), &qr(1, 10)).unwrap();
        assert_eq!(d, qr(3, 10));
        let d = deadline_over_i(2, &qr(1, 20), &qr(1, 2), &qr(1, 100), &qr(1, 10)).unwrap();
        assert_eq!(d, qr(1, 10));
        assert!(deadline_over_i(2, &qi(1), &qr(1, 2), &qi(1), &qr(1, 10)).is_err());
    }

    #[test]
    fn identifiers_carry_the_mean_and_the_multiple_only_when_used() {
        let (rho, eps, d) = (qr(9, 10), qr(1, 100), Deadline::Seconds(qi(4)));
        let id = setting_id("det", 4, &rho, &qi(4), &eps, &qr(1, 2), &d);
        assert_eq!(id, point_id("det", 4, &rho, &qi(4), &eps));
        let id = setting_id("det", 4, &rho, &qi(4), &eps, &qi(5), &d);
        assert_eq!(id, "det|N=4|rho=9/10|d=4|eps=1/100|tau=5");
        let r = Deadline::OverI(qr(1, 2));
        let id = setting_id("det", 4, &qr(7, 10), &qi(275), &eps, &qi(50), &r);
        assert_eq!(id, "det|N=4|rho=7/10|d=275|eps=1/100|tau=50|d/I=1/2");
    }
}
