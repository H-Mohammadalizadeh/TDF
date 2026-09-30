//! `tdf point`: one setting at the four constraint levels (time budget, flow balance and the
//! single-class bounds alone; with the window constraint; with the phase constraint; with both),
//! with the lower and upper phase coefficients and the grid diagnostics. Used for the numerical
//! checks of grid step, precision, refinement sweeps, window subsets and running time.

use super::common::{grid_settings, parse_rational, point_id, profile_law, write_json};
use crate::feasibility::distributions::Model;
use crate::feasibility::evaluate;
use crate::feasibility::rational::Q;
use crate::feasibility::relaxation::Level;
use serde_json::{Value, json};
use std::fs;
use std::path::PathBuf;

/// Certify one setting: a symmetric setting given by a profile, or a model file.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// Reconfiguration-time profile: det, exp, gamma, tnorm, lognormal,
    /// pareto or pareto_heavy (required without --model).
    #[arg(long)]
    pub profile: Option<String>,
    /// Mean reconfiguration time in seconds (with --profile).
    #[arg(long, default_value = "1/2", value_parser = parse_rational)]
    pub mean: Q,
    /// Squared coefficient of variation of the reconfiguration time (with --profile).
    #[arg(long, default_value = "1/3", value_parser = parse_rational)]
    pub cv2: Q,
    /// Number of classes (required without --model).
    #[arg(long)]
    pub n: Option<usize>,
    /// Total load (required without --model).
    #[arg(long, value_parser = parse_rational)]
    pub rho: Option<Q>,
    /// Deadline of every class in seconds (required without --model).
    #[arg(long, value_parser = parse_rational)]
    pub d: Option<Q>,
    /// Target miss probability of every class (with --profile).
    #[arg(long, default_value = "1/100", value_parser = parse_rational)]
    pub eps: Q,
    /// JSON model file (per-class rates, deadlines and targets, and a reconfiguration-time
    /// distribution per ordered pair) instead of a symmetric setting.
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// Write the full evaluation to this JSON file (default: not written).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Grid step in seconds (default 1/2000 up to 4 classes, 1/1000 up to 8, 1/500 beyond).
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
    /// Worker threads (default: all cores).
    #[arg(long)]
    pub threads: Option<usize>,
}

fn model_of(args: &Args) -> Result<Model, String> {
    if let Some(f) = &args.model {
        let text = fs::read_to_string(f).map_err(|e| format!("{}: {e}", f.display()))?;
        return Model::from_json(&serde_json::from_str(&text).map_err(|e| e.to_string())?);
    }
    let name = args
        .profile
        .as_deref()
        .ok_or("--profile or --model is required")?;
    let law = profile_law(name, &args.mean, &args.cv2)?;
    let n = args.n.ok_or("--n is required")?;
    let rho = args.rho.as_ref().ok_or("--rho is required")?;
    let d = args.d.as_ref().ok_or("--d is required")?;
    Ok(Model::balanced(
        &point_id(name, n, rho, d, &args.eps),
        &law,
        n,
        rho,
        d,
        &args.eps,
    ))
}

/// Evaluate the setting, write the full result to `--out` when given and
/// return a summary.
pub fn run(args: &Args) -> Result<Value, String> {
    let model = model_of(args)?;
    let settings = grid_settings(
        model.n,
        args.step.as_deref(),
        args.sweeps,
        args.bits,
        args.all_subsets,
    )?;
    let ev = evaluate(&model, &settings, &Level::all(), None)?;
    let v = json!({
        "model": model.to_json(), "laws": model.describe_laws(settings.bits), "settings": settings.to_json(),
        "evaluation": ev.evaluation.to_json(), "upper_coefficients": ev.upper.to_json(), "lower_coefficients": ev.lower.to_json(),
        "diagnostics": ev.diagnostics
    });
    if let Some(o) = &args.out {
        if let Some(parent) = o.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        write_json(o, &v)?;
    }
    let levels: Vec<Value> = v["evaluation"]["levels"].as_array().unwrap().iter().map(|l| json!({"level": l["level"], "category": l["category"], "lb_max": l["min_max"]["certified_value"], "lb_avg": l["min_arrival_weighted"]["certified_value"], "rows": l["rows"]})).collect();
    Ok(
        json!({"model": v["model"]["name"], "laws": v["laws"], "levels": levels, "diagnostics": {"step": v["diagnostics"]["step"], "cells": v["diagnostics"]["cells"], "targets": v["diagnostics"]["targets"], "timing_ms": v["diagnostics"]["timing_ms"]}, "timing_ms": v["evaluation"]["timing_ms"], "out": args.out}),
    )
}
