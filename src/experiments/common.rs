//! Helpers shared by the commands: output files, run manifests with the
//! digests of the sources, grid settings from the command-line overrides,
//! the named distributions (profiles), setting identifiers and the JSON
//! summaries of constraint levels and certificates.

use crate::feasibility::LevelResult;
use crate::feasibility::distributions::{Distribution, profile};
use crate::feasibility::grid::GridSettings;
use crate::feasibility::rational::{Q, parse_q, to_f64};
use crate::feasibility::simplex::Certified;
use serde_json::{Value, json};
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Parse an integer, a fraction `a/b` or a decimal exactly (for the
/// command-line options).
pub fn parse_rational(s: &str) -> Result<Q, String> {
    parse_q(s)
}

/// Write `value` as indented JSON.
pub fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    fs::write(
        path,
        serde_json::to_string_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{}: {e}", path.display()))
}

/// Write a CSV table (fields are written as given).
pub fn write_csv(path: &Path, header: &[&str], rows: &[Vec<String>]) -> Result<(), String> {
    let mut text = header.join(",");
    text.push('\n');
    for r in rows {
        text.push_str(&r.join(","));
        text.push('\n');
    }
    fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Write lines of text, each followed by a newline.
pub fn write_lines(path: &Path, lines: &[String]) -> Result<(), String> {
    fs::write(path, lines.join("\n") + "\n").map_err(|e| format!("{}: {e}", path.display()))
}

/// The sources of the crate whose digests the manifests record, relative
/// to the crate directory.
const SOURCES: [&str; 32] = [
    "src/lib.rs",
    "src/main.rs",
    "src/digest.rs",
    "src/feasibility/mod.rs",
    "src/feasibility/coefficients.rs",
    "src/feasibility/distributions.rs",
    "src/feasibility/gaps.rs",
    "src/feasibility/grid.rs",
    "src/feasibility/model.rs",
    "src/feasibility/rational.rs",
    "src/feasibility/reference.rs",
    "src/feasibility/relaxation.rs",
    "src/feasibility/simplex.rs",
    "src/feasibility/special.rs",
    "src/feasibility/windows.rs",
    "src/sim/mod.rs",
    "src/sim/engine.rs",
    "src/sim/laws.rs",
    "src/sim/policies.rs",
    "src/sim/rng.rs",
    "src/sim/runner.rs",
    "src/experiments/mod.rs",
    "src/experiments/common.rs",
    "src/experiments/map.rs",
    "src/experiments/screen.rs",
    "src/experiments/bracket.rs",
    "src/experiments/witness.rs",
    "src/experiments/verdicts.rs",
    "src/witness/mod.rs",
    "src/witness/bound.rs",
    "src/witness/fp.rs",
    "src/witness/laws.rs",
];

/// Digests of the sources, read from the crate directory recorded at build
/// time (`null` for a file that is not found), their combined digest and
/// the package version.
fn source_digest() -> Value {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut per_file = serde_json::Map::new();
    for f in SOURCES {
        let text = fs::read_to_string(format!("{root}/{f}")).ok();
        per_file.insert(
            f.into(),
            text.map(|t| json!(crate::digest::digest(&json!(t))))
                .unwrap_or(Value::Null),
        );
    }
    let all = crate::digest::digest(&Value::Object(per_file.clone()));
    json!({"files": per_file, "combined": all, "package_version": env!("CARGO_PKG_VERSION")})
}

/// The manifest of a run: the command, its arguments, the grid settings,
/// the source digests and the start time.
pub fn manifest(command: &str, args: Value, settings: Option<&GridSettings>) -> Value {
    json!({
        "command": command, "args": args,
        "grid": settings.map(|s| s.to_json()).unwrap_or(Value::Null),
        "sources": source_digest(),
        "started_unix_seconds": SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
    })
}

/// The default grid settings for `n` classes with the given overrides.
pub fn grid_settings(
    n: usize,
    step: Option<&str>,
    sweeps: Option<usize>,
    bits: Option<u32>,
    all_subsets: bool,
) -> Result<GridSettings, String> {
    let mut s = GridSettings::default_for(n);
    s.all_subsets = all_subsets;
    if let Some(st) = step {
        s.step = parse_q(st)?;
    }
    if let Some(k) = sweeps {
        s.sweeps = k;
    }
    if let Some(b) = bits {
        s.bits = b;
    }
    Ok(s)
}

/// A named distribution (one of the six profiles) with the given mean and squared coefficient
/// of variation.
pub fn profile_law(name: &str, mean: &Q, cv2: &Q) -> Result<Distribution, String> {
    profile(name, mean, cv2)
}

/// The identifier of a symmetric setting, e.g. `gamma|N=4|rho=1/5|d=2|eps=1/100`.
/// `rho` is the total load, `d` the deadline and `eps` the target.
pub fn point_id(profile_name: &str, n: usize, rho: &Q, d: &Q, eps: &Q) -> String {
    format!("{profile_name}|N={n}|rho={rho}|d={d}|eps={eps}")
}

/// A file name derived from an identifier.
pub fn file_stem(id: &str) -> String {
    id.replace(['|', '/', '=', ' '], "_")
}

/// The constraint families with nonzero multipliers in a certificate.
pub fn witness_families(c: &Certified) -> Vec<String> {
    let mut fams: Vec<String> = c
        .witness
        .as_ref()
        .map(|w| {
            w.iter()
                .map(|(label, _)| {
                    let l = label.as_str();
                    if l.starts_with("phase[") {
                        "phase"
                    } else if l.starts_with("gap_window[") {
                        "gap_window"
                    } else if l.starts_with("gap_weak[") || l.starts_with("gap_strong[") {
                        "gap"
                    } else if l.starts_with("window[") {
                        "window"
                    } else if l.starts_with("time_accounting") {
                        "time"
                    } else if l.starts_with("flow[") {
                        "flow"
                    } else if l.starts_with("mm1_dedicated[") {
                        "single_class"
                    } else if l.starts_with("service_only[") {
                        "service_only"
                    } else if l.starts_with("target[") {
                        "target"
                    } else {
                        "bounds"
                    }
                    .to_string()
                })
                .collect()
        })
        .unwrap_or_default();
    fams.sort();
    fams.dedup();
    fams
}

/// A short JSON summary of one constraint level.
pub fn level_summary(l: &LevelResult) -> Value {
    json!({
        "level": l.level.name(), "category": l.category(), "rows": l.rows,
        "lb_max": l.lower_bound_max(), "lb_avg": l.lower_bound_avg(),
        "min_max_status": l.min_max.status, "min_max_exact_optimum": l.min_max.exact_optimum,
        "min_max_exact_simplex": l.min_max.used_exact_simplex,
        "feasibility_status": l.feasibility.status, "witness_families": witness_families(&l.feasibility)
    })
}

/// A rational as a decimal string for CSV, empty when absent.
pub fn opt_f(x: &Option<Q>) -> String {
    x.as_ref()
        .map(|v| to_f64(v).to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::SOURCES;
    use std::path::Path;

    fn rust_files(dir: &Path, root: &Path, out: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                rust_files(&path, root, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(
                    path.strip_prefix(root)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/"),
                );
            }
        }
    }

    #[test]
    fn the_manifest_digest_covers_every_source_file() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut found = Vec::new();
        rust_files(&root.join("src"), root, &mut found);
        found.sort();
        let mut listed: Vec<String> = SOURCES.iter().map(|s| s.to_string()).collect();
        listed.sort();
        assert_eq!(found, listed);
    }
}
