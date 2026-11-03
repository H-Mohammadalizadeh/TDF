//! `tdf verdicts`: the three-way verdict of every setting and target from the necessary side,
//! `map.csv` of `tdf map` (the full test is certified infeasible), and the sufficient side,
//! `witness.csv` of `tdf witness` (a certified upper bound on the miss probability of an
//! admissible policy is at or below the target): `certified_infeasible`, `certified_feasible`
//! or `undetermined`. Both at once, `impossible`, is a correctness failure and must not occur;
//! neither may a certified lower bound of the relaxation above the upper bound (`lb_above_upper`).
//! The upper bound is that of cyclic exhaustive polling (`hi_f64`, the outward f64 of `hi`) or,
//! with identical classes, the smaller of it and the OJF bound (`ojf_hi`; `--no-ojf` leaves it
//! out). A target below the `--ojf-above` of `tdf witness` could need an OJF bound that was left
//! out; such rows are counted (`ojf_left_out_but_needed`, zero when every target is at least that
//! value). The tables are joined on the identifier of `tdf map` without the target. Output:
//! `verdicts.csv` (one row per setting and target), `counts.csv` (per load and target, then over
//! all loads) and `summary.json`.

use super::common::{write_csv, write_json};
use super::witness::strip_target;
use crate::feasibility::rational::{Q, parse_q, to_f64};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

/// Join the necessary and the sufficient side into verdicts.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// `map.csv` of `tdf map`.
    #[arg(long)]
    pub map: PathBuf,
    /// `witness.csv` of `tdf witness`.
    #[arg(long)]
    pub witness: PathBuf,
    /// Output directory.
    #[arg(long)]
    pub out: PathBuf,
    /// Use the bound of cyclic exhaustive polling only (ignore the OJF bound).
    #[arg(long)]
    pub no_ojf: bool,
}

/// A CSV table split at commas (no field of `map.csv` or `witness.csv` contains one).
struct Table {
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    fn read(path: &Path) -> Result<Table, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let split = |l: &str| l.split(',').map(String::from).collect::<Vec<_>>();
        let header = split(
            lines
                .next()
                .ok_or_else(|| format!("{}: empty", path.display()))?,
        );
        let rows: Vec<Vec<String>> = lines.map(split).collect();
        if let Some((k, r)) = rows
            .iter()
            .enumerate()
            .find(|(_, r)| r.len() != header.len())
        {
            return Err(format!(
                "{}: line {} has {} fields, the header {}",
                path.display(),
                k + 2,
                r.len(),
                header.len()
            ));
        }
        Ok(Table { header, rows })
    }

    fn col(&self, name: &str) -> Result<usize, String> {
        self.header
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| format!("column {name} missing"))
    }
}

/// An upper bound written as an f64 (`hi_f64`, `ojf_hi`) as an exact rational; `None` if empty.
fn bound(s: &str) -> Result<Option<Q>, String> {
    if s.is_empty() {
        return Ok(None);
    }
    let v: f64 = s.parse().map_err(|_| format!("not a number: {s}"))?;
    Q::from_float(v)
        .map(Some)
        .ok_or_else(|| format!("not finite: {s}"))
}

/// Verdict counts of a load and target (or of a target over all loads).
#[derive(Clone, Debug, Default)]
struct Counts {
    settings: usize,
    infeasible: usize,
    feasible: usize,
    undetermined: usize,
    impossible: usize,
    /// certified feasible by cyclic exhaustive polling, by OJF
    feasible_ec: usize,
    feasible_ojf: usize,
}

impl Counts {
    fn add(&mut self, verdict: &str, ec: bool, ojf: bool) {
        self.settings += 1;
        match verdict {
            "certified_infeasible" => self.infeasible += 1,
            "certified_feasible" => self.feasible += 1,
            "undetermined" => self.undetermined += 1,
            _ => self.impossible += 1,
        }
        self.feasible_ec += usize::from(ec);
        self.feasible_ojf += usize::from(ojf);
    }

    fn fields(&self) -> Vec<String> {
        [
            self.settings,
            self.infeasible,
            self.feasible,
            self.undetermined,
            self.impossible,
            self.feasible_ec,
            self.feasible_ojf,
        ]
        .iter()
        .map(|x| x.to_string())
        .collect()
    }

    fn to_json(&self) -> Value {
        json!({"settings": self.settings, "certified_infeasible": self.infeasible, "certified_feasible": self.feasible,
               "undetermined": self.undetermined, "impossible": self.impossible,
               "feasible_by_ec": self.feasible_ec, "feasible_by_ojf": self.feasible_ojf})
    }
}

/// The verdict from the two sides.
fn verdict(infeasible: bool, feasible: bool) -> &'static str {
    match (infeasible, feasible) {
        (true, true) => "impossible",
        (true, false) => "certified_infeasible",
        (false, true) => "certified_feasible",
        (false, false) => "undetermined",
    }
}

/// Join the tables and write the verdicts and their counts; returns the summary.
pub fn run(args: &Args) -> Result<Value, String> {
    fs::create_dir_all(&args.out).map_err(|e| e.to_string())?;
    let wt = Table::read(&args.witness)?;
    let (wid, whi, wojf) = (wt.col("id")?, wt.col("hi_f64")?, wt.col("ojf_hi")?);
    let wsrc = wt.col("ojf_source")?;
    let mut upper: HashMap<String, (Q, Option<Q>)> = HashMap::new();
    // settings whose OJF bound was left out as not needed (`tdf witness --ojf-above`)
    let mut not_needed: HashMap<String, bool> = HashMap::new();
    for r in &wt.rows {
        let ec = bound(&r[whi])?.ok_or_else(|| format!("{}: no hi_f64", r[wid]))?;
        let ojf = if args.no_ojf { None } else { bound(&r[wojf])? };
        not_needed.insert(r[wid].clone(), r[wsrc] == "not_needed");
        if upper.insert(r[wid].clone(), (ec, ojf)).is_some() {
            return Err(format!(
                "{}: duplicate setting in {}",
                r[wid],
                args.witness.display()
            ));
        }
    }
    let mt = Table::read(&args.map)?;
    let c = |name: &str| mt.col(name);
    let (mid, mprof, mn, mrho, md) = (c("id")?, c("profile")?, c("n")?, c("rho")?, c("d")?);
    let (meps, mcat, mlb) = (c("eps")?, c("full_category")?, c("full_lb_max")?);
    let mut rows = Vec::with_capacity(mt.rows.len());
    let mut per_load: BTreeMap<(Q, Q), Counts> = BTreeMap::new();
    let mut per_target: BTreeMap<Q, Counts> = BTreeMap::new();
    let mut used: HashMap<String, usize> = HashMap::new();
    let (mut missing, mut lb_above, mut inconclusive) = (0usize, 0usize, 0usize);
    let mut skipped_needed = 0usize;
    let f = |q: &Q| format!("{:e}", to_f64(q));
    for r in &mt.rows {
        let key = strip_target(&r[mid]);
        let (eps, rho) = (parse_q(&r[meps])?, parse_q(&r[mrho])?);
        let infeasible = r[mcat] == "certified_infeasible";
        inconclusive += usize::from(r[mcat] == "numerically_inconclusive");
        let up = upper.get(&key);
        let ec_ok = up.is_some_and(|(ec, _)| *ec <= eps);
        // a left-out OJF bound could matter only where the witness misses the target
        skipped_needed +=
            usize::from(!args.no_ojf && !ec_ok && not_needed.get(&key) == Some(&true));
        match up {
            Some(_) => *used.entry(key).or_default() += 1,
            None => missing += 1,
        }
        let ojf_ok = up.and_then(|(_, o)| o.as_ref()).is_some_and(|o| *o <= eps);
        let v = verdict(infeasible, ec_ok || ojf_ok);
        let best = up.map(|(ec, o)| match o {
            Some(o) if o < ec => o.clone(),
            _ => ec.clone(),
        });
        let lb = bound(&r[mlb])?;
        if let (Some(lb), Some(b)) = (&lb, &best) {
            lb_above += usize::from(lb > b);
        }
        let by: Vec<&str> = [(ec_ok, "ec"), (ojf_ok, "ojf")]
            .iter()
            .filter(|(ok, _)| *ok)
            .map(|(_, name)| *name)
            .collect();
        per_load
            .entry((eps.clone(), rho))
            .or_default()
            .add(v, ec_ok, ojf_ok);
        per_target.entry(eps).or_default().add(v, ec_ok, ojf_ok);
        rows.push(vec![
            r[mid].clone(),
            r[mprof].clone(),
            r[mn].clone(),
            r[mrho].clone(),
            r[md].clone(),
            r[meps].clone(),
            r[mcat].clone(),
            r[mlb].clone(),
            up.map(|(ec, _)| f(ec)).unwrap_or_default(),
            up.and_then(|(_, o)| o.as_ref()).map(f).unwrap_or_default(),
            best.as_ref().map(f).unwrap_or_default(),
            v.to_string(),
            by.join("+"),
        ]);
    }
    write_csv(
        &args.out.join("verdicts.csv"),
        &[
            "id",
            "profile",
            "n",
            "rho",
            "d",
            "eps",
            "full_category",
            "full_lb_max",
            "ec_hi",
            "ojf_hi",
            "upper",
            "verdict",
            "certified_by",
        ],
        &rows,
    )?;
    let mut table = Vec::new();
    for (eps, total) in &per_target {
        for ((e, rho), cnt) in per_load.iter().filter(|((e, _), _)| e == eps) {
            let mut line = vec![rho.to_string(), e.to_string()];
            line.extend(cnt.fields());
            table.push(line);
        }
        let mut line = vec!["all".to_string(), eps.to_string()];
        line.extend(total.fields());
        table.push(line);
    }
    write_csv(
        &args.out.join("counts.csv"),
        &[
            "rho",
            "eps",
            "settings",
            "certified_infeasible",
            "certified_feasible",
            "undetermined",
            "impossible",
            "feasible_by_ec",
            "feasible_by_ojf",
        ],
        &table,
    )?;
    let impossible: usize = per_target.values().map(|c| c.impossible).sum();
    let summary = json!({
        "map_rows": mt.rows.len(), "witness_rows": wt.rows.len(), "ojf": !args.no_ojf,
        "missing_witness": missing, "unused_witness_rows": upper.len() - used.len(),
        "impossible": impossible, "lb_above_upper": lb_above, "numerically_inconclusive": inconclusive,
        "ojf_left_out_but_needed": skipped_needed,
        "by_target": per_target.iter().map(|(e, c)| (e.to_string(), c.to_json())).collect::<serde_json::Map<_, _>>()
    });
    write_json(&args.out.join("summary.json"), &summary)?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdicts_and_bounds() {
        assert_eq!(verdict(true, false), "certified_infeasible");
        assert_eq!(verdict(false, true), "certified_feasible");
        assert_eq!(verdict(false, false), "undetermined");
        assert_eq!(verdict(true, true), "impossible");
        assert_eq!(bound("").unwrap(), None);
        assert_eq!(bound("2.5e-1").unwrap(), Some(Q::new(1.into(), 4.into())));
        assert!(bound("x").is_err());
    }
}
