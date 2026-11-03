//! `tdf`: the command-line interface. It parses the arguments, sizes the thread pool and runs the
//! command of `tdf::experiments`, which writes its results to the path given by `--out`; the
//! command's JSON summary goes to standard output, an error to standard error with exit status 1.

use clap::{Parser, Subcommand};
use tdf::experiments::{bracket, map, screen, verdicts, witness};

#[derive(Parser)]
#[command(
    name = "tdf",
    version,
    about = "Certified tail-delay feasibility tests for reconfigurable systems"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Certify the evaluation grid of symmetric settings (the feasibility test).
    Map(map::Args),
    /// Certify an upper bound on the miss probability under exhaustive cyclic polling (EC) and
    /// OJF on the grid of `map` (the sufficient side).
    Witness(witness::Args),
    /// Join `map.csv` and `witness.csv` into certified-infeasible, certified-feasible and
    /// undetermined verdicts.
    Verdicts(verdicts::Args),
    /// Certify the interval of the best achievable miss probability (symmetric settings).
    Bracket(bracket::Args),
    /// Simulate EC or OJF against the certified bounds.
    Screen(screen::Args),
}

/// Size the global thread pool when `--threads` is given.
fn set_threads(requested: Option<usize>) -> Result<(), String> {
    if let Some(t) = requested {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn run(command: Command) -> Result<serde_json::Value, String> {
    match command {
        Command::Map(a) => set_threads(a.threads).and_then(|_| map::run(&a)),
        Command::Witness(a) => set_threads(a.threads).and_then(|_| witness::run(&a)),
        Command::Verdicts(a) => verdicts::run(&a),
        Command::Bracket(a) => set_threads(a.threads).and_then(|_| bracket::run(&a)),
        Command::Screen(a) => set_threads(a.threads).and_then(|_| screen::run(&a)),
    }
}

fn main() {
    match run(Cli::parse().command) {
        Ok(summary) => println!("{}", serde_json::to_string(&summary).unwrap()),
        Err(e) => {
            eprintln!("tdf: {e}");
            std::process::exit(1);
        }
    }
}
