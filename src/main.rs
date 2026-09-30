//! `tdf`: the command-line interface. It parses the arguments, sizes the thread pool and runs the
//! command of `tdf::experiments`, which writes its results to the path given by `--out`; the
//! command's JSON summary goes to standard output, an error to standard error with exit status 1.

use clap::{Parser, Subcommand};
use tdf::experiments::{
    bracket, directional, distributions, map, ojf, point, region, screen, sweep, validate,
};

#[derive(Parser)]
#[command(
    name = "tdf",
    version,
    about = "Certified tail-delay feasibility tests and policy simulations for reconfigurable networks"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Certify one setting at the four constraint levels.
    Point(point::Args),
    /// Certify the evaluation grid of symmetric settings.
    Map(map::Args),
    /// Tabulate the named reconfiguration-time distributions (profiles).
    Distributions(distributions::Args),
    /// Check the coefficient constructions and the simulator's distributions.
    Validate(validate::Args),
    /// Simulate EC and HOL-BMW against the certified bounds.
    Screen(screen::Args),
    /// Certify the interval of the best achievable miss probability (symmetric settings).
    Bracket(bracket::Args),
    /// Simulate OJF (oldest job first) against its reference analysis.
    Ojf(ojf::Args),
    /// Sweep the common target: verdicts, certified intervals and reconfiguration-rate ranges.
    Sweep(sweep::Args),
    /// Certify the range of the total reconfiguration rate over a fine grid of targets.
    Region(region::Args),
    /// Evaluate the three-class case with direction-dependent reconfiguration times.
    Directional(directional::Args),
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
        Command::Point(a) => set_threads(a.threads).and_then(|_| point::run(&a)),
        Command::Map(a) => set_threads(a.threads).and_then(|_| map::run(&a)),
        Command::Distributions(a) => distributions::run(&a),
        Command::Validate(a) => set_threads(a.threads).and_then(|_| validate::run(&a)),
        Command::Screen(a) => set_threads(a.threads).and_then(|_| screen::run(&a)),
        Command::Bracket(a) => set_threads(a.threads).and_then(|_| bracket::run(&a)),
        Command::Ojf(a) => set_threads(a.threads).and_then(|_| ojf::run(&a)),
        Command::Sweep(a) => set_threads(a.threads).and_then(|_| sweep::run(&a)),
        Command::Region(a) => set_threads(a.threads).and_then(|_| region::run(&a)),
        Command::Directional(a) => set_threads(a.threads).and_then(|_| directional::run(&a)),
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
