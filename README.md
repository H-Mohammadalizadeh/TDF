# Tail-Delay Feasibility in Reconfigurable Networks

Code for the paper *Before Scheduling: Tail-Delay Feasibility in Reconfigurable Networks* submitted to IEEE/IFIP NOMS 2027.

Hossein Mohammadalizadeh and Holger Karl (Hasso Plattner Institute, University of Potsdam).

A reconfigurable network serves one class of jobs per configuration and needs a random
reconfiguration time to switch between configurations. `tdf` tests whether per-class tail-delay
targets can be met by any scheduling policy: it combines policy-independent necessary conditions
into a linear relaxation and verifies every verdict in exact rational arithmetic. A discrete-event
simulator runs the policies EC, HOL-BMW and OJF against the certified bounds, and `plotting/`
draws the paper's figures and numbers from the results.

## Directory map

```
src/
  feasibility/    necessary conditions, certified coefficients, LP relaxation, exact certificates,
                  OJF reference analysis
  sim/            discrete-event simulator and the policies EC, HOL-BMW and OJF
  experiments/    the tdf commands, one module each
  main.rs         command-line interface
tests/            tests against closed forms and exact references
scripts/          the paper's experiments (01_grid.sh ... 06_scheduling.sh); reproduce.sh runs all
plotting/         the paper's figures and number macros (paper.py)
results/          experiment outputs (not tracked)
figures/          generated figures and numbers (not tracked)
```

## Usage

Requires Rust (the version pinned in `rust-toolchain.toml`) and Python 3 with the packages in
`plotting/requirements.txt`.

```
cargo build --release         # builds target/release/tdf
cargo test
scripts/reproduce.sh          # every experiment, then the figures (about 1.5 h on 16 cores)
python3 plotting/paper.py     # figures and numbers from existing results
```

`scripts/reproduce.sh 2 3 4 5 6` skips the evaluation grid, which takes most of the time, and
`target/release/tdf <command> --help` describes each command. Seeds are fixed and certificates
are exact, so a rerun reproduces the results; only timing fields differ.

## License

MIT, see `LICENSE`.
