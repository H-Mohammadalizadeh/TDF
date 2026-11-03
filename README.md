# Tail-Delay Feasibility in Reconfigurable Systems

Code and data for *Before Scheduling: Tail-Delay Feasibility in Reconfigurable Systems* by Hossein
Mohammadalizadeh and Holger Karl (Hasso Plattner Institute, University of Potsdam), submitted to
IEEE/IFIP NOMS 2027.

`tdf` tests whether per-class tail-delay targets can be met by any scheduling policy. A linear test of
policy-independent necessary conditions, verified in exact arithmetic, certifies infeasibility; certified
miss bounds for exhaustive cyclic polling (EC) and oldest job first (OJF) certify feasibility.

## Reproduce

Requires Rust (the version in `rust-toolchain.toml`) and Python 3 with `plotting/requirements.txt`.
`scripts/reproduce.sh` builds `tdf`, runs the experiments into `results/`, writes the numbers quoted in
the paper to `results/numbers.txt` and draws Figs. 1–3 into `figures/` (about two hours on 16 cores). From
the included results, `scripts/tables.sh`, `scripts/numbers.sh` and `python3 plotting/figures.py`
repeat the last three steps in seconds.

| Script | Paper |
| --- | --- |
| `scripts/grid.sh` | Table I, Sections VI-A and VI-D |
| `scripts/edges.sh` | Fig. 1 |
| `scripts/gap.sh` | Fig. 2 |
| `scripts/classes.sh` | Fig. 3 |
| `scripts/bracket.sh` | Appendix A |

`cargo test --release` runs the tests.

## License

MIT, see `LICENSE`.
