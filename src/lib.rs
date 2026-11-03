//! Certified tail-delay feasibility tests for reconfigurable systems. The system serves `N` job
//! classes, one per configuration; changing the configuration takes a random reconfiguration time
//! without service, and each class has a delay threshold and a target for its miss probability.
//! Before a policy is designed, [`feasibility`] combines policy-independent necessary conditions
//! into a linear relaxation whose verdicts are verified in exact rational arithmetic, and brackets
//! the best achievable miss probability in the symmetric case. [`witness`] certifies the
//! sufficient side: an upper bound on the miss probability under exhaustive cyclic polling (EC).
//! [`sim`] simulates EC and OJF against these bounds, [`experiments`] implements the commands of
//! the `tdf` binary, and [`digest`] provides SHA-256 for stream seeds, cache keys and run
//! manifests.

#![allow(
    // Index loops follow the subscripts of the formulas they implement.
    clippy::needless_range_loop,
    // Nested result tables are spelled out where they are built.
    clippy::type_complexity,
    // A generator holds the MT19937 state inline; a run has only a few generators.
    clippy::large_enum_variant
)]

pub mod digest;
pub mod experiments;
pub mod feasibility;
pub mod sim;
pub mod witness;
