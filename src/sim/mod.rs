//! Discrete-event simulation of the reconfigurable system.
//!
//! * [`engine`]: the event loop, the policy interface and the run record.
//! * [`laws`]: service-time and reconfiguration-time distributions with samplers.
//! * [`policies`]: the scheduling policies EC (`ec`, exhaustive cyclic polling) and OJF (`ojf`).
//! * [`rng`]: the random streams and their seeding.
//! * [`runner`]: run specifications, observation windows and condensed runs of a model.

pub mod engine;
pub mod laws;
pub mod policies;
pub mod rng;
pub mod runner;
