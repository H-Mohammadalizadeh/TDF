//! The commands of `tdf`, one module each ([`map`], [`witness`], [`verdicts`], [`bracket`],
//! [`screen`]), and the helpers of [`common`]. Every command writes its results into the
//! directory given by `--out` and returns a short JSON summary, which the binary prints. A
//! *profile* is a named reconfiguration-time distribution at a given mean and squared coefficient
//! of variation. With `--cache`, the phase coefficients of a model are read from that directory
//! when present and written otherwise; they depend only on the service rates, deadlines,
//! reconfiguration-time distributions and grid settings, so one table serves every load and
//! target.

pub mod bracket;
pub mod common;
pub mod map;
pub mod screen;
pub mod verdicts;
pub mod witness;
