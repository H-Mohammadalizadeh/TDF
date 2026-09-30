//! The commands of `tdf`, one module each ([`point`], [`map`], [`distributions`], [`validate`],
//! [`screen`], [`bracket`], [`ojf`], [`sweep`], [`region`], [`directional`]), and the helpers of
//! [`common`]. Every command writes its results under `--out` (a JSON file for `point`, a
//! directory otherwise) and returns a short JSON summary, which the binary prints. A *profile* is
//! a named reconfiguration-time distribution at a given mean and squared coefficient of variation.
//! With `--cache`, the phase coefficients of a model are read from that directory when present and
//! written otherwise; they depend only on the service rates, deadlines, reconfiguration-time
//! distributions and grid settings, so one table serves every load and target.

pub mod bracket;
pub mod common;
pub mod directional;
pub mod distributions;
pub mod map;
pub mod ojf;
pub mod point;
pub mod region;
pub mod screen;
pub mod sweep;
pub mod validate;
