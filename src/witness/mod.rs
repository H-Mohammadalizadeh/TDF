//! The sufficient side of the feasibility test: a certified upper bound on the miss
//! probability of every class under one admissible policy, classical exhaustive cyclic polling
//! (EC, the witness policy). The network visits the configurations in a fixed cyclic order, serves
//! each class until its queue is empty, then reconfigures to the next configuration even when
//! that queue is empty, and never idles. A setting is certified feasible when the bound `hi` of
//! every class is at or below its target.
//!
//! With exponential service, the Fuhrmann-Cooper decomposition gives
//! `m_i = 1 - E[u_i(I_i)] / E[I_i]`, where `I_i` is the stationary intervisit time of class `i`
//! and `u_i(x) = W_i(min(x, d_i))` is nondecreasing and concave, so replacing `I_i` by any
//! variable below it in the increasing-concave order (with the exact mean in the denominator)
//! can only raise the result. [`bound`] builds such a variable on the lattice `g Z` from the
//! descendant-set form of `I_i` (truncation to `K` past cycles, deletions and mean-preserving
//! spreads) and encloses its lattice law in binary64 with directed rounding ([`fp`]); the
//! reconfiguration-time laws and their spreads come from [`laws`], the constants and the final
//! assembly are exact rationals ([`crate::feasibility::rational`],
//! [`crate::feasibility::special`]). The upper end `hi >= m_i` holds for every lattice step,
//! truncation depth and cap; these choices only affect its distance to `m_i`. The lower end `lo`
//! adds the deleted means and a second-order bound of the spreads, so `[lo, hi]` brackets the
//! miss probability of the witness policy.

pub mod bound;
pub mod fp;
pub mod laws;

pub use bound::{Opts, Out, System, compute, default_step};
pub use laws::Law;
