//! Window constraints of the relaxation (window rows, in identifiers): the
//! minimum reconfiguration rate that the deadlines impose. They are generated
//! for a declared set of window lengths and class subsets (or for every
//! subset, for comparison), each with a certified lower bound as right-hand
//! side; [`WindowRow`] states the constraint and its derivation.

use super::model::DiscreteModel;
use super::rational::{Q, exp_neg};
use num_traits::{One, Zero};
use std::collections::HashMap;

/// One window constraint `h sum nu + sum_{i in S} lambda_i a_i x_i >= rhs`.
///
/// In a window of length `h`, a class-`i` job that arrives in the first
/// `a_i = (h - d_i)^+` time units must complete within the window to meet
/// its deadline; the number of such jobs is Poisson with mean
/// `lambda_i a_i`. If `K` classes of the subset `S` have such a job and `M`
/// of these jobs miss, at least `K - M` configurations are visited in the
/// window, which takes at least `(K - 1)^+ - M` reconfigurations. Averaging
/// over time gives, for the total reconfiguration rate `nu` and the miss
/// probabilities `x_i`,
///
/// ```text
/// h nu + sum_{i in S} lambda_i a_i x_i
///     >= sum_{i in S} (1 - exp(-lambda_i a_i)) - 1 + exp(-sum_{i in S} lambda_i a_i).
/// ```
#[derive(Clone, Debug)]
pub struct WindowRow {
    /// Window length.
    pub h: Q,
    /// The class subset `S`.
    pub subset: Vec<usize>,
    /// `lambda_i a_i` for every class (zero outside the subset).
    pub lam_a: Vec<Q>,
    /// A lower bound on
    /// `sum_{i in S} (1 - exp(-lambda_i a_i)) - 1 + exp(-sum_{i in S} lambda_i a_i)`.
    pub rhs: Q,
}

/// The declared window lengths: every deadline plus each of the offsets
/// 1/4, 1/2, 1, 3/2, 2, 3, 4, 6, 8, and the fixed length 5.
pub fn default_windows(model: &DiscreteModel) -> Vec<Q> {
    let offsets = ["1/4", "1/2", "1", "3/2", "2", "3", "4", "6", "8"];
    let mut out: Vec<Q> = vec![num_traits::FromPrimitive::from_i64(5).unwrap()];
    for d in &model.d {
        for o in offsets {
            out.push(d + super::rational::parse_q(o).unwrap());
        }
    }
    out.sort();
    out.dedup();
    out
}

/// The declared subsets: all active classes, every pair of active classes
/// and, with more than three active classes, every leave-one-out set.
/// Subsets of fewer than two classes give vacuous constraints and are omitted.
pub fn default_subsets(model: &DiscreteModel) -> Vec<Vec<usize>> {
    let active: Vec<usize> = (0..model.n).filter(|i| model.active(*i)).collect();
    let mut out = vec![active.clone()];
    for a in 0..active.len() {
        for b in a + 1..active.len() {
            out.push(vec![active[a], active[b]]);
        }
    }
    if active.len() > 3 {
        for a in 0..active.len() {
            out.push(
                active
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| *k != a)
                    .map(|(_, i)| *i)
                    .collect(),
            );
        }
    }
    out.sort();
    out.dedup();
    out.retain(|s| s.len() >= 2);
    out
}

/// Every subset of at least two active classes (`2^k - k - 1` subsets for
/// `k` active classes), to compare against the declared subsets.
pub fn all_subsets(model: &DiscreteModel) -> Vec<Vec<usize>> {
    let active: Vec<usize> = (0..model.n).filter(|i| model.active(*i)).collect();
    let k = active.len();
    let mut out = Vec::new();
    for mask in 1u64..(1u64 << k) {
        if mask.count_ones() < 2 {
            continue;
        }
        out.push(
            (0..k)
                .filter(|b| mask & (1 << b) != 0)
                .map(|b| active[b])
                .collect(),
        );
    }
    out.sort();
    out
}

/// The window constraints of `model` for every window length and subset.
/// A constraint whose right-hand side is not positive is vacuous (all its
/// terms are nonnegative) and dropped. Every exponential is enclosed with
/// `bits` fractional bits, and the end that lowers the right-hand side is used.
pub fn window_rows(
    model: &DiscreteModel,
    windows: &[Q],
    subsets: &[Vec<usize>],
    bits: u32,
) -> Vec<WindowRow> {
    let mut cache: HashMap<Q, (Q, Q)> = HashMap::new();
    let mut enclose = |x: &Q| -> (Q, Q) {
        cache
            .entry(x.clone())
            .or_insert_with(|| exp_neg(x, bits))
            .clone()
    };
    let mut rows = Vec::new();
    for h in windows {
        let lam_a: Vec<Q> = (0..model.n)
            .map(|i| {
                let a = h - &model.d[i];
                if a > Q::zero() {
                    &model.lam[i] * a
                } else {
                    Q::zero()
                }
            })
            .collect();
        for subset in subsets {
            let mut sum_q = Q::zero();
            let mut total = Q::zero();
            let mut row_lam_a = vec![Q::zero(); model.n];
            for &i in subset {
                if lam_a[i].is_zero() {
                    continue;
                }
                let (_, hi) = enclose(&lam_a[i]);
                sum_q += Q::one() - hi;
                total += &lam_a[i];
                row_lam_a[i] = lam_a[i].clone();
            }
            if total.is_zero() {
                continue;
            }
            let (lo, _) = enclose(&total);
            let rhs = sum_q - Q::one() + lo;
            if rhs <= Q::zero() {
                continue;
            }
            rows.push(WindowRow {
                h: h.clone(),
                subset: subset.clone(),
                lam_a: row_lam_a,
                rhs,
            });
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::super::model::two_class_example;
    use super::super::rational::{qi, to_f64};
    use super::*;

    #[test]
    fn window_rows_of_the_two_class_example_reproduce_the_counting_argument() {
        let m = two_class_example();
        let rows = window_rows(&m, &[qi(5)], &[vec![0, 1]], 128);
        assert_eq!(rows.len(), 1);
        let q = (1.0 - (-2.0f64).exp()).powi(2);
        assert!((to_f64(&rows[0].rhs) - q).abs() < 1e-14);
        assert_eq!(rows[0].lam_a, vec![qi(2), qi(2)]);
        // a window of length at most the deadline gives no constraint
        assert!(window_rows(&m, &[qi(3)], &[vec![0, 1]], 128).is_empty());
    }
}
