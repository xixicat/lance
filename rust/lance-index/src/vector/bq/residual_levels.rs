// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Greedy residual 1-bit codes in one shared Fast rotation.
//!
//! Residuals stay in the rotated space. Each level stores a packed `±1` code,
//! the coordinate scale `alpha = ||e||_1 / d`, the residual norm, and the inner
//! product of that residual with the original rotated vector.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use lance_core::{Error, Result};

use super::rotation::{apply_fast_rotation, fast_rotation_signs_len, random_fast_rotation_signs};

/// One residual 1-bit level in the already rotated space.
#[derive(Clone, Debug)]
pub struct LevelCode {
    /// LSB-first packed signs. Bit 1 means `+1`, bit 0 means `-1`.
    pub packed: Vec<u8>,
    pub alpha: f32,
    pub radius: f32,
    pub bias: f32,
}

/// Residual codes for one rotated vector.
#[derive(Clone, Debug)]
pub struct EncodedVector {
    pub dim: usize,
    pub norm_sq: f32,
    pub levels: Vec<LevelCode>,
}

/// Index-wide Fast rotation shared by every vector in a partition.
pub struct ResidualEncoder {
    dim: usize,
    signs: Vec<u8>,
}

impl ResidualEncoder {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            signs: random_fast_rotation_signs(dim),
        }
    }

    pub fn with_signs(dim: usize, signs: Vec<u8>) -> Result<Self> {
        let expected = fast_rotation_signs_len(dim);
        if signs.len() != expected {
            return Err(Error::invalid_input(format!(
                "residual encoder signs len {}, expected {} for dim {}",
                signs.len(),
                expected,
                dim
            )));
        }
        Ok(Self { dim, signs })
    }

    pub fn rotate(&self, vector: &[f32]) -> Result<Vec<f32>> {
        if vector.len() != self.dim {
            return Err(Error::invalid_input(format!(
                "residual encoder dim {}, got vector len {}",
                self.dim,
                vector.len()
            )));
        }
        let mut rotated = vec![0.0; self.dim];
        apply_fast_rotation(vector, &mut rotated, &self.signs);
        Ok(rotated)
    }

    pub fn encode(&self, vector: &[f32], levels: usize, joint: bool) -> Result<EncodedVector> {
        let rotated = self.rotate(vector)?;
        encode_rotated(&rotated, levels, joint)
    }
}

pub fn encode_rotated(rotated: &[f32], levels: usize, joint: bool) -> Result<EncodedVector> {
    validate_levels(levels)?;
    if rotated.is_empty() {
        return Err(Error::invalid_input(
            "residual encoder requires a non-empty rotated vector".to_string(),
        ));
    }

    let dim = rotated.len();
    let norm_sq = dot(rotated, rotated);
    let mut residual = rotated.to_vec();
    let mut pm1 = Vec::with_capacity(levels);
    let mut alphas = Vec::with_capacity(levels);

    for _ in 0..levels {
        let mut signs = vec![1i8; dim];
        let l1 = assign_signs(&residual, &mut signs);
        let alpha = l1 / dim as f32;
        for (dst, &sign) in residual.iter_mut().zip(signs.iter()) {
            *dst -= alpha * f32::from(sign);
        }
        pm1.push(signs);
        alphas.push(alpha);
    }

    if joint {
        refit_joint(rotated, &pm1, &mut alphas);
    }

    let levels = materialize_levels(rotated, &pm1, &alphas);
    Ok(EncodedVector {
        dim,
        norm_sq,
        levels,
    })
}

fn validate_levels(levels: usize) -> Result<()> {
    if !(1..=8).contains(&levels) {
        return Err(Error::invalid_input(format!(
            "residual levels must be in 1..=8, got {levels}"
        )));
    }
    Ok(())
}

fn assign_signs(residual: &[f32], signs: &mut [i8]) -> f32 {
    let mut l1 = 0.0f32;
    for (sign, &value) in signs.iter_mut().zip(residual.iter()) {
        l1 += value.abs();
        *sign = if value.is_sign_negative() { -1 } else { 1 };
    }
    l1
}

fn materialize_levels(rotated: &[f32], pm1: &[Vec<i8>], alphas: &[f32]) -> Vec<LevelCode> {
    let dim = rotated.len();
    let mut levels = Vec::with_capacity(alphas.len());
    let mut residual = rotated.to_vec();
    for (signs, &alpha) in pm1.iter().zip(alphas.iter()) {
        for (dst, &sign) in residual.iter_mut().zip(signs.iter()) {
            *dst -= alpha * f32::from(sign);
        }
        levels.push(LevelCode {
            packed: pack_signs(signs),
            alpha,
            radius: dot(&residual, &residual).sqrt(),
            bias: dot(&residual, rotated),
        });
        debug_assert_eq!(signs.len(), dim);
    }
    levels
}

fn pack_signs(signs: &[i8]) -> Vec<u8> {
    let mut packed = vec![0u8; signs.len().div_ceil(8)];
    for (idx, &sign) in signs.iter().enumerate() {
        if sign > 0 {
            packed[idx / 8] |= 1u8 << (idx % 8);
        }
    }
    packed
}

fn refit_joint(rotated: &[f32], pm1: &[Vec<i8>], alphas: &mut [f32]) {
    let Some(solved) = solve_codes(rotated, pm1) else {
        return;
    };
    for (dst, value) in alphas.iter_mut().zip(solved) {
        *dst = value;
    }
}

// Indexed updates of the Gram matrix are clearer than iterator adapters here.
#[allow(clippy::needless_range_loop)]
fn solve_codes(rotated: &[f32], pm1: &[Vec<i8>]) -> Option<Vec<f32>> {
    let levels = pm1.len();
    let mut gram = vec![vec![0.0f64; levels]; levels];
    let mut proj = vec![0.0f64; levels];
    for (dim, &value) in rotated.iter().enumerate() {
        let value = f64::from(value);
        for i in 0..levels {
            let sign_i = f64::from(pm1[i][dim]);
            proj[i] += sign_i * value;
            for j in i..levels {
                gram[i][j] += sign_i * f64::from(pm1[j][dim]);
            }
        }
    }
    for i in 0..levels {
        for j in 0..i {
            gram[i][j] = gram[j][i];
        }
    }
    solve_gram(&gram, &proj).map(|values| values.into_iter().map(|v| v as f32).collect())
}

// Gauss-Jordan updates both the pivot row and every other row by column index.
#[allow(clippy::needless_range_loop)]
pub(crate) fn solve_gram(gram: &[Vec<f64>], proj: &[f64]) -> Option<Vec<f64>> {
    let n = proj.len();
    if gram.len() != n || gram.iter().any(|row| row.len() != n) || n == 0 || n > 8 {
        return None;
    }
    let mut a = gram.to_vec();
    let mut b = proj.to_vec();
    for col in 0..n {
        let mut pivot = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if a[pivot][col].abs() < 1.0e-8 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let div = a[col][col];
        for value in a[col].iter_mut().skip(col) {
            *value /= div;
        }
        b[col] /= div;
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            for col_j in col..n {
                a[row][col_j] -= factor * a[col][col_j];
            }
            b[row] -= factor * b[col];
        }
    }
    Some(b)
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(l, r)| l * r).sum()
}

pub fn dot_packed_pm1(packed: &[u8], rotated_query: &[f32]) -> f32 {
    rotated_query
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            let bit = packed.get(idx / 8).copied().unwrap_or(0);
            if (bit >> (idx % 8)) & 1 == 1 {
                *value
            } else {
                -value
            }
        })
        .sum()
}

pub fn estimated_l2_sq(rotated_query: &[f32], encoded: &EncodedVector) -> f32 {
    let score = encoded
        .levels
        .iter()
        .map(|level| level.alpha * dot_packed_pm1(&level.packed, rotated_query))
        .sum::<f32>();
    let query_sq = dot(rotated_query, rotated_query);
    encoded.norm_sq + query_sq - 2.0 * score
}

pub fn lower_bound_sq(
    rotated_query: &[f32],
    encoded: &EncodedVector,
    done: usize,
    gamma: f32,
) -> f32 {
    debug_assert!((1..=encoded.levels.len()).contains(&done));
    let query_sq = dot(rotated_query, rotated_query);
    let query_norm = query_sq.sqrt();
    let score = encoded
        .levels
        .iter()
        .take(done)
        .map(|level| level.alpha * dot_packed_pm1(&level.packed, rotated_query))
        .sum::<f32>();
    let level = &encoded.levels[done - 1];
    let cauchy = level.radius * query_norm;
    let sigma = cauchy / (encoded.dim as f32).sqrt();
    let cap = cauchy.min(level.bias + gamma * sigma);
    encoded.norm_sq + query_sq - 2.0 * (score + cap)
}

pub fn threshold_cut_sq(
    est_sq: f32,
    radius_m: f32,
    query_norm: f32,
    dim: usize,
    gamma: f32,
) -> f32 {
    est_sq + 2.0 * gamma * radius_m * query_norm / (dim as f32).sqrt()
}

#[derive(Clone, Debug)]
pub struct SearchHit {
    pub id: usize,
    pub est_sq: f32,
}

#[derive(Clone, Debug)]
pub struct SearchStats {
    pub hits: Vec<SearchHit>,
    pub pruned: usize,
}

pub fn search_partition(
    rotated_query: &[f32],
    encoded: &[EncodedVector],
    k: usize,
    gamma: f32,
    prune: bool,
) -> SearchStats {
    #[derive(Clone, Copy)]
    struct HeapItem {
        est_sq: f32,
        radius_m: f32,
        id: usize,
    }

    impl PartialEq for HeapItem {
        fn eq(&self, other: &Self) -> bool {
            self.id == other.id && self.est_sq.to_bits() == other.est_sq.to_bits()
        }
    }
    impl Eq for HeapItem {}
    impl PartialOrd for HeapItem {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for HeapItem {
        fn cmp(&self, other: &Self) -> Ordering {
            self.est_sq
                .total_cmp(&other.est_sq)
                .then_with(|| self.id.cmp(&other.id))
        }
    }

    let query_norm = dot(rotated_query, rotated_query).sqrt();
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
    let mut pruned = 0usize;
    for (id, row) in encoded.iter().enumerate() {
        if row.dim != rotated_query.len() || row.levels.is_empty() {
            continue;
        }
        let mut score = 0.0f32;
        let mut dropped = false;
        for done in 1..=row.levels.len() {
            let level = &row.levels[done - 1];
            score += level.alpha * dot_packed_pm1(&level.packed, rotated_query);
            if prune
                && heap.len() >= k
                && let Some(worst) = heap.peek()
            {
                let cauchy = level.radius * query_norm;
                let sigma = cauchy / (row.dim as f32).sqrt();
                let cap = cauchy.min(level.bias + gamma * sigma);
                let lower = row.norm_sq + query_norm * query_norm - 2.0 * (score + cap);
                let cut =
                    threshold_cut_sq(worst.est_sq, worst.radius_m, query_norm, row.dim, gamma);
                if lower > cut {
                    dropped = true;
                    break;
                }
            }
        }
        if dropped {
            pruned += 1;
            continue;
        }
        let est_sq = row.norm_sq + query_norm * query_norm - 2.0 * score;
        let radius_m = row.levels.last().map(|level| level.radius).unwrap_or(0.0);
        let item = HeapItem {
            est_sq,
            radius_m,
            id,
        };
        if heap.len() < k {
            heap.push(item);
        } else if heap.peek().is_some_and(|worst| item < *worst) {
            heap.pop();
            heap.push(item);
        }
    }

    let mut hits: Vec<SearchHit> = heap
        .into_iter()
        .map(|item| SearchHit {
            id: item.id,
            est_sq: item.est_sq,
        })
        .collect();
    hits.sort_by(|left, right| {
        left.est_sq
            .total_cmp(&right.est_sq)
            .then_with(|| left.id.cmp(&right.id))
    });
    SearchStats { hits, pruned }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<f32> {
        vec![0.5, -1.25, 0.0, 2.0, -0.75, 0.25, 3.5, -2.0]
    }

    fn unpack_signs(packed: &[u8], dim: usize) -> Vec<i8> {
        (0..dim)
            .map(|idx| {
                let bit = packed.get(idx / 8).copied().unwrap_or(0);
                if (bit >> (idx % 8)) & 1 == 1 { 1 } else { -1 }
            })
            .collect()
    }

    #[test]
    fn greedy_residual_is_orthogonal_to_its_sign() {
        let original = sample();
        let encoded = encode_rotated(&original, 3, false).unwrap();
        assert_eq!(encoded.levels.len(), 3);
        let mut residual = original.clone();
        for (level_idx, level) in encoded.levels.iter().enumerate() {
            let signs = unpack_signs(&level.packed, original.len());
            for (dst, sign) in residual.iter_mut().zip(signs.iter()) {
                *dst -= level.alpha * f32::from(*sign);
            }
            let alignment: f32 = residual
                .iter()
                .zip(signs.iter())
                .map(|(value, sign)| value * f32::from(*sign))
                .sum();
            assert!(
                alignment.abs() <= 1.0e-4,
                "level {level_idx} residual is not orthogonal to its sign: {alignment}"
            );
            let bias: f32 = residual
                .iter()
                .zip(original.iter())
                .map(|(l, r)| l * r)
                .sum();
            assert!(
                (bias - level.bias).abs() <= 1.0e-4,
                "level {level_idx} bias {} != {}",
                level.bias,
                bias
            );
            let radius = residual
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            assert!((radius - level.radius).abs() <= 1.0e-4);
        }
        // The first subtraction removes a vector parallel to its sign, so that
        // residual is orthogonal to the whole prefix reconstruction.
        let first = &encoded.levels[0];
        let gap = (first.bias - first.radius * first.radius).abs();
        assert!(
            gap <= 1.0e-3 * (1.0 + first.radius * first.radius),
            "bias {} radius {} gap {}",
            first.bias,
            first.radius,
            gap
        );
    }

    #[test]
    fn zero_vector_has_zero_coefficients() {
        let encoded = encode_rotated(&[0.0; 8], 2, false).unwrap();
        assert_eq!(encoded.norm_sq, 0.0);
        for level in &encoded.levels {
            assert_eq!(level.alpha, 0.0);
            assert_eq!(level.radius, 0.0);
            assert_eq!(level.bias, 0.0);
        }
    }

    #[test]
    fn rejects_level_count_outside_1_to_8() {
        let error = encode_rotated(&sample(), 0, false).unwrap_err();
        assert!(error.to_string().contains("levels"));
        let error = encode_rotated(&sample(), 9, false).unwrap_err();
        assert!(error.to_string().contains("levels"));
    }

    #[test]
    fn joint_refit_does_not_increase_final_residual() {
        let encoded_greedy = encode_rotated(&sample(), 4, false).unwrap();
        let encoded_joint = encode_rotated(&sample(), 4, true).unwrap();
        let greedy = encoded_greedy.levels.last().unwrap().radius;
        let joint = encoded_joint.levels.last().unwrap().radius;
        assert!(
            joint <= greedy + 1.0e-3,
            "joint radius {joint} exceeded greedy radius {greedy}"
        );
        // Four signs in dimension 8 are independent here, so least squares
        // leaves the final residual orthogonal to every selected sign.
        let final_level = encoded_joint.levels.last().unwrap();
        let gap = (final_level.bias - final_level.radius * final_level.radius).abs();
        assert!(
            gap <= 1.0e-3 * (1.0 + final_level.radius * final_level.radius),
            "joint bias {} radius {}",
            final_level.bias,
            final_level.radius
        );

        let encoded_greedy = encode_rotated(&sample(), 8, false).unwrap();
        let encoded_joint = encode_rotated(&sample(), 8, true).unwrap();
        let greedy = encoded_greedy.levels.last().unwrap().radius;
        let joint = encoded_joint.levels.last().unwrap().radius;
        assert!(
            joint <= greedy + 1.0e-3,
            "8-level joint radius {joint} exceeded greedy radius {greedy}"
        );
    }

    #[test]
    fn singular_gram_returns_none() {
        let gram = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        assert!(solve_gram(&gram, &[1.0, 1.0]).is_none());
    }

    #[test]
    fn self_query_lower_bound_stays_non_positive() {
        let rotated = sample();
        let encoded = encode_rotated(&rotated, 3, true).unwrap();
        for done in 1..=3 {
            for gamma in [0.0, 3.0, (rotated.len() as f32).sqrt()] {
                let bound = lower_bound_sq(&rotated, &encoded, done, gamma);
                assert!(bound <= 1.0e-2, "done {done} gamma {gamma} bound {bound}");
            }
        }
    }

    #[test]
    fn safe_gamma_matches_cauchy_cap() {
        let dim = 8usize;
        let radius = 0.5f32;
        let query_norm = 2.0f32;
        let cut = threshold_cut_sq(1.0, radius, query_norm, dim, (dim as f32).sqrt());
        let expected = 1.0 + 2.0 * radius * query_norm;
        assert!(
            (cut - expected).abs() < 1.0e-5,
            "cut {cut} expected {expected}"
        );
    }

    #[test]
    fn packed_pm1_dot_matches_explicit_signs() {
        let query = sample();
        let signs = [1i8, -1, 1, 1, -1, -1, 1, -1];
        let mut packed = vec![0u8; 1];
        for (idx, sign) in signs.iter().enumerate() {
            if *sign > 0 {
                packed[0] |= 1u8 << idx;
            }
        }
        let expected: f32 = query
            .iter()
            .zip(signs.iter())
            .map(|(value, sign)| value * f32::from(*sign))
            .sum();
        let got = dot_packed_pm1(&packed, &query);
        assert!(
            (got - expected).abs() < 1.0e-5,
            "got {got} expected {expected}"
        );
    }

    #[test]
    fn recall_at_10_on_gaussian_rows_meets_floor() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand_distr::{Distribution, StandardNormal};

        let dim = 32usize;
        let rows = 80usize;
        let queries = 8usize;
        let k = 10usize;
        let mut rng = StdRng::seed_from_u64(42);
        let normal = StandardNormal;
        let encoder = ResidualEncoder::new(dim);
        let base: Vec<Vec<f32>> = (0..rows)
            .map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect())
            .collect();
        let encoded: Vec<_> = base
            .iter()
            .map(|row| encoder.encode(row, 4, true).unwrap())
            .collect();
        let mut recall = 0.0f32;
        for _ in 0..queries {
            let query: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
            let rotated_query = encoder.rotate(&query).unwrap();
            let rotated_base: Vec<Vec<f32>> = base
                .iter()
                .map(|row| encoder.rotate(row).unwrap())
                .collect();
            let mut exact: Vec<(f32, usize)> = rotated_base
                .iter()
                .enumerate()
                .map(|(id, row)| {
                    let dist = row
                        .iter()
                        .zip(rotated_query.iter())
                        .map(|(left, right)| {
                            let diff = left - right;
                            diff * diff
                        })
                        .sum::<f32>();
                    (dist, id)
                })
                .collect();
            exact.sort_by(|left, right| {
                left.0
                    .total_cmp(&right.0)
                    .then_with(|| left.1.cmp(&right.1))
            });
            let truth: std::collections::HashSet<usize> =
                exact.iter().take(k).map(|(_, id)| *id).collect();
            let stats = search_partition(&rotated_query, &encoded, k, dim as f32, false);
            let hit = stats
                .hits
                .iter()
                .filter(|hit| truth.contains(&hit.id))
                .count();
            recall += hit as f32 / k as f32;
        }
        recall /= queries as f32;
        assert!(
            recall >= 0.5,
            "gaussian residual-level recall@10 was {recall}, floor is 0.5"
        );
    }
}
