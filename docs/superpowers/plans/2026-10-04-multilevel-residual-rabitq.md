# Multi-level Residual RaBitQ Spike Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `IVF_MRQ` index that quantizes IVF residuals with one Fast rotation and 1–8 greedy 1-bit residual levels, and compare its recall with `IVF_RQ` at the same level / `num_bits` budget.

**Architecture:** The math lives in `lance-index` as `residual_levels.rs` and does not change `RabitQuantizer`. `IVF_MRQ` is a new `IndexType` (`IvfMrq = 108`) and a new quantizer, wired through the same IVF builder as `IVF_RQ`. `IVF_RQ` files stay readable and writable exactly as they are today. A fixed-seed example prints the algorithm table. A dataset test builds both indexes on the same rows and records recall.

**Tech Stack:** Rust, `lance-index` crate, existing `vector/bq/rotation.rs`, `rand` 0.9, `rand_distr`.

## Global Constraints

- One shared Fast rotation per encoder. Do not rotate again at each level.
- Residual update is `e_k = e_{k-1} - α_k * sign(e_{k-1})` with `α_k = ||e_{k-1}||_1 / d` and `sign(0) = +1`.
- Level count is `1..=8`. `1` is the baseline inside this encoder. It is not IVF_RQ `num_bits`.
- Distance is squared L2 in the rotated space: `N_sq + ||q_r||^2 - 2 S`.
- Joint least squares is on by default for the spike binary. A singular `BᵀB` keeps the greedy coefficients.
- Store per level: packed `±1` code, `alpha`, `radius = ||e_k||_2`, `bias = <e_k, o_r>`.
- Candidate cap: `U_k = min(R_k L_q, b_k + γ R_k L_q / sqrt(d))`.
- Heap cut: `D_cut^2 = D_est^2(A) + 2 γ R_m(A) L_q / sqrt(d)`, using A's own final radius.
- `γ = sqrt(d)` is the Cauchy safe setting. `γ = 3` is an aggressive setting, not a 99.7% recall guarantee.
- `IVF_MRQ` is a new index. Do not reuse `num_bits`, `__blocked_ex_codes`, or `RabitQuantizer` storage for it.
- `IVF_RQ` behavior stays unchanged. Match arms grow a new variant; they do not change existing `Rabit` arms.
- `IndexType::IvfMrq` discriminant is `108`. Its format version is `3`. `IvfRq` stays version `2`.
- Python parameter is `levels`, default `4`. Passing `num_bits` to `IVF_MRQ` is an error.
- Comparison budget is `IVF_MRQ levels = m` against `IVF_RQ num_bits = m` for `m ∈ {1, 4, 8}`. Both must reach recall@10 `>= 0.5`. `IVF_MRQ` must not fall more than `0.05` below `IVF_RQ` at the same `m`.
- Code comments and names are English. Tests assert; the example binary is the only place that prints the table.
- Power-of-two dimensions in tests (`8`, `32`, `128`). The example may also run `768`.

---

## File structure

- Create `rust/lance-index/src/vector/bq/residual_levels.rs`. Encoding, joint refit, rotated L2 estimate, lower bound, partition search.
- Create `rust/lance-index/src/vector/mrq/mod.rs`. `MrqQuantizer`, metadata, and storage. This is the IVF quantizer. It calls `residual_levels`, not `RabitQuantizer`.
- Modify `rust/lance-index/src/vector/bq.rs`. Add `pub mod residual_levels;`.
- Modify `rust/lance-index/src/vector/quantizer.rs`. Add `QuantizationType::Mrq` and `Quantizer::Mrq`.
- Modify `rust/lance-index-core/src/lib.rs`. Add `IndexType::IvfMrq = 108`.
- Modify `protos/index.proto`. Add `MultiResidualQuantization` as a new `compression` arm.
- Modify `rust/lance/src/index/vector.rs`. Add `StageParams::MRQ` and `VectorIndexParams::ivf_mrq`.
- Modify `python/src/dataset.rs`. Accept `index_type="IVF_MRQ"` and kwargs `levels`.
- Create `rust/lance-index/examples/residual_levels_spike.rs`. Algorithm table only.
- Modify `python/python/tests/test_vector_index.py`. Recall comparison against `IVF_RQ`.
- Do not modify `RabitQuantizer` math, `num_bits` validation, or the IVF_RQ auxiliary column set.

---

### Task 1: Greedy residual encoder

**Files:**
- Create: `rust/lance-index/src/vector/bq/residual_levels.rs`
- Modify: `rust/lance-index/src/vector/bq.rs` (the `pub mod` list next to `pub mod rotation;`)
- Test: `rust/lance-index/src/vector/bq/residual_levels.rs` (`mod tests`)

**Interfaces:**
- Consumes: `crate::vector::bq::rotation::{apply_fast_rotation, fast_rotation_signs_len, random_fast_rotation_signs}`
- Produces:
  - `ResidualEncoder::new(dim: usize) -> Self`
  - `ResidualEncoder::encode(&self, vector: &[f32], levels: usize, joint: bool) -> Result<EncodedVector>`
  - `encode_rotated(rotated: &[f32], levels: usize, joint: bool) -> Result<EncodedVector>`
  - `EncodedVector { dim: usize, norm_sq: f32, levels: Vec<LevelCode> }`
  - `LevelCode { packed: Vec<u8>, alpha: f32, radius: f32, bias: f32 }`

- [ ] **Step 1: Write the failing test**

Add this module declaration to `rust/lance-index/src/vector/bq.rs` immediately after `pub mod rotation;`:

```rust
pub mod residual_levels;
```

Create `rust/lance-index/src/vector/bq/residual_levels.rs` with only the test and a stub type so the test compiles after step 3. For step 1, write the test file contents below and run it; the expected failure is `cannot find function encode_rotated`.

```rust
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<f32> {
        vec![0.5, -1.25, 0.0, 2.0, -0.75, 0.25, 3.5, -2.0]
    }

    #[test]
    fn greedy_residual_is_orthogonal_to_its_sign() {
        let encoded = encode_rotated(&sample(), 3, false).unwrap();
        assert_eq!(encoded.levels.len(), 3);
        for level in &encoded.levels {
            let gap = (level.bias - level.radius * level.radius).abs();
            assert!(
                gap <= 1.0e-3 * (1.0 + level.radius * level.radius),
                "bias {} radius {} gap {}",
                level.bias,
                level.radius,
                gap
            );
        }
    }

    #[test]
    fn zero_vector_has_zero_coefficients() {
        let encoded = encode_rotated(&vec![0.0; 8], 2, false).unwrap();
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
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p lance-index rejects_level_count_outside_1_to_8 --lib`

Expected: FAIL because `encode_rotated` is not defined.

- [ ] **Step 3: Write the greedy encoder**

Replace the file with the implementation below. `joint = true` is accepted and behaves as greedy until Task 2 replaces `refit_joint`.

```rust
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

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
            *dst -= alpha * f32::from(*sign);
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

fn refit_joint(_rotated: &[f32], _pm1: &[Vec<i8>], _alphas: &mut [f32]) {}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(l, r)| l * r).sum()
}
```

Keep the `mod tests` block from Step 1 at the bottom of the same file.

- [ ] **Step 4: Run the tests**

Run: `cargo test -p lance-index residual_levels --lib`

Expected: `greedy_residual_is_orthogonal_to_its_sign`, `zero_vector_has_zero_coefficients`, and `rejects_level_count_outside_1_to_8` PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/lance-index/src/vector/bq.rs rust/lance-index/src/vector/bq/residual_levels.rs
git commit -m "feat(index): encode greedy residual 1-bit levels"
```

---

### Task 2: Joint least-squares refit

**Files:**
- Modify: `rust/lance-index/src/vector/bq/residual_levels.rs`
- Test: same file

**Interfaces:**
- Consumes: `encode_rotated`, `LevelCode::radius`
- Produces: `pub(crate) fn solve_gram(gram: &[Vec<f64>], proj: &[f64]) -> Option<Vec<f64>>`. `encode_rotated(..., joint: true)` replaces greedy `alpha` when the gram matrix is invertible, then recomputes `radius` and `bias` from the original rotated vector.

- [ ] **Step 1: Write the failing test**

Append inside `mod tests`:

```rust
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
    }

    #[test]
    fn singular_gram_returns_none() {
        let gram = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        assert!(solve_gram(&gram, &[1.0, 1.0]).is_none());
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p lance-index joint_refit_does_not_increase_final_residual --lib`

Expected: FAIL because `solve_gram` is not defined and joint refit is a no-op. If the no-op happens to pass the radius assertion, `singular_gram_returns_none` still fails to compile.

- [ ] **Step 3: Implement the 4×4 refit**

Replace `refit_joint` and add `solve_gram` above `dot`:

```rust
fn refit_joint(rotated: &[f32], pm1: &[Vec<i8>], alphas: &mut [f32]) {
    let Some(solved) = solve_codes(rotated, pm1) else {
        return;
    };
    for (dst, value) in alphas.iter_mut().zip(solved) {
        *dst = value;
    }
}

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
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p lance-index residual_levels --lib`

Expected: PASS, including `joint_refit_does_not_increase_final_residual` and `singular_gram_returns_none`. Also run `encode_rotated(&sample(), 8, true)` from an extra assertion in `joint_refit_does_not_increase_final_residual`: the 8-level joint radius is `<=` the 8-level greedy radius plus `1e-3`. The gram solver must accept `n = 8`.

- [ ] **Step 5: Commit**

```bash
git add rust/lance-index/src/vector/bq/residual_levels.rs
git commit -m "feat(index): refit residual level coefficients by least squares"
```

---

### Task 3: Estimate, lower bound, and partition search

**Files:**
- Modify: `rust/lance-index/src/vector/bq/residual_levels.rs`
- Test: same file

**Interfaces:**
- Consumes: `EncodedVector`, `LevelCode`
- Produces:
  - `dot_packed_pm1(packed: &[u8], rotated_query: &[f32]) -> f32`
  - `estimated_l2_sq(rotated_query: &[f32], encoded: &EncodedVector) -> f32`
  - `lower_bound_sq(rotated_query: &[f32], encoded: &EncodedVector, done: usize, gamma: f32) -> f32`
  - `threshold_cut_sq(est_sq: f32, radius_m: f32, query_norm: f32, dim: usize, gamma: f32) -> f32`
  - `search_partition(rotated_query: &[f32], encoded: &[EncodedVector], k: usize, gamma: f32, prune: bool) -> SearchStats`
  - `SearchStats { hits: Vec<SearchHit>, pruned: usize }`
  - `SearchHit { id: usize, est_sq: f32 }`

- [ ] **Step 1: Write the failing test**

Append inside `mod tests`:

```rust
    #[test]
    fn self_query_lower_bound_stays_non_positive() {
        let rotated = sample();
        let encoded = encode_rotated(&rotated, 3, true).unwrap();
        for done in 1..=3 {
            for gamma in [0.0, 3.0, (rotated.len() as f32).sqrt()] {
                let bound = lower_bound_sq(&rotated, &encoded, done, gamma);
                assert!(
                    bound <= 1.0e-2,
                    "done {done} gamma {gamma} bound {bound}"
                );
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
        assert!((cut - expected).abs() < 1.0e-5, "cut {cut} expected {expected}");
    }

    #[test]
    fn packed_pm1_dot_matches_explicit_signs() {
        let query = sample();
        let signs = vec![1i8, -1, 1, 1, -1, -1, 1, -1];
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
        assert!((got - expected).abs() < 1.0e-5, "got {got} expected {expected}");
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p lance-index self_query_lower_bound_stays_non_positive --lib`

Expected: FAIL because `lower_bound_sq` is not defined.

- [ ] **Step 3: Implement scoring and search**

Add this block after `dot`:

```rust
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
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

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
            if prune && heap.len() >= k {
                let worst = heap.peek().expect("heap has k items");
                let cap = {
                    let cauchy = level.radius * query_norm;
                    let sigma = cauchy / (row.dim as f32).sqrt();
                    cauchy.min(level.bias + gamma * sigma)
                };
                let lower = row.norm_sq + query_norm * query_norm - 2.0 * (score + cap);
                let cut = threshold_cut_sq(worst.est_sq, worst.radius_m, query_norm, row.dim, gamma);
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
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p lance-index residual_levels --lib`

Expected: PASS. `self_query_lower_bound_stays_non_positive` is the fusion check: with `bias` included, a query equal to the stored vector has a non-positive lower bound at every prefix for `γ` of `0`, `3`, and `sqrt(d)`.

- [ ] **Step 5: Commit**

```bash
git add rust/lance-index/src/vector/bq/residual_levels.rs
git commit -m "feat(index): score and prune residual 1-bit levels"
```

---

### Task 4: Recall unit gate

**Files:**
- Modify: `rust/lance-index/src/vector/bq/residual_levels.rs` tests only
- Test: `recall_at_10_on_gaussian_rows_meets_floor`

**Interfaces:**
- Consumes: `ResidualEncoder`, `search_partition`, `encode`
- Produces: one deterministic recall assertion used as the fast CI gate. The wider table remains in the example.

- [ ] **Step 1: Write the failing test**

Append inside `mod tests`:

```rust
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
            exact.sort_by(|left, right| left.0.total_cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
            let truth: std::collections::HashSet<usize> =
                exact.iter().take(k).map(|(_, id)| *id).collect();
            let stats = search_partition(&rotated_query, &encoded, k, dim as f32, false);
            let hit = stats.hits.iter().filter(|hit| truth.contains(&hit.id)).count();
            recall += hit as f32 / k as f32;
        }
        recall /= queries as f32;
        assert!(
            recall >= 0.5,
            "gaussian residual-level recall@10 was {recall}, floor is 0.5"
        );
    }
```

The `0.5` floor is the repository vector-index recall floor. It is a CI tripwire, not the puncture target. The example in Task 5 prints the number that the follow-up decision uses.

- [ ] **Step 2: Run the test**

Run: `cargo test -p lance-index recall_at_10_on_gaussian_rows_meets_floor --lib`

Expected: PASS with the assertion message showing the recall if it fails. If it fails because recall is below `0.5`, stop. Do not start Task 5's gate loosening and do not start the IVF follow-up. Record the recall and treat the algorithm as rejected at this setting.

- [ ] **Step 3: No production code changes if the test passes**

This task only adds the test. If the test fails for a mechanical reason (dimension mismatch, empty hits), fix `search_partition` or the test setup and re-run. Do not lower the floor.

- [ ] **Step 4: Commit**

```bash
git add rust/lance-index/src/vector/bq/residual_levels.rs
git commit -m "test(index): gate residual-level recall on gaussian rows"
```

---

### Task 5: Puncture example

**Files:**
- Create: `rust/lance-index/examples/residual_levels_spike.rs`

**Interfaces:**
- Consumes: `ResidualEncoder`, `search_partition`, `lower_bound_sq`, `encode_rotated` from `lance_index::vector::bq::residual_levels`
- Produces: a process that prints one TSV table and exits `0` only when every gate below holds.

Gates, evaluated on the run, not by editing constants after seeing a bad result:

- Copied-row query: `lower_bound_sq` of that row against itself is `<= 1e-2` at every level for `γ ∈ {3, sqrt(d)}`.
- Gaussian `dim=128`, `rows=400`, `queries=40`, `k=10`, `m=8`, joint refit, pruning off: mean recall@10 `>= 0.5`.
- Same data, `m=8` recall `+ 0.02 >= m=1` recall.
- Same data, prune ratio at `γ = sqrt(d)` is `<=` prune ratio at `γ = 3`.

- [ ] **Step 1: Write the example**

```rust
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashSet;
use std::process::ExitCode;

use lance_index::vector::bq::residual_levels::{
    ResidualEncoder, encode_rotated, lower_bound_sq, search_partition,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

struct Dataset {
    name: &'static str,
    dim: usize,
    rows: Vec<Vec<f32>>,
}

fn gaussian(dim: usize, rows: usize, rng: &mut StdRng) -> Dataset {
    let normal = StandardNormal;
    Dataset {
        name: "gaussian",
        dim,
        rows: (0..rows)
            .map(|_| (0..dim).map(|_| normal.sample(rng)).collect())
            .collect(),
    }
}

fn clustered(dim: usize, rows: usize, rng: &mut StdRng) -> Dataset {
    let normal = StandardNormal;
    let centers = 20usize;
    let centers: Vec<Vec<f32>> = (0..centers)
        .map(|_| (0..dim).map(|_| normal.sample(rng)).collect())
        .collect();
    Dataset {
        name: "clustered",
        dim,
        rows: (0..rows)
            .map(|idx| {
                let center = &centers[idx % centers];
                center
                    .iter()
                    .map(|value| value + 0.15 * normal.sample(rng))
                    .collect()
            })
            .collect(),
    }
}

fn recall_for(
    encoder: &ResidualEncoder,
    rows: &[Vec<f32>],
    queries: &[Vec<f32>],
    levels: usize,
    k: usize,
    gamma: f32,
    prune: bool,
) -> (f32, f32) {
    let encoded: Vec<_> = rows
        .iter()
        .map(|row| encoder.encode(row, levels, true).unwrap())
        .collect();
    let mut recall = 0.0f32;
    let mut pruned = 0.0f32;
    for query in queries {
        let rotated_query = encoder.rotate(query).unwrap();
        let mut exact: Vec<(f32, usize)> = rows
            .iter()
            .enumerate()
            .map(|(id, row)| {
                let rotated = encoder.rotate(row).unwrap();
                let dist = rotated
                    .iter()
                    .zip(rotated_query.iter())
                    .map(|(left, right)| {
                        let diff = left - right;
                        diff * diff
                    })
                    .sum();
                (dist, id)
            })
            .collect();
        exact.sort_by(|left, right| left.0.total_cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
        let truth: HashSet<usize> = exact.into_iter().take(k).map(|(_, id)| id).collect();
        let stats = search_partition(&rotated_query, &encoded, k, gamma, prune);
        let hits = stats
            .hits
            .iter()
            .filter(|hit| truth.contains(&hit.id))
            .count();
        recall += hits as f32 / k as f32;
        pruned += stats.pruned as f32 / rows.len() as f32;
    }
    (recall / queries.len() as f32, pruned / queries.len() as f32)
}

fn main() -> ExitCode {
    let mut rng = StdRng::seed_from_u64(42);
    let mut failed = false;
    println!("dataset\tdim\trows\tqueries\tm\tgamma\tprune\trecall_at_10\tprune_ratio");

    let probe = encode_rotated(&vec![0.2, -0.4, 0.6, -0.8, 1.0, -1.2, 0.3, -0.7], 4, true).unwrap();
    let rotated = vec![0.2, -0.4, 0.6, -0.8, 1.0, -1.2, 0.3, -0.7];
    for done in 1..=probe.levels.len() {
        for gamma in [3.0f32, (probe.dim as f32).sqrt()] {
            let bound = lower_bound_sq(&rotated, &probe, done, gamma);
            if bound > 1.0e-2 {
                eprintln!("self-query bound failed done={done} gamma={gamma} bound={bound}");
                failed = true;
            }
        }
    }

    for (dim, rows, queries_n) in [(128usize, 400usize, 40usize), (768, 200, 20)] {
        let data = gaussian(dim, rows, &mut rng);
        let queries: Vec<Vec<f32>> = (0..queries_n)
            .map(|_| {
                (0..dim)
                    .map(|_| rand_distr::StandardNormal.sample(&mut rng))
                    .collect()
            })
            .collect();
        let encoder = ResidualEncoder::new(dim);
        let gamma_safe = (dim as f32).sqrt();
        let mut recall_m1 = 0.0f32;
        let mut recall_m8 = 0.0f32;
        let mut prune_safe = 0.0f32;
        let mut prune_aggressive = 0.0f32;
        for levels in [1usize, 2, 4, 8] {
            let (recall, _) = recall_for(&encoder, &data.rows, &queries, levels, 10, gamma_safe, false);
            println!(
                "gaussian\t{dim}\t{rows}\t{queries_n}\t{levels}\t{gamma_safe:.4}\toff\t{recall:.4}\t0"
            );
            if levels == 1 {
                recall_m1 = recall;
            }
            if levels == 8 {
                recall_m8 = recall;
            }
        }
        for (label, gamma) in [("aggressive", 3.0f32), ("safe", gamma_safe)] {
            let (recall, prune_ratio) =
                recall_for(&encoder, &data.rows, &queries, 8, 10, gamma, true);
            println!(
                "gaussian\t{dim}\t{rows}\t{queries_n}\t8\t{gamma:.4}\t{label}\t{recall:.4}\t{prune_ratio:.4}"
            );
            if label == "safe" {
                prune_safe = prune_ratio;
            } else {
                prune_aggressive = prune_ratio;
            }
        }
        if dim == 128 && recall_m8 < 0.5 {
            eprintln!("gate failed: dim 128 m=8 recall {recall_m8} < 0.5");
            failed = true;
        }
        if dim == 128 && recall_m8 + 0.02 < recall_m1 {
            eprintln!("gate failed: m=8 recall {recall_m8} regressed past m=1 recall {recall_m1}");
            failed = true;
        }
        if dim == 128 && prune_safe > prune_aggressive + 1.0e-6 {
            eprintln!(
                "gate failed: safe prune ratio {prune_safe} exceeded aggressive prune ratio {prune_aggressive}"
            );
            failed = true;
        }

        let clustered = clustered(dim, rows, &mut rng);
        let (recall, _) = recall_for(&encoder, &clustered.rows, &queries, 8, 10, gamma_safe, false);
        println!(
            "clustered\t{dim}\t{rows}\t{queries_n}\t8\t{gamma_safe:.4}\toff\t{recall:.4}\t0"
        );
    }

    if failed { ExitCode::from(1) } else { ExitCode::SUCCESS }
}
```

- [ ] **Step 2: Run the unit tests again**

Run: `cargo test -p lance-index residual_levels --lib`

Expected: PASS. The example is not part of `--lib`.

- [ ] **Step 3: Run the puncture**

Run: `cargo run -p lance-index --example residual_levels_spike --release`

Expected: a TSV header `dataset dim rows queries m gamma prune recall_at_10 prune_ratio`, then rows for gaussian `m ∈ {1, 2, 4, 8}` and both prune settings at dim `128` and `768`, plus clustered `m=8`. Exit code `0` if the dim-128 gates hold. Exit code `1` stops the index work.

Interpret the table this way:

- `m=8` recall at or below `m=1` means extra residual levels are not buying neighbors. Stop before Task 6.
- `prune_ratio` at `γ=3` near `0` means the bound is still too loose. Ship `IVF_MRQ` with `γ = sqrt(d)` only.
- `prune_ratio` at `γ=3` high while its `recall_at_10` falls well below the unpruned `m=8` row means the aggressive bound deletes true neighbors. Do not expose `γ=3` as the default.
- Clustered recall is diagnostic only. It is not a gate.

- [ ] **Step 4: Commit**

```bash
git add rust/lance-index/examples/residual_levels_spike.rs
git commit -m "test(index): add residual-level recall and prune spike"
```

---

### Task 6: Register `IVF_MRQ` without changing `IVF_RQ`

**Files:**
- Modify: `rust/lance-index-core/src/lib.rs`
- Modify: `rust/lance-index/src/lib.rs`
- Modify: `protos/index.proto`
- Test: `rust/lance-index/src/lib.rs` (`test_index_type_try_from_str_covers_all_parseable_variants`)

**Interfaces:**
- Consumes: existing `IndexType::IvfRq = 107`, version `2`
- Produces:
  - `IndexType::IvfMrq = 108`
  - `Display` / `TryFrom<&str>` accept `"IVF_MRQ"`
  - `IndexType::IvfMrq.version() == 3`
  - `IndexType::IvfRq.version()` stays `2`
  - `IVF_MRQ_INDEX_VERSION: u32 = 3`
  - `max_vector_version()` returns `3`
  - Protobuf `VectorIndexDetails.MultiResidualQuantization { uint32 levels = 1; }` as `compression` field `10`

- [ ] **Step 1: Write the failing test**

In `rust/lance-index/src/lib.rs`, extend `test_index_type_try_from_str_covers_all_parseable_variants` with `("IVF_MRQ", IndexType::IvfMrq)` and add:

```rust
    #[test]
    fn test_ivf_mrq_version_does_not_change_ivf_rq() {
        assert_eq!(IndexType::IvfRq.version(), 2);
        assert_eq!(IndexType::IvfMrq.version(), 3);
        assert_eq!(IndexType::max_vector_version(), IVF_MRQ_INDEX_VERSION);
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p lance-index test_ivf_mrq_version_does_not_change_ivf_rq --lib`

Expected: FAIL because `IndexType::IvfMrq` does not exist.

- [ ] **Step 3: Add the variant and the proto arm**

In `IndexType`, after `IvfRq = 107;`:

```rust
    IvfMrq = 108,
```

Add the variant to `Display`, both `TryFrom` impls, `version` (`3`), `target_partition_size` (`4096`, same as `IvfRq`), `max_vector_version`, and `matches_details` (still `VectorIndexDetails`). Update the exhaustive test vectors in `lance-index/src/lib.rs`.

In `protos/index.proto`, inside `VectorIndexDetails`, before the `oneof compression`:

```protobuf
  message MultiResidualQuantization {
    // Number of residual 1-bit levels. Valid range is 1..=8.
    // Absent on old writers. Readers of IVF_MRQ require this field.
    uint32 levels = 1;
  }
```

Inside `oneof compression`, after `FlatCompression flat = 8;`:

```protobuf
    MultiResidualQuantization mrq = 10;
```

Regenerate Rust protobuf with the repo's existing proto build (`cargo check -p lance-index`). Do not edit generated files by hand.

- [ ] **Step 4: Run the tests**

Run: `cargo test -p lance-index test_ivf_mrq_version_does_not_change_ivf_rq --lib`

Expected: PASS. `IndexType::IvfRq` tests still expect version `2`.

- [ ] **Step 5: Commit**

```bash
git add rust/lance-index-core/src/lib.rs rust/lance-index/src/lib.rs protos/index.proto
git commit -m "feat(index): add IVF_MRQ index type"
```

---

### Task 7: `MrqQuantizer` and the IVF build path

**Files:**
- Create: `rust/lance-index/src/vector/mrq/mod.rs`
- Modify: `rust/lance-index/src/vector.rs` (`pub mod mrq;`)
- Modify: `rust/lance-index/src/vector/quantizer.rs`
- Modify: `rust/lance/src/index/vector.rs`
- Modify: `python/src/dataset.rs`

**Interfaces:**
- Consumes: `encode_rotated`, `search_partition`, `ResidualEncoder`, `EncodedVector`
- Produces:
  - `MrqBuildParams { pub levels: u8 }` with `levels` validated in `1..=8`
  - `MrqQuantizer` implementing `Quantization`
  - `Quantizer::Mrq(MrqQuantizer)` and `QuantizationType::Mrq` displayed as `"MRQ"`
  - `StageParams::MRQ(MrqBuildParams)`
  - `VectorIndexParams::ivf_mrq(num_partitions: usize, levels: u8, distance_type: DistanceType) -> Self`
  - `VectorIndexParams::index_type()` returns `IndexType::IvfMrq` when the last stage is `StageParams::MRQ`
  - Python `index_type="IVF_MRQ"` reads kwargs `levels` (default `4`)

Storage columns, written by the quantizer and read back by `try_from_batch`:

| Column | Type | Meaning |
| --- | --- | --- |
| `_rowid` | uint64 | existing IVF row id |
| `__mrq_codes` | fixed-size list of uint8, width `levels * ceil(dim / 8)` | level-major packed signs |
| `__mrq_alpha` | fixed-size list of float32, width `levels` | per-level scale |
| `__mrq_radius` | fixed-size list of float32, width `levels` | `R_k` |
| `__mrq_bias` | fixed-size list of float32, width `levels` | `b_k` |
| `__mrq_norm_sq` | float32 | rotated residual norm squared |

The rotation signs live in quantizer metadata, one vector per index, the same way `RabitQuantizationMetadata.fast_rotation_signs` is shared. Build still runs `ResidualTransform` before quantization, as `IvfIndexBuilder` already does for `RabitQuantizer`.

`VectorStore` distance for a partition calls `search_partition` with `gamma = sqrt(dim)` by default. Do not call `RabitDistCalculator`.

- [ ] **Step 1: Write the failing parameter test**

In `rust/lance/src/index/vector.rs` tests, or a new `mrq` unit test next to `VectorIndexParams::ivf_rq`:

```rust
#[test]
fn ivf_mrq_params_report_mrq_index_type() {
    let params = VectorIndexParams::ivf_mrq(16, 8, DistanceType::L2);
    assert_eq!(params.index_type(), IndexType::IvfMrq);
    let VectorIndexParams { stages, .. } = params;
    match stages.last() {
        Some(StageParams::MRQ(mrq)) => assert_eq!(mrq.levels, 8),
        other => panic!("expected MRQ stage, got {other:?}"),
    }
}

#[test]
fn ivf_mrq_rejects_nine_levels() {
    let error = MrqBuildParams::new(9).unwrap_err();
    assert!(error.to_string().contains("1..=8"));
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p lance ivf_mrq_params_report_mrq_index_type --lib`

Expected: FAIL because `ivf_mrq` is not defined.

- [ ] **Step 3: Implement the quantizer and the parameter plumbing**

`MrqBuildParams::new`:

```rust
impl MrqBuildParams {
    pub fn new(levels: u8) -> Result<Self> {
        if !(1..=8).contains(&levels) {
            return Err(Error::invalid_input(format!(
                "IVF_MRQ levels must be in 1..=8, got {levels}"
            )));
        }
        Ok(Self { levels })
    }
}
```

`VectorIndexParams::ivf_mrq` follows `ivf_rq`, with `StageParams::MRQ` in place of `StageParams::RQ`. `index_type()` gains:

```rust
(2, _, Some(StageParams::MRQ(_))) => IndexType::IvfMrq,
```

The local build match in `build_vector_index` and the distributed match in `IndexType::IvfRq` each gain an `IndexType::IvfMrq` arm that constructs `IvfIndexBuilder::<FlatIndex, MrqQuantizer>` with `MrqBuildParams`. Copy the `IvfRq` arm and change only the quantizer type and the stage enum. Leave the `IvfRq` arm body untouched.

In `python/src/dataset.rs`:

- Add `"IVF_MRQ"` to the vector-index type list around the `"IVF_RQ"` pattern.
- In the params match, add:

```rust
"IVF_MRQ" => {
    let mut levels: u8 = 4;
    if let Some(kwargs) = kwargs
        && let Some(value) = kwargs.get_item("levels")?
    {
        levels = value.extract()?;
    }
    if kwargs.and_then(|kwargs| kwargs.get_item("num_bits").ok()).flatten().is_some() {
        return Err(PyValueError::new_err(
            "IVF_MRQ uses `levels` (1..=8), not `num_bits`",
        ));
    }
    let mrq_params = MrqBuildParams::new(levels)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(Box::new(VectorIndexParams::with_ivf_mrq_params(
        m_type, ivf_params, mrq_params,
    )))
}
```

Quantizer `quantize` rotates the residual batch with the shared signs and calls `encode_rotated(..., joint: true)`. `from_metadata` restores the same signs.

Every `match` on `Quantizer` and `QuantizationType` must compile. Add `Mrq` arms that return the MRQ column or metadata key. Do not change the `Rabit` arm expressions.

- [ ] **Step 4: Run the parameter tests**

Run: `cargo test -p lance ivf_mrq_params_report_mrq_index_type --lib`

Expected: PASS. `cargo check -p lance --tests` is clean, including the new match arms.

- [ ] **Step 5: Commit**

```bash
git add rust/lance-index/src/vector/mrq/mod.rs rust/lance-index/src/vector.rs \
  rust/lance-index/src/vector/quantizer.rs rust/lance/src/index/vector.rs python/src/dataset.rs
git commit -m "feat(index): build IVF_MRQ from residual 1-bit levels"
```

---

### Task 8: Recall comparison against `IVF_RQ`

**Files:**
- Modify: `python/python/tests/test_vector_index.py`
- Modify: `python/python/lance/dataset.py` docstring for `create_index`
- Modify: `docs/src/format/index/vector/index.md` with an `IVF_MRQ` auxiliary schema section

**Interfaces:**
- Consumes: `dataset.create_index(..., index_type="IVF_MRQ", levels=m)` and the existing `IVF_RQ` `num_bits` argument
- Produces: one parametrized test, `test_ivf_mrq_matches_ivf_rq_recall`, and a format note that `IVF_MRQ` is a new index version `3`

- [ ] **Step 1: Write the failing test**

Add to `python/python/tests/test_vector_index.py`, beside `test_create_ivf_rq_index`:

```python
@pytest.mark.parametrize("budget", [1, 4, 8])
def test_ivf_mrq_matches_ivf_rq_recall(tmp_path, budget):
    dim = 32
    rows = 256
    k = 10
    uri = tmp_path / "mrq.lance"
    data = np.random.default_rng(0).standard_normal((rows, dim)).astype(np.float32)
    table = pa.table({"id": pa.array(np.arange(rows)), "vector": pa.array(data)})
    dataset = lance.write_dataset(table, uri)
    queries = data[:8]

    def recall(index_type, **params):
        dataset.create_index(
            "vector",
            index_type,
            num_partitions=4,
            replace=True,
            metric="L2",
            **params,
        )
        hits = 0
        for row in range(len(queries)):
            exact = np.argsort(np.linalg.norm(data - queries[row], axis=1))[:k]
            found = dataset.to_table(
                nearest={
                    "column": "vector",
                    "q": queries[row],
                    "k": k,
                }
            ).column("id").to_pylist()
            hits += len(set(exact.tolist()) & set(found))
        return hits / (len(queries) * k)

    rq_recall = recall("IVF_RQ", num_bits=budget)
    mrq_recall = recall("IVF_MRQ", levels=budget)
    assert rq_recall >= 0.5, rq_recall
    assert mrq_recall >= 0.5, mrq_recall
    assert mrq_recall + 0.05 >= rq_recall, (mrq_recall, rq_recall, budget)
```

Use the dataset helper style already used by `test_create_ivf_rq_index` if that test builds the table differently. Keep the assertions.

- [ ] **Step 2: Run the test to verify it fails**

From `python/`:

Run: `uv run pytest python/tests/test_vector_index.py::test_ivf_mrq_matches_ivf_rq_recall -q`

Expected: FAIL because `IVF_MRQ` is rejected or recall is below the floor. A missing-index-type error fails the task until Task 7 is in the same build.

- [ ] **Step 3: Document the format and keep the test as the comparison gate**

In `docs/src/format/index/vector/index.md`, add an `IVF_MRQ` auxiliary schema with the six columns from Task 7. State that `levels` is `1..=8`, version is `3`, and `IVF_RQ` readers do not parse these columns.

In the `create_index` docstring, document `levels` for `IVF_MRQ` the same way `num_bits` is documented for `IVF_RQ`. Default `4`.

No production search change belongs in this step unless the test shows the index cannot be queried. Default search `γ` stays `sqrt(dim)`.

- [ ] **Step 4: Run the comparison**

Run: `uv run pytest python/tests/test_vector_index.py::test_ivf_mrq_matches_ivf_rq_recall -q`

Expected: PASS for budgets `1`, `4`, and `8`. If `mrq_recall + 0.05 < rq_recall`, stop and fix the quantizer. Do not weaken `0.05` or `0.5`.

- [ ] **Step 5: Commit**

```bash
git add python/python/tests/test_vector_index.py python/python/lance/dataset.py docs/src/format/index/vector/index.md
git commit -m "test(index): compare IVF_MRQ recall with IVF_RQ"
```

---

## Self-review

- Spec coverage: levels `1..=8`, one rotation, greedy `α`, 8×8 joint refit, bias-corrected cap, `IndexType::IvfMrq = 108`, version `3`, and the `IVF_RQ` comparison each have a task.
- `IVF_RQ` version stays `2`. The new proto field number is `10`. `num_bits` is rejected on `IVF_MRQ`.
- The self-query test locks the `b_k` term. Removing `level.bias` from `lower_bound_sq` fails `self_query_lower_bound_stays_non_positive`.
- Comparison budget is `levels = num_bits` at `1`, `4`, and `8`. The recall floor is `0.5`, and `IVF_MRQ` may trail `IVF_RQ` by at most `0.05`.
- Placeholder scan: no TBD steps. Java is not in this plan; Python is the public entry. Add Java only if a later request asks for it.

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-10-04-multilevel-residual-rabitq.md`. Two execution options:

1. Subagent-Driven (recommended) — a fresh subagent per task, with review between tasks.
2. Inline Execution — execute the tasks in this session, batch by batch, with checkpoints.

Which approach?
