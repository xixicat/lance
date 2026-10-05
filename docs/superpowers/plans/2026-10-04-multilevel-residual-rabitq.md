# 多级残差 RaBitQ（IVF_MRQ）实现计划

> **给实现者：** 按任务顺序执行。推荐每个任务单独实现并核对测试。步骤使用 `- [ ]` 复选框跟踪。

**目标：** 新增 `IVF_MRQ` 索引。它对 IVF 残差做一次 Fast 旋转，再用 1 到 8 级贪婪 1-bit 残差量化。并在相同级数 / `num_bits` 预算下和 `IVF_RQ` 比较召回。

**结构：** 数学放在 `lance-index` 的 `residual_levels.rs`，不改 `RabitQuantizer`。`IVF_MRQ` 是新的 `IndexType`（`IvfMrq = 108`）和新的量化器，走和 `IVF_RQ` 相同的 IVF 构建器。现有 `IVF_RQ` 文件的读写行为保持不变。固定种子的例子打印算法表。数据集测试在同一批向量上同时建两种索引并记录召回。

**实现修订：** 联合最小二乘之后，系数再乘 `||r||² / ⟨r, r̂⟩`，使重建在原残差方向上无偏。索引不再存 `__mrq_radius` 和 `__mrq_bias`。分区打分复用 RaBitQ accurate 模式的 1-bit FastScan（u16 查找表），每个查询建一次查找表，每一级扫一次。u8 表的误差会改写 top-k，所以阈值附近的行再用标量内积校正。`||q||₁ · Σ|α|` 硬上界留在标量 `search_partition`。IVF 热路径实测按 32 行分批早停比整级 FastScan 更慢，所以热路径扫完每一级。下面正文里关于 `b_k`、`R_k` 和 `γ` 的段落是最初的设计记录。

**技术栈：** Rust、`lance-index` 包、现有 `vector/bq/rotation.rs`、`rand` 0.9、`rand_distr`。

## 全局约束

- 每个编码器只共享一把 Fast 旋转。每一级不要重新旋转。
- 残差更新为 `e_k = e_{k-1} - α_k * sign(e_{k-1})`，其中 `α_k = ||e_{k-1}||_1 / d`，`sign(0) = +1`。
- 级数范围是 `1..=8`。`1` 是本编码器内部的基线，不是 IVF_RQ 的 `num_bits`。
- 距离是旋转空间里的 L2 平方：`N_sq + ||q_r||^2 - 2 S`。
- 穿刺程序默认打开联合最小二乘。`BᵀB` 奇异时保留贪婪系数。
- 每一级保存：打包的 `±1` 码、`alpha`、`radius = ||e_k||_2`、`bias = <e_k, o_r>`。
- 候选内积上限：`U_k = min(R_k L_q, b_k + γ R_k L_q / sqrt(d))`。
- 堆门槛：`D_cut^2 = D_est^2(A) + 2 γ R_m(A) L_q / sqrt(d)`，`R_m(A)` 是堆顶那一行自己的最终残差模长。
- `γ = sqrt(d)` 是柯西安全档。`γ = 3` 是激进档，不是 99.7% 召回保证。
- `IVF_MRQ` 是新索引。不要把它接到 `num_bits`、`__blocked_ex_codes` 或 `RabitQuantizer` 的存储上。
- `IVF_RQ` 的行为不变。`match` 只增加新分支，不改已有 `Rabit` 分支里的表达式。
- `IndexType::IvfMrq` 的判别值是 `108`。格式版本是 `3`。`IvfRq` 保持版本 `2`。
- Python 参数名是 `levels`，默认 `4`。给 `IVF_MRQ` 传 `num_bits` 必须报错。
- 对标预算是 `IVF_MRQ levels = m` 对比 `IVF_RQ num_bits = m`，`m ∈ {1, 4, 8}`。两边的召回率@10 都要 `>= 0.5`。同一 `m` 下 `IVF_MRQ` 不能比 `IVF_RQ` 低过 `0.05`。
- 代码注释和名字用英文。测试用断言。只有例子程序可以打印结果表。
- 单元测试用 2 的幂维度（`8`、`32`、`128`）。例子程序可以再跑 `768`。

---

## 文件

- 新建 `rust/lance-index/src/vector/bq/residual_levels.rs`。负责编码、联合重估、旋转空间 L2 估计、下界和分区搜索。
- 新建 `rust/lance-index/src/vector/mrq/mod.rs`。放 `MrqQuantizer`、元数据和存储。这是 IVF 量化器，调用 `residual_levels`，不调用 `RabitQuantizer`。
- 修改 `rust/lance-index/src/vector/bq.rs`。增加 `pub mod residual_levels;`。
- 修改 `rust/lance-index/src/vector/quantizer.rs`。增加 `QuantizationType::Mrq` 和 `Quantizer::Mrq`。
- 修改 `rust/lance-index-core/src/lib.rs`。增加 `IndexType::IvfMrq = 108`。
- 修改 `protos/index.proto`。在 `compression` 里增加 `MultiResidualQuantization`。
- 修改 `rust/lance/src/index/vector.rs`。增加 `StageParams::MRQ` 和 `VectorIndexParams::ivf_mrq`。
- 修改 `python/src/dataset.rs`。接受 `index_type="IVF_MRQ"` 和参数 `levels`。
- 新建 `rust/lance-index/examples/residual_levels_spike.rs`。只打印算法表。
- 修改 `python/python/tests/test_vector_index.py`。和 `IVF_RQ` 做召回对标。
- 不要改 `RabitQuantizer` 的数学、`num_bits` 校验，以及 IVF_RQ 的辅助列集合。

---

### 任务 1：贪婪残差编码

**文件：**
- 新建：`rust/lance-index/src/vector/bq/residual_levels.rs`
- 修改：`rust/lance-index/src/vector/bq.rs`（加在 `pub mod rotation;` 旁边的 `pub mod` 列表）
- 测试：`rust/lance-index/src/vector/bq/residual_levels.rs`（`mod tests`）

**接口：**
- 依赖：`crate::vector::bq::rotation::{apply_fast_rotation, fast_rotation_signs_len, random_fast_rotation_signs}`
- 产出：
  - `ResidualEncoder::new(dim: usize) -> Self`
  - `ResidualEncoder::encode(&self, vector: &[f32], levels: usize, joint: bool) -> Result<EncodedVector>`
  - `encode_rotated(rotated: &[f32], levels: usize, joint: bool) -> Result<EncodedVector>`
  - `EncodedVector { dim: usize, norm_sq: f32, levels: Vec<LevelCode> }`
  - `LevelCode { packed: Vec<u8>, alpha: f32, radius: f32, bias: f32 }`

- [ ] **步骤 1：先写失败测试**

在 `rust/lance-index/src/vector/bq.rs` 里、`pub mod rotation;` 后面立刻加上这个模块声明：

```rust
pub mod residual_levels;
```

新建 `rust/lance-index/src/vector/bq/residual_levels.rs`，这一步只放下面的测试。预期失败信息是 `cannot find function encode_rotated`。

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

- [ ] **步骤 2：运行测试，确认它失败**

运行：`cargo test -p lance-index rejects_level_count_outside_1_to_8 --lib`

预期：失败，因为还没有 `encode_rotated`。

- [ ] **步骤 3：写贪婪编码器**

用下面的实现替换文件。`joint = true` 先被接受，行为与贪婪相同，直到任务 2 替换 `refit_joint`。

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

把步骤 1 的 `mod tests` 原样留在同一文件底部。

- [ ] **步骤 4：运行测试**

运行：`cargo test -p lance-index residual_levels --lib`

预期：`greedy_residual_is_orthogonal_to_its_sign`、`zero_vector_has_zero_coefficients`、`rejects_level_count_outside_1_to_8` 通过。

- [ ] **步骤 5：提交**

```bash
git add rust/lance-index/src/vector/bq.rs rust/lance-index/src/vector/bq/residual_levels.rs
git commit -m "feat(index): encode greedy residual 1-bit levels"
```

---

### 任务 2：联合最小二乘重估

**文件：**
- 修改：`rust/lance-index/src/vector/bq/residual_levels.rs`
- 测试：同一文件

**接口：**
- 依赖：`encode_rotated`、`LevelCode::radius`
- 产出：`pub(crate) fn solve_gram(gram: &[Vec<f64>], proj: &[f64]) -> Option<Vec<f64>>`。`encode_rotated(..., joint: true)` 在格拉姆矩阵可逆时替换贪婪 `alpha`，再从原始旋转向量重算 `radius` 和 `bias`。

- [ ] **步骤 1：先写失败测试**

追加到 `mod tests` 里面：

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

- [ ] **步骤 2：运行测试，确认它失败**

运行：`cargo test -p lance-index joint_refit_does_not_increase_final_residual --lib`

预期：失败，因为还没有 `solve_gram`，联合重估仍是空操作。就算空操作碰巧通过了模长断言，`singular_gram_returns_none` 仍然编译失败。

- [ ] **步骤 3：实现最多 8×8 的重估**

替换 `refit_joint`，并在 `dot` 前面加上 `solve_gram`：

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

- [ ] **步骤 4：运行测试**

运行：`cargo test -p lance-index residual_levels --lib`

预期：通过，包括 `joint_refit_does_not_increase_final_residual` 和 `singular_gram_returns_none`。在 `joint_refit_does_not_increase_final_residual` 里再断言 `encode_rotated(&sample(), 8, true)`：8 级联合残差模长 `<=` 8 级贪婪残差模长加 `1e-3`。格拉姆求解必须接受 `n = 8`。

- [ ] **步骤 5：提交**

```bash
git add rust/lance-index/src/vector/bq/residual_levels.rs
git commit -m "feat(index): refit residual level coefficients by least squares"
```

---

### 任务 3：距离估计、下界和分区搜索

**文件：**
- 修改：`rust/lance-index/src/vector/bq/residual_levels.rs`
- 测试：同一文件

**接口：**
- 依赖：`EncodedVector`、`LevelCode`
- 产出：
  - `dot_packed_pm1(packed: &[u8], rotated_query: &[f32]) -> f32`
  - `estimated_l2_sq(rotated_query: &[f32], encoded: &EncodedVector) -> f32`
  - `lower_bound_sq(rotated_query: &[f32], encoded: &EncodedVector, done: usize, gamma: f32) -> f32`
  - `threshold_cut_sq(est_sq: f32, radius_m: f32, query_norm: f32, dim: usize, gamma: f32) -> f32`
  - `search_partition(rotated_query: &[f32], encoded: &[EncodedVector], k: usize, gamma: f32, prune: bool) -> SearchStats`
  - `SearchStats { hits: Vec<SearchHit>, pruned: usize }`
  - `SearchHit { id: usize, est_sq: f32 }`

- [ ] **步骤 1：先写失败测试**

追加到 `mod tests` 里面：

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

- [ ] **步骤 2：运行测试，确认它失败**

运行：`cargo test -p lance-index self_query_lower_bound_stays_non_positive --lib`

预期：失败，因为还没有 `lower_bound_sq`。

- [ ] **步骤 3：实现打分和搜索**

把下面这块加在 `dot` 后面：

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

- [ ] **步骤 4：运行测试**

运行：`cargo test -p lance-index residual_levels --lib`

预期：通过。`self_query_lower_bound_stays_non_positive` 锁住融合公式：查询等于底库向量时，带上 `bias` 之后，每一级前缀在 `γ` 为 `0`、`3`、`sqrt(d)` 时下界都不大于 0。

- [ ] **步骤 5：提交**

```bash
git add rust/lance-index/src/vector/bq/residual_levels.rs
git commit -m "feat(index): score and prune residual 1-bit levels"
```

---

### 任务 4：召回单元门禁

**文件：**
- 只改测试：`rust/lance-index/src/vector/bq/residual_levels.rs`
- 测试：`recall_at_10_on_gaussian_rows_meets_floor`

**接口：**
- 依赖：`ResidualEncoder`、`search_partition`、`encode`
- 产出：一条固定种子的召回断言，作为快速 CI 门禁。更宽的表留在例子程序里。

- [ ] **步骤 1：先写失败测试**

追加到 `mod tests` 里面：

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

`0.5` 是仓库对向量索引的召回下限。它是 CI 绊线，不是穿刺目标。任务 5 的例子会打印后续决策要用的数字。

- [ ] **步骤 2：运行测试**

运行：`cargo test -p lance-index recall_at_10_on_gaussian_rows_meets_floor --lib`

预期：通过。若失败，断言信息里会带上召回。若召回低于 `0.5`，停下来。不要放宽任务 5 的门禁，也不要开始 IVF 接入。记下召回，这一组参数下算法视为不成立。

- [ ] **步骤 3：测试通过时不改产品代码**

这一任务只加测试。如果是维度对不上或命中为空这类实现错误，修 `search_partition` 或测试数据后重跑。不要把 `0.5` 降下去。

- [ ] **步骤 4：提交**

```bash
git add rust/lance-index/src/vector/bq/residual_levels.rs
git commit -m "test(index): gate residual-level recall on gaussian rows"
```

---

### 任务 5：穿刺例子

**文件：**
- 新建：`rust/lance-index/examples/residual_levels_spike.rs`

**接口：**
- 依赖：`lance_index::vector::bq::residual_levels` 里的 `ResidualEncoder`、`search_partition`、`lower_bound_sq`、`encode_rotated`
- 产出：一个进程，打印一张 TSV 表。下面每条门禁都成立时退出码才是 `0`。

门禁在运行时判定。看到不好的结果之后不要改这些常数：

- 用底库向量自己当查询：每一级、`γ ∈ {3, sqrt(d)}` 时，`lower_bound_sq` `<= 1e-2`。
- 高斯数据 `dim=128`、`rows=400`、`queries=40`、`k=10`、`m=8`、打开联合重估、关闭剪枝：平均召回率@10 `>= 0.5`。
- 同一批数据上，`m=8` 的召回 `+ 0.02 >= m=1` 的召回。
- 同一批数据上，`γ = sqrt(d)` 的剪枝率 `<=` `γ = 3` 的剪枝率。

- [ ] **步骤 1：写例子程序**

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

- [ ] **步骤 2：再跑一遍单元测试**

运行：`cargo test -p lance-index residual_levels --lib`

预期：通过。例子程序不在 `--lib` 里。

- [ ] **步骤 3：跑穿刺**

运行：`cargo run -p lance-index --example residual_levels_spike --release`

预期：TSV 表头是 `dataset dim rows queries m gamma prune recall_at_10 prune_ratio`。接着是 `dim` 为 `128` 和 `768` 的高斯行，`m ∈ {1, 2, 4, 8}`，以及两种剪枝设置，再加上聚类数据的 `m=8`。128 维门禁通过则退出码 `0`。退出码 `1` 时停止后续索引工作。

按下面的规则读这张表：

- `m=8` 召回不高于 `m=1`：多出来的级数没有带来近邻。停在任务 6 之前。
- `γ=3` 的 `prune_ratio` 接近 `0`：界仍然太松。`IVF_MRQ` 只交付 `γ = sqrt(d)`。
- `γ=3` 剪掉很多行，同时召回率@10 明显低于不剪枝的 `m=8`：激进界在删真近邻。不要把 `γ=3` 做成默认值。
- 聚类数据的召回只作诊断，不是门禁。

- [ ] **步骤 4：提交**

```bash
git add rust/lance-index/examples/residual_levels_spike.rs
git commit -m "test(index): add residual-level recall and prune spike"
```

---

### 任务 6：注册 `IVF_MRQ`，不改 `IVF_RQ`

**文件：**
- 修改：`rust/lance-index-core/src/lib.rs`
- 修改：`rust/lance-index/src/lib.rs`
- 修改：`protos/index.proto`
- 测试：`rust/lance-index/src/lib.rs`（`test_index_type_try_from_str_covers_all_parseable_variants`）

**接口：**
- 依赖：现有 `IndexType::IvfRq = 107`，版本 `2`
- 产出：
  - `IndexType::IvfMrq = 108`
  - `Display` / `TryFrom<&str>` 接受 `"IVF_MRQ"`
  - `IndexType::IvfMrq.version() == 3`
  - `IndexType::IvfRq.version()` 保持 `2`
  - `IVF_MRQ_INDEX_VERSION: u32 = 3`
  - `max_vector_version()` 返回 `3`
  - 协议：`VectorIndexDetails.MultiResidualQuantization { uint32 levels = 1; }`，作为 `compression` 的字段 `10`

- [ ] **步骤 1：先写失败测试**

在 `rust/lance-index/src/lib.rs` 里，给 `test_index_type_try_from_str_covers_all_parseable_variants` 加上 `("IVF_MRQ", IndexType::IvfMrq)`，并增加：

```rust
    #[test]
    fn test_ivf_mrq_version_does_not_change_ivf_rq() {
        assert_eq!(IndexType::IvfRq.version(), 2);
        assert_eq!(IndexType::IvfMrq.version(), 3);
        assert_eq!(IndexType::max_vector_version(), IVF_MRQ_INDEX_VERSION);
    }
```

- [ ] **步骤 2：运行测试，确认它失败**

运行：`cargo test -p lance-index test_ivf_mrq_version_does_not_change_ivf_rq --lib`

预期：失败，因为还没有 `IndexType::IvfMrq`。

- [ ] **步骤 3：加上枚举变体和协议分支**

在 `IndexType` 里、`IvfRq = 107;` 后面加上：

```rust
    IvfMrq = 108,
```

把这个变体补进 `Display`、两个 `TryFrom`、`version`（`3`）、`target_partition_size`（`4096`，与 `IvfRq` 相同）、`max_vector_version` 和 `matches_details`（仍然是 `VectorIndexDetails`）。同步更新 `lance-index/src/lib.rs` 里穷尽枚举的测试列表。

在 `protos/index.proto` 的 `VectorIndexDetails` 里、`oneof compression` 之前加上：

```protobuf
  message MultiResidualQuantization {
    // Number of residual 1-bit levels. Valid range is 1..=8.
    // Absent on old writers. Readers of IVF_MRQ require this field.
    uint32 levels = 1;
  }
```

在 `oneof compression` 里、`FlatCompression flat = 8;` 后面加上：

```protobuf
    MultiResidualQuantization mrq = 10;
```

用仓库现有的协议构建重新生成 Rust 代码（`cargo check -p lance-index`）。不要手改生成文件。

- [ ] **步骤 4：运行测试**

运行：`cargo test -p lance-index test_ivf_mrq_version_does_not_change_ivf_rq --lib`

预期：通过。`IndexType::IvfRq` 的测试仍然期望版本 `2`。

- [ ] **步骤 5：提交**

```bash
git add rust/lance-index-core/src/lib.rs rust/lance-index/src/lib.rs protos/index.proto
git commit -m "feat(index): add IVF_MRQ index type"
```

---

### 任务 7：`MrqQuantizer` 和 IVF 构建路径

**文件：**
- 新建：`rust/lance-index/src/vector/mrq/mod.rs`
- 修改：`rust/lance-index/src/vector.rs`（`pub mod mrq;`）
- 修改：`rust/lance-index/src/vector/quantizer.rs`
- 修改：`rust/lance/src/index/vector.rs`
- 修改：`python/src/dataset.rs`

**接口：**
- 依赖：`encode_rotated`、`search_partition`、`ResidualEncoder`、`EncodedVector`
- 产出：
  - `MrqBuildParams { pub levels: u8 }`，`levels` 校验范围 `1..=8`
  - 实现 `Quantization` 的 `MrqQuantizer`
  - `Quantizer::Mrq(MrqQuantizer)`，`QuantizationType::Mrq` 的显示字符串是 `"MRQ"`
  - `StageParams::MRQ(MrqBuildParams)`
  - `VectorIndexParams::ivf_mrq(num_partitions: usize, levels: u8, distance_type: DistanceType) -> Self`
  - 最后一阶段是 `StageParams::MRQ` 时，`VectorIndexParams::index_type()` 返回 `IndexType::IvfMrq`
  - Python 的 `index_type="IVF_MRQ"` 读取参数 `levels`（默认 `4`）

量化器写入、`try_from_batch` 读回的存储列：

| 列 | 类型 | 含义 |
| --- | --- | --- |
| `_rowid` | uint64 | 现有 IVF 行号 |
| `__mrq_codes` | `uint8` 定长列表，宽度 `levels * ceil(dim / 8)` | 按级排列的打包符号 |
| `__mrq_alpha` | `float32` 定长列表，宽度 `levels` | 每一级的缩放系数 |
| `__mrq_radius` | `float32` 定长列表，宽度 `levels` | `R_k` |
| `__mrq_bias` | `float32` 定长列表，宽度 `levels` | `b_k` |
| `__mrq_norm_sq` | float32 | 旋转后残差的模长平方 |

旋转符号放在量化器元数据里，整份索引共享一把，方式和 `RabitQuantizationMetadata.fast_rotation_signs` 相同。构建时仍然先做 `ResidualTransform` 再量化，`IvfIndexBuilder` 对 `RabitQuantizer` 已经是这个顺序。

分区距离走 `VectorStore`，默认用 `gamma = sqrt(dim)` 调用 `search_partition`。不要调用 `RabitDistCalculator`。

- [ ] **步骤 1：先写失败的参数测试**

写在 `rust/lance/src/index/vector.rs` 的测试里，放在 `VectorIndexParams::ivf_rq` 旁边：

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

- [ ] **步骤 2：运行测试，确认它失败**

运行：`cargo test -p lance ivf_mrq_params_report_mrq_index_type --lib`

预期：失败，因为还没有 `ivf_mrq`。

- [ ] **步骤 3：实现量化器和参数接线**

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

`VectorIndexParams::ivf_mrq` 照着 `ivf_rq` 写，把 `StageParams::RQ` 换成 `StageParams::MRQ`。`index_type()` 增加：

```rust
(2, _, Some(StageParams::MRQ(_))) => IndexType::IvfMrq,
```

`build_vector_index` 里的本地构建分支，以及分布式构建里现有的 `IndexType::IvfRq` 分支，各自增加一个 `IndexType::IvfMrq` 分支。新分支构造 `IvfIndexBuilder::<FlatIndex, MrqQuantizer>`，参数用 `MrqBuildParams`。照着 `IvfRq` 分支抄，只改量化器类型和阶段枚举。`IvfRq` 分支的函数体不要动。

在 `python/src/dataset.rs` 里：

- 在 `"IVF_RQ"` 那一行的向量索引类型列表里加上 `"IVF_MRQ"`。
- 在参数 `match` 里加上：

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

量化器的 `quantize` 用共享符号旋转这一批残差，再调用 `encode_rotated(..., joint: true)`。`from_metadata` 恢复同一组符号。

`Quantizer` 和 `QuantizationType` 上的每个 `match` 都要能编译。为 `Mrq` 增加分支，返回 MRQ 的列名或元数据键。不要改 `Rabit` 分支里的表达式。

- [ ] **步骤 4：运行参数测试**

运行：`cargo test -p lance ivf_mrq_params_report_mrq_index_type --lib`

预期：通过。`cargo check -p lance --tests` 干净通过，包括新增的 `match` 分支。

- [ ] **步骤 5：提交**

```bash
git add rust/lance-index/src/vector/mrq/mod.rs rust/lance-index/src/vector.rs \
  rust/lance-index/src/vector/quantizer.rs rust/lance/src/index/vector.rs python/src/dataset.rs
git commit -m "feat(index): build IVF_MRQ from residual 1-bit levels"
```

---

### 任务 8：和 `IVF_RQ` 对标召回

**文件：**
- 修改：`python/python/tests/test_vector_index.py`
- 修改：`python/python/lance/dataset.py` 里 `create_index` 的文档
- 修改：`docs/src/format/index/vector/index.md`，补上 `IVF_MRQ` 的辅助文件格式

**接口：**
- 依赖：`dataset.create_index(..., index_type="IVF_MRQ", levels=m)`，以及现有 `IVF_RQ` 的 `num_bits` 参数
- 产出：参数化测试 `test_ivf_mrq_matches_ivf_rq_recall`，以及格式说明：`IVF_MRQ` 是版本 `3` 的新索引

- [ ] **步骤 1：先写失败测试**

加到 `python/python/tests/test_vector_index.py`，放在 `test_create_ivf_rq_index` 旁边：

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

如果 `test_create_ivf_rq_index` 建表的写法和这里不一样，就沿用那个测试的写法。断言保持不变。

- [ ] **步骤 2：运行测试，确认它失败**

在 `python/` 目录下运行：

运行：`uv run pytest python/tests/test_vector_index.py::test_ivf_mrq_matches_ivf_rq_recall -q`

预期：失败，因为 `IVF_MRQ` 被拒绝，或者召回低于下限。在任务 7 还没进同一次构建时，缺索引类型的错误也算这一步的失败。

- [ ] **步骤 3：写格式说明，测试继续当对标门禁**

在 `docs/src/format/index/vector/index.md` 里加上 `IVF_MRQ` 辅助文件格式，列就是任务 7 的六列。写明 `levels` 为 `1..=8`，版本为 `3`，`IVF_RQ` 的读取器不解析这些列。

在 `create_index` 的文档里，按 `IVF_RQ` 文档写 `num_bits` 的方式写 `IVF_MRQ` 的 `levels`。默认 `4`。

除非测试表明索引建出来却不能查，这一步不要再改搜索实现。默认查询 `γ` 保持 `sqrt(dim)`。

- [ ] **步骤 4：跑对标**

运行：`uv run pytest python/tests/test_vector_index.py::test_ivf_mrq_matches_ivf_rq_recall -q`

预期：预算 `1`、`4`、`8` 都通过。如果 `mrq_recall + 0.05 < rq_recall`，停下来修量化器。不要把 `0.05` 或 `0.5` 放宽。

- [ ] **步骤 5：提交**

```bash
git add python/python/tests/test_vector_index.py python/python/lance/dataset.py docs/src/format/index/vector/index.md
git commit -m "test(index): compare IVF_MRQ recall with IVF_RQ"
```

---

## 自检

- 规格覆盖：级数 `1..=8`、一次旋转、贪婪 `α`、8×8 联合重估、带偏差的内积上限、`IndexType::IvfMrq = 108`、版本 `3`、以及和 `IVF_RQ` 的对标，各自都有任务。
- `IVF_RQ` 版本保持 `2`。新的协议字段号是 `10`。`IVF_MRQ` 拒绝 `num_bits`。
- 自查询测试锁住 `b_k`。从 `lower_bound_sq` 去掉 `level.bias` 后，`self_query_lower_bound_stays_non_positive` 会失败。
- 对标预算是 `levels = num_bits`，取 `1`、`4`、`8`。召回下限是 `0.5`，`IVF_MRQ` 最多比 `IVF_RQ` 低 `0.05`。
- 没有待定步骤。本计划不包含 Java。对外入口是 Python。只有后续明确要求时再加 Java。

## 执行方式

计划已写在 `docs/superpowers/plans/2026-10-04-multilevel-residual-rabitq.md`。两种执行方式：

1. 按任务拆开做。每完成一个任务核对一次测试。
2. 在当前会话里按任务顺序直接实现，做完一批再核对。

要哪一种？
