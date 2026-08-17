# RaBitQ Blog Revisions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce three technically corrected Chinese RaBitQ articles based exclusively on the Lance community `v10.0.0` release.

**Architecture:** Treat the tagged source, RaBitQ paper, and RaBitQ-Library estimator documentation as the shared fact base. Create three independent Markdown drafts with different editorial structures, then run one cross-document verification pass for formulas, terminology, citations, and unsupported claims.

**Tech Stack:** Markdown, LaTeX, Git, ripgrep, Python standard library

## Global Constraints

- Analyze Lance tag `v10.0.0`, commit `95f2f36b22043c3face00afe088c34e0742d01df`.
- Do not compare against `e1dbd14288b40cc4f83264328bbc336f4c697dc9` or any legacy implementation.
- Do not modify Lance source code, existing website navigation, or existing published documentation.
- Do not present a high-probability RaBitQ bound as a deterministic guarantee.
- Do not include unsupported values such as “千万分之一漏召回” or “80% I/O 剪枝”.
- Use repository-relative source paths and symbol names instead of unstable line numbers.
- Each article must stand alone and link to the RaBitQ paper, RaBitQ-Library estimator documentation, and Lance `v10.0.0` source tag.

---

### Task 1: Verify the `v10.0.0` fact base

**Files:**
- Read from tag: `rust/lance-index/src/vector/bq.rs`
- Read from tag: `rust/lance-index/src/vector/bq/builder.rs`
- Read from tag: `rust/lance-index/src/vector/bq/rotation.rs`
- Read from tag: `rust/lance-index/src/vector/bq/transform.rs`
- Read from tag: `rust/lance-index/src/vector/bq/storage.rs`
- Read from tag: `rust/lance-index/src/vector/bq/prune.rs`
- Read from tag: `rust/lance/src/index/vector.rs`

**Interfaces:**
- Consumes: Git tag `v10.0.0` and public RaBitQ references.
- Produces: A verified shared fact set used verbatim by Tasks 2–4.

- [ ] **Step 1: Confirm the release identity**

Run:

```bash
git rev-parse 'v10.0.0^{commit}'
git show -s --format='%H %s' 'v10.0.0^{commit}'
```

Expected:

```text
95f2f36b22043c3face00afe088c34e0742d01df
95f2f36b22043c3face00afe088c34e0742d01df chore: release version 10.0.0
```

- [ ] **Step 2: Locate the exact implementation symbols in the tag**

Run:

```bash
git grep -n -E 'compute_raw_query_factors|error_factor_value|quantize_ex_code|best_ex_rescale_factor|apply_fast_rotation|raw_query_lower_bound|RABIT_ERROR_EPSILON' v10.0.0 -- \
  rust/lance-index/src/vector/bq
```

Expected: matches in `builder.rs`, `rotation.rs`, `transform.rs`, and `storage.rs`.

- [ ] **Step 3: Verify the public index combinations**

Run:

```bash
git grep -n -E 'IVF_RQ|IvfRq|QuantizationType::Rabit' v10.0.0 -- \
  rust/lance/src/index/vector.rs rust/lance/src/index/vector
```

Expected: `IVF_RQ` uses the flat IVF sub-index path; no `IVF_HNSW_RQ` constructor is present.

- [ ] **Step 4: Record the shared mathematical facts for the drafting tasks**

Use these exact facts:

```text
r = o - c
z = W r
bf_i = +0.5 when z_i is sign-positive, otherwise -0.5
binary_res_dot = <z, bf>
binary_cent_dot = <Wc, bf>
L2 add = ||r||² + 2 ||r||² binary_cent_dot / binary_res_dot
L2 scale = -2 ||r||² / binary_res_dot
alignment = ||r||² ||bf||² / <z,bf>², with ||bf||² = d/4
angular_error = sqrt(max(alignment - 1, 0) / (d - 1))
L2 error_factor = 2 ||r|| epsilon_0 angular_error
query_error = ||q-c||
lower_bound = binary_estimate - error_factor * query_error
epsilon_0 = 1.9
ex_bits = num_bits - 1
full_code = (sign_code << ex_bits) + ex_code
centered_code = full_code - (2^ex_bits - 0.5)
```

- [ ] **Step 5: Verify the external references**

Open and confirm:

```text
https://doi.org/10.1145/3654970
https://vectordb-ntu.github.io/RaBitQ-Library/rabitq/estimator/
https://github.com/lance-format/lance/tree/v10.0.0
```

Expected: the first is the RaBitQ paper, the second documents the estimator and probability bound, and the third is the source baseline.

### Task 2: Write the original-structure revision

**Files:**
- Create: `docs/superpowers/drafts/lance-rabitq-original-style.md`

**Interfaces:**
- Consumes: Task 1 fact set and the user’s original five-section narrative.
- Produces: A long-form article preserving the original editorial voice while correcting its mathematics.

- [ ] **Step 1: Create the draft with the retained five-part structure**

The document must contain:

```text
# 揭秘 Lance 向量索引中的 RaBitQ：从数学推导到工程实现
> 源码基线
1. 算法破局点：随机正交旋转
2. 距离估计器：精确项与近似交叉项
3. 概率误差界与下界剪枝
4. 多比特扩展：自适应网格与增量估计
5. 总结
参考资料
```

Required corrections:

```text
- Explain that ||r||² remains exact; do not derive from ||alpha bf||² ≈ ||r||².
- Write rotated inner products as <Wq,bf> and <Wc,bf>.
- Describe Fast rotation as four FHT-Kac rounds, with Matrix as a separate option.
- Give the complete L2 error radius, including 2 and ||q-c||.
- State that lower-bound gating is a high-probability mechanism for eligible multi-bit IVF_RQ scans.
- Describe per-vector t optimization before the symmetric half-integer ex-code grid.
- Qualify training-free and 32x compression claims.
```

- [ ] **Step 2: Check that the retained style does not reintroduce disproven claims**

Run:

```bash
rg -n '能量守恒近似|千万分之一|80%|IVF_HNSW_RQ[[:space:]]*(已支持|已经支持|可构建|可以构建|is supported|supported)|(公开)?支持[[:space:]]*IVF_HNSW_RQ|一行代码都不需要改|严格安全|绝对安全' \
  docs/superpowers/drafts/lance-rabitq-original-style.md
```

Expected: no matches.

- [ ] **Step 3: Commit the original-structure draft**

```bash
git add docs/superpowers/drafts/lance-rabitq-original-style.md
git commit -m "docs: add corrected original-style RaBitQ article"
```

### Task 3: Write the source-oriented main article

**Files:**
- Create: `docs/superpowers/drafts/lance-rabitq-source-analysis.md`

**Interfaces:**
- Consumes: Task 1 fact set.
- Produces: The recommended article organized by the Lance `v10.0.0` execution path.

- [ ] **Step 1: Create the source-oriented article**

Use this structure:

```text
# Lance v10.0.0 中的 RaBitQ：从概率估计器到多比特剪枝
> 源码基线与阅读范围
1. RaBitQ 在 IVF_RQ 中的位置
2. 两种旋转实现
3. 1-bit 编码及其尺度约定
4. 不依赖向量重构的距离估计器
5. add/scale/query 三类因子的代码映射
6. 概率误差界如何变成查询下界
7. 多比特 ex-code 与增量重估
8. 查询热路径和 SIMD 实现
9. 工程边界与容易误读之处
10. 总结
参考资料
```

Include pseudocode equivalent to:

```text
binary_dot = binary_ip - 0.5 * sum(rotated_query)
estimate = binary_dot * scale_factor + add_factor + query_factor
margin = error_factor * query_error
lower_bound = estimate - margin
```

Explicitly explain that the pruning SIMD path avoids FMA to preserve scalar rounding behavior.

- [ ] **Step 2: Verify all mentioned symbols exist in `v10.0.0`**

Run:

```bash
symbols='compute_raw_query_factors error_factor_value quantize_ex_code best_ex_rescale_factor apply_fast_rotation raw_query_lower_bound'
for symbol in $symbols; do
  git grep -q "$symbol" v10.0.0 -- rust/lance-index/src/vector/bq || exit 1
done
```

Expected: exit status 0.

- [ ] **Step 3: Commit the source-oriented draft**

```bash
git add docs/superpowers/drafts/lance-rabitq-source-analysis.md
git commit -m "docs: add source-oriented RaBitQ article"
```

### Task 4: Write the concise introduction

**Files:**
- Create: `docs/superpowers/drafts/lance-rabitq-introduction.md`

**Interfaces:**
- Consumes: Task 1 fact set.
- Produces: A shorter article for readers who need the concepts without a line-by-line source walkthrough.

- [ ] **Step 1: Create the concise article**

Use this structure:

```text
# 用一篇文章理解 Lance RaBitQ
1. 普通二值量化为什么不够
2. 随机旋转解决什么问题
3. 只保存符号，如何估算距离
4. 误差范围如何帮助减少多比特计算
5. 精度不够时怎样增加位数
6. 使用时需要记住的限制
参考资料
```

Keep only these formulas:

```text
r = o - c, z = Wr
d²(q,o) = ||q-c||² + ||r||² - 2<q-c,r>
<q-c,r> ≈ ||r||² <W(q-c),bf> / <Wr,bf>
lower_bound = estimate - error_factor * ||q-c||
```

Explain probability bounds and compression overhead in plain language. Do not introduce source symbols that are not explained in the same paragraph.

- [ ] **Step 2: Check that the concise version remains substantially shorter**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

root = Path("docs/superpowers/drafts")
short = len((root / "lance-rabitq-introduction.md").read_text())
long = len((root / "lance-rabitq-source-analysis.md").read_text())
assert short < long * 0.65, (short, long)
print(f"concise={short}, source_analysis={long}")
PY
```

Expected: the assertion passes and prints both character counts.

- [ ] **Step 3: Commit the concise draft**

```bash
git add docs/superpowers/drafts/lance-rabitq-introduction.md
git commit -m "docs: add concise RaBitQ introduction"
```

### Task 5: Cross-document verification and delivery

**Files:**
- Verify: `docs/superpowers/drafts/lance-rabitq-original-style.md`
- Verify: `docs/superpowers/drafts/lance-rabitq-source-analysis.md`
- Verify: `docs/superpowers/drafts/lance-rabitq-introduction.md`
- Copy after verification: `/opt/cursor/artifacts/lance-rabitq-original-style.md`
- Copy after verification: `/opt/cursor/artifacts/lance-rabitq-source-analysis.md`
- Copy after verification: `/opt/cursor/artifacts/lance-rabitq-introduction.md`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces: Three validated repository drafts and three downloadable artifacts.

- [ ] **Step 1: Commit and push all draft changes before testing**

Run:

```bash
git push -u origin cursor/rabitq-blog-revisions-2943
```

Expected: all three article commits are present on the remote branch.

- [ ] **Step 2: Validate baseline, links, Markdown fences, and forbidden claims**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import re

root = Path("docs/superpowers/drafts")
paths = sorted(root.glob("lance-rabitq-*.md"))
assert len(paths) == 3, paths

required = [
    "v10.0.0",
    "95f2f36b22043c3face00afe088c34e0742d01df",
    "https://doi.org/10.1145/3654970",
    "https://vectordb-ntu.github.io/RaBitQ-Library/rabitq/estimator/",
    "https://github.com/lance-format/lance/tree/v10.0.0",
]
forbidden = [
    "e1dbd14288b40cc4f83264328bbc336f4c697dc9",
    "千万分之一",
    "80% 磁盘",
]
unsupported_claim_patterns = [
    r"IVF_HNSW_RQ\s*(?:已支持|已经支持|可构建|可以构建|is supported|supported\b)",
    r"(?:公开)?支持\s*IVF_HNSW_RQ",
]

for path in paths:
    text = path.read_text()
    assert text.startswith("# "), path
    assert text.count("```") % 2 == 0, path
    assert text.count("$$") % 2 == 0, path
    for value in required:
        assert value in text, (path, value)
    for value in forbidden:
        assert value not in text, (path, value)
    for pattern in unsupported_claim_patterns:
        assert not re.search(pattern, text), (path, pattern)
    if "IVF_HNSW_RQ" in text:
        assert "unsupported" in text, path
    print(f"PASS {path}: {len(text)} chars")
PY
```

Expected: three `PASS` lines.

- [ ] **Step 3: Validate whitespace and inspect the final diff**

Run:

```bash
git diff --check main...HEAD
git diff --stat main...HEAD
git status --short
```

Expected: `git diff --check` has no output and `git status --short` is empty.

- [ ] **Step 4: Copy the verified Markdown files to the artifact directory**

Run:

```bash
cp docs/superpowers/drafts/lance-rabitq-original-style.md /opt/cursor/artifacts/lance-rabitq-original-style.md
cp docs/superpowers/drafts/lance-rabitq-source-analysis.md /opt/cursor/artifacts/lance-rabitq-source-analysis.md
cp docs/superpowers/drafts/lance-rabitq-introduction.md /opt/cursor/artifacts/lance-rabitq-introduction.md
```

Expected: all three files are available as downloadable artifacts.

- [ ] **Step 5: Push any verification fixes and update the draft pull request**

If verification required edits:

```bash
git add docs/superpowers/drafts
git commit -m "docs: polish RaBitQ article revisions"
git push -u origin cursor/rabitq-blog-revisions-2943
```

Update the pull request summary with the three delivered variants and the validation results.
