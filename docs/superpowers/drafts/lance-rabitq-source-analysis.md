# Lance v10.0.0 中的 RaBitQ：从概率估计器到多比特剪枝

> **源码基线与阅读范围**
>
> 本文只讨论 Lance Git 标签 [`v10.0.0`](https://github.com/lance-format/lance/tree/95f2f36b22043c3face00afe088c34e0742d01df)（提交 `95f2f36b22043c3face00afe088c34e0742d01df`）中的实现。文中的行为、常量、公式和公开构建边界均以该提交为准，不用其他版本补全或解释。主要路径位于 `rust/lance-index/src/vector/bq/`；上层索引组装位于 `rust/lance/src/index/vector.rs`。

## 1. RaBitQ 在 IVF_RQ 中的位置

Lance 的 `IVF_RQ` 不是把整条向量直接压成比特后做全表扫描。设原始数据向量为 \(\mathbf{o}\)，维度为 \(D\)，它先用 IVF 将 \(\mathbf{o}\) 分到质心 \(\mathbf{c}\) 所在的分区，再令残差

\[
\mathbf{r}=\mathbf{o}-\mathbf{c}
\]

进入 RaBitQ。构建端旋转残差、写入二值码和每行因子；查询端只探测选中的 IVF 分区，并把旋转后的查询通过查表内积映射成距离估计。换言之，IVF 负责缩小候选集合，RQ 负责压缩分区内向量并快速排序候选。

`RQBuildParams` 保存总位宽 `num_bits`、旋转类型和可选的预构建旋转。位宽被严格限制在 \(1\ldots9\)：第一位始终是符号位，其余

\[
e=\texttt{num\_bits}-1\in[0,8]
\]

位称为 ex-code。默认配置是 1 bit 和 `Fast` 旋转。[`rust/lance-index/src/vector/bq.rs::validate_rq_num_bits`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq.rs) 定义范围，[`rust/lance-index/src/vector/bq.rs::RQBuildParams`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq.rs) 则把这些选择带入构建流程。

这个分层很重要：后文的 `add_factor`、`query_factor` 以及误差项都含有质心语义。脱离 IVF 残差上下文，只看一条二值内积公式，会漏掉距离估计器的一半。

## 2. 两种旋转实现

RaBitQ 先用同一个随机正交变换处理数据和查询，使能量更均匀地分散到各维。`v10.0.0` 有两种实现。

### Matrix：显式稠密正交矩阵

`RQRotationType::Matrix` 生成随机正交矩阵，并把矩阵本身写入元数据。构建时以矩阵乘法旋转残差；查询时用同一矩阵旋转查询或查询残差。这一路径直观，但需要保存 \(D\times D\) 矩阵，计算和存储成本也随之增加。创建逻辑见 [`rust/lance-index/src/vector/bq/builder.rs::RabitQuantizer::new_with_rotation`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq/builder.rs)。

### Fast：无矩阵的四轮随机变换

默认的 `Fast` 路径只保存随机符号位，不物化稠密矩阵。`apply_fast_rotation` 固定运行四轮。当 \(D\) 是 2 的幂时，每一轮都是“Rademacher 随机符号翻转 → 全向量原地 FWHT → 乘 \(1/\sqrt D\) 归一化”。

当 \(D\) 不是 2 的幂时，令 \(m\) 为不大于 \(D\) 的最大 2 的幂。四轮中的**每一轮**都严格执行：

1. Rademacher 随机符号翻转；
2. 交替在 head（偶数轮）或 tail（奇数轮）的 \(m\) 维窗口执行原地 FWHT，并乘 \(1/\sqrt m\) 归一化；
3. 随后对整个输出执行一次固定角度的 Kac 式成对混合。

第四轮也包含 Kac mixing；四轮全部结束后，源码还对整个输出乘 `0.25`，补偿交替截断 FWHT 与 Kac 步骤带来的尺度变化。[`rust/lance-index/src/vector/bq/rotation.rs::apply_fast_rotation`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq/rotation.rs) 给出了完整流水线；符号翻转在 x86 上还会按运行时能力选择 AVX2。

两种旋转的协议相同：数据、查询、质心必须使用同一份旋转元数据。分布式 `IVF_RQ` 因而允许通过 `RQBuildParams::rotation` 注入同一个预构建模型，避免不同 shard 各自生成随机旋转。

## 3. 1-bit 编码及其尺度约定

设旋转后的残差为

\[
\mathbf{z}=\mathbf{W}\mathbf{r}.
\]

`pack_sign_bits` 对每维写一个符号位：`is_sign_positive()` 为真写 1，否则写 0，且每字节采用低位优先。为了把 \(\{0,1\}\) 码变为以零为中心的向量，估计器使用单个符号 \(\mathbf{u}\) 表示整条量化向量，
避免写成两个字母时被读成向量乘积：

\[
\mathbf{u}_i=b_i-\frac12\in\left\{-\frac12,+\frac12\right\}.
\]

因此数据端计算的 `binary_res_dot` 是

\[
\mathbf{z}^\mathsf{T}\mathbf{u}=\frac12\sum_{i=1}^{D} |\mathbf{z}_i|,
\]

而码向量的平方范数固定为 \(D/4\)。这解释了源码中看似特殊的 `0.5`：它不是查询侧随意加入的校正，而是 \(\{0,1\}\) 存储表示与 \(\{-0.5,+0.5\}\) 数学表示之间的中心化约定。

查询侧 FastScan 首先得到

\[
\texttt{binary\_ip}=\sum_{i=1}^{D} b_i(\mathbf{W}\mathbf{q})_i.
\]

于是实际需要的中心化内积为

\[
(\mathbf{W}\mathbf{q})^\mathsf{T}\mathbf{u}
=\texttt{binary\_ip}-\frac12\sum_{i=1}^{D}(\mathbf{W}\mathbf{q})_i.
\]

源码把 \(\sum_{i=1}^{D}(\mathbf{W}\mathbf{q})_i\) 缓存在 `sum_q` 中；[`rust/lance-index/src/vector/bq/storage.rs::binary_distance_factor_params`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq/storage.rs) 和 [`rust/lance-index/src/vector/bq/storage.rs::raw_query_binary_distance`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq/storage.rs) 正是这套约定的查询端落点。

## 4. 不依赖向量重构的距离估计器

Lance 不先从 1-bit 码重构一个近似向量再计算距离。构建端为每行预计算仿射因子，查询端用二值内积直接估计距离。核心流程可以写成：

```text
binary_dot = binary_ip - 0.5 * sum(rotated_query)
estimate = binary_dot * scale_factor + add_factor + query_factor
margin = error_factor * query_error
lower_bound = estimate - margin
```

这里 `binary_ip` 来自分区内批量码扫描；`scale_factor`、`add_factor`、`error_factor` 随数据行存储；`query_factor` 和 `query_error` 随“查询 × IVF 分区”计算。数据行与查询的变量因此在最后几次乘加中才汇合，不需要恢复 \(\mathbf{r}\) 或 \(\mathbf{o}\)。

几何上，L2 只近似交叉内积。正交旋转给出

\[
\langle \mathbf{q}-\mathbf{c},\mathbf{r}\rangle=\langle \mathbf{W}(\mathbf{q}-\mathbf{c}),\mathbf{W}\mathbf{r}\rangle=\lVert \mathbf{r}\rVert\lVert \mathbf{q}-\mathbf{c}\rVert\langle\hat{\mathbf{z}},\hat{\mathbf{y}}\rangle,
\]

其中 \(\hat{\mathbf{z}}=\mathbf{W}\mathbf{r}/\lVert \mathbf{r}\rVert\)，\(\hat{\mathbf{y}}=\mathbf{W}(\mathbf{q}-\mathbf{c})/\lVert \mathbf{q}-\mathbf{c}\rVert\)。1-bit 码 \(\mathbf{u}\) 是 \(\hat{\mathbf{z}}\) 的方向代理。按 RaBitQ estimator，用与 \(\mathbf{u}\) 的内积比值估计这两个单位向量的内积：

\[
\langle\hat{\mathbf{z}},\hat{\mathbf{y}}\rangle
\approx
\frac{\langle \mathbf{u},\hat{\mathbf{y}}\rangle}{\langle \mathbf{u},\hat{\mathbf{z}}\rangle}
=\lVert \mathbf{r}\rVert\frac{\langle \mathbf{W}(\mathbf{q}-\mathbf{c}),\mathbf{u}\rangle}{\langle \mathbf{W}\mathbf{r},\mathbf{u}\rangle\,\lVert \mathbf{q}-\mathbf{c}\rVert},
\]

因此

\[
\langle \mathbf{q}-\mathbf{c},\mathbf{r}\rangle
\approx
\lVert \mathbf{r}\rVert^2
\frac{\langle \mathbf{W}(\mathbf{q}-\mathbf{c}),\mathbf{u}\rangle}{\langle \mathbf{W}\mathbf{r},\mathbf{u}\rangle}.
\]

查询热路径再把 \(\langle \mathbf{W}(\mathbf{q}-\mathbf{c}),\mathbf{u}\rangle\) 拆成 \(\langle \mathbf{W}\mathbf{q},\mathbf{u}\rangle-\langle \mathbf{W}\mathbf{c},\mathbf{u}\rangle\)，并把已知的 \(\lVert \mathbf{r}\rVert^2/\langle \mathbf{W}\mathbf{r},\mathbf{u}\rangle\) 折进每行 `scale_factor` 与 `add_factor`。这不是用 \(\alpha \mathbf{u}\) 去欧氏重建 \(\mathbf{r}\) 后再展开距离。

按计算时机标出后，L2 估计器是

\[
\begin{aligned}
\hat{d}^{2}(\mathbf{q},\mathbf{o})
&=
\underbrace{\lVert \mathbf{q}-\mathbf{c}\rVert^{2}}_{\text{在线 B：query\_factor}}
+
\underbrace{\bigl(n+2n\gamma/\beta\bigr)}_{\text{离线 A：add\_factor}}
\\
&\quad+
\underbrace{\bigl(-2n/\beta\bigr)}_{\text{离线 C：scale\_factor}}
\cdot
\underbrace{\langle \mathbf{W}\mathbf{q},\mathbf{u}\rangle}_{\text{在线交互 D：binary\_dot}}.
\end{aligned}
\]

其中 \(n=\lVert \mathbf{r}\rVert^2\)，\(\beta=\mathbf{z}^\mathsf{T}\mathbf{u}=\langle \mathbf{W}\mathbf{r},\mathbf{u}\rangle\)，\(\gamma=(\mathbf{W}\mathbf{c})^\mathsf{T}\mathbf{u}\)。A **不是**单独的残差能量 \(n\)：质心交叉项 \(2n\gamma/\beta\) 已并入 `add_factor`，所以 D 可以对原始旋转查询 \(\mathbf{W}\mathbf{q}\) 做内积，而不必每行再减 \(\mathbf{W}\mathbf{c}\)。

对 L2 和 Dot，`compute_raw_query_factors` 使用不同的仿射系数，但共享同一个几何结构。二值估计器的数据端系数为：

| 距离类型 | `scale_factor` | `add_factor` |
|---|---:|---:|
| L2 | \(-2n/\beta\) | \(n+2n\gamma/\beta\) |
| Dot | \(-n/\beta\) | \(1-\mathbf{r}^\mathsf{T}\mathbf{c}+n\gamma/\beta\) |

源码用 `factor_ratio` 处理分母为零的退化情形：返回 0，而不是产生无穷值。[`rust/lance-index/src/vector/bq/transform.rs::compute_raw_query_factors`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq/transform.rs) 展示了这两组公式。

## 5. 离线三项因子与在线三项的生命周期

Indexing 把 A、C 以及下一节的误差数据项 E 固化到每行；Online 只组合查询项和交互内积。1-bit raw-query 行布局是

```text
+------------------------+------------------+---------------------+---------------------+
| 1-bit bitmap (D bits)  | add (float32)    | scale (float32)     | error (float32)     |
+------------------------+------------------+---------------------+---------------------+
```

对应 RaBitQ-Library 的 `F_add` / `F_rescale` / `F_error`。三个数据因子的拆分对应三个生命周期，不能只按“常数项/乘数”理解，更不能把 `add` 写成单纯的 \(\lVert \mathbf{r}\rVert^{2}\)：

- `scale_factor`（离线 C）：每条数据的缩放。它把中心化码内积映射回该残差的范数尺度；L2 比 Dot 多一个系数 2。
- `add_factor`（离线 A）：每条数据在所属分区中的平移。它包含残差范数和质心交叉项；Dot 还包含距离约定中的 1。
- `query_factor`（在线 B）：每个查询在当前分区中的平移。L2 为 `dist_q_c`；Dot 在已有旋转质心时为 \(-(\mathbf{W}\mathbf{q})^\mathsf{T}(\mathbf{W}\mathbf{c})\)，否则使用 `dist_q_c - 1.0` 的等价上下文值。

交互项 D 使用 \(\{0,1\}\) 存储码：

\[
\langle \mathbf{W}\mathbf{q},\mathbf{u}\rangle
=\texttt{binary\_ip}-\frac12\sum_{i=1}^{D}(\mathbf{W}\mathbf{q})_i.
\]

\(\sum_{i=1}^{D}(\mathbf{W}\mathbf{q})_i\) 缓存在 `sum_q` 中，每个查询计算一次；`binary_ip` 才是分区内 FastScan 的高频计算。在线极简式是

```text
binary_dot = binary_ip - 0.5 * sum(rotated_query)
estimate   = add_factor + query_factor + scale_factor * binary_dot
margin     = error_factor * query_error
lower_bound = estimate - margin
```

B 和 F 都依赖质心 \(\mathbf{c}\)，因此是“查询 × 所探测 IVF 分区”级常量，不是全图一条全局 \(G_{add}\)/\(G_{erro\mathbf{r}}\)。`prepare_raw_query_context` 把 \(\mathbf{W}\mathbf{q}\)、1-bit 距离表、必要时补零到 64 维块边界的 ex-query，以及 `sum_q` 一次性准备好；进入不同 IVF 分区后，只重新结合该分区的旋转质心计算 `query_factor` 和 `query_error`。

| 项 | 公式（L2） | Lance 字段 | RaBitQ-Library | 计算阶段 |
|---|---|---|---|---|
| A | \(n+2n\gamma/\beta\) | `add_factor` | `F_add` | 离线，每行 |
| B | \(\lVert \mathbf{q}-\mathbf{c}\rVert^{2}\) | `query_factor` | `G_add` | 在线，每查询×分区 |
| C | \(-2n/\beta\) | `scale_factor` | `F_rescale` | 离线，每行 |
| D | \(\langle \mathbf{W}\mathbf{q},\mathbf{u}\rangle\) | `binary_dot` | `ip + c_B S_q` | 在线，每候选 |
| E | \(2\lVert \mathbf{r}\rVert\epsilon_0\cdot\texttt{angular\_error}\) | `error_factor` | `F_error` | 离线，每行 |
| F | \(\lVert \mathbf{q}-\mathbf{c}\rVert\) | `query_error` | `G_error` | 在线，每查询×分区 |

[`rust/lance-index/src/vector/bq/storage.rs::raw_query_factor`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq/storage.rs) 是查询项 B 的直接映射。[`rust/lance-index/src/vector/bq/storage.rs::prepare_raw_query_context`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq/storage.rs) 与 [`rust/lance-index/src/vector/bq/storage.rs::dist_calculator_with_scratch`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq/storage.rs) 串起查询级数据的复用路径。

## 6. 概率误差界如何变成查询下界

1-bit 估计不是精确距离。构建端根据旋转残差与符号码的对齐程度生成 `error_factor`。令码维度为 \(D\)，则源码先计算

\[
\texttt{alignment}
=\frac{n(D/4)}{\beta^2},
\]

再计算

\[
\texttt{angular\_error}
=\sqrt{\frac{\max(\texttt{alignment}-1,0)}{D-1}}.
\]

基础误差为

\[
\sqrt n\times 1.9\times\texttt{angular\_error}.
\]

常量 [`rust/lance-index/src/vector/bq/transform.rs::RABIT_ERROR_EPSILON`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq/transform.rs) 对应 RaBitQ estimator 中的 \(\epsilon_0\)，在 `v10.0.0` 中精确为 `1.9`；L2 的 `error_factor` 再乘 2，Dot 保持基础值。若 \(D\le1\)、\(n\le0\) 或 \(\beta=0\)，因子直接为 0。[`rust/lance-index/src/vector/bq/transform.rs::error_factor_value`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq/transform.rs) 是这一计算的唯一实现。这是 RaBitQ 的高概率角误差因子，不是柯西–施瓦茨下的重构残差 \(2\lVert \mathbf{r}-\alpha \mathbf{u}\rVert\)。

查询侧的 `query_error` 提供另一半尺度：L2 对 `dist_q_c.max(0.0)` 开方；Dot 在有旋转质心时计算 \(\lVert \mathbf{W}\mathbf{q}-\mathbf{W}\mathbf{c}\rVert=\lVert \mathbf{W}(\mathbf{q}-\mathbf{c})\rVert\)，没有时对 `dist_q_c.max(0.0)` 开方。误差半径按离线/在线拆开后是

\[
\Delta(\mathbf{q},\mathbf{o})
=
\underbrace{\texttt{error\_factor}}_{\text{离线 E}}
\cdot
\underbrace{\texttt{query\_error}}_{\text{在线 F}}.
\]

由此得到用于剪枝筛选的查询下界：

\[
\texttt{lower\_bound}
=\hat{d}^{2}(\mathbf{q},\mathbf{o})-\Delta(\mathbf{q},\mathbf{o}).
\]

这里的“下界”具有概率语义，而非逐候选的确定性认证语义：\(\epsilon_0=1.9\) 控制针对随机旋转的高概率置信界，但置信界仍存在失效概率。只有在该置信事件成立时，`lower_bound` 才是真实距离的下界；源码没有把它变成对每个候选都必然成立的 certified bound，也没有在 DiskANN / HNSW 图上按 \([d_{\mathrm{lower}},d_{\mathrm{upper}}]\) 做三区域确定性剪枝。本文不从 `1.9` 杜撰具体失效概率数字；其统计含义应以 [RaBitQ estimator 文档](https://vectordb-ntu.github.io/RaBitQ-Library/rabitq/estimator/) 和论文为准。

只有同时满足以下条件时，这个概率下界才进入 top-k gating：

1. `approx_mode != ApproxMode::Fast`；
2. 元数据使用 `RabitQueryEstimator::RawQuery`；
3. `num_bits > 1`；
4. 索引确实带有 `error_factors`。

这些原因都绕过 lower-bound gating，但后续计算并不相同：`ApproxMode::Fast` 或 `num_bits == 1` 使用 1-bit raw-query 距离；`ResidualQuery` 使用其兼容估计路径；只有缺少 `error_factors` 的适用多比特 `RawQuery` 才在不做 gating 的情况下计算完整 ex-code 距离。[`rust/lance-index/src/vector/bq/storage.rs::raw_query_lower_bound_gating_disabled_reason`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq/storage.rs) 给出禁用原因，[`rust/lance-index/src/vector/bq/storage.rs::distance_all_with_scratch`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq/storage.rs) 决定相应 fallback。

在置信界成立事件内，候选若满足 `lower_bound >= query_upper_bound`，或在堆已满后满足 `lower_bound >= heap_threshold`，可据此剪枝；其余候选才读取 ex-code 做更精确的多比特重估。概率下界只负责拒绝，不作为最终候选距离写入 top-k 堆。

## 7. 多比特 ex-code 与增量重估

多比特模式没有丢弃 1-bit 符号码，而是在它旁边增加 \(e\) 位 ex-code。构建端先归一化每维绝对值

\[
a_i=\frac{|\mathbf{z}_i|}{\lVert \mathbf{z}\rVert},
\]

再由 `best_ex_rescale_factor` 搜索尺度 \(t\)，使量化码与归一化绝对值的内积目标尽可能大。`EX_TIGHT_START` 的九个精确值为

```text
[0.0, 0.15, 0.20, 0.52, 0.59, 0.71, 0.75, 0.77, 0.81]
```

量化时使用 `1.0e-5` 的 `EX_QUANTIZATION_EPSILON`，把 \(t a_i\) 向下取整并截到 \([0,2^e-1]\)。负坐标会按 \(e\) 位掩码取反 ex-code。记无符号 ex-code 为 \(x_i\)，最终一维中心化码值可写为

\[
\mathbf{g}_i=(b_i\ll e)+x_i-\left(2^e-\frac12\right),
\]

它仍以 0 为中心。[`rust/lance-index/src/vector/bq/builder.rs::best_ex_rescale_factor` 与 `quantize_ex_code`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq/builder.rs) 给出了尺度搜索和有符号码映射。

查询端不必重新解出完整的多比特中心化码 \(\mathbf{g}\)。它已有 `binary_ip`，只需计算

\[
\texttt{full\_dot}
=2^e\texttt{binary\_ip}
+\texttt{ex\_dist}
-\left(2^e-\frac12\right)\texttt{sum\_q},
\]

其中 `ex_dist` 是 \(\mathbf{W}\mathbf{q}\) 与无符号 ex-code 的内积。随后使用单独持久化的 `ex_scale_factors`、`ex_add_factors` 和同一个 `query_factor` 得到多比特距离。[`rust/lance-index/src/vector/bq/storage.rs::raw_query_multi_bit_exact_distance`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq/storage.rs) 对应这条公式。

因此“增量重估”在执行路径上非常具体：先对整分区批量算 1-bit `binary_ip` 和概率下界，只对 survivors 读取 ex-code、计算 `ex_dist` 并用 ex 因子重估。它不是先重构浮点向量，也不是对所有行无条件执行多比特计算。

ex-code 采用 64 维分块、按位宽专门设计的交错布局，末块补零。`ex_bits` 从 1 到 8 都有标量、x86 和 AArch64 解包/点积实现。[`rust/lance-index/src/vector/bq/ex_dot.rs::ex_dot_kernel`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq/ex_dot.rs) 描述了逐位宽布局。

## 8. 查询热路径和 SIMD 实现

查询热路径可以按数据流读成五步：

1. 旋转查询一次，构造每 4 个二值维度一组、16 个码型的距离表；
2. 用 FastScan 批量得到分区每行的 `binary_ip`；
3. 以 16 行为一组计算 1-bit 下界，并同时与查询上界、top-k 堆阈值比较；
4. 从两个剪枝掩码的零位枚举 survivors；
5. 对 survivors 调用 ex-code 点积内核，计算多比特距离并更新堆。

二值距离表可量化为 u8 或 u16；异常的非有限反量化范围会回退到精确浮点距离表计算。ex-code 点积则按 CPU 动态分派：x86 优先 AVX-512F，其次 AVX2+FMA；AArch64 使用基线 NEON；否则走标量实现。这里的 FMA 用于 survivor 的 ex-code 点积吞吐，是允许的。

剪枝内核却有更严格的数值合同。`prune_mask_kernel` 在 x86 上选择 AVX-512F 或 AVX2，其他平台使用可自动向量化的 16-lane portable 实现；每次返回 `pruned_upper_bound` 和 `pruned_heap` 两个 `u16` 掩码。比较使用 ordered-quiet `>=`，所以 NaN 不会被剪掉。

尤其要注意：**剪枝 SIMD 路径明确禁止 FMA**。它按标量 `raw_query_lower_bound` 的运算顺序逐次执行减、乘、加、减。FMA 只舍入一次，可能使临界候选的下界与标量路径不同，进而错误剪掉本应保留的行。源码因此用 `_mm256_mul_ps`/`_mm256_add_ps` 和对应 AVX-512 指令显式保持舍入行为；这与 ex-code 点积内核使用 FMA 并不矛盾，因为前者决定候选是否存活，后者只为已存活候选计算重估值。[`rust/lance-index/src/vector/bq/prune.rs::prune_mask_kernel`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq/prune.rs) 记录了这一正确性合同。

组开始时读取的堆阈值可能稍旧，但堆阈值只会收紧；旧值最多放过更多 survivor，不会多剪。逐行重估前还会用实时阈值复查。这让 16-lane 分类既减少控制流，又保持与标量扫描一致的剪枝语义。

## 9. 工程边界与容易误读之处

**公开构建类型的边界。** `v10.0.0` 的 Python 公开入口中，RaBitQ 构建类型只有 `IVF_RQ`：它把 `num_bits`、`rabitq_model` 等参数组装为 `VectorIndexParams::with_ivf_rq_params`；公开类型列表中没有 `IVF_HNSW_RQ`。[`python/src/dataset.rs::Dataset::create_index`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/python/src/dataset.rs) 与 [`python/src/dataset.rs::prepare_vector_index_params`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/python/src/dataset.rs) 展示了该边界。

更底层的组合逻辑虽然能识别 `Hnsw + Rabit`，但尝试为它创建新 segment 会返回：

```text
Cannot build a fresh IVF_HNSW_RQ segment: this index type is unsupported
```

而 RQ storage 的 `dist_calculator_from_id` 也仍未实现，源码注释明确指出 HNSW_RABIT 依赖它。因此准确说法是：**该发布的公开新建路径只有 IVF_RQ；fresh IVF_HNSW_RQ 明确 unsupported**，不能因为内部枚举能表达这种组合就宣称它可公开构建。[`rust/lance/src/index/vector.rs::fresh_vector_segment_params`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance/src/index/vector.rs) 给出了错误分支。

此外还有几处常见误读：

- `num_bits=1` 有估计距离，但没有 ex-code，也不会启用多比特下界 gating。
- `ApproxMode::Fast` 会主动绕过 gating，并在 raw-query 多比特场景退回 1-bit 距离；“Fast”在这里是近似模式，不是 `Fast` 随机旋转，两者不要混淆。
- `error_factor` 不是最终误差；最终 margin 必须再乘分区相关的 `query_error`。
- `add_factor` 不是纯粹的全局偏置，它含每行残差与所属质心的关系。
- ex-code 点积可以使用 FMA；禁止 FMA 的是决定生死的 lower-bound 剪枝内核。
- `Matrix` 与 `Fast` 是同一旋转接口的两个实现，不代表两个不同的距离估计器；新建元数据都选择 `RawQuery` estimator。

## 10. 总结

Lance `v10.0.0` 的 RaBitQ 路径可以浓缩为一个分层估计系统：IVF 产生残差和候选分区；随机旋转把能量打散；1-bit 符号码提供可批量查表的基础内积；每行 `scale/add/error` 因子与每分区 `query/query_error` 因子把内积变成带概率裕量、仅在置信事件成立时有效的高概率距离下界；多比特 ex-code 只为未被该概率下界剪掉的候选增量重估。

性能设计与数值正确性在这里相互约束。距离表和 ex-code 点积尽量利用 AVX-512、AVX2/FMA 或 NEON；真正决定剪枝的 16-lane 内核却刻意不用 FMA，以复现标量舍入顺序。理解这条边界，也就理解了该实现为何既保存两套因子、两层码，又把“粗估—下界—精估”拆成三个明确阶段。

## 参考资料

1. [RaBitQ 论文（DOI）](https://doi.org/10.1145/3654970)
2. [RaBitQ-Library estimator 文档](https://vectordb-ntu.github.io/RaBitQ-Library/rabitq/estimator/)
3. [Lance `v10.0.0` 标签](https://github.com/lance-format/lance/tree/v10.0.0)
4. [Lance `v10.0.0` 源码树（固定提交）](https://github.com/lance-format/lance/tree/95f2f36b22043c3face00afe088c34e0742d01df)
5. [`rust/lance-index/src/vector/bq/builder.rs::RabitQuantizer`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq/builder.rs)
6. [`rust/lance-index/src/vector/bq/transform.rs::compute_raw_query_factors`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq/transform.rs)
7. [`rust/lance-index/src/vector/bq/storage.rs::distance_all_with_scratch`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq/storage.rs)
8. [`rust/lance-index/src/vector/bq/prune.rs::prune_mask_kernel`](https://github.com/lance-format/lance/blob/95f2f36b22043c3face00afe088c34e0742d01df/rust/lance-index/src/vector/bq/prune.rs)
9. [RaBitQ-Library 源码](https://github.com/VectorDB-NTU/RaBitQ-Library)
