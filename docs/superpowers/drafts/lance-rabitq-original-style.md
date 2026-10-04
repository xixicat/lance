# 揭秘 Lance 向量索引中的 RaBitQ：从数学推导到工程实现

> 源码基线
>
> 本文只讨论 Lance 社区正式标签
> [`v10.0.0`](https://github.com/lance-format/lance/tree/v10.0.0)，对应提交
> `95f2f36b22043c3face00afe088c34e0742d01df`。源码路径和符号均以该标签为准，
> 不借用其他版本的实现来补充或推断行为。

在向量检索中，压缩从来不只是“把 `float32` 换成几个 bit”。真正棘手的问题是：
当原向量不再完整保存在索引里时，如何仍然快速、可靠地比较候选距离？

RaBitQ 的答案可以概括为三步：先把残差随机旋转，让方向信息更均匀地分散到各个维度；
再用符号 bit 构造内积估计器；最后用概率误差界决定哪些候选值得读取更多 bit 做精排。
Lance `v10.0.0` 把这条链路拆成了可向量化的查询热路径。下文按同一顺序展开：
旋转与编码 → 交叉项估计 → 离线/在线因子 → 误差下界 → 多比特精排。

## 1. 算法破局点：随机正交旋转

设数据向量为 \(\mathbf{o}\)，维度为 \(D\)，它所属 IVF 分区的中心为 \(\mathbf{c}\)，残差为

$$
\mathbf{r}=\mathbf{o}-\mathbf{c}.
$$

RaBitQ 不直接量化 \(\mathbf{r}\)，而是先施加旋转 \(\mathbf{W}\)，得到 \(\mathbf{z}=\mathbf{W}\mathbf{r}\)。理论模型把 \(\mathbf{W}\) 视为
正交变换：它不改变残差范数和内积，却会打散“能量集中在少数坐标”的坏情形。旋转后
只保留每个坐标的符号，构造量化向量 \(\mathbf{u}\)（单个字母，以免被读成两个向量的乘积）：

$$
\mathbf{u}_i=
\begin{cases}
+0.5, & \mathbf{z}_i\text{ 的符号为正},\\
-0.5, & \text{否则}.
\end{cases}
$$

于是 \(\lVert \mathbf{u}\rVert^2=D/4\)，基础码每维只需 1 bit。\(\mathbf{u}\) 是后续内积比值估计器的
方向代理，不是用来把 \(\mathbf{r}\) 欧氏重建成某个 \(\alpha \mathbf{u}\)。原始的 \(\lVert \mathbf{r}\rVert^2\)
会作为数据侧辅助量精确保留。

Lance 提供两种彼此独立的旋转选项：

- `RQRotationType::Matrix` 明确构造稠密随机正交矩阵。
- 默认的 `RQRotationType::Fast` 与 RaBitQ 参考库对齐：`apply_fast_rotation` 固定四轮。
  若 \(D\) 是 2 的幂，每轮依次做随机符号翻转、FWHT 和归一化；否则四轮交替归一化首尾
  的 2 的幂 FWHT 子窗口，并做 Hadamard/Kac 混合，最后乘 \(0.25\) 补偿。IVF_RQ 要求
  \(D\) 可被 8 整除。精确算术下该流水线保持范数；源码因 `f32` 舍入而注释为
  `approximately orthonormal`。Fast 不物化稠密矩阵，也不是 Matrix 路径中的某一轮。

旋转只是换坐标系。真正决定距离估计质量的，是下一节如何使用这些符号。

## 2. 距离估计：交叉项、离线因子与在线热路径

以 L2 为例。极化恒等式把距离拆成两项精确量和一项交叉内积：

$$
\lVert \mathbf{q}-\mathbf{o}\rVert^2
=\lVert \mathbf{q}-\mathbf{c}\rVert^2+\lVert \mathbf{r}\rVert^2-2\langle \mathbf{q}-\mathbf{c},\mathbf{r}\rangle.
$$

\(\lVert \mathbf{q}-\mathbf{c}\rVert^2\) 在查询时按分区计算，\(\lVert \mathbf{r}\rVert^2\) 在建索引时预存。
只需近似 \(\langle \mathbf{q}-\mathbf{c},\mathbf{r}\rangle\)。

因为 \(\mathbf{W}\) 正交，内积不变。把旋转后的残差和查询残差写成单位向量
\(\hat{\mathbf{z}}=\mathbf{W}\mathbf{r}/\lVert \mathbf{r}\rVert\)、\(\hat{\mathbf{y}}=\mathbf{W}(\mathbf{q}-\mathbf{c})/\lVert \mathbf{q}-\mathbf{c}\rVert\)，交叉项就是余弦：

$$
\langle \mathbf{q}-\mathbf{c},\mathbf{r}\rangle=\langle \mathbf{W}(\mathbf{q}-\mathbf{c}),\mathbf{W}\mathbf{r}\rangle=\lVert \mathbf{r}\rVert\lVert \mathbf{q}-\mathbf{c}\rVert\langle\hat{\mathbf{z}},\hat{\mathbf{y}}\rangle.
$$

\(\mathbf{u}\) 只代理 \(\hat{\mathbf{z}}\) 的方向。RaBitQ 用它同时测量两个单位向量，再取比值估计余弦
——与官方 estimator 中的 \(\langle\bar{\mathbf{o}},\mathbf{q}\rangle/\langle\bar{\mathbf{o}},\mathbf{o}\rangle\) 相同。
\(\lVert \mathbf{q}-\mathbf{c}\rVert\) 在代回交叉项时消去，得到

$$
\langle \mathbf{q}-\mathbf{c},\mathbf{r}\rangle
\approx
\lVert \mathbf{r}\rVert^2
\frac{\langle \mathbf{W}(\mathbf{q}-\mathbf{c}),\mathbf{u}\rangle}{\langle \mathbf{W}\mathbf{r},\mathbf{u}\rangle}.
$$

该比值对 \(\mathbf{u}\) 的整体尺度不敏感，所以坐标取 \(\pm 0.5\) 只是为了和 \(\{0,1\}\) 存储码
中心化一致，并不要求 \(\lVert \mathbf{u}\rVert=1\)。

查询热路径再把 \(\langle \mathbf{W}(\mathbf{q}-\mathbf{c}),\mathbf{u}\rangle=\langle \mathbf{W}\mathbf{q},\mathbf{u}\rangle-\langle \mathbf{W}\mathbf{c},\mathbf{u}\rangle\) 代回
L2 展开。与质心有关的一半并进每行离线因子，热路径只对每条码计算 \(\langle \mathbf{W}\mathbf{q},\mathbf{u}\rangle\)。
按 **Indexing / Online** 标出后：

$$
\begin{aligned}
\hat{d}^{2}(\mathbf{q},\mathbf{o})
&=
\underbrace{\lVert \mathbf{q}-\mathbf{c}\rVert^{2}}_{\text{在线 B：query}}
+
\underbrace{\Biggl(\lVert \mathbf{r}\rVert^{2}
+\frac{2\lVert \mathbf{r}\rVert^{2}\langle \mathbf{W}\mathbf{c},\mathbf{u}\rangle}{\langle \mathbf{W}\mathbf{r},\mathbf{u}\rangle}\Biggr)}_{\text{离线 A：add}}
\\
&\quad+
\underbrace{\Biggl(\frac{-2\lVert \mathbf{r}\rVert^{2}}{\langle \mathbf{W}\mathbf{r},\mathbf{u}\rangle}\Biggr)}_{\text{离线 C：scale}}
\cdot
\underbrace{\langle \mathbf{W}\mathbf{q},\mathbf{u}\rangle}_{\text{在线交互 D}}
\end{aligned}
$$

这正是 `transform.rs` 里 `compute_raw_query_factors` 的 L2 拆分。注意 A **不是**单独的
\(\lVert \mathbf{r}\rVert^{2}\)：质心交叉项已经折进去，D 才能写成对原始旋转查询的内积。B 依赖
当前 IVF 中心，是“查询 × 分区”级常量，不是全库一条。D 是唯一必须对每个候选求的码内积。

\(\mathbf{u}_i=b_i-1/2\)，因此 D 落到 \(\{0,1\}\) 码的 SIMD 点积上：

$$
\langle \mathbf{W}\mathbf{q},\mathbf{u}\rangle
=\texttt{binary\_ip}-\frac12\texttt{sum\_q},\qquad
\texttt{binary\_ip}=\sum_{i=1}^{D}b_i(\mathbf{W}\mathbf{q})_i.
$$

`sum_q` 每个查询算一次；`binary_ip` 才是分区扫描里的高频查表内积。在线计算因此就是

```text
binary_dot = binary_ip - 0.5 * sum(rotated_query)   # 在线交互 D
estimate   = add + query + scale * binary_dot       # A + B + C·D
```

每行随索引固化的 1-bit 布局为

```text
+------------------------+------------------+---------------------+---------------------+
| 1-bit bitmap (D bits)  | add (float32)    | scale (float32)     | error (float32)     |
+------------------------+------------------+---------------------+---------------------+
```

`add` / `scale` / `error` 对应 RaBitQ-Library 的 `F_add` / `F_rescale` / `F_error`；
`query` 与下一节的 `query_error` 对应 `G_add` 与 `G_error`。

| 项 | 公式（L2） | Lance | 阶段 |
|---|---|---|---|
| A | \(\lVert \mathbf{r}\rVert^{2}+2\lVert \mathbf{r}\rVert^{2}\langle \mathbf{W}\mathbf{c},\mathbf{u}\rangle/\langle \mathbf{W}\mathbf{r},\mathbf{u}\rangle\) | `add` | 离线，每行 |
| B | \(\lVert \mathbf{q}-\mathbf{c}\rVert^{2}\) | `query` | 在线，每查询×分区 |
| C | \(-2\lVert \mathbf{r}\rVert^{2}/\langle \mathbf{W}\mathbf{r},\mathbf{u}\rangle\) | `scale` | 离线，每行 |
| D | \(\langle \mathbf{W}\mathbf{q},\mathbf{u}\rangle\) | `binary_dot` | 在线，每候选 |
| E | \(2\lVert \mathbf{r}\rVert\epsilon_0\cdot\text{angular\_error}\) | `error` | 离线，每行 |
| F | \(\lVert \mathbf{q}-\mathbf{c}\rVert\) | `query_error` | 在线，每查询×分区 |

查询向量只需旋转一次；每条数据只携带三个浮点因子和二值码。误差来自方向估计，不来自把
\(\lVert \mathbf{r}\rVert^{2}\) 当成量化范数。

## 3. 概率误差界与下界剪枝

1-bit 估计需要知道可能偏差多少，才能安全地筛候选。构建端用旋转残差与符号码的对齐程度
生成数据侧误差因子：

$$
\text{alignment}
=\frac{\lVert \mathbf{r}\rVert^2(D/4)}{\langle \mathbf{W}\mathbf{r},\mathbf{u}\rangle^2},\qquad
\text{angular\_error}
=\sqrt{\frac{\max(\text{alignment}-1,0)}{D-1}}.
$$

L2 下 \(\epsilon_0=1.9\)，误差半径按离线/在线拆开为

$$
R_{\mathrm{L2}}
=
\underbrace{\bigl(2\lVert \mathbf{r}\rVert\epsilon_0\cdot\text{angular\_error}\bigr)}_{\text{离线 E：error}}
\cdot
\underbrace{\lVert \mathbf{q}-\mathbf{c}\rVert}_{\text{在线 F：query\_error}}.
$$

E 随每行写入；F 与 B 一样在探测该分区时计算一次。在线只做一次乘法：

$$
\Delta(\mathbf{q},\mathbf{o})=\text{error\_factor}\cdot\text{query\_error},\qquad
\text{lower\_bound}=\hat{d}^{2}(\mathbf{q},\mathbf{o})-\Delta(\mathbf{q},\mathbf{o}).
$$

这是 RaBitQ 理论的高概率误差控制，不是柯西–施瓦茨的逐点确定性包络，也不是
\(\lVert \mathbf{r}-\alpha \mathbf{u}\rVert\) 那种重构残差。Lance 用这个下界与查询上界、当前 top-k 堆阈值
比较：下界已经不可能胜出的候选可跳过 ex-code 重估。这是 `IVF_RQ` 分区扫描上的 gating，
不是 DiskANN / HNSW 图上的三区域确定性剪枝。

`v10.0.0` 中，gating 只用于满足条件的多比特 `IVF_RQ` 扫描：估计器必须是 `RawQuery`，
`num_bits > 1`，误差因子列必须存在，且查询的 `ApproxMode` 不能是 `Fast`。这里的
`ApproxMode::Fast` 与第一节的 `RQRotationType::Fast` 不是同一概念。禁用 gating 只表示
不再用二值下界筛候选，并不自动改算完整多比特距离：`RawQuery` 在 `ApproxMode::Fast` 或
`num_bits == 1` 时返回 1-bit 距离，`ResidualQuery` 走兼容路径；只有适用的多比特
`RawQuery` 才继续算 ex-code，并在 gating 生效的 top-k 扫描中只为幸存者做这一步。

## 4. 多比特扩展与工程边界

1-bit 只回答正负。精度预算允许时，Lance 在符号 bit 之外增加 ex-code，而不是在全局固定
区间上做普通均匀量化。若每维总位数为 `num_bits`，则 \(\text{ex\_bits}=\text{num\_bits}-1\)。

`best_ex_rescale_factor` 先对**每条向量**扫描量化阈值，选择使码与该向量内积对齐更好的
缩放 \(t\)；随后 `quantize_ex_code` 才写入各维 ex-code。自适应缩放发生在落入对称网格
之前，不能描述成所有向量共享同一量程。

完整码为 \((\text{sign\_code}\ll\text{ex\_bits})+\text{ex\_code}\)，中心化后减去
\(2^{\text{ex\_bits}}-0.5\)，网格以 0 为中心、落在半整数点上。负坐标的 ex-code 在对应
bit mask 内取反。完整码另有一套 `ex_add_factors` / `ex_scale_factors`。

下界 gating 生效时，增量体现在计算顺序：先用符号码得到 `binary_estimate`，再用上一节
的高概率下界筛选，只对幸存者读取 ex-code 并重估距离。`v10.0.0` 接受每维 1 到 9 bit；
`num_bits=1` 时没有 ex-code，1-bit 路径本身就是估计器。其他模式按第三节的 fallback
走，不能概括为都先经过这道门控。

公开新建路径同样需要限定：`VectorIndexParams::ivf_rq` 与 `with_ivf_rq_params` 创建的是
IVF + Flat + Rabit；fresh 的 IVF + HNSW + Rabit 会返回 unsupported。源码里为读取或分派
保留的分支，不能扩大成公开支持的索引类型。

压缩率也需要准确表述。相对每维 32-bit 的 `float32` payload，纯 1-bit 主码在位数上是
理论 32 倍；真实索引还包含行号、IVF 分区、残差范数、add/scale/error、旋转元数据和对齐，
多比特模式还有 ex-code，不能把端到端大小直接称为 32 倍。RQ 量化器的 `sample_size()` 为 0，
量化阶段不需要训练码本；完整 `IVF_RQ` 仍需要 IVF 聚类中心。

## 5. 总结

RaBitQ 的精髓不是“用 1 bit 重构向量”，而是用随机旋转后的符号方向建立一个可校准的
内积估计器：

- \(\lVert \mathbf{r}\rVert^2\) 始终作为精确项保留，并与质心交叉项一起折进离线 `add`；
- 离线三项是 `add` / `scale` / `error`，在线三项是 `query`、`query_error` 和
  \(\langle \mathbf{W}\mathbf{q},\mathbf{u}\rangle\)；
- L2 误差半径同时包含系数 \(2\)、\(\lVert \mathbf{r}\rVert\) 和 \(\lVert \mathbf{q}-\mathbf{c}\rVert\)；
- 多比特编码先逐向量优化 \(t\)，再落入对称半整数网格；
- 高概率下界只在满足条件的多比特 `IVF_RQ` 扫描中承担 gating。

Lance `v10.0.0` 把这些对象映射成数据侧因子、查询侧因子和两阶段扫描。理解这套边界后，
1-bit 的价值就不再只是“更小”，而是以极低成本给后续精排提供有理论依据的候选顺序。

## 参考资料

1. Gao, J. and Long, C. *RaBitQ: Quantizing High-Dimensional Vectors with a
   Theoretical Error Bound for Approximate Nearest Neighbor Search*：
   <https://doi.org/10.1145/3654970>
2. RaBitQ-Library Estimator 文档：
   <https://vectordb-ntu.github.io/RaBitQ-Library/rabitq/estimator/>
3. Lance `v10.0.0` 源码标签：
   <https://github.com/lance-format/lance/tree/v10.0.0>
4. 本文对应的主要源码位置：`rust/lance-index/src/vector/bq/rotation.rs`、
   `rust/lance-index/src/vector/bq/builder.rs`、
   `rust/lance-index/src/vector/bq/transform.rs`、
   `rust/lance-index/src/vector/bq/storage.rs`、
   `rust/lance-index/src/vector/bq/prune.rs` 与 `rust/lance/src/index/vector.rs`。
