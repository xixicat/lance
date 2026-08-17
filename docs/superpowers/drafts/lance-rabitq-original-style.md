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
Lance `v10.0.0` 不仅实现了这条数学链路，还把它拆成了可向量化的查询热路径。

## 1. 算法破局点：随机正交旋转

设数据向量为 \(o\)，它所属 IVF 分区的中心为 \(c\)，残差为

$$
r=o-c.
$$

RaBitQ 不直接量化 \(r\)，而是先施加旋转 \(W\)：

$$
z=Wr.
$$

概念上，正交旋转不改变残差范数，却会打散“能量集中在少数坐标”的坏情形。旋转后，
只保留每个坐标的符号，构造

$$
bf_i=
\begin{cases}
+0.5, & z_i\text{ 的符号为正},\\
-0.5, & \text{否则}.
\end{cases}
$$

于是 \(\lVert bf\rVert^2=d/4\)，而一条向量的基础码只需每维 1 bit。这里的关键不是把
\(r\) 重构成某个缩放后的 \(bf\)，而是让 \(bf\) 成为后续**内积比值估计器**的方向代理。
原始的 \(\lVert r\rVert^2\) 会作为数据侧辅助量精确保留，不需要从二值码的范数反推。

Lance 提供两种彼此独立的旋转选项：

- `RQRotationType::Fast` 是默认选项。`rotation.rs` 中的
  `apply_fast_rotation` 固定执行四轮 FHT-Kac 流程。对于 2 的幂维度，每轮依次执行随机
  符号翻转、FWHT 和缩放；对于非 2 的幂维度，四轮交替处理首尾的 2 的幂窗口，并加入
  Kac 式成对混合，最后补偿缩放。它不物化稠密矩阵。
- `RQRotationType::Matrix` 是另一种选项，使用单独的稠密随机正交矩阵路径。它不是
  Fast 流程中的某一轮，也不应与 FHT-Kac 混为一谈。

这一步换来了一个更适合符号量化的坐标系。接下来真正决定距离估计质量的，是如何使用
这些符号，而不是如何“还原”原向量。

## 2. 距离估计器：精确项与近似交叉项

以 L2 距离为例。令查询相对分区中心的向量为 \(q-c\)，则

$$
\begin{aligned}
\lVert q-o\rVert^2
&=\lVert q-c-r\rVert^2\\
&=\lVert q-c\rVert^2+\lVert r\rVert^2-2\langle q-c,r\rangle.
\end{aligned}
$$

这个展开式中，前两项都可以精确计算或预存：

- 查询侧的 \(\lVert q-c\rVert^2\) 是 `query_factor`；
- 数据侧的 \(\lVert r\rVert^2\) 是预存的残差平方范数。

只有交叉项 \(\langle q-c,r\rangle\) 需要近似。旋转保持内积关系，RaBitQ 用

$$
\langle q-c,r\rangle
\approx
\lVert r\rVert^2
\frac{\langle W(q-c),bf\rangle}{\langle Wr,bf\rangle}.
$$

注意 \(W(q-c)=Wq-Wc\)，所以查询热路径实际需要的旋转坐标内积是
\(\langle Wq,bf\rangle\) 和 \(\langle Wc,bf\rangle\)，而不是把旋转前后的量混写：

$$
\langle W(q-c),bf\rangle
=\langle Wq,bf\rangle-\langle Wc,bf\rangle.
$$

记

$$
\text{binary\_res\_dot}=\langle Wr,bf\rangle,\qquad
\text{binary\_cent\_dot}=\langle Wc,bf\rangle.
$$

`transform.rs` 中的 `compute_raw_query_factors` 把 L2 估计器整理为三个部分：

$$
\begin{aligned}
\text{add}
&=\lVert r\rVert^2+
\frac{2\lVert r\rVert^2\text{binary\_cent\_dot}}
{\text{binary\_res\_dot}},\\
\text{scale}
&=\frac{-2\lVert r\rVert^2}{\text{binary\_res\_dot}},\\
\text{query}
&=\lVert q-c\rVert^2.
\end{aligned}
$$

对符号 bit 做点积后，`storage.rs` 中的查询计算等价于

```text
binary_dot = binary_ip - 0.5 * sum(rotated_query)
estimate = binary_dot * scale + add + query
```

其中 `binary_dot` 正是 \(\langle Wq,bf\rangle\) 的实现形式。这样组织有两个工程收益：
查询只需旋转一次；每条数据只携带自己的 `add`、`scale` 和二值码，就能复用同一份查询
结果。更重要的是，这个公式清楚地区分了**精确的平方范数项**与**近似的交叉内积项**：
误差来自方向估计，不来自把 \(\lVert r\rVert^2\) 当成量化范数。

## 3. 概率误差界与下界剪枝

近似距离只有在知道“可能偏差多少”时，才能安全地参与候选筛选。Lance 为每条数据计算
方向对齐量

$$
\text{alignment}
=
\frac{\lVert r\rVert^2\lVert bf\rVert^2}
{\langle Wr,bf\rangle^2},
\qquad
\lVert bf\rVert^2=\frac d4,
$$

以及

$$
\text{angular\_error}
=
\sqrt{
\frac{\max(\text{alignment}-1,0)}
{d-1}
}.
$$

在 `transform.rs` 的 L2 分支中，常数 \(\epsilon_0=1.9\)，数据侧误差因子为

$$
\text{error\_factor}
=2\lVert r\rVert\epsilon_0\cdot\text{angular\_error}.
$$

查询侧还必须乘上

$$
\text{query\_error}=\lVert q-c\rVert.
$$

因此完整的 L2 误差半径是

$$
R_{\mathrm{L2}}
=
2\lVert r\rVert\epsilon_0
\sqrt{
\frac{
\max\left(
\frac{\lVert r\rVert^2(d/4)}
{\langle Wr,bf\rangle^2}-1,
0
\right)}
{d-1}
}
\lVert q-c\rVert.
$$

最终查询下界为

$$
\text{lower\_bound}
=\text{binary\_estimate}
-\text{error\_factor}\cdot\text{query\_error}.
$$

这里必须强调“概率”二字：它来自 RaBitQ 理论的高概率误差控制，不是对每个向量都成立的
确定性承诺。Lance 用这个下界与查询上界、当前 top-k 堆阈值比较；下界已经不可能胜出的
候选可跳过后续 ex-code 重估，其余候选才进入多比特精排。

这条 gating 也有明确边界。在 `v10.0.0` 中，它用于满足条件的多比特 `IVF_RQ`
分区扫描：查询估计器必须是 `RawQuery`，`num_bits > 1`，误差因子列必须存在，而且查询
的 `ApproxMode` 不能是 `Fast`。这里的查询 `ApproxMode::Fast` 与第一节的
`RQRotationType::Fast` 是两个不同概念。条件不满足时，代码绕过下界 gating，直接计算
完整距离。

从公开新建路径看，`VectorIndexParams::ivf_rq` 和
`VectorIndexParams::with_ivf_rq_params` 创建的是 IVF、Flat 子索引与 Rabit 量化的组合；
fresh 的 IVF + HNSW + Rabit 组合会明确返回 unsupported 错误。源码中为读取、重映射或
泛型分派保留的分支，不能据此扩大成公开支持的索引类型。

## 4. 多比特扩展：自适应网格与增量估计

1-bit 只回答“正还是负”。当精度预算允许每维使用更多 bit 时，Lance 不会简单地在一个
全局固定区间上做普通均匀量化，而是在符号 bit 之外增加 ex-code。

若每维总位数为 `num_bits`，则

$$
\text{ex\_bits}=\text{num\_bits}-1.
$$

`builder.rs` 的 `best_ex_rescale_factor` 会先针对**每条向量**处理归一化后的绝对坐标，
扫描量化阈值，选择使码向量与该向量内积对齐更好的缩放 \(t\)。只有确定这个逐向量的
\(t\) 后，`quantize_ex_code` 才计算各维 ex-code。也就是说，自适应缩放发生在落入对称
网格之前，不能把它描述成所有向量共享同一量程。

设符号码为 `sign_code`，额外码为 `ex_code`，完整码是

$$
\text{full\_code}
=(\text{sign\_code}\ll\text{ex\_bits})+\text{ex\_code}.
$$

用于点积的中心化值为

$$
\text{centered\_code}
=\text{full\_code}
-\left(2^{\text{ex\_bits}}-0.5\right).
$$

因此网格以 0 为中心、落在半整数点上。负坐标的 ex-code 会在对应 bit mask 内取反，
与符号位共同形成对称编码。Lance 接着为完整码计算独立的 `ex_add_factors` 和
`ex_scale_factors`。

查询时，“增量”体现在计算顺序，而不是数学上凭空追加一个误差修正：

1. 先只读符号码，用二值 FastScan 得到 `binary_estimate`；
2. 用上一节的高概率下界筛选候选；
3. 仅对幸存者读取 ex-code，组合 `full_code`，再用 ex-code 对应的 add/scale 因子重估
   距离并更新 top-k 堆。

`v10.0.0` 接受每维 1 到 9 bit；只有 `num_bits > 1` 时才有 ex-code。1-bit 路径仍然是
估计器本身，多比特路径则把它变成低成本的第一道候选门。

这样的压缩也需要准确表述。相对于每维 32-bit 的 `float32` payload，纯 1-bit 主码在
位数上具有理论 32 倍压缩比；真实索引还包含行号、IVF 分区、残差范数、add/scale/error
因子、旋转元数据和存储对齐，多比特模式还包含 ex-code，所以端到端索引大小不能直接
宣称为 32 倍。类似地，RQ 量化器的 `sample_size()` 为 0，可以称其量化阶段不需要训练
码本；但完整 `IVF_RQ` 仍需要 IVF 聚类中心，不能把整个索引称为无需训练。

## 5. 总结

RaBitQ 的精髓不是“用 1 bit 重构向量”，而是用随机旋转后的符号方向建立一个可校准的
内积估计器：

- \(\lVert r\rVert^2\) 始终作为精确项保留；
- \(\langle Wq,bf\rangle\) 与 \(\langle Wc,bf\rangle\) 共同近似交叉项；
- 完整 L2 误差半径同时包含系数 \(2\)、\(\lVert r\rVert\) 和
  \(\lVert q-c\rVert\)；
- 多比特编码先逐向量优化 \(t\)，再落入对称半整数网格；
- 高概率下界只在满足条件的多比特 `IVF_RQ` 扫描中承担 gating，而不是无条件保证。

Lance `v10.0.0` 把这些数学对象映射成了清晰的数据侧因子、查询侧因子和两阶段扫描。
理解这套边界后，1-bit 的价值就不再只是“更小”，而是以极低成本给后续精排提供有理论
依据的候选顺序与筛选信号。

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
