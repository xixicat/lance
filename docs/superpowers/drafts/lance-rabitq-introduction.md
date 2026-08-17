# 用一篇文章理解 Lance RaBitQ

向量检索的瓶颈常常不是算力，而是内存带宽：候选向量越多，从存储中搬运的浮点数就越多。RaBitQ 的目标，是把每个坐标压成很少的比特，同时仍能快速估算查询向量与候选向量的距离。Lance v10.0.0 将它用于 `IVF_RQ`；本文仅以该标签对应的提交 `95f2f36b22043c3face00afe088c34e0742d01df` 为源码基线，下面以 L2 距离说明核心思路。

## 1. 普通二值量化为什么不够

最直接的二值量化只记录每个坐标的正负号。它很省空间，也能把内积变成位运算，但效果依赖坐标轴：同一个向量换一组坐标后，符号分布可能完全不同。若信息集中在少数坐标，丢掉幅值会造成很大的距离误差。

`IVF_RQ` 先用 IVF 把向量分到聚类中心附近。设原向量为 \(o\)，它所属的中心为 \(c\)，RaBitQ 实际编码的是残差 \(r\)。再用随机正交旋转 \(W\) 把残差变成 \(z\)：

```text
r = o - c, z = Wr
```

减去中心缩小了要编码的范围；旋转则把方向信息更均匀地摊到各坐标上，使“只留符号”不再过度依赖原始坐标轴。

## 2. 随机旋转解决什么问题

正交旋转不改变长度和内积，却让符号编码面对一个随机化后的方向。这里的“随机”很重要：RaBitQ 给出的误差控制是对随机旋转而言的高概率结论，不是每个候选都绝对成立的确定性保证。直观上，维度越高，单个坐标异常地主导结果的机会越小，符号向量越能代表整体方向。

Lance v10.0.0 默认采用矩阵无关的快速旋转：四轮随机符号翻转与 FHT/Kac 混合；也提供稠密正交矩阵选项。旋转模型在建索引时生成并持久化，查询使用同一模型，不能给不同索引分段随意换一套旋转后再把它们当成同一物理模型合并。

## 3. 只保存符号，如何估算距离

对查询向量 \(q\)，L2 距离可以按中心拆开：

```text
d²(q,o) = ||q-c||² + ||r||² - 2<q-c,r>
```

前两项都可以精确获得：查询到中心的距离在查询时计算，残差长度则在建索引时折入每行校正因子。真正需要近似的只有交叉内积。

令 \(bf\) 是旋转后残差 \(z=Wr\) 的符号向量，每个坐标取正或负的半单位值。因为正交旋转保持内积，可用旋转后的查询与 \(bf\) 的内积来估计交叉项，并用建索引时保存的比例校正幅值：

```text
<q-c,r> ≈ ||r||² <W(q-c),bf> / <Wr,bf>
```

因此查询阶段无需重构完整浮点残差：读取打包的符号码和少量浮点校正因子，就能得到距离估计。

## 4. 误差范围如何帮助减少多比特计算

概率界的工程价值不只是告诉我们“估计可能差多少”，还可以把误差半径变成保守下界。Lance 为候选保存与其量化误差有关的 `error_factor`；查询侧的尺度是 \(\lVert q-c\rVert\)。两者相乘得到本次查询的误差余量：

```text
lower_bound = estimate - error_factor * ||q-c||
```

如果这个下界已经不可能进入当前 top-k，候选就可跳过更贵的多比特距离计算。这里仍然是高概率剪枝，而非数学上“永不误剪”的严格下界。Lance v10.0.0 也不会无条件启用它：该门控用于具备 raw-query 估计器和误差因子的多比特 `IVF_RQ` 正常扫描；单比特、快速近似模式或缺少误差因子时会绕过门控。

## 5. 精度不够时怎样增加位数

一位编码只保存符号。Lance v10.0.0 允许每维使用 1 到 9 位：第一位仍走紧凑的符号码路径，其余位存入额外的 ex-code。建索引时，每个向量会先选择适合自身幅值分布的缩放，再把绝对值放到对称的半整数网格上量化。在第 4 节所述门控生效的多比特 `RawQuery`（直接旋转原始查询、再用分区质心因子修正的估计路径）扫描中，查询只对通过下界筛选的候选计算 ex-code 内积并重估距离；其他模式走各自的 fallback 路径。

增加位数通常会提高距离精度，但会同步增加索引体积和计算量，所以它不是免费的“精度旋钮”。合理做法是先用单比特结果和业务召回要求建立基线，再逐步增加位数并实测延迟、召回与索引大小。

## 6. 使用时需要记住的限制

**压缩率不是简单的 32 倍。** 从 32 位浮点坐标到单个符号位，纯编码部分的理论倍率是 32 倍；真实索引每行还要保存 8 字节 row ID。v10.0.0 的 raw-query 单比特布局还保存三个 4 字节浮点因子，所以有 20 字节固定开销。以 768 维为例，符号码占 96 字节，整行至少 116 字节，而原始 float32 向量为 3072 字节，约为 26.5 倍；文件元数据、IVF 中心和对齐还会继续影响总大小。多比特模式还会增加 ex-code 以及两项浮点校正因子。

**RaBitQ 自身不需要从样本学习码本，但 IVF 仍要训练。** Lance 的 RaBitQ 量化器无需采样数据来训练；然而 `IVF_RQ` 的 IVF 部分必须通过 k-means 得到聚类中心，或由调用方提供已经训练好的中心。把 RaBitQ 称为“免训练”时，必须限定为量化器，而不能扩大到整个索引。

最后，近似检索质量还受 IVF 分区数、查询探测的分区数、数据分布和位数共同影响。不要只根据理论压缩率选配置，应在自己的数据和查询集上同时评估召回、延迟、建索引成本与磁盘占用。

## 参考资料

- [RaBitQ-Library：Estimator](https://vectordb-ntu.github.io/RaBitQ-Library/rabitq/estimator/)：距离估计、概率误差界和增量估计的推导。
- [Lance v10.0.0 tag 页面](https://github.com/lance-format/lance/tree/v10.0.0)：本文核对源码的固定版本入口。
- [`docs/src/format/index/vector/index.md`](https://github.com/lance-format/lance/blob/v10.0.0/docs/src/format/index/vector/index.md)：`IVF_RQ` 的列布局、位数范围和元数据字段。
- [`rust/lance-index/src/vector/bq/builder.rs`](https://github.com/lance-format/lance/blob/v10.0.0/rust/lance-index/src/vector/bq/builder.rs)：`RabitQuantizer`、`quantize_ex_code`、`best_ex_rescale_factor`，对应符号编码和多比特缩放。
- [`rust/lance-index/src/vector/bq/rotation.rs`](https://github.com/lance-format/lance/blob/v10.0.0/rust/lance-index/src/vector/bq/rotation.rs)：`FAST_ROTATION_ROUNDS`、`apply_fast_rotation`，对应四轮快速随机旋转。
- [`rust/lance-index/src/vector/bq/transform.rs`](https://github.com/lance-format/lance/blob/v10.0.0/rust/lance-index/src/vector/bq/transform.rs)：`error_factor_value`、`compute_raw_query_factors`，对应误差因子和距离校正因子。
- [`rust/lance-index/src/vector/bq/storage.rs`](https://github.com/lance-format/lance/blob/v10.0.0/rust/lance-index/src/vector/bq/storage.rs)：`raw_query_lower_bound`、`raw_query_lower_bound_gating_disabled_reason`，对应查询下界及其启用条件。
- [`rust/lance-index/src/vector/bq.rs`](https://github.com/lance-format/lance/blob/v10.0.0/rust/lance-index/src/vector/bq.rs)：`validate_rq_num_bits`、`RABIT_MIN_NUM_BITS`、`RABIT_MAX_NUM_BITS`，对应位数校验。
- [`python/python/lance/indices/builder.py`](https://github.com/lance-format/lance/blob/v10.0.0/python/python/lance/indices/builder.py)：`IndicesBuilder.train_ivf`，对应 IVF 的 k-means 训练。
- [RaBitQ 论文](https://doi.org/10.1145/3654970)
