# MC-H Rebuttal 实验与回答设计

## 1. 目标与总体策略

本轮 rebuttal 不应继续用“粗粒度数据集 / 混合粒度数据集 / 细粒度数据集”三组平均结果间接支撑核心主张。需要把以下三个命题分开验证：

1. **同一数据集中的混合粒度能力**：一次聚类同时保留一部分粗类，并分开另一部分细类。
2. **H-DPP 的独立贡献**：在固定 DPCA 和 SSD 后，自适应 H-DPP 是否优于固定强度 H-DPP 和全局 uniformity。
3. **未知 K 能力**：动态 K 是独立扩展，不与已知 K 的主聚类结果混报，也不使用真实 K 设置搜索上界。

建议 rebuttal 的优先级如下：

| 优先级 | 内容 | 直接回应 |
|---|---|---|
| P0 | CIFAR-100-Mixed 单数据集受控实验 | R1-Q1，R2 核心质疑，meta-review |
| P0 | 固定 SSD 的 uniform / constant H-DPP / adaptive H-DPP 对照 | R1-Q2，R2 对表 7 的质疑 |
| P0 | 澄清主表使用真实 K，并说明 ACC 匹配协议 | R1-Q3，R2-Q3 |
| P1 | Frozen CLIP、adapted feature、DPCA-only 公平基线 | R2-Q2 |
| P1 | 更新流程和完整损失算法框 | R2-Q1，R3 清晰性问题 |
| P1 | 动态 K 的无泄漏重跑和误差分析 | R1-Q4，R2 动态 K 质疑 |
| P2 | 批次大小、K 误估、CLIP 依赖等限制 | meta-review 可复现性与局限性 |

最重要的回答策略是：**承认原稿证据不足和表 7 存在混淆，然后用最小、严格的实验补上；不要辩称现有跨数据集结果已经证明局部共存。**

---

## 2. 主实验：CIFAR-100-Mixed

### 2.1 为什么不能只跑 K=20 和 K=100

CIFAR-100 同时提供 20 个 superclass 和 100 个 subclass，每个 superclass 含 5 个 subclass。

分别用 `K=20` 和 `K=100` 跑两次，能证明同一方法可工作在两个语义尺度，但仍然是“一次只处理一个粒度”。这只能作为辅助实验，不能完全回答审稿人要求的“同一数据集内部粗细粒度共存”。

核心实验应构造一个**单次、平坦、混合粒度划分**：

- 从 20 个 superclass 中选 10 个作为粗粒度组。这 10 组各自合并为一个目标簇，共 10 个粗簇。
- 剩余 10 个 superclass 作为细粒度组，保留其中 50 个 subclass，共 50 个细簇。
- 因此同一数据集、同一次聚类的目标簇数为 `K_mix = 10 + 10 x 5 = 60`。
- 训练仍使用全部 CIFAR-100 图像；粗/细标签只用于构造评估目标，不能进入训练。

该设置直接测试固定排斥的两难：

- 排斥过强时，粗粒度组容易被其内部 subclass 过度切分。
- 排斥过弱时，细粒度组的相邻 subclass 容易被合并。
- 自适应方法应同时降低 coarse fragmentation 和 fine merging。

### 2.2 防止划分选择偏差

不能只选择一个对 MC-H 有利的 10/10 划分。建议发布 20 个确定性划分：

1. 使用固定随机种子生成 10 个包含 10 个 superclass 的集合。
2. 同时加入每个集合的补集，得到 10 对、共 20 个 coarse/fine 划分。
3. 约束每个 superclass 在 20 个划分中恰好作为 coarse 组出现 10 次。
4. 在补充材料中给出 superclass 名称和生成脚本。

关键的算力优势是：所有划分都有相同的 `K_mix=60`，而训练不使用标签。因此每个方法和训练种子只需训练一次，所得预测可以在全部 20 个混合标签划分上重复评估。

最低可接受版本是 1 个预先声明的主划分加其互补划分；推荐版本是上述 20 个平衡划分。

### 2.3 训练协议

所有变体必须严格共享：

- Frozen CLIP ViT-B/16 权重来源与 checkpoint。
- 相同的 train/test split、图像增强、batch size、epoch 数、优化器和学习率。
- 相同的 `K=60`、prototype 初始化、数据顺序和训练 seed。
- 相同 DPCA、SSD queue、SSD 更新频率和 subspace rank。
- 相同 checkpoint 规则。推荐报告固定 final epoch，避免使用带标签的验证 ACC/NMI/ARI 选取 `best.pt`。
- 训练阶段不得读取 superclass/subclass 标签、类别名称或文本 prompt。

建议种子为 `42, 43, 44`，报告 `mean +/- standard deviation`。若算力允许使用 5 个种子。

如果当前实现启用了类别名称构造的 semantic prototypes、semantic alignment、prototype anchor 或 retrieval loss，本实验必须关闭这些信息：

```yaml
class_num: 60
semantic_prototype_enabled: false
lambda_semantic: 0.0
lambda_semantic_retrieval: 0.0
lambda_prototype_anchor: 0.0
```

SSD 可以保留，但其语义先验只能来自每张图像的 frozen CLIP image feature queue，不能使用类别名称或真实标签。

### 2.4 核心对照矩阵

以下矩阵既回答单数据集混合粒度问题，也修复表 7 的 SSD 混淆。为保证 rebuttal 可快速完成，主运行只执行 B--D；A 已由原稿的组件消融覆盖，不是回答密度自适应质疑所必需的额外训练。

| ID | DPCA | SSD | Repulsion | 目的 |
|---|---:|---:|---|---|
| A | yes | yes | none, `alpha=0` | 无额外排斥基线 |
| B | yes | yes | global uniformity | 固定 SSD 后复现表 7 对照 |
| C | yes | yes | constant H-DPP | 只去掉 density schedule |
| D | yes | yes | adaptive H-DPP | 完整核心方法 |

**最小执行计划。** 使用 seeds `42, 43`，固定 final epoch，并仅运行 B--D，共 `3 x 2 = 6` 次 30-epoch 训练。每个 checkpoint 只在 20 个互补的混合标签划分上离线评估；不再报告与该 `K=60` 目标不一致的 100 类标准评估，因此这些划分不增加训练次数。

其中 C 与 D 必须使用完全相同的 log-determinant kernel、quality weights、warmup 和计算频率；唯一差异是：

```text
C: alpha_t = alpha_const
D: alpha_t = gamma(t) * clip(alpha_base * rho_bar_t, 0, alpha_max)
```

这是比“uniformity vs H-DPP”更干净的核心检验，因为它直接隔离 density adaptivity。

#### 固定强度的选择

不能只选一个较差的固定系数。建议：

- 对 `alpha_const` 做预注册网格，例如自适应模型 warmup 后平均系数的 `{0.25, 0.5, 1, 2}` 倍。
- 在一个独立 calibration run 上确定该尺度，此过程不使用 test labels。
- 主表报告验证集上最优 fixed-alpha，同时在附录报告完整 sweep。
- 额外报告 matched-budget 版本，使 constant H-DPP 的训练期累计系数 `sum_t alpha_t` 与 adaptive H-DPP 相同。

Uniformity 行使用原表 7 的 `lambda={0.05, 0.10, 0.20}`，但每一行都必须包含相同 SSD。正文明确承认：uniformity 在 CIFAR-10 这类粗粒度数据上可能相当或略优；论文主张应限定为混合/细粒度下更好的平衡，而不是“所有数据集和所有指标都更优”。

### 2.5 评估指标

对每个 60 簇预测，使用同一结果计算三组指标。

#### 全局混合划分

- `ACC_mix`：预测的 60 个簇与 60 个混合目标类做 Hungarian matching。
- `NMI_mix`。
- `ARI_mix`。

#### 粗粒度子集

只保留被指定为 coarse 的样本，使用其 10 个 superclass 标签：

- `ACC_coarse`、`NMI_coarse`、`ARI_coarse`。
- `Frag@90_coarse`：对每个真实粗类，覆盖其 90% 样本所需的预测簇数，再取平均；越低越好。
- completeness：检查一个粗类是否被过度切分。

#### 细粒度子集

只保留被指定为 fine 的样本，使用其 50 个 subclass 标签：

- `ACC_fine`、`NMI_fine`、`ARI_fine`。
- homogeneity：检查多个细类是否被错误合并。
- `MergeRate_fine`：落入同一预测簇的异类样本对比例；越低越好。

单独评估子集时必须保留原预测 cluster ID，不能在子集上重新聚类。全局指标用于惩罚 coarse/fine 两部分之间的交叉混合，避免子集指标掩盖跨组错误。

建议主表格式：

| Method | ACC_mix | NMI_mix | ARI_mix | ARI_coarse | ARI_fine | Frag@90 down | MergeRate down |
|---|---:|---:|---:|---:|---:|---:|---:|
| DPCA + SSD | | | | | | | |
| + Uniformity + SSD | | | | | | | |
| + Constant H-DPP + SSD | | | | | | | |
| + Adaptive H-DPP + SSD | | | | | | | |

数值先对 20 个标签划分求均值，再对训练种子汇总。建议同时用 hierarchical bootstrap 给出 95% CI，并对 D vs C 做 paired comparison，因为同一 seed、同一标签划分可天然配对。

### 2.6 最有说服力的图：粗细粒度 Pareto 曲线

对 constant H-DPP 和 uniformity 扫描固定强度，绘制：

- 横轴：`Frag@90_coarse`，越低越好。
- 纵轴：`ARI_fine` 或 `1 - MergeRate_fine`，越高越好。
- 每个固定强度对应一个点，形成“保粗 vs 分细”的 trade-off 曲线。
- adaptive H-DPP 只用一个预设配置，以单独点和置信区间显示。

理想结果不是仅在总 ACC 上提高 0.x，而是 adaptive 点位于固定强度曲线的 Pareto 前沿之外或显著更接近左上角。这一图最直接对应论文的动机。

### 2.7 非循环的机制验证

原文用公式 7 定义 `alpha` 随 `rho` 增长，再展示两者同序，属于定义性结果。应改为测量模型未直接定义的行为结果：

- 用 label-free kNN cosine similarity 定义每个样本的局部拥挤度。
- 记录 H-DPP 对每个样本的梯度范数 `||d(alpha L_dpp)/dz_i||`。
- 按局部拥挤度分成五个分位数，报告每组的平均梯度和最终 margin 改变量。
- 报告局部密度与 repulsive gradient 的 Spearman correlation。
- 同时报告 coarse fragmentation 与 fine merging 的变化，形成机制到结果的闭环。

注意：公式 7 本身是 batch-level scalar，而不是 per-sample coefficient。论文应准确表述为“batch-calibrated H-DPP whose kernel gradient is geometry-dependent”，除非实现确实计算了 cluster-wise 或 sample-wise `alpha_k`。不能仅凭标量调度声称每个 batch 内部存在显式的局部系数。

### 2.8 辅助实验：同一数据的两个端点

为便于解释，再在完全相同的 CIFAR-100 图像上分别跑：

- `K=20`，按 superclass 评估。
- `K=100`，按 subclass 评估。

每个 K 比较 constant H-DPP 与 adaptive H-DPP，保持其他设置一致。该结果证明单一超参数配置在两个端点都稳定，但应明确称为 auxiliary cross-resolution experiment，不把它当作局部共存的唯一证据。

---

## 3. 修订表 7：固定 SSD 的公平比较

在 CIFAR-10、ImageNet-50、Oxford Flowers 上重跑或补齐如下行：

| Configuration | CIFAR-10 | ImageNet-50 | Oxford Flowers |
|---|---:|---:|---:|
| DPCA + SSD | | | |
| DPCA + Uniformity(0.05) + SSD | | | |
| DPCA + Uniformity(0.10) + SSD | | | |
| DPCA + Uniformity(0.20) + SSD | | | |
| DPCA + Constant H-DPP + SSD | | | |
| DPCA + Adaptive H-DPP + SSD | | | |

执行要求：

- 至少 3 seeds，报告 ACC/NMI/ARI 的均值和标准差。
- 所有行使用真实 K，以隔离几何损失，不混入 K 估计误差。
- 所有行保持相同 SSD；另外可在附录给出 without-SSD 矩阵，但不能跨矩阵比较。
- constant H-DPP 必须出现，否则无法判断收益来自 log-det 本身还是 density schedule。
- 对 CIFAR-10 如 uniformity 优于 H-DPP，应如实报告，并将结论改为 H-DPP 在混合/细粒度上的优势及跨粒度稳定性。

---

## 4. CLIP 与 DPCA 公平基线

审稿人 2 要求的基线应在相同 backbone、数据 split 和 K 下统一重跑。建议选 CIFAR-10、ImageNet-50、Oxford Flowers，另将 CIFAR-100-Mixed 作为核心列。

| Baseline | Feature | Clustering head |
|---|---|---|
| Frozen CLIP + k-means | frozen CLIP image feature | Euclidean k-means |
| Frozen CLIP + spherical k-means | L2-normalized frozen feature | cosine / spherical k-means |
| Adapted `L_inv` + spherical k-means | 只用 invariance 训练的 adapter | post-hoc spherical k-means |
| Full adapted feature + spherical k-means | 完整模型的 student feature | post-hoc spherical k-means |
| DPCA only | adapted feature | prototype assignment |
| DPCA + SSD | adapted feature | prototype assignment |
| DPCA + constant H-DPP + SSD | adapted feature | prototype assignment |
| Full adaptive MC-H | adapted feature | prototype assignment |

统一设置：

- k-means 使用相同 K，`n_init=20`，报告相同 seeds。
- spherical k-means 使用 L2-normalized features 和 cosine assignment。
- 不允许某些行使用类别文本初始化而另一些不用；类别名称若使用，必须成为单独的显式 ablation。
- 主对照最好完全不使用类别名称，从而检验纯视觉无监督聚类。
- 不用 test labels 选 checkpoint 或超参数；若历史主表确实这样做，应在 rebuttal 中透明说明并补报 fixed-final-epoch 结果。

---

## 5. 主表 K 与 ACC 协议

建议明确回答：

> Tables 1--3 use the ground-truth number of clusters, as is standard in the compared known-K protocols. Dynamic-K in Table 4 is a separate extension and was not used to produce Tables 1--3. We will state this explicitly in every table caption and in Sec. 4.1.

如果核对 checkpoint 后发现主表实际使用估计 K，则不能使用上述文字，必须重新标注并补一组 known-K 公平结果。

当 `K_pred != K_gt` 时，至少报告 NMI、ARI 和两种 ACC：

1. `ACC_1to1`：在 `K_pred x K_gt` contingency matrix 上做矩形 Hungarian matching。每个预测簇至多映射一个真实类，未匹配的多余簇计错，因此会惩罚 over-clustering。
2. `ACC_manyto1`：每个预测簇映射到其多数真实类，作为 purity-like 补充指标；该指标不惩罚把一个真实类切成多个纯簇，不能单独使用。

正文主指标使用 `ACC_1to1`，同时给出 NMI/ARI。必须公开精确实现，不用 `max(label)+1` 隐式构造方阵，因为非连续 cluster ID 和 `K_pred != K_gt` 时容易产生含糊结果。

---

## 6. 动态 K：独立、无泄漏的补充实验

### 6.1 先修正协议

`freek` 分支可作为实现参考，但正式结果必须满足：

- `max_spectral_clusters` 不能等于真实 K。
- `initial_spectral_clusters` 不能按真实 K 的倍数设置。
- 搜索上界只能由与标签无关的资源约束决定，例如统一的 `min(batch_size, feature_dim)` 或所有数据集共享的固定上界。
- checkpoint 不能按带标签的 `val_acc + val_nmi + val_ari` 选择；使用 fixed final epoch 或 label-free assignment stability。
- CIFAR-100 必须明确使用 20 个 superclass 还是 100 个 subclass。不能用 100-class 数据训练/评估，却在表中把 20 作为 ground-truth K。

### 6.2 报告指标

对每个数据集报告：

- `K_gt`、每个 seed 的 `K_hat`、`mean +/- std`。
- absolute error `|K_hat-K|`。
- relative error `|K_hat-K|/K`。
- known-K 与 free-K 下的 ACC/NMI/ARI 差值。
- K 随 epoch 的轨迹和最终稳定性。

### 6.3 高估是否为语义子结构

最低成本且可信的验证：

- 对每个真实类计算它被多少预测簇拆分，以及各子簇的样本数和 purity。
- 报告 over-cluster 的 size distribution，排除大量极小簇。
- 对 CIFAR-100 可利用 superclass/subclass 双层标签：若按 20 superclass 评估得到 `K_hat=26`，检查额外簇是否稳定对应其中的 fine subclass。
- 对 Stanford Dogs 可报告 extra cluster 与 breed 内姿态/颜色等属性的一致性；没有正式属性标签时只提供定性样例，不能称其为已确认语义。
- ImageNet-1K 若无法完成系统验证，删除“高估反映内在语义子结构”的肯定句，改为“可能来自子结构，也可能是估计偏差”。

推荐的保守回答：

> We agree that the current evidence does not establish that over-estimated components are semantically meaningful. We will weaken this interpretation, report cardinality error explicitly, and treat semantic substructure as a hypothesis unless supported by the new split/purity analysis.

---

## 7. 更新流程与完整目标函数

论文需要加入算法框，并保证它与实际代码一致。建议每个 iteration 明确写成：

1. 对两种增强视图计算 frozen CLIP teacher features `t^a, t^b`。
2. 通过 adapter 得到归一化 student features `z^a, z^b`。
3. 由当前 prototypes / H-DPP spectrum 得到 logits 和跨视图 soft assignments。
4. 计算 cross-view partition consistency。公式 2 必须作为显式 loss 出现在最终目标中，不能再称为“implicitly used”。
5. 由 detached assignments 和 teacher features构建/读取 SSD queue，计算当前 step 的 SSD loss。
6. 由 student features 和 quality weights 构造 H-DPP kernel，计算 `L_dpp`、`rho` 和 `alpha_t`。
7. 计算完整训练目标并反向传播；必须列出代码中实际启用的 instance、cluster、semantic、retrieval、prototype-anchor、SSD 和 DPP 项。
8. optimizer step 后归一化 prototypes，再用 detached assignment 更新 SSD queue。
9. 若为 free-K，在声明的周期更新 spectral prototypes；说明 eigenvector matching、sign alignment 和 EMA 的顺序。

原稿公式 13 漏掉公式 2，而当前代码配置还可能启用 semantic alignment 和 prototype anchor。提交 rebuttal 前必须建立一张“论文公式 - 配置项 - 代码函数 - 表中 checkpoint”的审计表；任何实际启用但未写入论文的监督或文本语义项都要披露或关闭后重跑。

针对审稿人 3 的符号问题，逐项修订：

- `r_i`：quality head 输出的 scalar log-quality score。
- `mu_k`：在公式 8--9 中统一定义为 H-DPP kernel 的第 k 个 eigenvalue；SSD 中的 cluster mean 改用不同符号，避免重名。
- “frozen semantic prior”：明确为 frozen CLIP image feature；若使用 text feature，必须另行声明类别名称来源。
- SSD 各段是同一 iteration 中按“queue routing -> basis estimation -> residual covariance loss”的顺序执行。
- 对公式 12 的解释改为：`Sigma - P Sigma P` 保留至少一个端点位于正交补空间的 covariance block，因此惩罚非语义方向及其与语义子空间的交叉协方差；不要只用直觉句。

---

## 8. 统计与复现信息

对审稿人 1 的报告细节，统一答复并落实：

- 所有新增结果至少 3 个独立训练 seeds：`42, 43, 44`。
- `+/-` 表示跨独立训练 seeds 的 standard deviation，不是 standard error。
- Tables 1--3 若是单次最佳结果，不能声称它们是均值；应补成同一 seeds 的 mean +/- std，或明确区分主表单次结果与附录多次结果。
- k-means 初始化随机性不能冒充训练 seed；对每个训练 seed 内部使用 `n_init=20`。
- 保存每次运行的完整 YAML、git commit、预测 cluster IDs 和 contingency matrix。
- 同一 seed 的方法变体使用相同数据划分、初始化和 batch order，便于 paired test。

建议补充 batch size 敏感性：在 CIFAR-100-Mixed 上使用 `{128, 256, 512}`，报告性能和 `rho/alpha` 轨迹。由于 H-DPP 与 batch kernel 直接相关，这属于实质性限制，而非普通超参分析。

---

## 9. 面向审稿人的回答模板

### 9.1 核心混合粒度主张

> We agree that grouping separate benchmarks by granularity did not isolate within-dataset adaptation. We therefore construct CIFAR-100-Mixed from the official 20-superclass/100-subclass hierarchy. In each evaluation partition, 10 superclasses are retained as 10 coarse clusters while the remaining 10 are resolved into 50 fine clusters, yielding a single 60-cluster target over the same images. The model is trained once without labels or class names, and we evaluate global mixed-partition quality together with coarse-side fragmentation and fine-side merging over balanced superclass partitions and three seeds. We compare adaptive H-DPP against constant H-DPP and uniform repulsion with DPCA and SSD fixed.

结果出来后追加一句结论，但只描述实际观察。例如只有在 `ARI_fine` 提高而 `Frag@90_coarse` 不恶化时，才能称其支持 simultaneous balance。

### 9.2 H-DPP 与 SSD 的混淆

> We thank the reviewer for identifying the ambiguity in Table 7. The original comparison did not make the SSD control sufficiently explicit. We replace it with a factorial comparison in which DPCA and SSD are identical for all rows, and compare no repulsion, uniformity, constant-coefficient H-DPP, and density-adaptive H-DPP. The constant-H-DPP comparison isolates the density schedule while preserving the same kernel and log-determinant objective.

### 9.3 DPCA 增益最大

> We agree that DPCA provides the largest transition from the weak baseline because it is the component that first creates a usable partition. We will revise the framing: DPCA is the structural clustering interface, SSD is the stronger standalone geometric component in Table 5, and the specific role of H-DPP is to improve the coarse/fine separation-fragmentation trade-off after assignment and denoising are fixed. We no longer imply that H-DPP alone accounts for the majority of the total gain.

### 9.4 动态 K

> Tables 1--3 and the free-K study are reported as separate protocols. We explicitly state whether ground-truth K is used in each caption, remove ground-truth-dependent spectral caps, define rectangular Hungarian ACC for unequal cluster counts, and report cardinality error together with clustering quality. We also weaken the semantic-substructure interpretation where no attribute-level evidence is available.

### 9.5 冻结 CLIP 基线

> We add frozen CLIP k-means/spherical k-means, adapted-feature spherical k-means, and DPCA-only baselines under the same ViT-B/16 checkpoint, data split, K, and evaluation code. This separates the contribution of the pretrained representation, feature adaptation, prototype assignment, SSD, and H-DPP.

---

## 10. 结果判定与论文措辞

预先定义结论边界，避免结果出来后过度解释：

1. **强支持**：adaptive H-DPP 在 `ARI_mix`、`ARI_coarse`、`ARI_fine` 上整体优于最佳 constant H-DPP，且 Pareto 图同时降低 fragmentation 和 merging。
2. **部分支持**：总指标提高，但 coarse 或 fine 一侧没有优势。论文只能声称 improved average mixed-granularity performance，不能声称 simultaneous local adaptation。
3. **不支持 density schedule**：constant H-DPP 与 adaptive 无显著差异。保留 H-DPP log-det 贡献，但弱化 density-adaptive 为核心机制的表述。
4. **uniformity 在粗粒度更好**：如实承认，并强调方法目标是跨异质粒度的稳健平衡，而非每个单一粗粒度基准都最优。
5. **动态 K 高估无语义证据**：称为 estimation error，不再称为 discovery。

---

## 11. 最小执行清单

### Rebuttal 最小集

- [ ] 实现 CIFAR-100 superclass/subclass 双标签和 20 个平衡 mixed evaluation mappings。
- [ ] 在 `K=60`、3 seeds 下训练 A--D 四个核心变体。
- [ ] 输出全局、coarse-side、fine-side 指标和 Pareto 图。
- [ ] 在现有三个表 7 数据集上至少补 constant H-DPP + SSD 和 uniformity + SSD。
- [ ] 核对 Tables 1--3 的 K、checkpoint、完整 loss 和类别文本使用情况。
- [ ] 明确 `+/-` 的 seed 数和统计量。
- [ ] 在 rebuttal 中弱化未经验证的 K 子结构解释。

### 完整修订集

- [ ] 补齐 frozen/adapted CLIP 与 DPCA 基线。
- [ ] 无真实 K cap 重跑 free-K。
- [ ] 完成 K 高估的 split/purity 分析。
- [ ] 加入 batch-size sensitivity。
- [ ] 加入完整算法框、符号表和 limitation 段落。
- [ ] 发布逐次运行配置、预测和评估脚本。

---

## 12. 最终表格与图的推荐布局

Rebuttal 正文优先放两项：

1. **Table R1: CIFAR-100-Mixed controlled experiment**，包含 adaptive 与最佳 fixed-strength 对照，以及 coarse/fine 分组指标。
2. **Figure R1: coarse fragmentation vs fine separation Pareto plot**。

若还有空间，再放：

3. **Table R2: fixed-SSD uniformity / constant H-DPP / adaptive H-DPP**。
4. 一段文字明确 known-K 主表、free-K 表和 unequal-K ACC 协议。

这种组织优先回答可能改变评分的两个问题，而不是在 rebuttal 中重复现有的大规模 benchmark 优势。

---

## 13. 当前脚本实现

以下文件已经实现了本设计中无需改变训练目标的部分：

| 文件 | 作用 |
|---|---|
| `config/train/rebuttal/cifar100_mixed_vitb16_hypersphere_dpp_config.yaml` | `K=60` 视觉无标签主配置，关闭 class-name/text semantic 信号。 |
| `experiments/run_kmeans_baselines.py` | 在训练 protocol split 拟合 frozen/adapted Euclidean 与 spherical k-means，并在 official test split 预测。 |
| `experiments/evaluate_cifar100_mixed.py` | 从 checkpoint 或保存的预测读取官方 CIFAR-100 test 顺序，在 20 个平衡 coarse/fine 分区上汇总指标。 |
| `experiments/clustering_eval_utils.py` | 矩形 Hungarian ACC 与 spherical k-means 的共享实现。 |

执行命令、预测文件和 JSON 产物的路径见 `experiments/README.md` 中的 “Rebuttal: k-means 和 CIFAR-100-Mixed” 一节。训练 constant-H-DPP 与 uniformity 变体仍需在训练目标中显式实现对应开关；在此之前，新增脚本可先用于 frozen/adapted k-means 与现有模型 checkpoint 的统一评估。
