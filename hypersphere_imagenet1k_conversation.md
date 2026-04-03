# TAC / Hypersphere ImageNet-1K 对话记录

本文档整理了本轮对话中关于 TAC 语义引导机制、Hypersphere 项目在 ImageNet-1K 上表现不佳的原因分析，以及单卡训练和分布式 `all_gather` 修改建议。

## 1. TAC 里的语义引导是怎么做的

结论：

- TAC 不是直接用真实类别名监督聚类。
- 它先从大规模通用 noun 词库中构建一个外部文本语义空间。
- 然后为每张图像检索一个软文本对应物 `text counterpart`。
- 最后再让图像分支和文本分支在聚类空间里对齐。

核心流程：

1. 用 CLIP 文本编码器对 WordNet nouns 做文本 embedding。
2. 用图像特征做过聚类，得到较细粒度的图像语义中心。
3. 用这些图像中心筛掉无关 noun，只保留和当前数据分布匹配的词。
4. 对每张图像和筛后的 nouns 做相似度，得到一个 soft retrieval 文本特征。
5. 训练 cluster head，让图像输出和文本输出一致，并通过邻域蒸馏对齐局部结构。

关键点：

- TAC 的语义引导是实例级的软文本对应，不是固定类名监督。
- 文本分支主要在训练中提供引导，推理时最终仍然只用图像分支做聚类。

## 2. Hypersphere 在 ImageNet-1K 上表现不佳的主要原因

结合 `hypersphere` 项目代码和 `train.log`，分析结论如下。

### 2.1 你当前实现和 TAC 的关键差异

TAC：

- 从开放 noun 空间中为每个样本检索文本 counterpart。
- 语义引导是软的、实例级的。

Hypersphere 当前实现：

- 直接构建 `class_num` 个类名文本特征。
- 然后让 prototype logits 去对齐 CLIP 图像特征对这些类文本的 soft target。

这意味着：

- 在 `Image10 / Image50 / Image100 / Image200` 这种类别数较少的设置下，这种做法通常有效。
- 在 `ImageNet-1K` 这种大量近邻细粒度类别的设置下，它更像一种 closed-set 的弱监督分类先验，容错性会明显下降。

### 2.2 当前多卡训练的一个硬问题

当前 `train_parallel.py` 中：

- `NT-Xent`
- `SwAV Sinkhorn`
- `semantic alignment`
- `DPP`
- `subspace`

这些损失都是在每个 rank 的本地 batch 上算的，不是在全局 batch 上算的。

对 `ImageNet-1K` 而言，这个问题非常严重：

- 配置里 `world_size=2`
- 全局 batch 为 `1536`
- 每卡 `768`
- 两视图后每卡实际参与聚类损失的样本数是 `1536`

在 `1000` 个 prototype 下：

- 每卡平均每个 prototype 的 mass 约为 `1.536`
- 即使看全局，也只有 `3.072`

这会直接带来两个后果：

1. `subspace_loss` 常常根本激活不了
2. `class-aware DPP` 会在非常噪声的 cluster 统计上施加约束

这与日志一致：

- `subspace_loss` 在日志里一直是 `0.0`
- 说明当前 1K run 里这个正则基本没真正工作起来

### 2.3 DPP / subspace 的阈值对 1K 过紧

当前 run 里你使用了：

- `subspace_min_mass=4.0`
- `class_aware_dpp_min_mass=2.0`

在 `1000` prototype、局部 batch 训练的前提下，这两个阈值偏高，尤其是 `subspace_min_mass=4.0`。

因此当前现象其实更像：

- `subspace` 没工作
- `DPP` 在高噪声 cluster 统计上工作

### 2.4 ImageNet-1K 的文本标签本身更差

你的 `Image50 / 100 / 200` 可以使用手工整理过的 `config/labels/*.rtf` 标签文本。

但 `ImageNet1K` 只能依赖：

- `label_info.json`
- 或 `classes.py` 里的原始类名

ImageNet-1K 原始类名经常包含：

- 多个同义词
- 长短语
- 逗号分隔别名

这种文本比经过人工整理的子集类名更不利于 prompt 编码和类间分离。

### 2.5 日志体现的是“早期最好，后续被拉偏”

从日志看：

- `epoch 0` 的验证结果最好
- `epoch 1` 就开始轻微回落

这说明并不是完全学不动，而更像：

- 模型在训练早期已经达到一个还不错的 basin
- 后续正则或语义项继续施压
- 导致模型逐渐偏离最优点

日志信号包括：

- `epoch 0 val acc=0.4178`
- `epoch 1 val acc=0.4143`
- `NMI` 高，但 `ACC` 明显更低

这通常意味着：

- 模型学到了较强的粗语义结构
- 但没有把大量细粒度近邻类切开

## 3. 单卡训练会不会改善

结论：

- 会有改善的可能，而且很值得做验证实验。
- 但是否明显改善，取决于单卡能承受的 batch size。

判断标准：

- 如果单卡能直接跑你当前的全局 batch `1536`，那么我预计会有较明显改善。
- 如果单卡只能跑到 `768` 左右，那它本质上还是当前某个 rank 的局部问题，改善不会从根上解决。

原因：

- 当前多卡训练的问题核心，是损失项只看到了每卡局部 batch。
- 如果单卡可以真正看到更大的 batch，那么：
  - Sinkhorn 更稳定
  - prototype mass 更合理
  - DPP / subspace 的统计也更有意义

但即使单卡能跑更大 batch，`ImageNet-1K` 仍然有两个结构性问题不会自动消失：

1. 你现在的语义引导仍然是固定 `1000` 类名文本，而不是 TAC 那种实例检索文本 counterpart
2. 当前 `subspace_min_mass` 和 `class_aware_dpp_min_mass` 对 1K 仍然偏紧

## 4. 如果要改 `all_gather`，应该怎么改

### 4.1 总原则

当前多卡版本需要把参与以下损失的张量扩展为全局 batch：

- `student` features
- `teacher` features
- `prototype logits`
- `assignments`

但不能直接把本地拼好的 `teacher_cat / student_cat / logits_cat` 拿去 gather 后再 `chunk(2)`。

原因：

- 本地顺序是 `[view1_local, view2_local]`
- 如果按 rank gather，会变成 `[v1_r0, v2_r0, v1_r1, v2_r1, ...]`
- 这时再 `chunk(2)` 会把视图对应关系打乱

正确做法是：

1. 先在本地拆成 `view1` 和 `view2`
2. 分别对 `view1`、`view2` 做 gather
3. gather 完后再按 `[all_view1, all_view2]` 重新拼起来

### 4.2 建议新增两个 helper

一个无梯度 gather：

```python
@torch.no_grad()
def gather_no_grad(x: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size <= 1:
        return x
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x.detach())
    return torch.cat(gathered, dim=0)
```

一个仅保留本 rank 梯度的 gather：

```python
def gather_with_local_grad(x: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
    if world_size <= 1:
        return x
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x.detach())
    gathered[rank] = x
    return torch.cat(gathered, dim=0)
```

### 4.3 建议的训练内修改方式

当前本地得到：

```python
teacher_cat_local, student_cat_local, logits_cat_local = model(view1, view2)
```

先本地拆分：

```python
teacher1_local, teacher2_local = teacher_cat_local.chunk(2)
student1_local, student2_local = student_cat_local.chunk(2)
logits1_local, logits2_local = logits_cat_local.chunk(2)
```

然后 gather：

```python
teacher1 = gather_no_grad(teacher1_local, world_size)
teacher2 = gather_no_grad(teacher2_local, world_size)

student1 = gather_with_local_grad(student1_local.float(), world_size, rank)
student2 = gather_with_local_grad(student2_local.float(), world_size, rank)

logits1 = gather_with_local_grad(logits1_local.float(), world_size, rank)
logits2 = gather_with_local_grad(logits2_local.float(), world_size, rank)

teacher_cat = torch.cat([teacher1, teacher2], dim=0)
student_cat = torch.cat([student1, student2], dim=0)
logits_cat = torch.cat([logits1, logits2], dim=0)
```

### 4.4 哪些损失应该改成基于 gather 后的全局张量

以下都应该改成基于 gather 后的张量计算：

- `nt_xent_loss(student1, student2, ...)`
- `swav_cluster_loss(logits1, logits2, ...)`
- `semantic_cluster_alignment_loss(logits_cat, teacher_cat, ...)`
- `dpp_diversity_loss(student_cat, assignment_cat, ...)`
- `subspace_regularization_loss(student_cat, teacher_cat, assignment_cat, ...)`

`semantic_text_features` 不需要 gather，因为每个 rank 上都是同一份常量。

### 4.5 反向时的缩放问题

如果你采用这种“构造全局 loss，但只有本 rank chunk 保留梯度”的写法，那么 DDP 最终会在梯度同步时做平均。

这会让梯度尺度比单卡全局 batch 小 `world_size` 倍。

因此建议：

```python
loss = (
    config.lambda_instance * inst_loss
    + config.lambda_cluster * cluster_loss
    + config.lambda_semantic * semantic_loss
    + config.lambda_subspace * subspace_loss
    + alpha_div * dpp_loss
)

backward_loss = loss * world_size if world_size > 1 else loss
scaler.scale(backward_loss).backward()
```

记录日志时仍然记录原始 `loss`，不要记录 `backward_loss`。

### 4.6 queue 也要改

`queue.enqueue(...)` 也需要用 gather 后的全局结果：

```python
queue.enqueue(teacher_cat, assignment_cat)
```

否则 queue 仍然只记住每卡局部统计，subspace 依然会因为统计不完整而很弱。

## 5. 对 ImageNet-1K 的实际建议

建议按这个顺序验证：

1. 先做一个单卡实验，尽量把单卡 batch 提高到显存允许的上限。
2. 如果单卡明显优于当前多卡结果，基本可以确认当前多卡 loss 的作用范围有问题。
3. 然后把 `all_gather` 改上。
4. 改完后再调整 1K 专用配置。

推荐先尝试的 1K 配置方向：

- 放松 `subspace_min_mass`
- 放松 `class_aware_dpp_min_mass`
- 如果仍不稳定，可先暂时关闭 `class_aware_dpp`
- 如果语义项仍然造成后期回落，可先降低 `lambda_semantic`

一个相对保守的起点：

- `subspace_min_mass: 2.0`
- `class_aware_dpp_min_mass: 1.0` 或 `1.5`

## 6. 总结

最重要的判断有三点：

1. 当前 ImageNet-1K 不佳，核心不是单一超参，而是多卡训练下损失只看本地 batch，导致 1K 场景的统计失真。
2. 单卡训练如果能承受足够大的 batch，很可能会改善。
3. 真正稳妥的修复方案，是把多卡训练里的关键损失改成基于全局 batch 的 `all_gather` 版本。

如果后续需要，可以在此基础上继续补充：

- `train_parallel.py` 的具体改动 patch
- ImageNet-1K 专用配置建议
- 对 semantic prototype 进一步改为 TAC 风格实例级文本 counterpart 的设计方案
