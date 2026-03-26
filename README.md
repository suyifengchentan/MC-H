# Hypersphere-DPP CUB Prototype

这个目录现在包含一版面向 `CUB_200_2011`、`Image10`、`Image50` 和 `Image200` 的自包含原型实现，对应 `ViT-L/14 + Hypersphere DPP` 的训练与评估流程。

## 方法概要

- 冻结 `open_clip` 的 `ViT-L-14` 图像编码器，默认使用 LAION 预训练权重。
- 学生侧训练一个 residual MLP feature adapter，并维护 `K=class_num` 个可学习 prototypes。
- 总损失由四部分组成：
  - `L_inst`: 双视图 InfoNCE
  - `L_cluster`: SwAV/Sinkhorn online clustering
  - `L_sub`: 基于 teacher feature PCA anchor subspace 的子空间正则
  - `L_DPP`: hypersphere 上的 batch DPP diversity loss
- DPP 权重采用 density-aware 自适应调度：
  - `alpha_div(t) = gamma(t) * min(alpha_div_base * rho, 1)`

## 目录结构

- `train.py`: 训练入口
- `eval.py`: 评估入口
- `hypersphere_dpp/`: 配置、数据、模型、损失、子空间队列、评估指标等实现
- `config/train/fine_grained/cub_vitl14_hypersphere_dpp_config.yaml`: 训练配置
- `config/test/fine_grained/cub_vitl14_hypersphere_dpp_config.yaml`: 评估配置
- `tests/`: 基础单元测试
- `save/`: 运行产物目录占位，实际权重和日志默认写到这里

## 数据协议

- 支持 `CUB_200_2011`、`Image10`、`Image50`、`Image200`
- 训练只使用官方 `train` split
- validation 从 `train` 中按 `validation_ratio` 划分
- 官方评估 split 由 `official_eval_split` 控制：
  - `CUB_200_2011`: 默认 `test`
  - `Image10`: 默认 `test`
  - `Image50`: 默认 `val`
  - `Image200`: 默认 `val`

这版实现不复用 `datasets/data_loader_orthogonal.py` 中的 train/test 合并逻辑。

## 依赖

见 `requirements.txt`：

- `torch`
- `torchvision`
- `open_clip_torch`
- `scikit-learn`
- `scipy`
- `pyyaml`

## 运行方式

安装依赖后，训练命令：

```bash
python3 train.py --config config/train/fine_grained/cub_vitl14_hypersphere_dpp_config.yaml
```

训练运行时会在 `model_path` 下生成：

- `train.log`: 可读训练日志
- `metrics.jsonl`: 结构化 epoch 指标
- `best.pt` / `last.pt`: checkpoint

评估命令：

```bash
python3 eval.py \
  --config config/test/fine_grained/cub_vitl14_hypersphere_dpp_config.yaml \
  --checkpoint save/hypersphere_dpp/CUB_200_2011_CLIP_ViTL14_HypersphereDPP/best.pt
```

可选地指定设备：

```bash
python3 train.py --config ... --device cuda:0
python3 eval.py --config ... --checkpoint ... --device cuda:0
```

精度模式由配置项 `precision` 控制，当前支持：

- `fp16`: 仅在 CUDA 上启用 autocast 和 GradScaler
- `bf16`: 在 CUDA/CPU 上启用 autocast，不使用 GradScaler
- `fp32`: 关闭 AMP

训练期间的 `val` 评估频率由配置项 `eval_interval` 控制：

- 当 `eval_interval = 1` 时，每轮训练后都做一次 `val`
- 当 `eval_interval = N` 时，每 `N` 轮训练后做一次 `val`

`best.pt` 的替换只发生在这些 `val` 评估轮次上，并且依据 `val_acc + val_nmi + val_ari` 是否超过当前最优。

## 配置说明

保留并支持的核心字段包括：

- `dataset`
- `dataset_dir`
- `class_num`
- `validation_ratio`
- `validation_seed`
- `official_eval_split`
- `image_size`
- `feature_dim`
- `batch_size`
- `eval_batch_size`
- `epochs`
- `learning_rate`
- `weight_decay`
- `instance_temperature`
- `cluster_temperature`
- `lambda_instance`
- `lambda_cluster`
- `lambda_subspace`
- `alpha_div_base`
- `backbone_name`
- `pretrained`
- `pretrained_checkpoint`
- `adapter_hidden_dim`
- `adapter_dropout`
- `dpp_warmup_epochs`
- `dpp_every_n_steps`
- `dpp_jitter`
- `sinkhorn_iters`
- `sinkhorn_epsilon`
- `subspace_rank`
- `subspace_queue_size`
- `subspace_min_mass`
- `workers`
- `seed`
- `model_path`

对于旧实验配置里未接入首版原型的字段，例如：

- `semantic_prototype_*`
- `distill_confidence_*`
- `gamma_compactness`
- `gamma_space`

配置加载器会给出 warning，并忽略这些字段。

## 当前限制

- prototype 数量固定等于 `class_num`
- 没有接入离线 k-means
- 没有接入 text semantic prototypes
- 若环境离线，`open_clip` 预训练权重需要提前准备本地 checkpoint，并通过 `pretrained_checkpoint` 指定

## ImageNet 系列增强

针对 `Image10`、`Image50`、`Image200`，训练增强不再沿用 CUB 的细粒度设置，而是按自然图像与分辨率特性区分：

- `imagenet10_two_view`: 更强的随机裁剪和颜色扰动，适合粗粒度 ImageNet 子集
- `imagenet50_two_view`: 中等强度裁剪与颜色增强，兼顾语义稳定性与视图多样性
- `tiny_imagenet200_two_view`: 温和裁剪、较小颜色扰动，不使用模糊，避免低分辨率图像被破坏

Image10 和 Image50 默认假设数据目录为标准 `ImageFolder` 结构：

```text
root/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg 或 test/<class_name>/*.jpg
```

Image200 默认按标准 Tiny-ImageNet 目录解析，其中 `val` 通过 `val_annotations.txt` 读取标签。

## Git Ignore 规则

- 默认忽略所有文件，再显式保留源码、配置和文档类文本文件
- 保留目录结构通过 `.gitkeep` 完成
- 训练权重、日志、`jsonl` 指标文件和其他二进制产物不会被纳入版本控制

## 测试

语法检查：

```bash
python3 -m py_compile train.py eval.py hypersphere_dpp/*.py datasets/__init__.py tests/*.py
```

单元测试：

```bash
python3 -m unittest discover -s tests
```

如果测试报 `ModuleNotFoundError`，说明当前 Python 环境尚未安装依赖，而不是实现文件本身存在语法错误。
