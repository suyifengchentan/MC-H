# Experiments: Validation and Ablation

本目录把 [`methodology.md`](/home/geyingjie/hypersphere/methodology.md) 的核心主张映射为可执行实验。

## 实验映射

1. `Section 3.1 / 3.6`（统一目标在不同粒度下可用）  
   使用 `validation_methodology.yaml` 在 `coarse / medium / fine` 数据集上跑同一训练框架。
2. `Section 3.3 / 3.3.3`（DPP 的密度自适应）  
   对比 `*_full` 与 `*_no_dpp`；同时统计 `mean_train_rho` 与 `mean_train_alpha_div`。
3. `Section 3.4`（自适应权重）  
   在 CUB 上做 `alpha_div_base` 与 `dpp_warmup_epochs` 消融。
4. `Section 3.5 / 3.6`（吸引-排斥平衡与各损失项作用）  
   在 CUB 上做 `no_instance / no_cluster / no_subspace / no_dpp` 消融。
5. `Section 3.7`（训练策略）  
   在 CUB 上做 `dpp_every_n_steps` 频率消融。

## 文件说明

- [`suites/validation_methodology.yaml`](/home/geyingjie/hypersphere/experiments/suites/validation_methodology.yaml): 验证实验矩阵
- [`suites/ablation_methodology.yaml`](/home/geyingjie/hypersphere/experiments/suites/ablation_methodology.yaml): 消融实验矩阵
- [`run_suite.py`](/home/geyingjie/hypersphere/experiments/run_suite.py): 实验运行器（生成配置 + 可选执行 train/eval）
- [`collect_results.py`](/home/geyingjie/hypersphere/experiments/collect_results.py): 聚合 `metrics.jsonl` 与 `eval_*.json` 输出 CSV
- [`run_validation.sh`](/home/geyingjie/hypersphere/experiments/run_validation.sh): 运行验证实验套件
- [`run_ablation.sh`](/home/geyingjie/hypersphere/experiments/run_ablation.sh): 运行消融实验套件
- [`figures/generate_fig3.py`](/home/geyingjie/hypersphere/experiments/figures/generate_fig3.py): 图3（DPP 密度自适应排斥机制）
- [`figures/generate_fig6.py`](/home/geyingjie/hypersphere/experiments/figures/generate_fig6.py): 图6（课程调度与训练动态）
- [`figures/generate_fig7.py`](/home/geyingjie/hypersphere/experiments/figures/generate_fig7.py): 图7（子空间正则化可视化）
- [`figures/ieee_style.py`](/home/geyingjie/hypersphere/experiments/figures/ieee_style.py): 统一 IEEE 风格样式（Computer Modern Roman 字体、线宽、网格、图例）

## 使用方式

1. 仅生成实验计划和配置（默认不执行训练）：

```bash
python experiments/run_suite.py --suite experiments/suites/validation_methodology.yaml
python experiments/run_suite.py --suite experiments/suites/ablation_methodology.yaml
```

2. 执行指定 case（建议先小规模验证）：

```bash
python experiments/run_suite.py \
  --suite experiments/suites/validation_methodology.yaml \
  --case fine_cub_full \
  --seed 42 \
  --execute \
  --device cuda:0
```

3. 全局覆盖超参（所有 case 生效）：

```bash
python experiments/run_suite.py \
  --suite experiments/suites/ablation_methodology.yaml \
  --override epochs=120 \
  --override eval_interval=3 \
  --execute
```

4. 汇总结果：

```bash
python experiments/collect_results.py \
  --suite experiments/suites/validation_methodology.yaml

python experiments/collect_results.py \
  --suite experiments/suites/ablation_methodology.yaml
```

汇总 CSV 默认写到：

- `experiments/generated/methodology_validation/summary.csv`
- `experiments/generated/methodology_ablation/summary.csv`

## 产物路径规范

每个运行实例都落在：

`save/experiments/<suite_name>/<case_id>/seed_<seed>/`

关键文件：

- `train.log`
- `metrics.jsonl`
- `best.pt` / `last.pt`
- `eval_val.json` / `eval_test.json`（执行 eval 时生成）

## 论文图生成

三张图默认使用 IEEE 风格（Computer Modern Roman 字体、紧凑字号、规范线宽和图例样式）。

```bash
python experiments/figures/generate_fig3.py
python experiments/figures/generate_fig6.py
python experiments/figures/generate_fig7.py
```

默认输出目录：

- `experiments/generated/figures/fig3_density_adaptive_repulsion.png`
- `experiments/generated/figures/fig6_curriculum_scheduling.png`
- `experiments/generated/figures/fig7_subspace_regularization.png`
