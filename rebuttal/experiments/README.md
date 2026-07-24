# Rebuttal Experiment Runners

Every runner is dry-run by default. It generates resolved YAML files and a
`plan.json` under `rebuttal/generated/`; it starts subprocesses only with
`--execute`. The module-level comment in each runner names the rebuttal question
it addresses.

Use the unified entry instead of invoking a runner directly:

```bash
.venv/bin/python rebuttal/experiments/run.py mixed-controlled --devices auto
.venv/bin/python rebuttal/experiments/run.py mixed-controlled --devices auto --execute
```

`--devices auto` selects every visible CUDA device. One selected device runs
`train.py` directly; multiple selected devices run each training workflow with
`torchrun` and DDP. Evaluation and k-means run once on the first selected device.
Use `--devices 0,2` to select a subset. DDP honors `distributed_batch_size_mode`:
`per_rank` keeps `batch_size` per GPU, while `global` divides it across ranks.

- `run_cifar100_mixed_controlled.py`: R1-Q1 and the central R2 concern.
- `run_cifar100_endpoints.py`: auxiliary R1-Q1 endpoint evidence.
- `run_table7_fixed_ssd.py`: R1-Q2 and the Table 7 concern in R2.
- `run_cifar100_mixed_kmeans_baselines.py`: frozen and full-adapted R2-Q2 baselines.
- `run_cifar100_mixed_component_baselines.py`: instance-only, DPCA-only, and SSD controls for R2-Q2.
- `run_cifar100_mixed_batch_size.py`: batch-kernel limitation and reproducibility.
- `audit_free_k_overclustering.py`: R1-Q4 and R2-Q3 analysis for existing free-K predictions.
