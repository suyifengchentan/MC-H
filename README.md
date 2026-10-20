# Hypersphere-DPP: Free-K Branch

This branch contains the experimental **free-K** variant of Hypersphere-DPP. Instead of fixing the number of prototypes to the number of classes, it estimates the cluster cardinality from the feature spectrum and maintains a variable-size prototype bank.

The cleaned fixed-K implementation lives on the `main` branch.

## Method

- Freeze an `open_clip` image encoder (default `ViT-B-16`; `ViT-L-14` is also supported).
- Train a residual MLP feature adapter and a spectral quality head.
- Build a quality-weighted feature Gram matrix and estimate the cluster count from its spectrum.
- Construct spectral prototypes from the leading eigenvectors.
- Stabilize the estimated cardinality with EMA smoothing and hysteresis.
- Stabilize prototype identities across updates with Hungarian matching and momentum.
- Use sharpened assignments or variable-K Sinkhorn assignments for online clustering.
- Keep the InfoNCE, subspace, DPP diversity, and optional semantic alignment losses from Hypersphere-DPP.

The free-K switch is:

```yaml
clustering_mode: spectral_free_k
```

With `clustering_mode: fixed_k`, the branch falls back to the standard fixed-`K` model.

## Repository layout

```text
.
├── train.py                 # training entry point (also supports torchrun)
├── eval.py                  # checkpoint evaluation entry point
├── src/                     # core implementation
│   ├── config.py            # config parsing and validation
│   ├── data.py              # datasets, dataloaders, train/val split
│   ├── model.py             # fixed-K and spectral free-K models
│   ├── losses.py            # InfoNCE, Sinkhorn, spectral subspace, DPP
│   ├── subspace.py          # PCA subspace queue and regularization
│   ├── semantic.py          # text semantic prototypes
│   ├── metrics.py           # ACC / NMI / ARI
│   ├── transforms.py        # per-dataset augmentations
│   ├── teacher_cache.py     # precomputed teacher feature reader
│   ├── checkpoint.py        # checkpoint I/O
│   ├── logging_utils.py     # train.log / metrics.jsonl
│   └── utils.py             # distributed, device, seed, and misc helpers
├── datasets/                # dataset readers
├── config/
│   ├── train/               # training configs
│   │   └── freek/           # free-K capacity sweeps and ablations
│   ├── test/                # evaluation configs
│   └── labels/              # ImageNet subset class definitions
├── scripts/                 # ImageNet HDF5 conversion
├── results/                 # all training outputs (gitignored)
├── requirements.txt
└── README.md
```

`results/` only keeps directory placeholders in git. Logs, metrics, predictions, and checkpoints are not committed.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If CLIP weights cannot be downloaded, set a local checkpoint in the config:

```yaml
pretrained_checkpoint: "/path/to/open_clip_pytorch_model.bin"
```

## Data protocol

All paths are read from `dataset_dir` in the config. Supported datasets:

| Dataset | Official evaluation split |
| --- | --- |
| `CIFAR10` / `CIFAR100` | `test` |
| `CUB_200_2011` | `test` |
| `FGVC_Aircraft` | `val` / `test` |
| `Image10` | `test` |
| `ImageNet1K` | `val` |
| `Image50` / `Image100` / `Image200` | `val` |
| `Oxford_Flowers` | `valid` |
| `Oxford_Pet` | `test` |
| `STL10` | `test` |
| `Stanford_Dogs` | `test` |

Training uses the official `train` split and creates a validation subset according to `validation_ratio`.

- `Image10` uses an ImageFolder-style layout: `root/<class>/*`, `root/train/<class>/*`, or split aliases such as `root/train.X1/*`.
- `ImageNet1K`, `Image50`, `Image100`, and `Image200` use sharded HDF5:

```text
dataset_dir/
  data/train-*.h5
  data/validation-*.h5
  label_info.json
```

Each HDF5 shard contains at least `image_bytes`, `label`, `path`, and `wnid`. The subset definitions for `Image50` / `Image100` / `Image200` are stored in `config/labels/`.

HDF5 conversion:

```bash
python3 scripts/convert_imagenet_subsets_to_hdf5.py \
  --imagenet-root ~/hypersphere-data/imagenet-1k \
  --datasets ImageNet1K Image50 Image100 Image200
```

## Training

Adjust `dataset_dir`, `pretrained_checkpoint`, `model_path`, and the batch sizes before running.

### Single process / single GPU

```bash
python3 train.py \
  --config config/train/freek/cifar100_vitb16_freek_stable_cardinality.yaml
```

Override the device:

```bash
python3 train.py \
  --config config/train/freek/cifar100_vitb16_freek_sinkhorn_k72.yaml \
  --device cuda:1
```

### Multi-GPU

`train.py` handles `torchrun` distributed training directly; this branch has no separate `train_parallel.py`.

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --config config/train/freek/cifar100_vitb16_freek_stable_cardinality.yaml
```

Use `distributed_batch_size_mode: global` to treat `batch_size` as the total batch size, or `per_rank` to use it on every rank.

### Outputs

Outputs are written to `model_path`, for example:

```text
results/freek/cifar100_mixed_freek_stable_cardinality_ddp_cache/seed_42/
  train.log
  metrics.jsonl
  best.pt
  last.pt
```

Checkpoint selection uses the clustering selection score configured by the training loop. Set `best_checkpoint_patience: 0` to disable early stopping.

## Free-K configuration

`class_num` acts as the prototype capacity for the persistent capacity sweeps. The effective free-K range is controlled by:

| Field | Description |
| --- | --- |
| `clustering_mode` | `fixed_k` or `spectral_free_k` |
| `min_spectral_clusters` | Lower bound for the estimated cluster count |
| `max_spectral_clusters` | Upper bound; `0` means no explicit upper bound |
| `initial_spectral_clusters` | Initial prototype capacity; must be at least `max_spectral_clusters` when both are set |
| `spectral_assignment_temperature` | Temperature for spectral assignment logits |
| `spectral_assignment_mode` | `sharpen` or `swav_sinkhorn` |
| `spectral_sharpen_temperature` | Target sharpening temperature for `sharpen` mode |
| `spectral_proto_momentum` | Momentum for Hungarian-matched prototype updates |
| `spectral_cardinality_ema_momentum` | EMA momentum for the estimated cluster count |
| `spectral_cardinality_hysteresis` | Dead band before resizing the prototype bank |
| `spectral_eval_refresh_enabled` | Refresh spectral prototypes before evaluation |
| `spectral_eval_refresh_passes` | Number of refresh passes over the evaluation set |
| `spectral_eval_refresh_temperature` | Assignment temperature during refresh |
| `spectral_eval_refresh_min_mass` | Minimum cluster mass kept during refresh |
| `diversity_weight_mode` | `adaptive` or `constant` DPP weighting |
| `subspace_anchor_mode` | `auto`, `teacher_pca`, or `spectral_prototype` |

Example:

```yaml
clustering_mode: "spectral_free_k"
class_num: 60
min_spectral_clusters: 2
max_spectral_clusters: 60
initial_spectral_clusters: 60
spectral_assignment_mode: "sharpen"
spectral_cardinality_ema_momentum: 0.9
spectral_cardinality_hysteresis: 2
subspace_anchor_mode: "auto"
```

The configs under `config/train/freek/` cover persistent capacity sweeps (`K = 48/60/72`) and free-K variants with sharpened or Sinkhorn assignments.

## Evaluation

```bash
python3 eval.py \
  --config config/test/coarse_grained/cifar100_vitb16_hypersphere_dpp_config.yaml \
  --checkpoint results/freek/cifar100_mixed_freek_stable_cardinality_ddp_cache/seed_42/best.pt \
  --split both \
  --device cuda:0
```

To export test predictions in official dataset order:

```bash
python3 eval.py \
  --config config/test/coarse_grained/cifar100_vitb16_hypersphere_dpp_config.yaml \
  --checkpoint results/freek/cifar100_mixed_freek_stable_cardinality_ddp_cache/seed_42/best.pt \
  --split test \
  --save-test-predictions results/freek/cifar100_test_predictions.npy
```

Evaluation reports ACC, NMI, and ARI, plus the estimated cluster count.

## Cached teacher features

Cached teacher views are supported for `CIFAR10`, `CIFAR100`, `Oxford_Flowers`, and `Image50`. Enable them with:

```yaml
cached_teacher_views_enabled: true
cached_teacher_num_views: 8
cached_teacher_dtype: "float16"
```

The cache directory is `<dataset_dir>/teacher_cache_<backbone>_views<N>_memmap/`, unless `cached_teacher_views_dir` is set explicitly. Generate the cache before training:

```bash
python3 scripts/precompute_teacher_feature_cache.py \
  --config config/train/freek/cifar100_vitb16_freek_stable_cardinality.yaml \
  --device cuda:0
```

The script can resume after interruption; pass `--overwrite` to recompute from scratch. `cached_teacher_num_views` and `cached_teacher_dtype` must match the cache.

## Notes

- Use the `main` branch for the cleaned fixed-`K` implementation.
- This branch focuses on free-K cardinality estimation and its ablations.
- Run entry points from the repository root so that `src` and `datasets` are importable.
