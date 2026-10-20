# Hypersphere-DPP

A self-contained image representation learning and clustering framework built around a **frozen CLIP image encoder, a learnable residual adapter, and a hypersphere DPP regularizer**. The repository keeps only the core code needed for training, evaluation, data loading, and preprocessing.

## Method

- Freeze an `open_clip` image encoder (default `ViT-B-16`; `ViT-L-14` is also supported).
- Train a residual MLP feature adapter and maintain `K = class_num` learnable prototypes.
- The total loss combines:
  - `L_inst`: two-view InfoNCE
  - `L_cluster`: SwAV / Sinkhorn online clustering
  - `L_sub`: subspace regularization based on a PCA anchor subspace of teacher features
  - `L_DPP`: batch DPP diversity loss on the hypersphere
  - `L_semantic` (optional): CLIP text semantic prototype alignment
- The DPP weight follows a density-aware adaptive schedule:
  - `alpha_div(t) = gamma(t) * min(alpha_div_base * rho, 1)`

## Repository layout

```text
.
├── train.py                 # single-process training entry point
├── train_parallel.py        # torchrun multi-GPU data-parallel entry point
├── eval.py                  # checkpoint evaluation entry point
├── src/                     # core implementation
│   ├── config.py            # config parsing and validation
│   ├── data.py              # datasets, dataloaders, train/val split
│   ├── model.py             # CLIP backbone, adapter, prototypes
│   ├── losses.py            # InfoNCE, SwAV, DPP, uniformity
│   ├── subspace.py          # PCA subspace queue and regularization
│   ├── semantic.py          # text semantic prototypes
│   ├── metrics.py           # ACC / NMI / ARI
│   ├── transforms.py        # per-dataset train/eval augmentations
│   ├── teacher_cache.py     # precomputed teacher feature reader
│   ├── checkpoint.py        # checkpoint I/O
│   ├── logging_utils.py     # train.log / metrics.jsonl
│   └── utils.py             # distributed, device, seed, and misc helpers
├── datasets/                # dataset readers
├── config/
│   ├── train/               # training configs
│   ├── test/                # evaluation configs
│   └── labels/              # ImageNet subset class definitions
├── scripts/                 # teacher-cache / HDF5 preprocessing tools
├── results/                 # all training outputs (gitignored)
├── requirements.txt
└── README.md
```

`results/` only keeps directory placeholders in git. Runtime artifacts such as `train.log`, `metrics.jsonl`, and `*.pt` are not committed.

## Branches

| Branch | Description |
| --- | --- |
| `main` | Cleaned, maintained implementation using the `src/` + `datasets/` + `results/` layout. The number of prototypes is fixed to `K = class_num`. |
| `freek` | Experimental free-K variant. Estimates the number of clusters from the feature spectrum instead of fixing `K`, and uses spectral prototypes, variable-K Sinkhorn assignments, and label-blind cardinality estimation. |

The `freek` branch keeps the earlier repository layout (`hypersphere_dpp/`, `save/`) and its own experimental configs, including the CIFAR-100 capacity sweeps under `config/train/rebuttal/`. Its core switch is the config field:

```yaml
clustering_mode: spectral_free_k
```

Related options include `min_spectral_clusters`, `max_spectral_clusters`, `initial_spectral_clusters`, `spectral_assignment_temperature`, `spectral_proto_momentum`, `spectral_cardinality_ema_momentum`, `spectral_eval_refresh_*`, `diversity_weight_mode`, and `subspace_anchor_mode`. The branch also contains experiments for label-blind K estimation, semantic audits, persistent prototype capacity sweeps, and cached distributed free-K training.

To inspect or run the experimental variant:

```bash
git checkout freek
```

Use the instructions in the `README.md` on that branch, since its directory layout and output paths differ from `main`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies are listed in `requirements.txt`: `torch`, `torchvision`, `open_clip_torch`, `scikit-learn`, `scipy`, `pyyaml`, `pyarrow`, `h5py`, `kornia`, and others.

If the machine cannot download CLIP weights, prepare a local checkpoint and set it in the config:

```yaml
pretrained_checkpoint: "/path/to/open_clip_pytorch_model.bin"
```

## Data protocol

All data paths are read from `dataset_dir` in the config file. Supported datasets:

| Dataset | Official evaluation split |
| --- | --- |
| `CUB_200_2011` | `test` |
| `FGVC_Aircraft` | `val` / `test` |
| `Oxford_Flowers` | `valid` |
| `Oxford_Pet` | `test` |
| `Stanford_Dogs` | `test` |
| `CIFAR10` / `CIFAR100` | `test` |
| `STL10` | `test` |
| `Image10` | `test` |
| `ImageNet1K` | `val` |
| `Image50` / `Image100` / `Image200` | `val` |
| `TinyImageNet200` | `val` |

Training uses only the official `train` split, and carves out a validation subset according to `validation_ratio`. The official evaluation split can be overridden with `official_eval_split`.

### ImageNet HDF5 format

`ImageNet1K`, `Image50`, `Image100`, and `Image200` read sharded HDF5 data. `Image10` uses an ImageFolder layout (`root/train/<wnid>/*.JPEG`, `root/test/<wnid>/*.JPEG`).

HDF5 layout:

```text
dataset_dir/
  data/train-*.h5
  data/validation-*.h5
  label_info.json
```

Each shard contains at least:

```text
image_bytes: raw JPEG bytes (variable-length uint8 array)
label:       contiguous label
path:        original image path
wnid:        class wnid
```

For `Image50` / `Image100` / `Image200`, the subset and class order are defined by `config/labels/imagenet_*.rtf`. To export the HDF5 shards from ImageNet-1K parquet data:

```bash
python3 scripts/convert_imagenet_subsets_to_hdf5.py \
  --imagenet-root ~/hypersphere-data/imagenet-1k \
  --datasets ImageNet1K Image50 Image100 Image200
```

## Training

First adjust `dataset_dir`, `model_path`, `batch_size`, `workers`, and related fields in the config.

### Single process / single GPU

```bash
python3 train.py \
  --config config/train/fine_grained/cub_vitb16_hypersphere_dpp_config.yaml
```

Override the device:

```bash
python3 train.py \
  --config config/train/medium_grained/image50_vitb16_hypersphere_dpp_config.yaml \
  --device cuda:1
```

### Multi-GPU data parallel

```bash
torchrun --standalone --nproc_per_node=4 train_parallel.py \
  --config config/train/coarse_grained/imagenet1k_vitb16_hypersphere_dpp_parallel_config.yaml
```

`train_parallel.py` reuses the optimizer, scheduler, evaluation, and checkpoint logic from `train.py`, and distributes the dataloader across ranks.

Outputs are written to `model_path`, for example:

```text
results/hypersphere_dpp/CUB_200_2011_CLIP_ViTB16_HypersphereDPP/
  train.log
  metrics.jsonl
  best.pt
  last.pt
  eval_checkpoints/
```

`best.pt` is selected by `val_acc + val_nmi + val_ari` (or the corresponding selection score). Training stops early after `best_checkpoint_patience` consecutive evaluation rounds without improvement.

## Evaluation

```bash
python3 eval.py \
  --config config/test/fine_grained/cub_vitb16_hypersphere_dpp_config.yaml \
  --checkpoint results/hypersphere_dpp/CUB_200_2011_CLIP_ViTB16_HypersphereDPP/best.pt \
  --split both \
  --device cuda:0
```

`--split` accepts `val`, `test`, or `both`. Evaluation reports ACC, NMI, and ARI.

## Teacher feature cache (optional; enabled by default in some configs)

The training configs for `CUB_200_2011`, `CIFAR10`, `CIFAR100`, `Oxford_Flowers`, `Stanford_Dogs`, `Oxford_Pet`, `Image10`, and `Image50` enable cached teacher views by default. Generate the cache once before training:

```bash
python3 scripts/precompute_teacher_feature_cache.py \
  --config config/train/fine_grained/cub_vitb16_hypersphere_dpp_config.yaml \
  --device cuda:0
```

The cache is written to:

```text
<dataset_dir>/teacher_cache_vitb16_views8_memmap/
  data/train.mmap
  data/train.mmap.progress.json
  metadata.json
```

The script can resume after interruption; pass `--overwrite` to recompute from scratch. `cached_teacher_num_views` and `cached_teacher_dtype` must match the cache.

`ImageNet1K` uses a separate sharded cache format:

```bash
python3 scripts/precompute_imagenet1k_teacher_cache.py \
  --config config/train/coarse_grained/imagenet1k_vitb16_hypersphere_dpp_parallel_config.yaml \
  --device cuda:0
```

During training, two distinct views are sampled without replacement from each sample's cached candidate features.

## Key config fields

See `src/config.py` for the full set of fields and defaults. The most frequently modified ones are:

| Field | Description |
| --- | --- |
| `dataset` / `dataset_dir` | Dataset name and root directory |
| `class_num` | Number of classes; also determines the number of prototypes |
| `backbone_name` / `pretrained` | OpenCLIP model and pretrained weights |
| `pretrained_checkpoint` | Local checkpoint path for offline environments |
| `feature_dim` / `representation_dim` | Feature dimensions |
| `batch_size` / `eval_batch_size` | Train/eval batch size |
| `epochs` / `learning_rate` / `weight_decay` | Optimization settings |
| `precision` | `fp16`, `bf16`, or `fp32` |
| `lambda_instance` / `lambda_cluster` / `lambda_subspace` | Loss weights |
| `alpha_div_base` / `dpp_warmup_epochs` | DPP diversity weight and warmup |
| `validation_ratio` / `validation_seed` | Train/val split |
| `eval_interval` | Run validation every N epochs |
| `model_path` | Output directory, under `results/` by default |
| `workers` / `seed` | DataLoader workers and random seed |

## Notes

- The number of prototypes is fixed to `class_num` by default.
- Some config fields are kept for compatibility with older experiments; the config loader emits a warning and ignores unknown fields.
- Run entry points from the repository root so that `src` and `datasets` are importable.
