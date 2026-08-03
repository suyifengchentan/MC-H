"""Precompute CLIP-B/16 augmented teacher features for training sets.

The generated cache follows ``TEACHER_FEATURE_CANDIDATES_KEY``'s contract:
each row has shape ``[num_views, feature_dim]`` and is indexed by the original
training-set index.  Training subsequently samples two distinct views per
image in ``sample_two_teacher_feature_views``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import amp
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypersphere_dpp.config import SUPPORTED_CACHED_TEACHER_DATASETS, load_experiment_config
from hypersphere_dpp.data import _build_base_datasets, safe_default_collate
from hypersphere_dpp.model import FrozenCLIPBackbone
from hypersphere_dpp.transforms import build_gpu_train_transform, build_train_decode_transform
from hypersphere_dpp.utils import resolve_device, set_seed
from train import resolve_amp_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute augmented CLIP-B/16 teacher feature candidates for any "
            "non-ImageNet-1K training dataset."
        )
    )
    parser.add_argument("--config", required=True, help="Training YAML for one supported dataset.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Cache directory. Defaults below dataset_dir and matches cached_teacher_views_dir.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Image decode and inference batch size.")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader image decode workers.")
    parser.add_argument("--device", default=None, help="Device override, for example cuda:0.")
    parser.add_argument(
        "--num-views",
        type=int,
        default=None,
        help="Number of cached augmentations per image; defaults to cached_teacher_num_views (8).",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default=None,
        help="Cache element dtype; defaults to cached_teacher_dtype.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Discard and recreate an existing cache.")
    return parser.parse_args()


def storage_dtype(dtype_name: str) -> tuple[torch.dtype, np.dtype]:
    if dtype_name == "float16":
        return torch.float16, np.dtype(np.float16)
    if dtype_name == "float32":
        return torch.float32, np.dtype(np.float32)
    raise ValueError(f"Unsupported cache dtype: {dtype_name!r}.")


def cache_root_from_config(config, num_views: int, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser()
    if config.cached_teacher_views_dir:
        return Path(config.cached_teacher_views_dir)
    backbone_tag = config.backbone_name.lower().replace("-", "").replace("/", "")
    return Path(config.dataset_dir) / f"teacher_cache_{backbone_tag}_views{num_views}_memmap"


def build_train_dataset(config):
    decode_transform = build_train_decode_transform(config.image_size, config.transform_profile)
    train_dataset, _ = _build_base_datasets(config)
    if not hasattr(train_dataset, "transform"):
        raise TypeError(
            f"Training dataset for {config.dataset!r} does not expose a transform attribute."
        )
    train_dataset.transform = decode_transform
    return train_dataset


def progress_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".progress.json")


def expected_nbytes(num_samples: int, num_views: int, feature_dim: int, dtype: np.dtype) -> int:
    return math.prod((num_samples, num_views, feature_dim)) * dtype.itemsize


def read_progress(
    cache_path: Path,
    *,
    num_samples: int,
    num_views: int,
    feature_dim: int,
    dtype_name: str,
) -> int:
    path = progress_path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"Existing cache is missing its progress file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        progress = json.load(handle)
    expected_shape = [num_samples, num_views, feature_dim]
    if progress.get("shape") != expected_shape or progress.get("dtype") != dtype_name:
        raise ValueError(
            f"Existing cache progress does not match requested cache: {path} "
            f"(expected shape={expected_shape}, dtype={dtype_name})."
        )
    rows_written = int(progress.get("rows_written", -1))
    if not 0 <= rows_written <= num_samples:
        raise ValueError(f"Invalid rows_written={rows_written} in {path}.")
    return rows_written


def write_progress(
    cache_path: Path,
    *,
    num_samples: int,
    num_views: int,
    feature_dim: int,
    dtype_name: str,
    rows_written: int,
) -> None:
    with progress_path(cache_path).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "shape": [num_samples, num_views, feature_dim],
                "dtype": dtype_name,
                "rows_written": rows_written,
            },
            handle,
            indent=2,
        )


def open_or_create_cache(
    cache_path: Path,
    *,
    num_samples: int,
    num_views: int,
    feature_dim: int,
    dtype_name: str,
    numpy_dtype: np.dtype,
    overwrite: bool,
) -> tuple[np.memmap, int]:
    shape = (num_samples, num_views, feature_dim)
    if cache_path.exists() and not overwrite:
        actual_nbytes = cache_path.stat().st_size
        required_nbytes = expected_nbytes(*shape, numpy_dtype)
        if actual_nbytes != required_nbytes:
            raise ValueError(
                f"Existing cache has {actual_nbytes} bytes, expected {required_nbytes}: {cache_path}"
            )
        return np.memmap(cache_path, dtype=numpy_dtype, mode="r+", shape=shape), read_progress(
            cache_path,
            num_samples=num_samples,
            num_views=num_views,
            feature_dim=feature_dim,
            dtype_name=dtype_name,
        )

    if cache_path.exists():
        cache_path.unlink()
    if progress_path(cache_path).exists():
        progress_path(cache_path).unlink()
    cache = np.memmap(cache_path, dtype=numpy_dtype, mode="w+", shape=shape)
    cache.flush()
    write_progress(
        cache_path,
        num_samples=num_samples,
        num_views=num_views,
        feature_dim=feature_dim,
        dtype_name=dtype_name,
        rows_written=0,
    )
    return cache, 0


def write_metadata(
    cache_root: Path,
    *,
    config,
    num_samples: int,
    num_views: int,
    dtype_name: str,
) -> None:
    payload = {
        "dataset": config.dataset,
        "dataset_dir": config.dataset_dir,
        "split": "train",
        "num_samples": num_samples,
        "num_views": num_views,
        "feature_dim": config.feature_dim,
        "dtype": dtype_name,
        "storage_format": "memmap",
        "backbone_name": config.backbone_name,
        "pretrained": config.pretrained,
        "pretrained_checkpoint": config.pretrained_checkpoint,
        "image_size": config.image_size,
        "transform_profile": config.transform_profile,
        "shards": [{"name": "train.mmap", "start_index": 0, "num_rows": num_samples}],
    }
    with (cache_root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative.")

    config = load_experiment_config(args.config)
    if config.dataset not in SUPPORTED_CACHED_TEACHER_DATASETS:
        raise ValueError(
            "This script supports only "
            f"{sorted(SUPPORTED_CACHED_TEACHER_DATASETS)}, got {config.dataset!r}."
        )
    if config.backbone_name != "ViT-B-16" or config.feature_dim != 512:
        raise ValueError("This cache script requires CLIP ViT-B-16 with feature_dim=512.")

    num_views = args.num_views if args.num_views is not None else config.cached_teacher_num_views
    dtype_name = args.dtype or config.cached_teacher_dtype
    if num_views < 2:
        raise ValueError("--num-views must be at least 2.")
    if num_views != config.cached_teacher_num_views:
        raise ValueError(
            "--num-views must match cached_teacher_num_views in the training config "
            "so the generated cache can be loaded during training."
        )
    if dtype_name != config.cached_teacher_dtype:
        raise ValueError(
            "--dtype must match cached_teacher_dtype in the training config "
            "so the generated cache can be loaded during training."
        )

    device = resolve_device(args.device or config.device)
    save_dtype, numpy_dtype = storage_dtype(dtype_name)
    cache_root = cache_root_from_config(config, num_views, args.output_dir)
    data_root = cache_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    cache_path = data_root / "train.mmap"

    set_seed(config.seed)
    dataset = build_train_dataset(config)
    cache, write_offset = open_or_create_cache(
        cache_path,
        num_samples=len(dataset),
        num_views=num_views,
        feature_dim=config.feature_dim,
        dtype_name=dtype_name,
        numpy_dtype=numpy_dtype,
        overwrite=args.overwrite,
    )
    if write_offset == len(dataset):
        write_metadata(
            cache_root,
            config=config,
            num_samples=len(dataset),
            num_views=num_views,
            dtype_name=dtype_name,
        )
        print(f"cache already complete: {cache_root}", flush=True)
        return 0

    loader = DataLoader(
        torch.utils.data.Subset(dataset, range(write_offset, len(dataset))),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=safe_default_collate,
    )
    gpu_train_transform = build_gpu_train_transform(
        config.image_size,
        config.transform_profile,
    ).to(device)
    backbone = FrozenCLIPBackbone(
        backbone_name=config.backbone_name,
        pretrained=config.pretrained,
        pretrained_checkpoint=config.pretrained_checkpoint,
        device=device,
    ).to(device)
    use_amp, amp_device_type, amp_dtype, _ = resolve_amp_settings(config.precision, device)

    print(
        f"writing {config.dataset} cache: samples={len(dataset)} views={num_views} "
        f"feature_dim={config.feature_dim} dtype={dtype_name} device={device} root={cache_root}",
        flush=True,
    )
    for images, _labels, _indices in loader:
        images = images.to(device, non_blocking=True)
        feature_views: list[torch.Tensor] = []
        autocast_kwargs = {"device_type": amp_device_type, "enabled": use_amp}
        if amp_dtype is not None:
            autocast_kwargs["dtype"] = amp_dtype
        amp_context = amp.autocast(**autocast_kwargs) if use_amp else nullcontext()
        with torch.no_grad(), amp_context:
            while len(feature_views) < num_views:
                first_view, second_view = gpu_train_transform(images)
                feature_views.append(backbone(first_view))
                if len(feature_views) < num_views:
                    feature_views.append(backbone(second_view))

        stacked = torch.stack(feature_views, dim=1).to(save_dtype).cpu().numpy()
        batch_size = stacked.shape[0]
        cache[write_offset : write_offset + batch_size] = stacked
        write_offset += batch_size
        cache.flush()
        write_progress(
            cache_path,
            num_samples=len(dataset),
            num_views=num_views,
            feature_dim=config.feature_dim,
            dtype_name=dtype_name,
            rows_written=write_offset,
        )
        print(f"cached {write_offset}/{len(dataset)}", flush=True)

    if write_offset != len(dataset):
        raise RuntimeError(f"Cache row count mismatch: wrote {write_offset}, expected {len(dataset)}.")
    write_metadata(
        cache_root,
        config=config,
        num_samples=len(dataset),
        num_views=num_views,
        dtype_name=dtype_name,
    )
    print(f"completed cache: {cache_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
