from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import amp
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.imagenet_h5_dataset import ImageNetH5Dataset
from src.config import load_experiment_config
from src.data import safe_default_collate
from src.labels import load_label_wnids
from src.model import FrozenCLIPBackbone
from src.transforms import build_gpu_train_transform, build_train_decode_transform
from src.utils import ensure_dir, resolve_device, set_seed
from train import resolve_amp_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute ImageNet-1K teacher feature candidates from HDF5 shards into "
            "sharded memmap arrays."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an ImageNet1K training config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the memmap teacher cache output directory.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=("train",),
        help="Dataset split to cache. Only train is currently supported.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Per-step batch size for precomputing cached teacher features.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="DataLoader workers used for image decode.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Optional device override for single-process runs, for example cuda:0. "
            "Under torchrun multi-GPU mode, LOCAL_RANK decides the CUDA device."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cached memmap shards and metadata.",
    )
    return parser.parse_args()


def storage_dtype(dtype_name: str) -> tuple[torch.dtype, np.dtype, str]:
    if dtype_name == "float16":
        return torch.float16, np.dtype(np.float16), "float16"
    if dtype_name == "float32":
        return torch.float32, np.dtype(np.float32), "float32"
    raise ValueError(f"Unsupported cached teacher dtype: {dtype_name}")


def distributed_env() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def resolve_parallel_device(
    requested_device: str | None, world_size: int, local_rank: int
) -> torch.device:
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Distributed multi-GPU teacher cache precompute requires CUDA, but no CUDA device is available."
            )
        if requested_device and not requested_device.startswith("cuda"):
            raise ValueError(
                "Distributed multi-GPU teacher cache precompute only supports CUDA devices. "
                "Omit --device or provide a CUDA device spec."
            )
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return resolve_device(requested_device)


def init_process_group_if_needed(device: torch.device, world_size: int) -> None:
    if world_size <= 1 or dist.is_initialized():
        return
    backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(backend=backend)


def maybe_barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def destroy_process_group_if_needed(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def assigned_shard_indices(num_shards: int, rank: int, world_size: int) -> range:
    return range(rank, num_shards, world_size)


def log(message: str, *, rank: int, force: bool = False) -> None:
    if force or is_main_process(rank):
        print(f"[rank {rank}] {message}", flush=True)


def default_cache_root_from_config(config) -> Path:
    if config.imagenet1k_cached_teacher_views_dir:
        return Path(config.imagenet1k_cached_teacher_views_dir)
    backbone_tag = config.backbone_name.lower().replace("-", "").replace("/", "")
    cache_dir_name = (
        f"teacher_cache_{backbone_tag}_{config.transform_profile}_"
        f"views{config.imagenet1k_cached_teacher_num_views}_memmap"
    )
    return Path(config.dataset_dir) / cache_dir_name


def shard_progress_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".progress.json")


def _remove_existing_output(path: Path) -> None:
    if path.exists():
        path.unlink()
    progress_path = shard_progress_path(path)
    if progress_path.exists():
        progress_path.unlink()


def expected_memmap_nbytes(
    *,
    num_rows: int,
    num_views: int,
    feature_dim: int,
    dtype: np.dtype,
) -> int:
    return math.prod((num_rows, num_views, feature_dim)) * dtype.itemsize


def read_progress(
    output_path: Path,
    *,
    expected_num_rows: int,
    expected_num_views: int,
    expected_feature_dim: int,
    dtype_name: str,
) -> int:
    progress_path = shard_progress_path(output_path)
    if not progress_path.is_file():
        return expected_num_rows

    with progress_path.open("r", encoding="utf-8") as handle:
        progress = json.load(handle)

    if not isinstance(progress, dict):
        raise ValueError(f"Invalid progress file format: {progress_path}")

    shape = progress.get("shape")
    expected_shape = [expected_num_rows, expected_num_views, expected_feature_dim]
    if shape != expected_shape:
        raise ValueError(
            f"Progress file {progress_path} reports shape {shape}, expected {expected_shape}."
        )

    recorded_dtype = progress.get("dtype")
    if recorded_dtype != dtype_name:
        raise ValueError(
            f"Progress file {progress_path} reports dtype {recorded_dtype!r}, expected {dtype_name!r}."
        )

    rows_written = int(progress.get("rows_written", expected_num_rows))
    if rows_written < 0 or rows_written > expected_num_rows:
        raise ValueError(
            f"Invalid rows_written={rows_written} for {progress_path}; "
            f"expected a value in [0, {expected_num_rows}]."
        )
    return rows_written


def write_progress(
    output_path: Path,
    *,
    num_rows: int,
    num_views: int,
    feature_dim: int,
    dtype_name: str,
    rows_written: int,
) -> None:
    progress_path = shard_progress_path(output_path)
    payload = {
        "shape": [num_rows, num_views, feature_dim],
        "dtype": dtype_name,
        "rows_written": rows_written,
    }
    with progress_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def validate_existing_cache_shard(
    output_path: Path,
    *,
    expected_num_rows: int,
    expected_num_views: int,
    expected_feature_dim: int,
    dtype_name: str,
    numpy_dtype: np.dtype,
) -> int:
    if not output_path.is_file():
        raise FileNotFoundError(f"Expected cache shard at: {output_path}")

    expected_nbytes = expected_memmap_nbytes(
        num_rows=expected_num_rows,
        num_views=expected_num_views,
        feature_dim=expected_feature_dim,
        dtype=numpy_dtype,
    )
    actual_nbytes = output_path.stat().st_size
    if actual_nbytes != expected_nbytes:
        raise ValueError(
            f"Existing cache shard {output_path} has size {actual_nbytes} bytes, "
            f"expected {expected_nbytes} bytes."
        )

    return read_progress(
        output_path,
        expected_num_rows=expected_num_rows,
        expected_num_views=expected_num_views,
        expected_feature_dim=expected_feature_dim,
        dtype_name=dtype_name,
    )


def open_memmap_shard(
    output_path: Path,
    *,
    expected_num_rows: int,
    expected_num_views: int,
    expected_feature_dim: int,
    numpy_dtype: np.dtype,
    mode: str,
) -> np.memmap:
    return np.memmap(
        output_path,
        dtype=numpy_dtype,
        mode=mode,
        shape=(expected_num_rows, expected_num_views, expected_feature_dim),
    )


def prepare_cache_shard(
    output_path: Path,
    *,
    expected_num_rows: int,
    expected_num_views: int,
    expected_feature_dim: int,
    dtype_name: str,
    numpy_dtype: np.dtype,
    overwrite: bool,
) -> tuple[np.memmap, int]:
    if output_path.exists() and not overwrite:
        rows_written = validate_existing_cache_shard(
            output_path,
            expected_num_rows=expected_num_rows,
            expected_num_views=expected_num_views,
            expected_feature_dim=expected_feature_dim,
            dtype_name=dtype_name,
            numpy_dtype=numpy_dtype,
        )
        return (
            open_memmap_shard(
                output_path,
                expected_num_rows=expected_num_rows,
                expected_num_views=expected_num_views,
                expected_feature_dim=expected_feature_dim,
                numpy_dtype=numpy_dtype,
                mode="r+",
            ),
            rows_written,
        )

    _remove_existing_output(output_path)
    memmap_array = open_memmap_shard(
        output_path,
        expected_num_rows=expected_num_rows,
        expected_num_views=expected_num_views,
        expected_feature_dim=expected_feature_dim,
        numpy_dtype=numpy_dtype,
        mode="w+",
    )
    memmap_array.flush()
    write_progress(
        output_path,
        num_rows=expected_num_rows,
        num_views=expected_num_views,
        feature_dim=expected_feature_dim,
        dtype_name=dtype_name,
        rows_written=0,
    )
    return memmap_array, 0


def build_metadata(
    *,
    dataset: ImageNetH5Dataset,
    cache_root: Path,
    config,
    save_dtype_name: str,
    save_numpy_dtype: np.dtype,
    split: str,
) -> dict[str, object]:
    shard_summaries: list[dict[str, int | str]] = []
    for shard in dataset.shards:
        output_name = Path(shard.path.name).with_suffix(".mmap").name
        output_path = cache_root / "data" / output_name
        cached_rows = validate_existing_cache_shard(
            output_path,
            expected_num_rows=shard.num_rows,
            expected_num_views=config.imagenet1k_cached_teacher_num_views,
            expected_feature_dim=config.feature_dim,
            dtype_name=save_dtype_name,
            numpy_dtype=save_numpy_dtype,
        )
        if cached_rows != shard.num_rows:
            raise RuntimeError(
                f"Cache shard {output_path} is incomplete: {cached_rows}/{shard.num_rows} rows written."
            )
        shard_summaries.append(
            {
                "name": output_name,
                "num_rows": cached_rows,
                "start_index": shard.start_index,
            }
        )

    return {
        "dataset": config.dataset,
        "dataset_dir": config.dataset_dir,
        "split": split,
        "num_samples": len(dataset),
        "num_views": config.imagenet1k_cached_teacher_num_views,
        "feature_dim": config.feature_dim,
        "dtype": save_dtype_name,
        "storage_format": "memmap",
        "backbone_name": config.backbone_name,
        "pretrained": config.pretrained,
        "pretrained_checkpoint": config.pretrained_checkpoint,
        "image_size": config.image_size,
        "transform_profile": config.transform_profile,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "shards": shard_summaries,
    }


def main() -> int:
    args = parse_args()
    rank, world_size, local_rank = distributed_env()
    device = resolve_parallel_device(args.device, world_size, local_rank)
    init_process_group_if_needed(device, world_size)

    try:
        config = load_experiment_config(args.config)
        if config.dataset != "ImageNet1K":
            raise ValueError("This precompute script only supports dataset='ImageNet1K'.")

        set_seed(config.seed + rank)

        cache_root = Path(args.output_dir) if args.output_dir else default_cache_root_from_config(config)
        output_data_root = cache_root / "data"
        ensure_dir(output_data_root)

        save_dtype, save_numpy_dtype, save_dtype_name = storage_dtype(
            config.imagenet1k_cached_teacher_dtype
        )

        label_wnids = load_label_wnids(config.dataset)
        dataset = ImageNetH5Dataset(
            root=config.dataset_dir,
            split=args.split,
            transform=build_train_decode_transform(
                config.image_size,
                config.transform_profile,
            ),
            class_names=label_wnids or None,
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
        use_amp, amp_device_type, amp_dtype, _ = resolve_amp_settings(
            config.precision,
            device,
        )

        log(
            (
                f"starting teacher memmap precompute world_size={world_size} "
                f"local_rank={local_rank} device={device}"
            ),
            rank=rank,
        )
        if is_main_process(rank):
            log(f"dataset={config.dataset} split={args.split} size={len(dataset)}", rank=rank)
            log(f"cache_root={cache_root}", rank=rank)
            log(
                (
                    f"teacher_cache: num_views={config.imagenet1k_cached_teacher_num_views} "
                    f"feature_dim={config.feature_dim} dtype={save_dtype_name} storage=memmap"
                ),
                rank=rank,
            )

        for shard_index in assigned_shard_indices(len(dataset.shards), rank, world_size):
            shard = dataset.shards[shard_index]
            output_name = Path(shard.path.name).with_suffix(".mmap").name
            output_path = output_data_root / output_name
            teacher_features_memmap, write_offset = prepare_cache_shard(
                output_path,
                expected_num_rows=shard.num_rows,
                expected_num_views=config.imagenet1k_cached_teacher_num_views,
                expected_feature_dim=config.feature_dim,
                dtype_name=save_dtype_name,
                numpy_dtype=save_numpy_dtype,
                overwrite=args.overwrite,
            )
            if write_offset >= shard.num_rows:
                log(
                    f"[skip] {output_path.name} already complete ({write_offset} rows)",
                    rank=rank,
                    force=True,
                )
                continue

            if write_offset > 0:
                log(
                    f"[resume] {output_path.name} from row {write_offset}/{shard.num_rows}",
                    rank=rank,
                    force=True,
                )
            else:
                log(
                    f"[start] {output_path.name} from row 0/{shard.num_rows}",
                    rank=rank,
                    force=True,
                )

            subset = Subset(
                dataset,
                range(
                    shard.start_index + write_offset,
                    shard.start_index + shard.num_rows,
                ),
            )
            loader = DataLoader(
                subset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.workers > 0,
                collate_fn=safe_default_collate,
            )

            for images, _labels, _indices in loader:
                images = images.to(device, non_blocking=True)
                feature_views: list[torch.Tensor] = []
                view_batches_needed = (
                    config.imagenet1k_cached_teacher_num_views + 1
                ) // 2

                autocast_kwargs = {
                    "device_type": amp_device_type,
                    "enabled": use_amp,
                }
                if amp_dtype is not None:
                    autocast_kwargs["dtype"] = amp_dtype
                amp_context = amp.autocast(**autocast_kwargs) if use_amp else nullcontext()

                with torch.no_grad():
                    with amp_context:
                        for _ in range(view_batches_needed):
                            view_a, view_b = gpu_train_transform(images)
                            feature_views.append(backbone(view_a))
                            if len(feature_views) < config.imagenet1k_cached_teacher_num_views:
                                feature_views.append(backbone(view_b))

                stacked = torch.stack(
                    feature_views[: config.imagenet1k_cached_teacher_num_views],
                    dim=1,
                ).to(save_dtype)
                current_batch_size = stacked.size(0)
                teacher_features_memmap[
                    write_offset : write_offset + current_batch_size
                ] = stacked.cpu().numpy()
                teacher_features_memmap.flush()
                write_offset += current_batch_size
                write_progress(
                    output_path,
                    num_rows=shard.num_rows,
                    num_views=config.imagenet1k_cached_teacher_num_views,
                    feature_dim=config.feature_dim,
                    dtype_name=save_dtype_name,
                    rows_written=write_offset,
                )

            if write_offset != shard.num_rows:
                raise RuntimeError(
                    f"Teacher cache row count mismatch for {output_path}: "
                    f"expected {shard.num_rows}, wrote {write_offset}."
                )

            teacher_features_memmap.flush()
            write_progress(
                output_path,
                num_rows=shard.num_rows,
                num_views=config.imagenet1k_cached_teacher_num_views,
                feature_dim=config.feature_dim,
                dtype_name=save_dtype_name,
                rows_written=write_offset,
            )
            log(f"[done] {output_path.name} ({shard.num_rows} rows)", rank=rank, force=True)

        maybe_barrier(world_size)

        if is_main_process(rank):
            metadata = build_metadata(
                dataset=dataset,
                cache_root=cache_root,
                config=config,
                save_dtype_name=save_dtype_name,
                save_numpy_dtype=save_numpy_dtype,
                split=args.split,
            )
            metadata_path = cache_root / "metadata.json"
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
            log(f"wrote metadata to {metadata_path}", rank=rank)

        maybe_barrier(world_size)
        return 0
    finally:
        destroy_process_group_if_needed(world_size)


if __name__ == "__main__":
    raise SystemExit(main())
