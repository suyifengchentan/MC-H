from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import h5py
import torch
import torch.distributed as dist
from torch import amp
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.imagenet_shard_dataset import ImageNetH5Dataset
from hypersphere_dpp.config import load_experiment_config
from hypersphere_dpp.data import safe_default_collate
from hypersphere_dpp.labels import load_label_wnids
from hypersphere_dpp.model import FrozenCLIPBackbone
from hypersphere_dpp.transforms import build_gpu_train_transform, build_train_decode_transform
from hypersphere_dpp.utils import ensure_dir, resolve_device, set_seed
from train import resolve_amp_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute 6-way ImageNet-1K teacher feature candidates for cached 6-choose-2 training."
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
        help="Optional override for the teacher cache output directory.",
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
        help="Overwrite existing cached feature shards and metadata.",
    )
    return parser.parse_args()


def storage_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16, "float16"
    if dtype_name == "float32":
        return torch.float32, "float32"
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


def open_h5_file_with_retry(
    path: Path,
    mode: str,
    *,
    attempts: int = 20,
    sleep_seconds: float = 0.5,
    locking: bool | None = False,
) -> h5py.File:
    last_error: BlockingIOError | None = None
    for attempt in range(attempts):
        try:
            if locking is None:
                return h5py.File(path, mode)
            return h5py.File(path, mode, locking=locking)
        except TypeError:
            # Older h5py builds may not expose the locking kwarg. Fall back to
            # the default open behavior in that case.
            return h5py.File(path, mode)
        except BlockingIOError as error:
            last_error = error
            if attempt + 1 >= attempts:
                break
            time.sleep(sleep_seconds)

    raise BlockingIOError(
        f"Unable to open HDF5 file after {attempts} attempts due to file locking: {path}"
    ) from last_error


def validate_existing_cache_shard(
    output_path: Path,
    *,
    expected_num_rows: int,
    expected_num_views: int,
    expected_feature_dim: int,
) -> int:
    with open_h5_file_with_retry(output_path, "r") as handle:
        cached_shape = tuple(int(dim) for dim in handle["teacher_features"].shape)
    expected_shape = (
        expected_num_rows,
        expected_num_views,
        expected_feature_dim,
    )
    if cached_shape != expected_shape:
        raise ValueError(
            f"Existing cache shard {output_path} has shape {cached_shape}, expected {expected_shape}. "
            "Remove it or rerun with --overwrite."
        )
    return cached_shape[0]


def resolve_rows_written(
    handle: h5py.File,
    *,
    expected_num_rows: int,
) -> int:
    rows_written_raw = handle.attrs.get("rows_written")
    if rows_written_raw is None:
        # Legacy cache shards created before resume support do not carry
        # incremental progress metadata. Treat them as complete so they can
        # still be reused, but partial legacy shards should be deleted.
        return expected_num_rows

    rows_written = int(rows_written_raw)
    if rows_written < 0 or rows_written > expected_num_rows:
        raise ValueError(
            f"Invalid rows_written={rows_written} for cache shard; expected a value in [0, {expected_num_rows}]."
        )
    return rows_written


def prepare_cache_shard(
    output_path: Path,
    *,
    expected_num_rows: int,
    expected_num_views: int,
    expected_feature_dim: int,
    dtype_name: str,
    batch_size: int,
    overwrite: bool,
) -> tuple[h5py.File, h5py.Dataset, int]:
    if output_path.exists() and not overwrite:
        handle = open_h5_file_with_retry(output_path, "r+")
        if "teacher_features" not in handle:
            handle.close()
            raise KeyError(
                f"Existing cache shard is missing required dataset 'teacher_features': {output_path}"
            )

        teacher_features_ds = handle["teacher_features"]
        cached_shape = tuple(int(dim) for dim in teacher_features_ds.shape)
        expected_shape = (
            expected_num_rows,
            expected_num_views,
            expected_feature_dim,
        )
        if cached_shape != expected_shape:
            handle.close()
            raise ValueError(
                f"Existing cache shard {output_path} has shape {cached_shape}, expected {expected_shape}. "
                "Remove it or rerun with --overwrite."
            )

        rows_written = resolve_rows_written(
            handle,
            expected_num_rows=expected_num_rows,
        )
        return handle, teacher_features_ds, rows_written

    handle = open_h5_file_with_retry(output_path, "w")
    teacher_features_ds = handle.create_dataset(
        "teacher_features",
        shape=(
            expected_num_rows,
            expected_num_views,
            expected_feature_dim,
        ),
        dtype=dtype_name,
        chunks=(
            min(max(1, batch_size), expected_num_rows),
            expected_num_views,
            expected_feature_dim,
        ),
    )
    handle.attrs["rows_written"] = 0
    handle.flush()
    return handle, teacher_features_ds, 0


def build_metadata(
    *,
    dataset: ImageNetH5Dataset,
    cache_root: Path,
    config,
    save_dtype_name: str,
    split: str,
) -> dict[str, object]:
    shard_summaries: list[dict[str, int | str]] = []
    for shard in dataset.shards:
        output_path = cache_root / "data" / shard.path.name
        cached_rows = validate_existing_cache_shard(
            output_path,
            expected_num_rows=shard.num_rows,
            expected_num_views=config.imagenet1k_cached_teacher_num_views,
            expected_feature_dim=config.feature_dim,
        )
        shard_summaries.append(
            {
                "name": output_path.name,
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
        "backbone_name": config.backbone_name,
        "pretrained": config.pretrained,
        "pretrained_checkpoint": config.pretrained_checkpoint,
        "image_size": config.image_size,
        "transform_profile": config.transform_profile,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "shards": shard_summaries,
    }


def log(message: str, *, rank: int, force: bool = False) -> None:
    if force or is_main_process(rank):
        print(f"[rank {rank}] {message}", flush=True)


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

        cache_root = (
            Path(args.output_dir)
            if args.output_dir
            else config.resolved_imagenet1k_cached_teacher_views_dir
        )
        if cache_root is None:
            raise ValueError("Unable to resolve ImageNet1K teacher cache output directory.")
        output_data_root = cache_root / "data"
        ensure_dir(output_data_root)

        save_dtype, save_dtype_name = storage_dtype(config.imagenet1k_cached_teacher_dtype)

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
                f"starting teacher cache precompute world_size={world_size} "
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
                    f"feature_dim={config.feature_dim} dtype={save_dtype_name}"
                ),
                rank=rank,
            )

        for shard_index in assigned_shard_indices(len(dataset.shards), rank, world_size):
            shard = dataset.shards[shard_index]
            output_path = output_data_root / shard.path.name
            handle, teacher_features_ds, write_offset = prepare_cache_shard(
                output_path,
                expected_num_rows=shard.num_rows,
                expected_num_views=config.imagenet1k_cached_teacher_num_views,
                expected_feature_dim=config.feature_dim,
                dtype_name=save_dtype_name,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
            )
            if write_offset >= shard.num_rows:
                handle.close()
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

            with handle:
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
                    amp_context = (
                        amp.autocast(**autocast_kwargs) if use_amp else nullcontext()
                    )
                    with torch.no_grad():
                        with amp_context:
                            for _ in range(view_batches_needed):
                                view_a, view_b = gpu_train_transform(images)
                                feature_views.append(backbone(view_a))
                                if (
                                    len(feature_views)
                                    < config.imagenet1k_cached_teacher_num_views
                                ):
                                    feature_views.append(backbone(view_b))

                    stacked = torch.stack(
                        feature_views[: config.imagenet1k_cached_teacher_num_views],
                        dim=1,
                    ).to(save_dtype)
                    batch_size = stacked.size(0)
                    teacher_features_ds[
                        write_offset : write_offset + batch_size
                    ] = stacked.cpu().numpy()
                    write_offset += batch_size
                    handle.attrs["rows_written"] = write_offset
                    handle.flush()

                if write_offset != shard.num_rows:
                    raise RuntimeError(
                        f"Teacher cache row count mismatch for {output_path}: expected {shard.num_rows}, wrote {write_offset}."
                    )
                handle.attrs["rows_written"] = write_offset
                handle.flush()

            log(f"[done] {output_path.name} ({shard.num_rows} rows)", rank=rank, force=True)

        maybe_barrier(world_size)

        if is_main_process(rank):
            metadata = build_metadata(
                dataset=dataset,
                cache_root=cache_root,
                config=config,
                save_dtype_name=save_dtype_name,
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
