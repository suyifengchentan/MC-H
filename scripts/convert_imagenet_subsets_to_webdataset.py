from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypersphere_dpp.labels import default_label_config_path
from scripts.imagenet_subset_utils import (
    DEFAULT_DATASETS,
    SubsetSpec,
    build_subset_specs,
    discover_source_splits,
    load_imagenet_1k_classes,
    write_label_info,
)
from scripts.imagenet_webdataset_utils import write_webdataset_metadata, write_webdataset_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ImageNet-1k parquet shards directly into WebDataset tar shards. "
            "ImageNet1K keeps the full label space, while Image50/Image100/Image200 "
            "select rows by config/labels and remap labels before writing tar shards."
        )
    )
    parser.add_argument(
        "--imagenet-root",
        type=Path,
        default=Path("/home/geyingjie/imagenet-1k"),
        help="Path to the downloaded ImageNet-1k parquet dataset root.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=["ImageNet1K", "Image50", "Image100", "Image200"],
        help="Datasets to build from the source parquet shards.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional destination root. Only valid when converting a single dataset.",
    )
    parser.add_argument(
        "--max-shards-per-split",
        type=int,
        default=None,
        help="Optional limit for quick smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tar shards, indices, and metadata.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subset_specs = resolve_subset_specs(
        imagenet_root=args.imagenet_root,
        dataset_names=list(args.datasets),
        output_root=args.output_root,
    )

    for spec in subset_specs:
        print(f"[{spec.dataset_name}] writing to {spec.output_root}")
        split_summaries = convert_subset(
            imagenet_root=args.imagenet_root,
            spec=spec,
            max_shards_per_split=args.max_shards_per_split,
            overwrite=args.overwrite,
        )
        write_metadata_files(
            spec=spec,
            imagenet_root=args.imagenet_root,
            split_summaries=split_summaries,
        )

    return 0


def resolve_subset_specs(
    *,
    imagenet_root: Path,
    dataset_names: list[str],
    output_root: Path | None = None,
) -> list[SubsetSpec]:
    imagenet_items = load_imagenet_1k_classes(imagenet_root)
    subset_specs = build_subset_specs(dataset_names, imagenet_items)
    if output_root is None:
        return subset_specs
    if len(subset_specs) != 1:
        raise ValueError("--output-root can only be used when converting a single dataset.")
    return [replace(subset_specs[0], output_root=output_root.resolve())]


def convert_subset(
    *,
    imagenet_root: Path,
    spec: SubsetSpec,
    max_shards_per_split: int | None,
    overwrite: bool,
) -> dict[str, dict[str, int]]:
    source_data_root = imagenet_root / "data"
    output_data_root = spec.output_root / "data"
    output_data_root.mkdir(parents=True, exist_ok=True)

    split_summaries: dict[str, dict[str, int]] = {}
    for split_prefix in discover_source_splits(source_data_root):
        shard_paths = sorted(source_data_root.glob(f"{split_prefix}-*.parquet"))
        if max_shards_per_split is not None:
            shard_paths = shard_paths[:max_shards_per_split]

        written_shards = 0
        written_samples = 0
        next_start_index = 0
        for shard_path in shard_paths:
            tar_path = output_data_root / shard_path.with_suffix(".tar").name
            index_path = output_data_root / shard_path.with_suffix(".index.npz").name

            if tar_path.exists() or index_path.exists():
                if overwrite:
                    pass
                elif tar_path.exists() and index_path.exists():
                    with np.load(index_path, allow_pickle=False) as index_data:
                        num_samples = int(index_data["labels"].shape[0])
                    written_shards += 1
                    written_samples += num_samples
                    next_start_index += num_samples
                    continue
                else:
                    raise FileExistsError(
                        f"Incomplete output shard for split={split_prefix}: {tar_path} or {index_path}. "
                        "Rerun with --overwrite to replace it."
                    )

            num_samples = convert_one_shard(
                shard_path=shard_path,
                tar_path=tar_path,
                index_path=index_path,
                label_map=spec.source_label_to_subset_label,
                split=split_prefix,
                start_index=next_start_index,
            )
            if num_samples == 0:
                if tar_path.exists():
                    tar_path.unlink()
                if index_path.exists():
                    index_path.unlink()
                continue

            written_shards += 1
            written_samples += num_samples
            next_start_index += num_samples
            print(f"[{spec.dataset_name}] {split_prefix} {shard_path.name} -> {tar_path.name} ({num_samples} samples)")

        split_summaries[split_prefix] = {
            "num_shards": written_shards,
            "num_samples": written_samples,
        }

    return split_summaries


def convert_one_shard(
    *,
    shard_path: Path,
    tar_path: Path,
    index_path: Path,
    label_map: dict[int, int],
    split: str,
    start_index: int,
) -> int:
    parquet_file = pq.ParquetFile(shard_path)
    return write_webdataset_shard(
        tar_path=tar_path,
        index_path=index_path,
        split=split,
        start_index=start_index,
        rows=iter_parquet_rows(parquet_file=parquet_file, label_map=label_map),
    )


def iter_parquet_rows(
    *,
    parquet_file: pq.ParquetFile,
    label_map: dict[int, int],
):
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        table = parquet_file.read_row_group(row_group_index, columns=["image", "label"])
        image_rows = table.column("image").to_pylist()
        label_rows = table.column("label").to_pylist()
        for image_row, source_label in zip(image_rows, label_rows):
            subset_label = label_map.get(int(source_label))
            if subset_label is None:
                continue
            if not isinstance(image_row, dict):
                continue

            image_bytes = image_row.get("bytes")
            if image_bytes is None:
                continue

            yield bytes(image_bytes), subset_label, str(image_row.get("path", ""))


def write_metadata_files(
    *,
    spec: SubsetSpec,
    imagenet_root: Path,
    split_summaries: dict[str, dict[str, int]],
) -> None:
    spec.output_root.mkdir(parents=True, exist_ok=True)
    write_label_info(spec.output_root, spec)
    write_webdataset_metadata(
        output_root=spec.output_root,
        dataset_name=spec.dataset_name,
        source_root=imagenet_root,
        source_format={
            "type": "parquet_shards",
            "image_column": "image",
            "label_column": "label",
        },
        split_summaries=split_summaries,
        class_num=len(spec.entries),
        label_config_path=default_label_config_path(spec.dataset_name),
    )


if __name__ == "__main__":
    raise SystemExit(main())
