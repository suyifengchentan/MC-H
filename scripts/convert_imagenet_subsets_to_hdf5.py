from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.imagenet_subset_utils import (
    DEFAULT_DATASETS,
    SubsetSpec,
    build_label_info,
    build_subset_specs,
    discover_source_splits,
    load_imagenet_1k_classes,
)
from src.labels import default_label_config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ImageNet-1k parquet shards into HDF5 shards. "
            "ImageNet1K keeps the full label space, while Image50/Image100/Image200 "
            "select rows by config/labels and remap labels before writing H5."
        )
    )
    parser.add_argument(
        "--imagenet-root",
        type=Path,
        default=Path.home() / "hypersphere-data" / "imagenet-1k",
        help="Path to the downloaded ImageNet-1k parquet dataset root.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to build from the source parquet shards.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional destination root. Only valid when converting a single dataset.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="HDF5 chunk size along the sample dimension.",
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
        help="Overwrite existing HDF5 shard files and metadata.",
    )
    return parser.parse_args()


def resolve_subset_specs(
    *,
    imagenet_root: Path,
    dataset_names: list[str],
    output_root: Path | None = None,
) -> list[SubsetSpec]:
    if output_root is not None and len(dataset_names) != 1:
        raise ValueError("--output-root is only supported when converting a single dataset.")

    imagenet_items = load_imagenet_1k_classes(imagenet_root)
    subset_specs = build_subset_specs(dataset_names, imagenet_items)
    if output_root is None:
        return subset_specs

    [spec] = subset_specs
    return [
        SubsetSpec(
            dataset_name=spec.dataset_name,
            output_root=output_root,
            entries=spec.entries,
            source_label_to_subset_label=spec.source_label_to_subset_label,
            subset_label_to_wnid=spec.subset_label_to_wnid,
            subset_label_to_name=spec.subset_label_to_name,
        )
    ]


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
            chunk_size=args.chunk_size,
            max_shards_per_split=args.max_shards_per_split,
            overwrite=args.overwrite,
        )
        write_metadata_files(
            spec=spec,
            imagenet_root=args.imagenet_root,
            split_summaries=split_summaries,
        )

    return 0


def convert_subset(
    imagenet_root: Path,
    spec: SubsetSpec,
    chunk_size: int,
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
        for shard_path in shard_paths:
            output_path = output_data_root / shard_path.name.replace(".parquet", ".h5")
            if output_path.exists() and not overwrite:
                with h5py.File(output_path, "r") as handle:
                    written_samples += int(handle["label"].shape[0])
                written_shards += 1
                continue

            num_samples = convert_one_shard(
                shard_path=shard_path,
                output_path=output_path,
                label_map=spec.source_label_to_subset_label,
                subset_label_to_wnid=spec.subset_label_to_wnid,
                chunk_size=chunk_size,
            )
            if num_samples == 0:
                if output_path.exists():
                    output_path.unlink()
                continue

            written_shards += 1
            written_samples += num_samples
            print(
                f"[{spec.dataset_name}] {split_prefix} {shard_path.name} -> {output_path.name} ({num_samples} samples)"
            )

        split_summaries[split_prefix] = {
            "num_shards": written_shards,
            "num_samples": written_samples,
        }

    return split_summaries


def convert_one_shard(
    shard_path: Path,
    output_path: Path,
    label_map: dict[int, int],
    subset_label_to_wnid: dict[int, str],
    chunk_size: int,
) -> int:
    parquet_file = pq.ParquetFile(shard_path)
    image_arrays: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    wnids: list[str] = []

    for row_group_index in range(parquet_file.metadata.num_row_groups):
        table = parquet_file.read_row_group(row_group_index, columns=["image", "label"])
        image_rows = table.column("image").to_pylist()
        label_rows = table.column("label").to_pylist()
        for image_row, source_label in zip(image_rows, label_rows):
            source_label_int = int(source_label)
            subset_label = label_map.get(source_label_int)
            if subset_label is None:
                continue

            image_bytes = image_row.get("bytes")
            if image_bytes is None:
                continue

            image_arrays.append(np.frombuffer(image_bytes, dtype=np.uint8))
            labels.append(subset_label)
            paths.append(str(image_row.get("path", "")))
            wnids.append(subset_label_to_wnid[subset_label])

    if not labels:
        return 0

    write_hdf5_shard(
        output_path=output_path,
        image_arrays=image_arrays,
        labels=labels,
        paths=paths,
        wnids=wnids,
        chunk_size=chunk_size,
        source_shard_name=shard_path.name,
    )
    return len(labels)


def write_hdf5_shard(
    output_path: Path,
    image_arrays: list[np.ndarray],
    labels: list[int],
    paths: list[str],
    wnids: list[str],
    chunk_size: int,
    source_shard_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_len = max(1, min(chunk_size, len(labels)))
    uint8_vlen_dtype = h5py.vlen_dtype(np.dtype("uint8"))
    utf8_dtype = h5py.string_dtype("utf-8")

    with h5py.File(output_path, "w") as handle:
        image_dataset = handle.create_dataset(
            "image_bytes",
            shape=(len(image_arrays),),
            dtype=uint8_vlen_dtype,
            chunks=(chunk_len,),
        )
        for index, image_array in enumerate(image_arrays):
            image_dataset[index] = image_array

        handle.create_dataset(
            "label",
            data=np.asarray(labels, dtype=np.int32),
            chunks=(chunk_len,),
            compression="gzip",
            shuffle=True,
        )
        handle.create_dataset(
            "path",
            data=np.asarray(paths, dtype=object),
            dtype=utf8_dtype,
            chunks=(chunk_len,),
        )
        handle.create_dataset(
            "wnid",
            data=np.asarray(wnids, dtype=object),
            dtype=utf8_dtype,
            chunks=(chunk_len,),
        )
        handle.attrs["format"] = "imagenet-hdf5-shard-v1"
        handle.attrs["source_shard"] = source_shard_name
        handle.attrs["num_samples"] = len(labels)


def write_metadata_files(
    spec: SubsetSpec,
    imagenet_root: Path,
    split_summaries: dict[str, dict[str, int]],
) -> None:
    spec.output_root.mkdir(parents=True, exist_ok=True)
    label_config_path = default_label_config_path(spec.dataset_name)

    label_info = build_label_info(spec)
    with (spec.output_root / "label_info.json").open("w", encoding="utf-8") as handle:
        json.dump(label_info, handle, ensure_ascii=False, indent=2)

    metadata = {
        "dataset": spec.dataset_name,
        "source_root": str(imagenet_root),
        "label_config_path": str(label_config_path) if label_config_path is not None else None,
        "class_num": len(spec.entries),
        "splits": split_summaries,
        "format": {
            "type": "hdf5_shards",
            "image_dataset": "image_bytes",
            "label_dataset": "label",
            "path_dataset": "path",
            "wnid_dataset": "wnid",
        },
    }
    with (spec.output_root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
