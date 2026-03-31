from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypersphere_dpp.labels import LabelEntry, default_label_config_path, load_label_entries


DEFAULT_OUTPUT_ROOTS = {
    "ImageNet1K": Path("/home/geyingjie/hypersphere-data/ImageNet1K"),
    "Image50": Path("/home/geyingjie/hypersphere-data/Image50"),
    "Image100": Path("/home/geyingjie/hypersphere-data/Image100"),
    "Image200": Path("/home/geyingjie/hypersphere-data/Image200"),
}
DEFAULT_DATASETS = ["ImageNet1K", "Image50", "Image100", "Image200"]


@dataclass(frozen=True)
class SubsetSpec:
    dataset_name: str
    output_root: Path
    entries: list[LabelEntry]
    source_label_to_subset_label: dict[int, int]
    subset_label_to_wnid: dict[int, str]
    subset_label_to_name: dict[int, str]


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
        default=Path("/home/geyingjie/imagenet-1k"),
        help="Path to the downloaded ImageNet-1k parquet dataset root.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=sorted(DEFAULT_OUTPUT_ROOTS),
        help="Datasets to build from the source parquet shards.",
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


def main() -> int:
    args = parse_args()
    imagenet_items = load_imagenet_1k_classes(args.imagenet_root)
    subset_specs = build_subset_specs(args.datasets, imagenet_items)

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


def load_imagenet_1k_classes(imagenet_root: Path) -> list[tuple[str, str]]:
    classes_path = imagenet_root / "classes.py"
    if not classes_path.is_file():
        raise FileNotFoundError(f"Expected classes.py at: {classes_path}")

    spec = importlib.util.spec_from_file_location("imagenet_classes_module", classes_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import ImageNet classes from: {classes_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    imagenet_classes = getattr(module, "IMAGENET2012_CLASSES", None)
    if imagenet_classes is None:
        raise AttributeError(f"IMAGENET2012_CLASSES not found in {classes_path}")

    return list(imagenet_classes.items())


def build_subset_specs(dataset_names: list[str], imagenet_items: list[tuple[str, str]]) -> list[SubsetSpec]:
    wnid_to_source_label = {wnid: index for index, (wnid, _) in enumerate(imagenet_items)}
    subset_specs: list[SubsetSpec] = []

    for dataset_name in dataset_names:
        if dataset_name == "ImageNet1K":
            label_entries = [LabelEntry(wnid=wnid, name=name) for wnid, name in imagenet_items]
        else:
            label_entries = load_label_entries(dataset_name)
            if not label_entries:
                raise FileNotFoundError(
                    f"No label entries found for {dataset_name} at {default_label_config_path(dataset_name)}"
                )

        source_label_to_subset_label: dict[int, int] = {}
        subset_label_to_wnid: dict[int, str] = {}
        subset_label_to_name: dict[int, str] = {}

        for subset_label, entry in enumerate(label_entries):
            if entry.wnid not in wnid_to_source_label:
                raise KeyError(f"{dataset_name} label {entry.wnid!r} does not exist in ImageNet-1k classes.")
            source_label = wnid_to_source_label[entry.wnid]
            source_label_to_subset_label[source_label] = subset_label
            subset_label_to_wnid[subset_label] = entry.wnid
            subset_label_to_name[subset_label] = entry.name

        subset_specs.append(
            SubsetSpec(
                dataset_name=dataset_name,
                output_root=DEFAULT_OUTPUT_ROOTS[dataset_name],
                entries=label_entries,
                source_label_to_subset_label=source_label_to_subset_label,
                subset_label_to_wnid=subset_label_to_wnid,
                subset_label_to_name=subset_label_to_name,
            )
        )

    return subset_specs


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
            print(f"[{spec.dataset_name}] {split_prefix} {shard_path.name} -> {output_path.name} ({num_samples} samples)")

        split_summaries[split_prefix] = {
            "num_shards": written_shards,
            "num_samples": written_samples,
        }

    return split_summaries


def discover_source_splits(source_data_root: Path) -> list[str]:
    split_prefixes = [
        split_prefix
        for split_prefix in ("train", "validation", "test")
        if any(source_data_root.glob(f"{split_prefix}-*.parquet"))
    ]
    if not split_prefixes:
        raise FileNotFoundError(f"No parquet shards found under {source_data_root}")
    return split_prefixes


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

    label_info = {
        "idx_to_wnid": {str(index): spec.subset_label_to_wnid[index] for index in range(len(spec.entries))},
        "idx_to_class_name": {str(index): spec.subset_label_to_name[index] for index in range(len(spec.entries))},
        "class_to_idx": {spec.subset_label_to_wnid[index]: index for index in range(len(spec.entries))},
    }
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
