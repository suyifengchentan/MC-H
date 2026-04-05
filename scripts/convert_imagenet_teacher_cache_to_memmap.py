from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
try:
    import zarr
except ImportError:  # pragma: no cover - optional dependency in some environments
    zarr = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an ImageNet teacher cache from HDF5 shards to a contiguous random-read-friendly format."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Existing teacher cache root containing metadata.json and data/train-*.h5 shards.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output cache root. Defaults to <input-dir>_<format>.",
    )
    parser.add_argument(
        "--format",
        default="zarr",
        choices=("memmap", "zarr"),
        help="Target cache storage format.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output shard files.",
    )
    parser.add_argument(
        "--copy-block-rows",
        type=int,
        default=2048,
        help="Rows copied per HDF5 read when materializing each output shard.",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid metadata payload in {path}")
    return metadata


def ensure_hdf5_source(metadata: dict) -> None:
    storage_format = str(metadata.get("storage_format", "hdf5")).lower()
    if storage_format != "hdf5":
        raise ValueError(
            f"Only HDF5 teacher caches can be converted, got storage_format={storage_format!r}."
        )
    if not metadata.get("shards"):
        raise ValueError("Teacher cache metadata must include shard summaries for conversion.")


def convert_cache(
    *,
    input_root: Path,
    output_root: Path,
    output_format: str,
    overwrite: bool,
    copy_block_rows: int,
) -> None:
    metadata_path = input_root / "metadata.json"
    metadata = load_metadata(metadata_path)
    ensure_hdf5_source(metadata)

    num_views = int(metadata["num_views"])
    feature_dim = int(metadata["feature_dim"])
    dtype = np.dtype(str(metadata["dtype"]))

    input_data_root = input_root / "data"
    output_data_root = output_root / "data"
    output_data_root.mkdir(parents=True, exist_ok=True)

    converted_shards: list[dict[str, int | str]] = []
    for shard in metadata["shards"]:
        shard_name = str(shard["name"])
        shard_num_rows = int(shard["num_rows"])
        shard_start_index = int(shard["start_index"])
        input_path = input_data_root / shard_name
        if not input_path.is_file():
            raise FileNotFoundError(f"Input teacher cache shard not found: {input_path}")

        if output_format == "memmap":
            output_name = Path(shard_name).with_suffix(".mmap").name
        elif output_format == "zarr":
            output_name = Path(shard_name).with_suffix(".zarr").name
        else:
            raise ValueError(f"Unsupported output format: {output_format!r}")
        output_path = output_data_root / output_name
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output shard already exists: {output_path}. Rerun with --overwrite to replace it."
            )

        print(f"converting {input_path.name} -> {output_name} ({shard_num_rows} rows)", flush=True)
        if output_format == "memmap":
            target = np.memmap(
                output_path,
                mode="w+",
                dtype=dtype,
                shape=(shard_num_rows, num_views, feature_dim),
            )
        else:
            if zarr is None:
                raise ImportError("Converting to format='zarr' requires the 'zarr' package to be installed.")
            target = zarr.open(
                output_path,
                mode="w",
                shape=(shard_num_rows, num_views, feature_dim),
                chunks=(min(copy_block_rows, shard_num_rows), num_views, feature_dim),
                dtype=dtype,
            )

        with h5py.File(input_path, "r") as handle:
            dataset = handle["teacher_features"]
            expected_shape = (shard_num_rows, num_views, feature_dim)
            actual_shape = tuple(int(dim) for dim in dataset.shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Shard {input_path} has shape {actual_shape}, expected {expected_shape}."
                )

            for start in range(0, shard_num_rows, copy_block_rows):
                stop = min(start + copy_block_rows, shard_num_rows)
                target[start:stop] = np.asarray(dataset[start:stop], dtype=dtype)

        if output_format == "memmap":
            target.flush()
            del target

        converted_shards.append(
            {
                "name": output_name,
                "num_rows": shard_num_rows,
                "start_index": shard_start_index,
            }
        )

    converted_metadata = dict(metadata)
    converted_metadata["storage_format"] = output_format
    converted_metadata["shards"] = converted_shards
    output_metadata_path = output_root / "metadata.json"
    output_root.mkdir(parents=True, exist_ok=True)
    with output_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(converted_metadata, handle, indent=2)

    print(f"wrote {output_format} metadata to {output_metadata_path}", flush=True)


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_dir).resolve()
    output_root = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_root.with_name(f"{input_root.name}_{args.format}")
    )
    convert_cache(
        input_root=input_root,
        output_root=output_root,
        output_format=args.format,
        overwrite=args.overwrite,
        copy_block_rows=max(1, args.copy_block_rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
