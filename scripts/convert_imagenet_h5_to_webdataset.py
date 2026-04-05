from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hypersphere_dpp.hdf5_utils import open_readonly_h5


SUPPORTED_SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ImageNet-style HDF5 shards into WebDataset tar shards."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Source dataset root containing data/*.h5 and label_info.json.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Destination dataset root for WebDataset tar shards.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        choices=SUPPORTED_SPLITS,
        help="Splits to convert. Defaults to train and validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tar shards, indices, and metadata in the output root.",
    )
    return parser.parse_args()


def _copy_label_info(input_root: Path, output_root: Path) -> None:
    source = input_root / "label_info.json"
    if not source.is_file():
        raise FileNotFoundError(f"Expected label_info.json at {source}")
    destination = output_root / "label_info.json"
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _load_source_metadata(input_root: Path) -> dict:
    metadata_path = input_root / "metadata.json"
    if not metadata_path.is_file():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid metadata payload in {metadata_path}")
    return metadata


def _glob_split_shards(data_root: Path, split: str) -> list[Path]:
    shard_paths = sorted(data_root.glob(f"{split}-*.h5"))
    if not shard_paths:
        raise FileNotFoundError(f"Couldn't find source shards matching {split}-*.h5 under {data_root}")
    return shard_paths


def _normalize_image_extension(sample_path: str) -> str:
    suffix = Path(sample_path).suffix.lower().lstrip(".")
    if suffix in {"jpg", "jpeg", "png", "webp"}:
        return suffix
    return "jpg"


def _tar_add_bytes(handle: tarfile.TarFile, name: str, payload: bytes) -> int:
    offset_data = handle.fileobj.tell() + 512
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    handle.addfile(info, io.BytesIO(payload))
    return offset_data


def convert_split(
    *,
    input_root: Path,
    output_root: Path,
    split: str,
    overwrite: bool,
) -> dict[str, int]:
    data_root = input_root / "data"
    output_data_root = output_root / "data"
    output_data_root.mkdir(parents=True, exist_ok=True)

    shard_paths = _glob_split_shards(data_root, split)
    total_samples = 0
    for shard_path in shard_paths:
        tar_name = shard_path.with_suffix(".tar").name
        tar_path = output_data_root / tar_name
        index_path = output_data_root / shard_path.with_suffix(".index.npz").name

        if not overwrite and (tar_path.exists() or index_path.exists()):
            raise FileExistsError(
                f"Output shard already exists for split={split}: {tar_path} or {index_path}. "
                "Rerun with --overwrite to replace it."
            )

        print(f"[convert] {shard_path.name} -> {tar_name}", flush=True)
        offsets: list[int] = []
        sizes: list[int] = []
        labels: list[int] = []

        with open_readonly_h5(shard_path, swmr=True) as source_handle:
            image_dataset = source_handle["image_bytes"]
            label_dataset = source_handle["label"]
            path_dataset = source_handle["path"]
            num_rows = int(label_dataset.shape[0])

            with tarfile.open(tar_path, mode="w") as tar_handle:
                for row_index in range(num_rows):
                    image_value = image_dataset[row_index]
                    if isinstance(image_value, (bytes, bytearray)):
                        image_bytes = bytes(image_value)
                    else:
                        image_bytes = np.asarray(image_value, dtype=np.uint8).tobytes()

                    label = int(label_dataset[row_index])
                    sample_path = str(path_dataset[row_index])
                    image_ext = _normalize_image_extension(sample_path)
                    key = f"{split}-{total_samples + row_index:08d}"

                    image_member_name = f"{key}.{image_ext}"
                    offset_data = _tar_add_bytes(tar_handle, image_member_name, image_bytes)
                    _tar_add_bytes(tar_handle, f"{key}.cls", f"{label}\n".encode("utf-8"))

                    offsets.append(offset_data)
                    sizes.append(len(image_bytes))
                    labels.append(label)

        np.savez(
            index_path,
            offsets=np.asarray(offsets, dtype=np.int64),
            sizes=np.asarray(sizes, dtype=np.int64),
            labels=np.asarray(labels, dtype=np.int32),
        )
        total_samples += len(labels)

    return {
        "num_shards": len(shard_paths),
        "num_samples": total_samples,
    }


def convert_dataset(
    *,
    input_root: Path,
    output_root: Path,
    splits: list[str],
    overwrite: bool,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    _copy_label_info(input_root, output_root)

    source_metadata = _load_source_metadata(input_root)
    split_summaries = {
        split: convert_split(
            input_root=input_root,
            output_root=output_root,
            split=split,
            overwrite=overwrite,
        )
        for split in splits
    }

    metadata = {
        "dataset": source_metadata.get("dataset"),
        "source_root": str(input_root),
        "source_format": source_metadata.get("format", "hdf5"),
        "format": "webdataset",
        "sample_layout": "<key>.jpg + <key>.cls",
        "index_format": "npz(offsets, sizes, labels)",
        "splits": split_summaries,
    }
    metadata_path = output_root / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"[done] wrote metadata to {metadata_path}", flush=True)


def main() -> int:
    args = parse_args()
    convert_dataset(
        input_root=Path(args.input_root).resolve(),
        output_root=Path(args.output_root).resolve(),
        splits=list(args.splits),
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
