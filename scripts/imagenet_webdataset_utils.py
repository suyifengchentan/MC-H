from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def copy_label_info(input_root: Path, output_root: Path) -> None:
    source = input_root / "label_info.json"
    if not source.is_file():
        raise FileNotFoundError(f"Expected label_info.json at {source}")
    destination = output_root / "label_info.json"
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def load_source_metadata(input_root: Path) -> dict:
    metadata_path = input_root / "metadata.json"
    if not metadata_path.is_file():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid metadata payload in {metadata_path}")
    return metadata


def normalize_image_extension(sample_path: str) -> str:
    suffix = Path(sample_path).suffix.lower().lstrip(".")
    if suffix in {"jpg", "jpeg", "png", "webp"}:
        return suffix
    return "jpg"


def tar_add_bytes(handle: tarfile.TarFile, name: str, payload: bytes) -> int:
    offset_data = handle.fileobj.tell() + 512
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    handle.addfile(info, io.BytesIO(payload))
    return offset_data


def write_webdataset_shard(
    *,
    tar_path: Path,
    index_path: Path,
    split: str,
    start_index: int,
    rows: Iterable[tuple[bytes, int, str]],
) -> int:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    offsets: list[int] = []
    sizes: list[int] = []
    labels: list[int] = []

    with tarfile.open(tar_path, mode="w") as tar_handle:
        for local_index, (image_bytes, label, sample_path) in enumerate(rows):
            key = f"{split}-{start_index + local_index:08d}"
            image_ext = normalize_image_extension(sample_path)
            image_member_name = f"{key}.{image_ext}"
            offset_data = tar_add_bytes(tar_handle, image_member_name, image_bytes)
            tar_add_bytes(tar_handle, f"{key}.cls", f"{label}\n".encode("utf-8"))

            offsets.append(offset_data)
            sizes.append(len(image_bytes))
            labels.append(int(label))

    np.savez(
        index_path,
        offsets=np.asarray(offsets, dtype=np.int64),
        sizes=np.asarray(sizes, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int32),
    )
    return len(labels)


def write_webdataset_metadata(
    *,
    output_root: Path,
    dataset_name: str | None,
    source_root: Path,
    source_format: object,
    split_summaries: dict[str, dict[str, int]],
    class_num: int | None = None,
    label_config_path: Path | None = None,
) -> None:
    metadata = {
        "dataset": dataset_name,
        "source_root": str(source_root),
        "source_format": source_format,
        "format": "webdataset",
        "sample_layout": "<key>.jpg + <key>.cls",
        "index_format": "npz(offsets, sizes, labels)",
        "splits": split_summaries,
    }
    if class_num is not None:
        metadata["class_num"] = int(class_num)
    if label_config_path is not None:
        metadata["label_config_path"] = str(label_config_path)

    metadata_path = output_root / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
