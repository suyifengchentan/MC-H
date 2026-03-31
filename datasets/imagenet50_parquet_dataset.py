from __future__ import annotations

import bisect
import importlib.util
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _ParquetShard:
    path: Path
    start_index: int
    num_rows: int
    row_group_offsets: tuple[int, ...]


def _load_imagenet2012_classes(root: Path) -> tuple[list[str], dict[str, int]]:
    classes_path = root / "classes.py"
    if not classes_path.is_file():
        raise FileNotFoundError(f"Expected ImageNet classes definition at: {classes_path}")

    spec = importlib.util.spec_from_file_location(
        "imagenet_classes_module",
        classes_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import ImageNet classes from: {classes_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    imagenet_classes = getattr(module, "IMAGENET2012_CLASSES", None)
    if not isinstance(imagenet_classes, dict) or not imagenet_classes:
        raise ValueError(f"IMAGENET2012_CLASSES is missing or invalid in {classes_path}")

    classes = [str(wnid).strip() for wnid in imagenet_classes]
    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
    return classes, class_to_idx


class ImageNetParquetDataset(Dataset):
    """
    通用的 ImageNet parquet shard 数据集包装器。

    目录结构要求:
    - root/data/train-*.parquet
    - root/data/validation-*.parquet 或 root/data/test-*.parquet
    - root/label_info.json 或 root/classes.py

    返回 (image, label, index)，与当前训练框架保持一致。
    """

    _SPLIT_TO_PREFIXES = {
        "train": ("train",),
        "val": ("validation", "val"),
        "test": ("test", "validation", "val"),
    }

    def __init__(self, root: str, split: str, transform=None):
        if split not in self._SPLIT_TO_PREFIXES:
            raise ValueError(
                f"ImageNetParquetDataset only supports split='train', 'val', or 'test', got {split!r}."
            )

        self.root = Path(root)
        self.data_root = self.root / "data"
        self.split = split
        self.transform = transform

        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Expected parquet shard directory at: {self.data_root}")

        self.classes, self.class_to_idx = self._load_classes()
        self.shards = self._build_shards(self._SPLIT_TO_PREFIXES[split])
        self._shard_start_indices = [shard.start_index for shard in self.shards]
        self._size = sum(shard.num_rows for shard in self.shards)

        self._parquet_files: dict[Path, pq.ParquetFile] = {}
        self._cached_row_group_key: tuple[int, int] | None = None
        self._cached_row_group_table = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_parquet_files"] = {}
        state["_cached_row_group_key"] = None
        state["_cached_row_group_table"] = None
        return state

    def _load_classes(self) -> tuple[list[str], dict[str, int]]:
        label_info_path = self.root / "label_info.json"
        if label_info_path.is_file():
            with label_info_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)

            idx_to_wnid = raw.get("idx_to_wnid")
            if not isinstance(idx_to_wnid, dict) or not idx_to_wnid:
                raise ValueError(f"Invalid idx_to_wnid mapping in {label_info_path}")

            sorted_pairs = sorted((int(index), wnid) for index, wnid in idx_to_wnid.items())
            classes = [wnid for _, wnid in sorted_pairs]
            class_to_idx = {wnid: index for index, wnid in sorted_pairs}
            return classes, class_to_idx

        return _load_imagenet2012_classes(self.root)

    def _build_shards(self, split_prefixes: tuple[str, ...]) -> list[_ParquetShard]:
        shard_paths: list[Path] = []
        for split_prefix in split_prefixes:
            shard_paths = sorted(self.data_root.glob(f"{split_prefix}-*.parquet"))
            if shard_paths:
                break
        if not shard_paths:
            expected = ", ".join(f"{prefix}-*.parquet" for prefix in split_prefixes)
            raise FileNotFoundError(
                f"Couldn't find parquet shards matching any of [{expected}] under {self.data_root}"
            )

        shards: list[_ParquetShard] = []
        start_index = 0
        for shard_path in shard_paths:
            parquet_file = pq.ParquetFile(shard_path)
            row_group_offsets: list[int] = []
            row_offset = 0
            for row_group_idx in range(parquet_file.metadata.num_row_groups):
                row_group_offsets.append(row_offset)
                row_offset += parquet_file.metadata.row_group(row_group_idx).num_rows

            shards.append(
                _ParquetShard(
                    path=shard_path,
                    start_index=start_index,
                    num_rows=parquet_file.metadata.num_rows,
                    row_group_offsets=tuple(row_group_offsets),
                )
            )
            start_index += parquet_file.metadata.num_rows

        return shards

    def _get_parquet_file(self, path: Path) -> pq.ParquetFile:
        parquet_file = self._parquet_files.get(path)
        if parquet_file is None:
            parquet_file = pq.ParquetFile(path)
            self._parquet_files[path] = parquet_file
        return parquet_file

    def _read_row(self, shard_index: int, row_group_index: int, row_index_in_group: int) -> tuple[bytes, int]:
        cache_key = (shard_index, row_group_index)
        if self._cached_row_group_key != cache_key:
            shard = self.shards[shard_index]
            parquet_file = self._get_parquet_file(shard.path)
            self._cached_row_group_table = parquet_file.read_row_group(
                row_group_index,
                columns=["image", "label"],
            )
            self._cached_row_group_key = cache_key

        image_info = self._cached_row_group_table.column("image")[row_index_in_group].as_py()
        label = int(self._cached_row_group_table.column("label")[row_index_in_group].as_py())
        image_bytes = image_info.get("bytes")
        if image_bytes is None:
            raise ValueError("Parquet row is missing image bytes.")
        return image_bytes, label

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")

        shard_index = bisect.bisect_right(self._shard_start_indices, index) - 1
        shard = self.shards[shard_index]
        local_index = index - shard.start_index

        row_group_index = bisect.bisect_right(shard.row_group_offsets, local_index) - 1
        row_group_start = shard.row_group_offsets[row_group_index]
        row_index_in_group = local_index - row_group_start

        image_bytes, label = self._read_row(shard_index, row_group_index, row_index_in_group)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label, index


ImageNet50ParquetDataset = ImageNetParquetDataset
