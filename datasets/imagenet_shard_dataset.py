from __future__ import annotations

import bisect
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _ShardIndex:
    offsets: np.ndarray
    sizes: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class _TarShard:
    path: Path
    index_path: Path
    start_index: int
    num_rows: int
    index: _ShardIndex


class ImageNetWebDataset(Dataset):
    """
    Map-style ImageNet tar-shard dataset with WebDataset-compatible samples.

    Expected layout:
    - root/data/train-*.tar
    - root/data/train-*.index.npz
    - root/data/validation-*.tar
    - root/data/validation-*.index.npz
    - root/label_info.json

    Each tar shard stores WebDataset-style samples such as:
    - <key>.jpg
    - <key>.cls
    - <key>.json

    The paired `.index.npz` stores contiguous byte offsets, byte sizes, and
    labels so we can keep random access without scanning tar members at runtime.
    """

    _SPLIT_TO_PREFIX = {
        "train": "train",
        "val": "validation",
        "test": "test",
    }

    def __init__(self, root: str, split: str, transform=None, class_names: list[str] | None = None):
        if split not in self._SPLIT_TO_PREFIX:
            raise ValueError(
                f"ImageNetWebDataset only supports split='train', 'val', or 'test', got {split!r}."
            )

        self.root = Path(root)
        self.data_root = self.root / "data"
        self.split = split
        self.transform = transform

        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Expected shard directory at: {self.data_root}")

        self.classes, self.class_to_idx = self._load_classes(class_names=class_names)
        self.shards = self._build_shards(self._SPLIT_TO_PREFIX[split])
        self._shard_start_indices = [shard.start_index for shard in self.shards]
        self._size = sum(shard.num_rows for shard in self.shards)
        self.labels = self._load_all_labels()
        self.targets = self.labels

        self._tar_files: dict[Path, BinaryIO] = {}

    def _close_tar_files(self) -> None:
        for handle in self._tar_files.values():
            try:
                handle.close()
            except Exception:
                pass
        self._tar_files = {}

    def __del__(self):
        self._close_tar_files()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tar_files"] = {}
        return state

    def _load_classes(self, class_names: list[str] | None) -> tuple[list[str], dict[str, int]]:
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

        if class_names:
            classes = list(class_names)
            class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
            return classes, class_to_idx

        raise FileNotFoundError(
            f"Expected label mapping file at {label_info_path}, or provide explicit class_names."
        )

    def _build_shards(self, split_prefix: str) -> list[_TarShard]:
        tar_paths = sorted(self.data_root.glob(f"{split_prefix}-*.tar"))
        if not tar_paths:
            raise FileNotFoundError(
                f"Couldn't find tar shards matching {split_prefix}-*.tar under {self.data_root}"
            )

        shards: list[_TarShard] = []
        start_index = 0
        for tar_path in tar_paths:
            index_path = tar_path.with_suffix(".index.npz")
            if not index_path.is_file():
                raise FileNotFoundError(f"Expected WebDataset shard index at: {index_path}")

            with np.load(index_path, allow_pickle=False) as index_data:
                offsets = np.asarray(index_data["offsets"], dtype=np.int64)
                sizes = np.asarray(index_data["sizes"], dtype=np.int64)
                labels = np.asarray(index_data["labels"], dtype=np.int64)

            if offsets.shape != sizes.shape or offsets.shape != labels.shape:
                raise ValueError(
                    f"Shard index arrays must share the same shape in {index_path}, "
                    f"got offsets={offsets.shape}, sizes={sizes.shape}, labels={labels.shape}."
                )

            shard_index = _ShardIndex(
                offsets=offsets,
                sizes=sizes,
                labels=labels,
            )
            num_rows = int(labels.shape[0])
            shards.append(
                _TarShard(
                    path=tar_path,
                    index_path=index_path,
                    start_index=start_index,
                    num_rows=num_rows,
                    index=shard_index,
                )
            )
            start_index += num_rows

        return shards

    def _load_all_labels(self) -> list[int]:
        labels: list[int] = []
        for shard in self.shards:
            labels.extend(shard.index.labels.tolist())
        return labels

    def _get_tar_file(self, path: Path) -> BinaryIO:
        handle = self._tar_files.get(path)
        if handle is None or handle.closed:
            handle = path.open("rb")
            self._tar_files[path] = handle
        return handle

    def _lookup_shard(self, index: int) -> tuple[_TarShard, int]:
        shard_index = bisect.bisect_right(self._shard_start_indices, index) - 1
        shard = self.shards[shard_index]
        local_index = index - shard.start_index
        return shard, local_index

    def _read_image_bytes(self, shard: _TarShard, local_index: int) -> tuple[bytes, int]:
        handle = self._get_tar_file(shard.path)
        offset = int(shard.index.offsets[local_index])
        size = int(shard.index.sizes[local_index])
        handle.seek(offset)
        image_bytes = handle.read(size)
        if len(image_bytes) != size:
            raise RuntimeError(
                f"Expected to read {size} bytes from {shard.path} at offset {offset}, "
                f"got {len(image_bytes)} bytes."
            )
        label = int(shard.index.labels[local_index])
        return image_bytes, label

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")

        shard, local_index = self._lookup_shard(index)
        image_bytes, label = self._read_image_bytes(shard, local_index)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label, index

    def __getitems__(self, indices: Sequence[int]):
        return [self[index] for index in indices]


ImageNetShardDataset = ImageNetWebDataset
