from __future__ import annotations

import bisect
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _ShardFile:
    path: Path
    start_index: int
    num_rows: int


class ImageNetH5Dataset(Dataset):
    """
    通用的 ImageNet HDF5 shard 数据集包装器。

    目录结构要求:
    - root/data/train-*.h5
    - root/data/validation-*.h5
    - root/label_info.json

    每个 HDF5 shard 至少需要包含:
    - image_bytes: vlen uint8 / bytes
    - label: int

    若 label_info.json 尚未写出，也可以显式传入 class_names。
    返回 (image, label, index)，与当前训练框架保持一致。
    """

    _SPLIT_TO_PREFIX = {
        "train": "train",
        "val": "validation",
        "test": "test",
    }

    def __init__(self, root: str, split: str, transform=None, class_names: list[str] | None = None):
        if split not in self._SPLIT_TO_PREFIX:
            raise ValueError(
                f"ImageNetH5Dataset only supports split='train', 'val', or 'test', got {split!r}."
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

        self._h5_files: dict[Path, h5py.File] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_files"] = {}
        return state

    def _open_readonly_h5(self, path: Path) -> h5py.File:
        last_error: BlockingIOError | None = None
        for attempt in range(20):
            try:
                return h5py.File(path, "r", locking=False, swmr=True)
            except TypeError:
                # Older h5py builds may not expose locking/swmr kwargs.
                return h5py.File(path, "r")
            except BlockingIOError as error:
                last_error = error
                if attempt == 19:
                    break
                time.sleep(0.5)

        raise BlockingIOError(
            f"Unable to open ImageNet HDF5 shard after 20 attempts due to file locking: {path}"
        ) from last_error

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

    def _build_shards(self, split_prefix: str) -> list[_ShardFile]:
        h5_paths = sorted(self.data_root.glob(f"{split_prefix}-*.h5"))
        if not h5_paths:
            raise FileNotFoundError(
                f"Couldn't find h5 shards matching {split_prefix}-*.h5 under {self.data_root}"
            )
        return self._build_h5_shards(h5_paths)

    def _build_h5_shards(self, shard_paths: list[Path]) -> list[_ShardFile]:
        shards: list[_ShardFile] = []
        start_index = 0
        for shard_path in shard_paths:
            with self._open_readonly_h5(shard_path) as handle:
                if "label" not in handle:
                    raise KeyError(f"HDF5 shard is missing required dataset 'label': {shard_path}")
                num_rows = int(handle["label"].shape[0])

            shards.append(
                _ShardFile(
                    path=shard_path,
                    start_index=start_index,
                    num_rows=num_rows,
                )
            )
            start_index += num_rows
        return shards

    def _load_all_labels(self) -> list[int]:
        labels: list[int] = []
        for shard in self.shards:
            with self._open_readonly_h5(shard.path) as handle:
                labels.extend(np.asarray(handle["label"], dtype=np.int64).tolist())
        return labels

    def _get_h5_file(self, path: Path) -> h5py.File:
        handle = self._h5_files.get(path)
        if handle is None:
            handle = self._open_readonly_h5(path)
            self._h5_files[path] = handle
        return handle

    def _read_h5_row(self, shard_index: int, row_index_in_shard: int) -> tuple[bytes, int]:
        shard = self.shards[shard_index]
        handle = self._get_h5_file(shard.path)
        if "image_bytes" not in handle:
            raise KeyError(f"HDF5 shard is missing required dataset 'image_bytes': {shard.path}")

        image_value = handle["image_bytes"][row_index_in_shard]
        if isinstance(image_value, (bytes, bytearray)):
            image_bytes = bytes(image_value)
        else:
            image_bytes = np.asarray(image_value, dtype=np.uint8).tobytes()

        label = int(handle["label"][row_index_in_shard])
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
        image_bytes, label = self._read_h5_row(shard_index, local_index)

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label, index


ImageNetShardDataset = ImageNetH5Dataset
