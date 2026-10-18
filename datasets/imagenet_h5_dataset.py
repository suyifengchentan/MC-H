from __future__ import annotations

import bisect
import importlib.util
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image
from torch.utils.data import Dataset

from src.hdf5_utils import open_readonly_h5


@dataclass(frozen=True)
class _H5Shard:
    path: Path
    start_index: int
    num_rows: int


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


class ImageNetH5Dataset(Dataset):
    """
    Map-style ImageNet HDF5 shard dataset.

    Expected layout:
    - root/data/train-*.h5
    - root/data/validation-*.h5 or root/data/test-*.h5
    - root/label_info.json or root/classes.py

    Each HDF5 shard is expected to store:
    - image_bytes: variable-length uint8 arrays
    - label: integer labels
    - path: image path bytes
    - wnid: class wnid bytes
    """

    _SPLIT_TO_PREFIXES = {
        "train": ("train",),
        "val": ("validation", "val"),
        "test": ("test", "validation", "val"),
    }

    def __init__(self, root: str, split: str, transform=None, class_names: list[str] | None = None):
        if split not in self._SPLIT_TO_PREFIXES:
            raise ValueError(
                f"ImageNetH5Dataset only supports split='train', 'val', or 'test', got {split!r}."
            )

        self.root = Path(root)
        self.data_root = self.root / "data"
        self.split = split
        self.transform = transform

        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Expected shard directory at: {self.data_root}")

        self._h5_files: dict[Path, object] = {}
        self.classes, self.class_to_idx = self._load_classes(class_names=class_names)
        self.shards = self._build_shards(self._SPLIT_TO_PREFIXES[split])
        self._shard_start_indices = [shard.start_index for shard in self.shards]
        self._size = sum(shard.num_rows for shard in self.shards)
        self.labels = self._load_all_labels()
        self.targets = self.labels

    def __del__(self):
        if hasattr(self, "_h5_files"):
            self._close_h5_files()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_files"] = {}
        return state

    def _close_h5_files(self) -> None:
        for handle in self._h5_files.values():
            try:
                handle.close()
            except Exception:
                pass
        self._h5_files = {}

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

        return _load_imagenet2012_classes(self.root)

    def _build_shards(self, split_prefixes: tuple[str, ...]) -> list[_H5Shard]:
        shard_paths: list[Path] = []
        for split_prefix in split_prefixes:
            shard_paths = sorted(self.data_root.glob(f"{split_prefix}-*.h5"))
            if shard_paths:
                break
        if not shard_paths:
            expected = ", ".join(f"{prefix}-*.h5" for prefix in split_prefixes)
            raise FileNotFoundError(
                f"Couldn't find HDF5 shards matching any of [{expected}] under {self.data_root}"
            )

        shards: list[_H5Shard] = []
        start_index = 0
        for shard_path in shard_paths:
            with open_readonly_h5(shard_path) as handle:
                if "label" not in handle:
                    raise KeyError(f"HDF5 shard is missing 'label' dataset: {shard_path}")
                num_rows = int(handle["label"].shape[0])
            shards.append(
                _H5Shard(
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
            handle = self._get_h5_file(shard.path)
            labels.extend(int(label) for label in handle["label"][:])
        return labels

    def _get_h5_file(self, path: Path):
        handle = self._h5_files.get(path)
        if handle is None:
            handle = open_readonly_h5(path)
            self._h5_files[path] = handle
        return handle

    def _lookup_shard(self, index: int) -> tuple[_H5Shard, int]:
        shard_index = bisect.bisect_right(self._shard_start_indices, index) - 1
        shard = self.shards[shard_index]
        local_index = index - shard.start_index
        return shard, local_index

    @staticmethod
    def _decode_image_bytes(raw_value) -> bytes:
        if isinstance(raw_value, bytes):
            return raw_value
        if hasattr(raw_value, "tobytes"):
            return raw_value.tobytes()
        return bytes(raw_value)

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")

        shard, local_index = self._lookup_shard(index)
        handle = self._get_h5_file(shard.path)
        image_bytes = self._decode_image_bytes(handle["image_bytes"][local_index])
        label = int(handle["label"][local_index])

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, label, index

    def __getitems__(self, indices: Sequence[int]):
        return [self[index] for index in indices]
