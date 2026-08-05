from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader


class Image10Dataset(datasets.ImageFolder):
    """Image10 loader backed by ImageFolder instead of ImageNet HDF5 shards.

    Supported layouts:
    - root/<wnid>/*.JPEG
    - root/train/<wnid>/*.JPEG and root/test/<wnid>/*.JPEG
    - root/train/<wnid>/*.JPEG and root/val/<wnid>/*.JPEG
    """

    _SPLIT_CANDIDATES = {
        "train": ("train",),
        "val": ("val", "validation", "test"),
        "test": ("test", "val", "validation"),
    }

    def __init__(self, root: str, split: str = "train", transform=None, target_transform=None):
        if split not in self._SPLIT_CANDIDATES:
            raise ValueError(f"Unsupported Image10 split: {split!r}")

        base_root = Path(root)
        split_root = self._resolve_split_root(base_root, split)

        if not split_root.is_dir():
            raise FileNotFoundError(f"Image10 split directory not found: {split_root}")

        super().__init__(
            root=str(split_root),
            loader=default_loader,
            transform=transform,
            target_transform=target_transform,
        )
        self.root = str(split_root)
        self.split = split
        self.labels = list(self.targets)

    @classmethod
    def _resolve_split_root(cls, base_root: Path, split: str) -> Path:
        for candidate in cls._SPLIT_CANDIDATES[split]:
            candidate_root = base_root / candidate
            if candidate_root.is_dir():
                return candidate_root
        return base_root

    def __getitem__(self, index: int):
        sample, target = super().__getitem__(index)
        return sample, target, index

    @staticmethod
    def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = [
            entry.name
            for entry in os.scandir(directory)
            if (
                entry.is_dir()
                and not entry.name.startswith(".")
                and not entry.name.startswith("teacher_cache_")
            )
        ]
        classes.sort()
        class_to_idx = {cls_name: index for index, cls_name in enumerate(classes)}
        return classes, class_to_idx
