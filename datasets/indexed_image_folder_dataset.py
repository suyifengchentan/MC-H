from __future__ import annotations

import os
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class IndexedImageFolderDataset(Dataset):
    """
    通用的 ImageFolder 风格包装器。

    支持三类目录布局：
    - root/train/<class>/* 与 root/val|test/<class>/*
    - root/train.X1|train.X2|... 与 root/val.X 这类带后缀 split
    - root/<class>/*

    返回 (image, label, index)，以兼容当前训练框架。
    """

    _SPLIT_NAMES = {"train", "val", "test"}
    _IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
    }

    def __init__(self, root: str, split: str | None = None, transform=None, target_transform=None):
        self.root_dir = root
        self.split = split or "all"
        self.transform = transform
        self.target_transform = target_transform

        self.resolved_roots = self._resolve_roots(root=root, split=split)
        self.root = str(self.resolved_roots[0])
        self.classes, self.class_to_idx = self.find_classes(self.resolved_roots)
        self.samples = self._make_dataset(self.resolved_roots, self.class_to_idx)
        if not self.samples:
            raise FileNotFoundError(f"Couldn't find any image file under split roots: {self.resolved_roots}")

        self.targets = [target for _, target in self.samples]
        self.imgs = self.samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = default_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, index

    @classmethod
    def _split_prefix(cls, name: str) -> str:
        return name.split(".", 1)[0]

    @classmethod
    def _list_visible_dirs(cls, root: str | Path) -> list[Path]:
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(f"Expected dataset directory does not exist: {root_path}")
        return sorted(
            entry
            for entry in root_path.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        )

    @classmethod
    def _is_split_dir(cls, name: str) -> bool:
        return cls._split_prefix(name) in cls._SPLIT_NAMES

    @classmethod
    def _has_direct_class_folders(cls, root: str | Path) -> bool:
        directories = cls._list_visible_dirs(root)
        return bool(directories) and not all(cls._is_split_dir(entry.name) for entry in directories)

    @classmethod
    def _resolve_roots(cls, root: str, split: str | None) -> list[Path]:
        root_path = Path(root)
        if split:
            exact_split_root = root_path / split
            if exact_split_root.is_dir():
                return [exact_split_root]

            alias_roots = [
                entry
                for entry in cls._list_visible_dirs(root_path)
                if cls._split_prefix(entry.name) == split
            ]
            if alias_roots:
                return alias_roots

        if cls._has_direct_class_folders(root_path):
            return [root_path]

        if split:
            raise FileNotFoundError(
                f"Expected split directory or split alias for '{split}' under: {root_path}"
            )
        raise FileNotFoundError(f"Expected ImageFolder-style class directories under: {root_path}")

    @classmethod
    def find_classes(cls, directories: list[Path]) -> tuple[list[str], dict[str, int]]:
        classes = sorted(
            {
                entry.name
                for directory in directories
                for entry in os.scandir(directory)
                if entry.is_dir() and not entry.name.startswith(".")
            }
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directories}.")
        return classes, {class_name: index for index, class_name in enumerate(classes)}

    @classmethod
    def _iter_image_paths(cls, directory: Path):
        for current_root, dir_names, file_names in os.walk(directory):
            dir_names[:] = sorted(name for name in dir_names if not name.startswith("."))
            for file_name in sorted(file_name for file_name in file_names if not file_name.startswith(".")):
                path = Path(current_root) / file_name
                if path.suffix.lower() in cls._IMAGE_EXTENSIONS:
                    yield path

    @classmethod
    def _make_dataset(cls, directories: list[Path], class_to_idx: dict[str, int]) -> list[tuple[str, int]]:
        samples: list[tuple[str, int]] = []
        for directory in directories:
            for class_name in sorted(class_to_idx):
                class_dir = directory / class_name
                if not class_dir.is_dir():
                    continue
                target = class_to_idx[class_name]
                for image_path in cls._iter_image_paths(class_dir):
                    samples.append((str(image_path), target))
        return samples
