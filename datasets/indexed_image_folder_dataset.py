from __future__ import annotations

import os

from torchvision.datasets import ImageFolder


class IndexedImageFolderDataset(ImageFolder):
    """
    通用的 ImageFolder 包装器。

    支持两类目录布局:
    - root/train/<class>/*.jpg 与 root/val|test/<class>/*.jpg
    - root/<class>/*.jpg

    当根目录本身就是类别文件夹布局时，`split=None` 会直接加载根目录；
    对于像 Image10 这种只有单一导出 split 的数据，如果请求的 split
    目录不存在，也会回退到根目录本身。

    返回 (image, label, index)，以兼容当前训练框架。
    """

    def __init__(self, root: str, split: str | None = None, transform=None, target_transform=None):
        resolved_root = self._resolve_root(root=root, split=split)

        super().__init__(
            root=resolved_root,
            transform=transform,
            target_transform=target_transform,
        )
        self.root_dir = root
        self.split = split or "all"

    @staticmethod
    def _has_direct_class_folders(root: str) -> bool:
        split_names = {"train", "val", "test"}
        class_dirs = []
        for entry in os.scandir(root):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            class_dirs.append(entry.name)
        return bool(class_dirs) and not all(name in split_names for name in class_dirs)

    @classmethod
    def _resolve_root(cls, root: str, split: str | None) -> str:
        if split:
            split_root = os.path.join(root, split)
            if os.path.isdir(split_root):
                return split_root

        if cls._has_direct_class_folders(root):
            return root

        if split:
            raise FileNotFoundError(f"Expected split directory does not exist: {os.path.join(root, split)}")
        raise FileNotFoundError(f"Expected ImageFolder-style class directories under: {root}")

    def find_classes(self, directory: str):
        classes = sorted(
            entry.name
            for entry in os.scandir(directory)
            if entry.is_dir() and not entry.name.startswith(".")
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        return classes, {class_name: index for index, class_name in enumerate(classes)}

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample, target, index
