from __future__ import annotations

import os

from torchvision.datasets import ImageFolder


class IndexedImageFolderDataset(ImageFolder):
    """
    通用的 ImageFolder 包装器。

    约定数据目录为:
    - root/train/<class>/*.jpg
    - root/val/<class>/*.jpg 或 root/test/<class>/*.jpg

    返回 (image, label, index)，以兼容当前训练框架。
    """

    def __init__(self, root: str, split: str, transform=None, target_transform=None):
        split_root = os.path.join(root, split)
        if not os.path.isdir(split_root):
            raise FileNotFoundError(f"Expected split directory does not exist: {split_root}")

        super().__init__(
            root=split_root,
            transform=transform,
            target_transform=target_transform,
        )
        self.split = split

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample, target, index

