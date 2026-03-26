from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset


class TinyImageNetDataset(Dataset):
    """
    Tiny-ImageNet 数据集包装器。

    支持标准目录布局：
    - train/<wnid>/images/*.JPEG
    - val/images/*.JPEG + val_annotations.txt
    - test/images/*.JPEG

    若 val 已经被整理成 val/<wnid>/*.JPEG，也会自动回退到类目录解析。
    test split 没有官方标签时，返回 label=-1。
    """

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

    def __init__(self, root: str, split: str, transform=None):
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"TinyImageNetDataset only supports split='train', 'val', or 'test', got {split!r}."
            )

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.class_to_idx = self._load_class_to_idx()
        self.classes = sorted(self.class_to_idx, key=self.class_to_idx.get)

        if split == "train":
            self.samples = self._load_train_samples()
        elif split == "val":
            self.samples = self._load_val_samples()
        else:
            self.samples = self._load_test_samples()

        if not self.samples:
            raise FileNotFoundError(f"Couldn't find any Tiny-ImageNet samples for split '{split}' under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target, index

    @classmethod
    def _iter_image_files(cls, directory: Path):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in cls._IMAGE_EXTENSIONS:
                yield path

    def _load_class_to_idx(self) -> Dict[str, int]:
        wnids_path = self.root / "wnids.txt"
        if wnids_path.is_file():
            with wnids_path.open("r", encoding="utf-8") as handle:
                classes = [line.strip() for line in handle if line.strip()]
            return {class_name: index for index, class_name in enumerate(classes)}

        train_root = self.root / "train"
        classes = sorted(entry.name for entry in train_root.iterdir() if entry.is_dir())
        return {class_name: index for index, class_name in enumerate(classes)}

    def _load_train_samples(self) -> List[Tuple[str, int]]:
        train_root = self.root / "train"
        if not train_root.is_dir():
            raise FileNotFoundError(f"Expected Tiny-ImageNet train directory at: {train_root}")

        samples: List[Tuple[str, int]] = []
        for class_name in self.classes:
            class_root = train_root / class_name
            if not class_root.is_dir():
                continue
            images_root = class_root / "images"
            search_root = images_root if images_root.is_dir() else class_root
            for image_path in self._iter_image_files(search_root):
                samples.append((str(image_path), self.class_to_idx[class_name]))
        return samples

    def _load_val_samples(self) -> List[Tuple[str, int]]:
        val_root = self.root / "val"
        if not val_root.is_dir():
            raise FileNotFoundError(f"Expected Tiny-ImageNet val directory at: {val_root}")

        annotation_path = val_root / "val_annotations.txt"
        images_root = val_root / "images"
        if annotation_path.is_file() and images_root.is_dir():
            samples: List[Tuple[str, int]] = []
            with annotation_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    image_name, class_name = parts[0], parts[1]
                    if class_name not in self.class_to_idx:
                        continue
                    image_path = images_root / image_name
                    samples.append((str(image_path), self.class_to_idx[class_name]))
            return samples

        samples: List[Tuple[str, int]] = []
        for class_name in self.classes:
            class_root = val_root / class_name
            if not class_root.is_dir():
                continue
            for image_path in self._iter_image_files(class_root):
                samples.append((str(image_path), self.class_to_idx[class_name]))
        return samples

    def _load_test_samples(self) -> List[Tuple[str, int]]:
        test_root = self.root / "test"
        if not test_root.is_dir():
            raise FileNotFoundError(f"Expected Tiny-ImageNet test directory at: {test_root}")

        images_root = test_root / "images"
        search_root = images_root if images_root.is_dir() else test_root
        return [(str(image_path), -1) for image_path in self._iter_image_files(search_root)]
