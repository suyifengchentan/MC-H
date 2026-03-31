from __future__ import annotations

import io
import zipfile
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
        self._storage_mode, self.dataset_root, self.archive_prefix = self._resolve_dataset_root(
            self.root
        )
        self._zip_handle: zipfile.ZipFile | None = None
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
        image_ref, target = self.samples[index]
        image = self._load_image(image_ref)
        if self.transform:
            image = self.transform(image)
        return image, target, index

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zip_handle"] = None
        return state

    @classmethod
    def _iter_image_files(cls, directory: Path):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in cls._IMAGE_EXTENSIONS:
                yield path

    @classmethod
    def _resolve_dataset_root(
        cls, root: Path
    ) -> tuple[str, Path, str]:
        if cls._looks_like_dataset_root(root):
            return "dir", root, ""

        nested_root = root / "Tiny-ImageNet-200"
        if cls._looks_like_dataset_root(nested_root):
            return "dir", nested_root, ""

        archive_path = cls._resolve_archive_path(root)
        if archive_path is not None:
            archive_prefix = cls._detect_archive_prefix(archive_path)
            return "zip", archive_path, archive_prefix

        raise FileNotFoundError(
            "Tiny-ImageNet root must either contain train/val/test directories or a Tiny-ImageNet zip archive: "
            f"{root}"
        )

    @staticmethod
    def _looks_like_dataset_root(root: Path) -> bool:
        return (root / "train").is_dir() and (root / "val").is_dir() and (root / "test").is_dir()

    @staticmethod
    def _resolve_archive_path(root: Path) -> Path | None:
        if root.is_file() and root.suffix.lower() == ".zip":
            return root

        candidates = sorted(root.glob("*.zip"))
        if not candidates:
            return None

        preferred = next(
            (candidate for candidate in candidates if candidate.name == "Tiny-ImageNet-200.zip"),
            None,
        )
        return preferred or candidates[0]

    @staticmethod
    def _detect_archive_prefix(archive_path: Path) -> str:
        with zipfile.ZipFile(archive_path) as archive:
            for member_name in archive.namelist():
                normalized = member_name.rstrip("/")
                if normalized.endswith("/wnids.txt"):
                    return normalized[: -len("wnids.txt")]
                if normalized == "wnids.txt":
                    return ""
        return ""

    def _ensure_zip_handle(self) -> zipfile.ZipFile:
        if self._zip_handle is None:
            self._zip_handle = zipfile.ZipFile(self.dataset_root)
        return self._zip_handle

    def _load_image(self, image_ref: str) -> Image.Image:
        if self._storage_mode == "dir":
            return Image.open(image_ref).convert("RGB")

        archive = self._ensure_zip_handle()
        with archive.open(image_ref) as handle:
            return Image.open(io.BytesIO(handle.read())).convert("RGB")

    def _archive_path(self, relative_path: str) -> str:
        return f"{self.archive_prefix}{relative_path}" if self.archive_prefix else relative_path

    def _load_class_to_idx(self) -> Dict[str, int]:
        classes = self._load_wnids()
        if not classes:
            train_root = self.dataset_root / "train"
            classes = sorted(entry.name for entry in train_root.iterdir() if entry.is_dir())
        return {class_name: index for index, class_name in enumerate(classes)}

    def _load_wnids(self) -> list[str]:
        if self._storage_mode == "dir":
            wnids_path = self.dataset_root / "wnids.txt"
            if not wnids_path.is_file():
                return []
            with wnids_path.open("r", encoding="utf-8") as handle:
                return [line.strip() for line in handle if line.strip()]

        archive = self._ensure_zip_handle()
        member_name = self._archive_path("wnids.txt")
        try:
            with archive.open(member_name) as handle:
                return [
                    line.decode("utf-8").strip()
                    for line in handle.readlines()
                    if line.decode("utf-8").strip()
                ]
        except KeyError:
            return []

    def _load_train_samples(self) -> List[Tuple[str, int]]:
        if self._storage_mode == "dir":
            train_root = self.dataset_root / "train"
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

        archive = self._ensure_zip_handle()
        samples: List[Tuple[str, int]] = []
        for class_name in self.classes:
            class_prefix = self._archive_path(f"train/{class_name}/")
            image_prefix = self._archive_path(f"train/{class_name}/images/")
            for member_name in sorted(archive.namelist()):
                if not member_name.startswith(class_prefix):
                    continue
                candidate_name = member_name.rstrip("/")
                if candidate_name == class_prefix.rstrip("/"):
                    continue
                suffix = Path(candidate_name).suffix.lower()
                if suffix not in self._IMAGE_EXTENSIONS:
                    continue
                if candidate_name.startswith(image_prefix) or candidate_name.startswith(class_prefix):
                    samples.append((candidate_name, self.class_to_idx[class_name]))
        return samples

    def _load_val_samples(self) -> List[Tuple[str, int]]:
        if self._storage_mode == "dir":
            val_root = self.dataset_root / "val"
            if not val_root.is_dir():
                raise FileNotFoundError(f"Expected Tiny-ImageNet val directory at: {val_root}")

            annotation_path = val_root / "val_annotations.txt"
            images_root = val_root / "images"
            if annotation_path.is_file() and images_root.is_dir():
                return self._parse_val_annotations(
                    annotation_lines=annotation_path.read_text(encoding="utf-8").splitlines(),
                    images_prefix=str(images_root),
                )

            samples: List[Tuple[str, int]] = []
            for class_name in self.classes:
                class_root = val_root / class_name
                if not class_root.is_dir():
                    continue
                for image_path in self._iter_image_files(class_root):
                    samples.append((str(image_path), self.class_to_idx[class_name]))
            return samples

        archive = self._ensure_zip_handle()
        annotation_name = self._archive_path("val/val_annotations.txt")
        images_prefix = self._archive_path("val/images")
        try:
            with archive.open(annotation_name) as handle:
                lines = handle.read().decode("utf-8").splitlines()
            return self._parse_val_annotations(lines, images_prefix)
        except KeyError:
            samples: List[Tuple[str, int]] = []
            for class_name in self.classes:
                class_prefix = self._archive_path(f"val/{class_name}/")
                for member_name in sorted(archive.namelist()):
                    candidate_name = member_name.rstrip("/")
                    if not candidate_name.startswith(class_prefix):
                        continue
                    if Path(candidate_name).suffix.lower() not in self._IMAGE_EXTENSIONS:
                        continue
                    samples.append((candidate_name, self.class_to_idx[class_name]))
            return samples

    def _parse_val_annotations(
        self, annotation_lines: list[str], images_prefix: str
    ) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        for line in annotation_lines:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            image_name, class_name = parts[0], parts[1]
            if class_name not in self.class_to_idx:
                continue
            image_ref = str(Path(images_prefix) / image_name)
            samples.append((image_ref, self.class_to_idx[class_name]))
        return samples

    def _load_test_samples(self) -> List[Tuple[str, int]]:
        if self._storage_mode == "dir":
            test_root = self.dataset_root / "test"
            if not test_root.is_dir():
                raise FileNotFoundError(f"Expected Tiny-ImageNet test directory at: {test_root}")

            images_root = test_root / "images"
            search_root = images_root if images_root.is_dir() else test_root
            return [(str(image_path), -1) for image_path in self._iter_image_files(search_root)]

        archive = self._ensure_zip_handle()
        images_prefix = self._archive_path("test/images/")
        samples: List[Tuple[str, int]] = []
        for member_name in sorted(archive.namelist()):
            candidate_name = member_name.rstrip("/")
            if not candidate_name.startswith(images_prefix):
                continue
            if Path(candidate_name).suffix.lower() not in self._IMAGE_EXTENSIONS:
                continue
            samples.append((candidate_name, -1))
        return samples


def load_tiny_imagenet_class_texts(root: str) -> list[str]:
    dataset = TinyImageNetDataset(root=root, split="train", transform=None)
    words_map = _load_tiny_imagenet_words_map(dataset)
    return [words_map.get(class_name, class_name) for class_name in dataset.classes]


def _load_tiny_imagenet_words_map(dataset: TinyImageNetDataset) -> dict[str, str]:
    if dataset._storage_mode == "dir":
        words_path = dataset.dataset_root / "words.txt"
        if not words_path.is_file():
            return {}
        lines = words_path.read_text(encoding="utf-8").splitlines()
    else:
        archive = dataset._ensure_zip_handle()
        words_member = dataset._archive_path("words.txt")
        try:
            with archive.open(words_member) as handle:
                lines = handle.read().decode("utf-8").splitlines()
        except KeyError:
            return {}

    words_map: dict[str, str] = {}
    for line in lines:
        parts = line.strip().split("\t", maxsplit=1)
        if len(parts) != 2:
            continue
        wnid, label_text = parts
        cleaned = label_text.split(",", 1)[0].strip()
        if wnid and cleaned:
            words_map[wnid] = cleaned
    return words_map
