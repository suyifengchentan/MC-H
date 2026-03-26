from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from datasets.indexed_image_folder_dataset import IndexedImageFolderDataset
from datasets.tiny_imagenet_dataset import TinyImageNetDataset


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


class DatasetFormatsTest(unittest.TestCase):
    def test_indexed_image_folder_resolves_split_alias_groups(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_image(root / "train.X1" / "n0001" / "a.jpg", (255, 0, 0))
            _write_image(root / "train.X2" / "n0002" / "b.jpg", (0, 255, 0))
            _write_image(root / "val.X" / "n0001" / "c.jpg", (0, 0, 255))
            _write_image(root / "val.X" / "n0002" / "d.jpg", (255, 255, 0))

            train_dataset = IndexedImageFolderDataset(str(root), split="train")
            val_dataset = IndexedImageFolderDataset(str(root), split="val")

            self.assertEqual(len(train_dataset), 2)
            self.assertEqual(len(val_dataset), 2)
            self.assertEqual(train_dataset.classes, ["n0001", "n0002"])
            self.assertEqual(train_dataset.class_to_idx["n0001"], 0)
            self.assertEqual(train_dataset.class_to_idx["n0002"], 1)
            self.assertEqual(sorted(target for _, target in train_dataset.samples), [0, 1])

    def test_tiny_imagenet_supports_standard_train_val_and_test(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            with (root / "wnids.txt").open("w", encoding="utf-8") as handle:
                handle.write("n0001\nn0002\n")

            _write_image(root / "train" / "n0001" / "images" / "train_1.JPEG", (255, 0, 0))
            _write_image(root / "train" / "n0002" / "images" / "train_2.JPEG", (0, 255, 0))
            _write_image(root / "val" / "images" / "val_1.JPEG", (0, 0, 255))
            _write_image(root / "val" / "images" / "val_2.JPEG", (255, 255, 0))
            _write_image(root / "test" / "images" / "test_1.JPEG", (0, 255, 255))

            with (root / "val" / "val_annotations.txt").open("w", encoding="utf-8") as handle:
                handle.write("val_1.JPEG\tn0002\t0\t0\t0\t0\n")
                handle.write("val_2.JPEG\tn0001\t0\t0\t0\t0\n")

            train_dataset = TinyImageNetDataset(str(root), split="train")
            val_dataset = TinyImageNetDataset(str(root), split="val")
            test_dataset = TinyImageNetDataset(str(root), split="test")

            self.assertEqual(len(train_dataset), 2)
            self.assertEqual(len(val_dataset), 2)
            self.assertEqual(len(test_dataset), 1)
            self.assertEqual(train_dataset.class_to_idx["n0001"], 0)
            self.assertEqual(train_dataset.class_to_idx["n0002"], 1)
            self.assertEqual({target for _, target in train_dataset.samples}, {0, 1})
            self.assertEqual({target for _, target in val_dataset.samples}, {0, 1})
            self.assertEqual(test_dataset.samples[0][1], -1)


if __name__ == "__main__":
    unittest.main()
