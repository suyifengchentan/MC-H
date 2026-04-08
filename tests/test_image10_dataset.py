from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from datasets.image10_dataset import Image10Dataset
from hypersphere_dpp.config import ExperimentConfig
from hypersphere_dpp.data import build_experiment_dataloaders


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


class Image10DatasetTest(unittest.TestCase):
    def test_image10_dataset_supports_root_level_imagefolder_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "imagenet-10"
            _write_image(root / "n02056570" / "sample_1.JPEG", (255, 0, 0))
            _write_image(root / "n02085936" / "sample_2.JPEG", (0, 255, 0))

            train_dataset = Image10Dataset(str(root), split="train")
            test_dataset = Image10Dataset(str(root), split="test")

            self.assertEqual(len(train_dataset), 2)
            self.assertEqual(len(test_dataset), 2)
            self.assertEqual(train_dataset.classes, ["n02056570", "n02085936"])
            self.assertEqual(train_dataset.class_to_idx["n02085936"], 1)

            image, label, sample_index = train_dataset[1]
            self.assertEqual(image.size, (8, 8))
            self.assertEqual(label, 1)
            self.assertEqual(sample_index, 1)

    def test_image10_dataloaders_use_imagefolder_instead_of_h5_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "imagenet-10"
            for class_name, color in (
                ("n02056570", (255, 0, 0)),
                ("n02085936", (0, 255, 0)),
            ):
                for sample_index in range(10):
                    _write_image(
                        root / class_name / f"sample_{sample_index}.JPEG",
                        (color[0], color[1], sample_index),
                    )

            config = ExperimentConfig(
                dataset="Image10",
                dataset_dir=str(root),
                class_num=2,
                image_size=8,
                feature_dim=512,
                backbone_name="ViT-B-16",
                pretrained="laion2b_s34b_b88k",
                batch_size=4,
                eval_batch_size=4,
                workers=0,
                validation_ratio=0.2,
                official_eval_split="test",
                semantic_prototype_enabled=False,
            )

            bundle = build_experiment_dataloaders(config)

            self.assertTrue(bundle.val_is_stratified)
            self.assertEqual(len(bundle.train_loader.dataset), 16)
            self.assertEqual(len(bundle.val_loader.dataset), 4)
            self.assertEqual(len(bundle.test_loader.dataset), 20)


if __name__ == "__main__":
    unittest.main()
