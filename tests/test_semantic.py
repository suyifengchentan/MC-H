from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hypersphere_dpp.semantic import load_class_texts


class SemanticClassTextTest(unittest.TestCase):
    def test_image10_uses_label_config_class_texts(self):
        self.assertEqual(
            load_class_texts("Image10", "/unused"),
            [
                "king penguin",
                "Maltese dog",
                "snow leopard",
                "airliner",
                "airship",
                "container ship",
                "soccer ball",
                "sports car",
                "trailer truck",
                "orange",
            ],
        )

    def test_imagenet1k_can_load_class_texts_from_h5_label_info(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with (root / "label_info.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "idx_to_wnid": {"0": "n00000001", "1": "n00000002"},
                        "idx_to_class_name": {"0": "class one", "1": "class two"},
                    },
                    handle,
                )

            self.assertEqual(
                load_class_texts("ImageNet1K", str(root)),
                ["class one", "class two"],
            )

    def test_imagenet1k_class_texts_are_canonicalized_to_primary_label(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with (root / "label_info.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "idx_to_wnid": {"0": "n00000001", "1": "n00000002"},
                        "idx_to_class_name": {
                            "0": "great white shark, white shark, man-eater",
                            "1": "water_ouzel, dipper",
                        },
                    },
                    handle,
                )

            self.assertEqual(
                load_class_texts("ImageNet1K", str(root)),
                ["great white shark", "water ouzel"],
            )


if __name__ == "__main__":
    unittest.main()
