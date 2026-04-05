from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import imagenet_subset_utils as module


class ImageNetSubsetUtilsTest(unittest.TestCase):
    def test_build_subset_specs_supports_imagenet1k_identity_mapping(self):
        imagenet_items = [
            ("n00000001", "class one"),
            ("n00000002", "class two"),
        ]

        [spec] = module.build_subset_specs(["ImageNet1K"], imagenet_items)

        self.assertEqual(spec.dataset_name, "ImageNet1K")
        self.assertEqual(spec.output_root, Path("/home/geyingjie/hypersphere-data/ImageNet1K"))
        self.assertEqual([entry.wnid for entry in spec.entries], ["n00000001", "n00000002"])
        self.assertEqual(spec.source_label_to_subset_label, {0: 0, 1: 1})
        self.assertEqual(spec.subset_label_to_wnid, {0: "n00000001", 1: "n00000002"})
        self.assertEqual(spec.subset_label_to_name, {0: "class one", 1: "class two"})

    def test_discover_source_splits_detects_test_shards(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = Path(tmp_dir)
            (data_root / "train-00000-of-00001.parquet").touch()
            (data_root / "test-00000-of-00001.parquet").touch()

            self.assertEqual(module.discover_source_splits(data_root), ["train", "test"])

    def test_build_label_info_contains_expected_mappings(self):
        imagenet_items = [
            ("n00000001", "class one"),
            ("n00000002", "class two"),
        ]

        [spec] = module.build_subset_specs(["ImageNet1K"], imagenet_items)
        label_info = module.build_label_info(spec)

        self.assertEqual(label_info["idx_to_wnid"], {"0": "n00000001", "1": "n00000002"})
        self.assertEqual(label_info["idx_to_class_name"], {"0": "class one", "1": "class two"})
        self.assertEqual(label_info["class_to_idx"], {"n00000001": 0, "n00000002": 1})


if __name__ == "__main__":
    unittest.main()
