from __future__ import annotations

import unittest

from hypersphere_dpp.labels import load_label_entries


class LabelConfigTest(unittest.TestCase):
    def test_imagenet_subset_label_files_are_parseable(self):
        image50 = load_label_entries("Image50")
        image100 = load_label_entries("Image100")
        image200 = load_label_entries("Image200")

        self.assertEqual(len(image50), 50)
        self.assertEqual(len(image100), 100)
        self.assertEqual(len(image200), 200)
        self.assertEqual(image50[0].wnid, "n01601694")
        self.assertEqual(image100[0].wnid, "n01558993")
        self.assertEqual(image200[0].wnid, "n01443537")


if __name__ == "__main__":
    unittest.main()
