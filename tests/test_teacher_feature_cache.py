from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from hypersphere_dpp.data import (
    TEACHER_FEATURE_CANDIDATES_KEY,
    TeacherFeatureTrainDataset,
    sample_two_teacher_feature_views,
)
from hypersphere_dpp.teacher_cache import TeacherFeatureCache


class _IndexedDataset(Dataset):
    def __init__(self):
        self.targets = [4, 8, 15]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return None, self.targets[index], index


class _DataBackedDataset(Dataset):
    def __init__(self):
        self.data = [("a.jpg", 4), ("b.jpg", 8), ("c.jpg", 15)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        _path, label = self.data[index]
        return None, label, index


class TeacherFeatureCacheTest(unittest.TestCase):
    def _create_cache(self, root: Path) -> None:
        data_root = root / "data"
        data_root.mkdir()
        values = np.arange(3 * 8 * 512, dtype=np.float16).reshape(3, 8, 512)
        values.tofile(data_root / "train.mmap")
        with (data_root / "train.mmap.progress.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {"shape": [3, 8, 512], "dtype": "float16", "rows_written": 3},
                handle,
            )
        with (root / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "num_views": 8,
                    "feature_dim": 512,
                    "dtype": "float16",
                    "storage_format": "memmap",
                    "shards": [{"name": "train.mmap", "start_index": 0, "num_rows": 3}],
                },
                handle,
            )

    def test_subset_returns_candidate_features_by_original_dataset_index(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir)
            self._create_cache(cache_root)
            cache = TeacherFeatureCache(cache_root, expected_num_views=8, expected_feature_dim=512)
            dataset = TeacherFeatureTrainDataset(Subset(_IndexedDataset(), [2, 0]), cache)

            second_sample = dataset[1]
            self.assertEqual(second_sample["index"], 0)
            self.assertEqual(second_sample["label"], 4)
            self.assertEqual(second_sample[TEACHER_FEATURE_CANDIDATES_KEY].shape, (8, 512))
            self.assertEqual(second_sample[TEACHER_FEATURE_CANDIDATES_KEY][0, 0].item(), 0.0)

    def test_subset_supports_data_backed_labels(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir)
            self._create_cache(cache_root)
            cache = TeacherFeatureCache(cache_root, expected_num_views=8, expected_feature_dim=512)
            dataset = TeacherFeatureTrainDataset(Subset(_DataBackedDataset(), [2, 0]), cache)

            self.assertEqual(dataset[0]["label"], 15)
            self.assertEqual(dataset[1]["label"], 4)
            self.assertEqual(
                [sample["label"] for sample in dataset.__getitems__([0, 1])],
                [15, 4],
            )

    def test_random_two_view_sampling_uses_distinct_candidates(self):
        candidates = torch.arange(2 * 8 * 512, dtype=torch.float32).reshape(2, 8, 512)
        first, second = sample_two_teacher_feature_views(candidates)

        self.assertEqual(first.shape, (2, 512))
        self.assertEqual(second.shape, (2, 512))
        self.assertTrue(torch.all(first[:, 0] != second[:, 0]))


if __name__ == "__main__":
    unittest.main()
