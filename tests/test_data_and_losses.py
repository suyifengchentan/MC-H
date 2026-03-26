from __future__ import annotations

import unittest

import torch

from hypersphere_dpp.data import split_train_indices
from hypersphere_dpp.losses import dpp_diversity_loss
from hypersphere_dpp.subspace import ClusterFeatureQueue


class DataAndLossesTest(unittest.TestCase):
    def test_split_reproducibility(self):
        train_a, val_a = split_train_indices(100, 0.2, 42)
        train_b, val_b = split_train_indices(100, 0.2, 42)

        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(len(train_a), 80)
        self.assertEqual(len(val_a), 20)
        self.assertEqual(set(train_a).intersection(val_a), set())

    def test_dpp_loss_is_finite(self):
        features = torch.nn.functional.normalize(torch.randn(8, 768), dim=-1)
        assignments = torch.softmax(torch.randn(8, 4), dim=-1)
        loss, rho, relative_quality = dpp_diversity_loss(features, assignments)

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(rho))
        self.assertEqual(relative_quality.shape, (8,))

    def test_queue_respects_capacity(self):
        queue = ClusterFeatureQueue(num_clusters=3, feature_dim=4, queue_size=2)
        teacher_features = torch.randn(5, 4)
        assignments = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        queue.enqueue(teacher_features, assignments)

        features, weights = queue.get(0, torch.device("cpu"))
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(weights.shape[0], 2)


if __name__ == "__main__":
    unittest.main()

