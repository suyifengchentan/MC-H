from __future__ import annotations

import unittest

import torch

from hypersphere_dpp.losses import dpp_diversity_loss


class DiversityLossTest(unittest.TestCase):
    def test_class_aware_dpp_is_finite(self):
        features = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
        assignments = torch.tensor(
            [
                [0.95, 0.05, 0.0],
                [0.90, 0.10, 0.0],
                [0.85, 0.15, 0.0],
                [0.10, 0.90, 0.0],
                [0.05, 0.95, 0.0],
                [0.0, 0.20, 0.80],
                [0.0, 0.10, 0.90],
                [0.0, 0.15, 0.85],
            ],
            dtype=torch.float32,
        )

        loss, rho, quality = dpp_diversity_loss(
            features=features,
            assignments=assignments,
            class_aware=True,
            class_aware_min_mass=1.0,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(rho))
        self.assertEqual(quality.numel(), 3)


if __name__ == "__main__":
    unittest.main()
