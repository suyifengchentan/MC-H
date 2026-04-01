from __future__ import annotations

import unittest

import torch

from hypersphere_dpp.losses import dpp_diversity_loss
from hypersphere_dpp.semantic import semantic_spectral_alignment_loss


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

    def test_semantic_spectral_alignment_is_finite_and_differentiable(self):
        prototypes = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1).requires_grad_(True)
        assignments = torch.tensor(
            [
                [0.80, 0.15, 0.05],
                [0.75, 0.20, 0.05],
                [0.10, 0.80, 0.10],
                [0.05, 0.15, 0.80],
            ],
            dtype=torch.float32,
        )
        teacher_features = torch.nn.functional.normalize(torch.randn(4, 4), dim=-1)
        text_features = torch.nn.functional.normalize(torch.randn(5, 4), dim=-1)
        teacher_logit_scale = torch.tensor(2.0)

        loss = semantic_spectral_alignment_loss(
            prototypes=prototypes,
            assignments=assignments,
            teacher_features=teacher_features,
            text_features=text_features,
            teacher_logit_scale=teacher_logit_scale,
            temperature=0.2,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(prototypes.grad)
        self.assertTrue(torch.isfinite(prototypes.grad).all())


if __name__ == "__main__":
    unittest.main()
