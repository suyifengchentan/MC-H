from __future__ import annotations

import unittest

import numpy as np

from experiments.clustering_eval_utils import (
    clustering_metrics,
    fit_spherical_kmeans,
    rectangular_hungarian_accuracy,
)
from experiments.evaluate_cifar100_mixed import (
    balanced_coarse_superclass_sets,
    build_mixed_labels,
    fine_merge_rate,
    fragmentation_at_coverage,
)
from rebuttal.experiments.common import (
    SPEED_OVERRIDES,
    apply_adaptive_semantic_guidance_policy,
)


class RebuttalExperimentTests(unittest.TestCase):
    def test_semantic_guidance_is_enabled_only_for_adaptive_hdpp(self):
        self.assertNotIn("semantic_prototype_enabled", SPEED_OVERRIDES)

        adaptive = {
            "repulsion_mode": "adaptive_hdpp",
            "semantic_prototype_enabled": False,
            "lambda_semantic": 0.05,
        }
        apply_adaptive_semantic_guidance_policy(adaptive)
        self.assertTrue(adaptive["semantic_prototype_enabled"])
        self.assertEqual(adaptive["lambda_semantic"], 0.05)

        constant = {
            "repulsion_mode": "constant_hdpp",
            "semantic_prototype_enabled": True,
            "lambda_semantic": 0.05,
            "lambda_semantic_retrieval": 0.2,
            "lambda_prototype_anchor": 0.3,
            "prototype_freeze_epochs": 1,
        }
        apply_adaptive_semantic_guidance_policy(constant)
        self.assertFalse(constant["semantic_prototype_enabled"])
        self.assertEqual(constant["lambda_semantic"], 0.0)
        self.assertEqual(constant["lambda_semantic_retrieval"], 0.0)
        self.assertEqual(constant["lambda_prototype_anchor"], 0.0)
        self.assertEqual(constant["prototype_freeze_epochs"], 0)

    def test_rectangular_accuracy_penalizes_extra_clusters(self):
        truth = np.array([0, 0, 1, 1])
        predictions = np.array([0, 1, 2, 2])
        self.assertAlmostEqual(rectangular_hungarian_accuracy(truth, predictions), 0.75)
        self.assertIn("ari", clustering_metrics(truth, predictions))

    def test_spherical_kmeans_finds_separated_directions(self):
        features = np.array(
            [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]], dtype=float
        )
        model = fit_spherical_kmeans(features, 2, seed=0, n_init=3)
        labels = model.predict(features)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_mixed_labels_have_ten_coarse_and_fifty_fine_targets(self):
        fine = np.repeat(np.arange(100), 2)
        coarse = np.repeat(np.arange(20), 10)
        mixed = build_mixed_labels(fine, coarse, tuple(range(10)))
        self.assertEqual(np.unique(mixed).size, 60)
        self.assertEqual(np.unique(mixed[coarse < 10]).size, 10)
        self.assertEqual(np.unique(mixed[coarse >= 10]).size, 50)

    def test_balanced_partitions_use_each_superclass_equally(self):
        partitions = balanced_coarse_superclass_sets(num_pairs=3, seed=4)
        counts = np.zeros(20, dtype=int)
        for partition in partitions:
            counts[list(partition)] += 1
        self.assertTrue(np.all(counts == 3))

    def test_fragmentation_and_merge_metrics(self):
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        predictions = np.array([0, 0, 1, 1, 2, 2, 2, 2])
        self.assertEqual(fragmentation_at_coverage(labels, predictions), 1.5)
        self.assertEqual(fine_merge_rate(labels, predictions), 0.0)


if __name__ == "__main__":
    unittest.main()
