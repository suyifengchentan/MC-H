from __future__ import annotations

import unittest
from collections import Counter

import torch
import torch.nn.functional as F

from hypersphere_dpp.data import split_train_indices
from hypersphere_dpp.losses import (
    _dpp_logdet_from_phi,
    dpp_diversity_loss,
    spectral_cluster_loss,
)
from hypersphere_dpp.subspace import (
    ClusterFeatureQueue,
    spectral_anchor_subspace_loss,
    subspace_regularization_loss,
)


class DataAndLossesTest(unittest.TestCase):
    def test_split_reproducibility(self):
        train_a, val_a = split_train_indices(100, 0.2, 42)
        train_b, val_b = split_train_indices(100, 0.2, 42)

        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(len(train_a), 80)
        self.assertEqual(len(val_a), 20)
        self.assertEqual(set(train_a).intersection(val_a), set())

    def test_stratified_split_reproducibility(self):
        labels = [0] * 10 + [1] * 10 + [2] * 10
        train_a, val_a = split_train_indices(30, 0.2, 42, labels=labels)
        train_b, val_b = split_train_indices(30, 0.2, 42, labels=labels)

        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(set(train_a).intersection(val_a), set())

    def test_stratified_split_keeps_all_classes_in_val(self):
        labels = [0] * 30 + [1] * 30 + [2] * 30 + [3] * 30
        train_indices, val_indices = split_train_indices(120, 0.1, 42, labels=labels)

        val_counts = Counter(labels[index] for index in val_indices)
        train_counts = Counter(labels[index] for index in train_indices)

        self.assertEqual(len(val_counts), 4)
        self.assertEqual(len(train_counts), 4)
        self.assertTrue(all(count >= 1 for count in val_counts.values()))
        self.assertTrue(all(count >= 1 for count in train_counts.values()))

    def test_dpp_loss_is_finite(self):
        features = torch.nn.functional.normalize(torch.randn(8, 768), dim=-1)
        assignments = torch.softmax(torch.randn(8, 4), dim=-1)
        loss, rho, relative_quality = dpp_diversity_loss(features, assignments)

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(rho))
        self.assertEqual(relative_quality.shape, (8,))

    def test_dpp_small_matrix_logdet_matches_original_formula(self):
        torch.manual_seed(0)
        features = F.normalize(torch.randn(16, 6), dim=-1)
        assignments = torch.softmax(torch.randn(16, 5), dim=-1)
        quality = assignments.max(dim=1).values.clamp_min(1e-6)
        centered_log_quality = quality.log() - quality.log().mean()
        relative_quality = centered_log_quality.exp()
        phi = features * relative_quality.unsqueeze(1)

        kernel = phi @ phi.t()
        matrix = (1.0 + 1e-5) * torch.eye(kernel.size(0), dtype=kernel.dtype) + kernel
        _, original_logabsdet = torch.linalg.slogdet(matrix)
        optimized_logabsdet = _dpp_logdet_from_phi(phi, 1e-5)

        self.assertTrue(torch.allclose(optimized_logabsdet, original_logabsdet, atol=1e-5, rtol=1e-5))

    def test_spectral_cluster_loss_is_finite(self):
        logits_a = torch.randn(8, 5)
        logits_b = torch.randn(8, 5)
        loss, assign_a, assign_b = spectral_cluster_loss(
            logits_a=logits_a,
            logits_b=logits_b,
            prediction_temperature=0.3,
            sharpen_temperature=0.1,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(assign_a.shape, logits_a.shape)
        self.assertEqual(assign_b.shape, logits_b.shape)

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

    def test_queue_state_roundtrip_keeps_values(self):
        queue = ClusterFeatureQueue(num_clusters=2, feature_dim=3, queue_size=4)
        teacher_features = torch.randn(4, 3)
        assignments = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        queue.enqueue(teacher_features, assignments)

        restored = ClusterFeatureQueue(num_clusters=2, feature_dim=3, queue_size=4)
        restored.load_state_dict(queue.state_dict())

        original_features, original_weights = queue.get(1, torch.device("cpu"))
        restored_features, restored_weights = restored.get(1, torch.device("cpu"))
        self.assertTrue(torch.allclose(original_features, restored_features))
        self.assertTrue(torch.allclose(original_weights, restored_weights))

    def test_subspace_topk_clusters_limits_active_clusters(self):
        queue = ClusterFeatureQueue(num_clusters=4, feature_dim=6, queue_size=8)
        teacher_features = torch.randn(12, 6)
        student_features = torch.randn(12, 6)
        assignments = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        queue.enqueue(teacher_features, assignments)

        loss, stats = subspace_regularization_loss(
            student_features=student_features,
            teacher_features=teacher_features,
            assignments=assignments,
            queue=queue,
            rank=2,
            min_mass=2.0,
            pca_q_offset=0,
            topk_clusters=2,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats.active_clusters, 2)

    def test_spectral_anchor_subspace_loss_is_finite(self):
        student_features = torch.randn(10, 6)
        assignments = torch.softmax(torch.randn(10, 4), dim=-1)
        prototypes = torch.nn.functional.normalize(torch.randn(4, 6), dim=-1)

        loss, stats = spectral_anchor_subspace_loss(
            student_features=student_features,
            assignments=assignments,
            anchor_prototypes=prototypes,
            min_mass=1.0,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(stats.active_clusters, 0)


if __name__ == "__main__":
    unittest.main()
