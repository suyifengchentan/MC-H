from __future__ import annotations

import unittest

import torch

from hypersphere_dpp.model import FeatureAdapter, PrototypeHead


class ModelComponentsTest(unittest.TestCase):
    def test_feature_adapter_outputs_unit_norm(self):
        adapter = FeatureAdapter(feature_dim=768, hidden_dim=1024, dropout=0.1)
        teacher_features = torch.nn.functional.normalize(torch.randn(4, 768), dim=-1)
        student_features = adapter(teacher_features)
        norms = student_features.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_feature_adapter_uses_two_smaller_residual_blocks(self):
        adapter = FeatureAdapter(feature_dim=768, hidden_dim=1024, dropout=0.1)
        self.assertEqual(adapter.block_hidden_dim, 512)
        self.assertEqual(adapter.fc1.weight.shape, (512, 768))
        self.assertEqual(adapter.fc2.weight.shape, (768, 512))
        self.assertEqual(adapter.refine_fc1.weight.shape, (512, 768))
        self.assertEqual(adapter.refine_fc2.weight.shape, (768, 512))
        self.assertEqual(tuple(adapter.main_scale.shape), ())
        self.assertEqual(tuple(adapter.refine_scale.shape), ())

    def test_feature_adapter_rejects_single_block_checkpoint(self):
        adapter = FeatureAdapter(feature_dim=512, hidden_dim=768, dropout=0.1)
        legacy_state_dict = {
            "norm.weight": torch.ones(512),
            "norm.bias": torch.zeros(512),
            "fc1.weight": torch.randn(768, 512),
            "fc1.bias": torch.randn(768),
            "fc2.weight": torch.randn(512, 768),
            "fc2.bias": torch.randn(512),
        }
        with self.assertRaisesRegex(RuntimeError, "two-block residual adapter"):
            adapter.load_trainable_state_dict(legacy_state_dict)

    def test_prototypes_are_normalized_after_manual_call(self):
        head = PrototypeHead(num_prototypes=5, feature_dim=768)
        head.normalize_()
        norms = head.prototypes.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
