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

    def test_prototypes_are_normalized_after_manual_call(self):
        head = PrototypeHead(num_prototypes=5, feature_dim=768)
        head.normalize_()
        norms = head.prototypes.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


if __name__ == "__main__":
    unittest.main()

