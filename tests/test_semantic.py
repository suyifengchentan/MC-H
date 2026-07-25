from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from hypersphere_dpp.semantic import (
    _encode_text_prompts_batched,
    _spherical_text_prototypes,
    load_class_texts,
)


class SemanticClassTextTest(unittest.TestCase):
    def test_spherical_text_prototypes_support_noncanonical_k(self):
        features = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]]
        )
        centers, assignments = _spherical_text_prototypes(features, 2)

        self.assertEqual(tuple(centers.shape), (2, 2))
        self.assertEqual(tuple(assignments.shape), (4,))
        self.assertTrue(torch.allclose(centers.norm(dim=1), torch.ones(2)))
        self.assertEqual(assignments[0], assignments[1])
        self.assertEqual(assignments[2], assignments[3])
        self.assertNotEqual(assignments[0], assignments[2])

    def test_spherical_text_prototypes_preserve_matching_k(self):
        features = torch.eye(3)
        centers, assignments = _spherical_text_prototypes(features, 3)

        self.assertTrue(torch.allclose(centers, features))
        self.assertTrue(torch.equal(assignments, torch.arange(3)))

    def test_image10_uses_label_config_class_texts(self):
        self.assertEqual(
            load_class_texts("Image10", "/unused"),
            [
                "king penguin",
                "Maltese dog",
                "snow leopard",
                "airliner",
                "airship",
                "container ship",
                "soccer ball",
                "sports car",
                "trailer truck",
                "orange",
            ],
        )

    def test_imagenet1k_can_load_class_texts_from_h5_label_info(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with (root / "label_info.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "idx_to_wnid": {"0": "n00000001", "1": "n00000002"},
                        "idx_to_class_name": {"0": "class one", "1": "class two"},
                    },
                    handle,
                )

            self.assertEqual(
                load_class_texts("ImageNet1K", str(root)),
                ["class one", "class two"],
            )

    def test_imagenet1k_class_texts_are_canonicalized_to_primary_label(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with (root / "label_info.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "idx_to_wnid": {"0": "n00000001", "1": "n00000002"},
                        "idx_to_class_name": {
                            "0": "great white shark, white shark, man-eater",
                            "1": "water_ouzel, dipper",
                        },
                    },
                    handle,
                )

            self.assertEqual(
                load_class_texts("ImageNet1K", str(root)),
                ["great white shark", "water ouzel"],
            )

    def test_encode_text_prompts_batched_matches_single_pass(self):
        class FakeTokenizer:
            def __call__(self, prompts):
                values = [
                    [float(prompt[1:]), float(prompt[1:]) + 0.5]
                    for prompt in prompts
                ]
                return torch.tensor(values, dtype=torch.float32)

        class FakeBackbone:
            def encode_text(self, tokenized):
                return tokenized + 1.0

        prompts = ["p0", "p1", "p2", "p3", "p4"]
        tokenizer = FakeTokenizer()
        backbone = FakeBackbone()

        expected = torch.nn.functional.normalize(
            backbone.encode_text(tokenizer(prompts)).float(), dim=-1
        )
        actual = _encode_text_prompts_batched(
            backbone_model=backbone,
            tokenizer=tokenizer,
            prompts=prompts,
            device=torch.device("cpu"),
            batch_size=2,
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_encode_text_prompts_batched_rejects_empty_prompt_lists(self):
        class FakeTokenizer:
            def __call__(self, prompts):
                return torch.tensor(prompts)

        class FakeBackbone:
            def encode_text(self, tokenized):
                return tokenized

        with self.assertRaises(ValueError):
            _encode_text_prompts_batched(
                backbone_model=FakeBackbone(),
                tokenizer=FakeTokenizer(),
                prompts=[],
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
