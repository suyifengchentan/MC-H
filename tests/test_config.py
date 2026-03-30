from __future__ import annotations

import tempfile
import textwrap
import unittest
import warnings

from hypersphere_dpp.config import load_experiment_config


class ConfigLoaderTest(unittest.TestCase):
    def test_unknown_fields_emit_warning(self):
        config_text = textwrap.dedent(
            """
            dataset: "CUB_200_2011"
            dataset_dir: "/tmp/cub"
            class_num: 200
            unsupported_future_flag: true
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_text)
            config_path = handle.name

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = load_experiment_config(config_path)

        self.assertEqual(config.dataset, "CUB_200_2011")
        self.assertTrue(
            any(
                "Unsupported config field 'unsupported_future_flag'"
                in str(w.message)
                for w in caught
            )
        )

    def test_semantic_and_class_aware_fields_are_supported(self):
        config_text = textwrap.dedent(
            """
            dataset: "Image200"
            dataset_dir: "/tmp/image200"
            class_num: 200
            feature_dim: 512
            backbone_name: "ViT-B-16"
            semantic_prototype_enabled: true
            semantic_prompt_style: "auto"
            semantic_prototype_aggregation: "mean"
            lambda_semantic: 0.2
            semantic_temperature: 0.2
            class_aware_dpp: true
            class_aware_dpp_min_mass: 2.0
            official_eval_split: "val"
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_text)
            config_path = handle.name

        config = load_experiment_config(config_path)

        self.assertTrue(config.semantic_prototype_enabled)
        self.assertEqual(config.semantic_prompt_style, "auto")
        self.assertEqual(config.semantic_prototype_aggregation, "mean")
        self.assertEqual(config.lambda_semantic, 0.2)
        self.assertEqual(config.semantic_temperature, 0.2)
        self.assertTrue(config.class_aware_dpp)
        self.assertEqual(config.class_aware_dpp_min_mass, 2.0)

    def test_image100_is_supported(self):
        config_text = textwrap.dedent(
            """
            dataset: "Image100"
            dataset_dir: "/tmp/image100"
            class_num: 100
            feature_dim: 512
            backbone_name: "ViT-B-16"
            pretrained: "laion2b_s34b_b88k"
            lr_scheduler: "cosine"
            min_learning_rate: 0.00001
            subspace_every_n_steps: 2
            subspace_pca_q_offset: 2
            subspace_topk_clusters: 48
            dpp_every_n_steps: 4
            official_eval_split: "val"
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_text)
            config_path = handle.name

        config = load_experiment_config(config_path)

        self.assertEqual(config.dataset, "Image100")
        self.assertEqual(config.official_eval_split, "val")
        self.assertEqual(config.feature_dim, 512)
        self.assertEqual(config.backbone_name, "ViT-B-16")
        self.assertEqual(config.pretrained, "laion2b_s34b_b88k")
        self.assertEqual(config.lr_scheduler, "cosine")
        self.assertEqual(config.min_learning_rate, 1e-5)
        self.assertEqual(config.subspace_every_n_steps, 2)
        self.assertEqual(config.subspace_pca_q_offset, 2)
        self.assertEqual(config.subspace_topk_clusters, 48)
        self.assertEqual(config.dpp_every_n_steps, 4)

    def test_oxford_flowers_valid_split_is_supported(self):
        config_text = textwrap.dedent(
            """
            dataset: "Oxford_Flowers"
            dataset_dir: "/tmp/oxford-flowers"
            class_num: 102
            official_eval_split: "valid"
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_text)
            config_path = handle.name

        config = load_experiment_config(config_path)

        self.assertEqual(config.dataset, "Oxford_Flowers")
        self.assertEqual(config.class_num, 102)
        self.assertEqual(config.official_eval_split, "valid")

    def test_min_learning_rate_cannot_exceed_learning_rate(self):
        config_text = textwrap.dedent(
            """
            dataset: "Image50"
            dataset_dir: "/tmp/image50"
            class_num: 50
            feature_dim: 512
            backbone_name: "ViT-B-16"
            learning_rate: 0.0001
            min_learning_rate: 0.0002
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_text)
            config_path = handle.name

        with self.assertRaisesRegex(
            ValueError, "min_learning_rate must be less than or equal"
        ):
            load_experiment_config(config_path)


if __name__ == "__main__":
    unittest.main()
