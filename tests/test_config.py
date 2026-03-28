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
            semantic_prototype_enabled: true
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
                "Unsupported config field 'semantic_prototype_enabled'"
                in str(w.message)
                for w in caught
            )
        )

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
