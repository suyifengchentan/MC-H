from __future__ import annotations

import tempfile
import textwrap
import unittest
import warnings
from pathlib import Path

from hypersphere_dpp.config import load_experiment_config


class ConfigLoaderTest(unittest.TestCase):
    def test_home_relative_paths_are_expanded(self):
        config_text = textwrap.dedent(
            """
            dataset: "CIFAR10"
            dataset_dir: "~/hypersphere-data/CIFAR10"
            pretrained_checkpoint: "~/clip-pretrained/open_clip_pytorch_model.bin"
            class_num: 10
            feature_dim: 512
            backbone_name: "ViT-B-16"
            official_eval_split: "test"
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_text)
            config_path = handle.name

        config = load_experiment_config(config_path)

        self.assertEqual(
            config.dataset_dir,
            str(Path.home() / "hypersphere-data" / "CIFAR10"),
        )
        self.assertEqual(
            config.pretrained_checkpoint,
            str(Path.home() / "clip-pretrained" / "open_clip_pytorch_model.bin"),
        )

    def test_dataset_dir_defaults_below_current_home(self):
        config_text = textwrap.dedent(
            """
            dataset: "CIFAR10"
            class_num: 10
            feature_dim: 512
            backbone_name: "ViT-B-16"
            official_eval_split: "test"
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_text)
            config_path = handle.name

        config = load_experiment_config(config_path)

        self.assertEqual(
            config.dataset_dir,
            str(Path.home() / "hypersphere-data" / "CIFAR10"),
        )

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

    def test_imagenet1k_is_supported(self):
        config_text = textwrap.dedent(
            """
            dataset: "ImageNet1K"
            dataset_dir: "/tmp/imagenet-1k"
            class_num: 1000
            feature_dim: 512
            backbone_name: "ViT-B-16"
            transform_profile: "imagenet1k_two_view"
            official_eval_split: "val"
            imagenet1k_cached_teacher_views_enabled: true
            imagenet1k_cached_teacher_num_views: 6
            imagenet1k_cached_teacher_dtype: "float16"
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_text)
            config_path = handle.name

        config = load_experiment_config(config_path)

        self.assertEqual(config.dataset, "ImageNet1K")
        self.assertEqual(config.class_num, 1000)
        self.assertEqual(config.transform_profile, "imagenet1k_two_view")
        self.assertEqual(config.official_eval_split, "val")
        self.assertTrue(config.uses_imagenet1k_cached_teacher_views)
        self.assertEqual(config.imagenet1k_cached_teacher_num_views, 6)
        self.assertEqual(config.imagenet1k_cached_teacher_dtype, "float16")

    def test_step_profile_fields_are_supported(self):
        config_text = textwrap.dedent(
            """
            dataset: "ImageNet1K"
            dataset_dir: "/tmp/imagenet-1k"
            class_num: 1000
            feature_dim: 512
            backbone_name: "ViT-B-16"
            step_profile_enabled: true
            step_profile_log_interval: 3
            step_profile_cuda_sync: false
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_text)
            config_path = handle.name

        config = load_experiment_config(config_path)

        self.assertTrue(config.step_profile_enabled)
        self.assertEqual(config.step_profile_log_interval, 3)
        self.assertFalse(config.step_profile_cuda_sync)

    def test_imagenet1k_single_and_parallel_train_configs_are_split(self):
        repo_root = Path(__file__).resolve().parents[1]
        single_config = load_experiment_config(
            repo_root
            / "config/train/coarse_grained/imagenet1k_vitb16_hypersphere_dpp_single_config.yaml"
        )
        parallel_config = load_experiment_config(
            repo_root
            / "config/train/coarse_grained/imagenet1k_vitb16_hypersphere_dpp_parallel_config.yaml"
        )

        self.assertEqual(single_config.dataset, "ImageNet1K")
        self.assertEqual(parallel_config.dataset, "ImageNet1K")
        self.assertEqual(single_config.batch_size, 2048)
        self.assertGreaterEqual(parallel_config.batch_size, single_config.batch_size)
        self.assertEqual(single_config.subspace_min_mass, 4.0)
        self.assertEqual(parallel_config.subspace_min_mass, 4.0)
        self.assertEqual(single_config.class_aware_dpp_min_mass, 2.0)
        self.assertEqual(parallel_config.class_aware_dpp_min_mass, 2.0)
        self.assertTrue(single_config.uses_imagenet1k_cached_teacher_views)
        self.assertTrue(parallel_config.uses_imagenet1k_cached_teacher_views)
        self.assertEqual(single_config.imagenet1k_cached_teacher_num_views, 6)
        self.assertEqual(parallel_config.imagenet1k_cached_teacher_num_views, 6)
        self.assertNotEqual(single_config.model_path, parallel_config.model_path)

    def test_imagenet1k_cached_teacher_views_are_rejected_for_other_datasets(self):
        config_text = textwrap.dedent(
            """
            dataset: "Image100"
            dataset_dir: "/tmp/image100"
            class_num: 100
            feature_dim: 512
            backbone_name: "ViT-B-16"
            imagenet1k_cached_teacher_views_enabled: true
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_text)
            config_path = handle.name

        with self.assertRaisesRegex(
            ValueError, "only supported when dataset='ImageNet1K'"
        ):
            load_experiment_config(config_path)

    def test_tiny_imagenet200_is_supported(self):
        config_text = textwrap.dedent(
            """
            dataset: "TinyImageNet200"
            dataset_dir: "/tmp/tiny-imagenet-200"
            class_num: 200
            feature_dim: 512
            backbone_name: "ViT-B-16"
            transform_profile: "tiny_imagenet200_two_view"
            official_eval_split: "val"
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_text)
            config_path = handle.name

        config = load_experiment_config(config_path)

        self.assertEqual(config.dataset, "TinyImageNet200")
        self.assertEqual(config.class_num, 200)
        self.assertEqual(config.transform_profile, "tiny_imagenet200_two_view")
        self.assertEqual(config.official_eval_split, "val")

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

    def test_cifar100_coarse_mode_is_supported(self):
        config_text = textwrap.dedent(
            """
            dataset: "CIFAR100"
            dataset_dir: "/tmp/cifar100"
            cifar100_label_mode: "coarse"
            class_num: 20
            feature_dim: 512
            backbone_name: "ViT-B-16"
            official_eval_split: "test"
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_text)
            config_path = handle.name

        config = load_experiment_config(config_path)

        self.assertEqual(config.dataset, "CIFAR100")
        self.assertEqual(config.cifar100_label_mode, "coarse")
        self.assertEqual(config.class_num, 20)

    def test_cifar100_coarse_mode_requires_20_classes(self):
        config_text = textwrap.dedent(
            """
            dataset: "CIFAR100"
            dataset_dir: "/tmp/cifar100"
            cifar100_label_mode: "coarse"
            class_num: 100
            feature_dim: 512
            backbone_name: "ViT-B-16"
            official_eval_split: "test"
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_text)
            config_path = handle.name

        with self.assertRaisesRegex(
            ValueError, "class_num for CIFAR100 must match cifar100_label_mode"
        ):
            load_experiment_config(config_path)

    def test_cifar100_noncanonical_class_num_requires_explicit_opt_in(self):
        config_text = textwrap.dedent(
            """
            dataset: "CIFAR100"
            dataset_dir: "/tmp/cifar100"
            cifar100_label_mode: "fine"
            cifar100_allow_noncanonical_class_num: true
            class_num: 60
            feature_dim: 512
            backbone_name: "ViT-B-16"
            official_eval_split: "test"
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_text)
            config_path = handle.name

        config = load_experiment_config(config_path)

        self.assertEqual(config.class_num, 60)
        self.assertTrue(config.cifar100_allow_noncanonical_class_num)

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

    def test_step_profile_log_interval_must_be_positive(self):
        config_text = textwrap.dedent(
            """
            dataset: "Image50"
            dataset_dir: "/tmp/image50"
            class_num: 50
            feature_dim: 512
            backbone_name: "ViT-B-16"
            step_profile_enabled: true
            step_profile_log_interval: 0
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(config_text)
            config_path = handle.name

        with self.assertRaisesRegex(
            ValueError, "step_profile_log_interval must be a positive integer"
        ):
            load_experiment_config(config_path)


if __name__ == "__main__":
    unittest.main()
