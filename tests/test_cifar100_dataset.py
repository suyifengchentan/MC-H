from __future__ import annotations

import unittest
from unittest import mock

import torchvision.datasets as tv_datasets

from datasets.cifar100_dataset import CIFAR100Dataset


class CIFAR100DatasetTest(unittest.TestCase):
    def _mock_super_init(self, instance, root, train, transform, target_transform, download):
        instance.root = root
        instance.train = train
        instance.transform = transform
        instance.target_transform = target_transform
        instance.base_folder = "cifar-100-python"
        instance.train_list = [["train", "unused"]]
        instance.test_list = [["test", "unused"]]
        instance.meta = {"filename": "meta"}
        instance.targets = [10, 11, 12]
        instance.classes = ["fine_a", "fine_b", "fine_c"]
        instance.class_to_idx = {name: idx for idx, name in enumerate(instance.classes)}

    def test_coarse_mode_replaces_targets_and_classes(self):
        with mock.patch.object(
            tv_datasets.CIFAR100,
            "__init__",
            autospec=True,
            side_effect=self._mock_super_init,
        ):
            with mock.patch.object(
                CIFAR100Dataset,
                "_load_coarse_label_names",
                return_value=["super_a", "super_b"],
            ):
                with mock.patch.object(
                    CIFAR100Dataset,
                    "_load_split_coarse_targets",
                    return_value=[1, 0, 1],
                ):
                    dataset = CIFAR100Dataset("/tmp/cifar100", train=True, label_mode="coarse")

        self.assertEqual(dataset.label_mode, "coarse")
        self.assertEqual(dataset.targets, [1, 0, 1])
        self.assertEqual(dataset.classes, ["super_a", "super_b"])
        self.assertEqual(dataset.fine_classes, ["fine_a", "fine_b", "fine_c"])
        self.assertEqual(dataset.coarse_classes, ["super_a", "super_b"])
        self.assertEqual(dataset.class_to_idx, {"super_a": 0, "super_b": 1})

    def test_fine_mode_keeps_original_targets(self):
        with mock.patch.object(
            tv_datasets.CIFAR100,
            "__init__",
            autospec=True,
            side_effect=self._mock_super_init,
        ):
            with mock.patch.object(
                CIFAR100Dataset,
                "_load_coarse_label_names",
                return_value=["super_a", "super_b"],
            ):
                dataset = CIFAR100Dataset("/tmp/cifar100", train=False, label_mode="fine")

        self.assertEqual(dataset.label_mode, "fine")
        self.assertEqual(dataset.targets, [10, 11, 12])
        self.assertEqual(dataset.classes, ["fine_a", "fine_b", "fine_c"])
        self.assertEqual(dataset.coarse_classes, ["super_a", "super_b"])


if __name__ == "__main__":
    unittest.main()
