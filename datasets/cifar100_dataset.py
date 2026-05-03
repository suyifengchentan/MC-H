import pickle
from pathlib import Path

import torchvision.datasets as datasets


class CIFAR100Dataset(datasets.CIFAR100):
    """Torchvision CIFAR100 wrapper that also returns sample indices."""

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        label_mode="fine",
    ):
        super(CIFAR100Dataset, self).__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        normalized_label_mode = str(label_mode).strip().lower()
        if normalized_label_mode not in {"fine", "coarse"}:
            raise ValueError(
                "label_mode must be either 'fine' or 'coarse', "
                f"got {label_mode!r}."
            )
        self.label_mode = normalized_label_mode
        self.fine_classes = list(self.classes)
        self.coarse_classes = self._load_coarse_label_names()
        if self.label_mode == "coarse":
            self.targets = self._load_split_coarse_targets()
            self.classes = list(self.coarse_classes)
            self.class_to_idx = {
                class_name: class_index
                for class_index, class_name in enumerate(self.classes)
            }

    def __getitem__(self, index):
        sample, target = super(CIFAR100Dataset, self).__getitem__(index)
        return sample, target, index

    def _dataset_file_path(self, filename: str) -> Path:
        return Path(self.root) / self.base_folder / filename

    def _load_pickle(self, filename: str):
        with self._dataset_file_path(filename).open("rb") as handle:
            return pickle.load(handle, encoding="latin1")

    def _load_coarse_label_names(self) -> list[str]:
        meta = self._load_pickle(self.meta["filename"])
        return [str(name) for name in meta["coarse_label_names"]]

    def _load_split_coarse_targets(self) -> list[int]:
        split_filename = self.train_list[0][0] if self.train else self.test_list[0][0]
        split = self._load_pickle(split_filename)
        return [int(label) for label in split["coarse_labels"]]
