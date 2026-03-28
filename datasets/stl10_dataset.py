import torchvision.datasets as datasets


class STL10Dataset(datasets.STL10):
    """Torchvision STL10 wrapper that also returns sample indices."""

    def __init__(
        self, root, split="train", transform=None, target_transform=None, download=False
    ):
        super(STL10Dataset, self).__init__(
            root=root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __getitem__(self, index):
        sample, target = super(STL10Dataset, self).__getitem__(index)
        return sample, target, index
