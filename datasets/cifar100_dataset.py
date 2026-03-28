import torchvision.datasets as datasets


class CIFAR100Dataset(datasets.CIFAR100):
    """Torchvision CIFAR100 wrapper that also returns sample indices."""

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        super(CIFAR100Dataset, self).__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __getitem__(self, index):
        sample, target = super(CIFAR100Dataset, self).__getitem__(index)
        return sample, target, index
