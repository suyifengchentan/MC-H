# datasets/cifar10_dataset.py

import torchvision.datasets as datasets


class CIFAR10Dataset(datasets.CIFAR10):
    """
    对 torchvision.datasets.CIFAR10 的一个简单包装器。

    它重载了 __getitem__ 方法，使其返回 (sample, target, index)。
    这与我们项目中所有其他的数据集类保持一致。
    """

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        print(f"[{self.__class__.__name__}] 正在加载: {'训练集' if train else '测试集'}")

        # 调用父类 (torchvision.datasets.CIFAR10) 的构造函数
        super(CIFAR10Dataset, self).__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download
        )

    def __getitem__(self, index):
        """
        重载 __getitem__ 方法。
        """
        # 调用父类的 __getitem__ 方法，它返回 (sample, target)
        sample, target = super(CIFAR10Dataset, self).__getitem__(index)

        # 返回我们所有脚本都期望的三个值
        return sample, target, index