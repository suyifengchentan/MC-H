# datasets/oxford_flowers_dataset.py (数字排序修正版)

import os
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
from typing import List, Dict, Tuple


class OxfordFlowersDataset(datasets.ImageFolder):
    """
    自定义的 Oxford-102 Flowers 数据集类。

    关键修正：重写 (override) find_classes 方法，
    使其对 "1", "10", "2" 这样的文件夹名进行 *数字排序*，
    而不是 *字母排序*。
    """

    def __init__(self, root, train=True, transform=None, target_transform=None):

        split_dir = 'train' if train else 'valid'
        path = os.path.join(root, split_dir)

        print(f"[{self.__class__.__name__}] 正在从以下路径加载图像: {path}")

        # 调用父类 (ImageFolder) 的构造函数
        # ImageFolder.__init__ 会自动调用我们 *重写后* 的 find_classes 方法
        super(OxfordFlowersDataset, self).__init__(
            root=path,
            loader=default_loader,
            transform=transform,
            target_transform=target_transform
        )

    def __getitem__(self, index):
        """
        重载 __getitem__ 方法，使其返回 (sample, target, index)。
        """
        sample, target = super(OxfordFlowersDataset, self).__getitem__(index)
        return sample, target, index

    # vvvvvvvvvvvv 这是最关键的新增代码 vvvvvvvvvvvv
    @staticmethod
    def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        重写此方法以实现数字排序。
        """
        # 1. 找到所有是目录的子文件夹
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]

        # 2. 关键：不使用 sorted(classes)，而是使用 sorted(classes, key=int)
        #    这会将 ["1", "10", "2"] 正确地排序为 ["1", "2", "10"]
        try:
            classes.sort(key=int)
            print(f"    - [find_classes] 成功按 *数字* 顺序找到 {len(classes)} 个类。")
        except ValueError:
            # 如果文件夹名不是纯数字，则退回到字母排序
            print(f"    - [find_classes] 警告：文件夹名不是纯数字，退回到 *字母* 排序。")
            classes.sort()

        # 3. 创建 class_to_idx 映射
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^