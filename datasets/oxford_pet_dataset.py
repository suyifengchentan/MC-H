# datasets/oxford_pet_dataset.py

import os
from torch.utils.data import Dataset
from PIL import Image


def _resolve_images_root(root: str) -> str:
    candidates = (
        os.path.join(root, "images", "images"),
        os.path.join(root, "images"),
    )
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Oxford_Pet images directory not found below: {root}")


def _resolve_annotation_path(root: str, split: str) -> str:
    annotation_file = f"{split}.txt"
    candidates = (
        os.path.join(root, "annotations", annotation_file),
        os.path.join(root, "annotations", "annotations", annotation_file),
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Oxford_Pet annotation file for split {split!r} not found below: {root}"
    )


class OxfordPetDataset(Dataset):
    """
    自定义的 Oxford-IIIT Pet 数据集类。
    它通过解析 'trainval.txt' 或 'test.txt' 来加载数据。
    """

    def __init__(self, root, split='trainval', transform=None):
        """
        构造函数

        参数:
            root (string): Oxford-IIIT Pet 数据集的根目录。
                           (e.g., '/.../data/oxford_pet/')
            split (string): 'trainval' 或 'test'
            transform (callable, optional): 应用于样本的图像变换。
        """
        self.root = root
        self.transform = transform
        self.images_base_path = _resolve_images_root(self.root)

        annotation_path = _resolve_annotation_path(self.root, split)

        print(f"[{self.__class__.__name__}] 正在从以下路径加载 {split} 数据: {annotation_path}")

        self.image_files = []
        self.labels = []

        # 2. 读取 .txt 文件
        with open(annotation_path, 'r') as f:
            for line in f:
                # 跳过注释行
                if line.startswith('#'):
                    continue

                parts = line.strip().split()
                if not parts:
                    continue

                # 示例: Abyssinian_100 1 1 1
                image_name = parts[0] + ".jpg"  # 添加 .jpg 扩展名
                class_id = int(parts[1])  # CLASS-ID 是 1-37

                self.image_files.append(image_name)
                self.labels.append(class_id - 1)  # 转换为 0-36 索引

    def __len__(self):
        """返回数据集中的样本数量。"""
        return len(self.image_files)

    def __getitem__(self, idx):
        """根据索引 idx 获取一个样本。"""
        image_name = self.image_files[idx]
        label = self.labels[idx]

        image_path = os.path.join(self.images_base_path, image_name)

        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"错误: 找不到图像文件 {image_path}。")
            # 在出错时返回一个黑色图像
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            label = 0  # 标签也设为0

        if self.transform:
            image = self.transform(image)

        # 返回我们框架所需的三个值
        return image, label, idx
