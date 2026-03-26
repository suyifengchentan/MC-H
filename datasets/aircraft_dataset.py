# datasets/aircraft_dataset.py

import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class AircraftDataset(Dataset):
    """
    自定义的 FGVC-Aircraft 数据集类。
    通过解析官方的 .csv 文件来加载图像、类名和标签。
    """

    def __init__(self, root, split='train', transform=None):
        """
        构造函数

        参数:
            root (string): FGVC-Aircraft 数据集的根目录。
                           (e.g., '/.../data/FGVC-Aircraft/')
            split (string): 'train', 'val', 或 'test'
            transform (callable, optional): 应用于样本的图像变换。
        """
        self.root = root
        self.transform = transform

        # 1. 定义图像文件的根路径
        #    (根据您提供的路径 .../FGVC-Aircraft/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images/)
        self.images_base_path = os.path.join(
            self.root, 'fgvc-aircraft-2013b', 'fgvc-aircraft-2013b', 'data', 'images'
        )

        # 2. 定义要加载的CSV文件
        csv_file = f"{split}.csv"
        csv_path = os.path.join(self.root, csv_file)

        print(f"[{self.__class__.__name__}] 正在从以下路径加载 {split} 数据: {csv_path}")

        # 3. 使用 pandas 加载CSV
        self.data_frame = pd.read_csv(csv_path)

        # 4. 提取信息
        #    假设 'filename' 列包含完整的 .jpg 文件名
        self.image_files = self.data_frame['filename'].values
        self.labels = self.data_frame['Labels'].values  # 0-99

    def __len__(self):
        """返回数据集中的样本数量。"""
        return len(self.image_files)

    def __getitem__(self, idx):
        """根据索引 idx 获取一个样本。"""
        # 1. 获取文件名和标签
        image_name = self.image_files[idx]
        label = self.labels[idx]

        # 2. 构建完整图像路径
        image_path = os.path.join(self.images_base_path, image_name)

        # 3. 加载图像
        image = Image.open(image_path).convert('RGB')

        # 4. 应用变换
        if self.transform:
            image = self.transform(image)

        # 返回我们框架所需的三个值
        return image, label, idx