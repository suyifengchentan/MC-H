# stanford_dogs_dataset.py (修正版)

import os
from torch.utils.data import Dataset
from PIL import Image
import scipy.io


def _resolve_images_folder(root):
    direct_path = os.path.join(root, 'Images')
    nested_path = os.path.join(root, 'images', 'Images')

    if os.path.isdir(direct_path):
        return direct_path
    if os.path.isdir(nested_path):
        return nested_path

    raise FileNotFoundError(f"在 {root} 下找不到 'Images' 或 'images/Images' 目录。")


class StanfordDogsDataset(Dataset):
    """
    自定义的 Stanford Dogs 数据集类。
    通过解析官方的 .mat 文件来加载训练/测试划分和标签。
    """

    def __init__(self, root, train=True, transform=None):
        """
        构造函数

        参数:
            root (string): Stanford Dogs 数据集的根目录。
            train (bool, optional): 如果为 True, 创建训练集; 否则创建测试集。
            transform (callable, optional): 应用于样本的图像变换。
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.images_folder = _resolve_images_folder(self.root)

        split_file = 'train_list.mat' if self.train else 'test_list.mat'

        # vvvvvvvvvvvv 这是关键的修正 vvvvvvvvvvvvvv
        # 在路径中加入了 'lists' 子文件夹
        split_path = os.path.join(self.root, 'lists', split_file)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # 加载 .mat 文件
        split_mat = scipy.io.loadmat(split_path)

        # --- 解析 .mat 文件内容 ---
        self.image_files = [f[0][0] for f in split_mat['file_list']]
        self.labels = [int(l[0]) - 1 for l in split_mat['labels']]
        self.data = list(zip(self.image_files, self.labels))

    def __len__(self):
        """返回数据集中的样本数量。"""
        return len(self.data)

    def __getitem__(self, idx):
        """根据索引 idx 获取一个样本。"""
        image_path, label = self.data[idx]
        full_image_path = os.path.join(self.images_folder, image_path)

        image = Image.open(full_image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 适配我们最新的训练脚本，返回三个值
        return image, label, idx