# cub_dataset.py (修正版)

import os
from torch.utils.data import Dataset
from PIL import Image


class CUB200Dataset(Dataset):
    """
    自定义的 CUB-200-2011 数据集类。
    它通过解析官方的 .txt 文件来加载数据和标签，并根据 train/test split 文件来划分数据集。
    """

    def __init__(self, root, train=True, transform=None):
        """
        构造函数

        参数:
            root (string): CUB_200_2011 数据集的根目录。
            train (bool, optional): 如果为 True, 创建训练集; 否则创建测试集。
            transform (callable, optional): 应用于样本的图像变换。
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.data = []  # 用来存储 (图片路径, 标签) 的列表

        self._load_metadata()

    def _load_metadata(self):
        """
        解析 .txt 元数据文件，并根据 train/test 标志筛选数据。
        """
        images_path = os.path.join(self.root, 'images.txt')
        labels_path = os.path.join(self.root, 'image_class_labels.txt')
        split_path = os.path.join(self.root, 'train_test_split.txt')

        id_to_path = {}
        with open(images_path, 'r') as f:
            for line in f:
                image_id, path = line.strip().split()
                id_to_path[image_id] = path

        id_to_label = {}
        with open(labels_path, 'r') as f:
            for line in f:
                image_id, label = line.strip().split()
                id_to_label[image_id] = int(label) - 1

        id_to_train_flag = {}
        with open(split_path, 'r') as f:
            for line in f:
                image_id, is_train = line.strip().split()
                id_to_train_flag[image_id] = (is_train == '1')

        for image_id in id_to_path.keys():
            is_train_sample = id_to_train_flag[image_id]

            if self.train == is_train_sample:
                image_path = id_to_path[image_id]
                label = id_to_label[image_id]
                self.data.append((image_path, label))

    def __len__(self):
        """
        返回数据集中的样本数量。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引 idx 获取一个样本。
        现在返回 (图像, 真实标签, 样本索引)。
        """
        image_path, label = self.data[idx]
        full_image_path = os.path.join(self.root, 'images', image_path)
        image = Image.open(full_image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # vvvvvvvvvvvv 这是关键的修正 vvvvvvvvvvvvvv
        # 增加了返回样本索引 idx
        return image, label, idx
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
