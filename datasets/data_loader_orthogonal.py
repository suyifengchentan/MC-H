# data_loader_orthogonal.py

import os
from torch.utils import data
import torchvision

from .cub_dataset import CUB200Dataset
from .stanford_dogs_dataset import StanfordDogsDataset
from .oxford_flowers_dataset import OxfordFlowersDataset
from .cifar10_dataset import CIFAR10Dataset
from .aircraft_dataset import AircraftDataset
from .oxford_pet_dataset import OxfordPetDataset

def _get_cub_dataset_orthogonal(root, transform):
    """
    为正交实验加载CUB数据集。
    合并训练集和测试集用于无监督训练。
    """
    train_dataset = CUB200Dataset(root=root, train=True, transform=transform)
    test_dataset = CUB200Dataset(root=root, train=False, transform=transform)
    dataset = data.ConcatDataset([train_dataset, test_dataset])
    class_num = 200
    return dataset, class_num


def _get_stanford_dogs_dataset(root, transform):
    """
    加载并配置Stanford Dogs数据集的函数。
    合并训练集和测试集用于无监督训练。
    """
    train_dataset = StanfordDogsDataset(root=root, train=True, transform=transform)
    test_dataset = StanfordDogsDataset(root=root, train=False, transform=transform)
    dataset = data.ConcatDataset([train_dataset, test_dataset])
    class_num = 120 # Stanford Dogs 数据集有120个类别
    return dataset, class_num

def _get_oxford_flowers_dataset(root, transform):
    """
    加载并配置 Oxford-102 Flowers 数据集。
    我们将合并 'train' 和 'valid' 文件夹作为我们的无监督训练集。
    """
    # train=True 加载 'train' 文件夹
    train_dataset = OxfordFlowersDataset(root=root, train=True, transform=transform)

    # train=False 加载 'valid' 文件夹
    valid_dataset = OxfordFlowersDataset(root=root, train=False, transform=transform)

    dataset = data.ConcatDataset([train_dataset, valid_dataset])
    class_num = 102 # Oxford Flowers 数据集有102个类别
    return dataset, class_num

def _get_aircraft_dataset(root, transform):
    """
    加载 FGVC-Aircraft 数据集。
    (转导式)：合并 'train', 'val', 和 'test' CSV文件
    作为我们的无监督训练集。
    """
    train_dataset = AircraftDataset(root=root, split='train', transform=transform)
    val_dataset = AircraftDataset(root=root, split='val', transform=transform)
    test_dataset = AircraftDataset(root=root, split='test', transform=transform)

    dataset = data.ConcatDataset([train_dataset, val_dataset, test_dataset])
    class_num = 100 # FGVC-Aircraft 有100个类别
    return dataset, class_num

def _get_oxford_pet_dataset(root, transform):
    """
    加载 Oxford-IIIT Pet 数据集。
    (转导式)：合并 'trainval.txt' 和 'test.txt'
    作为我们的无监督训练集。
    """
    trainval_dataset = OxfordPetDataset(root=root, split='trainval', transform=transform)
    test_dataset = OxfordPetDataset(root=root, split='test', transform=transform)

    dataset = data.ConcatDataset([trainval_dataset, test_dataset])
    class_num = 37 # Oxford-Pet 有 37 个类别
    return dataset, class_num

def _get_cifar10_dataset(root, transform): # <== 新增函数
    """
    加载 CIFAR-10 数据集。
    (归纳式)：只使用 'train=True' (5个databatch) 的部分进行无监督训练。
    """
    dataset = CIFAR10Dataset(root=root, train=True, transform=transform, download=True)
    class_num = 10
    return dataset, class_num



dataset_factory_orthogonal = {
    "CUB_200_2011": _get_cub_dataset_orthogonal,
    "Stanford_Dogs": _get_stanford_dogs_dataset,
    "Oxford_Flowers": _get_oxford_flowers_dataset,
    "CIFAR10": _get_cifar10_dataset,
    "FGVC_Aircraft": _get_aircraft_dataset,
    "Oxford_Pet": _get_oxford_pet_dataset,
}


def get_dataset_orthogonal(name, root, transform):
    """
    正交实验的数据集工厂主函数。
    """
    if name not in dataset_factory_orthogonal:
        raise KeyError(f"未知的数据集: {name}.")

    print(f"正在从正交数据工厂加载数据集: {name}...")
    return dataset_factory_orthogonal[name](root, transform)

# datasets/data_loader_orthogonal.py (严谨修正版)
# 修正：所有函数只加载 'train=True' 的部分，以防止数据泄露

# import os
# from torch.utils import data
# import torchvision
#
# # --- 导入所有我们支持的数据集类 ---
# from .cub_dataset import CUB200Dataset
# from .stanford_dogs_dataset import StanfordDogsDataset
# from .oxford_flowers_dataset import OxfordFlowersDataset
#
#
# def _get_cub_dataset_orthogonal(root, transform):
#     """
#     加载 CUB-200-2011 数据集。
#     修正：只使用 'train=True' (train_list.mat) 的部分进行无监督训练。
#     """
#     dataset = CUB200Dataset(root=root, train=True, transform=transform)
#     class_num = 200
#     return dataset, class_num
#
#
# def _get_stanford_dogs_dataset(root, transform):
#     """
#     加载 Stanford Dogs 数据集。
#     修正：只使用 'train=True' (train_list.mat) 的部分进行无监督训练。
#     """
#     dataset = StanfordDogsDataset(root=root, train=True, transform=transform)
#     class_num = 120
#     return dataset, class_num
#
#
# def _get_oxford_flowers_dataset(root, transform):
#     """
#     加载 Oxford-102 Flowers 数据集。
#     修正：只使用 'train=True' (即 'train' 文件夹) 进行无监督训练。
#     'valid' 文件夹将被保留，作为 cluster.py 的评估测试集。
#     """
#     dataset = OxfordFlowersDataset(root=root, train=True, transform=transform)
#     class_num = 102
#     return dataset, class_num
#
#
# # --- 数据集工厂 (现在是正确的) ---
# dataset_factory_orthogonal = {
#     "CUB_200_2011": _get_cub_dataset_orthogonal,
#     "Stanford_Dogs": _get_stanford_dogs_dataset,
#     "Oxford_Flowers": _get_oxford_flowers_dataset,
# }
#
#
# def get_dataset_orthogonal(name, root, transform):
#     """
#     通用的数据集工厂函数。
#     """
#     if name not in dataset_factory_orthogonal:
#         raise KeyError(f"未知的据集: {name}。请在 data_loader_orthogonal.py 中注册。")
#
#     print(f"--- 正在从正交数据工厂加载 *训练专用* 数据集: {name} ---")
#     return dataset_factory_orthogonal[name](root, transform)

