"""
OceanNPY Dataset - 适配 ocean-preprocess 预处理输出的数据集类（带陆地掩码支持）

从 ocean-preprocess 工具生成的目录结构加载数据：
    dataset_root/
    ├── train/hr/{var}/*.npy    (每个 npy 文件: [H, W])
    ├── train/lr/{var}/*.npy    (每个 npy 文件: [h, w])
    ├── valid/hr/{var}/*.npy
    ├── valid/lr/{var}/*.npy
    ├── test/hr/{var}/*.npy
    └── test/lr/{var}/*.npy

多个变量按 channels 维度堆叠: [N, H, W, C]

@author Leizheng
@date 2026-02-06
@version 2.0.0

@changelog
  - 2026-02-06 Leizheng: v2.0.0 添加陆地掩码支持
    - _load_split() 中从 HR 数据第一个时间步生成 mask
    - NaN 填充为 0（在归一化之前）
    - 新增 mask_hr / mask_lr 属性供 trainer 使用
  - 原始版本: v1.0.0
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.normalizer import UnitGaussianNormalizer, GaussianNormalizer


class OceanNPYDataset:
    """
    从 ocean-preprocess 预处理输出目录加载数据（带陆地掩码支持）。

    Args:
        data_args (dict):
            - dataset_root (str): 预处理输出根目录
            - dyn_vars (list[str]): 动态变量列表，如 ["temp", "salt"]
            - normalize (bool): 是否归一化 (默认 True)
            - normalizer_type (str): 'PGN' 或 'GN' (默认 'PGN')
            - train_batchsize (int): 训练 batch size
            - eval_batchsize (int): 评估 batch size
    """

    def __init__(self, data_args, **kwargs):
        dataset_root = data_args['dataset_root']
        dyn_vars = data_args['dyn_vars']
        normalize = data_args.get('normalize', True)
        normalizer_type = data_args.get('normalizer_type', 'PGN')

        # 加载三个 split 的数据
        train_hr, train_lr = self._load_split(dataset_root, 'train', dyn_vars)
        valid_hr, valid_lr = self._load_split(dataset_root, 'valid', dyn_vars)
        test_hr, test_lr = self._load_split(dataset_root, 'test', dyn_vars)

        # 从训练集 HR 第一个样本生成 mask（陆地位置在所有时间步是固定的）
        # mask: True = 海洋（有效像素），False = 陆地（NaN 位置）
        # 使用所有通道的 NaN 取并集：任一通道为 NaN 则该像素为陆地
        mask_hr = ~torch.isnan(train_hr[0:1])  # [1, H, W, C]
        # 合并所有通道：只要任一通道是 NaN（False），该像素就是陆地
        mask_hr_spatial = mask_hr.all(dim=-1, keepdim=True)  # [1, H, W, 1]

        mask_lr = ~torch.isnan(train_lr[0:1])  # [1, h, w, C]
        mask_lr_spatial = mask_lr.all(dim=-1, keepdim=True)  # [1, h, w, 1]

        # 统计掩码信息
        hr_total = mask_hr_spatial.numel()
        hr_ocean = mask_hr_spatial.sum().item()
        hr_land = hr_total - hr_ocean
        lr_total = mask_lr_spatial.numel()
        lr_ocean = mask_lr_spatial.sum().item()
        lr_land = lr_total - lr_ocean

        print(f'[OceanNPY] HR mask: {hr_ocean}/{hr_total} ocean pixels ({hr_land} land, {hr_land/hr_total*100:.1f}%)')
        print(f'[OceanNPY] LR mask: {lr_ocean}/{lr_total} ocean pixels ({lr_land} land, {lr_land/lr_total*100:.1f}%)')

        # NaN 填充为 0（必须在归一化之前）
        train_hr = torch.nan_to_num(train_hr, nan=0.0)
        train_lr = torch.nan_to_num(train_lr, nan=0.0)
        valid_hr = torch.nan_to_num(valid_hr, nan=0.0)
        valid_lr = torch.nan_to_num(valid_lr, nan=0.0)
        test_hr = torch.nan_to_num(test_hr, nan=0.0)
        test_lr = torch.nan_to_num(test_lr, nan=0.0)

        print(f'[OceanNPY] Train: HR {train_hr.shape}, LR {train_lr.shape}')
        print(f'[OceanNPY] Valid: HR {valid_hr.shape}, LR {valid_lr.shape}')
        print(f'[OceanNPY] Test:  HR {test_hr.shape}, LR {test_lr.shape}')

        # 归一化（在训练集上拟合，应用到所有 split）
        if normalize:
            B, H, W, C = train_hr.shape
            train_hr_flat = train_hr.reshape(B, -1, C)
            if normalizer_type == 'PGN':
                normalizer = UnitGaussianNormalizer(train_hr_flat)
            else:
                normalizer = GaussianNormalizer(train_hr_flat)

            train_hr = normalizer.encode(train_hr_flat).reshape(B, H, W, C)
            train_lr = normalizer.encode(train_lr.reshape(train_lr.shape[0], -1, C)).reshape(train_lr.shape)

            valid_hr = normalizer.encode(valid_hr.reshape(valid_hr.shape[0], -1, C)).reshape(valid_hr.shape)
            valid_lr = normalizer.encode(valid_lr.reshape(valid_lr.shape[0], -1, C)).reshape(valid_lr.shape)

            test_hr = normalizer.encode(test_hr.reshape(test_hr.shape[0], -1, C)).reshape(test_hr.shape)
            test_lr = normalizer.encode(test_lr.reshape(test_lr.shape[0], -1, C)).reshape(test_lr.shape)
        else:
            normalizer = None

        self.normalizer = normalizer
        # 保存 mask 供 trainer 使用
        self.mask_hr = mask_hr_spatial  # [1, H, W, 1] bool
        self.mask_lr = mask_lr_spatial  # [1, h, w, 1] bool

        self.train_dataset = OceanNPYDatasetBase(train_lr, train_hr, mode='train')
        self.valid_dataset = OceanNPYDatasetBase(valid_lr, valid_hr, mode='valid')
        self.test_dataset = OceanNPYDatasetBase(test_lr, test_hr, mode='test')

    def _load_split(self, dataset_root, split, dyn_vars):
        """
        加载某个 split 的 HR 和 LR 数据。

        Returns:
            hr_data: [N, H, W, C] tensor
            lr_data: [N, h, w, C] tensor
        """
        hr_arrays = []
        lr_arrays = []

        for var in dyn_vars:
            hr_dir = os.path.join(dataset_root, split, 'hr', var)
            lr_dir = os.path.join(dataset_root, split, 'lr', var)

            if not os.path.isdir(hr_dir):
                raise FileNotFoundError(f"HR directory not found: {hr_dir}")
            if not os.path.isdir(lr_dir):
                raise FileNotFoundError(f"LR directory not found: {lr_dir}")

            # 获取文件列表并排序（保证 HR/LR 对应）
            hr_files = sorted(glob.glob(os.path.join(hr_dir, '*.npy')))
            lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.npy')))

            if len(hr_files) == 0:
                raise FileNotFoundError(f"No .npy files found in {hr_dir}")
            if len(hr_files) != len(lr_files):
                raise ValueError(
                    f"HR/LR file count mismatch for {var} in {split}: "
                    f"HR={len(hr_files)}, LR={len(lr_files)}"
                )

            # 验证文件名一一对应
            for hf, lf in zip(hr_files, lr_files):
                if os.path.basename(hf) != os.path.basename(lf):
                    raise ValueError(
                        f"Filename mismatch: HR={os.path.basename(hf)}, LR={os.path.basename(lf)}"
                    )

            # 加载 npy 文件: 每个文件 [H, W] → 堆叠为 [N, H, W]
            hr_var = np.stack([np.load(f) for f in hr_files], axis=0)  # [N, H, W]
            lr_var = np.stack([np.load(f) for f in lr_files], axis=0)  # [N, h, w]

            hr_arrays.append(hr_var)
            lr_arrays.append(lr_var)

        # 多变量堆叠为 channels: [N, H, W, C]
        hr_data = np.stack(hr_arrays, axis=-1)  # [N, H, W, C]
        lr_data = np.stack(lr_arrays, axis=-1)  # [N, h, w, C]

        return torch.tensor(hr_data, dtype=torch.float32), torch.tensor(lr_data, dtype=torch.float32)


class OceanNPYDatasetBase(Dataset):
    """
    PyTorch Dataset wrapper。

    Args:
        x (Tensor): LR input [N, h, w, C]
        y (Tensor): HR target [N, H, W, C]
        mode (str): 'train', 'valid', or 'test'
    """

    def __init__(self, x, y, mode='train', **kwargs):
        self.mode = mode
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
