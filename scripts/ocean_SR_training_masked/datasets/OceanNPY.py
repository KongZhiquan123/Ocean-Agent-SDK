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
@version 3.0.0

@changelog
  - 2026-02-07 Leizheng: v3.0.0 添加 Patch 训练支持
    - OceanNPYDatasetBase 支持 patch_size 参数，训练时随机裁剪 HR/LR patch
    - 裁剪同时裁剪对应的 mask，返回 (x, y, mask_hr_patch) 三元组
    - 仅训练集裁剪，验证/测试集仍使用全图
  - 2026-02-06 Leizheng: v2.1.0 修复 PGN 归一化器 HR/LR 空间分辨率不匹配
    - PGN 模式下 HR 和 LR 使用各自独立的 normalizer（空间维度不同不能共用）
    - GN 模式下 HR 和 LR 共用同一个 normalizer（全局标量统计量）
    - normalizer 改为 dict: {'hr': normalizer_hr, 'lr': normalizer_lr}
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
        # 注意：PGN (UnitGaussianNormalizer) 是逐空间点归一化，mean/std 形状 = [N_spatial, C]
        #       HR 和 LR 空间分辨率不同（如 174240 vs 10890），所以 PGN 必须分别拟合
        #       GN (GaussianNormalizer) 是全局标量归一化，mean/std 是标量，可以共用
        if normalize:
            B_hr, H, W, C = train_hr.shape
            B_lr = train_lr.shape[0]
            h, w = train_lr.shape[1], train_lr.shape[2]

            train_hr_flat = train_hr.reshape(B_hr, -1, C)
            train_lr_flat = train_lr.reshape(B_lr, -1, C)

            if normalizer_type == 'PGN':
                # PGN: HR 和 LR 各自独立的 normalizer
                normalizer_hr = UnitGaussianNormalizer(train_hr_flat)
                normalizer_lr = UnitGaussianNormalizer(train_lr_flat)
            else:
                # GN: 全局标量统计，HR 和 LR 共用同一个
                normalizer_hr = GaussianNormalizer(train_hr_flat)
                normalizer_lr = normalizer_hr

            train_hr = normalizer_hr.encode(train_hr_flat).reshape(B_hr, H, W, C)
            train_lr = normalizer_lr.encode(train_lr_flat).reshape(B_lr, h, w, C)

            valid_hr = normalizer_hr.encode(valid_hr.reshape(valid_hr.shape[0], -1, C)).reshape(valid_hr.shape)
            valid_lr = normalizer_lr.encode(valid_lr.reshape(valid_lr.shape[0], -1, C)).reshape(valid_lr.shape)

            test_hr = normalizer_hr.encode(test_hr.reshape(test_hr.shape[0], -1, C)).reshape(test_hr.shape)
            test_lr = normalizer_lr.encode(test_lr.reshape(test_lr.shape[0], -1, C)).reshape(test_lr.shape)

            print(f'[OceanNPY] Normalizer type: {normalizer_type}')
            if normalizer_type == 'PGN':
                print(f'[OceanNPY] HR normalizer: mean/std shape {normalizer_hr.mean.shape}')
                print(f'[OceanNPY] LR normalizer: mean/std shape {normalizer_lr.mean.shape}')
        else:
            normalizer_hr = None
            normalizer_lr = None

        # normalizer 保存为 dict，方便 trainer 在 decode 时区分 HR/LR
        self.normalizer = {'hr': normalizer_hr, 'lr': normalizer_lr}
        # 保存 mask 供 trainer 使用
        self.mask_hr = mask_hr_spatial  # [1, H, W, 1] bool
        self.mask_lr = mask_lr_spatial  # [1, h, w, 1] bool

        # Patch 训练参数
        patch_size = data_args.get('patch_size', None)
        scale = data_args.get('sample_factor', 1)

        if patch_size is not None:
            H, W = train_hr.shape[1], train_hr.shape[2]
            assert patch_size <= H and patch_size <= W, (
                f"patch_size ({patch_size}) must be <= HR spatial dims ({H}x{W})")
            assert patch_size % scale == 0, (
                f"patch_size ({patch_size}) must be divisible by scale ({scale})")
            print(f'[OceanNPY] Patch training: HR patch {patch_size}x{patch_size}, '
                  f'LR patch {patch_size//scale}x{patch_size//scale}')

        self.train_dataset = OceanNPYDatasetBase(
            train_lr, train_hr, mode='train',
            patch_size=patch_size, scale=scale, mask_hr=mask_hr_spatial)
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
    PyTorch Dataset wrapper，支持可选的 Patch 随机裁剪。

    Args:
        x (Tensor): LR input [N, h, w, C]
        y (Tensor): HR target [N, H, W, C]
        mode (str): 'train', 'valid', or 'test'
        patch_size (int|None): HR patch 尺寸，None 则不裁剪
        scale (int): 超分辨率倍数（用于推导 LR patch 坐标）
        mask_hr (Tensor|None): [1, H, W, 1] bool，HR 掩码

    Returns:
        训练且 patch_size 有效且 mask_hr 存在时: (x, y, mask_hr_patch)
        其他情况: (x, y)
    """

    def __init__(self, x, y, mode='train', patch_size=None, scale=1, mask_hr=None, **kwargs):
        self.mode = mode
        self.x = x
        self.y = y
        self.patch_size = patch_size
        self.scale = scale
        self.mask_hr = mask_hr  # [1, H, W, 1] bool or None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]  # [h, w, C]
        y = self.y[idx]  # [H, W, C]

        if self.patch_size is not None and self.mode == 'train':
            H, W, C = y.shape
            ps = self.patch_size

            # 随机裁剪 HR patch
            top = torch.randint(0, H - ps + 1, (1,)).item()
            left = torch.randint(0, W - ps + 1, (1,)).item()
            y = y[top:top+ps, left:left+ps, :]

            # 推导对应的 LR patch 坐标
            lr_ps = ps // self.scale
            lr_top = top // self.scale
            lr_left = left // self.scale
            x = x[lr_top:lr_top+lr_ps, lr_left:lr_left+lr_ps, :]

            # 裁剪对应的 mask patch
            if self.mask_hr is not None:
                mask_hr_patch = self.mask_hr[0, top:top+ps, left:left+ps, :]  # [ps, ps, 1]
                return x, y, mask_hr_patch

        return x, y
