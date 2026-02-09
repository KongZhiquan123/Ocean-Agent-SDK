"""
OceanNPY Dataset - 适配 ocean-preprocess 预处理输出的数据集类（带陆地掩码支持）

从 ocean-preprocess 工具生成的目录结构加载数据：
    dataset_root/
    ├── train/hr/{var}/*.npy    (每个 npy 文件: [H, W])
    ├── train/lr/{var}/*.npy    (每个 npy 文件: [h, w])
    ├── valid/hr/{var}/*.npy
    ├── valid/lr/{var}/*.npy
    ├── test/hr/{var}/*.npy
    ├── test/lr/{var}/*.npy
    └── static_variables/       (经纬度等静态变量)

多个变量按 channels 维度堆叠: [N, H, W, C]

@author Leizheng
@contributors kongzhiquan
@date 2026-02-06
@version 7.1.0

@changelog
  - 2026-02-09 kongzhiquan: v7.1.0 get_meta() 返回完整元数据
    - filenames/dyn_vars 传入 OceanNPYDatasetBase
    - get_coords() 重命名为 get_meta()，返回 dict 含经纬度、filename、dyn_vars
    - get_meta() 正确计算 sample_idx 索引 filename
  - 2026-02-09 kongzhiquan: v7.0.0 新增经纬度/日期/变量名元数据支持
    - _load_static_coords() 从 static_variables/ 加载经纬度
    - _load_split() 额外返回文件名列表
    - OceanNPYDataset 存储 lon/lat/dyn_vars/test_filenames 元数据
  - 2026-02-08 Leizheng: v6.1.0 修复 valid/test 网格切片索引越界
  - 2026-02-08 Leizheng: v6.0.0 patch 训练默认开启
    - 用户未指定 patch_size 时自动计算合理默认值（OOM 防护）
    - patch_size 需同时满足 scale 和 model_divisor 的整除要求
    - 数据过小时自动退化为全图训练
  - 2026-02-08 Leizheng: v5.0.0 验证/测试集也使用 patch 切片
    - valid/test 数据集传入 patch_size、scale、mask_hr
    - OceanNPYDatasetBase 非训练模式使用非重叠网格切片
    - __len__ 返回 样本数 × 每样本 patch 数
    - __getitem__ 按 (sample_idx, patch_idx) 解码
  - 2026-02-07 Leizheng: v4.0.0 读取 model_divisor，自动计算 patch_size
    - 当数据尺寸不能被 model_divisor 整除且未指定 patch_size 时自动计算
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

# 经纬度文件名匹配模式（与 base.py 一致）
LON_PATTERNS = ['lon', 'longitude', 'lon_rho', 'lon_u', 'lon_v']
LAT_PATTERNS = ['lat', 'latitude', 'lat_rho', 'lat_u', 'lat_v']


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
        train_hr, train_lr, train_filenames = self._load_split(dataset_root, 'train', dyn_vars)
        valid_hr, valid_lr, valid_filenames = self._load_split(dataset_root, 'valid', dyn_vars)
        test_hr, test_lr, test_filenames = self._load_split(dataset_root, 'test', dyn_vars)

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

        # 加载经纬度元数据（直接从 static_variables/hr/ 和 lr/ 加载）
        lon_hr, lat_hr, lon_lr, lat_lr = self._load_static_coords(dataset_root)

        self.lon_hr = lon_hr          # np.ndarray 1D/2D or None
        self.lat_hr = lat_hr          # np.ndarray 1D/2D or None
        self.lon_lr = lon_lr          # np.ndarray 1D/2D or None
        self.lat_lr = lat_lr          # np.ndarray 1D/2D or None
        self.dyn_vars = dyn_vars      # list[str]
        self.test_filenames = test_filenames  # list[str]

        # Patch 训练参数
        patch_size = data_args.get('patch_size', None)
        scale = data_args.get('sample_factor', 1)
        divisor = data_args.get('model_divisor', 1)
        H, W = train_hr.shape[1], train_hr.shape[2]

        # 默认开启 patch 训练（OOM 防护 + 尺寸对齐）
        # 当用户未指定 patch_size 时，自动计算合理的默认值
        if patch_size is None:
            from math import gcd
            max_dim = min(H, W)
            # patch_size 必须同时被 scale 和 divisor 整除
            lcm_factor = (scale * divisor) // gcd(scale, divisor)
            # 目标: 不超过数据尺寸的 1/2，不超过 256
            target = min(max_dim // 2, 256)
            auto_patch = (target // lcm_factor) * lcm_factor
            if auto_patch < lcm_factor and lcm_factor < max_dim:
                auto_patch = lcm_factor
            if 0 < auto_patch < max_dim:
                patch_size = auto_patch
                print(f'[OceanNPY] 默认开启 patch 训练: patch_size={patch_size} '
                      f'(数据 {H}x{W}, scale={scale}, divisor={divisor}, '
                      f'lcm={lcm_factor})')
            else:
                print(f'[OceanNPY] 数据尺寸 {H}x{W} 较小，使用全图训练')

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
            patch_size=patch_size, scale=scale, mask_hr=mask_hr_spatial,
            lon_hr=lon_hr, lat_hr=lat_hr, lon_lr=lon_lr, lat_lr=lat_lr,
            filenames=train_filenames, dyn_vars=dyn_vars)
        self.valid_dataset = OceanNPYDatasetBase(
            valid_lr, valid_hr, mode='valid',
            patch_size=patch_size, scale=scale, mask_hr=mask_hr_spatial,
            lon_hr=lon_hr, lat_hr=lat_hr, lon_lr=lon_lr, lat_lr=lat_lr,
            filenames=valid_filenames, dyn_vars=dyn_vars)
        self.test_dataset = OceanNPYDatasetBase(
            test_lr, test_hr, mode='test',
            patch_size=patch_size, scale=scale, mask_hr=mask_hr_spatial,
            lon_hr=lon_hr, lat_hr=lat_hr, lon_lr=lon_lr, lat_lr=lat_lr,
            filenames=test_filenames, dyn_vars=dyn_vars)

    def _load_split(self, dataset_root, split, dyn_vars):
        """
        加载某个 split 的 HR 和 LR 数据。

        Returns:
            hr_data: [N, H, W, C] tensor
            lr_data: [N, h, w, C] tensor
            filenames: list[str] 排序后的文件名（不含 .npy 扩展名）
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

        # 从第一个变量的 HR 文件列表提取文件名（所有变量文件名一致，已有校验保证）
        first_hr_dir = os.path.join(dataset_root, split, 'hr', dyn_vars[0])
        filenames = [
            os.path.splitext(os.path.basename(f))[0]
            for f in sorted(glob.glob(os.path.join(first_hr_dir, '*.npy')))
        ]

        return (torch.tensor(hr_data, dtype=torch.float32),
                torch.tensor(lr_data, dtype=torch.float32),
                filenames)

    @staticmethod
    def _find_lon_lat(npy_files):
        """从 npy 文件列表中匹配经纬度文件并加载。"""
        lon = None
        lat = None
        for fpath in npy_files:
            basename = os.path.splitext(os.path.basename(fpath))[0]
            # 文件名格式: xx_varname（xx 为两位数字前缀）
            parts = basename.split('_', 1)
            varname = parts[1] if len(parts) > 1 else parts[0]
            varname_lower = varname.lower()

            if lon is None and varname_lower in LON_PATTERNS:
                lon = np.load(fpath)
            elif lat is None and varname_lower in LAT_PATTERNS:
                lat = np.load(fpath)

            if lon is not None and lat is not None:
                break
        return lon, lat

    @staticmethod
    def _load_static_coords(dataset_root):
        """
        从 dataset_root/static_variables/hr/ 和 lr/ 子目录加载经纬度。

        目录结构：
          static_variables/
          ├── hr/  (00_lon_rho.npy, 10_lat_rho.npy, ...)
          └── lr/  (00_lon_rho.npy, 10_lat_rho.npy, ...)

        Returns:
            (lon_hr, lat_hr, lon_lr, lat_lr): 各为 np.ndarray (1D 或 2D) 或 None
        """
        static_dir = os.path.join(dataset_root, 'static_variables')
        if not os.path.isdir(static_dir):
            return None, None, None, None

        hr_dir = os.path.join(static_dir, 'hr')
        lr_dir = os.path.join(static_dir, 'lr')

        if os.path.isdir(hr_dir):
            hr_files = sorted(glob.glob(os.path.join(hr_dir, '*.npy')))
            lon_hr, lat_hr = OceanNPYDataset._find_lon_lat(hr_files)
        else:
            lon_hr, lat_hr = None, None

        if os.path.isdir(lr_dir):
            lr_files = sorted(glob.glob(os.path.join(lr_dir, '*.npy')))
            lon_lr, lat_lr = OceanNPYDataset._find_lon_lat(lr_files)
        else:
            lon_lr, lat_lr = None, None

        return lon_hr, lat_hr, lon_lr, lat_lr


class OceanNPYDatasetBase(Dataset):
    """
    PyTorch Dataset wrapper，支持可选的 Patch 裁剪。

    - train 模式: 随机裁剪 patch（数据增强）
    - valid/test 模式: 非重叠网格切片（确定性，覆盖大部分面积）
    - 无 patch_size: 返回全图

    Args:
        x (Tensor): LR input [N, h, w, C]
        y (Tensor): HR target [N, H, W, C]
        mode (str): 'train', 'valid', or 'test'
        patch_size (int|None): HR patch 尺寸，None 则不裁剪
        scale (int): 超分辨率倍数（用于推导 LR patch 坐标）
        mask_hr (Tensor|None): [1, H, W, 1] bool，HR 掩码
        lon_hr (np.ndarray|None): HR 经度，1D [W] 或 2D [H, W]
        lat_hr (np.ndarray|None): HR 纬度，1D [H] 或 2D [H, W]
        lon_lr (np.ndarray|None): LR 经度，1D [w] 或 2D [h, w]
        lat_lr (np.ndarray|None): LR 纬度，1D [h] 或 2D [h, w]
        filenames (list[str]|None): 样本文件名列表（不含扩展名）
        dyn_vars (list[str]|None): 动态变量名列表

    Returns:
        有 patch_size 且有 mask_hr 时: (x, y, mask_hr_patch)
        其他情况: (x, y)
    """

    def __init__(self, x, y, mode='train', patch_size=None, scale=1, mask_hr=None,
                 lon_hr=None, lat_hr=None, lon_lr=None, lat_lr=None,
                 filenames=None, dyn_vars=None, **kwargs):
        self.mode = mode
        self.x = x
        self.y = y
        self.patch_size = patch_size
        self.scale = scale
        self.mask_hr = mask_hr  # [1, H, W, 1] bool or None
        self.lon_hr = lon_hr    # np.ndarray 1D/2D or None
        self.lat_hr = lat_hr    # np.ndarray 1D/2D or None
        self.lon_lr = lon_lr    # np.ndarray 1D/2D or None
        self.lat_lr = lat_lr    # np.ndarray 1D/2D or None
        self.filenames = filenames  # list[str] or None
        self.dyn_vars = dyn_vars    # list[str] or None
        self.mode = mode
        # 非训练模式 + 有 patch_size：预计算非重叠网格位置
        if patch_size is not None and mode != 'train':
            H, W = y.shape[1], y.shape[2]
            self._grid_positions = [
                (top, left)
                for top in range(0, H - patch_size + 1, patch_size)
                for left in range(0, W - patch_size + 1, patch_size)
            ]
            if len(self._grid_positions) == 0:
                # patch_size > 数据尺寸，退化为全图（不裁剪）
                self._grid_positions = None
            else:
                covered_h = (H // patch_size) * patch_size
                covered_w = (W // patch_size) * patch_size
                if covered_h < H or covered_w < W:
                    print(f'[OceanNPY] {mode} 网格切片: {len(self._grid_positions)} patches '
                          f'({covered_h}x{covered_w} / {H}x{W}, '
                          f'覆盖率 {covered_h*covered_w/(H*W)*100:.1f}%)')
        else:
            self._grid_positions = None

    def __len__(self):
        if self._grid_positions is not None:
            return len(self.x) * len(self._grid_positions)
        return len(self.x)

    def __getitem__(self, idx):
        if self._grid_positions is not None:
            # 非训练模式的网格切片
            n_patches = len(self._grid_positions)
            sample_idx = idx // n_patches
            patch_idx = idx % n_patches
            top, left = self._grid_positions[patch_idx]

            x = self.x[sample_idx]
            y = self.y[sample_idx]
            ps = self.patch_size

            y = y[top:top+ps, left:left+ps, :]

            lr_ps = ps // self.scale
            lr_top = top // self.scale
            lr_left = left // self.scale
            x = x[lr_top:lr_top+lr_ps, lr_left:lr_left+lr_ps, :]

            if self.mask_hr is not None:
                mask_hr_patch = self.mask_hr[0, top:top+ps, left:left+ps, :]
                return x, y, mask_hr_patch

            return x, y

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

    def get_meta(self, idx):
        """
        返回第 idx 个样本对应的元数据（经纬度、文件名、变量名）。
        注意：
        - 当 mode='train' 且 patch_size 不为 None 时，patch会是随机裁剪的，因此经纬度等元数据不适用
        - 以及，分布式训练时同一 idx 可能对应不同 patch，元数据也不适用。因此在训练模式下不支持 get_meta()。
        
        Returns:
            dict:
                lon_hr, lat_hr, lon_lr, lat_lr — numpy 数组或 None（含 patch 裁剪）
                filename — str or None（该样本对应的文件名）
                dyn_vars — list[str] or None（变量名列表）
        """
        if self.mode == 'train':
            raise NotImplementedError("get_meta() not supported in 'train' mode")
        
        lon_hr = self.lon_hr
        lat_hr = self.lat_hr
        lon_lr = self.lon_lr
        lat_lr = self.lat_lr

        if self._grid_positions is not None:
            n_patches = len(self._grid_positions)
            sample_idx = idx // n_patches
            patch_idx = idx % n_patches
            top, left = self._grid_positions[patch_idx]
            ps = self.patch_size

            lr_ps = ps // self.scale
            lr_top = top // self.scale
            lr_left = left // self.scale

            if lon_hr is not None:
                lon_hr = lon_hr[left:left+ps] if lon_hr.ndim == 1 else lon_hr[top:top+ps, left:left+ps]
            if lat_hr is not None:
                lat_hr = lat_hr[top:top+ps] if lat_hr.ndim == 1 else lat_hr[top:top+ps, left:left+ps]
            if lon_lr is not None:
                lon_lr = lon_lr[lr_left:lr_left+lr_ps] if lon_lr.ndim == 1 else lon_lr[lr_top:lr_top+lr_ps, lr_left:lr_left+lr_ps]
            if lat_lr is not None:
                lat_lr = lat_lr[lr_top:lr_top+lr_ps] if lat_lr.ndim == 1 else lat_lr[lr_top:lr_top+lr_ps, lr_left:lr_left+lr_ps]
        else:
            sample_idx = idx

        filename = self.filenames[sample_idx] if self.filenames is not None else None

        return {
            'lon_hr': lon_hr,
            'lat_hr': lat_hr,
            'lon_lr': lon_lr,
            'lat_lr': lat_lr,
            'filename': filename,
            'dyn_vars': self.dyn_vars,
        }
