#!/usr/bin/env python3
"""
downsample.py - 海洋数据下采样脚本

@author liuzhengyang
@contributor leizheng, kongzhiquan
@date 2026-02-03
@version 3.1.1

功能:
- 读取 hr/ 目录下的 NPY 文件
- 下采样后保存到 lr/ 目录
- 支持 NaN 处理（先填 0，下采样后恢复 NaN）
- 支持多种插值方法
- v3.1.0: 静态变量类型特定插值（经纬度=linear, 掩码=nearest）
- v3.1.1: 支持 1D 数组下采样（经纬度坐标数组）

用法:
    python downsample.py --dataset_root /path/to/dataset --scale 4 --method area

目录结构 (v3.1.0):
    dataset_root/
    ├── train/
    │   ├── raw/              ← 区域裁剪后的原始数据（可选）
    │   ├── hr/
    │   │   ├── uo/          ← 每个变量一个子目录
    │   │   │   ├── 000000.npy
    │   │   │   └── ...
    │   │   └── vo/
    │   └── lr/              ← 下采样后保存到这里
    │       ├── uo/
    │       └── vo/
    ├── valid/
    ├── test/
    └── static_variables/
        ├── raw/             ← 区域裁剪后的静态变量（可选）
        ├── hr/              ← 尺寸裁剪后的静态变量
        └── lr/              ← 下采样后的静态变量

Changelog:
    - 2026-02-05 kongzhiquan v3.1.1: 支持 1D 数组下采样
        - 修复静态变量 lr 目录为空的问题
        - 经纬度坐标数组（1D）使用 scipy.ndimage.zoom 线性插值
    - 2026-02-05 kongzhiquan v3.1.0: 静态变量类型特定插值
        - 经纬度坐标使用线性插值 (bilinear)
        - 掩码使用最近邻插值 (nearest)
        - 其他静态变量使用用户指定的方法
        - 支持新目录结构: static_variables/hr/ → static_variables/lr/
        - 兼容旧目录结构: static_variables/ → static_variables/lr/
    - 2026-02-04 leizheng v2.0.0: 支持逐时间步文件格式
        - 适配 convert_npy.py v2.9.0 的新目录结构
        - hr/uo/000000.npy, hr/vo/000000.npy 等
        - 输出到 lr/uo/000000.npy, lr/vo/000000.npy 等
    - 2026-02-03 leizheng v1.0.0: 适配 Ocean-Agent-SDK 目录结构
        - 新增 --dataset_root 参数
        - 新增 --scale 参数（替代 --size）
        - 自动扫描 hr/ 目录下的文件
        - 输出到对应的 lr/ 目录
    - 2026-02-03 liuzhengyang: 原始版本
        - NaN 处理逻辑
        - 多种插值方法支持
"""

import os
import sys
import argparse
import glob
import json
from datetime import datetime, timezone
from typing import List, Tuple, Optional

import cv2
import numpy as np

__all__ = [
    'get_cv2_interpolation',
    'resize_with_nan_handling',
    'process_file',
    'process_split',
    'get_static_var_type',
    'process_static_variables',
    'INTERPOLATION_METHODS',
]


# ========================================
# 插值方法映射
# ========================================

INTERPOLATION_METHODS = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    'cubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


def get_cv2_interpolation(method_name: str) -> int:
    """
    根据字符串返回 OpenCV 的插值方法
    """
    method_key = method_name.lower()
    if method_key not in INTERPOLATION_METHODS:
        valid_keys = ", ".join(INTERPOLATION_METHODS.keys())
        print(f"[Error] 不支持的插值方法: '{method_name}'. 支持的方法: {valid_keys}")
        sys.exit(1)
    return INTERPOLATION_METHODS[method_key]


# ========================================
# 核心处理函数
# ========================================

def resize_with_nan_handling(image: np.ndarray, target_size: Tuple[int, int], method_flag: int) -> np.ndarray:
    """
    处理单帧图像的下采样，包含 NaN 处理逻辑

    Args:
        image: 输入图像 (2D numpy array)
        target_size: (width, height)
        method_flag: OpenCV 插值方法标志

    Returns:
        下采样后的图像
    """
    # 1. 记录 NaN 位置
    nan_mask = np.isnan(image)

    # 2. NaN 填充为 0，防止计算时污染周边像素
    image_zero_filled = np.nan_to_num(image, nan=0.0)

    # 3. 下采样数据（使用用户指定的方法）
    resized_data = cv2.resize(image_zero_filled, target_size, interpolation=method_flag)

    # 4. 下采样 NaN Mask（强制使用 INTER_NEAREST）
    # 将 mask 转为 uint8 进行最近邻插值，保证结果只有 0 或 1
    resized_mask = cv2.resize(nan_mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)

    # 5. 恢复 NaN
    resized_data[resized_mask.astype(bool)] = np.nan

    return resized_data


def process_file(
    src_path: str,
    dst_path: str,
    scale: int,
    method_flag: int
) -> Optional[dict]:
    """
    读取、下采样、保存单个文件

    Args:
        src_path: 源文件路径
        dst_path: 目标文件路径
        scale: 下采样倍数
        method_flag: 插值方法

    Returns:
        处理结果信息，失败返回 None
    """
    if not os.path.exists(src_path):
        print(f"[Warning] 文件不存在，跳过: {src_path}")
        return None

    try:
        data = np.load(src_path)
    except Exception as e:
        print(f"[Error] 读取文件失败 {src_path}: {e}")
        return None

    # 获取原始形状
    original_shape = data.shape

    if data.ndim == 1:
        # 1D 数组（如经纬度坐标）
        n = data.shape[0]
        target_n = n // scale
        if target_n < 1:
            print(f"[Error] 下采样后尺寸太小: {target_n}, 原始: {n}, scale={scale}")
            return None
        # 使用线性插值对 1D 数组下采样
        from scipy.ndimage import zoom
        result_data = zoom(data, 1.0 / scale, order=1)  # order=1 为线性插值
        # 确保精确的目标长度
        if len(result_data) != target_n:
            # 使用简单的等距采样作为备选
            indices = np.linspace(0, n - 1, target_n).astype(int)
            result_data = data[indices]
        # 保存
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        try:
            np.save(dst_path, result_data)
            print(f"[OK] {os.path.basename(src_path)}: {original_shape} -> {result_data.shape}")
            return {
                "src": src_path,
                "dst": dst_path,
                "original_shape": list(original_shape),
                "result_shape": list(result_data.shape),
                "scale": scale
            }
        except Exception as e:
            print(f"[Error] 保存文件失败 {dst_path}: {e}")
            return None

    # 以下处理 2D/3D/4D 数据
    if data.ndim == 2:
        h, w = data.shape
    elif data.ndim == 3:
        t, h, w = data.shape
    elif data.ndim == 4:
        # (T, D, H, W) 格式，取最后两维
        t, d, h, w = data.shape
    else:
        print(f"[Error] 数据维度不支持 (仅支持 2D/3D/4D): {src_path}, shape={data.shape}")
        return None

    # 计算目标尺寸
    target_w = w // scale
    target_h = h // scale

    if target_w < 1 or target_h < 1:
        print(f"[Error] 下采样后尺寸太小: ({target_w}, {target_h}), 原始: ({w}, {h}), scale={scale}")
        return None

    target_size = (target_w, target_h)

    # 开始处理 2D/3D/4D 数据
    if data.ndim == 2:
        # 2D 数据
        result_data = resize_with_nan_handling(data, target_size, method_flag)
    elif data.ndim == 3:
        # 3D 数据 (T, H, W)，逐帧处理
        t, h, w = data.shape
        result_data = np.zeros((t, target_h, target_w), dtype=data.dtype)
        for i in range(t):
            result_data[i] = resize_with_nan_handling(data[i], target_size, method_flag)
    elif data.ndim == 4:
        # 4D 数据 (T, D, H, W)，逐帧逐深度处理
        t, d, h, w = data.shape
        result_data = np.zeros((t, d, target_h, target_w), dtype=data.dtype)
        for ti in range(t):
            for di in range(d):
                result_data[ti, di] = resize_with_nan_handling(data[ti, di], target_size, method_flag)

    # 确保目标目录存在
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # 保存
    try:
        np.save(dst_path, result_data)
        print(f"[OK] {os.path.basename(src_path)}: {original_shape} -> {result_data.shape}")
        return {
            "src": src_path,
            "dst": dst_path,
            "original_shape": list(original_shape),
            "result_shape": list(result_data.shape),
            "scale": scale
        }
    except Exception as e:
        print(f"[Error] 保存文件失败 {dst_path}: {e}")
        return None


def process_split(
    dataset_root: str,
    split: str,
    scale: int,
    method_flag: int
) -> List[dict]:
    """
    处理单个数据集划分（train/valid/test）

    Args:
        dataset_root: 数据集根目录
        split: 划分名称（train/valid/test）
        scale: 下采样倍数
        method_flag: 插值方法

    Returns:
        处理结果列表
    """
    hr_dir = os.path.join(dataset_root, split, 'hr')
    lr_dir = os.path.join(dataset_root, split, 'lr')

    if not os.path.exists(hr_dir):
        print(f"[Warning] HR 目录不存在，跳过: {hr_dir}")
        return []

    # v2.0.0: 扫描 hr 目录下的子目录（每个变量一个子目录）
    var_dirs = [d for d in os.listdir(hr_dir) if os.path.isdir(os.path.join(hr_dir, d))]

    # 如果没有子目录，尝试旧格式（直接扫描 .npy 文件）
    if not var_dirs:
        npy_files = glob.glob(os.path.join(hr_dir, '*.npy'))
        if not npy_files:
            print(f"[Warning] HR 目录为空: {hr_dir}")
            return []

        # 旧格式：直接处理 hr/*.npy 文件
        print(f"\n处理 {split} 数据集 [旧格式] ({len(npy_files)} 个文件)...")
        print(f"  HR 目录: {hr_dir}")
        print(f"  LR 目录: {lr_dir}")

        results = []
        for src_path in sorted(npy_files):
            filename = os.path.basename(src_path)
            dst_path = os.path.join(lr_dir, filename)
            result = process_file(src_path, dst_path, scale, method_flag)
            if result:
                results.append(result)
        return results

    # v2.0.0 新格式：hr/var_name/000000.npy
    total_files = 0
    for var_name in var_dirs:
        var_hr_dir = os.path.join(hr_dir, var_name)
        npy_files = glob.glob(os.path.join(var_hr_dir, '*.npy'))
        total_files += len(npy_files)

    print(f"\n处理 {split} 数据集 [新格式] ({len(var_dirs)} 个变量, {total_files} 个文件)...")
    print(f"  HR 目录: {hr_dir}")
    print(f"  LR 目录: {lr_dir}")

    results = []
    for var_name in sorted(var_dirs):
        var_hr_dir = os.path.join(hr_dir, var_name)
        var_lr_dir = os.path.join(lr_dir, var_name)
        os.makedirs(var_lr_dir, exist_ok=True)

        npy_files = glob.glob(os.path.join(var_hr_dir, '*.npy'))
        print(f"  变量 {var_name}: {len(npy_files)} 个文件")

        for src_path in sorted(npy_files):
            filename = os.path.basename(src_path)
            dst_path = os.path.join(var_lr_dir, filename)
            result = process_file(src_path, dst_path, scale, method_flag)
            if result:
                result['variable'] = var_name
                results.append(result)

    return results


# ========================================
# 静态变量类型判断
# ========================================

# 经度变量名模式
LON_PATTERNS = ['lon', 'longitude', 'lon_rho', 'lon_u', 'lon_v']
# 纬度变量名模式
LAT_PATTERNS = ['lat', 'latitude', 'lat_rho', 'lat_u', 'lat_v']
# 掩码变量名模式
MASK_PATTERNS = ['mask', 'land_mask', 'mask_rho', 'mask_u', 'mask_v']


def get_static_var_type(filename: str) -> str:
    """
    根据文件名判断静态变量类型

    Args:
        filename: 文件名（如 00_longitude.npy, 90_mask_rho.npy）

    Returns:
        变量类型: 'lon', 'lat', 'mask', 'other'
    """
    # 移除前缀编号和扩展名
    name_lower = filename.lower()

    # 检查是否是经度
    for pattern in LON_PATTERNS:
        if pattern in name_lower:
            return 'lon'

    # 检查是否是纬度
    for pattern in LAT_PATTERNS:
        if pattern in name_lower:
            return 'lat'

    # 检查是否是掩码
    for pattern in MASK_PATTERNS:
        if pattern in name_lower:
            return 'mask'

    return 'other'


def process_static_variables(
    dataset_root: str,
    scale: int,
    method_flag: int
) -> List[dict]:
    """
    处理静态变量（支持新的目录结构和类型特定插值）

    目录结构:
    - 优先从 static_variables/hr/ 读取（如果存在）
    - 否则从 static_variables/ 读取
    - 输出到 static_variables/lr/

    插值规则:
    - 经纬度坐标: 线性插值 (bilinear)
    - 掩码: 最近邻插值 (nearest)
    - 其他: 使用用户指定的方法

    Args:
        dataset_root: 数据集根目录
        scale: 下采样倍数
        method_flag: 默认插值方法（用于非经纬度非掩码变量）

    Returns:
        处理结果列表
    """
    # 确定源目录（优先使用 hr/ 子目录）
    static_hr_dir = os.path.join(dataset_root, 'static_variables', 'hr')
    static_dir = os.path.join(dataset_root, 'static_variables')

    if os.path.exists(static_hr_dir) and os.listdir(static_hr_dir):
        source_dir = static_hr_dir
        print(f"[Info] 使用 static_variables/hr/ 作为源目录")
    elif os.path.exists(static_dir):
        source_dir = static_dir
    else:
        print(f"[Info] 静态变量目录不存在，跳过")
        return []

    # 扫描静态变量文件
    npy_files = glob.glob(os.path.join(source_dir, '*.npy'))

    if not npy_files:
        print(f"[Info] 静态变量目录为空: {source_dir}")
        return []

    # 输出目录
    lr_static_dir = os.path.join(dataset_root, 'static_variables', 'lr')
    os.makedirs(lr_static_dir, exist_ok=True)

    print(f"\n处理静态变量 ({len(npy_files)} 个文件)...")
    print(f"  源目录: {source_dir}")
    print(f"  目标目录: {lr_static_dir}")
    print(f"  插值规则: 经纬度=linear, 掩码=nearest, 其他=用户指定")

    results = []
    for src_path in sorted(npy_files):
        filename = os.path.basename(src_path)
        dst_path = os.path.join(lr_static_dir, filename)

        # 根据变量类型选择插值方法
        var_type = get_static_var_type(filename)
        if var_type in ['lon', 'lat']:
            actual_method = cv2.INTER_LINEAR
            method_name = 'linear'
        elif var_type == 'mask':
            actual_method = cv2.INTER_NEAREST
            method_name = 'nearest'
        else:
            actual_method = method_flag
            method_name = 'user_specified'

        print(f"  {filename}: 类型={var_type}, 插值={method_name}")

        result = process_file(src_path, dst_path, scale, actual_method)
        if result:
            result['var_type'] = var_type
            result['interpolation'] = method_name
            results.append(result)

    return results


# ========================================
# 主函数
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description="海洋数据下采样脚本 - 从 hr/ 下采样到 lr/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 4倍下采样，使用 area 插值
  python downsample.py --dataset_root /path/to/dataset --scale 4 --method area

  # 只处理 train 和 valid
  python downsample.py --dataset_root /path/to/dataset --scale 4 --splits train valid

  # 同时处理静态变量
  python downsample.py --dataset_root /path/to/dataset --scale 4 --include_static
        """
    )

    parser.add_argument(
        '--dataset_root',
        required=True,
        type=str,
        help='数据集根目录（包含 train/valid/test 子目录）'
    )

    parser.add_argument(
        '--scale',
        required=True,
        type=int,
        help='下采样倍数（如 4 表示尺寸缩小为 1/4）'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='area',
        choices=['nearest', 'linear', 'cubic', 'area', 'lanczos'],
        help='插值方法（默认: area，推荐用于下采样）'
    )

    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'valid', 'test'],
        help='要处理的数据集划分（默认: train valid test）'
    )

    parser.add_argument(
        '--include_static',
        action='store_true',
        help='是否同时处理静态变量'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出结果 JSON 文件路径（可选）'
    )

    args = parser.parse_args()

    # 验证参数
    if args.scale < 1:
        print(f"[Error] scale 必须 >= 1，当前值: {args.scale}")
        sys.exit(1)

    if not os.path.exists(args.dataset_root):
        print(f"[Error] 数据集根目录不存在: {args.dataset_root}")
        sys.exit(1)

    method_flag = get_cv2_interpolation(args.method)

    # 打印配置
    print("=" * 60)
    print("海洋数据下采样")
    print("=" * 60)
    print(f"数据集根目录: {args.dataset_root}")
    print(f"下采样倍数: {args.scale}x")
    print(f"插值方法: {args.method}")
    print(f"处理划分: {', '.join(args.splits)}")
    print(f"处理静态变量: {'是' if args.include_static else '否'}")
    print("=" * 60)

    all_results = {
        "dataset_root": args.dataset_root,
        "scale": args.scale,
        "method": args.method,
        "splits": {},
        "static_variables": [],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    # 处理各个数据集划分
    for split in args.splits:
        results = process_split(args.dataset_root, split, args.scale, method_flag)
        all_results["splits"][split] = results

    # 处理静态变量（如果需要）
    if args.include_static:
        static_results = process_static_variables(args.dataset_root, args.scale, method_flag)
        all_results["static_variables"] = static_results

    # 统计
    total_files = sum(len(r) for r in all_results["splits"].values())
    total_files += len(all_results["static_variables"])

    print("\n" + "=" * 60)
    print(f"处理完成！共处理 {total_files} 个文件")
    print("=" * 60)

    # 输出结果 JSON
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {args.output}")

    # 同时输出到 stdout
    print(json.dumps(all_results, ensure_ascii=False))


if __name__ == "__main__":
    main()
