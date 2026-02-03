#!/usr/bin/env python3
"""
downsample.py - 海洋数据下采样脚本

@author liuzhengyang
@contributor leizheng
@date 2026-02-03
@version 1.0.0

功能:
- 读取 hr/ 目录下的 NPY 文件
- 下采样后保存到 lr/ 目录
- 支持 NaN 处理（先填 0，下采样后恢复 NaN）
- 支持多种插值方法

用法:
    python downsample.py --dataset_root /path/to/dataset --scale 4 --method area

目录结构:
    dataset_root/
    ├── train/
    │   ├── hr/          ← 源数据
    │   │   ├── var1.npy
    │   │   └── var2.npy
    │   └── lr/          ← 下采样后保存到这里
    ├── valid/
    │   ├── hr/
    │   └── lr/
    ├── test/
    │   ├── hr/
    │   └── lr/
    └── static_variables/

Changelog:
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

    # 开始处理
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

    # 扫描 hr 目录下的所有 .npy 文件
    npy_files = glob.glob(os.path.join(hr_dir, '*.npy'))

    if not npy_files:
        print(f"[Warning] HR 目录为空: {hr_dir}")
        return []

    print(f"\n处理 {split} 数据集 ({len(npy_files)} 个文件)...")
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


def process_static_variables(
    dataset_root: str,
    scale: int,
    method_flag: int
) -> List[dict]:
    """
    处理静态变量（如果需要的话）

    Args:
        dataset_root: 数据集根目录
        scale: 下采样倍数
        method_flag: 插值方法

    Returns:
        处理结果列表
    """
    static_dir = os.path.join(dataset_root, 'static_variables')

    if not os.path.exists(static_dir):
        print(f"[Info] 静态变量目录不存在，跳过: {static_dir}")
        return []

    # 扫描静态变量文件
    npy_files = glob.glob(os.path.join(static_dir, '*.npy'))

    if not npy_files:
        print(f"[Info] 静态变量目录为空: {static_dir}")
        return []

    # 静态变量的 LR 版本保存到 static_variables_lr/
    lr_static_dir = os.path.join(dataset_root, 'static_variables_lr')

    print(f"\n处理静态变量 ({len(npy_files)} 个文件)...")
    print(f"  源目录: {static_dir}")
    print(f"  目标目录: {lr_static_dir}")

    results = []
    for src_path in sorted(npy_files):
        filename = os.path.basename(src_path)
        dst_path = os.path.join(lr_static_dir, filename)
        result = process_file(src_path, dst_path, scale, method_flag)
        if result:
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
