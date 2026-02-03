#!/usr/bin/env python3
"""
metrics.py - 下采样数据质量指标检测

@author liuzhengyang
@contributor leizheng
@date 2026-02-03
@version 1.1.0

功能:
- 计算 HR 与 LR 之间的质量指标
- LR 临时上采样到 HR 尺寸进行比较（不保存）
- HR 作为基准数据（Relative L2 的分母）
- 支持 2D/3D/4D 数据格式

指标:
- SSIM: 结构相似性 (越接近 1 越好)
- Relative L2 Error: 相对 L2 误差 (越小越好)
- MSE: 均方误差
- RMSE: 均方根误差

用法:
    python metrics.py --dataset_root /path/to/dataset --scale 4

输出:
    dataset_root/metrics_result.json

Changelog:
    - 2026-02-03 leizheng v1.1.0: 鲁棒性修复
        - 修复 HR/LR 维度不一致导致的错误
        - 修复 HR/LR 时间步不同导致的错误（只比较共同时间步）
        - 修复空数据导致的错误
        - 修复空列表 mean 返回 NaN 的问题
    - 2026-02-03 leizheng v1.0.0: 适配 Ocean-Agent-SDK 目录结构
        - 新增 --dataset_root 参数
        - LR 上采样到 HR 尺寸后比较
        - HR 作为基准（Relative L2 分母）
        - 添加 4D 数据支持
        - 输出 metrics_result.json 与 train 同级
    - 2026-02-03 liuzhengyang: 原始版本
        - SSIM/MSE/RMSE/Relative L2 计算
        - NaN 区域处理
"""

import os
import sys
import argparse
import glob
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ========================================
# 数据验证函数
# ========================================

def validate_data_pair(hr_data: np.ndarray, lr_data: np.ndarray, var_name: str) -> Tuple[bool, str, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    验证 HR 和 LR 数据对是否可以比较

    Args:
        hr_data: HR 数据
        lr_data: LR 数据
        var_name: 变量名（用于日志）

    Returns:
        (is_valid, message, (hr_aligned, lr_aligned) or None)
    """
    # 检查空数据
    if hr_data.size == 0:
        return False, "HR 数据为空", None
    if lr_data.size == 0:
        return False, "LR 数据为空", None

    # 检查维度
    if hr_data.ndim != lr_data.ndim:
        return False, f"维度不一致: HR={hr_data.ndim}D, LR={lr_data.ndim}D", None

    # 检查维度范围
    if hr_data.ndim not in [2, 3, 4]:
        return False, f"不支持的维度: {hr_data.ndim}D（仅支持 2D/3D/4D）", None

    # 对于 3D/4D 数据，检查时间步
    if hr_data.ndim >= 3:
        hr_t = hr_data.shape[0]
        lr_t = lr_data.shape[0]

        if hr_t == 0 or lr_t == 0:
            return False, f"时间步为 0: HR={hr_t}, LR={lr_t}", None

        # 如果时间步不同，取共同的部分
        if hr_t != lr_t:
            min_t = min(hr_t, lr_t)
            hr_data = hr_data[:min_t]
            lr_data = lr_data[:min_t]
            print(f"    [Info] 时间步不同 (HR={hr_t}, LR={lr_t})，只比较前 {min_t} 帧")

    # 对于 4D 数据，检查深度维度
    if hr_data.ndim == 4:
        hr_d = hr_data.shape[1]
        lr_d = lr_data.shape[1]

        if hr_d != lr_d:
            min_d = min(hr_d, lr_d)
            hr_data = hr_data[:, :min_d]
            lr_data = lr_data[:, :min_d]
            print(f"    [Info] 深度不同 (HR={hr_d}, LR={lr_d})，只比较前 {min_d} 层")

    return True, "OK", (hr_data, lr_data)


# ========================================
# 上采样函数
# ========================================

def upsample_slice(lr_slice: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    将 2D LR 切片上采样到目标尺寸（处理 NaN）

    Args:
        lr_slice: LR 2D 数据
        target_size: (width, height)

    Returns:
        上采样后的数据
    """
    # 记录 NaN 位置
    nan_mask = np.isnan(lr_slice)

    # NaN 填充为 0
    lr_filled = np.nan_to_num(lr_slice, nan=0.0)

    # 上采样数据（bicubic）
    upsampled = cv2.resize(lr_filled, target_size, interpolation=cv2.INTER_CUBIC)

    # 上采样 NaN mask（nearest）
    mask_upsampled = cv2.resize(nan_mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)

    # 恢复 NaN
    upsampled[mask_upsampled.astype(bool)] = np.nan

    return upsampled


def upsample_to_hr_shape(lr_data: np.ndarray, hr_shape: Tuple) -> np.ndarray:
    """
    将 LR 数据上采样到 HR 的 shape

    Args:
        lr_data: LR 数据
        hr_shape: HR 的 shape

    Returns:
        上采样后的 LR 数据（与 HR shape 相同）
    """
    ndim = lr_data.ndim

    # 目标尺寸 (W, H)
    target_size = (hr_shape[-1], hr_shape[-2])

    if ndim == 2:
        return upsample_slice(lr_data, target_size)

    elif ndim == 3:
        # (T, H, W)
        t = lr_data.shape[0]
        result = np.zeros((t, hr_shape[-2], hr_shape[-1]), dtype=lr_data.dtype)
        for i in range(t):
            result[i] = upsample_slice(lr_data[i], target_size)
        return result

    elif ndim == 4:
        # (T, D, H, W)
        t, d = lr_data.shape[:2]
        result = np.zeros((t, d, hr_shape[-2], hr_shape[-1]), dtype=lr_data.dtype)
        for ti in range(t):
            for di in range(d):
                result[ti, di] = upsample_slice(lr_data[ti, di], target_size)
        return result

    else:
        raise ValueError(f"不支持的维度: {ndim}D")


# ========================================
# SSIM 计算（处理 NaN）
# ========================================

def calculate_ssim_masked(img_a: np.ndarray, img_b: np.ndarray, data_range: float) -> float:
    """
    计算忽略 NaN 区域的 SSIM
    """
    # 获取共同的有效区域掩码
    valid_mask = np.isfinite(img_a) & np.isfinite(img_b)

    if np.sum(valid_mask) == 0:
        return 0.0

    # 填充 NaN 为 0
    img_a_filled = np.nan_to_num(img_a, nan=0.0)
    img_b_filled = np.nan_to_num(img_b, nan=0.0)

    # 计算 SSIM Map
    win_size = min(7, min(img_a.shape))
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        win_size = 3

    try:
        _, ssim_map = ssim(
            img_a_filled,
            img_b_filled,
            data_range=data_range,
            win_size=win_size,
            full=True
        )
    except ValueError:
        return 0.0

    # 只取有效区域的 SSIM 均值
    score = np.mean(ssim_map[valid_mask])
    return float(score)


# ========================================
# 指标计算
# ========================================

def compute_metrics(hr_data: np.ndarray, lr_upsampled: np.ndarray) -> Optional[Dict]:
    """
    计算 HR 与上采样后的 LR 之间的指标

    Args:
        hr_data: HR 数据（基准）
        lr_upsampled: 上采样后的 LR 数据

    Returns:
        指标字典
    """
    # 检查 Shape
    if hr_data.shape != lr_upsampled.shape:
        print(f"[Error] Shape mismatch after upsampling: HR={hr_data.shape}, LR_up={lr_upsampled.shape}")
        return None

    # 获取共同的有效数据掩码
    valid_mask = np.isfinite(hr_data) & np.isfinite(lr_upsampled)

    # 如果完全没有有效像素
    if np.sum(valid_mask) == 0:
        return {
            "ssim": 0.0,
            "relative_l2": 0.0,
            "mse": 0.0,
            "rmse": 0.0
        }

    hr_valid = hr_data[valid_mask]
    lr_valid = lr_upsampled[valid_mask]

    # --- MSE ---
    mse = np.mean((hr_valid - lr_valid) ** 2)

    # --- RMSE ---
    rmse = np.sqrt(mse)

    # --- Relative L2 Error ---
    # norm(hr - lr) / norm(hr)  ← HR 作为分母
    diff_norm = np.linalg.norm(hr_valid - lr_valid)
    hr_norm = np.linalg.norm(hr_valid)

    if hr_norm == 0:
        rel_l2 = 0.0
    else:
        rel_l2 = diff_norm / hr_norm

    # --- SSIM ---
    d_max = np.nanmax(hr_data)
    d_min = np.nanmin(hr_data)
    data_range = d_max - d_min
    if data_range == 0:
        data_range = 1.0

    ssim_score = 0.0
    ndim = hr_data.ndim

    if ndim == 2:
        ssim_score = calculate_ssim_masked(hr_data, lr_upsampled, data_range)

    elif ndim == 3:
        # (T, H, W) 逐帧计算取平均
        t_steps = hr_data.shape[0]
        frame_scores = []
        for t in range(t_steps):
            s = calculate_ssim_masked(hr_data[t], lr_upsampled[t], data_range)
            frame_scores.append(s)
        ssim_score = np.mean(frame_scores) if frame_scores else 0.0

    elif ndim == 4:
        # (T, D, H, W) 逐帧逐层计算取平均
        t_steps, d_steps = hr_data.shape[:2]
        all_scores = []
        for ti in range(t_steps):
            for di in range(d_steps):
                s = calculate_ssim_masked(hr_data[ti, di], lr_upsampled[ti, di], data_range)
                all_scores.append(s)
        ssim_score = np.mean(all_scores) if all_scores else 0.0

    return {
        "ssim": float(ssim_score),
        "relative_l2": float(rel_l2),
        "mse": float(mse),
        "rmse": float(rmse)
    }


def process_split(dataset_root: str, split: str, scale: int) -> Dict:
    """
    处理单个数据集划分

    Args:
        dataset_root: 数据集根目录
        split: 划分名称
        scale: 下采样倍数

    Returns:
        该划分的指标结果
    """
    hr_dir = os.path.join(dataset_root, split, 'hr')
    lr_dir = os.path.join(dataset_root, split, 'lr')

    if not os.path.exists(hr_dir):
        print(f"[Warning] HR 目录不存在，跳过: {hr_dir}")
        return {}

    if not os.path.exists(lr_dir):
        print(f"[Warning] LR 目录不存在，跳过: {lr_dir}")
        return {}

    # 扫描 HR 目录
    npy_files = glob.glob(os.path.join(hr_dir, '*.npy'))

    if not npy_files:
        print(f"[Warning] HR 目录为空: {hr_dir}")
        return {}

    print(f"\n处理 {split} 数据集 ({len(npy_files)} 个变量)...")

    results = {}

    for hr_path in sorted(npy_files):
        filename = os.path.basename(hr_path)
        var_name = os.path.splitext(filename)[0]
        lr_path = os.path.join(lr_dir, filename)

        if not os.path.exists(lr_path):
            print(f"  [Skip] LR 文件不存在: {filename}")
            continue

        try:
            # 读取数据
            hr_data = np.load(hr_path)
            lr_data = np.load(lr_path)

            # 验证数据对
            is_valid, msg, aligned = validate_data_pair(hr_data, lr_data, var_name)
            if not is_valid:
                print(f"  [Skip] {var_name}: {msg}")
                continue

            hr_data, lr_data = aligned

            # 上采样 LR 到 HR 尺寸（临时，不保存）
            lr_upsampled = upsample_to_hr_shape(lr_data, hr_data.shape)

            # 计算指标
            metrics = compute_metrics(hr_data, lr_upsampled)

            if metrics:
                results[var_name] = metrics
                print(f"  [OK] {var_name}: SSIM={metrics['ssim']:.4f}, RelL2={metrics['relative_l2']:.4f}")

        except Exception as e:
            print(f"  [Error] {var_name}: {e}")

    return results


# ========================================
# 主函数
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description="下采样数据质量指标检测 - HR vs LR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查所有划分
  python metrics.py --dataset_root /path/to/dataset --scale 4

  # 只检查 train
  python metrics.py --dataset_root /path/to/dataset --scale 4 --splits train

指标说明:
  - SSIM: 结构相似性 (0~1, 越接近 1 越好)
  - Relative L2: 相对 L2 误差 (越小越好, HR 作为分母)
  - MSE: 均方误差
  - RMSE: 均方根误差
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
        help='下采样倍数（用于验证，实际上采样根据 HR shape 自动计算）'
    )

    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'valid', 'test'],
        help='要检查的数据集划分（默认: train valid test）'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件路径（默认: dataset_root/metrics_result.json）'
    )

    args = parser.parse_args()

    # 验证参数
    if not os.path.exists(args.dataset_root):
        print(f"[Error] 数据集根目录不存在: {args.dataset_root}")
        sys.exit(1)

    # 输出路径
    output_path = args.output or os.path.join(args.dataset_root, 'metrics_result.json')

    # 打印配置
    print("=" * 60)
    print("下采样数据质量指标检测")
    print("=" * 60)
    print(f"数据集根目录: {args.dataset_root}")
    print(f"下采样倍数: {args.scale}x")
    print(f"检查划分: {', '.join(args.splits)}")
    print(f"输出文件: {output_path}")
    print("=" * 60)
    print("注: LR 临时上采样到 HR 尺寸进行比较，HR 作为基准（Relative L2 分母）")

    all_results = {
        "config": {
            "dataset_root": args.dataset_root,
            "scale": args.scale,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "splits": {}
    }

    # 处理各个划分
    for split in args.splits:
        results = process_split(args.dataset_root, split, args.scale)
        if results:
            all_results["splits"][split] = results

    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("指标检测完成！")
    print(f"结果保存在: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
