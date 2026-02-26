#!/usr/bin/env python3
"""
@file forecast_visualize.py

@description 海洋预报数据可视化（Step C）
             从 NPY 目录读取预报数据，生成样本帧、时序统计图和分布直方图

@author Leizheng
@date 2026-02-25
@version 1.3.0

@changelog
  - 2026-02-26 Leizheng: v1.4.0 增强坐标变量加载逻辑
    - _load_coord_var 三级搜索：首选 var_names.json 配置 → 形状匹配 → 关键词兜底
    - 读取 var_names.json 中的 lon_var/lat_var/spatial_shape 指导坐标选择
    - 解决 ROMS 数据坐标变量（lon_rho/lat_rho）无法正确加载的问题
  - 2026-02-26 Leizheng: v1.3.0 修复 frames 图三大问题
    - 1D 坐标匹配：lon(W,) / lat(H,) 与 data(H,W) 按维度长度匹配，构建 extent
    - 标题增加日期范围和样本数：{var} | {split} ({first} ~ {last}, n=N)
    - colorbar 改用 gridspec 预留独立列（cax），消除与子图重叠
  - 2026-02-26 Leizheng: v1.2.0 移除中文字体配置，所有标签改用英文
    - 删除 _configure_fonts()，不再强制覆盖 font.family
    - 根因：Droid Sans Fallback 缺少拉丁字符 glyph，导致所有文字变方框
  - 2026-02-26 Leizheng: v1.1.0 新增分布直方图
    - 新增 plot_distribution_histogram() 函数
    - 均匀采样帧合并值域，绘制填充直方图并标注 P5/P95 分位数
  - 2026-02-25 Leizheng: v1.0.0 初始版本
    - 支持从 {split}/{var_name}/*.npy 目录结构读取数据
    - 每个变量生成：样本帧空间分布图 + 时序统计图
    - 英文标签（不依赖系统中文字体）
    - 自动加载 static_variables/ 中的经纬度坐标

用法:
    python forecast_visualize.py --dataset_root /path/to/data \\
        --splits train valid test --out_dir /path/to/output
"""

import argparse
import glob
import json
import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
except ImportError:
    print(json.dumps({"status": "error", "errors": ["需要安装 matplotlib: pip install matplotlib"]}))
    sys.exit(1)

# ========================================
# 辅助函数
# ========================================

def _load_coord_var(
    static_dir: str,
    keyword: str,
    preferred_var: Optional[str] = None,
    target_shape: Optional[tuple] = None
) -> Optional[np.ndarray]:
    """
    从 static_variables/ 加载经纬度坐标。

    搜索优先级：
    1. preferred_var（var_names.json 中的 lon_var/lat_var）
    2. 匹配 target_shape 的 2D 数组（如 lon_rho (400,441) 匹配数据的 HxW）
    3. 任何包含 keyword 的文件（旧行为兜底）

    Args:
        static_dir: static_variables/ 目录路径
        keyword: 搜索关键词（'lon' 或 'lat'）
        preferred_var: 首选变量名（来自 var_names.json）
        target_shape: 目标空间形状 (H, W)，用于筛选匹配的坐标数组
    """
    if not os.path.isdir(static_dir):
        return None

    files = sorted(os.listdir(static_dir))

    # 辅助：从文件名提取变量名（去掉 "NN_" 编号前缀）
    def _var_name(f: str) -> str:
        base = f.replace('.npy', '')
        return base.split('_', 1)[-1] if '_' in base else base

    # 策略1：精确匹配 preferred_var
    if preferred_var:
        for f in files:
            if f.endswith('.npy') and _var_name(f) == preferred_var:
                try:
                    arr = np.load(os.path.join(static_dir, f))
                    return arr
                except Exception:
                    pass

    # 策略2：匹配 target_shape 的 2D 坐标数组
    if target_shape:
        for f in files:
            if keyword in f.lower() and f.endswith('.npy'):
                try:
                    arr = np.load(os.path.join(static_dir, f))
                    if arr.ndim == 2 and arr.shape == target_shape:
                        return arr
                except Exception:
                    pass

    # 策略3：关键词匹配兜底
    for f in files:
        if keyword in f.lower() and f.endswith('.npy'):
            try:
                return np.load(os.path.join(static_dir, f))
            except Exception:
                pass

    return None


def _get_sorted_npy_files(split_var_dir: str) -> List[str]:
    """获取目录下按名称排序的 NPY 文件列表"""
    files = sorted(glob.glob(os.path.join(split_var_dir, '*.npy')))
    return files


def _safe_load(path: str) -> Optional[np.ndarray]:
    """安全加载 NPY 文件，失败返回 None"""
    try:
        arr = np.load(path)
        return arr.astype(np.float32)
    except Exception:
        return None


def _nan_stats(arr: np.ndarray) -> Tuple[float, float]:
    """计算忽略 NaN 的均值和标准差"""
    mean = float(np.nanmean(arr)) if arr.size > 0 else 0.0
    std = float(np.nanstd(arr)) if arr.size > 0 else 0.0
    return mean, std


# ========================================
# 图表生成
# ========================================

def plot_sample_frames(
    npy_files: List[str],
    var_name: str,
    split: str,
    lon_arr: Optional[np.ndarray],
    lat_arr: Optional[np.ndarray],
    out_path: str,
    n_samples: int = 4
):
    """
    Generate spatial distribution plots from uniformly sampled frames.
    Supports both 1D coordinate axes and 2D coordinate grids.
    Uses a dedicated colorbar axis to avoid overlap with subplots.
    """
    if not npy_files:
        return

    n = min(n_samples, len(npy_files))
    if n == 0:
        return
    indices = [int(i * (len(npy_files) - 1) / max(n - 1, 1)) for i in range(n)]
    selected = [npy_files[i] for i in indices]

    arrays = []
    for fpath in selected:
        data = _safe_load(fpath)
        if data is not None:
            if data.ndim > 2:
                data = data.reshape(-1, *data.shape[-2:])[0]
            arrays.append(data)

    if not arrays:
        return

    all_vals = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
    vabs = float(np.nanpercentile(np.abs(all_vals), 99)) if all_vals.size > 0 else 1.0
    vmin, vmax = -vabs, vabs

    # Build extent from coordinate arrays (1D or 2D)
    extent = None
    has_coords = False
    if lon_arr is not None and lat_arr is not None:
        h, w = arrays[0].shape[:2]
        if lon_arr.ndim == 1 and lat_arr.ndim == 1 and lon_arr.shape[0] == w and lat_arr.shape[0] == h:
            # 1D axes: lon matches W dimension, lat matches H dimension
            extent = [float(np.nanmin(lon_arr)), float(np.nanmax(lon_arr)),
                      float(np.nanmin(lat_arr)), float(np.nanmax(lat_arr))]
            has_coords = True
        elif lon_arr.shape == arrays[0].shape and lat_arr.shape == arrays[0].shape:
            # 2D grids matching data shape
            extent = [float(np.nanmin(lon_arr)), float(np.nanmax(lon_arr)),
                      float(np.nanmin(lat_arr)), float(np.nanmax(lat_arr))]
            has_coords = True

    # Title with date range from filenames
    first_date = os.path.splitext(os.path.basename(npy_files[0]))[0]
    last_date = os.path.splitext(os.path.basename(npy_files[-1]))[0]
    title = f'{var_name}  |  {split}  ({first_date} ~ {last_date}, n={len(npy_files)})'

    # Use gridspec to reserve a dedicated colorbar column (avoids overlap)
    fig_width = 4.0 * n + 0.6  # extra width for colorbar
    fig = plt.figure(figsize=(fig_width, 4.0))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [0.04], wspace=0.15)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    cbar_ax = fig.add_subplot(gs[0, n])

    fig.suptitle(title, fontsize=11, y=1.02)

    im_last = None
    arr_idx = 0
    for col, fpath in enumerate(selected):
        ax = axes[col]
        if arr_idx >= len(arrays):
            ax.set_visible(False)
            continue
        data = arrays[arr_idx]
        arr_idx += 1

        fname = os.path.splitext(os.path.basename(fpath))[0]

        if has_coords and extent is not None:
            im = ax.imshow(data, origin='lower', extent=extent, aspect='auto',
                           cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_xlabel('Lon (°E)', fontsize=7)
            if col == 0:
                ax.set_ylabel('Lat (°N)', fontsize=7)
            else:
                ax.set_yticklabels([])
        else:
            im = ax.imshow(data, origin='lower', aspect='auto',
                           cmap='viridis', vmin=vmin, vmax=vmax)
            if col > 0:
                ax.set_yticklabels([])

        ax.set_title(fname, fontsize=8)
        ax.tick_params(labelsize=6)
        im_last = im

    if im_last is not None:
        cbar = fig.colorbar(im_last, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=7)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_time_series_stats(
    npy_files: List[str],
    var_name: str,
    split: str,
    out_path: str,
    max_files: int = 1000
):
    """
    生成时序统计图：均值和标准差随时间变化
    """
    if not npy_files:
        return

    files = npy_files[:max_files]
    filenames = [os.path.splitext(os.path.basename(f))[0] for f in files]
    means = []
    stds = []

    for fpath in files:
        data = _safe_load(fpath)
        if data is None:
            means.append(float('nan'))
            stds.append(float('nan'))
            continue
        m, s = _nan_stats(data)
        means.append(m)
        stds.append(s)

    x = list(range(len(files)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    fig.suptitle(f'{var_name} | {split} — time-series statistics', fontsize=13)

    ax1.plot(x, means, color='steelblue', linewidth=0.8)
    ax1.set_ylabel('Spatial Mean', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Spatial Mean over Time', fontsize=10)

    ax2.plot(x, stds, color='tomato', linewidth=0.8)
    ax2.set_ylabel('Spatial Std', fontsize=9)
    ax2.set_xlabel('Time Step (index)', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Spatial Std over Time', fontsize=10)

    # X 轴刻度（均匀采样）
    n_ticks = min(10, len(x))
    tick_indices = [int(i * (len(x) - 1) / max(n_ticks - 1, 1)) for i in range(n_ticks)]
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels([filenames[i][:10] for i in tick_indices], rotation=30, fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def plot_distribution_histogram(
    npy_files: List[str],
    var_name: str,
    split: str,
    out_path: str,
    n_samples: int = 50,
    n_bins: int = 60
) -> None:
    """
    Generate a value distribution histogram by sampling up to n_samples frames.

    Args:
        npy_files: Sorted list of NPY file paths for this variable/split.
        var_name: Variable name (used in title and filename).
        split: Dataset split name (train/valid/test).
        out_path: Output PNG path.
        n_samples: Maximum number of frames to sample (default 50).
        n_bins: Number of histogram bins (default 60).
    """
    if not npy_files:
        return

    # Uniformly sample up to n_samples files
    n = min(n_samples, len(npy_files))
    if n == 0:
        return
    indices = [int(i * (len(npy_files) - 1) / max(n - 1, 1)) for i in range(n)]
    selected = [npy_files[i] for i in indices]

    # Collect all finite values from sampled frames
    all_values: List[np.ndarray] = []
    for fpath in selected:
        arr = _safe_load(fpath)
        if arr is None:
            continue
        flat = arr.ravel()
        finite_mask = np.isfinite(flat)
        if np.any(finite_mask):
            all_values.append(flat[finite_mask])

    if not all_values:
        return

    combined = np.concatenate(all_values)
    if combined.size == 0:
        return

    # Compute percentile markers
    p5 = float(np.percentile(combined, 5))
    p95 = float(np.percentile(combined, 95))

    # Build histogram
    counts, bin_edges = np.histogram(combined, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(bin_centers, counts, width=bin_width * 0.9,
           color='steelblue', alpha=0.75, edgecolor='none')

    # Percentile vertical lines
    ax.axvline(p5, color='darkorange', linestyle='--', linewidth=1.2,
               label=f'P5 = {p5:.3g}')
    ax.axvline(p95, color='crimson', linestyle='--', linewidth=1.2,
               label=f'P95 = {p95:.3g}')

    ax.set_title(f'{var_name}  |  {split}  — value distribution', fontsize=12)
    ax.set_xlabel('Value', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ========================================
# 主函数
# ========================================

def visualize_forecast(dataset_root: str, splits: List[str], out_dir: str) -> dict:
    """
    生成预报数据可视化图表

    Args:
        dataset_root: 数据集根目录（包含 train/valid/test 子目录）
        splits: 要可视化的划分列表
        out_dir: 输出目录

    Returns:
        结果字典
    """
    result = {
        "status": "success",
        "dataset_root": dataset_root,
        "out_dir": out_dir,
        "generated_files": [],
        "warnings": [],
        "errors": []
    }

    # 从 var_names.json 获取变量列表和坐标配置
    static_dir = os.path.join(dataset_root, 'static_variables')
    var_names_path = os.path.join(dataset_root, 'var_names.json')
    dyn_vars = None
    lon_var_name = None
    lat_var_name = None
    spatial_shape = None
    if os.path.exists(var_names_path):
        try:
            with open(var_names_path, 'r', encoding='utf-8') as f:
                var_names_data = json.load(f)
            dyn_vars = var_names_data.get('dynamic', [])
            lon_var_name = var_names_data.get('lon_var')
            lat_var_name = var_names_data.get('lat_var')
            spatial_shape = var_names_data.get('spatial_shape')
        except Exception as e:
            result["warnings"].append(f"读取 var_names.json 失败: {e}")

    # 推断目标 HxW 形状（用于精确匹配坐标数组）
    target_hw = None
    if spatial_shape and len(spatial_shape) >= 2:
        target_hw = tuple(spatial_shape[-2:])

    # 加载经纬度坐标（优先使用 var_names.json 配置 + 形状匹配）
    lon_arr = _load_coord_var(static_dir, 'lon', preferred_var=lon_var_name, target_shape=target_hw)
    lat_arr = _load_coord_var(static_dir, 'lat', preferred_var=lat_var_name, target_shape=target_hw)

    generated_count = 0

    for split in splits:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.isdir(split_dir):
            result["warnings"].append(f"split 目录不存在: {split_dir}")
            continue

        # 自动检测变量目录
        if dyn_vars:
            var_dirs = [(v, os.path.join(split_dir, v)) for v in dyn_vars
                        if os.path.isdir(os.path.join(split_dir, v))]
        else:
            var_dirs = [(d, os.path.join(split_dir, d))
                        for d in os.listdir(split_dir)
                        if os.path.isdir(os.path.join(split_dir, d))]

        for var_name, var_dir in var_dirs:
            npy_files = _get_sorted_npy_files(var_dir)
            if not npy_files:
                result["warnings"].append(f"变量目录为空: {split}/{var_name}")
                continue

            split_out_dir = os.path.join(out_dir, split)

            # 1. 样本帧图
            frames_path = os.path.join(split_out_dir, f'{var_name}_frames.png')
            try:
                plot_sample_frames(npy_files, var_name, split, lon_arr, lat_arr, frames_path)
                result["generated_files"].append(frames_path)
                generated_count += 1
                print(f"  ✅ {split}/{var_name}_frames.png", file=sys.stderr)
            except Exception as e:
                result["warnings"].append(f"生成 {split}/{var_name}_frames.png 失败: {e}")

            # 2. 时序统计图（训练集全量，其他 split 限制前 500）
            max_ts = 2000 if split == 'train' else 500
            stats_path = os.path.join(split_out_dir, f'{var_name}_timeseries.png')
            try:
                plot_time_series_stats(npy_files, var_name, split, stats_path, max_files=max_ts)
                result["generated_files"].append(stats_path)
                generated_count += 1
                print(f"  ✅ {split}/{var_name}_timeseries.png", file=sys.stderr)
            except Exception as e:
                result["warnings"].append(f"生成 {split}/{var_name}_timeseries.png 失败: {e}")

            # 3. 分布直方图
            hist_path = os.path.join(split_out_dir, f'{var_name}_distribution.png')
            try:
                plot_distribution_histogram(npy_files, var_name, split, hist_path)
                result["generated_files"].append(hist_path)
                generated_count += 1
                print(f"  ✅ {split}/{var_name}_distribution.png", file=sys.stderr)
            except Exception as e:
                result["warnings"].append(f"生成 {split}/{var_name}_distribution.png 失败: {e}")

    result["message"] = f"可视化完成，共生成 {generated_count} 张图片"
    print(f"✅ {result['message']}", file=sys.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(description="预报数据可视化")
    parser.add_argument("--dataset_root", required=True, help="数据集根目录")
    parser.add_argument("--splits", nargs='+', default=['train', 'valid', 'test'],
                        help="要可视化的 split 列表")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    args = parser.parse_args()

    result = visualize_forecast(args.dataset_root, args.splits, args.out_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
