#!/usr/bin/env python3
"""
visualize_check.py - 数据处理后的可视化检查脚本

@author liuzhengyang
@contributor leizheng, kongzhiquan
@date 2026-02-04
@version 3.1.2

功能:
- 对比 HR 和 LR 数据的可视化
- 每个变量抽取 1 帧进行检查
- 支持 2D/3D/4D 数据格式
- 自动加载经纬度坐标显示真实地理范围
- 【新增】统计分布图：均值、方差、直方图
- 输出到 visualisation_data_process/ 目录

用法:
    python visualize_check.py --dataset_root /path/to/dataset

目录结构 (v3.1.0 新格式):
    dataset_root/
    ├── train/
    │   ├── raw/   ← 区域裁剪后的原始数据（两步裁剪模式）
    │   ├── hr/
    │   └── lr/
    ├── valid/
    │   ├── hr/
    │   └── lr/
    ├── test/
    │   ├── hr/
    │   └── lr/
    ├── static_variables/
    │   ├── raw/   ← 区域裁剪后的静态变量
    │   ├── hr/    ← 尺寸裁剪后的静态变量
    │   └── lr/    ← 下采样后的静态变量
    └── visualisation_data_process/   ← 图片输出到这里
        ├── train/
        │   ├── {var}_compare.png      # HR vs LR 空间对比
        │   └── {var}_statistics.png   # 统计分布图
        ├── valid/
        ├── test/
        └── statistics_summary.png     # 全局统计汇总

Changelog:
    - 2026-02-05 kongzhiquan v3.1.2: 修复单时间步 X 轴显示异常
        - 当只有 1 个时间步时，X 轴显示 "t=0" 而非异常范围
        - 添加 xlim 边界处理避免 matplotlib 自动缩放问题
    - 2026-02-05 kongzhiquan v3.1.1: 修复坐标文件搜索路径
        - 支持 static_variables/hr 和 static_variables/lr 子目录
        - 修复 v3.1.0 区域裁剪后坐标不显示的问题
    - 2026-02-04 kongzhiquan v3.0.0: 新增统计分布图
        - 新增均值/方差时序图
        - 新增 HR/LR 数据值直方图对比
        - 新增全局统计汇总图
        - 重命名原对比图为 {var}_compare.png
    - 2026-02-04 leizheng v2.0.0: 支持逐时间步文件格式
        - 适配 convert_npy.py v2.9.0 的新目录结构
        - 自动检测新格式（hr/uo/）或旧格式（hr/uo.npy）
        - 新格式下取第一个文件用于可视化
    - 2026-02-04 leizheng v1.4.0: 修复坐标文件匹配
        - 支持带编号前缀的文件名（如 20_latitude.npy）
        - 使用 glob 模式匹配多种命名格式
        - 添加调试日志显示加载的坐标文件
    - 2026-02-04 leizheng v1.3.0: 支持经纬度坐标
        - 自动加载 static_variables/ 中的 lat.npy/lon.npy
        - 坐标轴显示真实经纬度 (°E, °N)
        - 无坐标时回退到像素坐标
    - 2026-02-04 leizheng v1.2.0: 添加坐标轴标签
    - 2026-02-03 leizheng v1.1.0: 鲁棒性修复
        - 修复 HR/LR 维度不一致导致的错误
        - 修复 HR/LR 时间步不同导致的越界
        - 修复空数据导致的越界
        - HR 和 LR 分别提取切片，互不影响
    - 2026-02-03 leizheng v1.0.0: 适配 Ocean-Agent-SDK 目录结构
        - 新增 --dataset_root 参数
        - 修改为 train/valid/test + hr/lr 目录结构
        - 每个变量只抽 1 帧检查
        - 添加 4D 数据支持
    - 2026-02-03 liuzhengyang: 原始版本
        - HR vs LR 对比绘图
        - NaN 区域灰色背景显示
"""

import os
import sys
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'extract_2d_slice',
    'plot_and_save',
    'plot_statistics',
    'plot_summary_statistics',
    'load_coordinates',
    'process_split',
]


def extract_2d_slice(data: np.ndarray, name: str) -> tuple:
    """
    从多维数组中提取 2D 切片用于可视化

    Args:
        data: 输入数组
        name: 数据名称（用于日志）

    Returns:
        (slice_2d, slice_info) 或 (None, error_msg)
    """
    ndim = data.ndim

    # 检查空数据
    if data.size == 0 or (ndim >= 1 and data.shape[0] == 0):
        return None, "数据为空"

    if ndim == 2:
        # (H, W)
        return data, ""

    elif ndim == 3:
        # (T, H, W)
        t = data.shape[0]
        idx = t // 2
        return data[idx], f"t={idx}/{t}"

    elif ndim == 4:
        # (T, D, H, W)
        t, d = data.shape[:2]
        t_idx = t // 2
        d_idx = 0
        return data[t_idx, d_idx], f"t={t_idx}/{t}, d={d_idx}/{d}"

    else:
        return None, f"不支持的维度: {ndim}D"


def plot_and_save(
    hr_data: np.ndarray,
    lr_data: np.ndarray,
    save_path: str,
    var_name: str,
    hr_lat: np.ndarray = None,
    hr_lon: np.ndarray = None,
    lr_lat: np.ndarray = None,
    lr_lon: np.ndarray = None
):
    """
    绘制 HR vs LR 对比图并保存

    Args:
        hr_data: HR 数据
        lr_data: LR 数据
        save_path: 保存路径
        var_name: 变量名（用于标题）
        hr_lat: HR 纬度数组 (可选)
        hr_lon: HR 经度数组 (可选)
        lr_lat: LR 纬度数组 (可选)
        lr_lon: LR 经度数组 (可选)
    """
    # 绘图参数
    cmap = 'viridis'
    bg_color = 'lightgray'  # NaN 区域显示为灰色

    # 分别提取 HR 和 LR 的 2D 切片
    hr_slice, hr_info = extract_2d_slice(hr_data, f"HR:{var_name}")
    lr_slice, lr_info = extract_2d_slice(lr_data, f"LR:{var_name}")

    # 检查提取是否成功
    if hr_slice is None:
        print(f"[Skip] HR 数据无法提取: {var_name} - {hr_info}")
        return

    if lr_slice is None:
        print(f"[Skip] LR 数据无法提取: {var_name} - {lr_info}")
        return

    # 检查提取后是否为 2D
    if hr_slice.ndim != 2:
        print(f"[Skip] HR 切片不是 2D: {var_name}, shape={hr_slice.shape}")
        return

    if lr_slice.ndim != 2:
        print(f"[Skip] LR 切片不是 2D: {var_name}, shape={lr_slice.shape}")
        return

    # 构建切片信息字符串
    hr_slice_info = f" ({hr_info})" if hr_info else ""
    lr_slice_info = f" ({lr_info})" if lr_info else ""

    # 计算 extent（如果有坐标信息）
    hr_extent = None
    lr_extent = None
    use_geo_coords = False

    if hr_lat is not None and hr_lon is not None:
        try:
            # extent = [left, right, bottom, top] = [lon_min, lon_max, lat_min, lat_max]
            hr_extent = [hr_lon.min(), hr_lon.max(), hr_lat.min(), hr_lat.max()]
            use_geo_coords = True
        except Exception:
            pass

    if lr_lat is not None and lr_lon is not None:
        try:
            lr_extent = [lr_lon.min(), lr_lon.max(), lr_lat.min(), lr_lat.max()]
            use_geo_coords = True
        except Exception:
            pass

    # 坐标轴标签
    if use_geo_coords:
        xlabel = "Longitude (°E)"
        ylabel = "Latitude (°N)"
    else:
        xlabel = "W (pixels)"
        ylabel = "H (pixels)"

    # 创建画布: 1行2列
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # HR
    ax_hr = axes[0]
    ax_hr.set_facecolor(bg_color)
    im_hr = ax_hr.imshow(hr_slice, cmap=cmap, origin='lower', extent=hr_extent, aspect='auto' if hr_extent else 'equal')
    ax_hr.set_title(f"HR: {var_name}{hr_slice_info}\nshape: {hr_data.shape}")
    ax_hr.set_xlabel(xlabel)
    ax_hr.set_ylabel(ylabel)
    plt.colorbar(im_hr, ax=ax_hr, fraction=0.046, pad=0.04)

    # LR
    ax_lr = axes[1]
    ax_lr.set_facecolor(bg_color)
    im_lr = ax_lr.imshow(lr_slice, cmap=cmap, origin='lower', extent=lr_extent, aspect='auto' if lr_extent else 'equal')
    ax_lr.set_title(f"LR: {var_name}{lr_slice_info}\nshape: {lr_data.shape}")
    ax_lr.set_xlabel(xlabel)
    ax_lr.set_ylabel(ylabel)
    plt.colorbar(im_lr, ax=ax_lr, fraction=0.046, pad=0.04)

    # 布局调整
    plt.tight_layout()

    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[Saved] {save_path}")


def plot_statistics(
    hr_data_list: list,
    lr_data_list: list,
    save_path: str,
    var_name: str
):
    """
    绘制统计分布图：均值/方差时序 + 直方图

    Args:
        hr_data_list: HR 数据列表（每个时间步的 2D 数组）
        lr_data_list: LR 数据列表（每个时间步的 2D 数组）
        save_path: 保存路径
        var_name: 变量名（用于标题）
    """
    # 计算每个时间步的统计量
    hr_means = []
    hr_stds = []
    lr_means = []
    lr_stds = []

    for hr_data in hr_data_list:
        valid_hr = hr_data[~np.isnan(hr_data)]
        if len(valid_hr) > 0:
            hr_means.append(np.mean(valid_hr))
            hr_stds.append(np.std(valid_hr))
        else:
            hr_means.append(np.nan)
            hr_stds.append(np.nan)

    for lr_data in lr_data_list:
        valid_lr = lr_data[~np.isnan(lr_data)]
        if len(valid_lr) > 0:
            lr_means.append(np.mean(valid_lr))
            lr_stds.append(np.std(valid_lr))
        else:
            lr_means.append(np.nan)
            lr_stds.append(np.nan)

    hr_means = np.array(hr_means)
    hr_stds = np.array(hr_stds)
    lr_means = np.array(lr_means)
    lr_stds = np.array(lr_stds)

    # 收集所有有效值用于直方图
    hr_all_values = np.concatenate([d[~np.isnan(d)].flatten() for d in hr_data_list if d.size > 0])
    lr_all_values = np.concatenate([d[~np.isnan(d)].flatten() for d in lr_data_list if d.size > 0])

    # 创建画布: 2行2列
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    time_steps = np.arange(len(hr_means))
    n_steps = len(hr_means)

    # 1. 均值时序图 (左上)
    ax1 = axes[0, 0]
    ax1.plot(time_steps, hr_means, 'b-', label='HR Mean', linewidth=1.5, alpha=0.8)
    ax1.plot(time_steps, lr_means, 'r--', label='LR Mean', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Mean Value')
    ax1.set_title(f'{var_name} - Mean over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # 修复单时间步 X 轴显示异常（v3.1.2）
    if n_steps == 1:
        ax1.set_xlim(-0.5, 0.5)
        ax1.set_xticks([0])
        ax1.set_xticklabels(['t=0'])
    elif n_steps > 1:
        ax1.set_xlim(-0.5, n_steps - 0.5)

    # 2. 方差/标准差时序图 (右上)
    ax2 = axes[0, 1]
    ax2.plot(time_steps, hr_stds, 'b-', label='HR Std', linewidth=1.5, alpha=0.8)
    ax2.plot(time_steps, lr_stds, 'r--', label='LR Std', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title(f'{var_name} - Std Dev over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # 修复单时间步 X 轴显示异常（v3.1.2）
    if n_steps == 1:
        ax2.set_xlim(-0.5, 0.5)
        ax2.set_xticks([0])
        ax2.set_xticklabels(['t=0'])
    elif n_steps > 1:
        ax2.set_xlim(-0.5, n_steps - 0.5)

    # 3. HR 直方图 (左下)
    ax3 = axes[1, 0]
    if len(hr_all_values) > 0:
        # 计算合适的 bin 范围
        hr_min, hr_max = np.percentile(hr_all_values, [1, 99])
        bins = np.linspace(hr_min, hr_max, 50)
        ax3.hist(hr_all_values, bins=bins, color='blue', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax3.axvline(np.mean(hr_all_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(hr_all_values):.4f}')
        ax3.axvline(np.median(hr_all_values), color='green', linestyle=':', linewidth=2, label=f'Median: {np.median(hr_all_values):.4f}')
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'{var_name} HR - Value Distribution\nStd: {np.std(hr_all_values):.4f}')
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No valid HR data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title(f'{var_name} HR - Value Distribution')

    # 4. LR 直方图 (右下)
    ax4 = axes[1, 1]
    if len(lr_all_values) > 0:
        # 计算合适的 bin 范围
        lr_min, lr_max = np.percentile(lr_all_values, [1, 99])
        bins = np.linspace(lr_min, lr_max, 50)
        ax4.hist(lr_all_values, bins=bins, color='orange', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax4.axvline(np.mean(lr_all_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(lr_all_values):.4f}')
        ax4.axvline(np.median(lr_all_values), color='green', linestyle=':', linewidth=2, label=f'Median: {np.median(lr_all_values):.4f}')
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'{var_name} LR - Value Distribution\nStd: {np.std(lr_all_values):.4f}')
        ax4.legend(fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No valid LR data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title(f'{var_name} LR - Value Distribution')

    # 布局调整
    plt.tight_layout()

    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[Saved] {save_path}")

    # 返回统计信息用于汇总
    return {
        'var_name': var_name,
        'hr_mean': np.nanmean(hr_means),
        'hr_std': np.nanmean(hr_stds),
        'lr_mean': np.nanmean(lr_means),
        'lr_std': np.nanmean(lr_stds),
        'n_timesteps': len(hr_means)
    }


def plot_summary_statistics(all_stats: list, save_path: str, dataset_name: str = ""):
    """
    绘制全局统计汇总图

    Args:
        all_stats: 所有变量的统计信息列表
        save_path: 保存路径
        dataset_name: 数据集名称
    """
    if not all_stats:
        print("[Warning] 没有统计数据可汇总")
        return

    var_names = [s['var_name'] for s in all_stats]
    hr_means = [s['hr_mean'] for s in all_stats]
    hr_stds = [s['hr_std'] for s in all_stats]
    lr_means = [s['lr_mean'] for s in all_stats]
    lr_stds = [s['lr_std'] for s in all_stats]

    x = np.arange(len(var_names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 均值对比条形图
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, hr_means, width, label='HR', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, lr_means, width, label='LR', color='coral', alpha=0.8)
    ax1.set_xlabel('Variable')
    ax1.set_ylabel('Mean Value')
    ax1.set_title(f'{dataset_name} - Mean Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(var_names, rotation=0, ha='center')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 在条形图上添加数值标签（根据正负值调整位置）
    for bar, val in zip(bars1, hr_means):
        if not np.isnan(val):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            ax1.text(bar.get_x() + bar.get_width()/2, height, f'{val:.4g}',
                    ha='center', va=va, fontsize=7, rotation=0)
    for bar, val in zip(bars2, lr_means):
        if not np.isnan(val):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            ax1.text(bar.get_x() + bar.get_width()/2, height, f'{val:.4g}',
                    ha='center', va=va, fontsize=7, rotation=0)

    # 2. 标准差对比条形图
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, hr_stds, width, label='HR', color='steelblue', alpha=0.8)
    bars4 = ax2.bar(x + width/2, lr_stds, width, label='LR', color='coral', alpha=0.8)
    ax2.set_xlabel('Variable')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title(f'{dataset_name} - Std Dev Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(var_names, rotation=0, ha='center')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 在条形图上添加数值标签（根据正负值调整位置）
    for bar, val in zip(bars3, hr_stds):
        if not np.isnan(val):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            ax2.text(bar.get_x() + bar.get_width()/2, height, f'{val:.4g}',
                    ha='center', va=va, fontsize=7, rotation=0)
    for bar, val in zip(bars4, lr_stds):
        if not np.isnan(val):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            ax2.text(bar.get_x() + bar.get_width()/2, height, f'{val:.4g}',
                    ha='center', va=va, fontsize=7, rotation=0)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[Saved] {save_path}")


def load_coordinates(dataset_root: str, subdir: str = None) -> tuple:
    """
    尝试加载经纬度坐标

    Args:
        dataset_root: 数据集根目录
        subdir: 子目录 (如 'static_variables', 'lr_static_variables')

    Returns:
        (lat, lon) 或 (None, None)
    """
    # 可能的坐标文件名（支持带编号前缀的格式，如 00_lon.npy, 10_lat.npy）
    # 使用 glob 模式匹配
    lat_patterns = [
        'lat.npy', 'latitude.npy', 'LAT.npy', 'LATITUDE.npy',
        'lat_rho.npy', 'lat_psi.npy', 'lat_u.npy', 'lat_v.npy',
        '*_lat.npy', '*_latitude.npy', '*_lat_rho.npy'
    ]
    lon_patterns = [
        'lon.npy', 'longitude.npy', 'LON.npy', 'LONGITUDE.npy',
        'lon_rho.npy', 'lon_psi.npy', 'lon_u.npy', 'lon_v.npy',
        '*_lon.npy', '*_longitude.npy', '*_lon_rho.npy'
    ]

    # 搜索路径（v3.1.1 更新：支持新的 hr/lr 子目录结构）
    search_dirs = []
    if subdir:
        search_dirs.append(os.path.join(dataset_root, subdir))
        # 新增：支持 static_variables/hr 和 static_variables/lr 结构
        search_dirs.append(os.path.join(dataset_root, subdir, 'hr'))
        search_dirs.append(os.path.join(dataset_root, subdir, 'lr'))
    search_dirs.append(os.path.join(dataset_root, 'static_variables'))
    search_dirs.append(os.path.join(dataset_root, 'static_variables', 'hr'))
    search_dirs.append(os.path.join(dataset_root, 'static_variables', 'lr'))
    search_dirs.append(dataset_root)

    lat = None
    lon = None

    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        # 尝试加载 lat
        if lat is None:
            for lat_pattern in lat_patterns:
                # 使用 glob 模式匹配
                lat_matches = glob.glob(os.path.join(search_dir, lat_pattern))
                if lat_matches:
                    lat_path = lat_matches[0]  # 取第一个匹配的文件
                    try:
                        lat = np.load(lat_path)
                        print(f"  加载纬度坐标: {lat_path}")
                        # 如果是2D数组，取第一列（假设lat沿H方向变化）
                        if lat.ndim == 2:
                            lat = lat[:, 0]
                        break
                    except Exception as e:
                        print(f"  加载纬度失败: {lat_path}, {e}")

        # 尝试加载 lon
        if lon is None:
            for lon_pattern in lon_patterns:
                # 使用 glob 模式匹配
                lon_matches = glob.glob(os.path.join(search_dir, lon_pattern))
                if lon_matches:
                    lon_path = lon_matches[0]  # 取第一个匹配的文件
                    try:
                        lon = np.load(lon_path)
                        print(f"  加载经度坐标: {lon_path}")
                        # 如果是2D数组，取第一行（假设lon沿W方向变化）
                        if lon.ndim == 2:
                            lon = lon[0, :]
                        break
                    except Exception as e:
                        print(f"  加载经度失败: {lon_path}, {e}")

        if lat is not None and lon is not None:
            break

    return lat, lon


def process_split(dataset_root: str, split: str, out_dir: str, hr_lat: np.ndarray = None, hr_lon: np.ndarray = None, lr_lat: np.ndarray = None, lr_lon: np.ndarray = None):
    """
    处理单个数据集划分

    Args:
        dataset_root: 数据集根目录
        split: 划分名称 (train/valid/test)
        out_dir: 输出目录
        hr_lat: HR 纬度数组 (可选)
        hr_lon: HR 经度数组 (可选)
        lr_lat: LR 纬度数组 (可选)
        lr_lon: LR 经度数组 (可选)

    Returns:
        list: 该 split 的所有变量统计信息
    """
    hr_dir = os.path.join(dataset_root, split, 'hr')
    lr_dir = os.path.join(dataset_root, split, 'lr')

    all_stats = []

    if not os.path.exists(hr_dir):
        print(f"[Warning] HR 目录不存在，跳过: {hr_dir}")
        return all_stats

    if not os.path.exists(lr_dir):
        print(f"[Warning] LR 目录不存在，跳过: {lr_dir}")
        return all_stats

    # v2.0.0: 检查是新格式（子目录）还是旧格式（直接 .npy 文件）
    var_dirs = [d for d in os.listdir(hr_dir) if os.path.isdir(os.path.join(hr_dir, d))]

    if var_dirs:
        # 新格式: hr/uo/000000.npy
        print(f"\n处理 {split} 数据集 [新格式] ({len(var_dirs)} 个变量)...")

        for var_name in sorted(var_dirs):
            var_hr_dir = os.path.join(hr_dir, var_name)
            var_lr_dir = os.path.join(lr_dir, var_name)

            if not os.path.exists(var_lr_dir):
                print(f"[Skip] LR 变量目录不存在: {var_lr_dir}")
                continue

            # 获取所有文件
            hr_files = sorted(glob.glob(os.path.join(var_hr_dir, '*.npy')))
            lr_files = sorted(glob.glob(os.path.join(var_lr_dir, '*.npy')))

            if not hr_files or not lr_files:
                print(f"[Skip] {var_name} 目录为空")
                continue

            try:
                # 1. 生成空间对比图（取中间时间步）
                mid_idx = len(hr_files) // 2
                hr_data = np.load(hr_files[mid_idx])
                lr_data = np.load(lr_files[min(mid_idx, len(lr_files)-1)])

                compare_path = os.path.join(out_dir, split, f"{var_name}_compare.png")
                plot_and_save(hr_data, lr_data, compare_path, var_name, hr_lat, hr_lon, lr_lat, lr_lon)

                # 2. 生成统计分布图（加载所有时间步）
                print(f"  计算 {var_name} 统计量 ({len(hr_files)} 个时间步)...")
                hr_data_list = [np.load(f) for f in hr_files]
                lr_data_list = [np.load(f) for f in lr_files]

                stats_path = os.path.join(out_dir, split, f"{var_name}_statistics.png")
                stats = plot_statistics(hr_data_list, lr_data_list, stats_path, var_name)
                if stats:
                    all_stats.append(stats)

            except Exception as e:
                print(f"[Error] 处理 {var_name} 失败: {e}")
    else:
        # 旧格式: hr/uo.npy
        npy_files = glob.glob(os.path.join(hr_dir, '*.npy'))

        if not npy_files:
            print(f"[Warning] HR 目录为空: {hr_dir}")
            return all_stats

        print(f"\n处理 {split} 数据集 [旧格式] ({len(npy_files)} 个变量)...")

        for hr_path in sorted(npy_files):
            filename = os.path.basename(hr_path)
            var_name = os.path.splitext(filename)[0]
            lr_path = os.path.join(lr_dir, filename)

            # 检查 LR 文件是否存在
            if not os.path.exists(lr_path):
                print(f"[Skip] LR 文件不存在: {lr_path}")
                continue

            try:
                hr_data = np.load(hr_path)
                lr_data = np.load(lr_path)

                # 1. 生成空间对比图
                compare_path = os.path.join(out_dir, split, f"{var_name}_compare.png")
                plot_and_save(hr_data, lr_data, compare_path, var_name, hr_lat, hr_lon, lr_lat, lr_lon)

                # 2. 生成统计分布图（旧格式：整个数组作为时序）
                # 假设旧格式是 (T, H, W)，将每个时间步作为一个数据
                if hr_data.ndim >= 3:
                    hr_data_list = [hr_data[t] for t in range(hr_data.shape[0])]
                    lr_data_list = [lr_data[t] for t in range(lr_data.shape[0])]
                else:
                    # 2D 数据，作为单个时间步
                    hr_data_list = [hr_data]
                    lr_data_list = [lr_data]

                stats_path = os.path.join(out_dir, split, f"{var_name}_statistics.png")
                stats = plot_statistics(hr_data_list, lr_data_list, stats_path, var_name)
                if stats:
                    all_stats.append(stats)

            except Exception as e:
                print(f"[Error] 处理 {var_name} 失败: {e}")

    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="数据处理后的可视化检查 - HR vs LR 对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查所有划分
  python visualize_check.py --dataset_root /path/to/dataset

  # 只检查 train
  python visualize_check.py --dataset_root /path/to/dataset --splits train
        """
    )

    parser.add_argument(
        '--dataset_root',
        required=True,
        type=str,
        help='数据集根目录（包含 train/valid/test 子目录）'
    )

    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'valid', 'test'],
        help='要检查的数据集划分（默认: train valid test）'
    )

    parser.add_argument(
        '--out_dir',
        type=str,
        default=None,
        help='输出目录（默认: dataset_root/visualisation_data_process/）'
    )

    args = parser.parse_args()

    # 验证参数
    if not os.path.exists(args.dataset_root):
        print(f"[Error] 数据集根目录不存在: {args.dataset_root}")
        sys.exit(1)

    # 输出目录
    out_dir = args.out_dir or os.path.join(args.dataset_root, 'visualisation_data_process')

    # 打印配置
    print("=" * 60)
    print("数据处理可视化检查")
    print("=" * 60)
    print(f"数据集根目录: {args.dataset_root}")
    print(f"检查划分: {', '.join(args.splits)}")
    print(f"输出目录: {out_dir}")
    print("=" * 60)

    # 尝试加载坐标
    hr_lat, hr_lon = load_coordinates(args.dataset_root, 'static_variables')
    lr_lat, lr_lon = load_coordinates(args.dataset_root, 'lr_static_variables')

    if hr_lat is not None and hr_lon is not None:
        print(f"HR 坐标已加载: lat shape={hr_lat.shape}, lon shape={hr_lon.shape}")
        print(f"  经度范围: {hr_lon.min():.2f}°E ~ {hr_lon.max():.2f}°E")
        print(f"  纬度范围: {hr_lat.min():.2f}°N ~ {hr_lat.max():.2f}°N")
    else:
        print("HR 坐标未找到，将使用像素坐标")

    if lr_lat is not None and lr_lon is not None:
        print(f"LR 坐标已加载: lat shape={lr_lat.shape}, lon shape={lr_lon.shape}")
    else:
        print("LR 坐标未找到，将使用 HR 坐标范围（如果可用）")
        # 如果没有 LR 坐标但有 HR 坐标，使用 HR 坐标（范围相同）
        if hr_lat is not None and hr_lon is not None:
            lr_lat, lr_lon = hr_lat, hr_lon

    print("=" * 60)

    # 处理各个划分并收集统计信息
    all_split_stats = {}
    for split in args.splits:
        stats = process_split(args.dataset_root, split, out_dir, hr_lat, hr_lon, lr_lat, lr_lon)
        if stats:
            all_split_stats[split] = stats

    # 生成全局统计汇总图（合并所有 split 的第一个变量统计）
    if all_split_stats:
        # 按变量合并统计
        var_stats_combined = {}
        for split, stats_list in all_split_stats.items():
            for s in stats_list:
                var_name = s['var_name']
                if var_name not in var_stats_combined:
                    var_stats_combined[var_name] = s
                # 可以选择合并或只用第一个 split 的数据

        if var_stats_combined:
            summary_path = os.path.join(out_dir, 'statistics_summary.png')
            plot_summary_statistics(
                list(var_stats_combined.values()),
                summary_path,
                dataset_name=os.path.basename(args.dataset_root)
            )

    print("\n" + "=" * 60)
    print("可视化检查完成！")
    print(f"图片保存在: {out_dir}")
    print("\n生成的图片类型：")
    print("  - {var}_compare.png    : HR vs LR 空间对比图")
    print("  - {var}_statistics.png : 均值/方差时序 + 直方图")
    print("  - statistics_summary.png : 全局统计汇总")
    print("=" * 60)


if __name__ == "__main__":
    main()
