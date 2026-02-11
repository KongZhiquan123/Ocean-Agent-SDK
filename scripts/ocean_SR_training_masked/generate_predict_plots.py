#!/usr/bin/env python3
"""
@file generate_predict_plots.py

@description Predict 输出可视化脚本 — 对 predictions/ 目录中的 SR NPY 文件生成
    LR / SR / HR / |SR-HR| 对比图。支持经纬度坐标显示和陆地掩码。

@author Leizheng
@date 2026-02-11
@version 1.1.0

@changelog
    - 2026-02-11 Leizheng: v1.1.0 LR 面板经纬度坐标支持
        - load_static_coords() 新增加载 LR 经纬度（static_variables/lr/）
        - plot_predict_comparison() LR 面板使用独立 LR 坐标网格
        - 与 plot_sample.py 坐标处理逻辑对齐
    - 2026-02-11 Leizheng: v1.0.1 修复通道不匹配 IndexError + 移除死代码 --scale
        - load_sample_data() 缺失变量文件填 NaN 占位，保持通道对齐
        - plot_predict_comparison() 安全索引，防止 hr/lr 通道数少于 sr
        - 移除未使用的 --scale 参数
    - 2026-02-11 Leizheng: v1.0.0 初始版本
        - 从 predictions/*.npy 读取 SR 输出
        - 从 dataset_root/test/hr|lr/ 读取对应 HR/LR
        - 从 static_variables/ 读取经纬度
        - 生成 pcolormesh 对比图（与 plot_sample.py 风格一致）
        - 输出 __result__ JSON 标记供 TypeScript 解析

用法:
    python generate_predict_plots.py \
        --log_dir /path/to/training_output \
        --dataset_root /path/to/dataset \
        --dyn_vars temp,salt \
        --max_samples 4

输出: log_dir/plots/predict_comparison_XX.png
"""

import os
import sys
import json
import argparse
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _make_grid_edges(centers):
    """从中心点坐标构造 pcolormesh 所需的边界坐标（N+1 个点）。"""
    if centers.ndim != 1 or len(centers) < 2:
        return centers
    mid = 0.5 * (centers[:-1] + centers[1:])
    left = 2 * centers[0] - mid[0]
    right = 2 * centers[-1] - mid[-1]
    return np.concatenate([[left], mid, [right]])


def load_static_coords(dataset_root):
    """从 static_variables/ 加载经纬度坐标（HR + LR）。

    HR 坐标从 static_variables/ 目录加载。
    LR 坐标从 static_variables/lr/ 子目录加载（如果存在）。
    """
    static_dir = os.path.join(dataset_root, 'static_variables')
    result = {'lon_hr': None, 'lat_hr': None, 'lon_lr': None, 'lat_lr': None}
    if not os.path.isdir(static_dir):
        return result

    # HR 坐标
    for fname in os.listdir(static_dir):
        fpath = os.path.join(static_dir, fname)
        if not os.path.isfile(fpath) or not fname.endswith('.npy'):
            continue
        lower = fname.lower()
        if 'lon' in lower:
            result['lon_hr'] = np.load(fpath)
        elif 'lat' in lower:
            result['lat_hr'] = np.load(fpath)

    # LR 坐标：先尝试 static_variables/lr/ 子目录
    lr_dir = os.path.join(static_dir, 'lr')
    if os.path.isdir(lr_dir):
        for fname in os.listdir(lr_dir):
            fpath = os.path.join(lr_dir, fname)
            if not fname.endswith('.npy'):
                continue
            lower = fname.lower()
            if 'lon' in lower:
                result['lon_lr'] = np.load(fpath)
            elif 'lat' in lower:
                result['lat_lr'] = np.load(fpath)

    return result


def load_sample_data(dataset_root, dyn_vars, filename):
    """从 dataset_root/test/ 加载 HR 和 LR 数据，按变量拼接通道。

    缺失的变量文件用 None 占位，保证返回的通道数与 dyn_vars 长度一致。

    Returns:
        dict with:
            'hr' [H, W, C] numpy array (or None if all missing)
            'lr' [h, w, C] numpy array (or None if all missing)
            'hr_valid' list[bool] — 每个通道是否有实际数据（非 NaN 填充）
    """
    hr_list, lr_list = [], []
    for var in dyn_vars:
        hr_path = os.path.join(dataset_root, 'test', 'hr', var, f'{filename}.npy')
        lr_path = os.path.join(dataset_root, 'test', 'lr', var, f'{filename}.npy')
        hr_list.append(np.load(hr_path) if os.path.exists(hr_path) else None)
        lr_list.append(np.load(lr_path) if os.path.exists(lr_path) else None)

    # 记录哪些通道有实际数据
    hr_valid = [h is not None for h in hr_list]

    # 找到第一个非 None 元素获取形状，缺失通道填 NaN
    hr_non_none = [h for h in hr_list if h is not None]
    lr_non_none = [l for l in lr_list if l is not None]

    if hr_non_none:
        ref_shape = hr_non_none[0].shape
        hr_filled = [h if h is not None else np.full(ref_shape, np.nan) for h in hr_list]
        hr = np.stack(hr_filled, axis=-1)  # [H, W, C]
    else:
        hr = None

    if lr_non_none:
        ref_shape = lr_non_none[0].shape
        lr_filled = [l if l is not None else np.full(ref_shape, np.nan) for l in lr_list]
        lr = np.stack(lr_filled, axis=-1)  # [h, w, C]
    else:
        lr = None

    return {'hr': hr, 'lr': lr, 'hr_valid': hr_valid}


def detect_mask_from_hr(hr, hr_valid_mask=None):
    """从 HR 数据检测陆地掩码（NaN 位置为陆地）。

    仅使用实际存在的通道检测 mask，忽略因缺失变量填充的 NaN 通道。

    Args:
        hr: [H, W, C] numpy array
        hr_valid_mask: 长度为 C 的布尔列表，True 表示该通道有实际数据。
            None 表示所有通道都有效。

    Returns:
        mask [H, W] — 1=ocean, 0=land; or None if no NaN in valid channels.
    """
    if hr is None:
        return None
    if hr_valid_mask is not None:
        # 只用有实际数据的通道来检测 mask
        valid_indices = [i for i, v in enumerate(hr_valid_mask) if v]
        if not valid_indices:
            return None
        hr_subset = hr[..., valid_indices]
    else:
        hr_subset = hr
    nan_mask = np.isnan(hr_subset)
    if not nan_mask.any():
        return None
    # 任意有效通道是 NaN 即为陆地
    land = nan_mask.any(axis=-1)  # [H, W]
    return (~land).astype(np.float32)


def plot_predict_comparison(sr, hr, lr, mask, lon_hr, lat_hr, lon_lr, lat_lr,
                            dyn_vars, filename, output_path):
    """绘制单个样本的 LR/SR/HR/Error 对比图。"""
    n_channels = sr.shape[-1]
    col_titles = ['LR (input)', 'SR (prediction)', 'HR (ground truth)', '|SR - HR| error']
    n_cols = 4

    fig, axes = plt.subplots(n_channels, n_cols, figsize=(4.2 * n_cols, 3.6 * n_channels + 0.8))
    if n_channels == 1:
        axes = axes[np.newaxis, :]

    # HR 坐标网格
    has_geo = lon_hr is not None and lat_hr is not None
    lon_hr_grid = lat_hr_grid = None
    hr_shading = 'auto'
    if has_geo:
        if lon_hr.ndim == 1:
            lon_hr_grid = _make_grid_edges(lon_hr)
            lat_hr_grid = _make_grid_edges(lat_hr)
            hr_shading = 'flat'
        else:
            lon_hr_grid = lon_hr
            lat_hr_grid = lat_hr
            hr_shading = 'auto'

    # LR 坐标网格
    has_geo_lr = lon_lr is not None and lat_lr is not None
    lon_lr_grid = lat_lr_grid = None
    lr_shading = 'auto'
    if has_geo_lr:
        if lon_lr.ndim == 1:
            lon_lr_grid = _make_grid_edges(lon_lr)
            lat_lr_grid = _make_grid_edges(lat_lr)
            lr_shading = 'flat'
        else:
            lon_lr_grid = lon_lr
            lat_lr_grid = lat_lr
            lr_shading = 'auto'

    for ch in range(n_channels):
        sr_ch = sr[..., ch]
        hr_ch = hr[..., ch] if (hr is not None and ch < hr.shape[-1]) else np.full_like(sr_ch, np.nan)
        lr_ch = lr[..., ch] if (lr is not None and ch < lr.shape[-1]) else None
        err = np.abs(sr_ch - hr_ch)

        # 统一色标
        vals = [sr_ch]
        if hr is not None:
            vals.append(hr_ch)
        if lr_ch is not None:
            vals.append(lr_ch)
        all_vals = np.concatenate([v.ravel() for v in vals])
        finite = all_vals[np.isfinite(all_vals)]
        if len(finite) > 0:
            vmin, vmax = float(finite.min()), float(finite.max())
        else:
            vmin, vmax = 0, 1

        panels = [lr_ch, sr_ch, hr_ch, err]
        cmaps = ['viridis', 'viridis', 'viridis', 'cividis']
        vmins = [vmin, vmin, vmin, 0]
        err_max = float(err[np.isfinite(err)].max()) if np.any(np.isfinite(err)) and err[np.isfinite(err)].max() > 0 else 1
        vmaxs = [vmax, vmax, vmax, err_max]
        is_lr = [True, False, False, False]

        for col, (panel, cmap, lo, hi, lr_flag) in enumerate(
            zip(panels, cmaps, vmins, vmaxs, is_lr)
        ):
            ax = axes[ch, col]
            if panel is None:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                if ch == 0:
                    ax.set_title(col_titles[col], fontsize=11, fontweight='bold', pad=8)
                continue

            display = np.copy(panel).astype(float)

            # mask 处理
            if mask is not None:
                ph, pw = display.shape
                mh, mw = mask.shape
                if (mh, mw) != (ph, pw):
                    row_idx = (np.arange(ph) * mh / ph).astype(int)
                    col_idx = (np.arange(pw) * mw / pw).astype(int)
                    m = mask[np.ix_(row_idx, col_idx)]
                else:
                    m = mask
                display[m == 0] = np.nan

            # 绘制
            use_geo_lr = lr_flag and has_geo_lr
            use_geo_hr = (not lr_flag) and has_geo

            if use_geo_lr and lon_lr_grid is not None and lat_lr_grid is not None:
                im = ax.pcolormesh(lon_lr_grid, lat_lr_grid, display,
                                   cmap=cmap, vmin=lo, vmax=hi, shading=lr_shading)
            elif use_geo_hr and lon_hr_grid is not None and lat_hr_grid is not None:
                im = ax.pcolormesh(lon_hr_grid, lat_hr_grid, display,
                                   cmap=cmap, vmin=lo, vmax=hi, shading=hr_shading)
            else:
                im = ax.pcolormesh(display, cmap=cmap, vmin=lo, vmax=hi, shading='auto')

            if col == 2 or col == 3:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if ch == 0:
                ax.set_title(col_titles[col], fontsize=11, fontweight='bold', pad=8)

            if col == 0:
                label = dyn_vars[ch] if dyn_vars and ch < len(dyn_vars) else f'Channel {ch}'
                ax.set_ylabel(label, fontsize=11, fontweight='medium')

            if use_geo_lr or use_geo_hr:
                ax.set_aspect('auto')
                ax.tick_params(labelsize=8)
                if ch == n_channels - 1:
                    ax.set_xlabel('Longitude', fontsize=9)
                if col == 0:
                    ax.set_ylabel(
                        (dyn_vars[ch] if dyn_vars and ch < len(dyn_vars) else f'Channel {ch}')
                        + '\nLatitude',
                        fontsize=9, fontweight='medium',
                    )
            else:
                ax.set_xticks([])
                ax.set_yticks([])

    # 标题
    try:
        from datetime import datetime
        date_str = datetime.strptime(filename, "%Y%m%d").strftime("%B %d, %Y")
        suptitle = f'Predict: {date_str}'
    except (ValueError, TypeError):
        suptitle = f'Predict: {filename}'
    fig.suptitle(suptitle, fontsize=15, fontweight='bold', y=1.0)

    plt.tight_layout(pad=1.2, rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate predict output visualization plots")
    parser.add_argument('--log_dir', required=True, type=str, help='Training/predict log directory')
    parser.add_argument('--dataset_root', required=True, type=str, help='Preprocessed dataset root')
    parser.add_argument('--dyn_vars', required=True, type=str, help='Comma-separated dynamic variable names')
    parser.add_argument('--max_samples', type=int, default=4, help='Max samples to visualize')
    args = parser.parse_args()

    log_dir = args.log_dir
    dataset_root = args.dataset_root
    dyn_vars = [v.strip() for v in args.dyn_vars.split(',') if v.strip()]
    max_samples = args.max_samples

    predictions_dir = os.path.join(log_dir, 'predictions')
    if not os.path.isdir(predictions_dir):
        print(f"[Error] Predictions directory does not exist: {predictions_dir}")
        sys.exit(1)

    # 收集 SR 文件
    sr_files = sorted(glob.glob(os.path.join(predictions_dir, '*_sr.npy')))
    if not sr_files:
        print(f"[Error] No *_sr.npy files found in {predictions_dir}")
        sys.exit(1)

    sr_files = sr_files[:max_samples]

    # 加载经纬度坐标
    coords = load_static_coords(dataset_root)
    lon_hr = coords['lon_hr']
    lat_hr = coords['lat_hr']
    lon_lr = coords['lon_lr']
    lat_lr = coords['lat_lr']

    output_dir = os.path.join(log_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)

    generated_plots = []
    print(f"[Info] Generating predict comparison plots for {len(sr_files)} samples...")

    for idx, sr_path in enumerate(sr_files):
        basename = os.path.basename(sr_path)
        # 文件名格式: {filename}_sr.npy → 提取 filename
        filename = basename.replace('_sr.npy', '')

        sr = np.load(sr_path)  # [H, W, C]
        sample_data = load_sample_data(dataset_root, dyn_vars, filename)
        hr = sample_data['hr']
        lr = sample_data['lr']
        hr_valid = sample_data['hr_valid']

        # 检测 mask（只用实际存在的 HR 通道，避免 NaN 填充通道误判）
        mask = detect_mask_from_hr(hr, hr_valid_mask=hr_valid)

        output_path = os.path.join(output_dir, f'predict_comparison_{idx:02d}.png')
        plot_predict_comparison(
            sr=sr, hr=hr, lr=lr, mask=mask,
            lon_hr=lon_hr, lat_hr=lat_hr,
            lon_lr=lon_lr, lat_lr=lat_lr,
            dyn_vars=dyn_vars, filename=filename,
            output_path=output_path,
        )
        generated_plots.append(output_path)
        print(f"  - [{idx+1}/{len(sr_files)}] {filename}: {output_path}")

    if generated_plots:
        print(f"\n[Success] Generated {len(generated_plots)} predict comparison plots in: {output_dir}")
        result = {
            "status": "success",
            "output_dir": output_dir,
            "plots": generated_plots,
            "n_samples": len(generated_plots),
        }
        print(f"__result__{json.dumps(result)}__result__")
    else:
        print("[Warning] No plots generated")
        sys.exit(1)


if __name__ == "__main__":
    main()
