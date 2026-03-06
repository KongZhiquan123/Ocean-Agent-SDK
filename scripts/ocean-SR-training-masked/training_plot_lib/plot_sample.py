"""
@file plot_sample.py

@description Test sample LR/SR/HR comparison visualization with error maps.
    Uses pcolormesh for geo-referenced plotting when lon/lat metadata is available.
@author kongzhiquan
@contributors Leizheng
@date 2026-02-09
@version 2.2.0

@changelog
    - 2026-02-24 Leizheng: v2.2.0 修复非日期格式文件名导致 strptime 崩溃
        - try/except 包裹日期解析，非日期文件名（如序号 "000000"）回退为原始名称
    - 2026-02-09 kongzhiquan: v2.1.0 修复 2D 经纬度坐标无法传入 pcolormesh 的问题
        - 2D lon/lat 直接作为坐标网格传给 pcolormesh（shading='auto'）
        - 1D lon/lat 仍转换为 edges 后使用 shading='flat'
    - 2026-02-09 kongzhiquan: v2.0.0 pcolormesh + 经纬度/日期/变量名显示
        - imshow → pcolormesh，有经纬度时显示地理坐标轴
        - 读取 dyn_vars 作为行标签替代 Channel 编号
        - 读取 filename 显示在 suptitle 中
        - 无元数据时向后兼容（像素坐标 + 通用标题）
    - 2026-02-09 kongzhiquan: v1.0.0 extracted from generate_training_plots.py
"""

from datetime import datetime
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .style import add_figure_border


def _make_grid_edges(centers: np.ndarray) -> np.ndarray:
    """从中心点坐标构造 pcolormesh 所需的边界坐标（N+1 个点）。

    对于 N 个中心点，生成 N+1 个边界：
    - 内部边界取相邻中心点的中点
    - 两端边界向外延伸半个间距
    """
    if centers.ndim != 1 or len(centers) < 2:
        return centers
    mid = 0.5 * (centers[:-1] + centers[1:])
    left = 2 * centers[0] - mid[0]
    right = 2 * centers[-1] - mid[-1]
    return np.concatenate([[left], mid, [right]])


def plot_sample_comparison(log_dir: str, output_dir: str) -> Optional[str]:
    """绘制测试样本 LR / SR / HR 对比图 + 误差图

    从 log_dir/test_samples.npz 加载数据，每个通道一行，
    列依次为 LR、SR、HR、|SR - HR| 误差。
    当 npz 中包含经纬度元数据时使用 pcolormesh 显示地理坐标。
    """
    npz_path = os.path.join(log_dir, 'test_samples.npz')
    if not os.path.exists(npz_path):
        return None

    data = np.load(npz_path, allow_pickle=True)
    lr = data['lr']   # [N, H_lr, W_lr, C]
    sr = data['sr']   # [N, H_hr, W_hr, C]
    hr = data['hr']   # [N, H_hr, W_hr, C]
    mask_hr = data['mask_hr'] if 'mask_hr' in data else None

    # 可选元数据
    lon_hr = data['lon_hr'] if 'lon_hr' in data else None
    lat_hr = data['lat_hr'] if 'lat_hr' in data else None
    lon_lr = data['lon_lr'] if 'lon_lr' in data else None
    lat_lr = data['lat_lr'] if 'lat_lr' in data else None
    dyn_vars = list(data['dyn_vars']) if 'dyn_vars' in data else None
    filename = str(data['filename']) if 'filename' in data else None

    has_geo = lon_hr is not None and lat_hr is not None
    has_geo_lr = lon_lr is not None and lat_lr is not None

    # 构建 pcolormesh 坐标：1D → edges (N+1)，2D → 直接使用
    lon_hr_grid = lat_hr_grid = lon_lr_grid = lat_lr_grid = None
    hr_shading = lr_shading = 'auto'
    if has_geo:
        if lon_hr.ndim == 1:
            lon_hr_grid = _make_grid_edges(lon_hr)
            lat_hr_grid = _make_grid_edges(lat_hr)
            hr_shading = 'flat'
        else:
            lon_hr_grid = lon_hr
            lat_hr_grid = lat_hr
            hr_shading = 'auto'
    if has_geo_lr:
        if lon_lr.ndim == 1:
            lon_lr_grid = _make_grid_edges(lon_lr)
            lat_lr_grid = _make_grid_edges(lat_lr)
            lr_shading = 'flat'
        else:
            lon_lr_grid = lon_lr
            lat_lr_grid = lat_lr
            lr_shading = 'auto'

    # 只可视化第一条样本
    lr0 = lr[0]  # [H_lr, W_lr, C]
    sr0 = sr[0]  # [H_hr, W_hr, C]
    hr0 = hr[0]  # [H_hr, W_hr, C]
    mask0 = mask_hr[0].squeeze() if mask_hr is not None else None

    n_channels = hr0.shape[-1]
    col_titles = ['LR (input)', 'SR (prediction)', 'HR (ground truth)', '|SR - HR| error']
    n_cols = 4

    fig, axes = plt.subplots(n_channels, n_cols, figsize=(4.2 * n_cols, 3.6 * n_channels + 0.8))
    if n_channels == 1:
        axes = axes[np.newaxis, :]

    for ch in range(n_channels):
        lr_ch = lr0[..., ch]
        sr_ch = sr0[..., ch]
        hr_ch = hr0[..., ch]
        err = np.abs(sr_ch - hr_ch)

        # 统一色标范围（LR/SR/HR 共享）
        vmin = min(float(lr_ch.min()), float(sr_ch.min()), float(hr_ch.min()))
        vmax = max(float(lr_ch.max()), float(sr_ch.max()), float(hr_ch.max()))

        panels = [lr_ch, sr_ch, hr_ch, err]
        cmaps = ['viridis', 'viridis', 'viridis', 'cividis']
        vmins = [vmin, vmin, vmin, 0]
        vmaxs = [vmax, vmax, vmax, float(err.max()) if err.max() > 0 else 1]
        # LR 用 lr 坐标，其余用 hr 坐标
        is_lr = [True, False, False, False]

        for col, (panel, cmap, lo, hi, lr_flag) in enumerate(
            zip(panels, cmaps, vmins, vmaxs, is_lr)
        ):
            ax = axes[ch, col]
            display = np.copy(panel).astype(float)

            # mask 处理：陆地像素设为 NaN
            if mask0 is not None:
                ph, pw = display.shape
                mh, mw = mask0.shape
                if (mh, mw) != (ph, pw):
                    row_idx = (np.arange(ph) * mh / ph).astype(int)
                    col_idx = (np.arange(pw) * mw / pw).astype(int)
                    m = mask0[np.ix_(row_idx, col_idx)]
                else:
                    m = mask0
                display[m == 0] = np.nan

            # 选择坐标
            use_geo_lr = lr_flag and has_geo_lr
            use_geo_hr = (not lr_flag) and has_geo

            if use_geo_lr and lon_lr_grid is not None and lat_lr_grid is not None:
                im = ax.pcolormesh(lon_lr_grid, lat_lr_grid, display,
                                   cmap=cmap, vmin=lo, vmax=hi, shading=lr_shading)
            elif use_geo_hr and lon_hr_grid is not None and lat_hr_grid is not None:
                im = ax.pcolormesh(lon_hr_grid, lat_hr_grid, display,
                                   cmap=cmap, vmin=lo, vmax=hi, shading=hr_shading)
            else:
                im = ax.pcolormesh(display, cmap=cmap, vmin=lo, vmax=hi,
                                   shading='auto')

            # colorbar: LR/SR/HR 共享色标 → 只在 HR 列(col=2)显示；error 列(col=3)单独显示
            if col == 2 or col == 3:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if ch == 0:
                ax.set_title(col_titles[col], fontsize=11, fontweight='bold', pad=8)

            # 行标签：变量名或 Channel 编号
            if col == 0:
                label = dyn_vars[ch] if dyn_vars and ch < len(dyn_vars) else f'Channel {ch}'
                ax.set_ylabel(label, fontsize=11, fontweight='medium')

            # 坐标轴标签
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
    if filename:
        try:
            date_str = datetime.strptime(filename, "%Y%m%d").strftime("%B %d, %Y")
            suptitle = f'Date: {date_str}  LR vs SR vs HR'
        except (ValueError, TypeError):
            suptitle = f'Sample: {filename}  LR vs SR vs HR'
    else:
        suptitle = 'Test Sample: LR vs SR vs HR Comparison'
    fig.suptitle(suptitle, fontsize=15, fontweight='bold', y=1.0)

    add_figure_border(fig)
    plt.tight_layout(pad=1.2, rect=[0, 0, 1, 0.97])
    output_path = os.path.join(output_dir, 'sample_comparison.png')
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()
    return output_path