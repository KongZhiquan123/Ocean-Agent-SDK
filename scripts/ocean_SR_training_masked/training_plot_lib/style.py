"""
@file style.py

@description Matplotlib style constants and utility functions for training plots.
@author kongzhiquan
@date 2026-02-09
@version 1.0.0

@changelog
    - 2026-02-09 kongzhiquan: v1.0.0 extracted from generate_training_plots.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# 现代化样式配置 — import 时立即生效
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.unicode_minus': False,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.labelweight': 'medium',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'axes.edgecolor': '#333333',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#cccccc',
    'figure.facecolor': '#fafafa',
    'axes.facecolor': '#ffffff',
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'savefig.facecolor': '#fafafa',
    'savefig.edgecolor': 'none',
})

# 现代配色方案
COLORS = {
    'primary': '#2563eb',
    'secondary': '#dc2626',
    'success': '#16a34a',
    'warning': '#ea580c',
    'purple': '#9333ea',
    'cyan': '#0891b2',
    'pink': '#db2777',
    'gray': '#6b7280',
}

# 渐变色用于填充
GRADIENT_COLORS = {
    'train': ('#3b82f6', '#93c5fd'),
    'valid': ('#ef4444', '#fca5a5'),
    'lr': ('#22c55e', '#86efac'),
}


def add_figure_border(fig, color='#e5e7eb', linewidth=2, padding=0.02):
    """为图表添加圆角边框"""
    rect = FancyBboxPatch(
        (padding, padding), 1 - 2 * padding, 1 - 2 * padding,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=linewidth,
        edgecolor=color,
        facecolor='none',
        transform=fig.transFigure,
        clip_on=False
    )
    fig.patches.append(rect)


def calc_marker_interval(n_points: int, target_markers: int = 15) -> int:
    """计算 marker 间隔，使得显示的 marker 数量适中"""
    if n_points <= target_markers:
        return 1
    return max(1, n_points // target_markers)
