"""
@file plot_lr.py

@description Learning rate schedule curve plotting.
@author kongzhiquan
@date 2026-02-09
@version 1.0.0

@changelog
    - 2026-02-09 kongzhiquan: v1.0.0 extracted from generate_training_plots.py
"""

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt

from .style import COLORS, add_figure_border, calc_marker_interval


def plot_lr_curve(log_data: Dict, output_dir: str) -> Optional[str]:
    """绘制学习率曲线"""
    epoch_train = log_data.get('epoch_train', [])

    if not epoch_train:
        return None

    epochs = [e['epoch'] for e in epoch_train]
    lrs = [e.get('lr', 0) for e in epoch_train]

    if all(lr == 0 or lr is None for lr in lrs):
        return None

    fig, ax = plt.subplots(figsize=(11, 5))

    # 计算 marker 间隔
    marker_interval = calc_marker_interval(len(epochs))

    # 绘制学习率曲线（不带 marker）
    ax.plot(epochs, lrs,
            color=COLORS['success'],
            linewidth=2.5,
            zorder=3)
    # 单独绘制稀疏的 marker
    ax.scatter(epochs[::marker_interval], lrs[::marker_interval],
               color=COLORS['success'], s=40,
               facecolors='white', edgecolors=COLORS['success'],
               linewidths=2, zorder=4)
    ax.fill_between(epochs, lrs, alpha=0.15, color=COLORS['success'])

    # 标记起始和结束学习率
    if len(lrs) > 1:
        # 起始点
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor='white',
                          edgecolor=COLORS['primary'], alpha=0.95, linewidth=1.2)
        ax.annotate(f'Start: {lrs[0]:.2e}',
                    xy=(epochs[0], lrs[0]),
                    xytext=(10, 15), textcoords='offset points',
                    fontsize=9, fontweight='medium',
                    color=COLORS['primary'], bbox=bbox_props)

        # 结束点
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor='white',
                          edgecolor=COLORS['secondary'], alpha=0.95, linewidth=1.2)
        ax.annotate(f'End: {lrs[-1]:.2e}',
                    xy=(epochs[-1], lrs[-1]),
                    xytext=(-60, 15), textcoords='offset points',
                    fontsize=9, fontweight='medium',
                    color=COLORS['secondary'], bbox=bbox_props)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule', pad=15)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    add_figure_border(fig)
    plt.tight_layout(pad=1.5)
    output_path = os.path.join(output_dir, 'lr_curve.png')
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()

    return output_path
