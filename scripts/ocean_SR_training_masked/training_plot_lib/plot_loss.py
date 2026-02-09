"""
@file plot_loss.py

@description Loss curve plotting for training and validation losses.
@author kongzhiquan
@date 2026-02-09
@version 1.0.0

@changelog
    - 2026-02-09 kongzhiquan: v1.0.0 extracted from generate_training_plots.py
"""

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, add_figure_border, calc_marker_interval


def plot_loss_curve(log_data: Dict, output_dir: str) -> Optional[str]:
    """绘制损失曲线"""
    epoch_train = log_data.get('epoch_train', [])
    epoch_valid = log_data.get('epoch_valid', [])

    if not epoch_train:
        return None

    fig, ax = plt.subplots(figsize=(11, 6))

    # 训练损失
    train_epochs = [e['epoch'] for e in epoch_train]
    train_losses = [e['metrics'].get('train_loss', 0) for e in epoch_train]

    # 计算 marker 间隔
    marker_interval = calc_marker_interval(len(train_epochs))

    # 绘制带填充的训练曲线（不带 marker）
    ax.plot(train_epochs, train_losses,
            color=COLORS['primary'],
            linewidth=2.5,
            label='Train Loss',
            zorder=3)
    # 单独绘制稀疏的 marker
    ax.scatter(train_epochs[::marker_interval], train_losses[::marker_interval],
               color=COLORS['primary'], s=40,
               facecolors='white', edgecolors=COLORS['primary'],
               linewidths=2, zorder=4)
    ax.fill_between(train_epochs, train_losses, alpha=0.15, color=COLORS['primary'])

    # 验证损失
    if epoch_valid:
        valid_epochs = [e['epoch'] for e in epoch_valid]
        valid_losses = [e['metrics'].get('valid_loss', 0) for e in epoch_valid]

        valid_marker_interval = calc_marker_interval(len(valid_epochs))

        # 绘制曲线（不带 marker）
        ax.plot(valid_epochs, valid_losses,
                color=COLORS['secondary'],
                linewidth=2.5,
                label='Valid Loss',
                zorder=3)
        # 单独绘制稀疏的 marker
        ax.scatter(valid_epochs[::valid_marker_interval], valid_losses[::valid_marker_interval],
                   color=COLORS['secondary'], s=40, marker='s',
                   facecolors='white', edgecolors=COLORS['secondary'],
                   linewidths=2, zorder=4)
        ax.fill_between(valid_epochs, valid_losses, alpha=0.15, color=COLORS['secondary'])

        # 标记最佳轮次
        best_idx = np.argmin(valid_losses)
        best_epoch = valid_epochs[best_idx]
        best_loss = valid_losses[best_idx]

        # 绘制最佳点的垂直线和标记
        ax.axvline(x=best_epoch, color=COLORS['success'], linestyle='--', alpha=0.6, linewidth=1.5)
        ax.scatter([best_epoch], [best_loss],
                   s=150, color=COLORS['success'],
                   marker='*', zorder=5,
                   edgecolors='white', linewidths=1.5)

        # 智能定位标注框：根据最佳点位置决定标注方向
        x_range = max(valid_epochs) - min(valid_epochs)
        y_range = max(valid_losses) - min(valid_losses)
        if best_epoch > min(valid_epochs) + x_range * 0.5:
            text_x = -80
        else:
            text_x = 20
        if best_loss > min(valid_losses) + y_range * 0.5:
            text_y = -40
        else:
            text_y = 30

        # 添加带背景的标注
        bbox_props = dict(boxstyle="round,pad=0.4", facecolor='white',
                          edgecolor=COLORS['success'], alpha=0.95, linewidth=1.5)
        ax.annotate(f'Best: {best_loss:.6f}\nEpoch {best_epoch}',
                    xy=(best_epoch, best_loss),
                    xytext=(text_x, text_y),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='medium',
                    color=COLORS['success'],
                    bbox=bbox_props,
                    arrowprops=dict(arrowstyle='->', color=COLORS['success'],
                                    connectionstyle='arc3,rad=0.2', linewidth=1.5))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curve', pad=15)

    # 美化图例
    legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_linewidth(1.2)

    # 添加边框
    add_figure_border(fig)

    plt.tight_layout(pad=1.5)
    output_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()

    return output_path
