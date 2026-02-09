"""
@file plot_metrics.py

@description Validation metrics curve plotting (MSE, RMSE, PSNR, SSIM).
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

from .style import COLORS, add_figure_border


def plot_metrics_curve(log_data: Dict, output_dir: str) -> Optional[str]:
    """绘制评估指标曲线"""
    epoch_valid = log_data.get('epoch_valid', [])

    if not epoch_valid:
        return None

    epochs = [e['epoch'] for e in epoch_valid]
    metrics_keys = ['mse', 'rmse', 'psnr', 'ssim']
    metrics_data = {k: [] for k in metrics_keys}

    for e in epoch_valid:
        m = e.get('metrics', {})
        for k in metrics_keys:
            metrics_data[k].append(m.get(k))

    has_data = any(any(v is not None for v in metrics_data[k]) for k in metrics_keys)
    if not has_data:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    titles = ['MSE (Mean Squared Error)', 'RMSE (Root Mean Squared Error)',
              'PSNR (Peak Signal-to-Noise Ratio)', 'SSIM (Structural Similarity)']
    colors = [COLORS['primary'], COLORS['warning'], COLORS['success'], COLORS['purple']]
    better_direction = ['lower', 'lower', 'higher', 'higher']

    for idx, (key, title, color, direction) in enumerate(zip(metrics_keys, titles, colors, better_direction)):
        ax = axes[idx]
        values = metrics_data[key]

        if all(v is None for v in values):
            ax.text(0.5, 0.5, 'No Data Available',
                    ha='center', va='center', fontsize=13,
                    color=COLORS['gray'], style='italic')
            ax.set_title(title, pad=10)
            ax.set_facecolor('#f9fafb')
            continue

        valid_data = [(e, v) for e, v in zip(epochs, values) if v is not None]
        if not valid_data:
            continue

        valid_epochs, valid_values = zip(*valid_data)

        # 绘制带填充的曲线
        ax.plot(valid_epochs, valid_values,
                color=color, linewidth=2.5,
                marker='o', markersize=6,
                markerfacecolor='white', markeredgewidth=2,
                zorder=3)
        ax.fill_between(valid_epochs, valid_values, alpha=0.12, color=color)

        # 找到最佳值
        if direction == 'lower':
            best_idx = np.argmin(valid_values)
        else:
            best_idx = np.argmax(valid_values)

        best_epoch = valid_epochs[best_idx]
        best_value = valid_values[best_idx]

        # 绘制最佳值水平线
        ax.axhline(y=best_value, color=color, linestyle=':', alpha=0.5, linewidth=1.5)

        # 绘制最佳点标记
        ax.scatter([best_epoch], [best_value],
                   color='#fbbf24', s=180, zorder=5,
                   marker='*', edgecolors=color, linewidths=1.5)

        # 智能定位标注框
        x_range = max(valid_epochs) - min(valid_epochs) if len(valid_epochs) > 1 else 1
        y_range = max(valid_values) - min(valid_values) if len(valid_values) > 1 else 1
        if best_epoch > min(valid_epochs) + x_range * 0.6:
            text_x = -70
        else:
            text_x = 15
        if best_value > min(valid_values) + y_range * 0.6:
            text_y = -35
        else:
            text_y = 20

        # 添加带背景的标注
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor='white',
                          edgecolor=color, alpha=0.95, linewidth=1.2)
        ax.annotate(f'Best: {best_value:.4f}\n(Epoch {best_epoch})',
                    xy=(best_epoch, best_value),
                    xytext=(text_x, text_y), textcoords='offset points',
                    fontsize=9, fontweight='medium',
                    color=color, bbox=bbox_props,
                    arrowprops=dict(arrowstyle='->', color=color,
                                    connectionstyle='arc3,rad=0.2', linewidth=1))

        ax.set_xlabel('Epoch')
        ax.set_ylabel(key.upper())
        ax.set_title(title, pad=10)

    # 添加总标题
    fig.suptitle('Validation Metrics Over Training',
                 fontsize=16, fontweight='bold', y=0.98)

    add_figure_border(fig)
    plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.96])
    output_path = os.path.join(output_dir, 'metrics_curve.png')
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()

    return output_path
