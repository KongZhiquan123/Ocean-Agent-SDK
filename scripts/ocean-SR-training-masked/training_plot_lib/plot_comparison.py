"""
@file plot_comparison.py

@description Validation vs Test metrics bar chart comparison.
@author kongzhiquan
@date 2026-02-09
@version 1.0.0

@changelog
    - 2026-02-09 kongzhiquan: v1.0.0 extracted from generate_training_plots.py
"""

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Patch

from .style import COLORS, add_figure_border


def plot_metrics_comparison(log_data: Dict, output_dir: str) -> Optional[str]:
    """绘制验证与测试指标对比"""
    final_valid = log_data.get('final_valid', {})
    final_test = log_data.get('final_test', {})

    valid_metrics = final_valid.get('metrics', {}) if final_valid else {}
    test_metrics = final_test.get('metrics', {}) if final_test else {}

    if not valid_metrics and not test_metrics:
        return None

    metrics_keys = ['mse', 'rmse', 'psnr', 'ssim']

    if all(valid_metrics.get(k) is None for k in metrics_keys) and \
       all(test_metrics.get(k) is None for k in metrics_keys):
        return None

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    titles = ['MSE', 'RMSE', 'PSNR', 'SSIM']
    colors_valid = COLORS['primary']
    colors_test = COLORS['secondary']

    for idx, (key, title) in enumerate(zip(metrics_keys, titles)):
        ax = axes[idx]
        v_val = valid_metrics.get(key)
        t_val = test_metrics.get(key)

        bars = []
        labels = []
        colors = []

        if v_val is not None:
            bars.append(v_val)
            labels.append('Valid')
            colors.append(colors_valid)
        if t_val is not None:
            bars.append(t_val)
            labels.append('Test')
            colors.append(colors_test)

        if bars:
            x = range(len(bars))
            bar_plot = ax.bar(x, bars, color=colors, width=0.55,
                              edgecolor='white', linewidth=2,
                              alpha=0.85, zorder=3)

            # 添加渐变效果（通过多层叠加模拟）
            for bar, val, color in zip(bar_plot, bars, colors):
                height = bar.get_height()
                # 添加数值标签
                text = ax.annotate(f'{val:.4f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 8), textcoords='offset points',
                                   ha='center', va='bottom',
                                   fontsize=11, fontweight='bold',
                                   color=color)
                # 添加白色描边效果
                text.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground='white'),
                    path_effects.Normal()
                ])

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=11, fontweight='medium')

            # 设置 y 轴范围，留出标签空间
            y_max = max(bars) * 1.2
            ax.set_ylim(0, y_max)
        else:
            ax.text(0.5, 0.5, 'No Data',
                    ha='center', va='center', fontsize=12,
                    color=COLORS['gray'], style='italic')
            ax.set_facecolor('#f9fafb')

        ax.set_title(title, pad=12)
        ax.grid(True, alpha=0.3, axis='y', zorder=0)

    # 添加图例
    legend_elements = [
        Patch(facecolor=colors_valid, edgecolor='white', label='Validation', alpha=0.85),
        Patch(facecolor=colors_test, edgecolor='white', label='Test', alpha=0.85)
    ]
    fig.legend(handles=legend_elements, loc='upper center',
               ncol=2, bbox_to_anchor=(0.5, 0.02),
               frameon=True, fancybox=True, shadow=True,
               fontsize=11)

    fig.suptitle('Validation vs Test Metrics Comparison',
                 fontsize=16, fontweight='bold', y=0.98)

    add_figure_border(fig)
    plt.tight_layout(pad=1.5, rect=[0, 0.08, 1, 0.94])
    output_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()

    return output_path
