"""
@file plot_summary.py

@description Training summary card-style layout visualization.
@author kongzhiquan
@date 2026-02-09
@version 1.0.0

@changelog
    - 2026-02-09 kongzhiquan: v1.0.0 extracted from generate_training_plots.py
"""

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .style import COLORS, add_figure_border


def plot_training_summary(log_data: Dict, output_dir: str) -> Optional[str]:
    """绘制训练总结卡片式布局"""
    start_event = log_data.get('training_start', {})
    end_event = log_data.get('training_end', {})
    final_test = log_data.get('final_test', {})

    if not end_event:
        return None

    # 提取数据
    model_name = start_event.get('model_name', 'N/A')
    dataset_name = start_event.get('dataset_name', 'N/A')
    model_params = start_event.get('model_params', 'N/A')
    total_epochs = start_event.get('total_epochs', 'N/A')
    actual_epochs = end_event.get('actual_epochs', 'N/A')
    best_epoch = end_event.get('best_epoch', 'N/A')
    early_stopped = end_event.get('early_stopped', False)
    duration = end_event.get('training_duration_seconds', 0)

    test_metrics = final_test.get('metrics', {}) if final_test else {}

    # 格式化时长
    if duration < 60:
        duration_str = f"{duration:.1f}s"
    elif duration < 3600:
        duration_str = f"{duration / 60:.1f} min"
    else:
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        duration_str = f"{hours}h {minutes}min"

    # 格式化参数
    params_str = f"{model_params}M" if isinstance(model_params, (int, float)) else str(model_params)

    # 创建图形
    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('#f8fafc')

    # 主标题
    fig.text(0.5, 0.95, 'Training Summary', fontsize=22, fontweight='bold',
             ha='center', va='top', color='#1e293b')

    # 定义卡片绘制函数
    def draw_card(ax, title, items, title_color=COLORS['primary']):
        """绘制一个信息卡片"""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # 卡片背景
        card_bg = FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor='white',
            edgecolor='#e2e8f0',
            linewidth=1.5,
            transform=ax.transAxes
        )
        ax.add_patch(card_bg)

        # 卡片标题背景
        title_bg = FancyBboxPatch(
            (0.02, 0.82), 0.96, 0.16,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor=title_color,
            edgecolor='none',
            transform=ax.transAxes
        )
        ax.add_patch(title_bg)

        # 卡片标题
        ax.text(0.5, 0.90, title, fontsize=13, fontweight='bold',
                ha='center', va='center', color='white', transform=ax.transAxes)

        # 内容区域
        n_items = len(items)
        if n_items == 0:
            return

        content_top = 0.78
        content_bottom = 0.08
        content_height = content_top - content_bottom
        row_height = content_height / n_items

        for i, (label, value) in enumerate(items):
            y_pos = content_top - (i + 0.5) * row_height

            # 交替背景色
            if i % 2 == 0:
                row_bg = FancyBboxPatch(
                    (0.04, y_pos - row_height * 0.45), 0.92, row_height * 0.9,
                    boxstyle="round,pad=0.005,rounding_size=0.01",
                    facecolor='#f8fafc',
                    edgecolor='none',
                    transform=ax.transAxes
                )
                ax.add_patch(row_bg)

            # 标签
            ax.text(0.08, y_pos, label, fontsize=10, fontweight='medium',
                    ha='left', va='center', color='#64748b', transform=ax.transAxes)

            # 值
            ax.text(0.92, y_pos, str(value), fontsize=10, fontweight='semibold',
                    ha='right', va='center', color='#1e293b', transform=ax.transAxes)

    # 定义指标卡片绘制函数
    def draw_metric_card(ax, label, value, color, better='lower'):
        """绘制单个指标卡片"""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # 卡片背景
        card_bg = FancyBboxPatch(
            (0.05, 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor='white',
            edgecolor=color,
            linewidth=2,
            transform=ax.transAxes
        )
        ax.add_patch(card_bg)

        # 顶部装饰条
        top_bar = FancyBboxPatch(
            (0.05, 0.85), 0.9, 0.1,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor=color,
            edgecolor='none',
            transform=ax.transAxes
        )
        ax.add_patch(top_bar)

        # 指标名称
        ax.text(0.5, 0.68, label, fontsize=11, fontweight='bold',
                ha='center', va='center', color='#475569', transform=ax.transAxes)

        # 指标值
        if value != 'N/A':
            ax.text(0.5, 0.40, value, fontsize=16, fontweight='bold',
                    ha='center', va='center', color=color, transform=ax.transAxes)
        else:
            ax.text(0.5, 0.40, 'N/A', fontsize=14, fontweight='medium',
                    ha='center', va='center', color='#94a3b8', style='italic',
                    transform=ax.transAxes)

        # 方向指示
        direction_text = '(lower is better)' if better == 'lower' else '(higher is better)'
        ax.text(0.5, 0.18, direction_text, fontsize=8,
                ha='center', va='center', color='#94a3b8', transform=ax.transAxes)

    # 创建网格布局
    # 上半部分：两个信息卡片并排
    ax_info1 = fig.add_axes([0.04, 0.48, 0.44, 0.42])
    ax_info2 = fig.add_axes([0.52, 0.48, 0.44, 0.42])

    # 下半部分：四个指标卡片
    ax_mse = fig.add_axes([0.04, 0.06, 0.22, 0.36])
    ax_rmse = fig.add_axes([0.28, 0.06, 0.22, 0.36])
    ax_psnr = fig.add_axes([0.52, 0.06, 0.22, 0.36])
    ax_ssim = fig.add_axes([0.76, 0.06, 0.22, 0.36])

    # 绘制模型信息卡片
    model_items = [
        ('Model', model_name),
        ('Dataset', dataset_name),
        ('Parameters', params_str),
    ]
    draw_card(ax_info1, 'Model Information', model_items, COLORS['primary'])

    # 绘制训练信息卡片
    training_items = [
        ('Planned Epochs', str(total_epochs)),
        ('Actual Epochs', str(actual_epochs)),
        ('Best Epoch', str(best_epoch)),
        ('Early Stopped', 'Yes' if early_stopped else 'No'),
        ('Duration', duration_str),
    ]
    draw_card(ax_info2, 'Training Progress', training_items, COLORS['success'])

    # 绘制指标卡片
    mse_val = f"{test_metrics.get('mse'):.6f}" if isinstance(test_metrics.get('mse'), float) else 'N/A'
    rmse_val = f"{test_metrics.get('rmse'):.6f}" if isinstance(test_metrics.get('rmse'), float) else 'N/A'
    psnr_val = f"{test_metrics.get('psnr'):.2f}" if isinstance(test_metrics.get('psnr'), float) else 'N/A'
    ssim_val = f"{test_metrics.get('ssim'):.6f}" if isinstance(test_metrics.get('ssim'), float) else 'N/A'

    draw_metric_card(ax_mse, 'Test MSE', mse_val, COLORS['primary'], 'lower')
    draw_metric_card(ax_rmse, 'Test RMSE', rmse_val, COLORS['warning'], 'lower')
    draw_metric_card(ax_psnr, 'Test PSNR', psnr_val, COLORS['success'], 'higher')
    draw_metric_card(ax_ssim, 'Test SSIM', ssim_val, COLORS['purple'], 'higher')

    # 添加外边框
    add_figure_border(fig, color='#cbd5e1', linewidth=2)

    output_path = os.path.join(output_dir, 'training_summary.png')
    plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='#f8fafc')
    plt.close()

    return output_path
