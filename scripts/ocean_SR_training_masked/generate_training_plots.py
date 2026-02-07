#!/usr/bin/env python3
"""
generate_training_plots.py - Ocean SR 训练可视化脚本

@author kongzhiquan
@date 2026-02-07
@version 1.2.0

@changelog
    - 2026-02-07 kongzhiquan: v1.2.0 美化图表样式，使用现代配色和视觉效果
    - 2026-02-07 kongzhiquan: v1.1.0 移除中文标签，仅使用英文
    - 2026-02-07 kongzhiquan: v1.0.0 初始版本

用法:
        python generate_training_plots.py --log_dir /path/to/log_dir

输出:
        log_dir/plots/
            - loss_curve.png
            - metrics_curve.png
            - lr_curve.png
            - metrics_comparison.png
            - training_summary.png
"""

import os
import re
import sys
import json
import argparse
from typing import Dict, List, Any, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects
import numpy as np

# 现代化样式配置
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
    'primary': '#2563eb',      # 蓝色
    'secondary': '#dc2626',    # 红色
    'success': '#16a34a',      # 绿色
    'warning': '#ea580c',      # 橙色
    'purple': '#9333ea',       # 紫色
    'cyan': '#0891b2',         # 青色
    'pink': '#db2777',         # 粉色
    'gray': '#6b7280',         # 灰色
}

# 渐变色用于填充
GRADIENT_COLORS = {
    'train': ('#3b82f6', '#93c5fd'),  # 蓝色渐变
    'valid': ('#ef4444', '#fca5a5'),  # 红色渐变
    'lr': ('#22c55e', '#86efac'),     # 绿色渐变
}


def parse_structured_log(log_path: str) -> Dict[str, Any]:
    """解析结构化日志并提取训练数据"""
    result = {
        'training_start': None,
        'training_end': None,
        'epoch_train': [],
        'epoch_valid': [],
        'final_valid': None,
        'final_test': None,
    }

    if not os.path.exists(log_path):
        return result

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return result

    event_pattern = re.compile(r'__event__(\{.*?\})__event__')
    matches = event_pattern.findall(content)

    for match in matches:
        try:
            event = json.loads(match)
            event_type = event.get('event')

            if event_type == 'training_start':
                result['training_start'] = event
            elif event_type == 'training_end':
                result['training_end'] = event
            elif event_type == 'epoch_train':
                result['epoch_train'].append(event)
            elif event_type == 'epoch_valid':
                result['epoch_valid'].append(event)
            elif event_type == 'final_valid':
                result['final_valid'] = event
            elif event_type == 'final_test':
                result['final_test'] = event
        except json.JSONDecodeError:
            continue

    return result


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
        # 如果最佳点在右半部分，标注放左边；否则放右边
        if best_epoch > min(valid_epochs) + x_range * 0.5:
            text_x = -80
        else:
            text_x = 20
        # 如果最佳点在上半部分，标注放下面；否则放上面
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

        # 智能定位标注框：根据最佳点位置决定标注方向
        x_range = max(valid_epochs) - min(valid_epochs) if len(valid_epochs) > 1 else 1
        y_range = max(valid_values) - min(valid_values) if len(valid_values) > 1 else 1
        # 如果最佳点在右半部分，标注放左边；否则放右边
        if best_epoch > min(valid_epochs) + x_range * 0.6:
            text_x = -70
        else:
            text_x = 15
        # 如果最佳点在上半部分，标注放下面；否则放上面
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
    from matplotlib.patches import Patch
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

        # 计算每行高度
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


def main():
    parser = argparse.ArgumentParser(description="Generate Ocean SR training visualization plots")
    parser.add_argument('--log_dir', required=True, type=str, help='Training log directory')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: log_dir/plots)')
    args = parser.parse_args()

    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        print(f"[Error] Log directory does not exist: {log_dir}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(log_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)

    log_data = {}
    for name in ['train.log', 'training.log']:
        log_path = os.path.join(log_dir, name)
        if os.path.exists(log_path):
            log_data = parse_structured_log(log_path)
            break

    if not log_data.get('epoch_train'):
        print(f"[Warning] No structured log events found, cannot generate plots")
        sys.exit(1)

    generated_plots = []

    print("[Info] Generating plots...")

    path = plot_loss_curve(log_data, output_dir)
    if path:
        generated_plots.append(path)
        print(f"  - Loss curve: {path}")

    path = plot_metrics_curve(log_data, output_dir)
    if path:
        generated_plots.append(path)
        print(f"  - Metrics curve: {path}")

    path = plot_lr_curve(log_data, output_dir)
    if path:
        generated_plots.append(path)
        print(f"  - Learning rate curve: {path}")

    path = plot_metrics_comparison(log_data, output_dir)
    if path:
        generated_plots.append(path)
        print(f"  - Metrics comparison: {path}")

    path = plot_training_summary(log_data, output_dir)
    if path:
        generated_plots.append(path)
        print(f"  - Training summary: {path}")

    if generated_plots:
        print(f"\n[Success] Generated {len(generated_plots)} plots in: {output_dir}")
        result = {
            "status": "success",
            "output_dir": output_dir,
            "plots": generated_plots
        }
        print(f"__result__{json.dumps(result)}__result__")
    else:
        print("[Warning] No plots generated")
        sys.exit(1)


if __name__ == "__main__":
    main()
