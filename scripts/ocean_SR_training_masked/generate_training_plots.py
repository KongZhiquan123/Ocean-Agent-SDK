#!/usr/bin/env python3
"""
generate_training_plots.py - Ocean SR 训练可视化脚本

@author kongzhiquan
@date 2026-02-07
@version 1.1.0

@changelog
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
import numpy as np

# 使用默认的英文字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


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


def plot_loss_curve(log_data: Dict, output_dir: str) -> Optional[str]:
    """绘制损失曲线"""
    epoch_train = log_data.get('epoch_train', [])
    epoch_valid = log_data.get('epoch_valid', [])

    if not epoch_train:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # 训练损失
    train_epochs = [e['epoch'] for e in epoch_train]
    train_losses = [e['metrics'].get('train_loss', 0) for e in epoch_train]
    ax.plot(train_epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)

    # 验证损失
    if epoch_valid:
        valid_epochs = [e['epoch'] for e in epoch_valid]
        valid_losses = [e['metrics'].get('valid_loss', 0) for e in epoch_valid]
        ax.plot(valid_epochs, valid_losses, 'r-', label='Valid Loss', linewidth=2, marker='s', markersize=3)

        # 标记最佳轮次
        best_idx = np.argmin(valid_losses)
        best_epoch = valid_epochs[best_idx]
        best_loss = valid_losses[best_idx]
        ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
        ax.annotate(f'Best: {best_loss:.6f}',
                    xy=(best_epoch, best_loss),
                    xytext=(best_epoch + 2, best_loss * 1.1),
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='green'))

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    titles = ['MSE (Mean Squared Error)', 'RMSE (Root Mean Squared Error)',
              'PSNR (Peak Signal-to-Noise Ratio)', 'SSIM (Structural Similarity)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    better_direction = ['lower', 'lower', 'higher', 'higher']

    for idx, (key, title, color, direction) in enumerate(zip(metrics_keys, titles, colors, better_direction)):
        ax = axes[idx]
        values = metrics_data[key]

        if all(v is None for v in values):
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=12)
            continue

        valid_data = [(e, v) for e, v in zip(epochs, values) if v is not None]
        if not valid_data:
            continue

        valid_epochs, valid_values = zip(*valid_data)
        ax.plot(valid_epochs, valid_values, color=color, linewidth=2, marker='o', markersize=4)

        if direction == 'lower':
            best_idx = np.argmin(valid_values)
        else:
            best_idx = np.argmax(valid_values)

        best_epoch = valid_epochs[best_idx]
        best_value = valid_values[best_idx]
        ax.axhline(y=best_value, color=color, linestyle='--', alpha=0.5)
        ax.scatter([best_epoch], [best_value], color='red', s=100, zorder=5, marker='*')
        ax.annotate(f'Best: {best_value:.4f}\n(Epoch {best_epoch})',
                    xy=(best_epoch, best_value),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, color='red')

        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(key.upper(), fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Validation Metrics Over Training', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'metrics_curve.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epochs, lrs, 'g-', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-3, 3))

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'lr_curve.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    titles = ['MSE', 'RMSE', 'PSNR', 'SSIM']
    colors_valid = '#3498db'
    colors_test = '#e74c3c'

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
            bar_plot = ax.bar(x, bars, color=colors, width=0.6, edgecolor='black', linewidth=1)

            for bar, val in zip(bar_plot, bars):
                height = bar.get_height()
                ax.annotate(f'{val:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Validation vs Test Metrics Comparison', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_training_summary(log_data: Dict, output_dir: str) -> Optional[str]:
    """绘制训练总结表格"""
    start_event = log_data.get('training_start', {})
    end_event = log_data.get('training_end', {})
    final_test = log_data.get('final_test', {})

    if not end_event:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    model_name = start_event.get('model_name', 'N/A')
    dataset_name = start_event.get('dataset_name', 'N/A')
    model_params = start_event.get('model_params', 'N/A')
    total_epochs = start_event.get('total_epochs', 'N/A')
    actual_epochs = end_event.get('actual_epochs', 'N/A')
    best_epoch = end_event.get('best_epoch', 'N/A')
    early_stopped = end_event.get('early_stopped', False)
    duration = end_event.get('training_duration_seconds', 0)

    test_metrics = final_test.get('metrics', {}) if final_test else {}

    if duration < 60:
        duration_str = f"{duration:.1f}s"
    elif duration < 3600:
        duration_str = f"{duration / 60:.1f}min"
    else:
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        duration_str = f"{hours}h {minutes}min"

    table_data = [
        ['Model', model_name],
        ['Dataset', dataset_name],
        ['Parameters', f"{model_params}M" if isinstance(model_params, (int, float)) else str(model_params)],
        ['Planned Epochs', str(total_epochs)],
        ['Actual Epochs', str(actual_epochs)],
        ['Best Epoch', str(best_epoch)],
        ['Early Stopped', 'Yes' if early_stopped else 'No'],
        ['Training Duration', duration_str],
        ['', ''],
        ['Test MSE', f"{test_metrics.get('mse', 'N/A'):.6f}" if isinstance(test_metrics.get('mse'), float) else 'N/A'],
        ['Test RMSE', f"{test_metrics.get('rmse', 'N/A'):.6f}" if isinstance(test_metrics.get('rmse'), float) else 'N/A'],
        ['Test PSNR', f"{test_metrics.get('psnr', 'N/A'):.4f}" if isinstance(test_metrics.get('psnr'), float) else 'N/A'],
        ['Test SSIM', f"{test_metrics.get('ssim', 'N/A'):.6f}" if isinstance(test_metrics.get('ssim'), float) else 'N/A'],
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=['Item', 'Value'],
        loc='center',
        cellLoc='left',
        colWidths=[0.4, 0.4]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for i in range(2):
        table[(0, i)].set_facecolor('#4a90d9')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    for i in range(1, len(table_data) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title('Training Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
