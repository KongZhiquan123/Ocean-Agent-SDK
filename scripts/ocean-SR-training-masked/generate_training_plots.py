#!/usr/bin/env python3
"""
generate_training_plots.py - Ocean SR 训练可视化脚本

@author kongzhiquan
@date 2026-02-07
@version 2.0.0

@changelog
    - 2026-02-09 kongzhiquan: v2.0.0 模块化重构，拆分为 training_plot_lib 包
    - 2026-02-09 kongzhiquan: v1.3.1 修复 mask_hr 维度/尺寸不匹配导致 IndexError
    - 2026-02-09 kongzhiquan: v1.3.0 新增测试样本 LR/SR/HR 对比可视化
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
            - sample_comparison.png
"""

import os
import sys
import json
import argparse

from training_plot_lib import (
    parse_structured_log,
    plot_loss_curve,
    plot_metrics_curve,
    plot_lr_curve,
    plot_metrics_comparison,
    plot_sample_comparison,
    plot_training_summary,
)


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

    path = plot_sample_comparison(log_dir, output_dir)
    if path:
        generated_plots.append(path)
        print(f"  - Sample comparison: {path}")

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
