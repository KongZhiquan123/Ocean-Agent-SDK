#!/usr/bin/env python3
"""
generate_training_report.py - 海洋超分辨率训练报告生成脚本

@author kongzhiquan
@date 2026-02-06
@version 2.2.0

@changelog
  - 2026-02-09 kongzhiquan: v2.2.0 新增测试样本对比图支持
    - find_plot_files() 增加 sample_comparison.png 检测
    - 报告可视化章节新增 LR/SR/HR 对比图嵌入
  - 2026-02-07 kongzhiquan: v2.1.0 添加可视化结果章节
    - 新增第 4 节可视化结果，插入训练图表
    - 自动检测 plots/ 目录下的图表文件
  - 2026-02-07 kongzhiquan: v2.0.0 重构为结构化日志解析
    - 使用 __event__{}__event__ 格式解析 JSON 事件
    - 按照 train_report.md 模板生成报告
    - 移除正则表达式解析，改用结构化数据

用法:
    python generate_training_report.py --log_dir /path/to/log_dir

输出:
    log_dir/training_report.md
"""

import os
import re
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any


def load_yaml_safe(file_path: str) -> Optional[Dict]:
    """安全加载 YAML 文件"""
    if not os.path.exists(file_path):
        return None
    try:
        import yaml
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[Warning] 无法加载 {file_path}: {e}")
        return None


def parse_structured_log(log_path: str) -> Dict[str, Any]:
    """解析结构化日志文件，提取 __event__{...}__event__ 格式的 JSON 事件"""
    result = {
        'training_start': None,
        'training_end': None,
        'epoch_history': [],
        'final_valid': None,
        'final_test': None,
        'early_stop': None,
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
    epoch_train_cache = {}

    for match in matches:
        try:
            event = json.loads(match)
            event_type = event.get('event')

            if event_type == 'training_start':
                result['training_start'] = event
            elif event_type == 'training_end':
                result['training_end'] = event
            elif event_type == 'epoch_train':
                epoch_train_cache[event.get('epoch')] = event
            elif event_type == 'epoch_valid':
                epoch = event.get('epoch')
                train_event = epoch_train_cache.get(epoch, {})
                result['epoch_history'].append({
                    'epoch': epoch,
                    'train_metrics': train_event.get('metrics', {}),
                    'valid_metrics': event.get('metrics', {}),
                    'lr': train_event.get('lr'),
                    'is_best': event.get('is_best', False),
                })
            elif event_type == 'final_valid':
                result['final_valid'] = event
            elif event_type == 'final_test':
                result['final_test'] = event
            elif event_type == 'early_stop':
                result['early_stop'] = event
        except json.JSONDecodeError:
            continue

    return result


def format_duration(seconds: float) -> str:
    """格式化时长"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}min"


def format_metric(value: Any, precision: int = 8) -> str:
    """格式化指标值"""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def find_plot_files(log_dir: str) -> Dict[str, str]:
    """查找 plots 目录下的图表文件"""
    plots_dir = os.path.join(log_dir, 'plots')
    plot_files = {}

    if not os.path.isdir(plots_dir):
        return plot_files

    expected_plots = {
        'loss_curve.png': '训练损失曲线',
        'metrics_curve.png': '验证指标随训练变化',
        'lr_curve.png': '学习率调度曲线',
        'metrics_comparison.png': '验证与测试指标对比',
        'training_summary.png': '训练摘要',
        'sample_comparison.png': '测试样本 LR/SR/HR 对比',
    }

    for filename, description in expected_plots.items():
        filepath = os.path.join(plots_dir, filename)
        if os.path.exists(filepath):
            plot_files[filename] = description

    return plot_files


def generate_report(log_dir: str, yaml_config: Optional[Dict], log_data: Dict) -> str:
    """按照 train_report.md 模板生成报告"""
    lines = []

    start_event = log_data.get('training_start') or {}
    end_event = log_data.get('training_end') or {}
    final_test = log_data.get('final_test') or {}
    final_valid = log_data.get('final_valid') or {}
    epoch_history = log_data.get('epoch_history') or []
    early_stop = log_data.get('early_stop')

    model_name = start_event.get('model_name') or (yaml_config or {}).get('model', {}).get('name', 'N/A')
    dataset_name = start_event.get('dataset_name') or (yaml_config or {}).get('data', {}).get('name', 'N/A')
    model_params = start_event.get('model_params', 'N/A')
    training_duration = end_event.get('training_duration_seconds', 0)
    best_epoch = end_event.get('best_epoch', final_test.get('best_epoch', 'N/A'))

    test_metrics = final_test.get('metrics') or end_event.get('final_test_metrics') or {}
    valid_metrics = final_valid.get('metrics') or end_event.get('final_valid_metrics') or {}

    # 查找图表文件
    plot_files = find_plot_files(log_dir)

    # ========== 报告头部 ==========
    lines.append("# 海洋超分辨率模型训练报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**模型**: {model_name}\n")
    lines.append(f"**数据集**: {dataset_name}\n")
    lines.append(f"**训练时长**: {format_duration(training_duration)}\n")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ========== 执行摘要 ==========
    lines.append("## 执行摘要")
    lines.append("")
    lines.append("### 主要成果")
    lines.append(f"- {'✅' if end_event else '❌'} **模型训练**: {'训练完成' if end_event else '训练未完成或日志缺失'}")
    lines.append(f"- {'✅' if test_metrics else '❌'} **测试性能**: {'已评估' if test_metrics else '未评估'}")
    lines.append(f"- ✅ **模型检查点**: best_model.pth")
    lines.append(f"- {'✅' if plot_files else '⚪'} **可视化**: {'已生成 ' + str(len(plot_files)) + ' 个图表' if plot_files else '未生成'}")
    lines.append(f"- {'✅' if early_stop else '⚪'} **训练稳定性**: {'触发早停' if early_stop else '正常完成'}")
    lines.append("")

    lines.append("### 关键指标")
    lines.append("")
    lines.append(f"- **参数量**: {model_params}M" if isinstance(model_params, (int, float)) else f"- **参数量**: {model_params}")
    lines.append(f"- **训练模式**: {'分布式' if start_event.get('distribute') else '单卡'}")
    lines.append(f"- **最佳 Epoch**: {best_epoch}")
    lines.append(f"- **最终 test_loss**: {format_metric(test_metrics.get('test_loss'))}")
    lines.append(f"- **最终 MSE**: {format_metric(test_metrics.get('mse'))}")
    lines.append(f"- **最终 RMSE**: {format_metric(test_metrics.get('rmse'))}")
    lines.append(f"- **最终 PSNR**: {format_metric(test_metrics.get('psnr'))}")
    lines.append(f"- **最终 SSIM**: {format_metric(test_metrics.get('ssim'))}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ========== 1. 训练配置 ==========
    lines.append("## 1. 训练配置")
    lines.append("")

    lines.append("### 1.1 模型结构")
    lines.append("")
    lines.append("| 配置项 | 值 |")
    lines.append("|--------|-----|")
    lines.append(f"| **模型名称** | {model_name} |")
    model_cfg = (yaml_config or {}).get('model', {})
    lines.append(f"| **参数量** | {model_params}M |" if isinstance(model_params, (int, float)) else f"| **参数量** | {model_params} |")
    for key in ['in_channels', 'out_channels', 'hidden_channels', 'num_layers']:
        if key in model_cfg:
            lines.append(f"| **{key}** | {model_cfg[key]} |")
    lines.append("")

    lines.append("### 1.2 数据配置")
    lines.append("")
    lines.append("| 配置项 | 值 |")
    lines.append("|--------|-----|")
    data_cfg = (yaml_config or {}).get('data', {})
    lines.append(f"| **数据集路径** | {data_cfg.get('dataset_root', 'N/A')} |")
    shape = data_cfg.get('shape', [])
    if shape and len(shape) >= 2:
        lines.append(f"| **空间分辨率 (H x W)** | {shape[-2]} x {shape[-1]} |")
    dyn_vars = data_cfg.get('dyn_vars', [])
    if dyn_vars:
        lines.append(f"| **输入变量** | {', '.join(dyn_vars)} |")
    lines.append(f"| **训练集样本数** | {start_event.get('train_samples', 'N/A')} |")
    lines.append(f"| **验证集样本数** | {start_event.get('valid_samples', 'N/A')} |")
    lines.append(f"| **测试集样本数** | {start_event.get('test_samples', 'N/A')} |")
    lines.append("")

    lines.append("### 1.3 训练超参数")
    lines.append("")
    lines.append("| 配置项 | 值 |")
    lines.append("|--------|-----|")
    lines.append(f"| **总 Epochs** | {start_event.get('total_epochs', 'N/A')} |")
    lines.append(f"| **批次大小 (batch_size)** | {start_event.get('batch_size', 'N/A')} |")
    lines.append(f"| **优化器** | {start_event.get('optimizer', 'N/A')} |")
    lines.append(f"| **初始学习率** | {start_event.get('learning_rate', 'N/A')} |")
    schedule_cfg = (yaml_config or {}).get('schedule', {})
    lines.append(f"| **学习率调度器** | {schedule_cfg.get('scheduler', 'N/A')} |")
    optim_cfg = (yaml_config or {}).get('optimize', {})
    lines.append(f"| **权重衰减 (weight_decay)** | {optim_cfg.get('weight_decay', 'N/A')} |")
    lines.append(f"| **早停耐心度 (patience)** | {start_event.get('patience', 'N/A')} |")
    lines.append(f"| **评估频率 (eval_freq)** | {start_event.get('eval_freq', 'N/A')} |")
    lines.append(f"| **损失函数** | MaskedLpLoss |")
    lines.append("")

    lines.append("### 1.4 硬件配置")
    lines.append("")
    lines.append("| 配置项 | 值 |")
    lines.append("|--------|-----|")
    lines.append(f"| **设备** | {start_event.get('device', 'N/A')} |")
    lines.append(f"| **训练模式** | {'分布式 (' + str(start_event.get('distribute_mode')) + ')' if start_event.get('distribute') else '单卡'} |")
    mask_info = start_event.get('mask_hr_info')
    if mask_info and mask_info.get('total_pixels'):
        ocean = mask_info.get('ocean_pixels', 0)
        total = mask_info.get('total_pixels', 1)
        land_pct = (1 - ocean / total) * 100
        lines.append(f"| **HR 掩码** | {ocean:,} 海洋 / {total:,} 总像素 ({land_pct:.1f}% 陆地) |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ========== 2. 训练过程 ==========
    lines.append("## 2. 训练过程")
    lines.append("")

    lines.append("### 2.1 训练时间线")
    lines.append("")
    lines.append("| 时间 | 事件 | 说明 |")
    lines.append("|------|------|------|")
    if start_event.get('timestamp'):
        lines.append(f"| {start_event['timestamp'][:19]} | 训练开始 | {model_name} on {dataset_name} |")
    if best_epoch != 'N/A':
        lines.append(f"| - | 最佳模型保存 | Epoch {best_epoch}, valid_loss = {format_metric(valid_metrics.get('valid_loss'), 6)} |")
    if early_stop:
        ts = early_stop.get('timestamp', '-')[:19] if early_stop.get('timestamp') else '-'
        lines.append(f"| {ts} | 早停触发 | Epoch {early_stop.get('epoch')}, patience = {early_stop.get('patience')} |")
    if end_event.get('timestamp'):
        lines.append(f"| {end_event['timestamp'][:19]} | 训练结束 | 共 {end_event.get('actual_epochs', 'N/A')} epochs |")
    lines.append("")

    lines.append("### 2.2 训练曲线")
    lines.append("")
    lines.append("#### 损失下降趋势")
    lines.append("")
    lines.append("<!-- AI_FILL: 描述 train_loss 和 valid_loss 的下降趋势，分析收敛情况 -->")
    lines.append("")

    lines.append("### 2.3 验证集性能演进")
    lines.append("")
    if epoch_history:
        lines.append("| Epoch | valid_loss | mse | rmse | psnr | ssim |")
        lines.append("|-------|------------|-----|------|------|------|")
        for record in epoch_history[-10:]:
            vm = record.get('valid_metrics', {})
            lines.append(f"| {record['epoch']} | {format_metric(vm.get('valid_loss'), 6)} | {format_metric(vm.get('mse'), 6)} | {format_metric(vm.get('rmse'), 6)} | {format_metric(vm.get('psnr'), 4)} | {format_metric(vm.get('ssim'), 6)} |")
    else:
        lines.append("*无验证历史数据*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ========== 3. 最终性能评估 ==========
    lines.append("## 3. 最终性能评估")
    lines.append("")

    lines.append("### 3.1 验证集指标")
    lines.append("")
    lines.append("| 指标 | 值 | 说明 |")
    lines.append("|------|-----|------|")
    lines.append(f"| **valid_loss** | {format_metric(valid_metrics.get('valid_loss'))} | 验证集损失 |")
    lines.append(f"| **mse** | {format_metric(valid_metrics.get('mse'))} | 均方误差 |")
    lines.append(f"| **rmse** | {format_metric(valid_metrics.get('rmse'))} | 均方根误差 |")
    lines.append(f"| **psnr** | {format_metric(valid_metrics.get('psnr'))} | 峰值信噪比 (dB) |")
    lines.append(f"| **ssim** | {format_metric(valid_metrics.get('ssim'))} | 结构相似性 [0, 1] |")
    lines.append("")

    lines.append("### 3.2 测试集指标 (最终评估)")
    lines.append("")
    lines.append("| 指标 | 值 | 说明 |")
    lines.append("|------|-----|------|")
    lines.append(f"| **test_loss** | {format_metric(test_metrics.get('test_loss'))} | 测试集损失 |")
    lines.append(f"| **mse** | {format_metric(test_metrics.get('mse'))} | 均方误差 |")
    lines.append(f"| **rmse** | {format_metric(test_metrics.get('rmse'))} | 均方根误差 |")
    lines.append(f"| **psnr** | {format_metric(test_metrics.get('psnr'))} | 峰值信噪比 (dB) |")
    lines.append(f"| **ssim** | {format_metric(test_metrics.get('ssim'))} | 结构相似性 [0, 1] |")
    lines.append("")

    lines.append("### 3.3 性能对比")
    lines.append("")
    lines.append("<!-- AI_FILL: 对比分析验证集与测试集的性能差异，评估模型泛化能力 -->")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ========== 4. 可视化结果 ==========
    lines.append("## 4. 可视化结果")
    lines.append("")

    lines.append("### 4.1 生成的可视化文件")
    lines.append("")
    if plot_files:
        lines.append("| 文件名 | 说明 |")
        lines.append("|--------|------|")
        for filename, description in plot_files.items():
            lines.append(f"| {filename} | {description} |")
    else:
        lines.append("*未生成可视化文件，请先运行 `ocean_sr_visualize` 工具*")
    lines.append("")

    lines.append("### 4.2 可视化内容说明")
    lines.append("")
    lines.append("<!-- AI_FILL: 分析可视化图表内容，说明每个图表展示的信息和关键发现 -->")
    lines.append("")

    lines.append("### 4.3 可视化图表展示")
    lines.append("")
    if plot_files:
        # Loss curve
        if 'loss_curve.png' in plot_files:
            lines.append("#### 训练损失曲线")
            lines.append("")
            lines.append("![训练损失曲线](plots/loss_curve.png)")
            lines.append("")

        # Metrics curve
        if 'metrics_curve.png' in plot_files:
            lines.append("#### 验证集指标变化曲线")
            lines.append("")
            lines.append("![验证集指标变化曲线](plots/metrics_curve.png)")
            lines.append("")

        # Learning rate curve
        if 'lr_curve.png' in plot_files:
            lines.append("#### 学习率变化曲线")
            lines.append("")
            lines.append("![学习率变化曲线](plots/lr_curve.png)")
            lines.append("")

        # Metrics comparison
        if 'metrics_comparison.png' in plot_files:
            lines.append("#### 验证集与测试集指标对比")
            lines.append("")
            lines.append("![验证集与测试集指标对比](plots/metrics_comparison.png)")
            lines.append("")

        # Training summary
        if 'training_summary.png' in plot_files:
            lines.append("#### 训练总结")
            lines.append("")
            lines.append("![训练总结](plots/training_summary.png)")
            lines.append("")

        # Sample comparison (LR / SR / HR)
        if 'sample_comparison.png' in plot_files:
            lines.append("#### 测试样本对比 (LR / SR / HR)")
            lines.append("")
            lines.append("下图展示了测试集样本的低分辨率输入、模型超分辨率输出、高分辨率真值以及绝对误差分布：")
            lines.append("")
            lines.append("![测试样本对比](plots/sample_comparison.png)")
            lines.append("")
    else:
        lines.append("*无可视化图表*")
        lines.append("")

    lines.append("---")
    lines.append("")

    # ========== 5. 模型检查点 ==========
    lines.append("## 5. 模型检查点")
    lines.append("")
    lines.append("### 5.1 保存的检查点")
    lines.append("")
    lines.append("| 文件名 | 类型 | 说明 |")
    lines.append("|--------|------|------|")
    lines.append(f"| best_model.pth | 最佳模型 | Epoch {best_epoch}, valid_loss = {format_metric(valid_metrics.get('valid_loss'), 6)} |")
    lines.append("")

    lines.append("### 5.2 辅助文件")
    lines.append("")
    lines.append("| 文件名 | 说明 |")
    lines.append("|--------|------|")
    lines.append("| config.yaml | 训练配置备份 |")
    lines.append("| train.log | 训练日志 |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ========== 6. 训练分析 ==========
    lines.append("## 6. 训练分析")
    lines.append("")
    lines.append("### 6.1 训练稳定性")
    lines.append("")
    lines.append("<!-- AI_FILL: 分析训练过程的稳定性，包括：loss 下降趋势、是否有异常波动、收敛速度评估 -->")
    lines.append("")

    lines.append("### 6.2 模型性能分析")
    lines.append("")
    lines.append("<!-- AI_FILL: 分析模型性能，包括：各指标变化趋势、与预期目标的对比、性能瓶颈分析 -->")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ========== 7. 计算性能 ==========
    lines.append("## 7. 计算性能")
    lines.append("")
    lines.append("### 7.1 训练效率")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| **总训练时长** | {format_duration(training_duration)} |")
    actual_epochs = end_event.get('actual_epochs', 0)
    if actual_epochs > 0 and training_duration > 0:
        avg_epoch_time = training_duration / actual_epochs
        lines.append(f"| **平均每 Epoch 耗时** | {format_duration(avg_epoch_time)} |")
    lines.append(f"| **实际训练 Epochs** | {actual_epochs} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ========== 8. 总结 ==========
    lines.append("## 8. 总结")
    lines.append("")
    lines.append("### 8.1 核心成就")
    lines.append("")
    lines.append("<!-- AI_FILL: 总结本次训练的核心成就，包括：模型性能亮点、训练效率、达成的目标（3-5点） -->")
    lines.append("")

    lines.append("### 8.2 关键数据")
    lines.append("")
    lines.append("| 项目 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| **训练样本数** | {start_event.get('train_samples', 'N/A')} |")
    lines.append(f"| **验证样本数** | {start_event.get('valid_samples', 'N/A')} |")
    lines.append(f"| **测试样本数** | {start_event.get('test_samples', 'N/A')} |")
    lines.append(f"| **模型参数量** | {model_params}M |" if isinstance(model_params, (int, float)) else f"| **模型参数量** | {model_params} |")
    lines.append(f"| **总训练时长** | {format_duration(training_duration)} |")
    lines.append(f"| **最佳 Epoch** | {best_epoch} |")
    lines.append(f"| **最终 test_loss** | {format_metric(test_metrics.get('test_loss'))} |")
    lines.append(f"| **最终 MSE** | {format_metric(test_metrics.get('mse'))} |")
    lines.append(f"| **最终 RMSE** | {format_metric(test_metrics.get('rmse'))} |")
    lines.append(f"| **最终 PSNR** | {format_metric(test_metrics.get('psnr'))} |")
    lines.append(f"| **最终 SSIM** | {format_metric(test_metrics.get('ssim'))} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*报告由 Ocean-Agent-SDK 训练报告系统自动生成*")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="生成海洋超分辨率训练 Markdown 报告")
    parser.add_argument('--log_dir', required=True, type=str, help='训练日志目录路径')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径（默认为 log_dir/training_report.md）')
    args = parser.parse_args()

    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        print(f"[Error] 日志目录不存在: {log_dir}")
        sys.exit(1)

    yaml_config = None
    for name in ['config.yaml', 'config.yml']:
        yaml_path = os.path.join(log_dir, name)
        cfg = load_yaml_safe(yaml_path)
        if cfg:
            yaml_config = cfg
            break

    log_data = {}
    for name in ['train.log', 'training.log']:
        log_path = os.path.join(log_dir, name)
        if os.path.exists(log_path):
            log_data = parse_structured_log(log_path)
            break

    if not log_data.get('training_start') and not yaml_config:
        print(f"[Warning] 未找到结构化日志事件，也未找到配置文件，报告可能不完整")

    try:
        report_content = generate_report(log_dir, yaml_config, log_data)
        output_path = args.output or os.path.join(log_dir, 'training_report.md')

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"[Success] 训练报告已生成: {output_path}")

    except Exception as e:
        print(f"[Error] 报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
