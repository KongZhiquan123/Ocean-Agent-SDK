#!/usr/bin/env python3
"""
generate_training_report.py - 海洋超分辨率训练报告生成脚本

@author Leizheng
@date 2026-02-06
@version 1.0.0

功能:
- 从训练日志目录读取 config.yaml + train.log
- 记录 4 阶段用户确认信息
- 生成包含训练概览、评估结果、掩码信息的 Markdown 报告
- 添加 Agent 分析占位符

用法:
    python generate_training_report.py --config report_config.json

输出:
    log_dir/training_report.md

Changelog:
    - 2026-02-06 Leizheng v1.0.0: 初始版本
        - 训练概览（模型、数据集、参数、GPU）
        - 用户确认记录（4 阶段）
        - 训练过程（loss、早停、最佳 epoch）
        - 评估结果（MSE/RMSE/PSNR/SSIM）
        - 掩码信息（陆地像素占比）
        - Agent 分析占位符
"""

import os
import sys
import json
import re
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any


def load_json_safe(file_path: str) -> Optional[Dict]:
    """安全加载 JSON 文件"""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[Warning] 无法加载 {file_path}: {e}")
        return None


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


def parse_train_log(log_path: str) -> Dict[str, Any]:
    """
    解析训练日志，提取关键信息。

    Returns:
        dict with keys:
        - epochs_trained: int
        - best_epoch: int or None
        - early_stopped: bool
        - final_train_loss: float or None
        - valid_metrics: dict or None
        - test_metrics: dict or None
        - mask_info: dict or None (HR/LR mask statistics)
    """
    result = {
        'epochs_trained': 0,
        'best_epoch': None,
        'early_stopped': False,
        'final_train_loss': None,
        'valid_metrics': None,
        'test_metrics': None,
        'mask_info': None,
    }

    if not os.path.exists(log_path):
        return result

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return result

    # 解析 epoch 信息
    epoch_pattern = re.compile(r'Epoch (\d+) \|')
    epochs = epoch_pattern.findall(content)
    if epochs:
        result['epochs_trained'] = max(int(e) for e in epochs) + 1

    # 解析早停
    if 'Early stop at epoch' in content:
        result['early_stopped'] = True
        early_match = re.search(r'Early stop at epoch (\d+)', content)
        if early_match:
            result['early_stopped_epoch'] = int(early_match.group(1))

    # 解析最后的 train_loss
    train_loss_pattern = re.compile(r'train_loss: ([\d.]+)')
    train_losses = train_loss_pattern.findall(content)
    if train_losses:
        result['final_train_loss'] = float(train_losses[-1])

    # 解析 Valid metrics
    valid_match = re.search(r'Valid metrics:.*?(\{.*?\}|valid_loss.*?)$', content, re.MULTILINE)
    if not valid_match:
        # 尝试解析格式化输出
        valid_metrics_pattern = re.compile(
            r'Valid metrics:.*?valid_loss: ([\d.eE+-]+).*?mse: ([\d.eE+-]+).*?rmse: ([\d.eE+-]+).*?psnr: ([\d.eE+-]+).*?ssim: ([\d.eE+-]+)'
        )
        vm = valid_metrics_pattern.search(content)
        if vm:
            result['valid_metrics'] = {
                'valid_loss': float(vm.group(1)),
                'mse': float(vm.group(2)),
                'rmse': float(vm.group(3)),
                'psnr': float(vm.group(4)),
                'ssim': float(vm.group(5)),
            }

    # 解析 Test metrics
    test_metrics_pattern = re.compile(
        r'Test metrics:.*?test_loss: ([\d.eE+-]+).*?mse: ([\d.eE+-]+).*?rmse: ([\d.eE+-]+).*?psnr: ([\d.eE+-]+).*?ssim: ([\d.eE+-]+)'
    )
    tm = test_metrics_pattern.search(content)
    if tm:
        result['test_metrics'] = {
            'test_loss': float(tm.group(1)),
            'mse': float(tm.group(2)),
            'rmse': float(tm.group(3)),
            'psnr': float(tm.group(4)),
            'ssim': float(tm.group(5)),
        }

    # 解析掩码信息
    hr_mask_pattern = re.compile(r'HR mask: (\d+)/(\d+) ocean pixels \((\d+) land, ([\d.]+)%\)')
    hr_match = hr_mask_pattern.search(content)
    lr_mask_pattern = re.compile(r'LR mask: (\d+)/(\d+) ocean pixels \((\d+) land, ([\d.]+)%\)')
    lr_match = lr_mask_pattern.search(content)

    if hr_match or lr_match:
        result['mask_info'] = {}
        if hr_match:
            result['mask_info']['hr'] = {
                'ocean_pixels': int(hr_match.group(1)),
                'total_pixels': int(hr_match.group(2)),
                'land_pixels': int(hr_match.group(3)),
                'land_percentage': float(hr_match.group(4)),
            }
        if lr_match:
            result['mask_info']['lr'] = {
                'ocean_pixels': int(lr_match.group(1)),
                'total_pixels': int(lr_match.group(2)),
                'land_pixels': int(lr_match.group(3)),
                'land_percentage': float(lr_match.group(4)),
            }

    return result


def analyze_training_quality() -> str:
    """生成训练质量分析占位符"""
    return """<!-- AGENT_ANALYSIS_PLACEHOLDER

⚠️ **重要提示**: 此部分需要由 Agent 根据实际训练结果进行分析。

Agent 应该：
1. 仔细阅读上述所有数据（训练概览、训练过程、评估结果、掩码信息等）
2. 识别关键问题和亮点
3. 提供具体的、有针对性的分析和建议

分析应包括但不限于：
- **训练收敛性**: loss 是否收敛？是否有过拟合迹象？
- **评估指标分析**: MSE/RMSE/PSNR/SSIM 各指标是否合理？哪些指标表现好/差？
- **掩码效果评估**: 陆地掩码是否正确排除了陆地像素？掩码比例是否合理？
- **模型选择评估**: 当前模型是否适合此数据集？是否需要尝试其他模型？
- **参数调优建议**: 学习率、batch_size、epochs 等参数是否合适？
- **数据质量**: 训练数据量是否充足？数据划分是否合理？
- **改进方向**: 下一步可以尝试哪些优化？

请用清晰、专业的语言编写分析，避免模板化的内容。

-->"""


def generate_report(config: Dict) -> str:
    """生成训练报告 Markdown"""

    log_dir = config['log_dir']
    user_confirmation = config.get('user_confirmation', {})

    # 尝试加载配置文件
    yaml_config = None
    for name in ['config.yaml', 'config.yml']:
        yaml_path = os.path.join(log_dir, name)
        cfg = load_yaml_safe(yaml_path)
        if cfg:
            yaml_config = cfg
            break

    # 解析训练日志
    log_info = {}
    for name in ['train.log', 'training.log']:
        log_path = os.path.join(log_dir, name)
        if os.path.exists(log_path):
            log_info = parse_train_log(log_path)
            break

    # 开始生成报告
    lines = []
    lines.append("# 海洋超分辨率训练报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**日志目录**: `{log_dir}`")
    lines.append(f"**框架版本**: v2.0.0（陆地掩码版）")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ========================================
    # 1. 训练概览
    # ========================================
    lines.append("## 1. 训练概览")
    lines.append("")

    if yaml_config:
        model_cfg = yaml_config.get('model', {})
        data_cfg = yaml_config.get('data', {})
        train_cfg = yaml_config.get('train', {})
        optim_cfg = yaml_config.get('optimize', {})

        lines.append("### 1.1 模型信息")
        lines.append("")
        lines.append(f"- **模型名称**: `{model_cfg.get('name', 'N/A')}`")
        lines.append(f"- **数据集**: `{data_cfg.get('name', 'N/A')}`")
        lines.append(f"- **数据目录**: `{data_cfg.get('dataset_root', 'N/A')}`")

        dyn_vars = data_cfg.get('dyn_vars', [])
        if dyn_vars:
            lines.append(f"- **动态变量**: {', '.join([f'`{v}`' for v in dyn_vars])}")

        shape = data_cfg.get('shape', [])
        if shape:
            lines.append(f"- **数据形状**: `{shape}`")
        lines.append("")

        lines.append("### 1.2 训练参数")
        lines.append("")
        lines.append(f"- **训练轮数**: {train_cfg.get('epochs', 'N/A')}")
        lines.append(f"- **学习率**: {optim_cfg.get('lr', 'N/A')}")
        lines.append(f"- **优化器**: {optim_cfg.get('optimizer', 'N/A')}")
        lines.append(f"- **Batch Size**: {data_cfg.get('train_batchsize', 'N/A')}")
        lines.append(f"- **早停耐心值**: {train_cfg.get('patience', 'N/A')}")
        lines.append(f"- **评估频率**: {train_cfg.get('eval_freq', 'N/A')}")
        lines.append(f"- **归一化**: {data_cfg.get('normalize', True)}")
        lines.append(f"- **归一化类型**: {data_cfg.get('normalizer_type', 'PGN')}")
        lines.append("")

        # GPU 信息
        distribute = train_cfg.get('distribute', False)
        if distribute:
            lines.append("### 1.3 GPU 配置")
            lines.append("")
            lines.append(f"- **分布式模式**: {train_cfg.get('distribute_mode', 'N/A')}")
            device_ids = train_cfg.get('device_ids', [])
            if device_ids:
                lines.append(f"- **GPU 列表**: {device_ids}")
            lines.append("")
    else:
        lines.append("⚠️ 未找到训练配置文件 (config.yaml)")
        lines.append("")

    # ========================================
    # 2. 用户确认记录
    # ========================================
    lines.append("## 2. 用户确认记录")
    lines.append("")
    lines.append("以下是训练流程中用户确认的关键选择（4 阶段确认）：")
    lines.append("")

    # 阶段 1: 数据确认
    lines.append("### 2.1 阶段 1：数据确认")
    lines.append("")
    stage1 = user_confirmation.get('stage1_data', {})
    if stage1:
        lines.append(f"- **数据目录**: `{stage1.get('dataset_root', 'N/A')}`")
        lines.append(f"- **输出目录**: `{stage1.get('log_dir', 'N/A')}`")
        dyn_vars = stage1.get('dyn_vars', [])
        if dyn_vars:
            lines.append(f"- **检测到的变量**: {', '.join([f'`{v}`' for v in dyn_vars])}")
        lines.append(f"- **超分辨率倍数**: {stage1.get('scale', 'N/A')}")
        if stage1.get('confirmed_at'):
            lines.append(f"- **确认时间**: {stage1.get('confirmed_at')}")
    else:
        lines.append("- ⚠️ 未记录")
    lines.append("")

    # 阶段 2: 模型选择
    lines.append("### 2.2 阶段 2：模型选择")
    lines.append("")
    stage2 = user_confirmation.get('stage2_model', {})
    if stage2:
        lines.append(f"- **选择的模型**: `{stage2.get('model_name', 'N/A')}`")
        if stage2.get('confirmed_at'):
            lines.append(f"- **确认时间**: {stage2.get('confirmed_at')}")
    else:
        lines.append("- ⚠️ 未记录")
    lines.append("")

    # 阶段 3: 参数确认
    lines.append("### 2.3 阶段 3：参数确认（含 GPU）")
    lines.append("")
    stage3 = user_confirmation.get('stage3_parameters', {})
    if stage3:
        lines.append(f"- **训练轮数**: {stage3.get('epochs', 'N/A')}")
        lines.append(f"- **学习率**: {stage3.get('lr', 'N/A')}")
        lines.append(f"- **Batch Size**: {stage3.get('batch_size', 'N/A')}")
        lines.append(f"- **GPU**: {stage3.get('device_ids', 'N/A')}")
        if stage3.get('confirmed_at'):
            lines.append(f"- **确认时间**: {stage3.get('confirmed_at')}")
    else:
        lines.append("- ⚠️ 未记录")
    lines.append("")

    # 阶段 4: 执行确认
    lines.append("### 2.4 阶段 4：执行确认")
    lines.append("")
    stage4 = user_confirmation.get('stage4_execution', {})
    if stage4:
        lines.append(f"- **用户确认执行**: {'✓ 是' if stage4.get('confirmed') else '✗ 否'}")
        if stage4.get('confirmed_at'):
            lines.append(f"- **确认时间**: {stage4.get('confirmed_at')}")
    else:
        lines.append("- ⚠️ 未记录")
    lines.append("")

    # ========================================
    # 3. 训练过程
    # ========================================
    lines.append("## 3. 训练过程")
    lines.append("")

    if log_info:
        lines.append(f"- **实际训练轮数**: {log_info.get('epochs_trained', 'N/A')}")

        if log_info.get('early_stopped'):
            lines.append(f"- **早停**: ✓ 在 epoch {log_info.get('early_stopped_epoch', 'N/A')} 触发")
        else:
            lines.append("- **早停**: ✗ 未触发")

        if log_info.get('final_train_loss') is not None:
            lines.append(f"- **最终 Train Loss**: {log_info['final_train_loss']:.8f}")

        lines.append("")
    else:
        lines.append("⚠️ 未找到训练日志 (train.log)")
        lines.append("")

    # ========================================
    # 4. 评估结果
    # ========================================
    lines.append("## 4. 评估结果")
    lines.append("")

    for idx, (split_name, metrics_key) in enumerate([('验证集', 'valid_metrics'), ('测试集', 'test_metrics')], start=1):
        metrics = log_info.get(metrics_key)
        if metrics:
            lines.append(f"### 4.{idx} {split_name}")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            for key, value in sorted(metrics.items()):
                if isinstance(value, float):
                    lines.append(f"| {key} | {value:.8f} |")
                else:
                    lines.append(f"| {key} | {value} |")
            lines.append("")

    if not log_info.get('valid_metrics') and not log_info.get('test_metrics'):
        lines.append("⚠️ 未找到评估结果（训练可能尚未完成或日志格式不匹配）")
        lines.append("")

    lines.append("**指标说明**:")
    lines.append("")
    lines.append("- **MSE**: 均方误差，越小越好（仅海洋格点）")
    lines.append("- **RMSE**: 均方根误差，越小越好（仅海洋格点）")
    lines.append("- **PSNR**: 峰值信噪比，越大越好（仅海洋格点）")
    lines.append("- **SSIM**: 结构相似性，0~1，越接近 1 越好（仅海洋格点）")
    lines.append("")

    # ========================================
    # 5. 掩码信息
    # ========================================
    lines.append("## 5. 陆地掩码信息")
    lines.append("")

    mask_info = log_info.get('mask_info')
    if mask_info:
        lines.append("本次训练使用了陆地掩码，以下为掩码统计：")
        lines.append("")

        if 'hr' in mask_info:
            hr = mask_info['hr']
            lines.append(f"### HR 掩码")
            lines.append("")
            lines.append(f"- **海洋像素**: {hr['ocean_pixels']:,}")
            lines.append(f"- **陆地像素**: {hr['land_pixels']:,}")
            lines.append(f"- **总像素**: {hr['total_pixels']:,}")
            lines.append(f"- **陆地占比**: {hr['land_percentage']:.1f}%")
            lines.append("")

        if 'lr' in mask_info:
            lr = mask_info['lr']
            lines.append(f"### LR 掩码")
            lines.append("")
            lines.append(f"- **海洋像素**: {lr['ocean_pixels']:,}")
            lines.append(f"- **陆地像素**: {lr['land_pixels']:,}")
            lines.append(f"- **总像素**: {lr['total_pixels']:,}")
            lines.append(f"- **陆地占比**: {lr['land_percentage']:.1f}%")
            lines.append("")

        lines.append("**掩码策略**：")
        lines.append("")
        lines.append("- 训练时：损失函数（MaskedLpLoss）只在海洋格点上计算")
        lines.append("- 扩散模型：loss 归一化分母使用有效像素数而非总像素数")
        lines.append("- 评估时：MSE/RMSE/PSNR/SSIM 都只在海洋格点上计算")
        lines.append("- NaN 处理：加载数据时将 NaN 填充为 0，通过 mask 排除陆地格点")
        lines.append("")
    else:
        lines.append("⚠️ 未检测到掩码信息（可能数据中没有 NaN / 陆地像素）")
        lines.append("")

    # ========================================
    # 6. 分析和建议
    # ========================================
    lines.append("## 6. 分析和建议")
    lines.append("")
    lines.append(analyze_training_quality())
    lines.append("")

    # ========================================
    # 7. 总结
    # ========================================
    lines.append("## 7. 总结")
    lines.append("")

    model_name = 'N/A'
    if yaml_config:
        model_name = yaml_config.get('model', {}).get('name', 'N/A')

    lines.append(f"本次使用 **{model_name}** 模型进行训练，")
    lines.append(f"共训练 **{log_info.get('epochs_trained', 'N/A')}** 个 epoch。")

    if log_info.get('early_stopped'):
        lines.append(f"训练在 epoch {log_info.get('early_stopped_epoch', 'N/A')} 触发早停。")

    if mask_info:
        hr_land_pct = mask_info.get('hr', {}).get('land_percentage', 0)
        lines.append(f"使用陆地掩码排除了 {hr_land_pct:.1f}% 的陆地像素。")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*报告由 Ocean-Agent-SDK 训练报告系统自动生成*")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="生成海洋超分辨率训练 Markdown 报告"
    )

    parser.add_argument(
        '--config',
        required=True,
        type=str,
        help='配置文件路径 (JSON 格式)'
    )

    args = parser.parse_args()

    # 加载配置
    if not os.path.exists(args.config):
        print(f"[Error] 配置文件不存在: {args.config}")
        sys.exit(1)

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 生成报告
    try:
        report_content = generate_report(config)

        # 写入文件
        output_path = config['output_path']
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
