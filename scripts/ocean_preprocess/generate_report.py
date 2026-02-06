#!/usr/bin/env python3
"""
generate_report.py - 海洋数据预处理报告生成脚本

@author kongzhiquan
@contributors leizheng
@date 2026-02-04
@version 3.1.2

功能:
- 整合预处理流程中的所有关键信息
- 记录 4 阶段用户确认信息
- 生成包含可视化图片的 Markdown 报告
- 添加数据质量分析和建议

用法:
    python generate_report.py --config report_config.json

输出:
    dataset_root/preprocessing_report.md

Changelog:
    - 2026-02-04 kongzhiquan v3.1.2: 支持显示多个静态变量的不同形状
        - 静态变量可能有不同维度（如一维的纬度经度就不同）
        - 改为列表形式显示每个静态变量的形状
    - 2026-02-04 kongzhiquan v3.1.1: 修复静态变量形状显示 N/A 的问题
        - 在查找静态变量形状时，也检查 coordinate 类别
        - 用户可能选择坐标变量（如 latitude, longitude）作为静态变量
    - 2026-02-04 kongzhiquan v3.1.0: 合并两个版本的功能
        - 同时读取 convert_result.json 和 preprocess_manifest.json
        - convert_result.json: 完整的转换结果、后置验证、warnings、errors
        - preprocess_manifest.json: 输入配置信息（dyn_vars, stat_vars等）
        - 新增 Section 2: 用户确认记录（4 阶段）
        - 新增全局统计汇总图展示 (statistics_summary.png)
        - 分离展示空间对比图和统计分布图
        - 更新输出文件结构说明（逐时间步保存）
        - 新增"转换警告"和"转换错误"小节
        - 改进数据提取逻辑，优先级：convert.config > convert根级别 > manifest
    - 2026-02-04 kongzhiquan v1.0.0: 初始版本
        - 整合 inspect/validate/convert/metrics 结果
        - 嵌入可视化图片
        - 生成 AI 分析和建议
"""

import os
import sys
import json
import argparse
import glob
from datetime import datetime
from typing import Dict, List, Optional, Any

__all__ = [
    'load_json_safe',
    'format_shape',
    'analyze_data_quality',
    'generate_report',
]


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


def format_shape(shape: List[int]) -> str:
    """格式化形状信息"""
    return f"[{', '.join(map(str, shape))}]"


def analyze_data_quality() -> str:
    """
    生成数据质量分析占位符

    注意：此函数不再生成具体分析内容，而是返回占位符。
    实际分析应由 Agent 根据数据情况自行编写。
    """
    return """<!-- AGENT_ANALYSIS_PLACEHOLDER

⚠️ **重要提示**: 此部分需要由 Agent 根据实际数据情况进行分析。

Agent 应该：
1. 仔细阅读上述所有数据（数据集概览、用户确认记录、验证结果、质量指标等）
2. 识别关键问题和亮点
3. 提供具体的、有针对性的分析和建议

分析应包括但不限于：
- **用户选择评估**: 用户选择的研究变量是否合理？静态/掩码变量是否正确？
- **SSIM 指标分析**: 哪些变量的结构相似性好/差？为什么？
- **Relative L2 误差分析**: 误差分布是否合理？是否有异常值？
- **数据量评估**: 数据量是否充足？训练集/验证集/测试集划分是否合理？
- **变量选择建议**: 是否所有变量都需要？是否有冗余？
- **验证结果解读**: 所有验证规则是否通过？如果有警告，如何处理？
- **下采样质量评估**: 下采样方法是否合适？是否需要调整？
- **潜在问题识别**: 数据中是否存在异常、缺失值、不一致等问题？
- **改进建议**: 如何提升数据质量？如何优化预处理流程？

请用清晰、专业的语言编写分析，避免模板化的内容。

-->"""


def generate_report(config: Dict) -> str:
    """生成 Markdown 报告"""

    # 加载各个步骤的结果
    inspect = load_json_safe(config.get('inspect_result_path', ''))
    validate = load_json_safe(config.get('validate_result_path', ''))
    convert = load_json_safe(config.get('convert_result_path', ''))  # convert_result.json (完整结果)
    manifest = load_json_safe(config.get('manifest_path', ''))  # preprocess_manifest.json (输入配置)
    metrics = load_json_safe(config.get('metrics_result_path', ''))

    # 用户确认信息（从 config 中获取）
    user_confirmation = config.get('user_confirmation', {})

    dataset_root = config['dataset_root']

    # 开始生成报告
    lines = []
    lines.append("# 海洋数据预处理报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**数据集路径**: `{dataset_root}`\n")
    lines.append(f"**流程版本**: v3.0.0（4 阶段强制确认）\n")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ========================================
    # 1. 数据集概览
    # ========================================
    lines.append("## 1. 数据集概览")
    lines.append("")

    if inspect:
        lines.append("### 1.1 基本信息")
        lines.append("")
        lines.append(f"- **文件数量**: {inspect.get('file_count', 'N/A')}")
        lines.append(f"- **动态文件**: {len(inspect.get('dynamic_files', []))}")
        lines.append(f"- **疑似静态文件**: {len(inspect.get('suspected_static_files', []))}")
        lines.append("")

        lines.append("### 1.2 检测到的变量（原始）")
        lines.append("")

        dyn_vars = inspect.get('dynamic_vars_candidates', [])
        stat_vars = inspect.get('static_vars_found', [])
        mask_vars = inspect.get('mask_vars_found', [])

        lines.append(f"- **动态变量候选** ({len(dyn_vars)}): {', '.join([f'`{v}`' for v in dyn_vars]) if dyn_vars else '无'}")
        lines.append(f"- **静态变量候选** ({len(stat_vars)}): {', '.join([f'`{v}`' for v in stat_vars]) if stat_vars else '无'}")
        lines.append(f"- **掩码变量候选** ({len(mask_vars)}): {', '.join([f'`{v}`' for v in mask_vars]) if mask_vars else '无'}")
        lines.append("")

        # 变量详细信息表格
        if 'variables' in inspect and inspect['variables']:
            lines.append("### 1.3 变量详细信息")
            lines.append("")
            lines.append("| 变量名 | 类别 | 形状 | 数据类型 | 单位 |")
            lines.append("|--------|------|------|----------|------|")

            for var_name, var_info in sorted(inspect['variables'].items()):
                category = var_info.get('category', 'unknown')
                shape = format_shape(var_info.get('shape', []))
                dtype = var_info.get('dtype', 'N/A')
                units = var_info.get('units', '-')
                lines.append(f"| `{var_name}` | {category} | {shape} | {dtype} | {units} |")

            lines.append("")
    else:
        lines.append("⚠️ 未找到数据检查结果 (inspect_result.json)")
        lines.append("")

    # ========================================
    # 2. 用户确认记录
    # ========================================
    lines.append("## 2. 用户确认记录")
    lines.append("")
    lines.append("以下是预处理流程中用户确认的关键选择（4 阶段强制确认）：")
    lines.append("")

    # 阶段 1: 研究变量选择
    lines.append("### 2.1 阶段 1：研究变量选择")
    lines.append("")
    stage1 = user_confirmation.get('stage1_research_vars', {})
    if stage1:
        selected_vars = stage1.get('selected', [])
        lines.append(f"- **选择的研究变量**: {', '.join([f'`{v}`' for v in selected_vars]) if selected_vars else '无'}")
        if stage1.get('confirmed_at'):
            lines.append(f"- **确认时间**: {stage1.get('confirmed_at')}")
    else:
        lines.append("- ⚠️ 未记录")
    lines.append("")

    # 阶段 2: 静态/掩码变量选择
    lines.append("### 2.2 阶段 2：静态/掩码变量选择")
    lines.append("")
    stage2 = user_confirmation.get('stage2_static_mask', {})
    if stage2:
        static_vars = stage2.get('static_vars', [])
        mask_vars_confirmed = stage2.get('mask_vars', [])
        coord_vars = stage2.get('coord_vars', {})
        lines.append(f"- **静态变量**: {', '.join([f'`{v}`' for v in static_vars]) if static_vars else '无'}")
        lines.append(f"- **掩码变量**: {', '.join([f'`{v}`' for v in mask_vars_confirmed]) if mask_vars_confirmed else '无'}")
        if coord_vars:
            lines.append(f"- **坐标变量**: lon=`{coord_vars.get('lon', 'N/A')}`, lat=`{coord_vars.get('lat', 'N/A')}`")
        if stage2.get('confirmed_at'):
            lines.append(f"- **确认时间**: {stage2.get('confirmed_at')}")
    else:
        lines.append("- ⚠️ 未记录")
    lines.append("")

    # 阶段 3: 处理参数确认
    lines.append("### 2.3 阶段 3：处理参数确认")
    lines.append("")
    stage3 = user_confirmation.get('stage3_parameters', {})
    if stage3:
        lines.append(f"- **下采样倍数 (scale)**: {stage3.get('scale', 'N/A')}")
        lines.append(f"- **下采样方法**: {stage3.get('downsample_method', 'N/A')}")
        lines.append(f"- **数据集划分**: train={stage3.get('train_ratio', 'N/A')}, valid={stage3.get('valid_ratio', 'N/A')}, test={stage3.get('test_ratio', 'N/A')}")

        # 裁剪信息
        h_slice = stage3.get('h_slice')
        w_slice = stage3.get('w_slice')
        if h_slice or w_slice:
            lines.append(f"- **裁剪设置**: H=`{h_slice or '无'}`, W=`{w_slice or '无'}`")

        # 裁剪推荐
        crop_recommendation = stage3.get('crop_recommendation')
        if crop_recommendation:
            lines.append(f"- **系统推荐裁剪**: {crop_recommendation}")

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
        if stage4.get('execution_started_at'):
            lines.append(f"- **开始执行时间**: {stage4.get('execution_started_at')}")
    else:
        lines.append("- ⚠️ 未记录")
    lines.append("")

    # ========================================
    # 3. 验证结果
    # ========================================
    lines.append("## 3. 验证结果")
    lines.append("")

    if validate:
        lines.append("### 3.1 张量约定验证")
        lines.append("")
        lines.append(f"- **状态**: `{validate.get('status', 'N/A')}`")
        lines.append(f"- **研究变量**: {', '.join([f'`{v}`' for v in validate.get('research_vars', [])])}")
        lines.append("")

        if 'tensor_convention' in validate:
            tc = validate['tensor_convention']
            lines.append("**张量约定**:")
            lines.append("")

            # 从 tensor_convention 中提取动态变量形状
            dynamic_shape = tc.get('dynamic_shape', None)
            static_shape = tc.get('static_shape', None)

            # 如果没有汇总形状，从各个变量中提取
            if not dynamic_shape:
                for var_name, var_info in tc.items():
                    if isinstance(var_info, dict) and var_info.get('category') == 'dynamic':
                        shape = var_info.get('original_shape', [])
                        interpretation = var_info.get('interpretation', '')
                        if shape:
                            dynamic_shape = f"{format_shape(shape)} {interpretation}"
                            break

            # 收集所有静态变量的形状（静态变量可能有不同维度，如纬度 4330，经度 2000）
            static_shapes = []
            for var_name, var_info in tc.items():
                if isinstance(var_info, dict) and var_info.get('category') in ['static', 'coordinate']:
                    shape = var_info.get('original_shape', [])
                    interpretation = var_info.get('interpretation', '')
                    if shape:
                        static_shapes.append((var_name, shape, interpretation))

            lines.append(f"- 动态变量形状: `{dynamic_shape or 'N/A'}`")

            if static_shapes:
                if len(static_shapes) == 1:
                    # 只有一个静态变量，保持原有格式
                    var_name, shape, interpretation = static_shapes[0]
                    lines.append(f"- 静态变量形状: `{format_shape(shape)} {interpretation}`")
                else:
                    # 多个静态变量，分别显示每个变量的形状
                    lines.append("- 静态变量形状:")
                    for var_name, shape, interpretation in sorted(static_shapes, key=lambda x: x[0]):
                        lines.append(f"  - `{var_name}`: `{format_shape(shape)}` {interpretation}")
            else:
                lines.append(f"- 静态变量形状: `{static_shape or 'N/A'}`")
            lines.append("")

        if validate.get('warnings'):
            lines.append("**警告**:")
            lines.append("")
            for warning in validate['warnings']:
                lines.append(f"- ⚠️ {warning}")
            lines.append("")

        if validate.get('errors'):
            lines.append("**错误**:")
            lines.append("")
            for error in validate['errors']:
                lines.append(f"- ❌ {error}")
            lines.append("")
    else:
        lines.append("⚠️ 未找到验证结果 (validate_result.json)")
        lines.append("")

    # ========================================
    # 4. 转换结果
    # ========================================
    lines.append("## 4. 转换结果")
    lines.append("")

    if convert:
        lines.append("### 4.1 数据集划分")
        lines.append("")

        # 优先从 convert 的 config 字段读取，其次从 convert 根级别，最后从 manifest
        config_info = convert.get('config', {}) if convert else {}

        if config_info:
            train_ratio = config_info.get('train_ratio', 'N/A')
            valid_ratio = config_info.get('valid_ratio', 'N/A')
            test_ratio = config_info.get('test_ratio', 'N/A')
        elif convert:
            train_ratio = convert.get('train_ratio', 'N/A')
            valid_ratio = convert.get('valid_ratio', 'N/A')
            test_ratio = convert.get('test_ratio', 'N/A')
        else:
            train_ratio = valid_ratio = test_ratio = 'N/A'

        lines.append(f"- **训练集比例**: {train_ratio}")
        lines.append(f"- **验证集比例**: {valid_ratio}")
        lines.append(f"- **测试集比例**: {test_ratio}")
        lines.append("")

        # 检查裁剪信息
        h_slice = config_info.get('h_slice') if config_info else (convert.get('h_slice') if convert else None)
        w_slice = config_info.get('w_slice') if config_info else (convert.get('w_slice') if convert else None)
        scale = config_info.get('scale') if config_info else (convert.get('scale') if convert else None)

        if h_slice or w_slice:
            lines.append("### 4.2 裁剪信息")
            lines.append("")
            lines.append(f"- **H 方向裁剪**: `{h_slice or '无'}`")
            lines.append(f"- **W 方向裁剪**: `{w_slice or '无'}`")
            lines.append(f"- **下采样倍数**: {scale or 'N/A'}")
            lines.append("")

        # 后置验证（只在 convert_result.json 中）
        if convert:
            lines.append("### 4.3 后置验证")
            lines.append("")

            # 读取 validation_rule1, validation_rule2, validation_rule3 字段
            rule1 = convert.get('validation_rule1', {})
            rule2 = convert.get('validation_rule2', {})
            rule3 = convert.get('validation_rule3', {})

            if rule1 or rule2 or rule3:
                # Rule 1
                if rule1:
                    status = "✅ 通过" if rule1.get('passed') else "❌ 失败"
                    lines.append(f"- **Rule 1 (输出完整性)**: {status}")
                    if rule1.get('errors'):
                        for error in rule1['errors']:
                            lines.append(f"  - ❌ {error}")
                    if rule1.get('warnings'):
                        for warning in rule1['warnings']:
                            lines.append(f"  - ⚠️ {warning}")

                # Rule 2
                if rule2:
                    status = "✅ 通过" if rule2.get('passed') else "❌ 失败"
                    lines.append(f"- **Rule 2 (掩码不可变性)**: {status}")
                    if rule2.get('errors'):
                        for error in rule2['errors']:
                            lines.append(f"  - ❌ {error}")
                    if rule2.get('warnings'):
                        for warning in rule2['warnings']:
                            lines.append(f"  - ⚠️ {warning}")

                # Rule 3
                if rule3:
                    status = "✅ 通过" if rule3.get('passed') else "❌ 失败"
                    lines.append(f"- **Rule 3 (排序确定性)**: {status}")
                    if rule3.get('errors'):
                        for error in rule3['errors']:
                            lines.append(f"  - ❌ {error}")
                    if rule3.get('warnings'):
                        for warning in rule3['warnings']:
                            lines.append(f"  - ⚠️ {warning}")

                lines.append("")

                # 添加后置验证汇总
                post_validation = convert.get('post_validation', {})
                if post_validation:
                    all_passed = post_validation.get('all_passed', False)
                    total_errors = post_validation.get('total_errors', 0)
                    total_warnings = post_validation.get('total_warnings', 0)
                    lines.append(f"**验证汇总**: {'✅ 全部通过' if all_passed else f'❌ {total_errors} 个错误, ⚠️ {total_warnings} 个警告'}")
                    lines.append("")
            else:
                # 尝试从旧格式读取
                validation = convert.get('validation', {})
                if validation:
                    lines.append(f"- **Rule 1 (输出完整性)**: `{validation.get('rule1_status', 'N/A')}`")
                    lines.append(f"- **Rule 2 (掩码不可变性)**: `{validation.get('rule2_status', 'N/A')}`")
                    lines.append(f"- **Rule 3 (排序确定性)**: `{validation.get('rule3_status', 'N/A')}`")
                    lines.append("")
                else:
                    lines.append("*后置验证信息未包含在转换结果中*")
                    lines.append("")

            subsection_idx = 4
            # 显示 warnings 和 errors（从 convert_result.json 根级别）
            if convert.get('warnings'):
                lines.append(f"### 4.{subsection_idx} 转换警告")
                lines.append("")
                for warning in convert['warnings']:
                    lines.append(f"- ⚠️ {warning}")
                lines.append("")
                subsection_idx += 1

            if convert.get('errors'):
                lines.append(f"### 4.{subsection_idx} 转换错误")
                lines.append("")
                for error in convert['errors']:
                    lines.append(f"- ❌ {error}")
                lines.append("")
                subsection_idx += 1
                
        # 输出文件统计
        lines.append(f"### 4.{subsection_idx} 输出文件结构")
        lines.append("")
        lines.append("**v3.0.0 新格式**：每个时间步单独保存为一个文件")
        lines.append("")
        lines.append("```")
        lines.append("output_base/")
        lines.append("├── train/")
        lines.append("│   ├── hr/")
        lines.append("│   │   ├── var1/          # 变量目录")
        lines.append("│   │   │   ├── 000000.npy # 时间步 0, 形状 [H, W]")
        lines.append("│   │   │   ├── 000001.npy # 时间步 1")
        lines.append("│   │   │   └── ...")
        lines.append("│   │   └── var2/")
        lines.append("│   └── lr/")
        lines.append("│       └── (同上)")
        lines.append("├── valid/")
        lines.append("├── test/")
        lines.append("└── static_variables/")
        lines.append("```")
        lines.append("")

        if 'saved_files' in convert:
            saved_files = convert['saved_files']
            lines.append(f"- **总文件数**: {len(saved_files)}")
            lines.append("")
    else:
        lines.append("⚠️ 未找到转换结果 (convert_result.json 或 preprocess_manifest.json)")
        lines.append("")

    # ========================================
    # 5. 质量指标
    # ========================================
    lines.append("## 5. 质量指标")
    lines.append("")

    if metrics and 'splits' in metrics:
        lines.append("### 5.1 指标概览")
        lines.append("")
        lines.append(f"- **下采样倍数**: {metrics.get('config', {}).get('scale', 'N/A')}")
        lines.append("")

        for split_name, split_data in sorted(metrics['splits'].items()):
            if not split_data:
                continue

            lines.append(f"### 5.2 {split_name.capitalize()} 数据集")
            lines.append("")
            lines.append("| 变量名 | SSIM ↑ | Relative L2 ↓ | MSE ↓ | RMSE ↓ |")
            lines.append("|--------|--------|----------------|-------|--------|")

            for var_name, var_metrics in sorted(split_data.items()):
                ssim_val = var_metrics.get('ssim', 0.0)
                l2_val = var_metrics.get('relative_l2', 0.0)
                mse_val = var_metrics.get('mse', 0.0)
                rmse_val = var_metrics.get('rmse', 0.0)

                lines.append(f"| `{var_name}` | {ssim_val:.4f} | {l2_val:.4f} | {mse_val:.6f} | {rmse_val:.6f} |")

            lines.append("")

        lines.append("**指标说明**:")
        lines.append("")
        lines.append("- **SSIM** (结构相似性): 0~1，越接近 1 表示结构越相似")
        lines.append("- **Relative L2**: 相对 L2 误差，越小越好（HR 作为基准）")
        lines.append("- **MSE**: 均方误差，越小越好")
        lines.append("- **RMSE**: 均方根误差，越小越好")
        lines.append("")
    else:
        lines.append("⚠️ 未找到质量指标结果 (metrics_result.json)")
        lines.append("")

    # ========================================
    # 6. 可视化对比
    # ========================================
    lines.append("## 6. 可视化对比")
    lines.append("")

    vis_dir = os.path.join(dataset_root, 'visualisation_data_process')
    if os.path.exists(vis_dir):
        # 6.0 全局统计汇总图
        summary_path = os.path.join(vis_dir, 'statistics_summary.png')
        if os.path.exists(summary_path):
            rel_path = os.path.relpath(summary_path, dataset_root)
            lines.append("### 6.0 全局统计汇总")
            lines.append("")
            lines.append("所有变量的均值和标准差对比（HR vs LR）：")
            lines.append("")
            lines.append(f"![全局统计汇总]({rel_path})")
            lines.append("")

        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(vis_dir, split)
            if not os.path.exists(split_dir):
                continue

            # 分类收集图片
            compare_files = sorted(glob.glob(os.path.join(split_dir, '*_compare.png')))
            stats_files = sorted(glob.glob(os.path.join(split_dir, '*_statistics.png')))

            # 兼容旧格式（没有 _compare 后缀的图片）
            if not compare_files and not stats_files:
                old_files = sorted(glob.glob(os.path.join(split_dir, '*.png')))
                compare_files = old_files

            if not compare_files and not stats_files:
                continue

            split_idx = ['train', 'valid', 'test'].index(split) + 1
            lines.append(f"### 6.{split_idx} {split.capitalize()} 数据集")
            lines.append("")

            # 获取变量列表
            var_names = set()
            for f in compare_files:
                name = os.path.basename(f).replace('_compare.png', '').replace('.png', '')
                var_names.add(name)
            for f in stats_files:
                name = os.path.basename(f).replace('_statistics.png', '')
                var_names.add(name)

            # 每个变量一个小节
            for var_idx, var_name in enumerate(sorted(var_names)):
                if var_idx >= 3:  # 每个 split 最多显示 3 个变量
                    lines.append(f"*（共 {len(var_names)} 个变量，仅显示前 3 个）*")
                    lines.append("")
                    break

                lines.append(f"#### {var_name}")
                lines.append("")

                # 空间对比图
                compare_path = os.path.join(split_dir, f'{var_name}_compare.png')
                if not os.path.exists(compare_path):
                    compare_path = os.path.join(split_dir, f'{var_name}.png')  # 兼容旧格式

                if os.path.exists(compare_path):
                    rel_path = os.path.relpath(compare_path, dataset_root)
                    lines.append("**HR vs LR 空间对比**:")
                    lines.append("")
                    lines.append(f"![{var_name} 空间对比]({rel_path})")
                    lines.append("")

                # 统计分布图
                stats_path = os.path.join(split_dir, f'{var_name}_statistics.png')
                if os.path.exists(stats_path):
                    rel_path = os.path.relpath(stats_path, dataset_root)
                    lines.append("**统计分布（均值/方差时序 + 直方图）**:")
                    lines.append("")
                    lines.append(f"![{var_name} 统计分布]({rel_path})")
                    lines.append("")
    else:
        lines.append("⚠️ 未找到可视化结果目录 (visualisation_data_process/)")
        lines.append("")

    # ========================================
    # 7. 分析和建议
    # ========================================
    lines.append("## 7. 分析和建议")
    lines.append("")

    analysis_placeholder = analyze_data_quality()
    lines.append(analysis_placeholder)
    lines.append("")

    # ========================================
    # 8. 总结
    # ========================================
    lines.append("## 8. 总结")
    lines.append("")

    # 统计信息
    total_files = inspect.get('file_count', 0) if inspect else 0
    total_vars = len(inspect.get('dynamic_vars_candidates', [])) if inspect else 0
    validation_status = validate.get('status', 'unknown') if validate else 'unknown'

    lines.append(f"本次预处理共处理 **{total_files}** 个文件，包含 **{total_vars}** 个动态变量。")
    lines.append(f"验证状态: `{validation_status}`")
    lines.append("")

    if metrics:
        lines.append("数据质量指标已计算完成，详见第 5 节。")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*报告由 Ocean-Agent-SDK 自动生成*")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="生成海洋数据预处理 Markdown 报告"
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"[Success] 报告已生成: {output_path}")

    except Exception as e:
        print(f"[Error] 报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
