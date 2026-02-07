---
name: ocean-SR-training
description: 海洋超分辨率模型训练技能 - 支持多种模型架构的训练、测试与推理（含陆地掩码处理 + OOM 防护）
version: 3.2.0
author: Leizheng
contributors: kongzhiquan
last_modified: 2026-02-07
---

<!--
Changelog:
  - 2026-02-07 kongzhiquan: v3.2.0 简化输出目录结构，移除子目录层级和代码快照
  - 2026-02-07 kongzhiquan: v3.1.0 可视化与报告增强
  - 2026-02-07 kongzhiquan: v3.0.0 后台训练模式
    - 训练启动后立即返回，不阻塞等待
    - 新增 ocean_sr_train_status 工具查询训练状态和日志
    - 服务器关闭时自动清理训练进程
    - 工作流更新：启动训练后等待用户指令
  - 2026-02-07 Leizheng: v3.0.0 OOM 防护 + 显存预估
    - 新增训练前 GPU 显存预估步骤（estimate_memory.py）
    - 支持 AMP 混合精度训练（use_amp）
    - 支持梯度检查点（gradient_checkpointing）
    - 支持 Patch 裁剪训练（patch_size）
    - 工作流新增"显存预估"阶段（步骤 5）
  - 2026-02-06 Leizheng: v2.0.0 陆地掩码 + 训练报告
  - 2026-02-06 Leizheng: v1.0.0 初始版本
-->

# 海洋超分辨率模型训练技能

## 核心原则

1. **禁止自动决策**：模型选择、训练参数、GPU 选择必须由用户确认
2. **错误附带建议**：遇到错误时，展示错误信息 + 可能的原因 + 修改建议
3. **错误不自动重试**：等待用户指示
4. **训练完成后主动询问**：检测到训练完成时，主动询问是否生成可视化和报告

---

## 可用工具

| 工具 | 用途 |
|------|------|
| `ocean_sr_check_gpu` | 查看可用 GPU |
| `ocean_sr_list_models` | 列出可用模型 |
| `ocean_sr_train` | 启动训练（后台执行） |
| `ocean_sr_train_status` | 查询训练状态/日志/终止训练 |
| `ocean_sr_visualize` | 生成训练可视化图表 |
| `ocean_sr_generate_report` | 生成训练报告 |

---

## 工作流程

```
1. 确认数据 → 用户提供预处理数据目录和输出目录
   ↓
2. 选择模型 → ocean_sr_list_models，用户选择
   ↓
3. 确认参数 → epochs, lr, batch_size, GPU 选择
   │  → OOM 防护参数: use_amp, gradient_checkpointing, patch_size
   ↓
4. 参数汇总 → 展示所有参数，等待"确认执行"
   ↓
5. 启动训练 → ocean_sr_train（后台执行，立即返回）
   ↓
6. 等待用户指令 → 【不要主动轮询】等待用户询问进度
   ↓
7. 查询状态 → 用户询问时调用 ocean_sr_train_status 工具
   │  若返回 process_status="completed"：
   │  【重要】主动询问："训练已完成，是否需要生成可视化图表和训练报告？"
   ↓
8. 生成可视化 → ocean_sr_visualize（用户确认后）
   ↓
9. 生成报告 → ocean_sr_generate_report
   │  → Agent 读取报告，补充 <!-- AI_FILL: ... --> 占位符
   ↓
10. 完成 → 展示报告路径和关键结果
```

---

## OOM 防护机制

训练前自动进行 GPU 显存预估，防止训练过程中 OOM 崩溃。

### OOM 时的建议优先级
1. 启用 `use_amp=true`（最易操作，效果显著）
2. 减小 `batch_size`（如 32 → 16）
3. 启用 `gradient_checkpointing=true`（有计算代价）
4. 设置 `patch_size`（如 64 或 128）
5. 使用多卡训练分摊显存

---

## 禁止行为

| 类别 | 禁止行为 |
|------|----------|
| **模型选择** | 自动决定使用哪个模型 |
| **参数决策** | 自动决定 epochs、lr、batch_size、GPU |
| **流程控制** | 跳过参数确认 |
| **错误处理** | 自动重试失败的训练、不给出修改建议 |
| **训练监控** | 训练启动后主动轮询状态（应等待用户询问） |

---

## 数据目录要求

需要 `ocean-preprocess` 预处理后的标准输出目录：

```
dataset_root/
├── train/
│   ├── hr/{var}/*.npy
│   └── lr/{var}/*.npy
├── valid/
│   ├── hr/{var}/*.npy
│   └── lr/{var}/*.npy
├── test/
│   ├── hr/{var}/*.npy
│   └── lr/{var}/*.npy
└── static_variables/     (可选)
```

---

## 输出目录结构

训练输出直接保存到 `log_dir` 指定的目录：

```
log_dir/                       ← 训练输出目录（由配置指定）
├── train-xxx.log              ← 进程日志
├── train-xxx.error.log        ← 错误日志
├── config.yaml                ← 训练配置
├── train.log                  ← 训练日志
├── best_model.pth             ← 最佳模型权重
├── training_report.md         ← 训练报告
└── plots/                     ← 可视化图表
    ├── loss_curve.png
    ├── metrics_curve.png
    ├── lr_curve.png
    ├── metrics_comparison.png
    └── training_summary.png
```

---

## 参考文档索引

| 文档 | 内容 | 何时读取 |
|------|------|----------|
| `references/models.md` | 模型详细说明和推荐参数 | 需要模型细节时 |
| `references/parameters.md` | 所有工具参数 | 需要参数细节时 |
| `references/background-training.md` | 后台训练模式详解 | 训练启动/状态查询时 |
| `references/visualization.md` | 可视化与报告生成 | 训练完成后生成报告时 |
| `references/examples.md` | 对话示例 | 需要参考示例时 |
| `references/errors.md` | 错误处理指南 | 遇到错误时 |
