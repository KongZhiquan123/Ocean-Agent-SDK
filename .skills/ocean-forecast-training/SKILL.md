---
name: ocean-forecast-training
description: 海洋时序预测模型训练技能 - 支持多种模型架构的训练、推理与自回归预测（含 OOM 防护 + per-variable 指标）
version: 1.1.0
author: Leizheng
last_modified: 2026-02-26
---

<!--
Changelog:
  - 2026-02-26 Leizheng: v1.1.0 add visualization.md reference, token v2
  - 2026-02-26 Leizheng: v1.0.0 initial version for ocean forecast training
-->

# 海洋时序预测模型训练技能

## 核心原则

1. **禁止自动决策**：模型选择、训练参数、GPU 选择必须由用户确认
2. **错误附带建议**：遇到错误时，展示错误信息 + 可能的原因 + 修改建议
3. **错误不自动重试**：展示错误分析后询问用户是否调整参数重试
4. **训练完成后主动询问**：检测到训练完成时，主动询问是否生成可视化和报告
5. **主动状态感知**：训练启动后立即等待，捕获快速完成或早期崩溃

---

## 与超分辨率训练的核心区别

| 对比项 | 超分辨率 (SR) | 时序预测 (Forecast) |
|--------|--------------|-------------------|
| 输入输出 | LR → HR（分辨率不同） | in_t 步 → out_t 步（同分辨率） |
| 关键参数 | `scale`, `patch_size` | `in_t`, `out_t`, `stride` |
| 评估指标 | PSNR, SSIM | RMSE, MAE |
| 模型特点 | 需上采样模块 | 无上采样，纯时序映射 |
| 数据格式 | hr/{var}/*.npy + lr/{var}/*.npy | {var}/*.npy（按日期命名） |
| 推理模式 | 单步 SR | 支持自回归 rollout |

---

## 可用工具

| 工具 | 用途 |
|------|------|
| `ocean_forecast_check_gpu` | 查看可用 GPU |
| `ocean_forecast_list_models` | 列出可用预测模型 |
| `ocean_forecast_train` | 启动训练或推理（含事件驱动启动监控） |
| `ocean_forecast_train_status` | 查询训练状态/日志/终止/等待状态变化 |
| `ocean_forecast_train_visualize` | 生成训练可视化图表（mode=train）或预测对比图（mode=predict） |
| `ocean_forecast_train_report` | 生成训练报告 |

---

## 工作流程

```
1. 检查 GPU → ocean_forecast_check_gpu
   ↓
2. 确认数据 → 用户提供 dataset_root（预处理输出目录）和 log_dir
   ↓
3. 验证数据 → ocean_forecast_train（Stage 1: 自动调用 validate_dataset.py）
   │  展示: dyn_vars, spatial_shape, splits, time_range
   ↓
4. 选择模型 → ocean_forecast_list_models，用户选择
   │  推荐: FNO2d > UNet2d > SwinTransformerV2 > Transformer
   ↓
5. 确认参数 → ocean_forecast_train（Stage 3）
   │  时序参数: in_t（默认 7）, out_t（默认 1）, stride（默认 1）
   │  训练参数: epochs, lr, batch_size(默认 4), GPU 选择
   │  OOM 防护: use_amp（默认开启）, gradient_checkpointing（默认开启）
   │  Agent 推荐: in_t 范围 5~14，out_t 范围 1~7，stride 通常 1
   ↓
6. 参数汇总 → 展示所有参数，等待"确认执行"
   ↓
7. 启动训练 → ocean_forecast_train（Stage 4, user_confirmed=true）
   │  工具内部等待 training_start 事件（最长 5 分钟）
   │  若返回 status="error"：展示错误 + 建议
   ↓
8. 首次等待 → ocean_forecast_train_status({ action: "wait", process_id, timeout: 120 })
   │  若 completed：主动询问是否生成可视化和报告
   │  若 failed：展示 error_summary + suggestions
   │  若 running（超时）：告知用户训练仍在运行
   ↓
9. 生成可视化 → ocean_forecast_train_visualize（用户确认后）
   │  **禁止在此步骤未成功前调用 ocean_forecast_train_report**
   ↓
10. 生成报告 → ocean_forecast_train_report
    │  Agent 读取报告，补充 <!-- AI_FILL: ... --> 占位符
    ↓
11. 完成 → 展示报告路径和关键结果
```

---

## 预测工作流程（Predict Mode）

predict 模式对测试集执行全量推理，支持自回归 rollout 多步预测。

### 触发条件
- 用户要求"对测试集做推理/预测/predict"
- 训练完成后需要生成完整预测输出

### 工作流

```
1. 确认参数 → dataset_root, log_dir, model_name, ckpt_path（可选）
   ↓
2. 启动推理 → ocean_forecast_train({ mode: "predict", dataset_root, log_dir, model_name, ... })
   │  工具内部等待 predict_start 事件（最长 5 分钟）
   ↓
3. 等待完成 → ocean_forecast_train_status({ action: "wait", process_id, timeout: 300 })
   │  若 completed：主动询问是否生成可视化
   │  若 failed：展示错误 + 建议
   ↓
4. 可视化 → ocean_forecast_train_visualize({ log_dir, mode: "predict", dataset_root })
   ↓
5. 完成 → 展示 predictions/ 路径和可视化图表
```

### predict 参数

| 参数 | 必需 | 说明 |
|------|------|------|
| `mode` | 是 | 固定为 `"predict"` |
| `dataset_root` | 是 | 预处理数据目录 |
| `log_dir` | 是 | 训练输出目录（需含 best_model.pth） |
| `model_name` | 是 | 模型名称 |
| `ckpt_path` | 否 | 模型权重路径（默认 log_dir/best_model.pth） |

### 输出目录

```
log_dir/
├── predictions/
│   ├── sample_000000_t0_var0_uo.npy
│   ├── sample_000000_t0_var1_vo.npy
│   └── ...
└── plots/
    ├── predict_sample_0_var_uo.png
    ├── predict_overview.png
    └── ...
```

---

## 主动状态检查

**重要**：如果之前启动过训练进程，Agent 在每次收到用户新消息时，
应先调用 ocean_forecast_train_status({ action: "list" }) 检查训练状态。
如果发现训练已完成或失败，优先告知用户训练结果，再处理用户当前请求。

---

## OOM 自动防护机制

训练前自动进行 GPU 显存预估并自动调参，防止训练过程中 OOM 崩溃。

### 自动防护流程
1. use_amp 默认开启（FFT 模型如 FNO2d、M2NO2d 自动关闭）
2. gradient_checkpointing 默认开启
3. 显存预估 > 85% 时自动降级：
   - 第一步：开启 AMP（如果未开启且模型支持）
   - 第二步：batch_size 减半（直到 1）
   - 最多 5 次尝试
4. 所有手段用尽仍不够 → 报错并建议使用更大显存 GPU

### OOM 时的手动建议优先级
1. 启用 `use_amp=true`（最易操作，效果显著）
2. 减小 `batch_size`（如 4 → 2 或 1）
3. 启用 `gradient_checkpointing=true`
4. 减小 `in_t`（减少输入时间步）
5. 使用多卡训练分摊显存

---

## 禁止行为

| 类别 | 禁止行为 |
|------|----------|
| **模型选择** | 自动决定使用哪个模型 |
| **参数决策** | 自动决定 epochs、lr、batch_size、GPU |
| **流程控制** | 跳过参数确认 |
| **错误处理** | 自动重试失败的训练、不给出修改建议 |

---

## 数据目录要求

需要 `ocean-forecast-data-preprocess` 预处理后的标准输出目录：

```
dataset_root/
├── var_names.json          ← 变量名和空间形状
├── time_index.json         ← 时间索引（可选）
├── train/
│   ├── {var_name}/
│   │   ├── 19930101.npy    ← (H, W) float32
│   │   ├── 19930102.npy
│   │   └── ...
│   └── ...
├── valid/
│   └── {var_name}/*.npy
├── test/
│   └── {var_name}/*.npy
└── static/                 (可选)
    └── {var_name}.npy
```

---

## 输出目录结构

```
log_dir/
├── train.log               ← 训练日志（含 __event__ 标记）
├── config.yaml             ← 训练配置备份
├── best_model.pth          ← 最佳模型权重
├── predictions/            ← predict 模式输出
├── training_report.md      ← 训练报告
└── plots/                  ← 可视化图表
    ├── loss_curve.png
    ├── metrics_curve.png
    ├── lr_curve.png
    ├── per_var_metrics.png
    ├── training_summary.png
    └── predict_*.png
```

---

## 参考文档索引

| 文档 | 内容 | 何时读取 |
|------|------|----------|
| `references/models.md` | 模型详细说明和推荐参数 | 需要模型细节时 |
| `references/parameters.md` | 所有工具参数 | 需要参数细节时 |
| `references/examples.md` | 对话示例 | 需要参考示例时 |
| `references/errors.md` | 错误处理指南 | 遇到错误时 |
| `references/visualization.md` | 可视化与报告生成流程 | 训练完成后 |
