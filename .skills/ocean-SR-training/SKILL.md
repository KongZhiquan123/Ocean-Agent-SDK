---
name: ocean-SR-training
description: 海洋超分辨率模型训练技能 - 支持多种模型架构的训练、测试、推理（含陆地掩码处理 + OOM 自动防护 + 错误实时反馈 + predict 全图预测）
version: 4.3.0
author: Leizheng
contributors: kongzhiquan
last_modified: 2026-02-11
---

<!--
Changelog:
  - 2026-02-11 Leizheng: v4.3.0 新增 predict 模式全图预测工作流
    - predict 快速通道：跳过训练工作流直接启动推理
    - predict 结构化事件：predict_start / predict_progress / predict_end
    - predict 可视化：generate_predict_plots.py 生成 SR 对比图
  - 2026-02-09 Leizheng: v4.2.0 默认参数与模型接入策略对齐
    - batch_size / eval_batch_size 默认值调整为 4
    - gradient_checkpointing 默认开启（允许用户手动关闭）
    - use_amp 默认策略改为：非 FFT 模型开启，FFT/频域模型关闭（允许 override 并强提示）
    - 模型列表以 list_models.py 为准；idm/wdno/remg 不再作为可训练模型
   - 2026-02-09 kongzhiquan: v4.1.1 可视化与报告生成步骤解耦
    - 将原步骤 8 拆分为独立的"生成可视化"和"生成报告"两个步骤
    - 新增可视化结果校验门控，禁止未成功时调用报告生成
  - 2026-02-09 Leizheng: v4.1.0 默认参数更新
    - batch_size / eval_batch_size 默认下调为 16
    - gradient_checkpointing 改为按模型/全图训练自动开启
  - 2026-02-07 kongzhiquan: v4.0.0 OOM 自动防护 + 训练错误实时反馈
    - OOM 防护：AMP 默认开启 + 自动循环调参（batch_size 自动减半）
    - 事件驱动启动监控：等待 training_start 事件，捕获早期崩溃
    - 失败分类：自动分析错误类型并提供建议
    - wait 模式：长轮询等待训练状态变化
    - 工作流更新：启动后主动 wait + 失败分析重试
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
3. **错误不自动重试**：展示错误分析后询问用户是否调整参数重试
4. **训练完成后主动询问**：检测到训练完成时，主动询问是否生成可视化和报告
5. **主动状态感知**：训练启动后立即等待，捕获快速完成或早期崩溃

---

## 可用工具

| 工具 | 用途 |
|------|------|
| `ocean_sr_check_gpu` | 查看可用 GPU |
| `ocean_sr_list_models` | 列出可用模型 |
| `ocean_sr_train` | 启动训练或推理（含事件驱动启动监控，predict 模式跳过训练工作流） |
| `ocean_sr_train_status` | 查询训练/推理状态/日志/终止训练/等待状态变化 |
| `ocean_sr_visualize` | 生成训练可视化图表（mode=train）或推理对比图（mode=predict） |
| `ocean_sr_generate_report` | 生成训练报告 |

---

## 工作流程

```
1. 确认数据 → 用户提供预处理数据目录和输出目录
   ↓
2. 选择模型 → ocean_sr_list_models，用户选择
   ↓
3. 确认参数 → epochs, lr, batch_size(默认4), GPU 选择
   │  → OOM 防护参数: use_amp（非 FFT 默认开启 / FFT 默认关闭）, gradient_checkpointing（默认开启）, patch_size
   ↓
4. 参数汇总 → 展示所有参数，等待"确认执行"
   ↓
5. 启动训练 → ocean_sr_train（含事件驱动启动监控）
   │  工具内部等待 training_start 事件（最长 5 分钟）
   │  若返回 status="error"：展示错误 + 建议，询问用户是否调整参数重试
   ↓
6. 首次等待 → ocean_sr_train_status({ action: "wait", process_id, timeout: 120 })
   │  等 2 分钟：
   │  若 process_status="completed"：主动询问是否生成可视化和报告
   │  若 process_status="failed"：展示 error_summary + suggestions
   │    → 询问用户："训练失败原因是 XXX，建议 YYY，是否调整参数重试？"
   │  若 process_status="running"（超时）：
   │    → 告知用户：训练仍在运行中（已完成 N 个 epoch），稍后再询问进度
   ↓
7. 后续询问 → 用户询问"训练怎么样了？"时
   │  调用 ocean_sr_train_status({ action: "wait", process_id, timeout: 120 })
   │  同样等 2 分钟，按上述逻辑处理
   ↓
8. 生成可视化 → ocean_sr_visualize（用户确认后）
   │  检查返回 status="success" 且 plots/ 目录下文件已生成
   │  若失败：展示错误，询问用户是否重试
   │  **禁止在此步骤未成功前调用 ocean_sr_generate_report**
   ↓
9. 生成报告 → ocean_sr_generate_report（仅在步骤 8 成功后执行）
   │  → Agent 读取报告，补充 <!-- AI_FILL: ... --> 占位符
   ↓
10. 完成 → 展示报告路径和关键结果
```

---

## 预测工作流程（Predict Mode）

predict 模式对测试集执行全图 SR 推理，输出物理值空间的 NPY 文件。
跳过训练工作流的 4 阶段确认（OOM/shape/FFT 检查），直接启动。

### 触发条件
- 用户要求"对测试集做推理/预测/predict"
- 训练完成后需要生成完整 SR 输出

### 工作流

```
1. 确认参数 → dataset_root, log_dir, model_name, ckpt_path（可选，默认 best_model.pth）
   ↓
2. 启动推理 → ocean_sr_train({ mode: "predict", dataset_root, log_dir, model_name, ... })
   │  工具内部等待 predict_start 事件（最长 5 分钟）
   │  若返回 status="error"：展示错误 + 建议
   ↓
3. 等待完成 → ocean_sr_train_status({ action: "wait", process_id, timeout: 300 })
   │  若 process_status="completed"：主动询问是否生成可视化
   │  若 process_status="failed"：展示错误 + 建议
   ↓
4. 可视化 → ocean_sr_visualize({ log_dir, mode: "predict", dataset_root, dyn_vars })
   ↓
5. 完成 → 展示 predictions/ 路径和可视化图表
```

### predict 参数

| 参数 | 必需 | 说明 |
|------|------|------|
| `mode` | 是 | 固定为 `"predict"` |
| `dataset_root` | 是 | 预处理数据目录 |
| `log_dir` | 是 | 训练输出目录（需含 best_model.pth 或指定 ckpt_path） |
| `model_name` | 是 | 模型名称 |
| `ckpt_path` | 否 | 模型权重路径（默认 log_dir/best_model.pth） |
| `dyn_vars` | 条件必填 | 动态变量列表（推理启动不需要，但 predict 可视化必填） |
| `scale` | 否 | 超分辨率倍数 |
| `patch_size` | 否 | HR Patch 尺寸（若训练时用了 patch） |

### 输出目录

```
log_dir/
├── predictions/
│   ├── {filename}_sr.npy    ← 全图 SR 输出 [H, W, C]
│   └── ...
├── test_samples.npz          ← LR/SR/HR 对比数据（前 4 条）
└── plots/
    ├── predict_comparison_00.png ← 可视化对比图
    └── ...
```

### 注意事项
- predict 不需要训练参数确认（不走 4 阶段状态机）
- predict 始终使用单卡推理
- 如果 log_dir 中没有 best_model.pth，必须显式传入 ckpt_path

---

## 主动状态检查

**重要**：如果之前启动过训练进程，Agent 在每次收到用户新消息时，
应先调用 ocean_sr_train_status({ action: "list" }) 检查训练状态。
如果发现训练已完成或失败，优先告知用户训练结果，再处理用户当前请求。

---

## OOM 自动防护机制

训练前自动进行 GPU 显存预估并自动调参，防止训练过程中 OOM 崩溃。

### 自动防护流程
1. use_amp 按模型默认：非 FFT 默认开启；FFT/频域模型默认关闭（可手动 override）
2. 显存预估 > 85% 时自动降级：
   - 第一步：开启 AMP（如果未开启）
   - 第二步：batch_size 减半（直到 1）
   - 最多 5 次尝试
3. 所有手段用尽仍不够 → 报错并建议使用更大显存 GPU 或设置 patch_size

### OOM 时的手动建议优先级
1. 启用 `use_amp=true`（最易操作，效果显著）
2. 减小 `batch_size`（如 4 → 2 或 1）
3. 启用 `gradient_checkpointing=true`（有计算代价；默认已开启，可显式确认）
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
├── test_samples.npz           ← 测试/推理样本数据
├── training_report.md         ← 训练报告
├── predictions/               ← predict 模式输出
│   ├── {filename}_sr.npy
│   └── ...
└── plots/                     ← 可视化图表
    ├── loss_curve.png
    ├── metrics_curve.png
    ├── lr_curve.png
    ├── metrics_comparison.png
    ├── training_summary.png
    ├── sample_comparison.png
    └── predict_comparison_XX.png ← predict 模式
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
