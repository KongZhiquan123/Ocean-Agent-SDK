---
name: ocean-SR-training
description: 海洋超分辨率模型训练技能 - 支持多种模型架构的训练、测试与推理（含陆地掩码处理）
version: 3.0.0
author: Leizheng
last_modified: 2026-02-07
---

<!--
Changelog:
  - 2026-02-07 kongzhiquan: v3.0.0 后台训练模式
    - 训练启动后立即返回，不阻塞等待
    - 新增 ocean_sr_train_status 工具查询训练状态和日志
    - 服务器关闭时自动清理训练进程
    - 工作流更新：启动训练后等待用户指令
  - 2026-02-06 Leizheng: v2.0.0 陆地掩码 + 训练报告
    - 训练框架升级为 masked 版本（ocean_SR_training_masked）
    - 损失函数/评估指标排除陆地像素
    - 新增训练报告生成工具 ocean_sr_generate_report
    - 工作流新增报告生成步骤
  - 2026-02-06 Leizheng: v1.0.0 初始版本
    - 支持 15 种超分辨率模型（10 种标准 + 5 种扩散模型）
    - 适配 ocean-preprocess NPY 输出目录结构
    - 多卡训练支持（DP/DDP）
    - GPU 查询功能
-->

# 海洋超分辨率模型训练技能

## 核心原则

1. **禁止自动决策**：模型选择、训练参数、GPU 选择必须由用户确认
2. **错误附带建议**：遇到错误时，展示错误信息 + 可能的原因 + 修改建议
3. **错误不自动重试**：等待用户指示
4. **输出持久化**：训练日志、模型权重、指标结果保存到文件

---

## 可用工具

| 工具 | 用途 | 使用时机 |
|------|------|----------|
| `ocean_sr_check_gpu` | 查看可用 GPU | 确认参数阶段，展示 GPU 供用户选择 |
| `ocean_sr_list_models` | 列出可用模型 | 用户选择模型时 |
| `ocean_sr_train` | 启动训练/测试 | 参数确认后启动（后台执行） |
| `ocean_sr_train_status` | 查询训练状态/日志 | 训练启动后，查看进度或终止训练 |
| `ocean_sr_generate_report` | 生成训练报告 | 训练完成后，生成 Markdown 报告 |

---

## 工作流程

```
1. 确认数据 → 用户提供预处理数据目录和输出目录
   │  → 检查目录结构是否符合预期（train/hr, train/lr 等）
   ↓
2. 选择模型 → ocean_sr_list_models
   │  → 展示所有模型（标准/扩散），用户选择
   ↓
3. 确认参数 → 询问用户（GPU 选择在此阶段）
   │  → epochs, lr, batch_size
   │  → ocean_sr_check_gpu 查看可用 GPU
   │  → 用户选择用哪些卡、几张卡
   ↓
4. 参数汇总 → 展示所有参数，等待"确认执行"
   │  → 生成 YAML 配置文件
   ↓
5. 启动训练 → ocean_sr_train (mode=train)
   │  → 【后台执行】立即返回 process_id，不阻塞等待
   │  → 单卡: main.py / 多卡 DDP: torchrun main_ddp.py
   │  → 陆地掩码自动处理（NaN → 0，mask 排除陆地格点）
   ↓
6. 等待用户指令 → 【重要】训练启动后，告知用户训练已开始
   │  → 告知用户可以：
   │     - 查看训练状态：ocean_sr_train_status({ process_id: "xxx" })
   │     - 查看最新日志：ocean_sr_train_status({ action: "logs", process_id: "xxx", tail: 50 })
   │     - 终止训练：ocean_sr_train_status({ action: "kill", process_id: "xxx" })
   │  → 【不要主动轮询】等待用户主动询问进度或给出下一步指令
   ↓
7. 训练完成后 → 用户询问时检查状态
   │  → 如果 status="completed"，展示结果并询问是否生成报告
   │  → 如果 status="failed"，展示错误日志和修改建议
   ↓
8. 生成训练报告 → ocean_sr_generate_report
   │  → 传入 log_dir 和 user_confirmation
   │  → Agent 读取报告，补充"分析与建议"部分
   ↓
9. 完成 → 向用户展示报告路径和关键结果
```

---

## 后台训练模式（v3.0.0 新增）

### 工作原理

训练任务在后台执行，`ocean_sr_train` 启动后立即返回，不会阻塞对话：

```
ocean_sr_train(...)
  → 返回 { status: "started", process_id: "train-xxx-xxx", ... }
  → 训练在后台持续运行
  → Agent 可以继续与用户对话
```

### 状态查询工具

使用 `ocean_sr_train_status` 查询训练状态：

| 操作 | 调用方式 | 说明 |
|------|----------|------|
| 查询状态 | `ocean_sr_train_status({ process_id: "xxx" })` | 返回运行状态、耗时等 |
| 查看日志 | `ocean_sr_train_status({ action: "logs", process_id: "xxx", tail: 50 })` | 返回最后 50 行日志 |
| 增量日志 | `ocean_sr_train_status({ action: "logs", process_id: "xxx", offset: 12345 })` | 从上次位置继续读取 |
| 终止训练 | `ocean_sr_train_status({ action: "kill", process_id: "xxx" })` | 发送终止信号 |
| 列出所有 | `ocean_sr_train_status({ action: "list" })` | 列出所有训练进程 |

### 训练状态

| 状态 | 含义 |
|------|------|
| `running` | 训练进行中 |
| `completed` | 训练成功完成（exit code = 0） |
| `failed` | 训练失败（exit code != 0） |
| `killed` | 被用户或系统终止 |

### 重要行为准则

1. **启动后不要主动轮询**：训练启动后，告知用户训练已开始，然后等待用户的下一步指令
2. **用户询问时才查询**：只有当用户主动询问训练进度时，才调用 `ocean_sr_train_status`
3. **保持对话可用**：训练在后台运行，Agent 可以继续回答用户的其他问题
4. **服务器关闭自动清理**：如果服务器关闭，训练进程会被自动终止

---

## 陆地掩码处理（v2.0.0 新增）

训练框架自动检测并排除海洋数据中的陆地像素（NaN）：

### 掩码生成
- 从训练集 HR 数据第一个时间步生成 mask
- `True` = 海洋（有效像素），`False` = 陆地（NaN 位置）
- NaN 填充为 0（在归一化之前）

### 训练时掩码使用
- **标准模型**（SwinIR, FNO 等）：使用 `MaskedLpLoss`，只在海洋格点计算 loss
- **扩散模型**（DDPM, ReMiG, WDNO）：loss 归一化分母使用有效像素数而非总像素数
- **ResShift**：训练 loss 由内部 diffusion 框架计算，评估阶段使用 masked metrics

### 评估时掩码使用
- 所有模型评估统一使用 `MaskedEvaluator`
- MSE/RMSE/PSNR/SSIM 只在海洋格点上计算

---

## 模型概览

### 标准模型（BaseTrainer）

| 模型名 | 说明 |
|--------|------|
| FNO2d | Fourier Neural Operator 2D |
| UNet2d | UNet 2D |
| M2NO2d | Multiplicative Multiresolution Neural Operator |
| SwinIR | SwinIR 超分辨率（推荐） |
| EDSR | Enhanced Deep Super-Resolution |
| HiNOTE | High-order Neural Operator |
| Galerkin_Transformer | Galerkin Transformer |
| MWT2d | Morlet Wavelet Transform 2D |
| SRNO | Super-Resolution Neural Operator |
| Swin_Transformer | Swin Transformer SR |

### 扩散模型

| 模型名 | Trainer | 说明 |
|--------|---------|------|
| DDPM | DDPMTrainer | Denoising Diffusion Probabilistic Model |
| SR3 | DDPMTrainer | SR3 Diffusion Model |
| MG-DDPM | DDPMTrainer | Multigrid DDPM |
| ReMiG | DDPMTrainer | ReMiG Diffusion Model |
| Resshift | ResshiftTrainer | Residual Shifting Diffusion |

---

## GPU 选择（在参数确认阶段）

调用 `ocean_sr_check_gpu` 展示可用 GPU，让用户选择。

| GPU 数量 | 推荐模式 | 运行方式 |
|----------|----------|----------|
| 1 张 | 单卡 | `python main.py --config config.yaml` |
| 2+ 张 | DDP（推荐） | `torchrun --nproc_per_node=N main_ddp.py --config config.yaml` |
| 2+ 张 | DP（备选） | `python main.py --config config.yaml`（配置中设置 DP） |

---

## 数据目录要求

需要 `ocean-preprocess` 预处理后的标准输出目录：

```
dataset_root/
├── train/
│   ├── hr/{var}/*.npy    (高分辨率)
│   └── lr/{var}/*.npy    (低分辨率)
├── valid/
│   ├── hr/{var}/*.npy
│   └── lr/{var}/*.npy
├── test/
│   ├── hr/{var}/*.npy
│   └── lr/{var}/*.npy
└── static_variables/     (可选)
```

---

## 错误处理原则

遇到错误时，必须同时展示：

1. **错误信息**：原始错误内容
2. **可能原因**：分析错误产生的原因
3. **修改建议**：给出具体的解决方案

示例：
```
错误：FileNotFoundError: HR directory not found: /data/output/train/hr/temp

可能原因：
- 预处理数据目录路径不正确
- 预处理尚未执行或未完成
- 变量名 "temp" 不在预处理输出中

修改建议：
1. 检查数据目录是否正确：ls /data/output/train/hr/
2. 确认预处理已完成且变量名匹配
3. 重新提供正确的数据目录路径
```

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

## 参考文档索引

| 文档 | 内容 | 何时读取 |
|------|------|----------|
| `references/models.md` | 模型详细说明和推荐参数 | 需要模型细节时 |
| `references/parameters.md` | 所有工具参数 | 需要参数细节时 |
| `references/examples.md` | 对话示例 | 需要参考示例时 |
| `references/errors.md` | 错误处理指南 | 遇到错误时 |

---

## 输出目录结构

```
log_dir/
├── train-1707123456-abc123.log    ← 进程管理器日志（实时输出）
├── train-1707123456-abc123.error.log
└── OceanNPY_SwinIR_0206_143025/   ← Python 训练框架创建
    ├── config.yaml          ← 完整配置备份
    ├── train.log            ← 训练日志（与进程日志内容相同）
    ├── best_model.pth       ← 最佳模型权重
    ├── training_report.md   ← 训练报告
    └── code/                ← 代码快照
```
