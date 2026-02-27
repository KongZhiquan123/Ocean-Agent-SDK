# 工具参数参考

> 版本: 4.2.0 | 最后更新: 2026-02-09

---

## ocean_sr_gpu_check - 查看可用 GPU

无需参数，直接调用。

**返回值**：
- `cuda_available`: CUDA 是否可用
- `gpu_count`: GPU 数量
- `gpus`: 每张 GPU 的详细信息（名称、总显存、空闲显存、已用显存）

---

## ocean_sr_model_list - 列出可用模型

无需参数，直接调用。

**返回值**：
- `models`: 模型列表，每个包含 name, category, trainer, description

---

## ocean_sr_train_start - 执行训练

### 必需参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `dataset_root` | string | 预处理数据根目录 |
| `log_dir` | string | 日志输出目录 |
| `model_name` | string | 模型名称（来自 list_models） |
| `dyn_vars` | string[] | 动态变量列表 |
| `scale` | number | 超分辨率倍数 |

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | string | "train" | 模式: "train" 或 "test" |
| `epochs` | number | 500 | 训练轮数 |
| `lr` | number | 0.001 | 学习率 |
| `batch_size` | number | 4 | 训练 batch size |
| `eval_batch_size` | number | 4 | 评估 batch size |
| `patience` | number | 10 | 早停耐心值 |
| `eval_freq` | number | 5 | 每 N 个 epoch 评估一次 |
| `seed` | number | 42 | 随机种子 |

### GPU 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `device_ids` | number[] | [0] | 使用的 GPU 列表 |
| `distribute` | boolean | false | 是否启用多卡训练 |
| `distribute_mode` | string | "DDP" | 多卡模式: "DP" 或 "DDP" |

### 优化器参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `optimizer` | string | "AdamW" | 优化器: AdamW / Adam / SGD |
| `weight_decay` | number | 0.001 | 权重衰减 |
| `scheduler` | string | "StepLR" | 学习率调度器 |
| `scheduler_step_size` | number | 300 | 调度器步长 |
| `scheduler_gamma` | number | 0.5 | 调度器衰减率 |

### 其他参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `normalize` | boolean | true | 是否归一化 |
| `normalizer_type` | string | "PGN" | 归一化类型: PGN / GN |
| `wandb` | boolean | false | 是否启用 WandB 日志 |
| `ckpt_path` | string | - | 恢复训练的检查点路径 |

### OOM 防护参数（v4.0.0 更新）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_amp` | boolean | auto | 启用 AMP 混合精度训练（减少约 40-50% 显存；非 FFT 默认 true / FFT 默认 false，允许 override 并强提示） |
| `gradient_checkpointing` | boolean | true | 启用梯度检查点（减少约 60% 激活显存，增加约 30% 计算；默认开启，可手动关闭） |
| `patch_size` | number | null | Patch 裁剪尺寸，null 为全图训练（需为 scale 整数倍） |

> 注意：显存预估为强制步骤，不可跳过。预估 > 85% 时系统会先尝试开启 AMP（若当前关闭），再自动降低 batch_size。
> eval_batch_size 默认 4（扩散模型超过 4 会自动限制为 4；非扩散模型可按需调整）。

### 模型尺寸自动适配（v3.0.0 新增）

训练框架会自动处理数据尺寸与模型架构的兼容性问题，**用户无需手动配置**：

| 模型 | 整除要求 | 自动处理方式 |
|------|---------|-------------|
| DDPM / SR3 / ReMiG | 32 (2^5) | `image_size` 自动向上对齐（如 400→416），推理后 crop 回原尺寸 |
| Resshift | 32 (2^5) | 推理时 interpolate 到对齐尺寸，采样后 crop 回原尺寸 |
| UNet2d | 16 (2^4) | 推理时 reflect pad 到对齐尺寸，输出后 crop 回原尺寸 |
| SwinIR | 8 (window_size) | 模型内部自带 padding，无需处理 |
| FNO2d / EDSR / HiNOTE / M2NO2d | 1 | 无约束，无需处理 |

**自动 patch_size**：当数据尺寸不能被模型整除且用户未指定 `patch_size` 时，训练框架会自动计算一个合适的 `patch_size` 用于训练。FNO2d/HiNOTE/MWT2d/M2NO2d 默认不切 patch（全图训练）。
