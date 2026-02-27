# 训练参数参考

## ocean_forecast_train 参数

### 阶段 1: 数据确认

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `dataset_root` | string | 是 | - | 预处理数据根目录 |
| `log_dir` | string | 是 | - | 训练输出目录 |

### 阶段 2: 模型选择

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model_name` | string | 是 | - | 模型名称（从 list_models 获取） |

### 阶段 3: 训练参数

#### 时序参数（预报特有）

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `in_t` | number | 7 | 3~30 | 输入时间步数（Agent 推荐 5~14） |
| `out_t` | number | 1 | 1~14 | 输出时间步数（Agent 推荐 1~7） |
| `stride` | number | 1 | 1~in_t | 滑动窗口步长 |

#### 训练核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `epochs` | number | 500 | 训练轮次 |
| `lr` | number | 0.001 | 初始学习率 |
| `batch_size` | number | 4 | 训练批次大小 |
| `eval_batch_size` | number | 4 | 验证批次大小 |
| `patience` | number | 10 | 早停耐心度 |
| `eval_freq` | number | 5 | 评估频率（每 N 个 epoch） |
| `seed` | number | 42 | 随机种子 |

#### 优化器参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `optimizer` | string | "AdamW" | 优化器 |
| `weight_decay` | number | 0.001 | 权重衰减 |
| `scheduler` | string | "StepLR" | 调度器 |
| `scheduler_step_size` | number | 300 | StepLR 步长 |
| `scheduler_gamma` | number | 0.5 | 调度器衰减因子 |

#### 数据参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dyn_vars` | string[] | 自动检测 | 动态变量列表 |
| `normalize` | boolean | true | 是否归一化 |
| `normalizer_type` | string | "PGN" | 归一化类型 |

#### GPU / 分布式参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `device_ids` | number[] | [0] | GPU 设备 ID 列表 |
| `distribute` | boolean | false | 是否分布式训练 |
| `distribute_mode` | string | "single" | 分布式模式 |

#### OOM 防护参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_amp` | boolean | true | 混合精度（FFT 模型自动关闭） |
| `gradient_checkpointing` | boolean | true | 梯度检查点 |

### 阶段 4: 执行确认

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `user_confirmed` | boolean | 是 | 必须为 true |
| `confirmation_token` | string | 是 | 系统生成的确认令牌 |

## ocean_forecast_train_status 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `action` | string | "status", "logs", "kill", "list", "wait", "watch" |
| `process_id` | string | 训练进程 ID |
| `tail` | number | 获取最后 N 行日志（默认 100） |
| `offset` | number | 日志字节偏移量 |
| `timeout` | number | wait/watch 超时秒数（默认 120） |

## ocean_forecast_train_visualize 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `log_dir` | string | 训练日志目录 |
| `mode` | string | "train"（默认）或 "predict" |
| `dataset_root` | string | predict 模式需要 |
| `n_samples` | number | 最多可视化样本数（默认 5） |

## ocean_forecast_train_report 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `log_dir` | string | 训练日志目录 |
| `output_path` | string | 报告输出路径（可选） |
