# 错误处理指南

## CUDA OOM（显存不足）

### 症状
```
RuntimeError: CUDA out of memory. Tried to allocate X GiB
```

### 原因
输入张量 (B, H, W, in_t * C) 过大，超出 GPU 显存。

### 解决方案（按优先级）
1. **减小 batch_size**: 4 → 2 → 1
2. **启用 AMP**: use_amp=true（非 FFT 模型）
3. **启用 gradient_checkpointing**: gradient_checkpointing=true
4. **减小 in_t**: 14 → 7 → 5
5. **使用更大显存的 GPU**

### 自动防护
工具会自动尝试减小 batch_size（最多 5 次），无需手动干预。

---

## 数据维度不匹配

### 症状
```
RuntimeError: shape mismatch
ValueError: Cannot detect spatial shape
```

### 原因
- NPY 文件形状不一致
- var_names.json 中的 spatial_shape 与实际不符
- dyn_vars 列表与实际目录不匹配

### 解决方案
1. 运行 `ocean_forecast_train_start` 第一阶段验证数据集
2. 检查 var_names.json 配置
3. 确保所有变量目录下的 NPY 文件形状一致

---

## 空间尺寸约束

### 症状
```
Warning: 空间尺寸 301×301 不能被 16 整除，UNet2d 等模型可能需要 padding
```

### 原因
UNet2d 的编码器-解码器结构需要空间尺寸能被 16 整除。

### 解决方案
1. 在预处理阶段裁剪/填充到合适尺寸
2. 改用 FNO2d（无尺寸约束）
3. 改用 Transformer（无尺寸约束）

---

## 训练样本不足

### 症状
```
Warning: train 仅有 12 个时间步，可能不足以训练
```

### 原因
in_t + out_t > 数据总时间步数，导致可用滑动窗口样本过少。

### 解决方案
1. 减小 in_t（如 7 → 3）
2. 减小 out_t（如 7 → 1）
3. 增大 stride（如 1 → 2，但会减少样本数）
4. 增大数据量（更长的时间范围）

---

## NaN 值问题

### 症状
- 训练 loss 变为 NaN
- 预测结果全为 NaN

### 原因
- 输入数据中 NaN 比例过高
- AMP 混合精度导致数值溢出（FFT 模型）
- 学习率过大

### 解决方案
1. 检查输入数据 NaN 比例（>50% 可能有问题）
2. 对 FFT 模型关闭 AMP: use_amp=false
3. 降低学习率: lr=0.0001
4. 在预处理阶段处理 NaN（如插值填充）

---

## 模型构建失败

### 症状
```
KeyError: 'model_name' not in MODEL_REGISTRY
```

### 原因
模型名称拼写错误或未注册。

### 解决方案
使用 `ocean_forecast_list_models` 获取正确的模型名称列表。

---

## FFT/AMP 兼容性

### 症状
```
RuntimeError: cuFFT doesn't support signals of half type
```

### 原因
FNO2d、M2NO2d 等 FFT 模型不兼容半精度（FP16）。

### 解决方案
关闭 AMP: use_amp=false（工具会自动处理）

---

## 进程管理问题

### 找不到进程
```
error: 未找到进程: xxx
```
- 进程可能已被清理（超时）
- 使用 `action="list"` 查看所有进程

### 进程已结束
```
error: 进程已结束，状态: failed
```
- 使用 `action="logs"` 查看日志
- 检查 error_summary 中的失败原因和建议
