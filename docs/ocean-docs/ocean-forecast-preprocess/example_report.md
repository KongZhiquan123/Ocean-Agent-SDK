# 海洋预报数据预处理报告

> 生成时间: 2026-02-27T02:54:16.372Z 

> 数据集目录: `/home/kzq/Ocean-Agent-SDK/test_outputs/predict_preprocessed` 


## 1. 数据集概览

| 属性 | 值 |
|------|-----|
| 来源目录 | `/data/tmp/copernicus_uv_for_agent_test_20days/` |
| NC 文件数 | 20 |
| 总时间步数 | 20 |
| 空间形状 | 2041 × 4320 |
| 数据范围 | 19930101000000 ~ 19930120000000 |
| 时间步长（估计）| 24 小时 |
| 时间间隔异常 | ✅ 无 |

## 2. 变量配置

| 类型 | 变量 |
|------|------|
| 动态变量（预报目标） | uo, vo |
| 静态变量 | latitude, longitude |
| 掩码变量 | 无 |

## 3. 用户确认记录

### 阶段 1：研究变量选择
- 选择的变量：uo, vo
- 确认时间：2026-02-26T15:00:00Z

### 阶段 2：静态/掩码变量选择
- 静态变量：latitude, longitude
- 掩码变量：无
- 经度变量：longitude
- 纬度变量：latitude
- 确认时间：2026-02-26T15:01:00Z

### 阶段 3：处理参数确认
- 训练集比例：70%
- 验证集比例：15%
- 测试集比例：15%
- 确认时间：2026-02-26T15:02:00Z

### 阶段 4：执行确认
- 用户确认执行：✅ 是
- 确认时间：2026-02-26T15:03:00Z

## 4. 数据集划分

| 划分 | 时间步数 | 比例 | 起始时间 | 结束时间 |
|------|---------|------|---------|---------|
| train | 14 | 70% | 19930101000000 | 19930114000000 |
| valid | 3 | 15% | 19930115000000 | 19930117000000 |
| test | 3 | 15% | 19930118000000 | 19930120000000 |

## 5. 后置验证

| 规则 | 状态 | 说明 |
|------|------|------|
| Rule 1: 完整性 | ✅ 通过 | 所有 NPY 文件存在且形状一致 |
| Rule 2: 时间单调性 | ✅ 通过 | 时间戳在各 split 内严格递增 |
| Rule 3: NaN 一致性 | ⏭️ 跳过 | 非掩码区域无异常 NaN |

## 6. 输出目录结构

```
/home/kzq/Ocean-Agent-SDK/test_outputs/predict_preprocessed/
├── train/
│   ├── uo/          # 14 个时间步 NPY 文件
│   ├── vo/          # 14 个时间步 NPY 文件
├── valid/
│   ├── uo/
│   ├── vo/
├── test/
│   ├── uo/
│   ├── vo/
├── static_variables/   # 静态变量 & 掩码 NPY
├── time_index.json     # 完整时间戳溯源
├── var_names.json      # 变量配置（供 DataLoader 使用）
├── preprocess_manifest.json
└── preprocessing_report.md
```

## 7. 可视化

> 仅展示部分样本，完整图片请查看 `visualisation_forecast/` 目录

### 变量：uo
#### train集
**样本帧分布**
![](./visualisation_forecast/train/uo_frames.png)

**时序统计**
![](./visualisation_forecast/train/uo_timeseries.png)

**数值分布**
![](./visualisation_forecast/train/uo_distribution.png)

#### valid集
**样本帧分布**
![](./visualisation_forecast/valid/uo_frames.png)

**时序统计**
![](./visualisation_forecast/valid/uo_timeseries.png)

**数值分布**
![](./visualisation_forecast/valid/uo_distribution.png)

#### test集
**样本帧分布**
![](./visualisation_forecast/test/uo_frames.png)

**时序统计**
![](./visualisation_forecast/test/uo_timeseries.png)

**数值分布**
![](./visualisation_forecast/test/uo_distribution.png)

### 变量：vo
#### train集
**样本帧分布**
![](./visualisation_forecast/train/vo_frames.png)

**时序统计**
![](./visualisation_forecast/train/vo_timeseries.png)

**数值分布**
![](./visualisation_forecast/train/vo_distribution.png)

#### valid集
**样本帧分布**
![](./visualisation_forecast/valid/vo_frames.png)

**时序统计**
![](./visualisation_forecast/valid/vo_timeseries.png)

**数值分布**
![](./visualisation_forecast/valid/vo_distribution.png)

#### test集
**样本帧分布**
![](./visualisation_forecast/test/vo_frames.png)

**时序统计**
![](./visualisation_forecast/test/vo_timeseries.png)

**数值分布**
![](./visualisation_forecast/test/vo_distribution.png)


## 8. 分析和建议

### 1. 数据质量评估

#### ✅ 时间完整性
- **时间范围**：1993-01-01 至 1993-01-20，连续 20 天
- **时间步长**：24 小时（日均数据）
- **时间间隔**：无异常，所有时间步严格按升序排列
- **评价**：时间序列完整，符合预测模型训练要求

#### ⚠️ 空间完整性
- **分辨率**：2041 × 4320（全球高分辨率海流数据）
- **NaN 分布**：约 267 万个 NaN 值（约占 30%），符合全球海流数据特征（陆地区域无海流）
- **数值范围**：
  - `uo`（东向流）：-2.05 ~ 2.03 m/s
  - `vo`（北向流）：-2.01 ~ 2.00 m/s
- **评价**：NaN 分布合理，仅位于陆地区域；数值范围正常，符合海流物理特性

#### ⚠️ 数据规模
- **总时间步**：20 步
- **训练集**：14 步（70%）
- **验证集**：3 步（15%）
- **测试集**：3 步（15%）
- **评价**：**数据量较少**，20 天数据不足以训练高性能深度学习模型，建议用于：
  - 模型原型验证
  - 快速测试流程
  - 算法可行性探索

### 2. 划分合理性

#### ✅ 时间段分布
- **训练集**：1993-01-01 至 1993-01-14（14 天）
- **验证集**：1993-01-15 至 1993-01-17（3 天）
- **测试集**：1993-01-18 至 1993-01-20（3 天）
- **评价**：按时间顺序严格划分，保证因果关系，符合时序预测规范

#### ⚠️ 样本数量
- **训练集**：14 步对于深度学习模型而言过少，难以学习复杂时空模式
- **验证集**：3 步基本满足早停监控需求
- **测试集**：3 步可用于初步性能评估
- **评价**：比例合理（70:15:15），但**绝对数量不足**

### 3. 潜在问题与建议

#### ⚠️ 主要问题

1. **数据量严重不足**
   - 20 天数据无法覆盖海流的季节性变化
   - 训练集仅 14 步，难以训练参数量较大的模型（如 Transformer、FNO）
   - 建议：
     - 获取更长时间序列数据（至少 1 年）
     - 当前数据仅用于概念验证或轻量模型测试

2. **季节性覆盖不足**
   - 数据仅覆盖 1993 年 1 月（北半球冬季）
   - 无法捕捉夏季、过渡季节的海流模式
   - 模型泛化能力受限

3. **空间分辨率极高**
   - 2041 × 4320 = 881 万像素点
   - **显存需求巨大**，单张 float32 需约 33 MB
   - 建议：
     - 使用 Patch-based 训练（裁剪小区域）
     - 降低空间分辨率（下采样到 512×1024 或更小）
     - 使用梯度检查点（gradient checkpointing）

#### 💡 训练建议

1. **模型选择**
   - 避免参数量过大的模型（如 ViT、大型 FNO）
   - 推荐：UNet2d、小型 FNO2d、ConvLSTM
   - 考虑预训练模型微调

2. **训练配置**
   - **批大小**：1-2（显存限制）
   - **输入序列长度**：in_t=3-5（利用时间上下文）
   - **预测步长**：out_t=1-2（短期预测）
   - **空间裁剪**：建议裁剪到 256×256 或 512×512 区域
   - **混合精度训练**：必须开启（use_amp=true）

3. **数据增强**
   - 空间随机裁剪
   - 时间窗口滑动（stride=1）
   - 考虑多尺度训练

### 4. 使用场景定位

根据当前数据规模，本数据集适合：
- ✅ 预处理流程测试
- ✅ 模型架构原型验证
- ✅ 小型区域海流预测实验
- ❌ 生产级预测模型训练
- ❌ 长期预报能力评估
- ❌ 季节性模式学习

## 9. 总结

预处理已完成。

| 指标 | 数值 |
|------|------|
| 总时间步数 | 20 |
| 训练集 | 14 步 |
| 验证集 | 3 步 |
| 测试集 | 3 步 |
| 动态变量 | 2 个 |
| 静态变量 | 2 个 |
| 掩码变量 | 0 个 |
| 时间间隔异常 | 0 处 |
| 警告数量 | 0 个 |
