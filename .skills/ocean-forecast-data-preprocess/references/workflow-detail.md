# 工作流程详解

> 版本: 1.3.0 | 最后更新: 2026-02-26

本文档详细说明海洋预测数据预处理的完整工作流程。

---

## Step 1：信息收集

从用户的消息中提取已有信息，追问缺失的：

| 信息 | 必需 | 如果缺失 |
|------|------|---------|
| 数据目录（nc_folder） | ✅ | 追问 |
| 输出目录（output_base） | ✅ | 追问 |
| 预测变量（dyn_vars） | ⚠️ | 可以先分析数据再让用户选 |
| 划分比例 | ⚠️ | 阶段 3 追问 |

**注意**：如果用户已经提供了变量名、划分比例等，先记下来，后续阶段用。

---

## 流程总览

```
用户提供数据目录 & 输出目录
    ↓
调用 ocean_forecast_preprocess_full（无 dyn_vars）
    ↓
Step A: 分析 NC 文件，检测所有变量
    ↓
⛔ 阶段 1: awaiting_variable_selection
    → 展示动态变量列表 + 时间信息
    → 询问：您要预测哪些变量？
    ↓ [用户选择 dyn_vars]
⛔ 阶段 2: awaiting_static_selection
    → 展示静态/坐标变量，展示疑似掩码变量
    → 询问：需要保存哪些静态变量？使用哪些掩码？
    ↓ [用户确认 stat_vars + mask_vars]
⛔ 阶段 3: awaiting_parameters
    → 询问训练/验证/测试集划分比例
    → 询问是否需要空间裁剪（h_slice / w_slice）
    ↓ [用户提供比例]
⛔ 阶段 4: awaiting_execution
    → 展示完整参数汇总
    → 返回 confirmation_token
    → 等待用户"确认执行"
    ↓ [用户确认，携带 user_confirmed=true + confirmation_token]
✅ Step B: 执行 forecast_preprocess.py
    → 按时间升序处理 NC 文件
    → 输出 train / valid / test 目录
    → 保存 static_variables、time_index.json、var_names.json
    ↓
✅ Step C: 执行可视化（调用 ocean_forecast_visualize）
    → 每变量生成：_frames.png + _timeseries.png + _distribution.png
    ↓
📊 可选: ocean_forecast_stats（量化 NaN 率、值域、P5/P95）
    ↓
📝 调用 ocean_forecast_generate_report 生成报告
```

---

## 阶段 1：变量选择（awaiting_variable_selection）

**触发条件**：调用 `ocean_forecast_preprocess_full` 时未提供 `dyn_vars`。

**工具行为**：
1. 扫描 NC 目录（最多采样 3 个文件）
2. 检测所有变量，推断动态/静态/掩码分类
3. 检测时间变量（time、ocean_time 等）
4. 返回 `status: "awaiting_variable_selection"`

**Agent 展示内容**：

```
✅ 数据分析完成！

📁 数据目录：/data/ocean/nc_files
📊 文件数量：1095 个 NC 文件
⏱ 时间范围：2020-01-01 → 2022-12-31
🔲 空间分辨率：H=720, W=1440

🌊 动态变量（时序数据，可用于预测）：
  - sst: sea surface temperature [T, H, W]
  - ssh: sea surface height [T, H, W]
  - sss: sea surface salinity [T, H, W]

请问您要预测哪些变量？（可多选）
```

---

## 阶段 2：静态与掩码选择（awaiting_static_selection）

**触发条件**：已提供 `dyn_vars`，但未提供 `stat_vars` 和 `mask_vars`。

**工具行为**：
- 展示检测到的静态/坐标变量（lon、lat 等）
- 展示疑似掩码变量（mask、land_sea 等）
- 返回 `status: "awaiting_static_selection"`

**Agent 展示内容**：

```
📍 静态/坐标变量（形状固定，不随时间变化）：
  - lon: 经度 [H, W]
  - lat: 纬度 [H, W]
  - depth: 海底深度 [H, W]

🗺 疑似掩码变量（用于区分海洋/陆地）：
  - mask: 海陆掩码 [H, W]  ← 推荐
  - valid_flag: 有效数据标记 [H, W]

请确认：
1. 需要保存哪些静态变量？（可填 [] 表示不保存）
2. 使用哪个掩码变量？（可填 [] 表示不使用）
```

---

## 阶段 3：参数配置（awaiting_parameters）

**触发条件**：已提供 `dyn_vars`、`stat_vars`、`mask_vars`，但未提供划分比例。

**工具行为**：
- 返回 `status: "awaiting_parameters"`
- 提示需要 `train_ratio`、`valid_ratio`、`test_ratio`

**Agent 展示内容**：

```
⚙️ 请配置数据集划分参数：

训练/验证/测试集比例（三者之和 = 1）：
  - 常用方案：0.7 / 0.1 / 0.2
  - 时序数据建议：0.7 / 0.15 / 0.15

可选：空间裁剪（如果不需要请留空）：
  - H 方向：如 "0:512"（从第 0 行裁到第 512 行）
  - W 方向：如 "0:1024"
```

---

## 阶段 4：执行确认（awaiting_execution）

**触发条件**：所有参数已提供，`user_confirmed` 为 false 或未提供。

**工具行为**：
- 汇总所有参数
- 生成 `confirmation_token`（SHA-256，防跳过）
- 返回 `status: "awaiting_execution"`

**Agent 展示内容**：

```
📋 参数汇总，请确认：

┌─────────────────────────────────────────────────┐
│  数据目录：/data/ocean/nc_files                 │
│  输出目录：/output/forecast_data                │
│  预测变量：sst, ssh                             │
│  静态变量：lon, lat                             │
│  掩码变量：mask                                 │
│  划分比例：训练 70% / 验证 10% / 测试 20%      │
│  预计文件：训练 766 / 验证 110 / 测试 219       │
└─────────────────────────────────────────────────┘

确认无误后请回复"确认执行"。
```

**判断用户是否确认的标准**：
- ✅ 用户明确说："确认执行"、"确认"、"开始"、"执行"、"好的，开始吧"
- ❌ 用户只是回答了某个问题，但没有明确确认
- ❌ 用户提供了参数值，但没有说确认

---

## 执行阶段：Step B（forecast_preprocess.py）

**处理逻辑**：

1. **时间排序**：读取每个 NC 文件内的时间变量，按时间升序排列（关键：保障因果关系）
2. **划分数据集**：按时间顺序切分 train / valid / test，不打乱
3. **格式转换**：每个时间步 → 一个 `.npy` 文件（文件名为日期，如 `20200101.npy`）
4. **目录结构**：每个动态变量独立目录；静态变量集中存放
5. **元数据输出**：
   - `time_index.json`：完整时间戳溯源（文件名 → 原始时间值）
   - `var_names.json`：变量配置（供 DataLoader 使用）
   - `preprocess_manifest.json`：处理参数记录

---

## 执行阶段：Step C（可视化）

**自动调用** `ocean_forecast_visualize`，每个变量生成三种图：

- `{var}_frames.png`：4 个均匀采样时间步的空间分布图
- `{var}_timeseries.png`：全时间序列的均值/标准差折线图
- `{var}_distribution.png`：值域分布直方图，标注 P5/P95 分位数（异常值判断）
- 输出到 `{output_base}/visualisation_forecast/{split}/`

---

## 可选阶段：统计分析（ocean_forecast_stats）

**在可视化完成后、报告生成前**，可选调用 `ocean_forecast_stats` 对已预处理数据做量化评估。

**输出** `{dataset_root}/data_stats.json`，包含 per-variable：
- `nan_rate`：NaN 占比（> 30% 自动生成警告）
- `min` / `max`：值域
- `mean` / `std`：均值和标准差
- `p5` / `p95`：分位数

**调用示例**：
```json
{
  "dataset_root": "/output/forecast_data"
}
```

**何时调用**：
- 用户要求量化数据质量时
- 可视化图发现分布异常（如直方图严重偏态）需要确认数值时
- 报告 `ai_analysis` 需要引用具体数值时

---

## 报告生成

**最后调用** `ocean_forecast_generate_report`，生成 `preprocessing_report.md`，包含：

- 数据集统计（文件数、时间范围、变量列表）
- 划分结果（各集合时间步数）
- 可视化图片路径（包含三类图：帧图、时序图、分布图）
- AI 分析评论（Agent 填写，可引用 `data_stats.json` 中的具体数值）
