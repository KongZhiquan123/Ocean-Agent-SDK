---
name: ocean-forecast-data-preprocess
description: 海洋预测数据预处理技能 - NC 格式时序数据转换为 NPY 格式，用于深度学习预测模型训练
version: 1.4.0
author: Leizheng
last_modified: 2026-03-10
---

<!--
Changelog:
  - 2026-03-10 Leizheng: v1.4.0
    - 新增按经纬度裁剪功能（crop_lon_range/crop_lat_range）
    - 支持 1D 规则网格和 2D 曲线网格坐标系统
    - 自动从 NC 文件加载经纬度坐标并计算像素索引
    - 新增空间裁剪说明章节
  - 2026-02-26 Leizheng: v1.3.0
    - 新增 ocean_forecast_preprocess_stats 工具（per-variable 统计量）
    - 更新可视化描述：新增分布直方图 _distribution.png
    - 完整工作流程新增统计步骤（步骤 7）
    - 输出目录结构补充 data_stats.json
  - 2026-02-26 Leizheng: v1.2.0
    - 说明 ocean_forecast_preprocess_report 的 ai_analysis 参数必填
    - 报告生成说明补充撰写指引
  - 2026-02-25 Leizheng: v1.1.0
    - 重构文档结构，参照 ocean-preprocess SKILL.md 风格精简
    - 改"预报"为"预测"，去除超分辨率对比内容
  - 2026-02-25 Leizheng: v1.0.0
    - 初始版本，4 阶段强制确认流程，严格时间排序
-->

# 海洋预测数据预处理技能

## 核心原则

1. **只做格式转换**：不做归一化，不改变数值，NC → NPY 原样转存
2. **时间严格排序**：按 NC 内时间变量升序排列，绝不打乱，保障预测序列因果关系
3. **禁止自动决策**：所有变量选择、参数设置必须由用户确认
4. **错误不自动重试**：遇到错误展示给用户，等待指示

---

## 可用工具

| 工具 | 用途 | 使用时机 |
|------|------|----------|
| `ocean_forecast_preprocess_full` | 完整预处理流程 | **推荐**：信息完整时使用 |
| `ocean_forecast_preprocess_visualize` | 生成可视化图片 | 预处理完成后检查数据质量 |
| `ocean_forecast_preprocess_stats` | 计算 per-variable 统计量 | 可视化后，生成报告前（可选） |
| `ocean_forecast_preprocess_report` | 生成预处理报告 | 可视化完成后 |
| `ocean_inspect_data` | 查看 NC 数据变量 | 用户只想了解数据结构时 |

---

## 4 阶段强制确认流程

工具在每个阶段**强制返回**，Agent 无法跳过确认步骤。

```
调用工具（无 dyn_vars）
    ↓
⛔ 阶段 1: awaiting_variable_selection
    │  → 展示动态变量，询问"您要预测哪些变量？"
    ↓
⛔ 阶段 2: awaiting_static_selection
    │  → 展示静态/掩码变量，询问"需要保存哪些？使用哪些掩码？"
    ↓
⛔ 阶段 3: awaiting_parameters
    │  → 询问"训练/验证/测试集划分比例？是否需要空间裁剪？"
    ↓
⛔ 阶段 4: awaiting_execution
    │  → 展示完整参数汇总，等待"确认执行"
    ↓
✅ 执行处理 (Step A 检查 → Step B 转换 → Step C 可视化)
    ↓
📊 可选：ocean_forecast_preprocess_stats（per-variable NaN 率、值域、分位数）
    ↓
📝 调用 ocean_forecast_preprocess_report（ai_analysis 必填）生成报告
```

**Token 机制**：阶段 4 返回 `confirmation_token`，下次调用必须携带，防止跳过阶段。

**报告生成说明**：调用 `ocean_forecast_preprocess_report` 时，`ai_analysis` 参数必须由 Agent 根据可视化结果和处理统计撰写，不能留空。应包含：数据质量评估、时间分布合理性、是否有异常值等观察。

---

## 快速参数参考

### ocean_forecast_preprocess_full 核心参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `nc_folder` | string | ✅ | NC 文件目录 |
| `output_base` | string | ✅ | 输出根目录 |
| `dyn_vars` | string[] | ✅ | 预测变量（阶段 1 后必填） |
| `stat_vars` | string[] | ✅ | 静态变量（阶段 2 后必填，可为 `[]`） |
| `mask_vars` | string[] | ✅ | 掩码变量（阶段 2 后必填，可为 `[]`） |
| `train_ratio` | number | ⚠️ | 训练集比例（阶段 3 后必填） |
| `valid_ratio` | number | ⚠️ | 验证集比例（阶段 3 后必填） |
| `test_ratio` | number | ⚠️ | 测试集比例（阶段 3 后必填） |
| `user_confirmed` | boolean | ⚠️ | 阶段 4 必需 |
| `confirmation_token` | string | ⚠️ | 阶段 4 必需 |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `h_slice` | - | H 方向裁剪（像素索引），如 `"0:512"` |
| `w_slice` | - | W 方向裁剪（像素索引），如 `"0:1024"` |
| `crop_lon_range` | - | 经度裁剪范围，如 `[120, 130]`（自动转换为像素索引） |
| `crop_lat_range` | - | 纬度裁剪范围，如 `[30, 40]`（自动转换为像素索引） |
| `use_date_filename` | `true` | 用日期命名文件 |
| `date_format` | `auto` | `auto`/`YYYYMMDD`/`YYYYMMDDHH`/`YYYYMMDDHHmm` |
| `time_var` | 自动检测 | 时间变量名（自动检测 time/ocean_time 等） |
| `chunk_size` | `200` | 批处理文件数（控制内存） |
| `max_files` | 无限制 | 最多处理文件数（调试用） |
| `skip_visualize` | `false` | 跳过可视化步骤 |

### 空间裁剪说明

**两种裁剪方式**：

1. **按像素索引裁剪**（`h_slice` / `w_slice`）
   - 直接指定像素范围，如 `"100:400"`
   - 需要事先知道数据的空间维度

2. **按经纬度裁剪**（`crop_lon_range` / `crop_lat_range`）✨ 推荐
   - 指定地理坐标范围，如 `[120, 130]` / `[30, 40]`
   - 自动从 NC 文件加载经纬度坐标并计算像素索引
   - 支持 1D 规则网格和 2D 曲线网格（如 ROMS）
   - **要求**：必须同时指定 `lon_var` 和 `lat_var` 参数

**优先级**：如果同时提供经纬度范围和像素索引，**经纬度范围优先**（会覆盖像素索引）。

**示例**：
```json
{
  "crop_lon_range": [120.0, 130.0],
  "crop_lat_range": [30.0, 40.0],
  "lon_var": "lon_rho",
  "lat_var": "lat_rho"
}
```

---

## 禁止行为

| 类别 | 禁止行为 |
|------|----------|
| **变量选择** | 自动决定 dyn_vars / stat_vars / mask_vars |
| **参数决策** | 自动决定划分比例、裁剪范围 |
| **流程控制** | 跳过确认阶段、跳过报告生成 |
| **错误处理** | 自动重试、自动更换路径 |

---

## 完整工作流程

```
1. 收集信息 → 数据目录、输出目录
2. 分析数据 → ocean_forecast_preprocess_full（触发 Step A 检查）
3. 用户选择 → 预测变量、静态变量、掩码变量
4. 确认参数 → 划分比例、可选空间裁剪
5. 执行前确认 → 展示所有参数，等待"确认执行"
6. 执行处理 → ocean_forecast_preprocess_full (user_confirmed=true)
7. 数据质量量化（可选）→ ocean_forecast_preprocess_stats（NaN 率、值域统计）
8. 生成报告 → ocean_forecast_preprocess_report + Agent 填写分析
```

---

## 参考文档索引

详细信息请按需读取以下文档：

| 文档 | 内容 | 何时读取 |
|------|------|----------|
| `references/workflow-detail.md` | 4 阶段流程详解 | 需要了解完整流程时 |
| `references/examples.md` | 对话示例 | 需要参考示例时 |
| `references/parameters.md` | 所有工具参数 | 需要参数细节时 |
| `references/errors.md` | 错误处理指南 | 遇到错误时 |
| `references/warnings.md` | 警告处理指南 | 遇到警告时 |

---

## 输出目录结构

```
output_base/
├── train/
│   ├── {var}/      ← 每个预测变量一个目录
│   │   ├── 20200101.npy   ← 按时间升序，文件名为日期
│   │   └── ...
│   └── ...
├── valid/
├── test/
├── static_variables/       ← 静态变量（坐标）和掩码
├── time_index.json         ← 完整时间戳溯源
├── var_names.json          ← 变量配置（供 DataLoader 使用）
├── data_stats.json         ← per-variable 统计量（ocean_forecast_preprocess_stats 生成）
├── preprocess_manifest.json
├── visualisation_forecast/ ← 可视化图片
│   ├── train/
│   │   ├── {var}_frames.png        ← 4 帧空间分布图
│   │   ├── {var}_timeseries.png    ← 时序均值/标准差图
│   │   └── {var}_distribution.png ← 值域分布直方图（含 P5/P95）
│   └── ...
└── preprocessing_report.md
```
