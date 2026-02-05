---
name: ocean-preprocess
description: 海洋数据预处理技能 - 专用于超分辨率场景的NC到NPY数据格式转换
version: 3.3.0
author: kongzhiquan
contributors: leizheng
last_modified: 2026-02-05
---

<!--
Changelog:
  - 2026-02-05 kongzhiquan: v3.3.0
    - 新增日期文件名功能（use_date_filename, date_format, time_var 参数）
    - 支持从 NC 文件提取时间戳作为 NPY 文件名
    - 自动检测日期格式（日/小时/分钟级数据）
    - 重复日期自动添加时间后缀
  - 2026-02-05 kongzhiquan: v3.2.0
    - 重构文档结构：核心内容精简至 ~150 行
    - 详细文档移至 references/ 目录
    - 新增 references/workflow-detail.md, examples.md, parameters.md, errors.md, warnings.md
  - 2026-02-05 kongzhiquan: v3.1.0 - 状态机架构、Token 机制、区域裁剪
  - 2026-02-04 leizheng: v2.9.0 - 4 阶段强制停止点
  - 2026-02-04 leizheng: v2.6.0 - 粗网格模式支持
-->

# 海洋数据预处理技能

## 核心原则

1. **数据预处理定义**：不破坏原有数据结构，不做标准化，只做格式转换
2. **警告优先**：任何警告必须暂停询问用户，不能自动继续
3. **禁止自动决策**：所有变量选择、参数设置必须由用户确认
4. **错误不自动重试**：遇到错误展示给用户，等待指示

---

## 可用工具

| 工具 | 用途 | 使用时机 |
|------|------|----------|
| `ocean_preprocess_full` | 完整预处理流程 | **推荐**：信息完整时使用 |
| `ocean_metrics` | 质量指标计算 | **必须**：预处理后调用 |
| `ocean_generate_report` | 生成报告 | 指标计算后生成 |
| `ocean_inspect_data` | 查看数据变量 | 用户只想了解数据时 |

---

## 4 阶段强制确认流程

工具在每个阶段**强制返回**，Agent 无法跳过确认步骤。

```
调用工具（无 dyn_vars）
    ↓
⛔ 阶段 1: awaiting_variable_selection
    │  → 展示动态变量，询问"您要研究哪些变量？"
    ↓
⛔ 阶段 2: awaiting_static_selection
    │  → 展示静态/掩码变量，询问"需要保存哪些？使用哪些掩码？"
    ↓
⛔ 阶段 3: awaiting_parameters
    │  → 询问"下采样倍数？插值方法？划分比例？"
    ↓
⛔ 阶段 4: awaiting_execution
    │  → 展示完整参数汇总，等待"确认执行"
    ↓
✅ 执行处理 (A→B→C→D→E)
    ↓
📊 调用 ocean_metrics 计算指标
    ↓
📝 调用 ocean_generate_report 生成报告
```

**Token 机制**：每个阶段返回 `confirmation_token`，下次调用必须携带，防止跳过阶段。

---

## 快速参数参考

### ocean_preprocess_full 核心参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `nc_folder` | string | ✅ | 数据目录 |
| `output_base` | string | ✅ | 输出目录 |
| `dyn_vars` | string[] | ✅ | 研究变量 |
| `user_confirmed` | boolean | ⚠️ | 阶段 4 必需 |
| `confirmation_token` | string | ⚠️ | 阶段 2+ 必需 |
| `scale` | number | ⚠️ | 下采样倍数（下采样模式） |
| `downsample_method` | string | ⚠️ | 插值方法（下采样模式） |
| `lr_nc_folder` | string | ⚠️ | LR 数据目录（粗网格模式） |

**完整参数说明**：见 `references/parameters.md`

---

## 禁止行为（核心）

| 类别 | 禁止行为 |
|------|----------|
| **变量选择** | 自动决定 dyn_vars / stat_vars / mask_vars |
| **参数决策** | 自动决定 scale、划分比例、NaN 处理方式 |
| **流程控制** | 跳过确认阶段、跳过质量指标计算 |
| **错误处理** | 自动重试、自动更换路径 |

---

## 两种超分模式

| 模式 | 说明 | 关键参数 |
|------|------|----------|
| **下采样** | HR 数据下采样生成 LR | `scale` + `downsample_method` |
| **粗网格** | HR/LR 数据由模型分别生成 | `lr_nc_folder` |

### 日期文件名（默认开启）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `use_date_filename` | 使用日期作为文件名 | `true` |
| `date_format` | 日期格式: `auto`/`YYYYMMDD`/`YYYYMMDDHH`/`YYYYMMDDHHmm` | `auto` |
| `time_var` | 时间变量名（自动检测 time/ocean_time） | - |

**输出文件名示例**：`20200101.npy`, `20200102.npy`（自动从 NC 文件提取日期）

---

## 完整工作流程

```
1. 收集信息 → 数据目录、输出目录
2. 分析数据 → ocean_inspect_data 或 ocean_preprocess_full
3. 用户选择 → 研究变量、静态变量、掩码变量
4. 确认参数 → scale、method、划分比例
5. 执行前确认 → 展示所有参数，等待"确认执行"
6. 执行处理 → ocean_preprocess_full (user_confirmed=true)
7. 计算指标 → ocean_metrics（必须）
8. 生成报告 → ocean_generate_report + Agent 填写分析
```

**详细流程**：见 `references/workflow-detail.md`

---

## 参考文档索引

详细信息请按需读取以下文档：

| 文档 | 内容 | 何时读取 |
|------|------|----------|
| `references/workflow-detail.md` | Step 1-7 详细流程 | 需要了解完整流程时 |
| `references/examples.md` | 对话示例 | 需要参考示例时 |
| `references/parameters.md` | 所有工具参数 | 需要参数细节时 |
| `references/errors.md` | 错误处理指南 | 遇到错误时 |
| `references/warnings.md` | 警告处理指南 | 遇到警告时 |

---

## 输出目录结构

```
output/
├── train/
│   ├── hr/         ← 高分辨率数据
│   └── lr/         ← 低分辨率数据
├── valid/
├── test/
├── static_variables/
├── visualisation_data_process/
├── metrics_result.json
└── preprocessing_report.md
```
