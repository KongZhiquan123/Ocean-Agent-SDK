---
name: ocean-preprocess
description: 海洋数据预处理技能 - 专用于超分辨率场景的NC到NPY数据格式转换
version: 3.0.0
author: kongzhiquan
contributors: leizheng
last_modified: 2026-02-04
---

<!--
Changelog:
  - 2026-02-04 kongzhiquan: v3.0.0
    - 重写工具调用流程，完整实现 4 阶段强制确认
    - 新增状态追踪机制，防止 Agent 失忆
    - 报告工具新增 user_confirmation 参数
    - 更新输出目录结构（逐时间步保存）
    - 可视化工具新增统计分布图（均值/方差时序、直方图）
  - 2026-02-04 leizheng: v2.9.0
    - 工具层面实现4阶段强制停止点
    - 阶段1: awaiting_variable_selection - 研究变量选择
    - 阶段2: awaiting_static_selection - 静态/掩码变量选择
    - 阶段3: awaiting_parameters - 处理参数确认（scale/method/split）
    - 阶段4: awaiting_execution - 执行前最终确认
    - NPY 文件改为逐时间步保存（train/hr/uo/000000.npy）
  - 2026-02-04 leizheng: v2.8.0
    - 强化变量选择流程，研究变量/静态变量/掩码变量必须由用户选择
    - 禁止行为清单分类整理（变量选择/参数决策/流程控制/错误处理）
  - 2026-02-04 kongzhiquan: v2.6.0
    - 新增报告生成工具 ocean_generate_report
    - 报告分析部分由 Agent 自行填写（占位符机制）
  - 2026-02-04 leizheng: v2.6.0
    - 支持粗网格模式（数值模型方式）
    - 支持从动态文件中提取静态变量
  - 2026-02-03 leizheng: v2.5.0
    - ocean_preprocess_full 集成下采样和可视化
    - 完整流程变为 A→B→C→D→E 五步
-->

# 海洋数据预处理技能

## 核心原则

1. **数据预处理定义**：不破坏原有数据结构的任何信息，不做标准化，只做格式转换

2. **⚠️ 4 阶段强制确认原则（v3.0 核心）**：
   - 工具会在 4 个关键节点**强制停止**，等待用户确认
   - Agent **禁止跳过任何阶段**，禁止猜测用户意图
   - 每个阶段都有明确的输入输出，形成完整的状态链

3. **⚠️ 状态追踪原则（防止失忆）**：
   - Agent 必须维护一个**会话状态对象**，记录每个阶段的用户选择
   - 每次工具调用时，必须传入之前收集的所有参数
   - 最终生成报告时，必须传入完整的 `user_confirmation` 对象

4. **⚠️ 禁止自动决策原则**：
   - **不得代替用户做任何数据处理决策**
   - 所有变量名必须从数据中检测到或由用户明确指定
   - 禁止使用硬编码默认值（如 "lon_rho", "mask_rho"）

---

## 可用工具

| 工具名 | 用途 | 何时使用 |
|--------|------|----------|
| `ocean_preprocess_full` | 完整流程 A→B→C→D→E | **主工具**，4 阶段交互式执行 |
| `ocean_generate_report` | 生成预处理报告 | 预处理完成后，**必须**生成报告 |
| `ocean_inspect_data` | 只查看数据 | 用户只想看变量列表时 |
| `ocean_downsample` | HR→LR 下采样 | 单独执行（full 已集成） |
| `ocean_visualize` | 可视化对比+统计分布 | 单独执行（full 已集成） |
| `ocean_metrics` | 质量指标 | 单独执行（full 已集成） |

---

## ⭐ 完整工作流程（4 阶段 + 报告）

### 流程总览

```
用户请求 → 阶段1 → 阶段2 → 阶段3 → 阶段4 → 执行 → 报告生成
              ↓        ↓        ↓        ↓
           研究变量  静态/掩码  处理参数  最终确认
```

### 状态追踪对象（Agent 必须维护）

在整个流程中，Agent 必须在内部维护一个状态对象，记录每个阶段的用户选择：

```
会话状态 = {
  // 基础信息（用户首次提供）
  nc_folder: "/data/ocean",
  output_base: "/output/dataset",

  // 阶段1 收集
  dyn_vars: null,           // 用户选择后填入

  // 阶段2 收集
  mask_vars: null,
  stat_vars: null,
  lon_var: null,
  lat_var: null,

  // 阶段3 收集
  scale: null,
  downsample_method: null,
  train_ratio: null,
  valid_ratio: null,
  test_ratio: null,
  h_slice: null,
  w_slice: null,

  // 阶段4 收集
  user_confirmed: false
}
```

---

## 阶段 0：启动分析

### 触发条件
用户请求预处理数据，提供了数据目录和输出目录。

### Agent 行为

1. **追问缺失的基础信息**（如果用户没提供）：
   ```
   请提供以下信息：
   1. NC 数据文件所在目录是？
   2. 处理结果输出到哪个目录？
   ```

2. **首次调用工具**（只传基础参数）：
   ```json
   {
     "nc_folder": "/data/ocean",
     "output_base": "/output/dataset"
   }
   ```

   **⚠️ 注意**：首次调用时**不要传** `dyn_vars`，让工具分析数据后返回候选列表。

### 工具返回
```json
{
  "overall_status": "awaiting_variable_selection",
  "message": "请选择研究变量",
  "step_a": {
    "dynamic_vars_candidates": ["uo", "vo", "temp", "salt", "chl"],
    "static_vars_found": ["lon", "lat", "h"],
    "mask_vars_found": ["mask"]
  }
}
```

---

## 阶段 1：研究变量选择

### 触发条件
工具返回 `overall_status: "awaiting_variable_selection"`

### Agent 必须做的事

1. **向用户展示检测到的动态变量**：
   ```
   我已分析您的数据，检测到以下动态变量候选：

   【可选的研究变量】
   - uo (东向流速)
   - vo (北向流速)
   - temp (温度)
   - salt (盐度)
   - chl (叶绿素)

   请告诉我您要研究哪些变量？
   ```

2. **等待用户回复**，记录选择：
   ```
   用户: 我要研究 uo 和 vo

   → 更新状态: dyn_vars = ["uo", "vo"]
   ```

3. **再次调用工具**，传入已收集的参数：
   ```json
   {
     "nc_folder": "/data/ocean",
     "output_base": "/output/dataset",
     "dyn_vars": ["uo", "vo"]
   }
   ```

---

## 阶段 2：静态/掩码变量选择

### 触发条件
工具返回 `overall_status: "awaiting_static_selection"`

### 工具返回内容
```json
{
  "overall_status": "awaiting_static_selection",
  "message": "请确认静态变量和掩码变量",
  "step_a": {
    "static_vars_found": ["lon", "lat", "h", "angle"],
    "mask_vars_found": ["mask", "mask_u", "mask_v"],
    "coord_vars_detected": {
      "lon_candidates": ["lon", "longitude"],
      "lat_candidates": ["lat", "latitude"]
    }
  }
}
```

### Agent 必须做的事

1. **向用户展示检测结果并逐一询问**：
   ```
   检测到以下变量，请逐一确认：

   【1. 静态变量】（会保存到 static_variables/ 目录）
   检测到: lon, lat, h, angle
   → 您需要保存哪些？

   【2. 掩码变量】（用于区分海洋/陆地）
   检测到: mask, mask_u, mask_v
   → 使用哪些作为掩码？

   【3. 坐标变量】（用于可视化）
   经度候选: lon, longitude
   纬度候选: lat, latitude
   → 使用哪个作为经度？哪个作为纬度？
   ```

2. **等待用户逐一回复**：
   ```
   用户: 静态变量保存 lon, lat；掩码用 mask；坐标就用 lon 和 lat

   → 更新状态:
     stat_vars = ["lon", "lat"]
     mask_vars = ["mask"]
     lon_var = "lon"
     lat_var = "lat"
   ```

3. **再次调用工具**：
   ```json
   {
     "nc_folder": "/data/ocean",
     "output_base": "/output/dataset",
     "dyn_vars": ["uo", "vo"],
     "stat_vars": ["lon", "lat"],
     "mask_vars": ["mask"],
     "lon_var": "lon",
     "lat_var": "lat"
   }
   ```

---

## 阶段 3：处理参数确认

### 触发条件
工具返回 `overall_status: "awaiting_parameters"`

### 工具返回内容
```json
{
  "overall_status": "awaiting_parameters",
  "message": "请确认处理参数",
  "data_info": {
    "time_steps": 365,
    "height": 681,
    "width": 1440
  },
  "crop_recommendation": {
    "needs_crop": true,
    "reason": "H=681 不能被 4 整除",
    "suggested_h_slice": "0:680",
    "suggested_w_slice": null
  }
}
```

### Agent 必须做的事

1. **向用户展示数据信息和推荐参数**：
   ```
   数据信息：
   - 时间步数: 365
   - 空间尺寸: 681 × 1440

   请确认以下处理参数：

   【1. 下采样设置】
   - scale (下采样倍数): ? (如 4 表示缩小为 1/4)
   - method (插值方法): ? (推荐 area)
     可选: area(推荐), cubic, linear, nearest, lanczos

   【2. 数据集划分】
   - 训练集比例: ? (如 0.7 = 70%)
   - 验证集比例: ? (如 0.15 = 15%)
   - 测试集比例: ? (如 0.15 = 15%)

   【3. 裁剪设置】
   ⚠️ 系统检测: H=681 不能被 4 整除
   → 推荐裁剪: h_slice="0:680"

   请告诉我您的选择，或接受推荐值。
   ```

2. **等待用户回复**：
   ```
   用户: scale=4，用 area 方法，train 70% valid 15% test 15%，裁剪用推荐的

   → 更新状态:
     scale = 4
     downsample_method = "area"
     train_ratio = 0.7
     valid_ratio = 0.15
     test_ratio = 0.15
     h_slice = "0:680"
   ```

3. **再次调用工具**：
   ```json
   {
     "nc_folder": "/data/ocean",
     "output_base": "/output/dataset",
     "dyn_vars": ["uo", "vo"],
     "stat_vars": ["lon", "lat"],
     "mask_vars": ["mask"],
     "lon_var": "lon",
     "lat_var": "lat",
     "scale": 4,
     "downsample_method": "area",
     "train_ratio": 0.7,
     "valid_ratio": 0.15,
     "test_ratio": 0.15,
     "h_slice": "0:680"
   }
   ```

---

## 阶段 4：执行确认

### 触发条件
工具返回 `overall_status: "awaiting_execution"`

### 工具返回内容
```json
{
  "overall_status": "awaiting_execution",
  "message": "所有参数已确认，请确认执行",
  "execution_preview": {
    "input": "/data/ocean (365 files)",
    "output": "/output/dataset",
    "research_vars": ["uo", "vo"],
    "static_vars": ["lon", "lat"],
    "mask_vars": ["mask"],
    "scale": 4,
    "split": "70% / 15% / 15%",
    "crop": "H: 0:680, W: 全部"
  }
}
```

### Agent 必须做的事

1. **向用户展示执行预览**：
   ```
   ═══════════════════════════════════════
   ✓ 所有参数已确认，准备执行预处理
   ═══════════════════════════════════════

   【执行预览】
   • 输入: /data/ocean (365 个文件)
   • 输出: /output/dataset
   • 研究变量: uo, vo
   • 静态变量: lon, lat
   • 掩码变量: mask
   • 下采样: 4× (area 方法)
   • 数据集划分: train 70% / valid 15% / test 15%
   • 裁剪: H 方向 0:680

   确认执行吗？(回复"确认"开始处理)
   ```

2. **等待用户确认**：
   ```
   用户: 确认

   → 更新状态: user_confirmed = true
   ```

3. **最终调用工具**，传入所有参数 + `user_confirmed: true`：
   ```json
   {
     "nc_folder": "/data/ocean",
     "output_base": "/output/dataset",
     "dyn_vars": ["uo", "vo"],
     "stat_vars": ["lon", "lat"],
     "mask_vars": ["mask"],
     "lon_var": "lon",
     "lat_var": "lat",
     "scale": 4,
     "downsample_method": "area",
     "train_ratio": 0.7,
     "valid_ratio": 0.15,
     "test_ratio": 0.15,
     "h_slice": "0:680",
     "user_confirmed": true
   }
   ```

---

## 阶段 5：执行完成 → 生成报告

### 触发条件
工具返回 `overall_status: "pass"`

### Agent 必须做的事

1. **检查警告**：
   - 如果返回结果中有 `warnings`，必须向用户展示
   - 等待用户确认后再继续

2. **调用报告生成工具**，传入 `user_confirmation` 对象：
   ```json
   {
     "dataset_root": "/output/dataset",
     "user_confirmation": {
       "stage1_research_vars": {
         "selected": ["uo", "vo"],
         "confirmed_at": "2026-02-04 10:30:00"
       },
       "stage2_static_mask": {
         "static_vars": ["lon", "lat"],
         "mask_vars": ["mask"],
         "coord_vars": { "lon": "lon", "lat": "lat" },
         "confirmed_at": "2026-02-04 10:31:00"
       },
       "stage3_parameters": {
         "scale": 4,
         "downsample_method": "area",
         "train_ratio": 0.7,
         "valid_ratio": 0.15,
         "test_ratio": 0.15,
         "h_slice": "0:680",
         "crop_recommendation": "H=681 不能被 4 整除，建议 0:680",
         "confirmed_at": "2026-02-04 10:32:00"
       },
       "stage4_execution": {
         "confirmed": true,
         "confirmed_at": "2026-02-04 10:33:00",
         "execution_started_at": "2026-02-04 10:33:05"
       }
     }
   }
   ```

3. **读取生成的报告**，找到占位符：
   ```markdown
   ## 7. 分析和建议

   <!-- AGENT_ANALYSIS_PLACEHOLDER ... -->
   ```

4. **分析数据并填写专业分析**（参考报告中的指标数据）

5. **替换占位符**，保存报告

6. **向用户展示报告摘要**：
   ```
   ✅ 预处理完成！

   【输出目录】
   /output/dataset/
   ├── train/hr/uo/, vo/  (255 个时间步)
   ├── train/lr/...
   ├── valid/...
   ├── test/...
   ├── static_variables/
   └── preprocessing_report.md  ← 报告已生成

   【质量指标摘要】
   • uo: SSIM=0.9234, Relative L2=3.45%
   • vo: SSIM=0.9187, Relative L2=4.89%

   详细分析请查看报告文件。
   ```

---

## 输出目录结构（v3.0 新格式）

```
/output/dataset/
├── train/
│   ├── hr/
│   │   ├── uo/               # 每个变量一个目录
│   │   │   ├── 000000.npy    # 时间步0, 形状 [H, W]
│   │   │   ├── 000001.npy    # 时间步1
│   │   │   └── ...
│   │   └── vo/
│   │       └── ...
│   └── lr/
│       ├── uo/
│       └── vo/
├── valid/
│   ├── hr/
│   └── lr/
├── test/
│   ├── hr/
│   └── lr/
├── static_variables/
│   ├── 00_lon.npy
│   ├── 10_lat.npy
│   └── 90_mask.npy
├── visualisation_data_process/
│   ├── train/
│   │   ├── uo_compare.png       # HR vs LR 空间对比图
│   │   ├── uo_statistics.png    # 均值/方差时序 + 直方图
│   │   └── ...
│   ├── valid/*.png
│   ├── test/*.png
│   └── statistics_summary.png   # 全局统计汇总
├── preprocess_manifest.json
├── metrics_result.json
└── preprocessing_report.md
```

---

## ⛔ 禁止行为清单

### 变量选择阶段
| 禁止行为 | 正确做法 |
|----------|----------|
| ❌ 猜测用户要研究哪些变量 | ✅ 展示候选列表，等用户选择 |
| ❌ 使用硬编码变量名 (lon_rho, mask_rho) | ✅ 从数据检测或让用户指定 |
| ❌ 自动选择掩码变量 | ✅ 展示检测到的掩码，让用户确认 |

### 参数决策阶段
| 禁止行为 | 正确做法 |
|----------|----------|
| ❌ 自动设置 scale/train_ratio 等 | ✅ 询问用户具体数值 |
| ❌ 自动决定裁剪参数 | ✅ 展示推荐值，让用户确认 |
| ❌ 自动决定 NaN 处理方式 | ✅ 询问用户是否允许 NaN |

### 流程控制阶段
| 禁止行为 | 正确做法 |
|----------|----------|
| ❌ 跳过任何确认阶段 | ✅ 每个阶段都必须等用户确认 |
| ❌ 收到 awaiting_* 后直接继续 | ✅ 展示信息，等待用户回复 |
| ❌ 忘记之前阶段收集的参数 | ✅ 维护状态对象，累积传参 |

### 报告生成阶段
| 禁止行为 | 正确做法 |
|----------|----------|
| ❌ 不生成报告就说"完成" | ✅ 必须调用 ocean_generate_report |
| ❌ 不传 user_confirmation | ✅ 传入完整的 4 阶段确认记录 |
| ❌ 不填写分析就说"报告已生成" | ✅ 读取报告，替换占位符，写入分析 |
| ❌ 使用模板化分析 | ✅ 根据实际数据指标编写专业分析 |

---

## 参数速查表

### ocean_preprocess_full 参数分阶段

| 阶段 | 参数 | 类型 | 说明 |
|------|------|------|------|
| **阶段0** | `nc_folder` | string | NC 文件目录 |
| | `output_base` | string | 输出目录 |
| | `static_file` | string? | 静态文件（可选） |
| | `dyn_file_pattern` | string? | 文件匹配模式，默认 `*.nc` |
| **阶段1** | `dyn_vars` | string[] | 研究变量列表 |
| **阶段2** | `mask_vars` | string[] | 掩码变量列表 |
| | `stat_vars` | string[] | 静态变量列表 |
| | `lon_var` | string | 经度变量名 |
| | `lat_var` | string | 纬度变量名 |
| **阶段3** | `scale` | number | 下采样倍数 |
| | `downsample_method` | string | 插值方法 (area/cubic/linear/nearest) |
| | `train_ratio` | number | 训练集比例 (如 0.7) |
| | `valid_ratio` | number | 验证集比例 (如 0.15) |
| | `test_ratio` | number | 测试集比例 (如 0.15) |
| | `h_slice` | string? | H 方向裁剪 (如 "0:680") |
| | `w_slice` | string? | W 方向裁剪 |
| **阶段4** | `user_confirmed` | boolean | 用户最终确认 (必须为 true) |

### ocean_generate_report 参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `dataset_root` | string | ✅ | 数据集根目录 |
| `user_confirmation` | object | ✅ 推荐 | 4 阶段用户确认记录 |
| `inspect_result_path` | string | 否 | 自动查找 |
| `validate_result_path` | string | 否 | 自动查找 |
| `convert_result_path` | string | 否 | 自动查找 |
| `metrics_result_path` | string | 否 | 自动查找 |

---

