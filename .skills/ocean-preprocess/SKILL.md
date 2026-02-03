---
name: ocean-preprocess
description: 海洋数据预处理技能 - 专用于超分辨率场景的NC到NPY数据格式转换
version: 2.5.0
author: kzq
contributors: leizheng
last_modified: 2026-02-03
---

<!--
Changelog:
  - 2026-02-03 leizheng: v2.5.0
    - ocean_preprocess_full 集成下采样和可视化
    - 完整流程变为 A→B→C→D→E 五步
    - scale 参数变为必须
  - 2026-02-03 leizheng: v2.4.0
    - 新增裁剪功能（h_slice/w_slice/scale 参数）
    - 新增下采样工具 ocean_downsample
    - 新增可视化工具 ocean_visualize
    - 新增指标检测工具 ocean_metrics
    - 更新完整流程文档
  - 2026-02-03 leizheng: v2.3.0
    - 支持 nc_files 参数明确指定文件列表
    - 支持单个文件路径自动转换为目录模式
    - 逐文件检测时间维度，识别静态文件混入
    - 更新禁止行为清单
  - 2026-02-03 leizheng: v2.2.1
    - 添加强制确认机制（必须同时提供 mask_vars 和 stat_vars）
  - 2026-02-03 leizheng: v2.2.0
    - 添加禁止自动决策原则
    - 添加禁止行为清单
    - 移除硬编码默认值要求
  - 2026-02-02 leizheng: v2.1.0
    - 添加警告优先原则
    - 添加警告处理指南
  - 2026-02-02 leizheng: v2.0.0
    - 完整重写 SKILL.md
    - 添加工具调用流程
    - 添加错误解释指南
    - 添加参数速查表
-->

# 海洋数据预处理技能

## 核心原则

1. **数据预处理定义**：不破坏原有数据结构的任何信息，不做标准化，只做格式转换

2. **⚠️ 警告优先原则（重要）**：
   - 在分析过程中如果有**任何不确定或异常**的地方，**必须先询问用户**再继续执行
   - **绝不能**在有严重警告的情况下直接执行处理
   - 即使工具返回 `pass` 状态，如果 `warnings` 中有内容，也要向用户展示并确认

3. **⚠️ 禁止自动决策原则（v2.2 新增）**：
   - **不得代替用户做任何数据处理决策**
   - 以下事项必须由用户明确确认：
     - NaN/Inf 值如何处理（是否允许、如何填充）
     - 掩码变量选择（哪些是掩码、哪个是主掩码）
     - 静态变量选择（哪些需要保存）
     - 坐标范围验证（是否需要、范围是多少）
   - 禁止使用硬编码默认值（如 "lon_rho", "mask_rho"）
   - 所有变量名必须从数据中检测到或由用户明确指定

4. **⚠️ 路径灵活处理原则（v2.3 新增）**：
   - nc_folder 可以是目录路径，也可以是单个文件路径
   - 如果提供单个文件路径，自动转换为目录 + 文件列表模式
   - 支持 nc_files 参数明确指定要处理的文件
   - **检测静态文件混入**：自动识别目录中没有时间维度的文件

5. **需要暂停并询问用户的情况**：
   - 形状不匹配（动态数据与静态数据维度不同）
   - 缺少掩码变量
   - NaN/Inf 值存在（即使 allow_nan=true）
   - 掩码非二值
   - 坐标范围异常
   - 任何 warnings 数组中的内容
   - 用户未明确指定 mask_vars 或 stat_vars 时
   - **检测到静态文件混入动态数据目录时**

---

## 可用工具

| 工具名 | 用途 | 什么时候用 |
|--------|------|-----------|
| `ocean_preprocess_full` | 一键执行完整流程 A→B→C→D→E（含下采样+可视化） | **推荐**，信息完整时直接用这个 |
| `ocean_downsample` | HR→LR 下采样 | 单独执行下采样（full 已集成） |
| `ocean_visualize` | HR vs LR 可视化对比 | 单独生成可视化（full 已集成） |
| `ocean_metrics` | 质量指标检测 | 计算 SSIM、Relative L2 等指标 |
| `ocean_inspect_data` | 只查看数据，不处理 | 用户只想看看有什么变量时 |
| `ocean_validate_tensor` | 只验证张量形状 | 一般不单独用 |
| `ocean_convert_npy` | 只执行转换 | 一般不单独用 |

---

## 工具调用流程（重要）

### 第一步：收集必需信息

在调用任何工具前，你必须确保用户提供了以下信息：

| 信息 | 对应参数 | 必需 | 如果缺失，追问示例 |
|------|----------|------|-------------------|
| 数据目录 | `nc_folder` | ✅ 是 | "请提供 NC 数据文件所在的目录路径" |
| 研究变量 | `dyn_vars` | ✅ 是 | "请指定您要研究的动态变量（如 uo, vo）" |
| 输出目录 | `output_base` | ✅ 是 | "请指定处理结果的输出目录" |
| 静态文件 | `static_file` | 否 | 可选，有则更完整 |
| 文件匹配模式 | `dyn_file_pattern` | 否 | 默认 `*.nc` |

**追问规则**：缺少必需信息时，先追问，不要猜测。

---

### 第二步：调用 ocean_preprocess_full

信息完整后，调用工具：

```json
{
  "nc_folder": "/用户提供的数据目录",
  "output_base": "/用户提供的输出目录",
  "dyn_vars": ["uo", "vo"]
}
```

**⚠️ 重要**：第一次调用时**不要**提供 `mask_vars` 和 `stat_vars`！
- 工具会分析数据并返回 `awaiting_confirmation` 状态
- 然后你必须向用户展示检测到的变量，等待用户确认
- 只有用户确认后，第二次调用时才能提供这些参数

---

### 第三步：处理工具返回结果

#### 情况 A：返回 `overall_status: "awaiting_confirmation"`

**含义**：工具检测到了疑似掩码变量或坐标变量，需要用户确认。

**返回结构**：
```json
{
  "overall_status": "awaiting_confirmation",
  "message": "数据分析完成，请用户确认变量分类...",
  "step_a": {
    "suspected_masks": ["mask_rho", "mask_u", "mask_v"],
    "suspected_coordinates": ["lon_rho", "lat_rho", "h", "angle"],
    "dynamic_vars_candidates": ["uo", "vo", "temp", "salt"]
  }
}
```

**你必须做的事**：

1. 向用户展示检测到的变量分类：
```
我已分析数据，检测到以下变量：

【动态变量候选】（可作为研究目标）
- uo, vo, temp, salt

【疑似掩码变量】
- mask_rho, mask_u, mask_v

【疑似坐标/静态变量】
- lon_rho, lat_rho, h, angle

请逐一确认：
1. 您指定的研究变量 uo, vo 是否正确？
2. 掩码变量应该使用哪些？（检测到: mask_rho, mask_u, mask_v）
3. 需要保存哪些静态变量？（检测到: lon_rho, lat_rho, h, angle）
4. NaN/Inf 值如何处理？
   - 数据中是否可能有 NaN？
   - 如果有，是否允许保留？
```

2. **必须等待用户逐一确认每个问题**，不得自动决定

3. **用户确认后，再次调用 ocean_preprocess_full，必须同时提供所有参数**：

```json
{
  "nc_folder": "/data/ocean",
  "output_base": "/output/processed",
  "dyn_vars": ["uo", "vo"],
  "static_file": "/data/ocean/grid.nc",
  "mask_vars": ["mask_rho", "mask_u", "mask_v"],
  "stat_vars": ["lon_rho", "lat_rho", "h", "angle", "mask_rho"],
  "lon_var": "lon_rho",
  "lat_var": "lat_rho",
  "allow_nan": false
}
```

**重要**：
- 必须**同时**提供 `mask_vars` 和 `stat_vars`，否则工具会返回错误
- `lon_var` 和 `lat_var` 必须是用户确认的或从数据中检测到的
- **禁止使用硬编码默认值**

---

#### 情况 B：返回 `overall_status: "pass"`

**含义**：处理成功完成。

**⚠️ 重要：检查 warnings**

即使状态是 `pass`，你也必须检查返回结果中的 `warnings` 数组。如果有警告，**必须向用户展示并确认是否接受**：

```json
{
  "overall_status": "pass",
  "step_c": {
    "warnings": ["形状不匹配: 动态数据 (2041, 4320) 与静态数据 (100, 200)"],
    "errors": []
  }
}
```

**有警告时你必须做的**：

1. **不要**直接说"处理完成"
2. **必须**向用户展示所有警告
3. **必须**询问用户是否接受这些警告继续
4. 等用户确认后才能报告处理完成

示例回复：
```
处理已完成，但检测到以下警告：

⚠️ 警告信息：
1. 形状不匹配: 动态数据 (2041×4320) 与静态数据 (100×200) 形状不同
2. 缺少掩码: 未找到 mask_u 掩码变量

这些警告可能影响后续使用。请确认：
- 形状不匹配是否是预期的？（动态和静态数据来自不同分辨率？）
- 缺少 mask_u 是否会影响您的分析？

请回复"确认接受"继续，或告诉我需要如何处理。
```

**无警告时**：
```
预处理完成！

输出目录结构：
/output/processed/
├── target_variables/
│   ├── uo.npy
│   └── vo.npy
└── static_variables/
    ├── 00_lon_rho.npy
    ├── 10_lat_rho.npy
    └── 90_mask_rho.npy

所有验证检查已通过，无警告。
```

---

#### 情况 C：返回 `overall_status: "error"`

**含义**：处理过程中出错。

**返回结构**：
```json
{
  "overall_status": "error",
  "message": "Step C 失败",
  "step_c": {
    "errors": ["动态变量 'uo' 含有非法值: NaN=1234, Inf=0"],
    "warnings": [...]
  }
}
```

**你必须做的**：从 `errors` 数组中提取错误信息，向用户清晰解释。

---

## 错误解释指南

当工具返回错误时，按以下方式向用户解释：

### 路径/文件错误（最常见）

| 错误关键词 | 向用户解释 |
|-----------|-----------|
| `没有找到任何动态变量` | "您提供的数据文件中没有找到带时间维度的变量。**这很可能是因为您把静态文件路径填到了动态数据目录**。动态数据文件应该包含如 uo, vo, temp 等随时间变化的变量。" |
| `研究变量不在动态变量候选列表中` | "您指定的研究变量在数据文件的动态变量中不存在。请检查变量名是否拼写正确，或者查看工具返回的可用动态变量列表。" |
| `未找到匹配的动态数据文件` | "在指定目录下没有找到 NC 文件。请检查：1) 目录路径是否正确；2) 文件匹配模式是否匹配您的文件名。" |

### 数据质量错误

| 错误关键词 | 向用户解释 |
|-----------|-----------|
| `含有非法值: NaN=xxx` | "检测到变量中存在 NaN（缺失值），共 xxx 个。这可能是数据源问题。如果这是预期的（如陆地区域填充），可以设置 allow_nan=true 跳过检查。" |
| `含有非法值: Inf=xxx` | "检测到变量中存在 Inf（无穷大），共 xxx 个。这通常是数值计算溢出导致，请检查数据源。" |
| `坐标变量 'xxx' 包含 NaN` | "坐标变量中存在 NaN，这是严重错误。坐标不允许有缺失值，请检查静态文件是否损坏。" |

### 维度错误

| 错误关键词 | 向用户解释 |
|-----------|-----------|
| `有零长度维度` | "变量的某个维度长度为 0，数据是空的。请检查文件是否完整，或文件匹配模式是否正确。" |
| `维度数量错误: 实际 2D` | "该变量是 2D，但动态变量应该是 3D [时间,高度,宽度] 或 4D [时间,深度,高度,宽度]。请确认这是否是动态变量。" |
| `维度数量错误: 实际 5D` | "该变量是 5D，超出支持范围（最多 4D）。可能需要先降维处理。" |

### 静态文件错误

| 错误关键词 | 向用户解释 |
|-----------|-----------|
| `静态文件不存在` | "指定的静态文件路径不存在，请检查路径是否正确。" |
| `掩码形状不匹配` | "掩码变量的形状与动态数据的空间维度不一致，无法正确应用掩码。请检查静态文件是否与动态数据匹配。" |

---

## ⚠️ 警告处理指南（必读）

**核心原则**：有警告时必须暂停并询问用户，不能直接继续。

### 需要暂停并询问的警告类型

| 警告类型 | 警告关键词 | 必须询问用户 |
|----------|-----------|-------------|
| 形状不匹配 | `形状不匹配`、`shape mismatch` | ✅ 是 |
| 缺少掩码 | `未找到掩码`、`缺少 mask` | ✅ 是 |
| NaN 值存在 | `含有 NaN`、`NaN=` | ✅ 是（即使 allow_nan=true） |
| 掩码非二值 | `掩码不是二值`、`not binary` | ✅ 是 |
| 坐标范围异常 | `超出范围`、`out of range` | ✅ 是 |
| 时间不单调 | `时间不单调`、`not monotonic` | ✅ 是 |
| 启发式验证失败 | `陆地零值比例`、`海洋零值比例` | ✅ 是 |

### 如何向用户展示警告

```
⚠️ 处理过程中发现以下警告：

1. **形状不匹配**: 动态数据 (2041×4320) 与静态数据 (100×200) 形状不同
   - 这意味着动态数据和静态数据来自不同分辨率的网格
   - 可能导致掩码无法正确对应到数据点

2. **缺少掩码**: 未找到 mask_u 掩码变量
   - u 方向的流速数据将没有陆地掩码保护
   - 可能影响后续的超分辨率训练

这些警告需要您确认是否继续：
- 如果这是预期的情况，请回复"确认继续"
- 如果需要更正，请告诉我正确的文件路径或配置
```

### 用户可能的回复及处理

| 用户回复 | 你应该做的 |
|----------|-----------|
| "确认继续"/"没关系"/"可以" | 记录用户已确认，报告最终结果 |
| "不对，应该是..."/"重新处理" | 根据用户新提供的信息重新调用工具 |
| "这是什么意思？" | 详细解释该警告的含义和可能影响 |

---

## 单独使用 ocean_inspect_data

当用户只想查看数据有什么变量时：

```json
{
  "nc_folder": "/data/ocean",
  "static_file": "/data/ocean/grid.nc"
}
```

**返回**：
```json
{
  "status": "success",
  "file_count": 365,
  "dynamic_vars_candidates": ["uo", "vo", "temp", "salt"],
  "suspected_masks": ["mask_rho", "mask_u"],
  "suspected_coordinates": ["lon_rho", "lat_rho", "h"]
}
```

**你应该做的**：向用户展示变量列表和分类建议。

---

## 常见用户问法处理

| 用户说 | 缺什么 | 你的回应 |
|--------|--------|---------|
| "帮我预处理海洋数据" | 全缺 | 追问数据目录、研究变量、输出目录 |
| "处理 /data/ocean 目录" | 变量、输出 | 追问研究变量、输出目录 |
| "预处理 uo vo 变量" | 目录、输出 | 追问数据目录、输出目录 |
| "把 /data 的 uo vo 输出到 /out" | 无 | 信息完整，调用工具 |
| "看看 /data/ocean 有什么变量" | - | 调用 ocean_inspect_data |

---

## 参数速查表

### ocean_preprocess_full

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| nc_folder | string | ✅ | - | NC 文件目录（也可以是单个文件路径，会自动转换） |
| nc_files | string[] | 否 | - | 明确指定要处理的文件列表（支持通配符如 `ocean_*.nc`） |
| output_base | string | ✅ | - | 输出目录 |
| dyn_vars | string[] | ✅ | - | 研究变量 |
| static_file | string | 否 | - | 静态文件路径 |
| dyn_file_pattern | string | 否 | "*.nc" | 文件匹配模式（当 nc_files 未指定时使用） |
| mask_vars | string[] | ⚠️ 推荐 | 从数据检测 | 掩码变量（必须用户确认） |
| stat_vars | string[] | ⚠️ 推荐 | 从数据检测 | 静态变量（必须用户确认） |
| lon_var | string | ⚠️ 推荐 | 从数据检测 | 经度变量名（禁止硬编码默认值） |
| lat_var | string | ⚠️ 推荐 | 从数据检测 | 纬度变量名（禁止硬编码默认值） |
| allow_nan | boolean | 否 | false | 允许 NaN/Inf（必须用户确认） |
| lon_range | [min,max] | 否 | - | 经度范围验证 |
| lat_range | [min,max] | 否 | - | 纬度范围验证 |

**⚠️ 注意**：标注为"推荐"的参数，如果未从数据中检测到，必须由用户明确提供。禁止使用硬编码默认值！

### nc_files 参数使用示例

```json
// 处理目录中所有 NC 文件
{ "nc_folder": "/data/ocean" }

// 只处理特定文件
{ "nc_folder": "/data/ocean", "nc_files": ["ocean_avg_001.nc", "ocean_avg_002.nc"] }

// 使用通配符
{ "nc_folder": "/data/ocean", "nc_files": ["ocean_avg_*.nc"] }

// 排除静态文件（只处理动态文件）
{ "nc_folder": "/data/ocean", "nc_files": ["ocean_avg_*.nc"], "static_file": "/data/ocean/grid.nc" }
```

### ocean_inspect_data

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| nc_folder | string | ✅ | - | NC 文件目录 |
| static_file | string | 否 | - | 静态文件路径 |
| dyn_file_pattern | string | 否 | "*.nc" | 文件匹配模式 |

---

## 检查清单

执行预处理前确认：
- [ ] 用户提供了数据**目录**路径（不是文件路径！）
- [ ] 用户明确指定了研究变量（不要猜测）
- [ ] 用户提供了输出目录路径
- [ ] 如果有静态文件，已获取路径

收到 awaiting_confirmation 后确认：
- [ ] 已向用户展示疑似变量
- [ ] 用户已确认掩码变量和静态变量
- [ ] 用户已确认 NaN 处理方式
- [ ] 第二次调用时**同时**提供了所有必要参数

---

## ⛔ 禁止行为清单（v2.3 更新）

以下行为是**严格禁止**的：

| 禁止行为 | 原因 | 正确做法 |
|----------|------|----------|
| 使用硬编码变量名（如 lon_rho, mask_rho） | 不同数据集变量名不同 | 从数据检测或让用户指定 |
| 自动决定 NaN 处理方式 | 用户对数据质量有不同要求 | 询问用户是否允许 NaN |
| 自动选择主掩码变量 | 不同场景需要不同掩码 | 让用户确认或告知选择 |
| 在有警告时继续处理 | 可能产生错误结果 | 展示警告并等待确认 |
| 自动推导 mask_u/mask_v | 推导方式可能不适合所有情况 | 让用户明确指定 |
| **收到 awaiting_confirmation 后直接继续处理** | 用户尚未确认变量配置 | **必须等待用户逐一确认** |
| **猜测用户要研究的变量** | 用户意图不明确 | 询问用户要研究哪些动态变量 |
| **第一次调用就提供 mask_vars/stat_vars** | 跳过了用户确认流程 | 第一次不提供，等用户确认后第二次再提供 |
| **检测到静态文件混入时不告知用户** | 可能导致处理错误 | 展示混入的文件列表，询问如何处理 |

---

## ⚠️ 强制确认机制（v2.2.1 新增）

**工具行为**：
- 如果调用 `ocean_preprocess_full` 时**没有同时提供** `mask_vars` 和 `stat_vars`
- 工具会**强制返回** `awaiting_confirmation` 状态
- 即使数据分析完全成功，也不会继续处理

**Agent 必须做的事**：
1. 第一次调用只提供 `nc_folder`, `output_base`, `dyn_vars`
2. 收到 `awaiting_confirmation` 后，向用户展示：
   - 检测到的动态变量候选
   - 检测到的疑似掩码变量
   - 检测到的疑似坐标变量
3. **逐一询问用户**：
   - "您要研究的变量是 xxx 吗？"
   - "掩码变量使用 xxx 吗？"
   - "需要保存哪些静态变量？"
   - "是否允许数据中有 NaN 值？"
   - "数据集划分比例是多少？（train/valid/test）"
   - "需要裁剪数据吗？如果需要，请指定 h_slice 和 w_slice"
4. **等待用户逐一确认**
5. 用户确认后，第二次调用时提供所有确认的参数

---

## 完整超分预处理流程（v2.5 更新）

### 流程概览

`ocean_preprocess_full` 现在已集成完整的 5 步流程：

```
NC 文件 → [Step A: 数据检查]
              ↓
         [Step B: 张量验证]
              ↓
         [Step C: 转换+裁剪+划分] → HR 数据
              ↓
         [Step D: 下采样] → LR 数据
              ↓
         [Step E: 可视化检查] → 对比图
```

### 一键执行完整流程

使用 `ocean_preprocess_full` 工具，提供必要参数后自动完成所有步骤：

```json
{
  "nc_folder": "/data/ocean",
  "output_base": "/output/dataset",
  "dyn_vars": ["chl", "no3"],
  "user_confirmed": true,
  "train_ratio": 0.7,
  "valid_ratio": 0.15,
  "test_ratio": 0.15,
  "h_slice": "0:680",
  "w_slice": "0:1440",
  "scale": 4,
  "downsample_method": "area"
}
```

**输出目录结构**：
```
/output/dataset/
├── train/
│   ├── hr/
│   │   ├── chl.npy
│   │   └── no3.npy
│   └── lr/
│       ├── chl.npy
│       └── no3.npy
├── valid/
│   ├── hr/
│   └── lr/
├── test/
│   ├── hr/
│   └── lr/
├── static_variables/
└── visualisation_data_process/
    ├── train/*.png
    ├── valid/*.png
    └── test/*.png
```

### 新增参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| scale | number | ✅ 是 | - | 下采样倍数（必须由用户指定） |
| downsample_method | string | ✅ 是 | - | 下采样插值方法：area/cubic/linear/nearest/lanczos |
| skip_downsample | boolean | 否 | false | 跳过下采样步骤 |
| skip_visualize | boolean | 否 | false | 跳过可视化步骤 |

**插值方法说明**：
- `area`（推荐）：区域平均，最接近真实低分辨率采样
- `cubic`：三次插值，较平滑
- `linear`：双线性插值
- `nearest`：最近邻插值，保留原始值
- `lanczos`：Lanczos 插值，高质量但计算较慢

### 单独执行各步骤（可选）

如果需要单独控制某个步骤，可以使用独立工具：

#### Step 1: NC → NPY 转换（含裁剪和划分）

使用 `ocean_downsample` 工具：

```json
{
  "dataset_root": "/output/dataset",
  "scale": 4,
  "method": "area"
}
```

**参数说明**：
- `scale`: 下采样倍数（如 4 表示尺寸缩小为 1/4）
- `method`: 插值方法
  - `area`（推荐）：区域平均，最接近真实低分辨率
  - `bicubic`：双三次插值
  - `nearest`：最近邻插值
  - `linear`：双线性插值
  - `lanczos`：Lanczos 插值

**输出**：
```
/output/dataset/
├── train/
│   ├── hr/  (已有)
│   └── lr/  ← 新生成
│       ├── chl.npy
│       └── no3.npy
├── valid/
│   ├── hr/
│   └── lr/  ← 新生成
└── test/
    ├── hr/
    └── lr/  ← 新生成
```

### Step 3: 可视化检查

使用 `ocean_visualize` 工具：

```json
{
  "dataset_root": "/output/dataset"
}
```

**输出**：
```
/output/dataset/
└── visualisation_data_process/
    ├── train/
    │   ├── chl.png
    │   └── no3.png
    ├── valid/
    └── test/
```

### Step 4: 质量指标检测

使用 `ocean_metrics` 工具：

```json
{
  "dataset_root": "/output/dataset",
  "scale": 4
}
```

**输出**：
```
/output/dataset/
└── metrics_result.json
```

**指标说明**：
- `SSIM`: 结构相似性 (0~1, 越接近 1 越好)
- `Relative L2`: 相对 L2 误差 (越小越好, HR 作为分母)
- `MSE`: 均方误差
- `RMSE`: 均方根误差

---

## 新工具参数速查表

### ocean_downsample

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| dataset_root | string | ✅ | - | 数据集根目录 |
| scale | number | ✅ | - | 下采样倍数 |
| method | string | 否 | "area" | 插值方法 |
| splits | string[] | 否 | ["train","valid","test"] | 要处理的划分 |
| include_static | boolean | 否 | false | 是否处理静态变量 |

### ocean_visualize

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| dataset_root | string | ✅ | - | 数据集根目录 |
| splits | string[] | 否 | ["train","valid","test"] | 要检查的划分 |
| out_dir | string | 否 | dataset_root/visualisation_data_process | 输出目录 |

### ocean_metrics

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| dataset_root | string | ✅ | - | 数据集根目录 |
| scale | number | ✅ | - | 下采样倍数 |
| splits | string[] | 否 | ["train","valid","test"] | 要检查的划分 |
| output | string | 否 | dataset_root/metrics_result.json | 输出文件路径 |

---

## 裁剪参数说明（v2.4 新增）

### 为什么需要裁剪？

超分辨率训练要求 HR 尺寸能被 scale 整除：
```
HR: (680, 1440)  ← 能被 4 整除
LR: (170, 360)   ← 680÷4=170, 1440÷4=360

❌ HR: (681, 1440)  ← 681 不能被 4 整除
```

### 裁剪参数格式

| 格式 | 含义 | 示例 |
|------|------|------|
| `"0:680"` | 取 [0, 680) | `data[..., 0:680, :]` |
| `":680"` | 取 [0, 680) | 同上 |
| `"1:"` | 从 1 开始到末尾 | `data[..., 1:, :]` |
| `"1:-1"` | 去掉首尾各 1 行 | `data[..., 1:-1, :]` |

### 裁剪验证

如果提供了 `scale` 参数，工具会自动验证裁剪后的尺寸能否被整除：
- 能整除 → 继续处理
- 不能整除 → 报错并提示建议值

