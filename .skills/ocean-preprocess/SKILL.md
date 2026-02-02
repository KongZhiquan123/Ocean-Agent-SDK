---
name: ocean-preprocess
description: 海洋数据预处理技能 - 专用于超分辨率场景的NC到NPY数据格式转换
version: 1.0.0
author: kzq
---

# 海洋数据预处理

专用于超分辨率场景的 NC 到 NPY 数据预处理技能。

## 使用场景

- 将 NC 格式海洋数据转换为 NPY 格式
- 超分辨率模型的数据准备
- COAWST、ROMS 等海洋模型输出数据处理

## 核心原则

**数据预处理定义：不破坏原有数据结构的任何信息，不做标准化，只做格式转换**

---

## 预处理流程 (A -> B -> C)

```
NC文件 -> [Step A] -> [Step B] -> [Step C] -> NPY文件
          查看定义    张量约定    格式存储
```

### Step A: 查看数据并定义变量

**工具**: `ocean_inspect_data`

**功能**:
- 从 NC 文件查看包含什么变量
- 自动分类：动态变量、静态变量、掩码变量
- **【重要】必须询问用户确认研究变量是什么，不能自动假设**

**参数**:
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| nc_folder | string | 是 | NC文件目录路径 |
| static_file | string | 否 | 静态NC文件路径 |
| file_filter | string | 否 | 文件名过滤关键字（默认 "avg"） |

### Step B: 进行张量约定验证

**工具**: `ocean_validate_tensor`

**功能**:
- 验证张量形状：动态变量 `[T, H, W]` 或 `[T, D, H, W]`，静态变量 `[H, W]`
- 生成 var_names 配置

**参数**:
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| inspect_result_path | string | 是 | Step A 结果文件路径 |
| research_vars | string[] | 是 | 用户确认的研究变量列表 |
| mask_vars | string[] | 否 | 掩码变量列表 |

### Step C: 转 NPY 格式存储

**工具**: `ocean_convert_npy`

**功能**:
- 按目录结构保存：`hr/变量.npy`, `static/变量.npy`
- 每个变量单独存储（不拼接通道）

**参数**:
| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| nc_folder | string | 是 | NC文件目录 |
| output_base | string | 是 | 输出基础目录 |
| research_vars | string[] | 是 | 研究变量列表 |
| static_file | string | 否 | 静态NC文件路径 |
| static_vars | string[] | 否 | 静态变量列表 |
| mask_vars | string[] | 否 | 掩码变量列表 |
| file_filter | string | 否 | 文件名过滤关键字 |

### 一键执行

**工具**: `ocean_preprocess_full`

需要用户预先指定研究变量，自动执行 A -> B -> C 完整流程。

---

## 防错规则

### 变量分类规则

| 类型 | 判断条件 | 示例 |
|------|----------|------|
| 动态变量 | 有时间维度 | u_eastward, v_northward, temp, salt |
| 静态变量 | 无时间维度，是网格属性 | h, angle |
| 掩码变量 | mask_u, mask_rho, mask_v | **【绝对不能修改】** |

### NC 文件处理规则

- **必须**使用 `sorted()` 排序确保时间顺序
- 拼接时按 `axis=0`（时间轴）

### 张量形状约定

```
动态变量: [T, H, W]      # 3D: 时间 x 高度 x 宽度
         [T, D, H, W]   # 4D: 时间 x 深度 x 高度 x 宽度

静态变量: [H, W]         # 2D: 高度 x 宽度

掩码变量: [H, W]         # 2D: 0/1 二值
```

### 事后验证规则

1. 检查目录结构是否存在（hr/, static/）
2. 检查维度是否正确
3. 如果不对，报错 "数据维度检查有问题，请检查 xxx 部分"
4. 全部通过返回 `status: "pass"`

---

## 输出目录结构

```
output_base/
├── hr/                      # 高分辨率动态数据
│   ├── u_eastward.npy      # shape: [T, H, W]
│   ├── v_northward.npy     # shape: [T, H, W]
│   └── ...
└── static/                  # 静态数据
    ├── mask_rho.npy        # shape: [H, W] - 不可变
    ├── h.npy               # shape: [H, W]
    └── ...
```

---

## 工作流程

1. 首先使用 `ocean_inspect_data` 查看数据
2. 向用户展示可用的动态变量，**询问用户确认研究变量**
3. 使用 `ocean_validate_tensor` 验证张量约定
4. 使用 `ocean_convert_npy` 进行格式转换
5. 报告最终结果

或者直接使用 `ocean_preprocess_full` 一键执行（需要用户预先指定研究变量）

---

## 快速检查清单

执行预处理时，按以下顺序检查：

- [ ] **文件排序**: 是否使用了 `sorted()`?
- [ ] **变量分类**: 动态/静态/掩码变量是否正确区分?
- [ ] **掩码保护**: mask_* 变量是否原样保存?
- [ ] **维度约定**: 动态 `[T,H,W]`，静态 `[H,W]`?
- [ ] **目录结构**: hr/ 和 static/ 目录是否正确?
- [ ] **研究变量**: 是否由用户明确指定?

---

## 使用示例

### COAWST 模型数据

```
nc_folder: /data/coawst/PRE_prognostic_results/
output_base: /data/processed/coawst/
research_vars: [u_eastward, v_northward, temp, salt]
static_file: /data/coawst/stat_file.nc
file_filter: avg
```

### ROMS 模型数据

```
nc_folder: /data/roms/output/
output_base: /data/processed/roms/
research_vars: [u, v, temp, salt, zeta]
static_file: /data/roms/grid.nc
file_filter: his
```

---

## 常见错误及解决方案

| 错误信息 | 原因 | 解决方案 |
|----------|------|----------|
| "研究变量 'xxx' 不存在" | 变量名拼写错误 | 先运行 Step A 查看可用变量 |
| "数据维度检查有问题" | shape 不符合约定 | 检查原始 NC 文件的维度 |
| "空间维度不一致" | 不同变量的 H,W 不同 | 可能需要插值对齐 |
| "掩码变量维度错误" | mask 不是 2D | 检查静态文件 |
