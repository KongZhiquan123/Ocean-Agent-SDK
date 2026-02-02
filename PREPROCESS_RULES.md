# 超分辨率数据预处理 - Agent防错规则参考

## 概述

数据预处理定义：**不破坏原有数据结构的任何信息，不做标准化，只做格式转换**

```
NC文件 → [Step A] → [Step B] → [Step C] → NPY文件
         查看定义    张量约定    格式存储
```

---

## Step A: 查看数据并定义变量

### 函数
```python
step_a_inspect_and_define(nc_folder_path, static_file_path, file_filter, config)
```

### 防错规则

| 规则ID | 规则内容 | 检查方法 |
|--------|----------|----------|
| **A1** | 区分动态/静态/无关变量 | 检查变量是否有时间维度 |
| **A2** | 陆地掩码变量不可变 | `mask_u`, `mask_rho`, `mask_v` 必须原样保留 |
| **A3** | NC文件必须排序 | 使用 `sorted()` 函数 |

### 变量分类规则

```python
# 动态变量：有时间维度
dims = ['time', 'eta_rho', 'xi_rho']  # → DYNAMIC

# 静态变量：无时间维度，但是网格属性
dims = ['eta_rho', 'xi_rho']  # → STATIC (如果在 static_vars 列表中)

# 掩码变量：特殊静态变量，绝对不能修改
MASK_VARS = ['mask_u', 'mask_rho', 'mask_v']

# 无关变量：坐标变量、元数据等
# → IGNORED
```

### 强制询问点
```
【重要】用户必须确认研究变量是什么？
- 不能自动假设
- 必须等待用户明确指定
```

---

## Step B: 进行张量约定

### 函数
```python
step_b_validate_tensor_convention(variables_info, research_vars, config)
```

### 防错规则

| 规则ID | 规则内容 | 检查方法 |
|--------|----------|----------|
| **B1** | 动态变量必须有时间维度 | ndim == 3 或 4 |
| **B2** | 静态变量不能有时间维度 | ndim == 2 |
| **B3** | 研究变量必须存在 | 检查是否在 variables_info 中 |
| **B4** | 掩码验证 | 检查掩码形状与数据空间维匹配 |

### 张量形状约定

```python
# 动态变量
[T, H, W]           # 3D: 时间 × 高度 × 宽度
[T, D, H, W]        # 4D: 时间 × 深度 × 高度 × 宽度

# 静态变量
[H, W]              # 2D: 高度 × 宽度

# 掩码变量
[H, W]              # 2D: 0/1 二值
```

### var_names 配置输出
```python
var_names_config = {
    "dynamic": ["u_eastward", "v_northward", "temp", ...],
    "static": ["mask_rho", "h", "angle", ...],
    "research": ["u_eastward", "v_northward"],  # 用户指定
    "mask": ["mask_u", "mask_rho", "mask_v"]    # 不可变
}
```

---

## Step C: 转NPY格式存储

### 函数
```python
step_c_convert_to_npy(nc_folder_path, output_base_dir, research_vars, static_file_path, config, file_filter)
```

### 防错规则

| 规则ID | 规则内容 | 检查方法 |
|--------|----------|----------|
| **C1** | 目录结构符合 OceanSRDataset | 检查 hr/ 和 static/ 目录 |
| **C2** | 维度检查 | 保存前验证 shape |
| **C3** | 掩码变量原样保存 | 保存后对比验证 |
| **C4** | NC文件排序后处理 | 使用 `sorted()` |

### 目录结构

```
output_base/
├── hr/                      # 高分辨率动态数据
│   ├── u_eastward.npy      # shape: [T, H, W]
│   ├── v_northward.npy     # shape: [T, H, W]
│   └── ...
├── static/                  # 静态数据
│   ├── mask_rho.npy        # shape: [H, W]
│   ├── h.npy               # shape: [H, W]
│   └── ...
└── plots/                   # 分布图（可选）
    └── variable_distribution.png
```

---

## 事后防错规则

### 事后规则 1: 目录结构检查

```python
# 检查清单
assert os.path.exists(output_base + "/hr/")
for var in research_vars:
    assert os.path.exists(f"{output_base}/hr/{var}.npy")
```

### 事后规则 2: 维度检查

```python
# 检查清单
for var in research_vars:
    data = np.load(f"{output_base}/hr/{var}.npy")
    if data.ndim not in [3, 4]:
        raise ValueError(f"数据维度检查有问题，请检查 {var} 部分")
```

### 全部通过
```
如果所有防错规则都通过 → 返回 status: "pass"
```

---

## 快速检查清单

Agent修改代码时，按以下顺序检查：

1. [ ] **文件排序**: 是否使用了 `sorted()`?
2. [ ] **变量分类**: 动态/静态/掩码变量是否正确区分?
3. [ ] **掩码保护**: mask_* 变量是否原样保存?
4. [ ] **维度约定**: 动态 [T,H,W]，静态 [H,W]?
5. [ ] **目录结构**: hr/ 和 static/ 目录是否正确?
6. [ ] **研究变量**: 是否由用户明确指定?

---

## 常见错误及解决方案

| 错误信息 | 原因 | 解决方案 |
|----------|------|----------|
| "研究变量 'xxx' 不存在" | 变量名拼写错误 | 先运行 Step A 查看可用变量 |
| "数据维度检查有问题" | shape 不符合约定 | 检查原始 NC 文件的维度 |
| "空间维度不一致" | 不同变量的 H,W 不同 | 可能需要插值对齐 |
| "掩码变量维度错误" | mask 不是 2D | 检查静态文件 |

---

## 使用示例

```python
from preprocess_tools import run_full_preprocessing

# 一键执行
result = run_full_preprocessing(
    nc_folder_path='/path/to/nc/files/',
    output_base_dir='/path/to/output/',
    research_vars=['u_eastward', 'v_northward'],  # 必须指定!
    static_file_path='/path/to/static.nc',
    file_filter="avg"
)

if result["overall_status"] == "pass":
    print("预处理成功!")
```

---

## 命令行使用

```bash
python preprocess_tools.py \
    --nc-folder /path/to/nc/files/ \
    --output /path/to/output/ \
    --static-file /path/to/static.nc \
    --research-vars u_eastward v_northward \
    --filter avg
```
