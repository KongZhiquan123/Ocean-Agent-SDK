# 工具参数参考

> 版本: 3.4.0 | 最后更新: 2026-02-07

本文档详细说明所有工具的参数定义。

---

## ocean_inspect_data - 数据分析工具

仅分析数据，不执行任何处理。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `nc_folder` | string | ✅ | NC 文件目录路径 |

**返回值**：
- `file_count`: 文件数量
- `dynamic_vars`: 动态变量列表（有时间维度）
- `static_vars`: 静态变量列表（无时间维度）
- `suspected_masks`: 疑似掩码变量
- `shape_info`: 各变量形状信息

---

## ocean_sr_preprocess_full - 完整预处理工具

执行完整的预处理流程，包含多阶段确认机制。

### 基本参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `nc_folder` | string | ✅ | HR 数据目录 |
| `output_base` | string | ✅ | 输出根目录 |
| `dyn_vars` | string[] | ✅ | 研究变量列表 |

### 确认参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `user_confirmed` | boolean | ⚠️ | 用户是否确认执行（阶段5必需） |
| `confirmation_token` | string | ⚠️ | 确认令牌（阶段2-5必需） |

### 变量配置参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `mask_vars` | string[] | ❌ | 掩码变量列表 |
| `stat_vars` | string[] | ❌ | 静态变量列表 |
| `allow_nan` | boolean | ❌ | 是否允许 NaN（默认 false） |

### 下采样参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `scale` | number | ⚠️ | 下采样倍数（下采样模式必需） |
| `downsample_method` | string | ⚠️ | 插值方法（下采样模式必需） |
| `crop_to_divisible` | boolean | ❌ | 是否裁剪以整除（默认 true） |

**插值方法选项**：
- `area`（推荐）：面积平均，保持物理量守恒
- `cubic`：三次插值，平滑但可能产生振铃
- `linear`：双线性插值
- `nearest`：最近邻，保持原始值
- `lanczos`：Lanczos 插值，高质量但较慢

### 粗网格模式参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `lr_nc_folder` | string | ⚠️ | LR 数据目录（粗网格模式必需） |

### 数据集划分参数

| 参数 | 类型 | 必需 | 默认值 |
|------|------|------|--------|
| `train_ratio` | number | ❌ | 0.7 |
| `valid_ratio` | number | ❌ | 0.15 |
| `test_ratio` | number | ❌ | 0.15 |

**注意**：三个比例之和必须等于 1.0。

### 日期文件名参数（默认开启）

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `use_date_filename` | boolean | ❌ | true | 是否使用日期作为文件名 |
| `date_format` | string | ❌ | "auto" | 日期格式选项 |
| `time_var` | string | ❌ | - | 时间变量名（默认自动检测） |

### 文件数限制参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `max_files` | number | ❌ | - | 限制处理的最大 NC 文件数量（按排序后取前 N 个） |

**使用场景**：
- 用户只想用部分数据做测试（如 `max_files: 10` 只处理前 10 个文件）
- 数据量很大时先用少量文件验证流程正确性

**日期格式选项**：
- `auto`：自动检测（推荐）
- `YYYYMMDD`：日级数据，如 `20200101.npy`
- `YYYYMMDDHH`：小时级数据，如 `2020010106.npy`
- `YYYYMMDDHHmm`：分钟级数据，如 `202001010630.npy`
- `YYYY-MM-DD`：带分隔符，如 `2020-01-01.npy`

**重复日期处理**：当同一日期有多个时间步时，自动添加时间后缀：
- `20200101_0000.npy`, `20200101_0600.npy`, `20200101_1200.npy`

**回退机制**：若时间提取失败，自动回退到序号命名（`000000.npy`）

---

## ocean_sr_preprocess_metrics - 质量指标计算

计算下采样数据的质量指标。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `dataset_root` | string | ✅ | 数据集根目录 |
| `scale` | number | ✅ | 下采样倍数 |
| `splits` | string[] | ❌ | 划分列表（默认 train/valid/test） |
| `output` | string | ❌ | 输出路径（默认自动生成） |

**质量指标说明**：
- `SSIM`: 结构相似性，0~1，越接近 1 越好
- `Relative L2`: 相对 L2 误差，越小越好
- `MSE`: 均方误差
- `RMSE`: 均方根误差

---

## ocean_sr_preprocess_report - 报告生成

生成预处理报告。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `dataset_root` | string | ✅ | 数据集根目录 |
| `scale` | number | ✅ | 下采样倍数 |
| `downsample_method` | string | ✅ | 使用的插值方法 |
| `dyn_vars` | string[] | ✅ | 研究变量列表 |
| `metrics_result` | object | ✅ | ocean_sr_preprocess_metrics 返回的结果 |
| `output_path` | string | ❌ | 报告输出路径 |

---

## 参数组合示例

### 下采样模式

```json
{
  "nc_folder": "/data/hr",
  "output_base": "/output",
  "dyn_vars": ["chl", "no3"],
  "user_confirmed": true,
  "confirmation_token": "xxx",
  "mask_vars": ["mask"],
  "stat_vars": ["lon", "lat"],
  "scale": 4,
  "downsample_method": "area",
  "train_ratio": 0.7,
  "valid_ratio": 0.15,
  "test_ratio": 0.15
}
```

### 粗网格模式

```json
{
  "nc_folder": "/data/hr",
  "output_base": "/output",
  "dyn_vars": ["temp", "salt"],
  "user_confirmed": true,
  "confirmation_token": "xxx",
  "mask_vars": ["mask_rho"],
  "stat_vars": ["lon_rho", "lat_rho"],
  "lr_nc_folder": "/data/lr",
  "train_ratio": 0.7,
  "valid_ratio": 0.15,
  "test_ratio": 0.15
}
```

### 启用日期文件名

```json
{
  "nc_folder": "/data/ocean",
  "output_base": "/output",
  "dyn_vars": ["temp", "salt"],
  "user_confirmed": true,
  "confirmation_token": "xxx",
  "scale": 4,
  "downsample_method": "area",
  "use_date_filename": true,
  "date_format": "auto"
}
```

输出文件名示例：`20200101.npy`, `20200102.npy`, ... 而非 `000000.npy`, `000001.npy`
