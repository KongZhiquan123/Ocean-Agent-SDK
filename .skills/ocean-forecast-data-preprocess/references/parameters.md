# 参数参考手册

> 版本: 1.2.0 | 最后更新: 2026-02-26

本文档列出所有工具的完整参数说明。

---

## ocean_forecast_preprocess_full

主工具，执行完整的 4 阶段预处理流程。

### 必需参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `nc_folder` | string | NC 文件所在目录（绝对路径） |
| `output_base` | string | 输出根目录（绝对路径，不存在会自动创建） |

### 阶段相关参数

| 参数 | 类型 | 必需阶段 | 说明 |
|------|------|----------|------|
| `dyn_vars` | string[] | 阶段 1 后 | 预测变量列表，如 `["sst", "ssh"]` |
| `stat_vars` | string[] | 阶段 2 后 | 静态变量列表，如 `["lon", "lat"]`，可为 `[]` |
| `mask_vars` | string[] | 阶段 2 后 | 掩码变量列表，如 `["mask"]`，可为 `[]` |
| `train_ratio` | number | 阶段 3 后 | 训练集比例，如 `0.7` |
| `valid_ratio` | number | 阶段 3 后 | 验证集比例，如 `0.1` |
| `test_ratio` | number | 阶段 3 后 | 测试集比例，如 `0.2` |
| `user_confirmed` | boolean | 阶段 4 | 必须为 `true` 才会执行 |
| `confirmation_token` | string | 阶段 4 | 由工具生成，防止跳过阶段 |

> **注意**：`train_ratio + valid_ratio + test_ratio` 必须等于 1.0（允许浮点误差 ±0.01）

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `h_slice` | string | — | H 方向空间裁剪，格式 `"start:end"`，如 `"0:512"` |
| `w_slice` | string | — | W 方向空间裁剪，格式 `"start:end"`，如 `"0:1024"` |
| `use_date_filename` | boolean | `true` | 用日期命名输出文件（`false` 则用序号） |
| `date_format` | string | `"auto"` | 文件名日期格式：`auto` / `YYYYMMDD` / `YYYYMMDDHH` / `YYYYMMDDHHmm` |
| `time_var` | string | 自动检测 | 时间变量名（自动尝试 `time`、`ocean_time`、`Time`） |
| `chunk_size` | number | `200` | 批处理文件数，控制内存使用 |
| `max_files` | number | 无限制 | 最多处理文件数（调试用，生产环境不要设置） |
| `skip_visualize` | boolean | `false` | 是否跳过 Step C 可视化步骤 |

### 返回值结构

**阶段 1 返回（awaiting_variable_selection）**：
```json
{
  "status": "awaiting_variable_selection",
  "message": "请选择预测变量",
  "step_a": {
    "nc_folder": "/data/ocean",
    "file_count": 1095,
    "dyn_candidates": ["sst", "ssh", "sss"],
    "stat_candidates": ["lon", "lat"],
    "mask_candidates": ["mask"],
    "time_info": {
      "time_var": "time",
      "start": "2020-01-01",
      "end": "2022-12-31",
      "total_steps": 1095
    },
    "spatial_shape": [720, 1440]
  }
}
```

**阶段 4 返回（awaiting_execution）**：
```json
{
  "status": "awaiting_execution",
  "message": "请确认执行",
  "params_summary": {
    "nc_folder": "/data/ocean",
    "output_base": "/output/forecast",
    "dyn_vars": ["sst", "ssh"],
    "stat_vars": ["lon", "lat"],
    "mask_vars": ["mask"],
    "train_ratio": 0.7,
    "valid_ratio": 0.1,
    "test_ratio": 0.2,
    "estimated_splits": {
      "train": 766,
      "valid": 110,
      "test": 219
    }
  },
  "confirmation_token": "abc123..."
}
```

**执行完成返回（pass）**：
```json
{
  "status": "pass",
  "message": "预处理完成",
  "step_b": {
    "splits": {
      "train": { "timestep_count": 766 },
      "valid": { "timestep_count": 110 },
      "test": { "timestep_count": 219 }
    },
    "time_info": {
      "total_steps": 1095,
      "start": "2020-01-01T00:00:00",
      "end": "2022-12-31T00:00:00"
    },
    "static_vars_saved": ["lon", "lat"],
    "output_base": "/output/forecast"
  },
  "step_c": {
    "figures": [
      "/output/forecast/visualisation_forecast/train/sst_frames.png",
      "/output/forecast/visualisation_forecast/train/sst_timeseries.png",
      "/output/forecast/visualisation_forecast/train/sst_distribution.png"
    ]
  }
}
```

---

## ocean_forecast_preprocess_visualize

生成数据质量可视化图片。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `dataset_root` | string | ✅ | 预处理输出根目录（含 train/valid/test） |
| `dyn_vars` | string[] | ✅ | 要可视化的变量列表 |
| `splits` | string[] | ❌ | 要检查的数据集，默认 `["train", "valid", "test"]` |
| `out_dir` | string | ❌ | 图片输出目录，默认 `{dataset_root}/visualisation_forecast/` |
| `n_samples` | number | ❌ | 采样帧数，默认 `4` |

**返回值**：
```json
{
  "status": "pass",
  "figures": [
    "/output/forecast/visualisation_forecast/train/sst_frames.png",
    "/output/forecast/visualisation_forecast/train/sst_timeseries.png",
    "/output/forecast/visualisation_forecast/train/sst_distribution.png"
  ]
}
```

---

## ocean_forecast_preprocess_stats

计算已预处理 NPY 数据的 per-variable 统计量，独立于主流程，预处理完成后可选调用。

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `dataset_root` | string | ✅ | — | 数据集根目录（含 train/valid/test 和 var_names.json） |
| `splits` | string[] | ❌ | `["train","valid","test"]` | 要统计的 split 列表 |
| `max_files` | number | ❌ | `200` | 每变量最大采样文件数（控制运行时长） |

**返回值**：
```json
{
  "status": "pass",
  "dataset_root": "/output/forecast_data",
  "stats": {
    "train": {
      "sst": {
        "nan_rate": 0.12,
        "min": -2.5, "max": 32.1,
        "mean": 15.3, "std": 8.2,
        "p5": 5.1, "p95": 29.8,
        "sample_files": 200
      }
    }
  },
  "time_boundary_check": {
    "passed": true,
    "splits": {
      "train": {"start": "20200101000000", "end": "20211231000000", "count": 730},
      "valid": {"start": "20220101000000", "end": "20220630000000", "count": 181},
      "test":  {"start": "20220701000000", "end": "20221231000000", "count": 184}
    },
    "boundaries": {
      "train_end_before_valid_start": {"passed": true, "train_end": "20211231000000", "valid_start": "20220101000000"},
      "valid_end_before_test_start":  {"passed": true, "valid_end": "20220630000000", "test_start": "20220701000000"}
    },
    "errors": []
  },
  "cross_split_check": {
    "warnings": [],
    "variables": {
      "sst": {
        "valid": {"mean_diff": 0.5, "mean_z_score": 0.06, "std_ratio": 1.02, "flag": "ok"},
        "test":  {"mean_diff": 1.2, "mean_z_score": 0.15, "std_ratio": 0.98, "flag": "ok"}
      }
    }
  },
  "warnings": [],
  "errors": [],
  "data_stats_path": "/output/forecast_data/data_stats.json",
  "message": "统计完成，共计算 6 个变量，结果已写入 data_stats.json"
}
```

**注意**：
- 全 NaN 变量：`min`/`max`/`mean`/`std`/`p5`/`p95` 均为 `null`，不报错
- `nan_rate > 0.3` 自动生成警告，Agent 应展示给用户
- `time_boundary_check.passed = false` 表示存在时间泄露风险，**必须告知用户**
- `cross_split_check` 中 `flag` 为 `warn_mean`/`warn_std`/`warn_both` 时需展示给用户确认

---

## ocean_forecast_preprocess_report

生成预处理报告 Markdown 文件。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `output_base` | string | ✅ | 预处理输出根目录 |
| `ai_analysis` | string | ❌ | Agent 对数据质量的分析评论 |

**报告内容**：
- 数据集概览（文件数、时间范围、变量列表）
- 训练/验证/测试集划分统计
- 可视化图片路径
- AI 分析评论（由 Agent 撰写）

---

## ocean_inspect_data

查看 NC 文件目录的变量结构，不执行任何处理。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `nc_folder` | string | ✅ | NC 文件目录路径（非单文件，传目录）

**返回值**：
```json
{
  "variables": {
    "sst": { "shape": [1, 720, 1440], "dtype": "float32" },
    "time": { "shape": [1], "dtype": "float64" }
  },
  "dimensions": { "time": 1, "lat": 720, "lon": 1440 }
}
```

---

## 参数使用示例

### 最简调用（触发阶段 1）

```json
{
  "nc_folder": "/data/ocean/nc_files",
  "output_base": "/output/forecast_data"
}
```

### 阶段 2 → 3 调用（已有 dyn_vars）

```json
{
  "nc_folder": "/data/ocean/nc_files",
  "output_base": "/output/forecast_data",
  "dyn_vars": ["sst", "ssh"]
}
```

### 阶段 4 确认执行

```json
{
  "nc_folder": "/data/ocean/nc_files",
  "output_base": "/output/forecast_data",
  "dyn_vars": ["sst", "ssh"],
  "stat_vars": ["lon", "lat"],
  "mask_vars": ["mask"],
  "train_ratio": 0.7,
  "valid_ratio": 0.1,
  "test_ratio": 0.2,
  "user_confirmed": true,
  "confirmation_token": "abc123..."
}
```
