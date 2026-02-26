# 错误处理指南

> 版本: 1.1.0 | 最后更新: 2026-02-25

本文档说明常见错误及处理方法。

---

## 错误类型一览

| 错误代码 | 含义 | 处理方式 |
|----------|------|----------|
| `path_not_found` | 路径不存在 | 提示用户检查路径 |
| `no_nc_files` | 目录下无 NC 文件 | 提示用户确认目录 |
| `token_invalid` | 确认令牌无效 | 重新从阶段 1 开始 |
| `token_missing` | 缺少确认令牌 | 补充 confirmation_token |
| `variable_not_found` | 变量不存在 | 展示可用变量让用户选择 |
| `ratio_invalid` | 划分比例之和不为 1 | 提示用户重新输入比例 |
| `time_var_not_found` | 无法检测时间变量 | 让用户手动指定 time_var |
| `permission_denied` | 权限不足 | 提示用户检查目录权限 |
| `python_error` | Python 脚本执行失败 | 展示详细错误，等待指示 |

---

## 路径相关错误

### path_not_found

**错误信息**：`数据目录不存在: /path/to/data`

**原因**：用户提供的路径不存在或拼写错误。

**处理**：
```
您提供的路径不存在：/path/to/data
请检查路径是否正确，或提供新的路径。
```

### no_nc_files

**错误信息**：`目录下没有找到 NC 文件`

**原因**：目录存在但没有 .nc 文件。

**处理**：
```
在目录 /path/to/data 下没有找到 NC 文件。
请确认：
1. 文件扩展名是否为 .nc
2. 是否需要指定子目录
```

---

## Token 相关错误

### token_invalid

**错误信息**：`确认令牌无效或已过期`

**原因**：
1. Token 已过期（超过 30 分钟）
2. 参数已变更导致 Token 失效
3. Token 被篡改

**处理**：
```
确认令牌已失效，需要重新开始流程。
我将重新分析您的数据...

[重新调用 ocean_forecast_preprocess_full，不带 confirmation_token]
```

### token_missing

**错误信息**：`阶段 4 需要 confirmation_token`

**原因**：调用执行阶段时未提供 Token。

**处理**：从上一次工具返回中获取 `confirmation_token` 并传入。

---

## 变量相关错误

### variable_not_found

**错误信息**：`变量 xxx 在数据中不存在`

**原因**：用户指定的变量名在 NC 文件中找不到。

**处理**：
```
变量 "xxx" 在数据中不存在。

可用的动态变量：
- sst: sea surface temperature
- ssh: sea surface height
- sss: sea surface salinity

请从以上列表中选择。
```

---

## 参数相关错误

### ratio_invalid

**错误信息**：`划分比例之和不等于 1.0: 0.7 + 0.1 + 0.15 = 0.95`

**原因**：三个比例之和不等于 1.0（允许误差 ±0.01）。

**处理**：
```
划分比例之和不等于 1：
  train_ratio=0.7, valid_ratio=0.1, test_ratio=0.15
  合计：0.95（差值：0.05）

请重新提供三个比例，确保它们的和 = 1。
例如：0.7 / 0.1 / 0.2 或 0.8 / 0.1 / 0.1
```

---

## 时间相关错误

### time_var_not_found

**错误信息**：`无法自动检测时间变量`

**原因**：NC 文件中没有名为 `time`、`ocean_time`、`Time` 等常见时间变量。

**处理**：
```
无法自动检测时间变量。
数据中的所有变量：sst, ssh, t_dim, coordinates

请告诉我时间变量的名称，以便正确排序时序数据。
（提示：通常是维度变量，形状为 [T]）

您可以通过 ocean_inspect_data 查看变量详情。
```

---

## 系统错误

### permission_denied

**错误信息**：`无法写入目录: /path/to/output`

**处理**：
```
无法写入输出目录，请检查：
1. 目录是否存在（工具会自动创建，但父目录需存在）
2. 是否有写入权限
3. 磁盘空间是否充足
```

### python_error

**错误信息**：`Python 脚本执行失败`

**处理**：
1. 展示完整的 Python 错误信息
2. 检查 Python 环境（是否安装 netCDF4、numpy、xarray）
3. 如果是依赖问题，提示安装：
```bash
/home/lz/miniconda3/envs/pytorch/bin/pip install netCDF4 xarray numpy
```
4. 等待用户指示，**不要自动重试**

---

## 错误恢复流程

当遇到错误时，Agent 应该：

1. **停止执行**：不要尝试自动修复或重试
2. **展示错误**：清楚说明什么出错了
3. **给出建议**：提供可能的解决方案
4. **等待指示**：让用户决定下一步

**展示格式**：
```
执行过程中遇到错误：

【错误类型】变量不存在
【错误详情】变量 "temperature" 在数据中不存在
【可能原因】变量名拼写不同（NC 中实际名称为 "sst"）

【建议操作】
1. 使用变量名 "sst" 重新选择
2. 用 ocean_inspect_data 确认准确变量名

请问您想如何处理？
```
