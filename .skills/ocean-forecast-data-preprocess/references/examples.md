# 对话示例

> 版本: 1.2.0 | 最后更新: 2026-02-26

本文档展示典型对话流程，帮助理解 4 阶段确认机制。

---

## 快速参考

| 用户意图 | Agent 操作 |
|---------|-----------|
| 想预处理 NC 数据 | 调用 `ocean_forecast_preprocess_full`（触发阶段 1） |
| 只想查看数据结构 | 调用 `ocean_inspect_data` |
| 已知所有参数，想跳过前几阶段 | 一次性提供所有参数，直接推进到阶段 4 |
| 阶段 2 不要静态变量/掩码 | 明确传入 `stat_vars=[]`、`mask_vars=[]` |
| 只需裁剪特定区域 | 阶段 3 提供 `h_slice` / `w_slice` |

---

## 示例 1：标准完整流程

### 用户开始

```
用户：我有一批海洋 NC 数据，想转成 NPY 格式用于预测模型训练。
      数据目录：/data/ocean/sst_ssh
      输出目录：/output/forecast

Agent：好的，我先分析您的数据结构。

[调用 ocean_forecast_preprocess_full]
  nc_folder: "/data/ocean/sst_ssh"
  output_base: "/output/forecast"
```

### 阶段 1：选择预测变量

```
[工具返回 awaiting_variable_selection]

Agent：✅ 数据分析完成！

📁 数据目录：/data/ocean/sst_ssh
📊 文件数量：1095 个 NC 文件
⏱ 时间范围：2020-01-01 → 2022-12-31（共 1095 步）
🔲 空间分辨率：H=720, W=1440

🌊 检测到的动态变量（时序数据）：
  - sst: 形状 [T, 720, 1440]
  - ssh: 形状 [T, 720, 1440]
  - sss: 形状 [T, 720, 1440]

请问您要预测哪些变量？（可多选）

用户：sst 和 ssh 就行
```

### 阶段 2：静态变量与掩码选择

```
[调用 ocean_forecast_preprocess_full，加入 dyn_vars=["sst","ssh"]]
[工具返回 awaiting_static_selection]

Agent：📍 静态/坐标变量：
  - lon: 经度 [720, 1440]
  - lat: 纬度 [720, 1440]

🗺 疑似掩码变量：
  - mask: 海陆掩码 [720, 1440]  ← 推荐

请确认：
1. 需要保存哪些静态变量？
2. 使用哪个掩码变量？

用户：静态变量保存 lon 和 lat，掩码用 mask
```

### 阶段 3：划分比例配置

```
[调用 ocean_forecast_preprocess_full，加入 stat_vars + mask_vars]
[工具返回 awaiting_parameters]

Agent：⚙️ 请配置数据集划分参数：

时序数据共 1095 个时间步，请提供训练/验证/测试比例（三者之和 = 1）：
  - 常用方案：0.7 / 0.1 / 0.2

是否需要空间裁剪？（如不需要请直接跳过）

用户：用 0.7 / 0.1 / 0.2，不需要裁剪
```

### 阶段 4：执行前确认

```
[调用 ocean_forecast_preprocess_full，加入比例参数]
[工具返回 awaiting_execution，携带 confirmation_token]

Agent：📋 参数汇总，请确认：

┌─────────────────────────────────────────┐
│  数据目录：/data/ocean/sst_ssh          │
│  输出目录：/output/forecast             │
│  预测变量：sst, ssh                     │
│  静态变量：lon, lat                     │
│  掩码变量：mask                         │
│  划分比例：训练 70% / 验证 10% / 测试 20%│
│  预计文件：训练 766 / 验证 110 / 测试 219│
└─────────────────────────────────────────┘

确认无误后请回复"确认执行"。

用户：确认执行
```

### 执行与报告

```
[调用 ocean_forecast_preprocess_full，user_confirmed=true + confirmation_token]
[工具执行 Step B → Step C]

Agent：✅ 预处理完成！

📊 处理结果：
  训练集：766 个时间步（2020-01-01 → 2022-02-05）
  验证集：110 个时间步（2022-02-06 → 2022-06-26）
  测试集：219 个时间步（2022-06-27 → 2022-12-31）

🖼 可视化图片已生成：
  /output/forecast/visualisation_forecast/sst_frames.png
  /output/forecast/visualisation_forecast/sst_timeseries.png

[调用 ocean_forecast_generate_report]

📝 预处理报告已生成：/output/forecast/preprocessing_report.md
```

---

## 示例 2：用户只想查看数据结构

```
用户：帮我看一下 /data/ocean/ 这个目录里的 NC 文件有什么变量

Agent：好的，我来查看目录结构。

[调用 ocean_inspect_data]
  nc_folder: "/data/ocean/"

Agent：📂 文件变量列表：

变量名    | 形状              | 类型     | 说明
----------|-------------------|----------|------
sst       | (1, 720, 1440)    | float32  | sea surface temp
ssh       | (1, 720, 1440)    | float32  | sea surface height
lon       | (720, 1440)       | float64  | longitude
lat       | (720, 1440)       | float64  | latitude
mask      | (720, 1440)       | int8     | land-sea mask
time      | (1,)              | float64  | time variable
```

---

## 示例 3：用户提前提供部分参数

```
用户：我知道我要选什么变量了，直接预测 sst，
      静态要 lon lat，掩码是 mask，
      比例用 0.8 / 0.1 / 0.1

Agent：好的，我帮您直接推进到参数确认阶段。

[调用 ocean_forecast_preprocess_full，提供 dyn_vars + stat_vars + mask_vars + 比例]
[工具直接推进到 awaiting_execution]

Agent：已跳过前几阶段，参数汇总如下：

（展示参数汇总，等待确认）
```

---

## 示例 4：空间裁剪场景

```
阶段 3 对话：

用户：比例用 0.7 / 0.15 / 0.15，
      只要中国近海区域，H 方向裁 100:500，W 方向裁 800:1200

Agent：明白，配置如下：
  划分比例：训练 70% / 验证 15% / 测试 15%
  H 裁剪：100:500（共 400 行）
  W 裁剪：800:1200（共 400 列）

[继续到阶段 4 确认]
```

---

## 示例 5：用户选择不保存静态变量

```
阶段 2 对话：

Agent：📍 检测到坐标变量：lon, lat
       🗺 检测到掩码变量：mask

用户：静态变量不用保存，掩码也不用

Agent：明白，stat_vars=[], mask_vars=[]。

[继续到阶段 3]
```
