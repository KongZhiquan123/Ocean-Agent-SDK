# 海洋数据预处理工具使用指南

## 概述

本工具集为 KODE SDK Agent 提供了超分辨率场景下的 NC -> NPY 数据预处理能力。Agent 可以针对不同的数据集，按照标准流程进行预处理。

## 工具列表

| 工具名 | 步骤 | 功能 |
|--------|------|------|
| `ocean_inspect_data` | Step A | 查看数据并定义变量 |
| `ocean_validate_tensor` | Step B | 进行张量约定验证 |
| `ocean_convert_npy` | Step C | 转换为 NPY 格式存储 |
| `ocean_preprocess_full` | 完整流程 | 一键执行 A→B→C |

## 快速开始

### 1. 使用 Agent 模板

```typescript
// 创建预处理 Agent
const agent = await Agent.create({
  templateId: 'ocean-preprocess-assistant'
}, deps)

// 发送预处理请求
await agent.send({
  message: '请对 /data/nc_files 目录下的数据进行预处理，输出到 /data/output'
})
```

### 2. 直接调用工具

```typescript
// Step A: 查看数据
const inspectResult = await agent.call('ocean_inspect_data', {
  nc_folder: '/data/nc_files',
  static_file: '/data/static.nc',
  file_filter: 'avg'
})

// 用户确认研究变量后...

// Step B: 验证张量
const validateResult = await agent.call('ocean_validate_tensor', {
  inspect_result_path: '/tmp/ocean_preprocess/inspect_result.json',
  research_vars: ['u_eastward', 'v_northward']
})

// Step C: 转换存储
const convertResult = await agent.call('ocean_convert_npy', {
  nc_folder: '/data/nc_files',
  output_base: '/data/output',
  research_vars: ['u_eastward', 'v_northward'],
  static_file: '/data/static.nc'
})
```

### 3. 一键执行

```typescript
const result = await agent.call('ocean_preprocess_full', {
  nc_folder: '/data/nc_files',
  output_base: '/data/output',
  research_vars: ['u_eastward', 'v_northward'],  // 必须指定
  static_file: '/data/static.nc',
  file_filter: 'avg'
})

if (result.overall_status === 'pass') {
  console.log('预处理成功!')
}
```

## 工具参数说明

### ocean_inspect_data

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| nc_folder | string | 是 | NC文件目录路径 |
| static_file | string | 否 | 静态NC文件路径 |
| file_filter | string | 否 | 文件名过滤关键字（默认 "avg"） |

### ocean_validate_tensor

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| inspect_result_path | string | 是 | Step A 结果文件路径 |
| research_vars | string[] | 是 | 用户确认的研究变量列表 |
| mask_vars | string[] | 否 | 掩码变量列表 |

### ocean_convert_npy

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| nc_folder | string | 是 | NC文件目录 |
| output_base | string | 是 | 输出基础目录 |
| research_vars | string[] | 是 | 研究变量列表 |
| static_file | string | 否 | 静态NC文件路径 |
| static_vars | string[] | 否 | 静态变量列表 |
| mask_vars | string[] | 否 | 掩码变量列表 |
| file_filter | string | 否 | 文件名过滤关键字 |

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

## 防错规则

### Step A 规则
- **A1**: 自动区分动态/静态/掩码变量
- **A2**: 陆地掩码（mask_u, mask_rho, mask_v）标记为不可修改
- **A3**: NC文件自动排序

### Step B 规则
- **B1**: 动态变量形状 [T, H, W] 或 [T, D, H, W]
- **B2**: 静态变量形状 [H, W]
- **B3**: 研究变量必须存在
- **B4**: 掩码形状与空间维匹配

### Step C 规则
- **C1**: 目录结构符合 OceanSRDataset
- **C2**: 保存前验证维度
- **C3**: 掩码变量原样保存
- **C4**: NC文件排序后处理

### 事后验证
- 检查目录结构
- 检查维度
- 全部通过返回 `status: "pass"`

## 自定义事件

Agent 会发送以下自定义事件：

```typescript
agent.on('tool_custom_event', (event) => {
  switch (event.eventType) {
    case 'step_started':
      console.log(`开始 ${event.data.step}: ${event.data.description}`)
      break
    case 'step_completed':
      console.log(`完成 ${event.data.step}`)
      break
    case 'step_failed':
      console.log(`失败 ${event.data.step}: ${event.data.error}`)
      break
    case 'pipeline_started':
      console.log('开始完整预处理流程')
      break
    case 'pipeline_completed':
      console.log('预处理流程完成')
      break
  }
})
```

## 适配不同数据集

工具设计为可以处理不同的数据集，只需调整参数：

### 示例 1：COAWST 模型数据

```typescript
await agent.call('ocean_preprocess_full', {
  nc_folder: '/data/coawst/PRE_prognostic_results/',
  output_base: '/data/processed/coawst/',
  research_vars: ['u_eastward', 'v_northward', 'temp', 'salt'],
  static_file: '/data/coawst/stat_file.nc',
  file_filter: 'avg'
})
```

### 示例 2：ROMS 模型数据

```typescript
await agent.call('ocean_preprocess_full', {
  nc_folder: '/data/roms/output/',
  output_base: '/data/processed/roms/',
  research_vars: ['u', 'v', 'temp', 'salt', 'zeta'],
  static_file: '/data/roms/grid.nc',
  file_filter: 'his'  // ROMS 通常用 'his' 文件
})
```

### 示例 3：自定义变量名

```typescript
// 先检查数据中有哪些变量
const inspectResult = await agent.call('ocean_inspect_data', {
  nc_folder: '/data/custom/',
  file_filter: ''  // 不过滤，读取所有NC文件
})

// 查看可用变量
console.log('动态变量:', inspectResult.dynamic_vars_candidates)
console.log('所有变量:', Object.keys(inspectResult.variables))

// 根据实际变量名进行处理
await agent.call('ocean_convert_npy', {
  nc_folder: '/data/custom/',
  output_base: '/data/processed/custom/',
  research_vars: ['velocity_u', 'velocity_v'],  // 使用实际变量名
  mask_vars: ['land_mask']  // 使用实际掩码名
})
```

## 错误处理

```typescript
const result = await agent.call('ocean_preprocess_full', { ... })

if (result.overall_status === 'error') {
  // 检查哪个步骤失败
  if (result.step_a?.status === 'error') {
    console.log('Step A 错误:', result.step_a.errors)
  }
  if (result.step_b?.status === 'error') {
    console.log('Step B 错误:', result.step_b.errors)
  }
  if (result.step_c?.status === 'error') {
    console.log('Step C 错误:', result.step_c.errors)
  }
}
```

## 依赖要求

服务器端需要安装 Python 及以下库：

```bash
pip install numpy xarray netCDF4 tqdm
```
