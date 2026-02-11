/**
 * @file convert.ts
 * @description Step C: NC 转 NPY 转换工具
 *              调用 Python 脚本执行转换和后置验证
 *
 * @author leizheng
 * @contributors kongzhiquan
 * @date 2026-02-04
 * @version 3.6.0
 *
 * @changelog
 *   - 2026-02-07 kongzhiquan: v3.6.0 新增 max_files 参数，限制处理的最大 NC 文件数量
 *   - 2026-02-07 Leizheng: v3.5.0 新增 nc_files 参数，支持显式指定文件列表
 *   - 2026-02-07 Leizheng: v3.4.0 智能路径处理，nc_folder 同时支持目录和单个文件路径
 *   - 2026-02-05 kongzhiquan: v3.3.0 日期文件名功能
 *     - 新增 use_date_filename, date_format, time_var 参数
 *     - 支持从 NC 文件提取时间戳作为 NPY 文件名
 *     - 自动检测日期格式（日/小时/分钟级数据）
 *   - 2026-02-05 kongzhiquan: v3.2.0 合并重构与区域裁剪功能
 *     - 移除 try-catch，统一由上层处理错误
 *     - 错误时直接 throw Error 而非返回 status: 'error'
 *     - 新增 enable_region_crop, crop_lon_range, crop_lat_range, crop_mode 参数
 *     - 参数传递给 Python 脚本 convert_npy.py
 *   - 2026-02-04 leizheng: v2.7.0 修复 1D 坐标裁剪
 *     - latitude 正确使用 h_slice 裁剪
 *     - longitude 正确使用 w_slice 裁剪
 *     - 修复可视化坐标轴显示错误
 *   - 2026-02-04 leizheng: v2.6.0 文件级并行处理
 *     - 使用 multiprocessing.Pool 替代 xr.open_mfdataset
 *     - 彻底解决段错误问题
 *   - 2026-02-04 leizheng: v2.5.1 同步 Python 修复
 *     - workers 参数现在正确配置 dask 线程数
 *     - 支持从动态文件提取静态变量
 *   - 2026-02-04 leizheng: v2.5.0 支持粗网格模式
 *     - 新增 output_subdir 参数（默认 'hr'，可设为 'lr'）
 *     - 用于支持粗网格数据直接输出到 lr/ 目录
 *   - 2026-02-03 leizheng: v2.4.0 裁剪与多线程
 *     - 新增 h_slice/w_slice 参数，在转换时直接裁剪
 *     - 新增 scale 参数，验证裁剪后尺寸能否被整除
 *     - 新增 workers 参数，多线程并行处理（默认 8，且不超过 CPU 核数）
 *   - 2026-02-03 leizheng: v2.3.0 数据集划分功能
 *     - 新增 train_ratio/valid_ratio/test_ratio 参数
 *     - 按时间顺序划分数据到 train/valid/test 目录
 *     - 输出目录结构改为 train/hr/, valid/hr/, test/hr/
 *   - 2026-02-03 leizheng: v2.2.0 P0 安全修复
 *     - 移除硬编码默认值（mask_vars, lon_var, lat_var, mask_src_var）
 *     - 所有变量名必须由调用方显式传入
 *   - 2026-02-02 leizheng: v2.1.0 增加 P0 特性
 *     - allow_nan: NaN/Inf 采样检测
 *     - lon_range/lat_range: 坐标范围验证
 *     - 多网格支持（C-grid staggered mesh）
 *   - 2026-02-02 leizheng: v2.0.0 重构为调用独立 Python 脚本
 *     - 静态变量添加编号前缀（00_lon_rho, 99_mask_rho）
 *     - 集成后置验证 (Rule 1/2/3)
 *     - 自动生成 preprocess_manifest.json
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findFirstPythonPath } from '@/utils/python-manager'
import os from 'node:os'
import path from 'node:path'

function shellEscapeDouble(str: string): string {
  return str.replace(/[\\"$`!]/g, '\\$&')
}

const DEFAULT_WORKERS = Math.max(1, Math.min(8, os.cpus().length || 1))

// ========================================
// 类型定义
// ========================================

export interface ConvertResult {
  status: 'pass' | 'error' | 'pending'
  output_dir: string
  saved_files: Record<string, any>
  post_validation: Record<string, any>
  validation_rule1?: Record<string, any>
  validation_rule2?: Record<string, any>
  validation_rule3?: Record<string, any>
  warnings: string[]
  errors: string[]
  message: string
}

// ========================================
// 工具定义
// ========================================

export const oceanConvertNpyTool = defineTool({
  name: 'ocean_convert_npy',
  description: `Step C: 转换为NPY格式并按目录结构存储

将NC文件中的变量转换为NPY格式，按 OceanSRDataset 要求的目录结构保存。

**核心特性**：
- 使用 xr.open_mfdataset 进行多文件惰性加载和自动拼接
- 静态变量自动添加编号前缀（00_lon_rho, 01_lat_rho, 99_mask_rho）
- 自动生成 preprocess_manifest.json
- 执行后置验证 (Rule 1/2/3)

**输出目录结构**：
- output_base/train/hr/变量.npy - 训练集高分辨率数据
- output_base/train/lr/ - 训练集低分辨率数据（预留）
- output_base/valid/hr/变量.npy - 验证集高分辨率数据
- output_base/valid/lr/ - 验证集低分辨率数据（预留）
- output_base/test/hr/变量.npy - 测试集高分辨率数据
- output_base/test/lr/ - 测试集低分辨率数据（预留）
- output_base/static_variables/编号_变量.npy - 静态变量（带编号）

**数据集划分**：
- 按时间顺序划分（不随机）
- 前 train_ratio 为训练集
- 中间 valid_ratio 为验证集
- 最后 test_ratio 为测试集

**编号规则**：
- 00-09: 经度变量 (lon_rho, lon_u, ...)
- 10-19: 纬度变量 (lat_rho, lat_u, ...)
- 20-89: 其他静态变量 (h, angle, f, ...)
- 90-99: 掩码变量 (mask_rho, mask_u, ...)

**后置验证 (Rule 1/2/3)**：
- Rule 1: 输出完整性与形状约定
- Rule 2: 掩码不可变性检查
- Rule 3: 排序确定性检查`,

  params: {
    nc_folder: {
      type: 'string',
      description: '动态NC文件所在目录 (dyn_dir)，也可以直接传入单个 .nc 文件路径'
    },
    nc_files: {
      type: 'array',
      items: { type: 'string' },
      description: '显式指定要处理的 NC 文件列表（文件名或完整路径）。提供时忽略 dyn_file_pattern',
      required: false
    },
    output_base: {
      type: 'string',
      description: '输出根目录 (output_base_dir)'
    },
    dyn_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '动态变量列表 (dyn_vars)'
    },
    static_file: {
      type: 'string',
      description: '静态NC文件路径 (stat_file)',
      required: false
    },
    stat_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '静态变量列表 (stat_vars)',
      required: false,
      default: []
    },
    mask_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '掩码变量列表（必须由用户指定或从数据检测）',
      required: false
      // P0 修复：移除硬编码默认值 ['mask_rho', 'mask_u', 'mask_v', 'mask_psi']
    },
    lon_var: {
      type: 'string',
      description: '经度参考变量名（必须由用户指定或从数据检测）',
      required: false
      // P0 修复：移除硬编码默认值 'lon_rho'
    },
    lat_var: {
      type: 'string',
      description: '纬度参考变量名（必须由用户指定或从数据检测）',
      required: false
      // P0 修复：移除硬编码默认值 'lat_rho'
    },
    dyn_file_pattern: {
      type: 'string',
      description: '动态文件的 glob 匹配模式，如 "*.nc"',
      required: false,
      default: '*.nc'
    },
    run_validation: {
      type: 'boolean',
      description: '是否执行后置验证 (Rule 1/2/3)',
      required: false,
      default: true
    },
    mask_src_var: {
      type: 'string',
      description: '用于精确对比的源掩码变量名（必须由用户指定或从数据检测）',
      required: false
      // P0 修复：移除硬编码默认值 'mask_rho'
    },
    mask_derive_op: {
      type: 'string',
      description: '掩码推导操作: identity, land_is_zero, ocean_is_one, invert01',
      required: false,
      default: 'identity'
    },
    allow_nan: {
      type: 'boolean',
      description: '是否允许 NaN/Inf 值存在（默认 false，检测到会报错）',
      required: false,
      default: false
    },
    lon_range: {
      type: 'array',
      items: { type: 'number' },
      description: '经度有效范围 [min, max]，如 [-180, 180]',
      required: false
    },
    lat_range: {
      type: 'array',
      items: { type: 'number' },
      description: '纬度有效范围 [min, max]，如 [-90, 90]',
      required: false
    },
    heuristic_check_var: {
      type: 'string',
      description: '用于启发式掩码验证的动态变量名（如 "uo"）',
      required: false
    },
    land_threshold_abs: {
      type: 'number',
      description: '陆地零值判定阈值（默认 1e-12）',
      required: false,
      default: 1e-12
    },
    heuristic_sample_size: {
      type: 'number',
      description: '启发式验证采样点数（默认 2000）',
      required: false,
      default: 2000
    },
    require_sorted: {
      type: 'boolean',
      description: '是否要求 NC 文件按字典序排序（默认 true）',
      required: false,
      default: true
    },
    train_ratio: {
      type: 'number',
      description: '【必须由用户指定】训练集比例（按时间顺序取前 N%），如 0.7',
      required: false
    },
    valid_ratio: {
      type: 'number',
      description: '【必须由用户指定】验证集比例（按时间顺序取中间 N%），如 0.15',
      required: false
    },
    test_ratio: {
      type: 'number',
      description: '【必须由用户指定】测试集比例（按时间顺序取最后 N%），如 0.15',
      required: false
    },
    h_slice: {
      type: 'string',
      description: 'H 方向裁剪切片，如 "0:680"（在转换时直接裁剪）',
      required: false
    },
    w_slice: {
      type: 'string',
      description: 'W 方向裁剪切片，如 "0:1440"（在转换时直接裁剪）',
      required: false
    },
    scale: {
      type: 'number',
      description: '下采样倍数（用于验证裁剪后尺寸能否被整除）',
      required: false
    },
    workers: {
      type: 'number',
      description: '并行线程数（默认 8，且不超过 CPU 核数）',
      required: false,
      default: DEFAULT_WORKERS
    },
    output_subdir: {
      type: 'string',
      description: '输出子目录名（默认 "hr"，粗网格数据时设为 "lr"）',
      required: false,
      default: 'hr'
    },
    // ========== 区域裁剪参数（v3.1.0 新增）==========
    enable_region_crop: {
      type: 'boolean',
      description: '是否启用区域裁剪（先裁剪到特定经纬度区域再进行下采样）',
      required: false,
      default: false
    },
    crop_lon_range: {
      type: 'array',
      items: { type: 'number' },
      description: '区域裁剪的经度范围 [min, max]，如 [100, 120]',
      required: false
    },
    crop_lat_range: {
      type: 'array',
      items: { type: 'number' },
      description: '区域裁剪的纬度范围 [min, max]，如 [20, 40]',
      required: false
    },
    crop_mode: {
      type: 'string',
      description: '区域裁剪模式: "one_step"（一步到位）或 "two_step"（两步裁剪，保存 raw）',
      required: false,
      default: 'two_step'
    },
    // ========== 日期文件名参数（v3.2.0 新增）==========
    use_date_filename: {
      type: 'boolean',
      description: '是否使用日期作为文件名（如 20200101.npy 而非 000000.npy），默认开启',
      required: false,
      default: true
    },
    date_format: {
      type: 'string',
      description: '日期格式: "auto"（自动检测）, "YYYYMMDD", "YYYYMMDDHH", "YYYYMMDDHHmm"',
      required: false,
      default: 'auto'
    },
    time_var: {
      type: 'string',
      description: '时间变量名（默认自动检测 time/ocean_time 等）',
      required: false
    },
    // ========== 文件数限制参数（v3.6.0 新增）==========
    max_files: {
      type: 'number',
      description: '可选：限制处理的最大 NC 文件数量（按排序后取前 N 个）',
      required: false
    }
  },

  attributes: {
    readonly: false,
    noEffect: false
  },

  async exec(args, ctx) {
    const {
      nc_folder: rawNcFolder,
      nc_files: rawNcFiles,
      output_base,
      dyn_vars,
      static_file,
      stat_vars = [],
      mask_vars = [],  // P0 修复：默认为空数组，由调用方负责传入
      lon_var,         // P0 修复：移除硬编码默认值
      lat_var,         // P0 修复：移除硬编码默认值
      dyn_file_pattern: rawFilePattern = '*.nc',
      run_validation = true,
      mask_src_var,    // P0 修复：移除硬编码默认值
      mask_derive_op = 'identity',
      allow_nan = false,
      lon_range,
      lat_range,
      heuristic_check_var,
      land_threshold_abs = 1e-12,
      heuristic_sample_size = 2000,
      require_sorted = true,
      train_ratio,
      valid_ratio,
      test_ratio,
      h_slice,
      w_slice,
      scale,
      workers = DEFAULT_WORKERS,
      output_subdir = 'hr',
      // 区域裁剪参数（v3.1.0 新增）
      enable_region_crop = false,
      crop_lon_range,
      crop_lat_range,
      crop_mode = 'two_step',
      // 日期文件名参数（v3.2.0 新增）
      use_date_filename = true,
      date_format = 'auto',
      time_var,
      // 文件数限制参数（v3.6.0 新增）
      max_files
    } = args

    // 智能路径处理：支持目录或单个 .nc 文件路径
    let nc_folder = rawNcFolder.trim()
    let dyn_file_pattern = rawFilePattern

    if (nc_folder.endsWith('.nc') || nc_folder.endsWith('.NC')) {
      const filePath = nc_folder
      const lastSlash = filePath.lastIndexOf('/')
      if (lastSlash === -1) {
        dyn_file_pattern = filePath
        nc_folder = '.'
      } else {
        nc_folder = filePath.substring(0, lastSlash)
        dyn_file_pattern = filePath.substring(lastSlash + 1)
      }
    }

    // 验证数据集划分比例（必须由用户指定）
    if (train_ratio === undefined || valid_ratio === undefined || test_ratio === undefined) {
      throw new Error('数据集划分比例必须由用户指定！请提供 train_ratio, valid_ratio, test_ratio 参数')
    }

    // 验证划分比例之和
    const totalRatio = train_ratio + valid_ratio + test_ratio
    if (Math.abs(totalRatio - 1.0) > 0.01) {
      throw new Error(`数据集划分比例之和必须为 1.0，当前为 ${totalRatio}`)
    }

    // 1. 检查 Python 环境
    const pythonPath = findFirstPythonPath()
    if (!pythonPath) {
      throw new Error('未找到可用的Python解释器，请安装Python或配置PYTHON/PYENV')
    }

    // 2. 准备路径
    const pythonCmd = `"${shellEscapeDouble(pythonPath)}"`
    const tempDir = path.resolve(ctx.sandbox.workDir, 'ocean_preprocess_temp')
    const configPath = path.join(tempDir, 'convert_config.json')
    const outputPath = path.join(tempDir, 'convert_result.json')

    // Python 脚本路径
    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean_preprocess/convert_npy.py')

    // 3. 准备配置
    const config = {
      nc_folder,
      nc_files: rawNcFiles || null,  // 显式文件列表，提供时优先于 glob
      output_base,
      dyn_vars,
      static_file: static_file || null,
      stat_vars,
      mask_vars,
      lon_var: lon_var || null,   // P0 修复：允许为空
      lat_var: lat_var || null,   // P0 修复：允许为空
      dyn_file_pattern,
      run_validation,
      mask_src_var: mask_src_var || (mask_vars.length > 0 ? mask_vars[0] : null),  // P0 修复：使用第一个掩码变量
      mask_derive_op,
      allow_nan,
      lon_range: lon_range || null,
      lat_range: lat_range || null,
      heuristic_check_var: heuristic_check_var || null,
      land_threshold_abs,
      heuristic_sample_size,
      require_sorted,
      train_ratio,
      valid_ratio,
      test_ratio,
      // 裁剪参数
      h_slice: h_slice || null,
      w_slice: w_slice || null,
      scale: scale || null,
      workers: Math.max(1, Math.min(workers, os.cpus().length || 1)),
      // 输出子目录
      output_subdir,
      // 区域裁剪参数（v3.1.0 新增）
      enable_region_crop,
      crop_lon_range: crop_lon_range || null,
      crop_lat_range: crop_lat_range || null,
      crop_mode,
      // 日期文件名参数（v3.2.0 新增）
      use_date_filename,
      date_format,
      time_var: time_var || null,
      // 文件数限制（v3.6.0 新增）
      max_files: max_files || null
    }

    // 4. 创建临时目录并写入配置
    await ctx.sandbox.exec(`mkdir -p "${shellEscapeDouble(tempDir)}"`)
    await ctx.sandbox.fs.write(configPath, JSON.stringify(config, null, 2))

    // 5. 执行 Python 脚本
    const result = await ctx.sandbox.exec(
      `${pythonCmd} "${shellEscapeDouble(scriptPath)}" --config "${shellEscapeDouble(configPath)}" --output "${shellEscapeDouble(outputPath)}"`,
      { timeoutMs: 1800000 }
    )

    if (result.code !== 0) {
      throw new Error(`Python执行失败: ${result.stderr}`)
    }

    // 6. 读取结果
    const jsonContent = await ctx.sandbox.fs.read(outputPath)
    const convertResult: ConvertResult = JSON.parse(jsonContent)
    if (convertResult.status === 'error') {
      throw new Error(`NC 转 NPY 转换失败: ${convertResult.errors.join('; ')}`)
    } else if (convertResult.status === 'pending') {
      throw new Error(`NC 转 NPY 转换未完成: ${convertResult.message}`)
    }

    return convertResult
  }
})
