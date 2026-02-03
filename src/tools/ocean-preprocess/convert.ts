/**
 * @file convert.ts
 * @description Step C: NC 转 NPY 转换工具
 *              调用 Python 脚本执行转换和后置验证
 *
 * @author leizheng
 * @date 2026-02-02
 * @version 2.2.0
 *
 * @changelog
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
import path from 'node:path'

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
- output_base/target_variables/变量.npy - 动态研究变量
- output_base/static_variables/编号_变量.npy - 静态变量（带编号）

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
      description: '动态NC文件所在目录 (dyn_dir)'
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
    }
  },

  attributes: {
    readonly: false,
    noEffect: false
  },

  async exec(args, ctx) {
    const {
      nc_folder,
      output_base,
      dyn_vars,
      static_file,
      stat_vars = [],
      mask_vars = [],  // P0 修复：默认为空数组，由调用方负责传入
      lon_var,         // P0 修复：移除硬编码默认值
      lat_var,         // P0 修复：移除硬编码默认值
      dyn_file_pattern = '*.nc',
      run_validation = true,
      mask_src_var,    // P0 修复：移除硬编码默认值
      mask_derive_op = 'identity',
      allow_nan = false,
      lon_range,
      lat_range,
      heuristic_check_var,
      land_threshold_abs = 1e-12,
      heuristic_sample_size = 2000,
      require_sorted = true
    } = args

    ctx.emit('step_started', { step: 'C', description: '转换为NPY格式存储' })

    // P0 修复：验证必要参数
    if (!mask_vars || mask_vars.length === 0) {
      const warningMsg = '未指定掩码变量（mask_vars），将跳过掩码相关处理'
      ctx.emit('warning', { step: 'C', message: warningMsg })
    }

    // 1. 检查 Python 环境
    const pythonPath = findFirstPythonPath()
    if (!pythonPath) {
      const errorMsg = '未找到可用的Python解释器，请安装Python或配置PYTHON/PYENV'
      ctx.emit('step_failed', { step: 'C', error: errorMsg })
      return {
        status: 'error',
        errors: [errorMsg],
        message: '转换失败'
      }
    }

    // 2. 准备路径
    const pythonCmd = `"${pythonPath}"`
    const tempDir = path.resolve(ctx.sandbox.workDir, 'ocean_preprocess_temp')
    const configPath = path.join(tempDir, 'convert_config.json')
    const outputPath = path.join(tempDir, 'convert_result.json')

    // Python 脚本路径
    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean_preprocess/convert_npy.py')

    // 3. 准备配置
    const config = {
      nc_folder,
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
      require_sorted
    }

    try {
      // 4. 创建临时目录并写入配置
      await ctx.sandbox.exec(`mkdir -p "${tempDir}"`)
      await ctx.sandbox.fs.write(configPath, JSON.stringify(config, null, 2))

      // 5. 执行 Python 脚本
      const result = await ctx.sandbox.exec(
        `${pythonCmd} "${scriptPath}" --config "${configPath}" --output "${outputPath}"`,
        { timeoutMs: 600000 }
      )

      if (result.code !== 0) {
        ctx.emit('step_failed', { step: 'C', error: result.stderr })
        return {
          status: 'error',
          errors: [`Python执行失败: ${result.stderr}`],
          message: '转换失败'
        }
      }

      // 6. 读取结果
      const jsonContent = await ctx.sandbox.fs.read(outputPath)
      const convertResult: ConvertResult = JSON.parse(jsonContent)

      ctx.emit('step_completed', {
        step: 'C',
        status: convertResult.status,
        saved_files: Object.keys(convertResult.saved_files),
        validation: convertResult.post_validation
      })

      return convertResult

    } catch (error: any) {
      ctx.emit('step_failed', { step: 'C', error: error.message })
      return {
        status: 'error',
        errors: [error.message],
        message: '转换执行异常'
      }
    }
  }
})
