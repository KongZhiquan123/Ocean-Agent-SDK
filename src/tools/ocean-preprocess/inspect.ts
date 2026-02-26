/**
 * @file inspect.ts
 * @description Step A: 数据检查与变量分类工具
 *              调用 Python 脚本分析 NC 文件
 *
 * @author leizheng
 * @contributors kongzhiquan
 * @date 2026-02-04
 * @version 2.6.0
 *
 * @changelog
 *   - 2026-02-25 kongzhiquan: v2.7.0 tempDir 改为基于 output_base 的 .ocean_preprocess_temp
 *     - 新增 output_base 参数，用于指定临时目录的基础路径
 *   - 2026-02-07 Leizheng: v2.6.0 智能路径处理，nc_folder 同时支持目录和单个文件路径
 *     - 传入 .nc 文件路径时自动拆分为目录+文件名，无需复制数据
 *   - 2026-02-05 kongzhiquan: v2.5.0 增强验证逻辑
 *     - 添加文件数量验证（file_count === 0 时抛出错误）
 *     - 添加动态变量候选验证（无动态变量时抛出错误）
 *     - 移除冗余的 try-catch
 *   - 2026-02-04 kongzhiquan: v2.4.0 添加 mask_vars 和 static_vars 参数支持
 *     - 支持通过参数传入掩码变量列表和静态变量列表
 *     - 用于精确控制变量分类逻辑
 *   - 2026-02-04 leizheng: v2.3.0 检测维度坐标
 *     - 同时检查 ds.data_vars 和 ds.coords
 *     - 自动检测 latitude, longitude, depth 等维度坐标
 *     - 支持 Copernicus 等数据集的坐标变量
 *   - 2026-02-03 leizheng: v2.2.0 添加 nc_files 参数支持明确指定文件列表
 *   - 2026-02-03 leizheng: v2.1.0 P0 安全修复
 *     - 添加路径验证（检测文件路径 vs 目录路径）
 *   - 2026-02-02 leizheng: v2.0.0 重构为调用独立 Python 脚本
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findFirstPythonPath } from '@/utils/python-manager'
import path from 'node:path'

function shellEscapeDouble(str: string): string {
  return str.replace(/[\\"$`!]/g, '\\$&')
}

// ========================================
// 类型定义
// ========================================

interface VariableInfo {
  name: string
  category: 'dynamic' | 'static' | 'mask' | 'ignored'
  dims: string[]
  shape: number[]
  dtype: string
  units: string
  long_name: string
  is_mask: boolean
  has_time: boolean
  suspected_type: 'suspected_mask' | 'suspected_coordinate' | 'dynamic' | 'static' | 'unknown'
}

export interface InspectResult {
  status: 'success' | 'error' | 'awaiting_confirmation'
  nc_folder: string
  file_count: number
  file_list: string[]
  variables: Record<string, VariableInfo>
  dynamic_vars_candidates: string[]
  static_vars_found: string[]
  mask_vars_found: string[]
  statistics: Record<string, any>
  warnings: string[]
  errors: string[]
  message: string
  suspected_masks: string[]
  suspected_coordinates: string[]
  // v2.2 新增
  dynamic_files: string[]
  suspected_static_files: string[]
  file_analysis: Record<string, any>
}

// ========================================
// 工具定义
// ========================================

export const oceanInspectDataTool = defineTool({
  name: 'ocean_inspect_data',
  description: `Step A: 查看NC数据并定义变量

用于超分辨率场景的数据预处理第一步。从NC文件中提取变量信息，自动分类动态/静态/掩码变量。

**v2.2 新功能**：
- 支持 nc_files 参数明确指定要处理的文件
- 逐个文件检测时间维度，自动识别混入目录的静态文件

**防错规则**：
- A1: 自动区分动态变量（有时间维）、静态变量（无时间维）、掩码变量（mask_*）
- A2: 陆地掩码变量会被标记为不可修改
- A3: NC文件会自动排序以确保时间顺序正确
- A4: 检测静态文件混入动态目录的情况

**返回**：变量列表、形状信息、统计信息、动态变量候选列表、文件分类

**重要**：执行后需要用户确认研究变量是什么`,

  params: {
    nc_folder: {
      type: 'string',
      description: 'NC文件所在目录的绝对路径，也可以直接传入单个 .nc 文件路径（自动提取所在目录）'
    },
    nc_files: {
      type: 'array',
      items: { type: 'string' },
      description: '可选：明确指定要处理的文件列表（支持简单通配符如 "ocean_avg_*.nc"）',
      required: false
    },
    static_file: {
      type: 'string',
      description: '静态NC文件的绝对路径（可选）',
      required: false
    },
    file_filter: {
      type: 'string',
      description: '文件名过滤关键字（可选）',
      required: false,
      default: ''
    },
    dyn_file_pattern: {
      type: 'string',
      description: '动态文件的 glob 匹配模式，如 "*.nc"（当 nc_files 未指定时使用）',
      required: false,
      default: '*.nc'
    },
    mask_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '可选：明确指定掩码变量列表（如 ["mask_rho", "mask_u"]），用于变量分类',
      required: false
    },
    static_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '可选：明确指定静态变量列表（如 ["h", "angle", "pm", "pn"]），用于变量分类',
      required: false
    },
    output_base: {
      type: 'string',
      description: '输出基础目录，用于存放临时文件'
    }
  },

  attributes: {
    readonly: true,
    noEffect: true
  },

  async exec(args, ctx) {
    const {
      nc_folder: rawNcFolder,
      nc_files: rawNcFiles,
      static_file,
      file_filter = '',
      dyn_file_pattern = '*.nc',
      mask_vars,
      static_vars,
      output_base
    } = args

    // 智能路径处理：支持目录或单个 .nc 文件路径
    let nc_folder = rawNcFolder.trim()
    let nc_files = rawNcFiles

    if (nc_folder.endsWith('.nc') || nc_folder.endsWith('.NC')) {
      const filePath = nc_folder
      const lastSlash = filePath.lastIndexOf('/')
      if (lastSlash === -1) {
        nc_folder = '.'
        nc_files = [filePath]
      } else {
        nc_folder = filePath.substring(0, lastSlash)
        nc_files = [filePath.substring(lastSlash + 1)]
      }
    }

    // 1. 检查 Python 环境
    const pythonPath = findFirstPythonPath()
    if (!pythonPath) {
      throw new Error('未找到可用的Python解释器，请安装Python或配置PYTHON/PYENV')
    }

    // 2. 准备路径
    const pythonCmd = `"${shellEscapeDouble(pythonPath)}"`
    const tempDir = path.resolve(output_base, '.ocean_preprocess_temp')
    const configPath = path.join(tempDir, 'inspect_config.json')
    const outputPath = path.join(tempDir, 'inspect_result.json')

    // Python 脚本路径（相对于项目根目录）
    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean_preprocess/inspect_data.py')

    // 3. 准备配置
    const config: Record<string, any> = {
      nc_folder,
      static_file: static_file || null,
      file_filter,
      dyn_file_pattern
    }

    // 如果指定了 nc_files，添加到配置
    if (nc_files && nc_files.length > 0) {
      config.nc_files = nc_files
    }

    // 如果指定了 mask_vars，添加到配置
    if (mask_vars && mask_vars.length > 0) {
      config.mask_vars = mask_vars
    }

    // 如果指定了 static_vars，添加到配置
    if (static_vars && static_vars.length > 0) {
      config.static_vars = static_vars
    }

    // 4. 创建临时目录并写入配置
    await ctx.sandbox.exec(`mkdir -p "${shellEscapeDouble(tempDir)}"`)
    await ctx.sandbox.fs.write(configPath, JSON.stringify(config, null, 2))

    // 5. 执行 Python 脚本
    const result = await ctx.sandbox.exec(
      `${pythonCmd} "${shellEscapeDouble(scriptPath)}" --config "${shellEscapeDouble(configPath)}" --output "${shellEscapeDouble(outputPath)}"`,
      { timeoutMs: 300000 }
    )

    if (result.code !== 0) {
      throw new Error(`Python执行失败: ${result.stderr}`)
    }

    // 6. 读取结果
    const jsonContent = await ctx.sandbox.fs.read(outputPath)
    const inspectResult: InspectResult = JSON.parse(jsonContent)

    // 7. 验证结果
    if (inspectResult.status === 'error') {
      throw new Error(`数据检查失败: ${inspectResult.errors.join('; ')}`)
    }
    // 检查是否找到动态数据文件
    if (inspectResult.file_count === 0) {
      throw new Error(`未找到匹配的动态数据文件！
- 搜索目录: ${nc_folder}
- 文件匹配模式: "${dyn_file_pattern}"
请检查：
1. nc_folder 路径是否正确
2. dyn_file_pattern 是否匹配你的文件名`)
    }

    // 检查是否找到任何动态变量候选
    const dynCandidates = inspectResult.dynamic_vars_candidates || []
    if (dynCandidates.length === 0) {
      const allVarNames = Object.keys(inspectResult.variables || {})
      throw new Error(`数据文件中没有找到任何动态变量（带时间维度的变量）！

这通常意味着您可能提供了静态文件而非动态数据文件。

【文件信息】
- 搜索目录: ${nc_folder}
- 找到文件数: ${inspectResult.file_count}
- 文件列表: ${(inspectResult.file_list || []).slice(0, 3).join(', ')}${(inspectResult.file_list || []).length > 3 ? '...' : ''}

【检测到的变量】（都没有时间维度）
${allVarNames.slice(0, 10).join(', ')}${allVarNames.length > 10 ? '...' : ''}

请检查：
1. 您是否将静态文件路径填到了动态数据目录？
2. 动态数据文件是否确实包含时间维度？
3. 时间维度的名称是否为标准名称（time, ocean_time, t 等）？`)
    }

    return inspectResult
  }
})
