/**
 * @file inspect.ts
 * @description Step A: 数据检查与变量分类工具
 *              调用 Python 脚本分析 NC 文件
 *
 * @author leizheng
 * @date 2026-02-02
 * @version 2.2.0
 *
 * @changelog
 *   - 2026-02-03 leizheng: v2.2.0 添加 nc_files 参数支持明确指定文件列表
 *   - 2026-02-03 leizheng: v2.1.0 P0 安全修复
 *     - 添加路径验证（检测文件路径 vs 目录路径）
 *   - 2026-02-02 leizheng: v2.0.0 重构为调用独立 Python 脚本
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findFirstPythonPath } from '@/utils/python-manager'
import path from 'node:path'

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
      description: 'NC文件所在目录的绝对路径'
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
    }
  },

  attributes: {
    readonly: true,
    noEffect: true
  },

  async exec(args, ctx) {
    const {
      nc_folder,
      nc_files,
      static_file,
      file_filter = '',
      dyn_file_pattern = '*.nc'
    } = args

    ctx.emit('step_started', { step: 'A', description: '查看数据并定义变量' })

    // 1. 检查 Python 环境
    const pythonPath = findFirstPythonPath()
    if (!pythonPath) {
      const errorMsg = '未找到可用的Python解释器，请安装Python或配置PYTHON/PYENV'
      ctx.emit('step_failed', { step: 'A', error: errorMsg })
      return {
        status: 'error',
        errors: [errorMsg],
        message: '数据检查失败'
      }
    }

    // 2. 准备路径
    const pythonCmd = `"${pythonPath}"`
    const tempDir = path.resolve(ctx.sandbox.workDir, 'ocean_preprocess_temp')
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

    try {
      // 4. 创建临时目录并写入配置
      await ctx.sandbox.exec(`mkdir -p "${tempDir}"`)
      await ctx.sandbox.fs.write(configPath, JSON.stringify(config, null, 2))

      // 5. 执行 Python 脚本
      const result = await ctx.sandbox.exec(
        `${pythonCmd} "${scriptPath}" --config "${configPath}" --output "${outputPath}"`,
        { timeoutMs: 300000 }
      )

      if (result.code !== 0) {
        ctx.emit('step_failed', { step: 'A', error: result.stderr })
        return {
          status: 'error',
          errors: [`Python执行失败: ${result.stderr}`],
          message: '数据检查失败'
        }
      }

      // 6. 读取结果
      const jsonContent = await ctx.sandbox.fs.read(outputPath)
      const inspectResult: InspectResult = JSON.parse(jsonContent)

      ctx.emit('step_completed', {
        step: 'A',
        file_count: inspectResult.file_count,
        dynamic_files: inspectResult.dynamic_files,
        suspected_static_files: inspectResult.suspected_static_files,
        dynamic_vars: inspectResult.dynamic_vars_candidates
      })

      return inspectResult

    } catch (error: any) {
      ctx.emit('step_failed', { step: 'A', error: error.message })
      return {
        status: 'error',
        errors: [error.message],
        message: '数据检查执行异常'
      }
    }
  }
})
