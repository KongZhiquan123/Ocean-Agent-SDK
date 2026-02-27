/**
 * @file stats.ts
 * @description 海洋预报数据统计工具
 *              调用 forecast_stats.py，计算已预处理 NPY 数据的 per-variable 统计量
 *
 * @author Leizheng
 * @date 2026-02-26
 * @version 1.1.0
 *
 * @changelog
 *   - 2026-02-26 Leizheng: v1.1.0 新增两项质量检查
 *     - time_boundary_check: 验证 train_end < valid_start < test_start
 *     - cross_split_check: 均值偏移 z-score + 方差比告警
 *   - 2026-02-26 Leizheng: v1.0.0 初始版本
 *     - 支持 NaN 率、值域、均值、标准差、P5/P95 分位数
 *     - NaN 率 > 0.3 自动生成警告
 *     - 结果写入 {dataset_root}/data_stats.json
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

interface VarStats {
  nan_rate: number
  min: number | null
  max: number | null
  mean: number | null
  std: number | null
  p5: number | null
  p95: number | null
  sample_files: number
}

interface SplitBoundaryCheck {
  passed: boolean | null
  note?: string
  [key: string]: string | boolean | null | undefined
}

interface TimeBoundaryCheck {
  passed: boolean
  splits: Record<string, { start: string | null; end: string | null; count: number }>
  boundaries: Record<string, SplitBoundaryCheck>
  errors: string[]
}

interface CrossSplitVarResult {
  mean_diff: number
  mean_z_score: number | null
  std_ratio: number | null
  flag: 'ok' | 'warn_mean' | 'warn_std' | 'warn_both'
}

interface CrossSplitCheck {
  warnings: string[]
  variables: Record<string, Record<string, CrossSplitVarResult>>
}

export interface ForecastStatsResult {
  status: 'pass' | 'error'
  dataset_root: string
  stats: Record<string, Record<string, VarStats>>
  time_boundary_check?: TimeBoundaryCheck
  cross_split_check?: CrossSplitCheck
  warnings: string[]
  errors: string[]
  data_stats_path?: string
  message?: string
}

// ========================================
// 工具定义
// ========================================

export const oceanForecastPreprocessStatsTool = defineTool({
  name: 'ocean_forecast_preprocess_stats',
  description: `计算预报数据集的 per-variable 统计量

读取已预处理的 NPY 文件，计算每个变量在 train/valid/test 各 split 中的：
- NaN 率（异常值占比）
- 值域（min / max）
- 均值和标准差
- P5 / P95 分位数

**变量列表**：自动从 {dataset_root}/var_names.json 读取。
**输出**：统计结果写入 {dataset_root}/data_stats.json。
**警告**：NaN 率 > 30% 时自动生成警告。

**独立调用**：此工具不内嵌于 full 流程，可在预处理完成后单独调用。`,

  params: {
    dataset_root: {
      type: 'string',
      description: '数据集根目录（包含 train/valid/test 子目录和 var_names.json）'
    },
    splits: {
      type: 'array',
      items: { type: 'string' },
      description: '要统计的 split 列表（默认: train, valid, test）',
      required: false,
      default: ['train', 'valid', 'test']
    },
    max_files: {
      type: 'number',
      description: '每变量最大采样文件数，控制运行时长（默认: 200）',
      required: false,
      default: 200
    }
  },

  attributes: {
    readonly: true,
    noEffect: true
  },

  async exec(args, ctx) {
    const {
      dataset_root,
      splits = ['train', 'valid', 'test'],
      max_files = 200
    } = args

    // 1. 检查 Python 环境
    const pythonPath = findFirstPythonPath()
    if (!pythonPath) {
      throw new Error('未找到可用的Python解释器，请安装Python或配置PYTHON/PYENV')
    }

    // 2. 准备路径
    const pythonCmd = `"${shellEscapeDouble(pythonPath)}"`
    const scriptPath = path.resolve(
      process.cwd(),
      'scripts/ocean-forecast-data-preprocess/forecast_stats.py'
    )
    const tempDir = path.resolve(dataset_root, '.ocean_forecast_temp')
    const configPath = path.join(tempDir, 'stats_config.json')
    const outputPath = path.join(tempDir, 'stats_result.json')

    // 3. 写入配置
    const config = {
      dataset_root,
      splits,
      max_files
    }
    await ctx.sandbox.exec(`mkdir -p "${shellEscapeDouble(tempDir)}"`)
    await ctx.sandbox.fs.write(configPath, JSON.stringify(config, null, 2))

    // 4. 执行 Python 脚本
    const result = await ctx.sandbox.exec(
      `${pythonCmd} "${shellEscapeDouble(scriptPath)}" --config "${shellEscapeDouble(configPath)}" --output "${shellEscapeDouble(outputPath)}"`,
      { timeoutMs: 300000 }
    )

    if (result.code !== 0) {
      throw new Error(`Python执行失败: ${result.stderr}`)
    }

    // 5. 读取结果
    const jsonContent = await ctx.sandbox.fs.read(outputPath)
    const statsResult: ForecastStatsResult = JSON.parse(jsonContent)

    if (statsResult.status === 'error') {
      throw new Error(`统计计算失败: ${statsResult.errors.join('; ')}`)
    }

    // 6. 统计总变量数
    let totalVars = 0
    for (const splitStats of Object.values(statsResult.stats || {})) {
      totalVars += Object.keys(splitStats).length
    }

    return {
      ...statsResult,
      data_stats_path: path.join(dataset_root, 'data_stats.json'),
      message: `统计完成，共计算 ${totalVars} 个变量，结果已写入 data_stats.json${statsResult.warnings.length > 0 ? `，有 ${statsResult.warnings.length} 条警告` : ''}`
    }
  }
})
