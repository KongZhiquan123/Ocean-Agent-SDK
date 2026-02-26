/**
 * @file metrics-tool.ts
 * @description 海洋数据质量指标检测工具 - 计算 HR vs LR 的质量指标
 *
 * @author leizheng
 * @contributors kongzhiquan
 * @date 2026-02-03
 * @version 1.2.0
 *
 * @changelog
 *   - 2026-02-25 kongzhiquan: v1.3.0 tempDir 改为基于 dataset_root 的 .ocean_preprocess_temp
 *   - 2026-02-05 kongzhiquan: v1.2.0 合并重构与沙盒修复
 *     - 移除 try-catch，统一由上层处理错误
 *     - 删除无用参数 output（改为固定输出到 ./ocean_preprocess_temp/metrics_result.json）
 *     - 错误时直接 throw Error 而非返回 status: 'error'
 *   - 2026-02-03 leizheng: v1.0.0 初始版本
 *     - 调用 metrics.py 计算质量指标
 *     - 支持 SSIM、Relative L2、MSE、RMSE
 *     - LR 临时上采样到 HR 尺寸进行比较
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findFirstPythonPath } from '@/utils/python-manager'
import path from 'node:path'

function shellEscapeDouble(str: string): string {
  return str.replace(/[\\"$`!]/g, '\\$&')
}

export interface MetricsResult {
  status: 'success' | 'error'
  config: {
    dataset_root: string
    scale: number
    timestamp: string
  }
  splits: Record<string, Record<string, {
    ssim: number
    relative_l2: number
    mse: number
    rmse: number
  }>>
  errors?: string[]
  message?: string
}

export const oceanMetricsTool = defineTool({
  name: 'ocean_metrics',
  description: `计算下采样数据质量指标

将 LR 数据临时上采样到 HR 尺寸，然后计算以下指标：
- SSIM: 结构相似性 (0~1, 越接近 1 越好)
- Relative L2: 相对 L2 误差 (越小越好, HR 作为分母)
- MSE: 均方误差
- RMSE: 均方根误差

**注意**：HR 作为基准数据，在计算 Relative L2 时作为分母。

**输出**：
- dataset_root/metrics_result.json`,

  params: {
    dataset_root: {
      type: 'string',
      description: '数据集根目录（包含 train/valid/test 子目录）'
    },
    scale: {
      type: 'number',
      description: '下采样倍数（用于验证）'
    },
    splits: {
      type: 'array',
      items: { type: 'string' },
      description: '要检查的数据集划分（默认: train, valid, test）',
      required: false,
      default: ['train', 'valid', 'test']
    }
  },

  attributes: {
    readonly: true,
    noEffect: true
  },

  async exec(args, ctx) {
    const {
      dataset_root,
      scale,
      splits = ['train', 'valid', 'test']
    } = args

    // 1. 检查 Python 环境
    const pythonPath = findFirstPythonPath()
    if (!pythonPath) {
      throw new Error('未找到可用的Python解释器')
    }

    // 2. 准备路径
    const pythonCmd = `"${shellEscapeDouble(pythonPath)}"`
    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean_preprocess/metrics.py')
    const tempDir = path.resolve(dataset_root, '.ocean_preprocess_temp')
    const outputPath = path.join(tempDir, 'metrics_result.json')

    // 3. 构建命令
    const splitsArg = splits.map((split) => `"${shellEscapeDouble(String(split))}"`).join(' ')
    const cmd = `${pythonCmd} "${shellEscapeDouble(scriptPath)}" --dataset_root "${shellEscapeDouble(dataset_root)}" --scale ${scale} --splits ${splitsArg} --output "${shellEscapeDouble(outputPath)}"`

    // 4. 执行 Python 脚本
    const result = await ctx.sandbox.exec(cmd, { timeoutMs: 600000 })

    if (result.code !== 0) {
      throw new Error(`Python执行失败: ${result.stderr}`)
    }

    // 5. 读取结果
    const jsonContent = await ctx.sandbox.fs.read(outputPath)
    const metricsResult: MetricsResult = JSON.parse(jsonContent)
    if (metricsResult.status === 'error') {
      throw new Error(`指标检测失败: ${metricsResult.errors?.join('; ') || '未知错误'}`)
    }
    // 统计变量数
    let totalVars = 0
    for (const splitResult of Object.values(metricsResult.splits || {})) {
      totalVars += Object.keys(splitResult).length
    }

    return {
      status: 'success',
      ...metricsResult,
      message: `指标检测完成，共检测 ${totalVars} 个变量，请调用ocean_generate_report工具生成预处理报告。`
    }
  }
})
