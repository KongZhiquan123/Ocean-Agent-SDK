/**
 * @file metrics-tool.ts
 * @description 海洋数据质量指标检测工具 - 计算 HR vs LR 的质量指标
 *
 * @author leizheng
 * @date 2026-02-03
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-03 leizheng: v1.0.0 初始版本
 *     - 调用 metrics.py 计算质量指标
 *     - 支持 SSIM、Relative L2、MSE、RMSE
 *     - LR 临时上采样到 HR 尺寸进行比较
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findFirstPythonPath } from '@/utils/python-manager'
import path from 'node:path'

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
    },
    output: {
      type: 'string',
      description: '输出文件路径（默认: dataset_root/metrics_result.json）',
      required: false
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
      splits = ['train', 'valid', 'test'],
      output
    } = args

    ctx.emit('metrics_started', {
      dataset_root,
      scale,
      splits
    })

    // 1. 检查 Python 环境
    const pythonPath = findFirstPythonPath()
    if (!pythonPath) {
      const errorMsg = '未找到可用的Python解释器'
      ctx.emit('metrics_failed', { error: errorMsg })
      return {
        status: 'error',
        errors: [errorMsg],
        message: '指标检测失败'
      }
    }

    // 2. 准备路径
    const pythonCmd = `"${pythonPath}"`
    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean_preprocess/metrics.py')
    const outputPath = output || path.join(dataset_root, 'metrics_result.json')

    // 3. 构建命令
    const splitsArg = splits.join(' ')
    const cmd = `${pythonCmd} "${scriptPath}" --dataset_root "${dataset_root}" --scale ${scale} --splits ${splitsArg} --output "${outputPath}"`

    try {
      // 4. 执行 Python 脚本
      const result = await ctx.sandbox.exec(cmd, { timeoutMs: 600000 })

      if (result.code !== 0) {
        ctx.emit('metrics_failed', { error: result.stderr })
        return {
          status: 'error',
          errors: [`Python执行失败: ${result.stderr}`],
          message: '指标检测失败'
        }
      }

      // 5. 读取结果
      const jsonContent = await ctx.sandbox.fs.read(outputPath)
      const metricsResult: MetricsResult = JSON.parse(jsonContent)

      // 统计变量数
      let totalVars = 0
      for (const splitResult of Object.values(metricsResult.splits || {})) {
        totalVars += Object.keys(splitResult).length
      }

      ctx.emit('metrics_completed', {
        dataset_root,
        scale,
        total_vars: totalVars,
        output_path: outputPath
      })

      return {
        status: 'success',
        ...metricsResult,
        message: `指标检测完成，共检测 ${totalVars} 个变量`
      }

    } catch (error: any) {
      ctx.emit('metrics_failed', { error: error.message })
      return {
        status: 'error',
        errors: [error.message],
        message: '指标检测执行异常'
      }
    }
  }
})
