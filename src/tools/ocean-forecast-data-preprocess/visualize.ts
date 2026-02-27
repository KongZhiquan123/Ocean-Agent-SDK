/**
 * @file visualize.ts
 * @description 海洋预报数据可视化工具 - 生成样本帧和时序统计图
 *
 * @author Leizheng
 * @date 2026-02-25
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-25 Leizheng: v1.0.0 初始版本
 *     - 调用 forecast_visualize.py 生成可视化图片
 *     - 生成样本帧空间分布图（{var}_frames.png）
 *     - 生成时序统计图（{var}_timeseries.png）
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findFirstPythonPath } from '@/utils/python-manager'
import path from 'node:path'

function shellEscapeDouble(str: string): string {
  return str.replace(/[\\"$`!]/g, '\\$&')
}

export interface ForecastVisualizeResult {
  status: 'success' | 'error'
  dataset_root: string
  out_dir: string
  splits: string[]
  generated_files?: string[]
  warnings?: string[]
  message?: string
}

export const oceanForecastPreprocessVisualizeTool = defineTool({
  name: 'ocean_forecast_preprocess_visualize',
  description: `生成海洋预报数据可视化图片

从预报数据集目录读取 NPY 文件，为每个变量生成：

**图片类型**：
1. **{var}_frames.png** - 样本帧空间分布图
   - 从时间轴均匀采样 4 帧
   - 显示真实经纬度坐标（如果有 static_variables/）
   - NaN 区域（陆地）显示为背景色

2. **{var}_timeseries.png** - 时序统计图
   - 空间均值随时间变化曲线
   - 空间标准差随时间变化曲线

3. **{var}_distribution.png** - 数值分布直方图
   - 均匀采样部分帧的像素值
   - 标注 P5/P95 分位数范围

**输出目录结构**：
- dataset_root/visualisation_forecast/train/{var}_frames.png
- dataset_root/visualisation_forecast/train/{var}_timeseries.png
- dataset_root/visualisation_forecast/train/{var}_distribution.png
- dataset_root/visualisation_forecast/valid/...
- dataset_root/visualisation_forecast/test/...`,

  params: {
    dataset_root: {
      type: 'string',
      description: '数据集根目录（包含 train/valid/test 子目录）'
    },
    splits: {
      type: 'array',
      items: { type: 'string' },
      description: '要可视化的数据集划分（默认: train, valid, test）',
      required: false,
      default: ['train', 'valid', 'test']
    },
    out_dir: {
      type: 'string',
      description: '输出目录（默认: dataset_root/visualisation_forecast/）',
      required: false
    }
  },

  attributes: {
    readonly: false,
    noEffect: false
  },

  async exec(args, ctx) {
    const {
      dataset_root,
      splits = ['train', 'valid', 'test'],
      out_dir
    } = args

    const pythonPath = findFirstPythonPath()
    if (!pythonPath) {
      throw new Error('未找到可用的Python解释器')
    }

    const pythonCmd = `"${shellEscapeDouble(pythonPath)}"`
    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean-forecast-data-preprocess/forecast_visualize.py')
    const outputDir = out_dir || path.join(dataset_root, 'visualisation_forecast')

    const splitsArg = (splits as string[]).map(s => `"${shellEscapeDouble(s)}"`).join(' ')
    const cmd = `${pythonCmd} "${shellEscapeDouble(scriptPath)}" --dataset_root "${shellEscapeDouble(dataset_root)}" --splits ${splitsArg} --out_dir "${shellEscapeDouble(outputDir)}"`

    const result = await ctx.sandbox.exec(cmd, { timeoutMs: 300000 })

    if (result.code !== 0) {
      throw new Error(`Python执行失败: ${result.stderr}`)
    }

    // 解析 stdout 中的 JSON 结果
    let pyResult: any = {}
    try {
      pyResult = JSON.parse(result.stdout.trim())
    } catch {
      // stdout 可能包含其他输出，尝试找最后一行 JSON
      const lines = result.stdout.trim().split('\n')
      for (let i = lines.length - 1; i >= 0; i--) {
        try {
          pyResult = JSON.parse(lines[i])
          break
        } catch { /* continue */ }
      }
    }

    return {
      status: 'success',
      dataset_root,
      out_dir: outputDir,
      splits: splits as string[],
      generated_files: pyResult.generated_files || [],
      warnings: pyResult.warnings || [],
      message: pyResult.message || `可视化完成，输出目录: ${outputDir}`
    } as ForecastVisualizeResult
  }
})
