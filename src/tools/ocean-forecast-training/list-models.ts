/**
 * @file list-models.ts
 *
 * @description List available ocean forecast models
 * @author Leizheng
 * @date 2026-02-26
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-26 Leizheng: v1.0.0 initial version
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findPythonWithModule, findFirstPythonPath } from '@/utils/python-manager'
import path from 'node:path'

export const oceanForecastListModelsTool = defineTool({
  name: 'ocean_forecast_list_models',
  description: `列出所有可用的海洋时序预测模型。

返回模型名称、类别（spectral/cnn/transformer/mlp/operator）、推荐标记和说明。
用于让用户了解和选择训练模型。`,

  params: {},

  attributes: {
    readonly: true,
    noEffect: true
  },

  async exec(_args, ctx) {
    const pythonPath = findPythonWithModule('torch') || findFirstPythonPath()
    if (!pythonPath) {
      throw new Error('未找到可用的 Python 解释器（需要安装 torch）')
    }

    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean-forecast-training/list_models.py')
    const result = await ctx.sandbox.exec(
      `"${pythonPath}" "${scriptPath}"`,
      { timeoutMs: 30000 }
    )

    if (result.code !== 0) {
      throw new Error(`模型列表获取失败: ${result.stderr}`)
    }

    return JSON.parse(result.stdout)
  }
})
