/**
 * @file list-models.ts
 *
 * @description 列出所有可用的超分辨率模型
 * @author Leizheng
 * @date 2026-02-06
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-06 Leizheng: v1.0.0 初始版本
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findPythonWithModule, findFirstPythonPath } from '@/utils/python-manager'
import path from 'node:path'

export const oceanSrListModelsTool = defineTool({
  name: 'ocean_sr_list_models',
  description: `列出所有可用的超分辨率模型。

返回模型名称、类别（standard/diffusion）、对应的 Trainer 类型和说明。
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

    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean_SR_training_masked/list_models.py')
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
