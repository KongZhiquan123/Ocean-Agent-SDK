/**
 * @file check-gpu.ts
 *
 * @description 查看可用 GPU 信息工具
 * @author Leizheng
 * @date 2026-02-06
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-06 Leizheng: v1.0.0 初始版本
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findFirstPythonPath } from '@/utils/python-manager'
import path from 'node:path'

export const oceanSrCheckGpuTool = defineTool({
  name: 'ocean_sr_check_gpu',
  description: `查看当前可用的 GPU 信息。

返回每张 GPU 的名称、总显存、空闲显存、已用显存。
用于训练前确认 GPU 资源，帮助用户选择使用哪些卡。`,

  params: {},

  attributes: {
    readonly: true,
    noEffect: true
  },

  async exec(_args, ctx) {
    const pythonPath = findFirstPythonPath()
    if (!pythonPath) {
      throw new Error('未找到可用的 Python 解释器')
    }

    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean_SR_training_masked/check_gpu.py')
    const result = await ctx.sandbox.exec(
      `"${pythonPath}" "${scriptPath}"`,
      { timeoutMs: 30000 }
    )

    if (result.code !== 0) {
      throw new Error(`GPU 检测失败: ${result.stderr}`)
    }

    return JSON.parse(result.stdout)
  }
})
