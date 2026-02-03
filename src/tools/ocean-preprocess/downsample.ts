/**
 * @file downsample.ts
 * @description 海洋数据下采样工具 - 将 HR 数据下采样生成 LR 数据
 *
 * @author leizheng
 * @date 2026-02-03
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-03 leizheng: v1.0.0 初始版本
 *     - 调用 downsample.py 执行下采样
 *     - 支持多种插值方法（area, bicubic, nearest 等）
 *     - 支持 NaN 处理
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findFirstPythonPath } from '@/utils/python-manager'
import path from 'node:path'

export interface DownsampleResult {
  status: 'success' | 'error'
  dataset_root: string
  scale: number
  method: string
  splits: Record<string, any[]>
  static_variables: any[]
  timestamp: string
  errors?: string[]
  message?: string
}

export const oceanDownsampleTool = defineTool({
  name: 'ocean_downsample',
  description: `将 HR 数据下采样生成 LR 数据

从 train/hr/, valid/hr/, test/hr/ 目录读取高分辨率数据，
下采样后保存到对应的 train/lr/, valid/lr/, test/lr/ 目录。

**特性**：
- 支持多种插值方法：area（推荐）、bicubic、nearest、linear、lanczos
- 自动处理 NaN 值（先填 0，下采样后恢复 NaN）
- 支持 2D/3D/4D 数据格式

**输入目录结构**：
- dataset_root/train/hr/*.npy
- dataset_root/valid/hr/*.npy
- dataset_root/test/hr/*.npy

**输出目录结构**：
- dataset_root/train/lr/*.npy
- dataset_root/valid/lr/*.npy
- dataset_root/test/lr/*.npy`,

  params: {
    dataset_root: {
      type: 'string',
      description: '数据集根目录（包含 train/valid/test 子目录）'
    },
    scale: {
      type: 'number',
      description: '下采样倍数（如 4 表示尺寸缩小为 1/4）'
    },
    method: {
      type: 'string',
      description: '插值方法：area（推荐）、bicubic、nearest、linear、lanczos',
      required: false,
      default: 'area'
    },
    splits: {
      type: 'array',
      items: { type: 'string' },
      description: '要处理的数据集划分（默认: train, valid, test）',
      required: false,
      default: ['train', 'valid', 'test']
    },
    include_static: {
      type: 'boolean',
      description: '是否同时处理静态变量',
      required: false,
      default: false
    }
  },

  attributes: {
    readonly: false,
    noEffect: false
  },

  async exec(args, ctx) {
    const {
      dataset_root,
      scale,
      method = 'area',
      splits = ['train', 'valid', 'test'],
      include_static = false
    } = args

    ctx.emit('downsample_started', {
      dataset_root,
      scale,
      method,
      splits
    })

    // 1. 检查 Python 环境
    const pythonPath = findFirstPythonPath()
    if (!pythonPath) {
      const errorMsg = '未找到可用的Python解释器'
      ctx.emit('downsample_failed', { error: errorMsg })
      return {
        status: 'error',
        errors: [errorMsg],
        message: '下采样失败'
      }
    }

    // 2. 准备路径
    const pythonCmd = `"${pythonPath}"`
    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean_preprocess/downsample.py')
    const tempDir = path.resolve(ctx.sandbox.workDir, 'ocean_preprocess_temp')
    const outputPath = path.join(tempDir, 'downsample_result.json')

    // 3. 构建命令
    const splitsArg = splits.join(' ')
    const staticArg = include_static ? '--include_static' : ''
    const cmd = `${pythonCmd} "${scriptPath}" --dataset_root "${dataset_root}" --scale ${scale} --method ${method} --splits ${splitsArg} ${staticArg} --output "${outputPath}"`

    try {
      // 4. 创建临时目录
      await ctx.sandbox.exec(`mkdir -p "${tempDir}"`)

      // 5. 执行 Python 脚本
      const result = await ctx.sandbox.exec(cmd, { timeoutMs: 600000 })

      if (result.code !== 0) {
        ctx.emit('downsample_failed', { error: result.stderr })
        return {
          status: 'error',
          errors: [`Python执行失败: ${result.stderr}`],
          message: '下采样失败'
        }
      }

      // 6. 读取结果
      const jsonContent = await ctx.sandbox.fs.read(outputPath)
      const downsampleResult: DownsampleResult = JSON.parse(jsonContent)

      // 统计处理的文件数
      let totalFiles = 0
      for (const splitResult of Object.values(downsampleResult.splits || {})) {
        totalFiles += (splitResult as any[]).length
      }
      totalFiles += (downsampleResult.static_variables || []).length

      ctx.emit('downsample_completed', {
        dataset_root,
        scale,
        method,
        total_files: totalFiles
      })

      return {
        status: 'success',
        ...downsampleResult,
        message: `下采样完成，共处理 ${totalFiles} 个文件`
      }

    } catch (error: any) {
      ctx.emit('downsample_failed', { error: error.message })
      return {
        status: 'error',
        errors: [error.message],
        message: '下采样执行异常'
      }
    }
  }
})
