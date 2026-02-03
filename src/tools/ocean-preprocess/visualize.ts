/**
 * @file visualize.ts
 * @description 海洋数据可视化检查工具 - 生成 HR vs LR 对比图
 *
 * @author leizheng
 * @date 2026-02-03
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-03 leizheng: v1.0.0 初始版本
 *     - 调用 visualize_check.py 生成对比图
 *     - 每个变量抽取 1 帧进行检查
 *     - 支持 2D/3D/4D 数据格式
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findFirstPythonPath } from '@/utils/python-manager'
import path from 'node:path'

export interface VisualizeResult {
  status: 'success' | 'error'
  dataset_root: string
  output_dir: string
  splits: string[]
  generated_files?: string[]
  errors?: string[]
  message?: string
}

export const oceanVisualizeTool = defineTool({
  name: 'ocean_visualize',
  description: `生成 HR vs LR 对比可视化图片

从 train/hr/ 和 train/lr/ 目录读取数据，生成对比图保存到 visualisation_data_process/ 目录。

**特性**：
- 每个变量抽取 1 帧进行检查（取中间时间步）
- 支持 2D/3D/4D 数据格式
- NaN 区域显示为灰色背景
- 显示数据 shape 和切片信息

**输出目录结构**：
- dataset_root/visualisation_data_process/train/*.png
- dataset_root/visualisation_data_process/valid/*.png
- dataset_root/visualisation_data_process/test/*.png`,

  params: {
    dataset_root: {
      type: 'string',
      description: '数据集根目录（包含 train/valid/test 子目录）'
    },
    splits: {
      type: 'array',
      items: { type: 'string' },
      description: '要检查的数据集划分（默认: train, valid, test）',
      required: false,
      default: ['train', 'valid', 'test']
    },
    out_dir: {
      type: 'string',
      description: '输出目录（默认: dataset_root/visualisation_data_process/）',
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

    ctx.emit('visualize_started', {
      dataset_root,
      splits
    })

    // 1. 检查 Python 环境
    const pythonPath = findFirstPythonPath()
    if (!pythonPath) {
      const errorMsg = '未找到可用的Python解释器'
      ctx.emit('visualize_failed', { error: errorMsg })
      return {
        status: 'error',
        errors: [errorMsg],
        message: '可视化失败'
      }
    }

    // 2. 准备路径
    const pythonCmd = `"${pythonPath}"`
    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean_preprocess/visualize_check.py')
    const outputDir = out_dir || path.join(dataset_root, 'visualisation_data_process')

    // 3. 构建命令
    const splitsArg = splits.join(' ')
    const cmd = `${pythonCmd} "${scriptPath}" --dataset_root "${dataset_root}" --splits ${splitsArg} --out_dir "${outputDir}"`

    try {
      // 4. 执行 Python 脚本
      const result = await ctx.sandbox.exec(cmd, { timeoutMs: 300000 })

      if (result.code !== 0) {
        ctx.emit('visualize_failed', { error: result.stderr })
        return {
          status: 'error',
          errors: [`Python执行失败: ${result.stderr}`],
          message: '可视化失败'
        }
      }

      // 5. 列出生成的文件
      const generatedFiles: string[] = []
      for (const split of splits) {
        const splitDir = path.join(outputDir, split)
        try {
          const lsResult = await ctx.sandbox.exec(`ls "${splitDir}"/*.png 2>/dev/null || true`)
          if (lsResult.stdout.trim()) {
            const files = lsResult.stdout.trim().split('\n')
            generatedFiles.push(...files)
          }
        } catch {
          // 目录可能不存在
        }
      }

      ctx.emit('visualize_completed', {
        dataset_root,
        output_dir: outputDir,
        generated_files: generatedFiles.length
      })

      return {
        status: 'success',
        dataset_root,
        output_dir: outputDir,
        splits,
        generated_files: generatedFiles,
        message: `可视化完成，生成 ${generatedFiles.length} 张图片`
      }

    } catch (error: any) {
      ctx.emit('visualize_failed', { error: error.message })
      return {
        status: 'error',
        errors: [error.message],
        message: '可视化执行异常'
      }
    }
  }
})
