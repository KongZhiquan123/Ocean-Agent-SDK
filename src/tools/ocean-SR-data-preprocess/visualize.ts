/**
 * @file visualize.ts
 * @description 海洋数据可视化检查工具 - 生成 HR vs LR 对比图和统计分布图
 *
 * @author leizheng
 * @contributors kongzhiquan
 * @date 2026-02-04
 * @version 3.1.0
 *
 * @changelog
 *   - 2026-02-05 kongzhiquan: v3.1.0 移除 try-catch，统一由上层处理错误
 *     - 错误时直接 throw Error 而非返回 status: 'error'
 *   - 2026-02-04 kongzhiquan: v3.0.0 新增统计分布图
 *     - 新增均值/方差时序图
 *     - 新增 HR/LR 数据值直方图对比
 *     - 新增全局统计汇总图 (statistics_summary.png)
 *     - 重命名原对比图为 {var}_compare.png
 *   - 2026-02-04 leizheng: v1.4.0 修复坐标文件匹配
 *     - 支持带编号前缀的文件名（如 20_latitude.npy）
 *     - 使用 glob 模式匹配多种命名格式
 *   - 2026-02-04 leizheng: v1.3.0 支持经纬度坐标显示
 *     - 自动加载 static_variables/ 中的 lat.npy/lon.npy
 *     - 坐标轴显示真实经纬度 (°E, °N)
 *   - 2026-02-04 leizheng: v1.2.0 添加坐标轴标签
 *   - 2026-02-03 leizheng: v1.0.0 初始版本
 *     - 调用 visualize_check.py 生成对比图
 *     - 每个变量抽取 1 帧进行检查
 *     - 支持 2D/3D/4D 数据格式
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findFirstPythonPath } from '@/utils/python-manager'
import path from 'node:path'

function shellEscapeDouble(str: string): string {
  return str.replace(/[\\"$`!]/g, '\\$&')
}

export interface VisualizeResult {
  status: 'success' | 'error'
  dataset_root: string
  output_dir: string
  splits: string[]
  generated_files?: string[]
  errors?: string[]
  message?: string
}

export const oceanSrPreprocessVisualizeTool = defineTool({
  name: 'ocean_sr_preprocess_visualize',
  description: `生成 HR vs LR 对比可视化图片和统计分布图

从 train/hr/ 和 train/lr/ 目录读取数据，生成对比图和统计图保存到 visualisation_data_process/ 目录。

**生成的图片类型（v3.0.0）**：
1. **{var}_compare.png** - HR vs LR 空间对比图
   - 取中间时间步进行对比
   - 显示真实经纬度坐标（如果有）
   - NaN 区域显示为灰色背景

2. **{var}_statistics.png** - 统计分布图（新增）
   - 均值随时间变化曲线（HR vs LR）
   - 标准差随时间变化曲线（HR vs LR）
   - HR 数据值直方图（含均值、中位数标注）
   - LR 数据值直方图（含均值、中位数标注）

3. **statistics_summary.png** - 全局统计汇总（新增）
   - 所有变量的均值对比条形图
   - 所有变量的标准差对比条形图

**特性**：
- 支持 2D/3D/4D 数据格式
- 自动加载 static_variables/ 中的经纬度坐标
- 坐标轴显示真实经纬度 (Longitude °E, Latitude °N)

**输出目录结构**：
- dataset_root/visualisation_data_process/train/{var}_compare.png
- dataset_root/visualisation_data_process/train/{var}_statistics.png
- dataset_root/visualisation_data_process/valid/...
- dataset_root/visualisation_data_process/test/...
- dataset_root/visualisation_data_process/statistics_summary.png`,

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

    // 1. 检查 Python 环境
    const pythonPath = findFirstPythonPath()
    if (!pythonPath) {
      throw new Error('未找到可用的Python解释器')
    }

    // 2. 准备路径
    const pythonCmd = `"${shellEscapeDouble(pythonPath)}"`
    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean-SR-data-preprocess/visualize_check.py')
    const outputDir = out_dir || path.join(dataset_root, 'visualisation_data_process')

    // 3. 构建命令
    const splitsArg = splits.map((split) => `"${shellEscapeDouble(String(split))}"`).join(' ')
    const cmd = `${pythonCmd} "${shellEscapeDouble(scriptPath)}" --dataset_root "${shellEscapeDouble(dataset_root)}" --splits ${splitsArg} --out_dir "${shellEscapeDouble(outputDir)}"`

    // 4. 执行 Python 脚本
    const result = await ctx.sandbox.exec(cmd, { timeoutMs: 300000 })

    if (result.code !== 0) {
      throw new Error(`Python执行失败: ${result.stderr}`)
    }

    // 5. 列出生成的文件
    const generatedFiles: string[] = []
    for (const split of splits) {
      const splitDir = path.join(outputDir, split)
      const lsResult = await ctx.sandbox.exec(`ls "${shellEscapeDouble(splitDir)}"/*.png 2>/dev/null || true`)
      if (lsResult.stdout.trim()) {
        const files = lsResult.stdout.trim().split('\n')
        generatedFiles.push(...files)
      }
    }

    return {
      status: 'success',
      dataset_root,
      output_dir: outputDir,
      splits,
      generated_files: generatedFiles,
      message: `可视化完成，生成 ${generatedFiles.length} 张图片（含对比图和统计分布图）`
    }
  }
})
