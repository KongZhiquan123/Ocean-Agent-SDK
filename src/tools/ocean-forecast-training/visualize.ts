/**
 * @file visualize.ts
 *
 * @description Ocean forecast training and prediction visualization tool
 * @author Leizheng
 * @date 2026-02-26
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-26 Leizheng: v1.0.0 initial version for ocean forecast training
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findPythonWithModule, findFirstPythonPath } from '@/utils/python-manager'
import path from 'node:path'

function shellEscapeDouble(str: string): string {
  return str.replace(/[\\"$`!]/g, '\\$&')
}

export const oceanForecastVisualizeTool = defineTool({
  name: 'ocean_forecast_visualize',
  description: `生成海洋时序预测训练可视化图表

**训练模式 (mode=train, 默认)**：
从训练日志目录读取结构化日志，生成以下图表：
1. **loss_curve.png** - 训练/验证损失曲线
2. **metrics_curve.png** - RMSE/MAE 指标变化曲线
3. **lr_curve.png** - 学习率变化曲线
4. **per_var_metrics.png** - 各变量 RMSE 对比
5. **training_summary.png** - 训练总结

**预测模式 (mode=predict)**：
从 predictions/ 目录读取 NPY 文件，生成预测 vs 真值对比热力图。

**输出目录**：log_dir/plots/`,

  params: {
    log_dir: {
      type: 'string',
      description: '训练日志目录'
    },
    mode: {
      type: 'string',
      description: '可视化模式: "train"（默认）或 "predict"',
      required: false,
      default: 'train'
    },
    dataset_root: {
      type: 'string',
      description: '预处理数据根目录（predict 模式需要，用于加载真值对比数据）',
      required: false
    },
    n_samples: {
      type: 'number',
      description: '最多可视化样本数（predict 模式，默认 5）',
      required: false,
      default: 5
    }
  },

  async exec(args, ctx) {
    const { log_dir, mode = 'train' } = args

    const pythonPath = findPythonWithModule('matplotlib') || findFirstPythonPath()
    if (!pythonPath) {
      throw new Error('未找到可用的 Python 解释器（需要安装 matplotlib）')
    }

    const plotsDir = path.join(log_dir, 'plots')
    let cmd: string

    if (mode === 'predict') {
      const scriptPath = path.resolve(
        process.cwd(),
        'scripts/ocean-forecast-training/generate_predict_plots.py'
      )
      const nSamples = args.n_samples ?? 5
      cmd = `"${shellEscapeDouble(pythonPath)}" "${shellEscapeDouble(scriptPath)}" --log_dir "${shellEscapeDouble(log_dir)}"`
      if (args.dataset_root) {
        const datasetRoot = path.resolve(ctx.sandbox.workDir, args.dataset_root)
        cmd += ` --dataset_root "${shellEscapeDouble(datasetRoot)}"`
      }
      cmd += ` --n_samples ${Number(nSamples) || 5}`
    } else {
      const scriptPath = path.resolve(
        process.cwd(),
        'scripts/ocean-forecast-training/generate_training_plots.py'
      )
      cmd = `"${shellEscapeDouble(pythonPath)}" "${shellEscapeDouble(scriptPath)}" --log_dir "${shellEscapeDouble(log_dir)}"`
    }

    const result = await ctx.sandbox.exec(cmd, { timeoutMs: 180000 })

    if (result.code !== 0) {
      throw new Error(`Python 执行失败: ${result.stderr}`)
    }

    const resultMatch = result.stdout.match(/__result__(\{.*?\})__result__/)
    let plots: string[] = []

    if (resultMatch) {
      try {
        const parsed = JSON.parse(resultMatch[1])
        plots = parsed.plots || []
      } catch {
        // parse error, use defaults
      }
    }

    return {
      status: 'success' as const,
      output_dir: plotsDir,
      plots,
      mode,
      message: `已生成 ${plots.length} 个可视化图表，保存在: ${plotsDir}`
    }
  }
})
