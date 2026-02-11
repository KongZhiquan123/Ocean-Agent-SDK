/**
 * @file visualize.ts
 *
 * @description 海洋超分辨率训练可视化工具
 * @author kongzhiquan
 * @contributors Leizheng
 * @date 2026-02-07
 * @version 1.1.0
 *
 * @changelog
 *   - 2026-02-11 Leizheng: v1.1.0 新增 predict 模式可视化
 *     - 添加 mode / dataset_root / dyn_vars / max_samples 参数
 *     - predict 模式调用 generate_predict_plots.py 生成 SR 对比图
 *   - 2026-02-07 kongzhiquan: v1.0.0 初始版本
 *     - 调用 Python 脚本生成训练可视化图表
 *     - 支持 loss 曲线、指标曲线、学习率曲线、指标对比、训练总结
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findPythonWithModule, findFirstPythonPath } from '@/utils/python-manager'
import path from 'node:path'

/**
 * 转义字符串使其可以安全嵌入 shell 双引号
 */
function shellEscapeDouble(str: string): string {
  return str.replace(/[\\"$`!]/g, '\\$&')
}

export const oceanSrVisualizeTool = defineTool({
  name: 'ocean_sr_visualize',
  description: `生成海洋超分辨率训练可视化图表

**训练模式 (mode=train, 默认)**：
从训练日志目录读取结构化日志，生成以下图表：
1. **loss_curve.png** - 训练/验证损失曲线，标注最佳 epoch
2. **metrics_curve.png** - MSE/RMSE/PSNR/SSIM 四个指标的变化曲线
3. **lr_curve.png** - 学习率变化曲线
4. **metrics_comparison.png** - 验证集与测试集指标对比柱状图
5. **training_summary.png** - 训练总结表格（模型、参数、时长、最终指标）
6. **sample_comparison.png** - 测试样本 LR/SR/HR 对比

**预测模式 (mode=predict)**：
从 predictions/ 目录读取 SR NPY 文件，对比原始 HR/LR 数据生成：
- **predict_comparison_XX.png** - 每个样本的 LR/SR/HR/Error 四面板对比图

**输出目录**：固定为 log_dir/plots/

**使用场景**：
- 训练完成后生成可视化报告
- predict 完成后可视化 SR 输出质量
- 分析训练过程中的收敛情况
- 对比验证集和测试集性能`,

  params: {
    log_dir: {
      type: 'string',
      description: '训练日志目录（包含 train.log 或 predictions/）'
    },
    mode: {
      type: 'string',
      description: '可视化模式: "train"（默认）或 "predict"',
      required: false,
      default: 'train'
    },
    dataset_root: {
      type: 'string',
      description: '预处理数据根目录（predict 模式需要，用于加载 HR/LR 对比数据）',
      required: false
    },
    dyn_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '动态变量列表（predict 模式需要，如 ["temp", "salt"]）',
      required: false
    },
    max_samples: {
      type: 'number',
      description: '最多可视化样本数（predict 模式，默认 4）',
      required: false,
      default: 4
    }
  },

  async exec(args, ctx) {
    const { log_dir, mode = 'train' } = args

    // 1. 检查 Python 环境（需要 matplotlib）
    const pythonPath = findPythonWithModule('matplotlib') || findFirstPythonPath()
    if (!pythonPath) {
      throw new Error('未找到可用的 Python 解释器（需要安装 matplotlib）')
    }

    const plotsDir = path.join(log_dir, 'plots')
    let cmd: string

    if (mode === 'predict') {
      // predict 模式：调用 generate_predict_plots.py
      if (!args.dataset_root) {
        throw new Error('predict 模式需要 dataset_root 参数')
      }
      if (!args.dyn_vars || !Array.isArray(args.dyn_vars) || args.dyn_vars.length === 0) {
        throw new Error('predict 模式需要 dyn_vars 参数（动态变量列表）')
      }
      const scriptPath = path.resolve(
        process.cwd(),
        'scripts/ocean_SR_training_masked/generate_predict_plots.py'
      )
      const datasetRoot = path.resolve(ctx.sandbox.workDir, args.dataset_root)
      const dynVarsStr = (args.dyn_vars as string[]).join(',')
      const maxSamples = args.max_samples ?? 4
      cmd = `"${shellEscapeDouble(pythonPath)}" "${shellEscapeDouble(scriptPath)}" --log_dir "${shellEscapeDouble(log_dir)}" --dataset_root "${shellEscapeDouble(datasetRoot)}" --dyn_vars "${shellEscapeDouble(dynVarsStr)}" --max_samples ${Number(maxSamples) || 4}`
    } else {
      // train 模式：调用 generate_training_plots.py
      const scriptPath = path.resolve(
        process.cwd(),
        'scripts/ocean_SR_training_masked/generate_training_plots.py'
      )
      cmd = `"${shellEscapeDouble(pythonPath)}" "${shellEscapeDouble(scriptPath)}" --log_dir "${shellEscapeDouble(log_dir)}"`
    }

    // 执行 Python 脚本
    const result = await ctx.sandbox.exec(cmd, { timeoutMs: 180000 })

    if (result.code !== 0) {
      throw new Error(`Python 执行失败: ${result.stderr}`)
    }

    // 解析结果
    const resultMatch = result.stdout.match(/__result__(\{.*?\})__result__/)
    let plots: string[] = []

    if (resultMatch) {
      try {
        const parsed = JSON.parse(resultMatch[1])
        plots = parsed.plots || []
      } catch {
        // 解析失败，使用默认值
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
