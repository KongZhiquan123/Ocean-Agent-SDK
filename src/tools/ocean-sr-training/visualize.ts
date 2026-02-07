/**
 * @file visualize.ts
 *
 * @description 海洋超分辨率训练可视化工具
 * @author kongzhiquan
 * @date 2026-02-07
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-07 kongzhiquan: v1.0.0 初始版本
 *     - 调用 Python 脚本生成训练可视化图表
 *     - 支持 loss 曲线、指标曲线、学习率曲线、指标对比、训练总结
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findPythonWithModule, findFirstPythonPath } from '@/utils/python-manager'
import path from 'node:path'

export const oceanSrVisualizeTool = defineTool({
  name: 'ocean_sr_visualize',
  description: `生成海洋超分辨率训练可视化图表

从训练日志目录读取结构化日志，生成以下图表：

**生成的图表**：
1. **loss_curve.png** - 训练/验证损失曲线，标注最佳 epoch
2. **metrics_curve.png** - MSE/RMSE/PSNR/SSIM 四个指标的变化曲线
3. **lr_curve.png** - 学习率变化曲线
4. **metrics_comparison.png** - 验证集与测试集指标对比柱状图
5. **training_summary.png** - 训练总结表格（模型、参数、时长、最终指标）

**输出目录**：
- 默认: log_dir/plots/

**使用场景**：
- 训练完成后生成可视化报告
- 分析训练过程中的收敛情况
- 对比验证集和测试集性能`,

  params: {
    log_dir: {
      type: 'string',
      description: '训练日志目录（包含 train.log）'
    },
    output_dir: {
      type: 'string',
      description: '图表输出目录（默认: log_dir/plots）',
      required: false
    }
  },

  async exec(args, ctx) {
    const { log_dir, output_dir } = args

    // 1. 检查 Python 环境（需要 matplotlib）
    const pythonPath = findPythonWithModule('matplotlib') || findFirstPythonPath()
    if (!pythonPath) {
      throw new Error('未找到可用的 Python 解释器（需要安装 matplotlib）')
    }

    // 2. 准备路径
    const scriptPath = path.resolve(
      process.cwd(),
      'scripts/ocean_SR_training_masked/generate_training_plots.py'
    )
    const plotsDir = output_dir || path.join(log_dir, 'plots')

    // 3. 构建命令行参数
    let cmd = `"${pythonPath}" "${scriptPath}" --log_dir "${log_dir}"`
    if (output_dir) {
      cmd += ` --output_dir "${output_dir}"`
    }

    // 4. 执行 Python 脚本
    const result = await ctx.sandbox.exec(cmd, { timeoutMs: 180000 })

    if (result.code !== 0) {
      throw new Error(`Python 执行失败: ${result.stderr}`)
    }

    // 5. 解析结果
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
      message: `已生成 ${plots.length} 个可视化图表，保存在: ${plotsDir}`
    }
  }
})
