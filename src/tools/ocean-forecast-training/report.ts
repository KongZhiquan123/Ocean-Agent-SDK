/**
 * @file report.ts
 *
 * @description Ocean forecast training report generation tool
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

export const oceanForecastReportTool = defineTool({
  name: 'ocean_forecast_train_report',
  description: `生成海洋时序预测训练 Markdown 报告

从训练日志目录读取 config.yaml 和 train.log（结构化日志），自动生成综合报告。

**报告内容**：
1. 执行摘要（主要成果、关键指标）
2. 训练配置（模型结构、时序参数 in_t/out_t/stride、超参数）
3. 训练过程（RMSE/MAE 性能演进、per-variable 指标）
4. 最终性能评估（验证集/测试集指标）
5. 可视化结果
6. 模型检查点
7. 训练分析

**Agent 的职责**：
1. 调用此工具生成初始报告
2. 读取生成的报告文件
3. 分析报告中的数据
4. 替换 <!-- AI_FILL: ... --> 占位符
5. 保存最终报告

**输出**：log_dir/training_report.md`,

  params: {
    log_dir: {
      type: 'string',
      description: '训练日志目录（包含 config.yaml 和 train.log）'
    },
    output_path: {
      type: 'string',
      description: '报告输出路径（默认: log_dir/training_report.md）',
      required: false
    }
  },

  async exec(args, ctx) {
    const { log_dir, output_path } = args

    const pythonPath = findPythonWithModule('yaml') || findFirstPythonPath()
    if (!pythonPath) {
      throw new Error('未找到可用的 Python 解释器（需要安装 pyyaml）')
    }

    const scriptPath = path.resolve(
      process.cwd(),
      'scripts/ocean-forecast-training/generate_training_report.py'
    )
    const reportPath = output_path || path.join(log_dir, 'training_report.md')

    let cmd = `"${pythonPath}" "${scriptPath}" --log_dir "${log_dir}"`
    if (output_path) {
      cmd += ` --output "${output_path}"`
    }

    const result = await ctx.sandbox.exec(cmd, { timeoutMs: 120000 })

    if (result.code !== 0) {
      throw new Error(`Python 执行失败: ${result.stderr}`)
    }

    return {
      status: 'success' as const,
      report_path: reportPath,
      message: `训练报告已生成: ${reportPath}，请读取报告并补充 <!-- AI_FILL: ... --> 占位符部分的分析内容。`
    }
  }
})
