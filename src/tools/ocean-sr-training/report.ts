/**
 * @file report.ts
 *
 * @description 海洋超分辨率训练报告生成工具
 * @author kongzhiquan
 * @date 2026-02-06
 * @version 2.0.0
 *
 * @changelog
 *   - 2026-02-07 kongzhiquan: v2.0.0 适配结构化日志解析
 *     - 简化参数，移除 user_confirmation（从结构化日志自动提取）
 *     - 改用 --log_dir 命令行参数
 *     - 移除临时配置文件，直接传参
 *   - 2026-02-06 kongzhiquan: v1.0.0 初始版本
 *     - 调用 Python 脚本生成训练报告 Markdown
 *     - 支持 4 阶段用户确认信息记录
 *     - 报告包含 Agent 分析占位符
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findPythonWithModule, findFirstPythonPath } from '@/utils/python-manager'
import path from 'node:path'

export const oceanSrReportTool = defineTool({
  name: 'ocean_sr_generate_report',
  description: `生成海洋超分辨率训练 Markdown 报告

从训练日志目录读取 config.yaml 和 train.log（结构化日志），自动生成综合报告。

**报告内容**：
1. 执行摘要（主要成果、关键指标）
2. 训练配置（模型结构、数据配置、超参数、硬件配置）
3. 训练过程（时间线、训练曲线、验证集性能演进）
4. 最终性能评估（验证集/测试集指标）
5. 模型检查点
6. 训练分析（占位符 - 需要 Agent 填写）
7. 计算性能
8. 总结

**Agent 的职责**：
1. 调用此工具生成初始报告
2. 读取生成的报告文件
3. 仔细分析报告中的所有数据
4. 编写专业的分析和建议，替换 <!-- AI_FILL: ... --> 占位符
5. 保存最终报告

**输出**：
- log_dir/training_report.md`,

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

    // 1. 检查 Python 环境
    const pythonPath = findPythonWithModule('yaml') || findFirstPythonPath()
    if (!pythonPath) {
      throw new Error('未找到可用的 Python 解释器（需要安装 pyyaml）')
    }

    // 2. 准备路径
    const scriptPath = path.resolve(
      process.cwd(),
      'scripts/ocean_SR_training_masked/generate_training_report.py'
    )
    const reportPath = output_path || path.join(log_dir, 'training_report.md')

    // 3. 构建命令行参数
    let cmd = `"${pythonPath}" "${scriptPath}" --log_dir "${log_dir}"`
    if (output_path) {
      cmd += ` --output "${output_path}"`
    }

    // 4. 执行 Python 脚本
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
