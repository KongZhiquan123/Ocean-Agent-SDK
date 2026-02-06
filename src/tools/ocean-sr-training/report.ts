/**
 * @file report.ts
 *
 * @description 海洋超分辨率训练报告生成工具
 * @author Leizheng
 * @date 2026-02-06
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-06 Leizheng: v1.0.0 初始版本
 *     - 调用 Python 脚本生成训练报告 Markdown
 *     - 支持 4 阶段用户确认信息记录
 *     - 报告包含 Agent 分析占位符
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findFirstPythonPath } from '@/utils/python-manager'
import path from 'node:path'

export const oceanSrReportTool = defineTool({
  name: 'ocean_sr_generate_report',
  description: `生成海洋超分辨率训练 Markdown 报告

从训练日志目录读取配置和日志，生成包含以下内容的综合报告：

**报告内容**：
1. 训练概览（模型名、数据集、训练参数、GPU 信息）
2. 用户确认记录（4 阶段确认信息）
3. 训练过程（loss、早停信息、最佳 epoch）
4. 评估结果（MSE/RMSE/PSNR/SSIM - 仅海洋格点）
5. 掩码信息（陆地像素占比、有效像素统计）
6. **分析和建议（占位符）** - 需要 Agent 自行填写
7. 总结

**Agent 的职责**：
1. 调用此工具生成初始报告（传入 user_confirmation）
2. 读取生成的报告文件
3. 仔细分析报告中的所有数据
4. 编写专业的分析和建议，替换占位符
5. 保存最终报告

**输出**：
- log_dir/training_report.md`,

  params: {
    log_dir: {
      type: 'string',
      description: '训练日志目录（包含 config.yaml 和 train.log）'
    },
    user_confirmation: {
      type: 'object',
      description: '用户确认信息（4 阶段），包含 stage1_data、stage2_model、stage3_parameters、stage4_execution',
      required: false
    },
    output_path: {
      type: 'string',
      description: '报告输出路径（默认: log_dir/training_report.md）',
      required: false
    }
  },

  async exec(args, ctx) {
    const { log_dir, user_confirmation, output_path } = args

    // 1. 检查 Python 环境
    const pythonPath = findFirstPythonPath()
    if (!pythonPath) {
      throw new Error('未找到可用的 Python 解释器')
    }

    // 2. 准备路径
    const scriptPath = path.resolve(
      process.cwd(),
      'scripts/ocean_SR_training_masked/generate_training_report.py'
    )
    const reportPath =
      output_path || path.join(log_dir, 'training_report.md')

    // 3. 准备配置
    const tempDir = path.resolve(ctx.sandbox.workDir, 'ocean_sr_temp')
    const configPath = path.join(tempDir, 'report_config.json')

    const config = {
      log_dir,
      user_confirmation: user_confirmation || {},
      output_path: reportPath
    }

    // 4. 写入配置
    await ctx.sandbox.exec(`mkdir -p "${tempDir}"`)
    await ctx.sandbox.fs.write(configPath, JSON.stringify(config, null, 2))

    // 5. 执行 Python 脚本
    const result = await ctx.sandbox.exec(
      `"${pythonPath}" "${scriptPath}" --config "${configPath}"`,
      { timeoutMs: 120000 }
    )

    if (result.code !== 0) {
      throw new Error(`Python 执行失败: ${result.stderr}`)
    }

    return {
      status: 'success' as const,
      report_path: reportPath,
      message: `训练报告已生成: ${reportPath}，请勿再手写一份新的报告，直接使用此报告并补充分析部分即可。`
    }
  }
})
