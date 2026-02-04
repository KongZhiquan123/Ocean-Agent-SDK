/**
 * @file report.ts
 * @description 海洋数据预处理报告生成工具 - 生成包含分析的 Markdown 报告
 *
 * @author kongzhiquan
 * @contributors leizheng
 * @date 2026-02-04
 * @version 3.1.0
 *
 * @changelog
 *   - 2026-02-04 kongzhiquan: v3.1.0 合并 manifest_path 和 user_confirmation 功能
 *     - 新增 manifest_path 参数，用于读取 preprocess_manifest.json
 *     - 新增 user_confirmation 参数记录 4 阶段确认信息
 *     - convert_result_path 现在读取 convert_result.json（完整结果）
 *     - 更新报告章节结构（新增 Section 2 用户确认记录）
 *     - 新增全局统计汇总图展示 (statistics_summary.png)
 *     - 分离展示空间对比图 (_compare.png) 和统计分布图 (_statistics.png)
 *     - Python脚本同时使用两个文件的数据生成更完整的报告
 *   - 2026-02-04 kongzhiquan: v1.0.0 初始版本
 *     - 整合预处理流程中的所有关键信息
 *     - 生成包含可视化图片的 Markdown 报告
 *     - 添加 AI 分析和建议
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findFirstPythonPath } from '@/utils/python-manager'
import path from 'node:path'

export interface UserConfirmation {
  stage1_research_vars?: {
    selected: string[]
    confirmed_at?: string
  }
  stage2_static_mask?: {
    static_vars: string[]
    mask_vars: string[]
    coord_vars?: {
      lon?: string
      lat?: string
    }
    confirmed_at?: string
  }
  stage3_parameters?: {
    scale?: number
    downsample_method?: string
    train_ratio?: number
    valid_ratio?: number
    test_ratio?: number
    h_slice?: string
    w_slice?: string
    crop_recommendation?: string
    confirmed_at?: string
  }
  stage4_execution?: {
    confirmed: boolean
    confirmed_at?: string
    execution_started_at?: string
  }
}

export interface ReportResult {
  status: 'success' | 'error'
  report_path: string
  errors?: string[]
  message?: string
}

export const oceanReportTool = defineTool({
  name: 'ocean_generate_report',
  description: `生成海洋数据预处理 Markdown 报告

从预处理流程的各个步骤中提取关键信息，生成一份包含以下内容的综合报告：

**报告内容（v3.0.0）**：
1. 数据集概览（文件数、检测到的变量候选、形状信息）
2. **用户确认记录**（4 阶段强制确认的选择）
   - 阶段 1：研究变量选择
   - 阶段 2：静态/掩码变量选择
   - 阶段 3：处理参数确认（scale、裁剪、划分比例）
   - 阶段 4：执行确认
3. 验证结果（张量约定、后置验证 Rule 1/2/3）
4. 转换结果（输出文件结构、数据集划分）
5. 质量指标（SSIM、Relative L2、MSE、RMSE）
6. **可视化图片**（自动嵌入）
   - 全局统计汇总图 (statistics_summary.png)
   - 每个变量的空间对比图 ({var}_compare.png)
   - 每个变量的统计分布图 ({var}_statistics.png)
7. **分析和建议（占位符）** - 需要 Agent 自行填写
8. 总结

**重要提示**：
- 报告的第 7 节"分析和建议"会包含一个占位符注释
- Agent 必须在生成报告后，读取报告内容，替换占位符为实际分析
- 分析应基于报告中的所有数据（用户确认记录、质量指标、验证结果等）
- 分析应具体、有针对性，避免模板化内容

**Agent 的职责**：
1. 调用此工具生成初始报告（传入 user_confirmation）
2. 读取生成的报告文件
3. 仔细分析报告中的所有数据
4. 编写专业的分析和建议，替换占位符
5. 保存最终报告

**输出**：
- dataset_root/preprocessing_report.md`,

  params: {
    dataset_root: {
      type: 'string',
      description: '数据集根目录（包含所有预处理结果）'
    },
    user_confirmation: {
      type: 'object',
      description: '用户确认信息（4 阶段），包含 stage1_research_vars、stage2_static_mask、stage3_parameters、stage4_execution',
      required: false
    },
    inspect_result_path: {
      type: 'string',
      description: 'Step A 的 inspect_result.json 路径',
      required: false
    },
    validate_result_path: {
      type: 'string',
      description: 'Step B 的 validate_result.json 路径',
      required: false
    },
    convert_result_path: {
      type: 'string',
      description: 'Step C 的完整转换结果路径（convert_result.json，包含验证、warnings、errors）',
      required: false
    },
    manifest_path: {
      type: 'string',
      description: '预处理清单路径（preprocess_manifest.json，包含输入配置）',
      required: false
    },
    metrics_result_path: {
      type: 'string',
      description: '质量指标结果路径（metrics_result.json）',
      required: false
    },
    output_path: {
      type: 'string',
      description: '报告输出路径（默认: dataset_root/preprocessing_report.md）',
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
      user_confirmation,
      inspect_result_path,
      validate_result_path,
      convert_result_path,
      manifest_path,
      metrics_result_path,
      output_path
    } = args

    ctx.emit('report_started', { dataset_root })

    // 1. 检查 Python 环境
    const pythonPath = findFirstPythonPath()
    if (!pythonPath) {
      const errorMsg = '未找到可用的Python解释器'
      ctx.emit('report_failed', { error: errorMsg })
      return {
        status: 'error',
        errors: [errorMsg],
        message: '报告生成失败'
      }
    }

    // 2. 准备路径
    const pythonCmd = `"${pythonPath}"`
    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean_preprocess/generate_report.py')
    const reportPath = output_path || path.join(dataset_root, 'preprocessing_report.md')

    // 3. 准备配置
    const tempDir = path.resolve(ctx.sandbox.workDir, 'ocean_preprocess_temp')
    const configPath = path.join(tempDir, 'report_config.json')

    const config = {
      dataset_root,
      user_confirmation: user_confirmation || {},
      inspect_result_path: inspect_result_path || path.join(tempDir, 'inspect_result.json'),
      validate_result_path: validate_result_path || path.join(tempDir, 'validate_result.json'),
      convert_result_path: convert_result_path || path.join(tempDir, 'convert_result.json'),
      manifest_path: manifest_path || path.join(dataset_root, 'preprocess_manifest.json'),
      metrics_result_path: metrics_result_path || path.join(dataset_root, 'metrics_result.json'),
      output_path: reportPath
    }

    try {
      // 4. 写入配置
      await ctx.sandbox.fs.write(configPath, JSON.stringify(config, null, 2))

      // 5. 执行 Python 脚本
      const result = await ctx.sandbox.exec(
        `${pythonCmd} "${scriptPath}" --config "${configPath}"`,
        { timeoutMs: 120000 }
      )

      if (result.code !== 0) {
        ctx.emit('report_failed', { error: result.stderr })
        return {
          status: 'error',
          errors: [`Python执行失败: ${result.stderr}`],
          message: '报告生成失败'
        }
      }

      ctx.emit('report_completed', {
        report_path: reportPath
      })

      return {
        status: 'success',
        report_path: reportPath,
        message: `报告已生成: ${reportPath}`
      }

    } catch (error: any) {
      ctx.emit('report_failed', { error: error.message })
      return {
        status: 'error',
        errors: [error.message],
        message: '报告生成执行异常'
      }
    }
  }
})
