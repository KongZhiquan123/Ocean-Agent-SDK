/**
 * @file report.ts
 * @description 海洋数据预处理报告生成工具 - 生成包含分析的 Markdown 报告
 *
 * @author kongzhiquan
 * @contributors leizheng
 * @date 2026-02-05
 * @version 3.3.0
 *
 * @changelog
 *   - 2026-02-25 kongzhiquan: v3.4.0 tempDir 改为基于 dataset_root 的 .ocean_preprocess_temp
 *   - 2026-02-05 kongzhiquan: v3.3.0 移除 try-catch 与冗余参数
 *     - 删除无用参数（inspect_result_path, validate_result_path 等），改为自动从固定路径读取
 *     - 错误时直接 throw Error 而非返回 status: 'error'
 *   - 2026-02-05 kongzhiquan: v3.2.0 使用 zod 添加 user_confirmation 参数严格校验
 *     - 校验 user_confirmation 必须包含 4 个阶段的确认信息
 *     - 校验每个阶段必须包含必要字段
 *     - 返回详细的错误提示，指导 Agent 正确填写
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
import { z } from 'zod'
import path from 'node:path'
import { shellEscapeDouble } from '@/utils/shell'

// 使用 zod 定义 user_confirmation 的 schema
const UserConfirmationSchema = z.object({
  stage1_research_vars: z.object({
    selected: z.array(z.string()).min(1, '必须选择至少一个研究变量'),
    confirmed_at: z.string().optional()
  }),

  stage2_static_mask: z.object({
    static_vars: z.array(z.string()),
    mask_vars: z.array(z.string()),
    coord_vars: z.object({
      lon: z.string().optional(),
      lat: z.string().optional()
    }).optional(),
    confirmed_at: z.string().optional()
  }),

  stage3_parameters: z.object({
    scale: z.number().optional(),
    downsample_method: z.string().optional(),
    train_ratio: z.number().min(0).max(1),
    valid_ratio: z.number().min(0).max(1),
    test_ratio: z.number().min(0).max(1),
    h_slice: z.string().optional(),
    w_slice: z.string().optional(),
    crop_recommendation: z.string().optional(),
    confirmed_at: z.string().optional()
  }).refine(
    data => !data.scale || data.downsample_method,
    { message: '指定了 scale 时必须同时指定 downsample_method' }
  ),

  stage4_execution: z.object({
    confirmed: z.literal(true, {
      errorMap: () => ({ message: 'confirmed 必须为 true，表示用户已确认执行' })
    }),
    confirmed_at: z.string().optional(),
    execution_started_at: z.string().optional()
  })
})

export type UserConfirmation = z.infer<typeof UserConfirmationSchema>

export interface ReportResult {
  status: 'success' | 'error'
  report_path: string
  errors?: string[]
  message?: string
}

/**
 * 格式化 zod 校验错误为可读的错误信息
 */
function formatZodErrors(error: z.ZodError): string[] {
  return error.errors.map(err => {
    const path = err.path.join('.')
    return path ? `${path}: ${err.message}` : err.message
  })
}

/**
 * 生成 user_confirmation 格式示例
 */
function getUserConfirmationExample(): string {
  return `{
  "stage1_research_vars": {
    "selected": ["chl", "no3"],
    "confirmed_at": "2026-02-04T10:30:00Z"
  },
  "stage2_static_mask": {
    "static_vars": ["lon", "lat", "mask"],
    "mask_vars": ["mask"],
    "coord_vars": { "lon": "lon", "lat": "lat" },
    "confirmed_at": "2026-02-04T10:31:00Z"
  },
  "stage3_parameters": {
    "scale": 4,
    "downsample_method": "area",
    "train_ratio": 0.7,
    "valid_ratio": 0.15,
    "test_ratio": 0.15,
    "confirmed_at": "2026-02-04T10:32:00Z"
  },
  "stage4_execution": {
    "confirmed": true,
    "confirmed_at": "2026-02-04T10:33:00Z"
  }
}`
}

export const oceanSrPreprocessReportTool = defineTool({
  name: 'ocean_sr_preprocess_report',
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
      output_path
    } = args

    // 0. 使用 zod 校验 user_confirmation 参数
    const parseResult = UserConfirmationSchema.safeParse(user_confirmation)
    if (!parseResult.success) {
      const validationErrors = formatZodErrors(parseResult.error)
      const errorMessage = [
        '⛔ user_confirmation 参数校验失败：',
        '',
        ...validationErrors.map(e => `  - ${e}`),
        '',
        '📋 正确的 user_confirmation 格式示例：',
        getUserConfirmationExample(),
        '',
        '⚠️ 即使用户接受了推荐配置，也必须将这些配置记录到 user_confirmation 中！'
      ].join('\n')

      throw new Error(errorMessage)
    }

    // 1. 检查 Python 环境
    const pythonPath = await findFirstPythonPath()
    if (!pythonPath) {
      throw new Error('未找到可用的Python解释器')
    }

    // 2. 准备路径
    const pythonCmd = `"${shellEscapeDouble(pythonPath)}"`
    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean-SR-data-preprocess/generate_report.py')
    const reportPath = output_path || path.join(dataset_root, 'preprocessing_report.md')

    // 3. 准备配置
    const tempDir = path.resolve(dataset_root, '.ocean_preprocess_temp')
    const configPath = path.join(tempDir, 'report_config.json')

    const config = {
      dataset_root,
      user_confirmation: parseResult.data,
      inspect_result_path: path.join(tempDir, 'inspect_result.json'),
      validate_result_path: path.join(tempDir, 'validate_result.json'),
      convert_result_path: path.join(tempDir, 'convert_result.json'),
      manifest_path: path.join(dataset_root, 'preprocess_manifest.json'),
      metrics_result_path: path.join(tempDir, 'metrics_result.json'),
      output_path: reportPath
    }

    // 4. 写入配置
    await ctx.sandbox.fs.write(configPath, JSON.stringify(config, null, 2))

    // 5. 执行 Python 脚本
    const result = await ctx.sandbox.exec(
      `${pythonCmd} "${shellEscapeDouble(scriptPath)}" --config "${shellEscapeDouble(configPath)}"`,
      { timeoutMs: 120000 }
    )

    if (result.code !== 0) {
      throw new Error(`Python执行失败: ${result.stderr}`)
    }

    return {
      status: 'success',
      report_path: reportPath,
      message: `报告已生成: ${reportPath}，请勿再手写一份新的报告，直接使用此报告并补充分析部分即可。`
    } as ReportResult
  }
})
