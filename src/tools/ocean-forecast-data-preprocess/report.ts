/**
 * @file report.ts
 * @description 海洋预报数据预处理报告生成工具
 *
 * @author Leizheng
 * @date 2026-02-25
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-25 Leizheng: v1.0.0 初始版本
 *     - 读取 preprocess_manifest.json / time_index.json / var_names.json
 *     - 在 TypeScript 内联生成 Markdown 报告（无 Python 依赖）
 *     - 使用 zod 校验 user_confirmation（4 阶段记录）
 *     - 报告末尾预留 AI 分析占位符
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { z } from 'zod'
import path from 'node:path'

// ========================================
// user_confirmation 校验 Schema（预报版）
// ========================================

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
    train_ratio: z.number().min(0).max(1),
    valid_ratio: z.number().min(0).max(1),
    test_ratio: z.number().min(0).max(1),
    h_slice: z.string().optional(),
    w_slice: z.string().optional(),
    confirmed_at: z.string().optional()
  }),

  stage4_execution: z.object({
    confirmed: z.literal(true, {
      errorMap: () => ({ message: 'confirmed 必须为 true' })
    }),
    confirmed_at: z.string().optional()
  })
})

export type UserConfirmation = z.infer<typeof UserConfirmationSchema>

export interface ForecastReportResult {
  status: 'success' | 'error'
  report_path: string
  message?: string
}

function formatZodErrors(error: z.ZodError): string[] {
  return error.errors.map(err => {
    const p = err.path.join('.')
    return p ? `${p}: ${err.message}` : err.message
  })
}

function getConfirmationExample(): string {
  return `{
  "stage1_research_vars": {
    "selected": ["uo", "vo"],
    "confirmed_at": "2026-02-25T10:30:00Z"
  },
  "stage2_static_mask": {
    "static_vars": ["lon_rho", "lat_rho"],
    "mask_vars": ["mask_rho"],
    "coord_vars": { "lon": "lon_rho", "lat": "lat_rho" },
    "confirmed_at": "2026-02-25T10:31:00Z"
  },
  "stage3_parameters": {
    "train_ratio": 0.7,
    "valid_ratio": 0.15,
    "test_ratio": 0.15,
    "confirmed_at": "2026-02-25T10:32:00Z"
  },
  "stage4_execution": {
    "confirmed": true,
    "confirmed_at": "2026-02-25T10:33:00Z"
  }
}`
}

// ========================================
// Markdown 报告生成
// ========================================

function generateMarkdownReport(
  manifest: any,
  timeIndex: any,
  varNames: any,
  confirmation: UserConfirmation,
  datasetRoot: string
): string {
  const now = new Date().toISOString()
  const dynVars: string[] = varNames?.dynamic || manifest?.dyn_vars || []
  const staticVars: string[] = varNames?.static || manifest?.stat_vars || []
  const maskVars: string[] = varNames?.mask || manifest?.mask_vars || []
  const spatialShape: number[] = varNames?.spatial_shape || manifest?.spatial_shape || []
  const splitCounts = manifest?.split_counts || {}
  const splitRatios = manifest?.split_ratios || {}
  const sourceFiles: string[] = manifest?.source_files || []
  const warnings: string[] = manifest?.warnings || []

  const globalTimeInfo = timeIndex?.global || {}
  const splitsTimeInfo = timeIndex?.splits || {}
  const timeGapsCount = globalTimeInfo.time_gaps_count || 0

  // 后置验证信息（如果嵌入了 manifest）
  const validation = manifest?.post_validation || {}

  const lines: string[] = []

  lines.push(`# 海洋预报数据预处理报告`)
  lines.push('')
  lines.push(`> 生成时间: ${now}`)
  lines.push(`> 数据集目录: \`${datasetRoot}\``)
  lines.push('')

  // ---- Section 1: 数据集概览 ----
  lines.push(`## 1. 数据集概览`)
  lines.push('')
  lines.push(`| 属性 | 值 |`)
  lines.push(`|------|-----|`)
  lines.push(`| 来源目录 | \`${manifest?.nc_folder || '未知'}\` |`)
  lines.push(`| NC 文件数 | ${sourceFiles.length} |`)
  lines.push(`| 总时间步数 | ${globalTimeInfo.total_steps || '未知'} |`)
  lines.push(`| 空间形状 | ${spatialShape.length > 0 ? spatialShape.join(' × ') : '未知'} |`)
  lines.push(`| 数据范围 | ${globalTimeInfo.start_ts || '?'} ~ ${globalTimeInfo.end_ts || '?'} |`)
  lines.push(`| 时间步长（估计）| ${globalTimeInfo.estimated_step_seconds != null ? `${Math.round(globalTimeInfo.estimated_step_seconds / 3600)} 小时` : '未知'} |`)
  lines.push(`| 时间间隔异常 | ${timeGapsCount > 0 ? `⚠️ ${timeGapsCount} 处` : '✅ 无'} |`)
  lines.push('')

  // ---- Section 2: 变量配置 ----
  lines.push(`## 2. 变量配置`)
  lines.push('')
  lines.push(`| 类型 | 变量 |`)
  lines.push(`|------|------|`)
  lines.push(`| 动态变量（预报目标） | ${dynVars.join(', ') || '无'} |`)
  lines.push(`| 静态变量 | ${staticVars.join(', ') || '无'} |`)
  lines.push(`| 掩码变量 | ${maskVars.join(', ') || '无'} |`)
  lines.push('')

  // ---- Section 3: 用户确认记录 ----
  lines.push(`## 3. 用户确认记录`)
  lines.push('')
  lines.push(`### 阶段 1：研究变量选择`)
  lines.push(`- 选择的变量：${confirmation.stage1_research_vars.selected.join(', ')}`)
  if (confirmation.stage1_research_vars.confirmed_at) {
    lines.push(`- 确认时间：${confirmation.stage1_research_vars.confirmed_at}`)
  }
  lines.push('')

  lines.push(`### 阶段 2：静态/掩码变量选择`)
  lines.push(`- 静态变量：${confirmation.stage2_static_mask.static_vars.join(', ') || '无'}`)
  lines.push(`- 掩码变量：${confirmation.stage2_static_mask.mask_vars.join(', ') || '无'}`)
  if (confirmation.stage2_static_mask.coord_vars) {
    const coords = confirmation.stage2_static_mask.coord_vars
    lines.push(`- 经度变量：${coords.lon || '未指定'}`)
    lines.push(`- 纬度变量：${coords.lat || '未指定'}`)
  }
  if (confirmation.stage2_static_mask.confirmed_at) {
    lines.push(`- 确认时间：${confirmation.stage2_static_mask.confirmed_at}`)
  }
  lines.push('')

  lines.push(`### 阶段 3：处理参数确认`)
  const s3 = confirmation.stage3_parameters
  lines.push(`- 训练集比例：${(s3.train_ratio * 100).toFixed(0)}%`)
  lines.push(`- 验证集比例：${(s3.valid_ratio * 100).toFixed(0)}%`)
  lines.push(`- 测试集比例：${(s3.test_ratio * 100).toFixed(0)}%`)
  if (s3.h_slice) lines.push(`- H 裁剪：\`${s3.h_slice}\``)
  if (s3.w_slice) lines.push(`- W 裁剪：\`${s3.w_slice}\``)
  if (s3.confirmed_at) lines.push(`- 确认时间：${s3.confirmed_at}`)
  lines.push('')

  lines.push(`### 阶段 4：执行确认`)
  lines.push(`- 用户确认执行：✅ 是`)
  if (confirmation.stage4_execution.confirmed_at) {
    lines.push(`- 确认时间：${confirmation.stage4_execution.confirmed_at}`)
  }
  lines.push('')

  // ---- Section 4: 数据集划分 ----
  lines.push(`## 4. 数据集划分`)
  lines.push('')
  lines.push(`| 划分 | 时间步数 | 比例 | 起始时间 | 结束时间 |`)
  lines.push(`|------|---------|------|---------|---------|`)
  for (const split of ['train', 'valid', 'test']) {
    const count = splitCounts[split] || 0
    const ratio = splitRatios[split] || 0
    const splitTime = splitsTimeInfo[split] || {}
    lines.push(`| ${split} | ${count} | ${(ratio * 100).toFixed(0)}% | ${splitTime.start_ts || '?'} | ${splitTime.end_ts || '?'} |`)
  }
  lines.push('')

  // ---- Section 5: 后置验证 ----
  lines.push(`## 5. 后置验证`)
  lines.push('')
  if (validation && Object.keys(validation).length > 0 && !validation.skipped) {
    const r1 = validation.rule1_integrity
    const r2 = validation.rule2_time_order
    const r3 = validation.rule3_nan_consistency

    lines.push(`| 规则 | 状态 | 说明 |`)
    lines.push(`|------|------|------|`)

    if (r1) {
      const r1Status = r1.passed ? '✅ 通过' : `❌ 失败 (${r1.errors?.length || 0} 个错误)`
      lines.push(`| Rule 1: 完整性 | ${r1Status} | 所有 NPY 文件存在且形状一致 |`)
    }
    if (r2) {
      const r2Status = r2.passed ? '✅ 通过' : `❌ 失败 (${r2.errors?.length || 0} 个错误)`
      lines.push(`| Rule 2: 时间单调性 | ${r2Status} | 时间戳在各 split 内严格递增 |`)
    }
    if (r3) {
      const r3Status = r3.skipped ? '⏭️ 跳过' : (r3.passed ? '✅ 通过' : `❌ 失败 (${r3.errors?.length || 0} 个错误)`)
      lines.push(`| Rule 3: NaN 一致性 | ${r3Status} | 非掩码区域无异常 NaN |`)
    }

    // 详细错误
    const allErrors = [
      ...(r1?.errors || []),
      ...(r2?.errors || []),
      ...(r3?.errors || [])
    ]
    if (allErrors.length > 0) {
      lines.push('')
      lines.push(`### 验证错误详情`)
      allErrors.slice(0, 20).forEach(e => lines.push(`- ${e}`))
      if (allErrors.length > 20) {
        lines.push(`- ... 还有 ${allErrors.length - 20} 个错误`)
      }
    }
  } else if (validation.skipped) {
    lines.push(`> 后置验证已跳过（run_validation=false）`)
  } else {
    lines.push(`> 无后置验证结果`)
  }
  lines.push('')

  // ---- Section 6: 输出目录结构 ----
  lines.push(`## 6. 输出目录结构`)
  lines.push('')
  lines.push(`\`\`\``)
  lines.push(`${datasetRoot}/`)
  lines.push(`├── train/`)
  dynVars.forEach(v => {
    const count = splitCounts.train || '?'
    lines.push(`│   ├── ${v}/          # ${count} 个时间步 NPY 文件`)
  })
  lines.push(`├── valid/`)
  dynVars.forEach(v => lines.push(`│   ├── ${v}/`))
  lines.push(`├── test/`)
  dynVars.forEach(v => lines.push(`│   ├── ${v}/`))
  lines.push(`├── static_variables/   # 静态变量 & 掩码 NPY`)
  lines.push(`├── time_index.json     # 完整时间戳溯源`)
  lines.push(`├── var_names.json      # 变量配置（供 DataLoader 使用）`)
  lines.push(`├── preprocess_manifest.json`)
  lines.push(`└── preprocessing_report.md`)
  lines.push(`\`\`\``)
  lines.push('')

  // ---- Section 7: 警告 ----
  if (warnings.length > 0) {
    lines.push(`## 7. 处理警告`)
    lines.push('')
    warnings.slice(0, 20).forEach(w => lines.push(`- ⚠️ ${w}`))
    if (warnings.length > 20) {
      lines.push(`- ... 还有 ${warnings.length - 20} 个警告`)
    }
    lines.push('')
  }

  // ---- Section 8: 分析和建议（Agent 填写） ----
  const analysisSection = warnings.length > 0 ? 8 : 7
  lines.push(`## ${analysisSection}. 分析和建议`)
  lines.push('')
  lines.push(`<!-- AI_ANALYSIS_PLACEHOLDER`)
  lines.push(``)
  lines.push(`【Agent 注意】请将此占位符替换为实际分析内容，分析应包含：`)
  lines.push(``)
  lines.push(`1. 数据质量评估`)
  lines.push(`   - 时间完整性：是否有时间间隔异常？影响是否严重？`)
  lines.push(`   - 空间完整性：NaN 分布是否合理（仅陆地区域）？`)
  lines.push(`   - 数据规模：总时间步数是否足够训练？`)
  lines.push(``)
  lines.push(`2. 划分合理性`)
  lines.push(`   - 训练/验证/测试集的时间段分布是否合理？`)
  lines.push(`   - 各 split 的数量是否满足训练需求？`)
  lines.push(``)
  lines.push(`3. 潜在问题与建议`)
  lines.push(`   - 是否存在季节性偏差（如训练集全为夏季数据）？`)
  lines.push(`   - 是否需要额外的数据增强？`)
  lines.push(`   - 对下游训练的建议（批大小、序列长度等）？`)
  lines.push(``)
  lines.push(`-->`)
  lines.push('')
  lines.push(`*（此处等待 Agent 填写专业分析）*`)
  lines.push('')

  // ---- Section 总结 ----
  const totalSection = analysisSection + 1
  lines.push(`## ${totalSection}. 总结`)
  lines.push('')
  lines.push(`预处理已完成。`)
  lines.push('')
  lines.push(`| 指标 | 数值 |`)
  lines.push(`|------|------|`)
  lines.push(`| 总时间步数 | ${globalTimeInfo.total_steps || '未知'} |`)
  lines.push(`| 训练集 | ${splitCounts.train || 0} 步 |`)
  lines.push(`| 验证集 | ${splitCounts.valid || 0} 步 |`)
  lines.push(`| 测试集 | ${splitCounts.test || 0} 步 |`)
  lines.push(`| 动态变量 | ${dynVars.length} 个 |`)
  lines.push(`| 静态变量 | ${staticVars.length} 个 |`)
  lines.push(`| 掩码变量 | ${maskVars.length} 个 |`)
  lines.push(`| 时间间隔异常 | ${timeGapsCount} 处 |`)
  lines.push(`| 警告数量 | ${warnings.length} 个 |`)
  lines.push('')

  return lines.join('\n')
}

// ========================================
// 工具定义
// ========================================

export const oceanForecastReportTool = defineTool({
  name: 'ocean_forecast_generate_report',
  description: `生成海洋预报数据预处理 Markdown 报告

读取预处理输出目录中的 JSON 文件，生成结构化报告。

**报告内容**：
1. 数据集概览（文件数、总时间步、空间形状、时间范围）
2. 变量配置（动态/静态/掩码变量）
3. 用户确认记录（4 阶段确认信息）
4. 数据集划分（train/valid/test 时间步数和范围）
5. 后置验证结果（Rule 1/2/3）
6. 输出目录结构
7. AI 分析占位符（需 Agent 填写）

**重要**：报告生成后，Agent 必须：
1. 读取报告文件
2. 替换 AI_ANALYSIS_PLACEHOLDER 占位符为实际分析
3. 分析应基于验证结果、时间信息、警告等具体数据

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
    const { dataset_root, user_confirmation, output_path } = args

    // 校验 user_confirmation
    const parseResult = UserConfirmationSchema.safeParse(user_confirmation)
    if (!parseResult.success) {
      const validationErrors = formatZodErrors(parseResult.error)
      throw new Error([
        '⛔ user_confirmation 参数校验失败：',
        '',
        ...validationErrors.map((e: string) => `  - ${e}`),
        '',
        '📋 正确的 user_confirmation 格式示例：',
        getConfirmationExample(),
        '',
        '⚠️ 即使用户接受了推荐配置，也必须将这些配置记录到 user_confirmation 中！'
      ].join('\n'))
    }

    // 读取 JSON 文件
    const manifestPath = path.join(dataset_root, 'preprocess_manifest.json')
    const timeIndexPath = path.join(dataset_root, 'time_index.json')
    const varNamesPath = path.join(dataset_root, 'var_names.json')

    let manifest: any = {}
    let timeIndex: any = {}
    let varNames: any = {}

    try {
      const manifestStr = await ctx.sandbox.fs.read(manifestPath)
      manifest = JSON.parse(manifestStr)
    } catch {
      // manifest 不存在时继续，但报告信息会不完整
    }

    try {
      const timeIndexStr = await ctx.sandbox.fs.read(timeIndexPath)
      timeIndex = JSON.parse(timeIndexStr)
    } catch {
      // 继续
    }

    try {
      const varNamesStr = await ctx.sandbox.fs.read(varNamesPath)
      varNames = JSON.parse(varNamesStr)
    } catch {
      // 继续
    }

    // 生成 Markdown 报告
    const reportContent = generateMarkdownReport(
      manifest,
      timeIndex,
      varNames,
      parseResult.data,
      dataset_root
    )

    // 写入报告文件
    const reportPath = output_path || path.join(dataset_root, 'preprocessing_report.md')
    await ctx.sandbox.fs.write(reportPath, reportContent)

    return {
      status: 'success',
      report_path: reportPath,
      message: `报告已生成: ${reportPath}。请读取报告并替换 AI_ANALYSIS_PLACEHOLDER 占位符为实际的专业分析内容。`
    } as ForecastReportResult
  }
})
