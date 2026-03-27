/**
 * @file tool-result-transformer.ts
 *
 * @description 工具结果格式转换器，根据工具名称裁剪 result，只保留后端需要的信息
 * @author kongzhiquan
 * @date 2026-02-10
 * @version 1.2.0
 *
 * @changelog
 *   - 2026-03-05 kongzhiquan: v1.2.0 新增训练工具专用转换器
 *     - ocean_sr_train_start / ocean_forecast_train_start 共用 transformSrTrainStart
 *     - 正确映射 awaiting_* / token_invalid → awaiting_confirmation
 *     - 正确映射 started → success（附带模型/模式信息）
 *     - 正确映射 error → failed
 *     - 从 SIMPLE_TOOL_LABELS 移除 ocean_sr_train_start，避免 awaiting 状态被误判为 success
 *   - 2026-03-05 kongzhiquan: v1.1.0 修复 transformSrPreprocessFull 与 transformForecastPreprocessFull 状态判断
 *     - 优先检查顶层 overall_status，正确处理 pass/error/awaiting 状态
 *     - 统一处理所有 awaiting_* 状态（不只是 awaiting_confirmation）
 *     - 区分步骤"已完成"与"正在执行"，改善用词准确性
 *   - 2026-02-10 kongzhiquan: v1.0.0 初始版本
 *     - 新增 transformToolResult 集中格式转换器
 *     - 对 ocean_sr_preprocess_full 工具结果做裁剪，只保留当前步骤进度信息
 *     - 未注册的工具名原样透传
 */
import type { ToolCallSnapshot } from "@shareai-lab/kode-sdk"
import { SIMPLE_TOOL_LABELS } from "./constants"

const TOOL_MESSAGE_MAX_LENGTH = 200 as const
const TOOL_FIELD_MAX_LENGTH = 1200 as const

const SR_PREPROCESS_FULL_STEP_LABELS: Record<string, string> = {
  step_a: '数据检查',
  step_b: '张量验证',
  step_c: 'HR 数据转换',
  step_c2: 'LR 数据转换',
  step_d: '下采样',
  step_e: '可视化',
}

const FORECAST_PREPROCESS_FULL_STEP_LABELS: Record<string, string> = {
  step_a: '数据检查',
  step_b: '数据转换',
  step_c: '可视化',
}

// 预处理工具确认参数时可能的状态
const PREPROCESS_AWAITING_LABELS: Record<string, string> = {
  awaiting_variable_selection: '正在等待研究变量选择',
  awaiting_static_selection: '正在等待静态/掩码变量选择',
  awaiting_region_selection: '正在等待用户确认区域裁剪',
  awaiting_parameters:  '正在等待用户确认处理参数',
  awaiting_execution: '正在等待用户最终确认执行',
  token_invalid: 'Token 校验失败，需要重新确认'
}

// 训练工具确认参数时可能的状态
const TRAIN_AWAITING_LABELS: Record<string, string> = {
  awaiting_data_confirmation: '正在等待数据目录确认',
  awaiting_model_selection: '正在等待模型选择',
  awaiting_parameters: '正在等待训练参数设置',
  awaiting_execution: '正在等待执行确认',
  token_invalid: 'Token 校验失败，需要重新确认',
}

/**
 * 集中的工具结果转换器
 * 根据工具名称裁剪 result，只保留后端需要的信息
 * 未注册的工具名原样透传
 */
interface TransformedToolResult {
  status: 'success' | 'failed' | 'awaiting_confirmation'
  message: string
  [key: string]: any
}

export function transformToolResult(toolCall: ToolCallSnapshot): TransformedToolResult {
  const { name: toolName, result } = toolCall
  if (process.env.NODE_ENV === 'development') {
    // 开发环境下输出原始结果以便调试
    return result
  }

  let transformed = result

  const toolTransformers = {
    ocean_sr_train_start: (r) => transformTrainStart(r, '超分训练'),
    ocean_forecast_train_start: (r) => transformTrainStart(r, '预报训练'),
    ocean_sr_preprocess_full: transformSrPreprocessFull,
    ocean_forecast_preprocess_full: transformForecastPreprocessFull,
    skills: transformSkillResult
  }

  const transformer = toolTransformers[toolName]
  if (transformer) {
    transformed = transformer(result)
  } else if (toolName.startsWith('fs_')) {
    transformed = transformFileOperationResult(toolCall)
  } else if (toolName.startsWith('bash_')) {
    transformed = transformBashResult(result)
  } else {
    const label = SIMPLE_TOOL_LABELS[toolName]
    if (label) {
      transformed = transformSimpleStatus(result, label)
    }
  }

  return truncateResultMessage(transformed)
}

function truncateResultMessage<T extends { message?: unknown }>(result: T): T {
  if (!result || typeof result !== 'object') return result
  if (typeof result.message !== 'string') return result
  if (result.message.length <= TOOL_MESSAGE_MAX_LENGTH) return result
  return {
    ...result,
    message: `${result.message.slice(0, TOOL_MESSAGE_MAX_LENGTH)}（内容已截断）`,
  }
}

function truncateField(value: unknown, max = TOOL_FIELD_MAX_LENGTH): string | undefined {
  if (typeof value !== 'string') return undefined
  const normalized = value.trim()
  if (!normalized) return undefined
  if (normalized.length <= max) return normalized
  return `${normalized.slice(0, max)}（内容已截断）`
}

function collectResultPaths(result: any): string[] {
  const values = [
    result?.path,
    result?.output_path,
    result?.outputPath,
    result?.log_path,
    result?.logPath,
    result?.logs_path,
    result?.logsPath,
  ]

  if (Array.isArray(result?.paths)) {
    values.push(...result.paths)
  }

  return Array.from(
    new Set(
      values.filter((value): value is string => typeof value === 'string' && value.trim().length > 0),
    ),
  )
}

function transformBashResult(result: any): {
  status: 'success' | 'failed'
  message: string
  exitCode?: number
  outputSnippet?: string
  stdoutSnippet?: string
  stderrSnippet?: string
  paths: string[]
} {
  if (!result) return { status: 'failed', message: 'Bash 执行失败，未返回结果', paths: [] }
  const exitCode = typeof result.code === 'number'
    ? result.code
    : (typeof result.exitCode === 'number' ? result.exitCode : undefined)
  const outputSnippet = truncateField(result.output)
  const stdoutSnippet = truncateField(result.stdout)
  const stderrSnippet = truncateField(result.stderr)
  const paths = collectResultPaths(result)
  const ok = exitCode != null ? exitCode === 0 : !result.error
  const primaryDetail =
    outputSnippet ||
    stdoutSnippet ||
    stderrSnippet ||
    truncateField(result.message) ||
    (ok ? '命令执行成功' : '命令执行失败')
  const message = `Bash 执行结果: ${primaryDetail}`

  return {
    status: ok ? 'success' : 'failed',
    message,
    exitCode,
    outputSnippet,
    stdoutSnippet,
    stderrSnippet,
    paths,
  }
}

function transformSkillResult(result: any): { status: 'success' | 'failed'; message: string } {
  if (!result) return { status: 'failed', message: '技能调用失败，未返回结果' }
  const ok = Boolean(result.ok)
  const message = ok ? '技能调用成功' : '技能调用失败'
  return { status: ok ? 'success' : 'failed', message }
}

function transformSrPreprocessFull(result: any): { status: 'success' | 'failed' | 'awaiting_confirmation'; message: string } {
  if (!result) return { status: 'failed', message: '超分预处理流程执行失败' }

  const overallStatus: string = result.overall_status || ''

  // 优先处理顶层 overall_status
  if (overallStatus === 'pass') {
    return { status: 'success', message: result.message || '超分预处理流程已全部完成' }
  }
  if (overallStatus === 'error') {
    return { status: 'failed', message: result.message || '超分预处理流程执行失败' }
  }
  // 处理 awaiting 状态
  const awaitingLabel = PREPROCESS_AWAITING_LABELS[overallStatus]
  if (awaitingLabel) {
    return { status: 'awaiting_confirmation', message: awaitingLabel }
  }

  // 顶层状态不明确时，按执行顺序倒序查找最新的非空、非跳过 step
  const stepKeys = ['step_e', 'step_d', 'step_c2', 'step_c', 'step_b', 'step_a'] as const
  const success_status = ['success', 'ok', 'completed', 'pass'] as const

  for (const key of stepKeys) {
    const step = result[key]
    if (!step || step.status === 'skipped') continue
    const label = SR_PREPROCESS_FULL_STEP_LABELS[key]
    const ok = success_status.includes(step.status)
    return {
      status: ok ? 'success' : 'failed',
      message: ok ? `超分预处理流程${label}步骤已完成` : `超分预处理流程${label}步骤失败`
    }
  }

  return { status: 'failed', message: result.message || '超分预处理流程正在执行未知状态' }
}

function transformForecastPreprocessFull(result: any): { status: 'success' | 'failed' | 'awaiting_confirmation'; message: string } {
  if (!result) return { status: 'failed', message: '预报预处理流程执行失败' }

  const overallStatus: string = result.overall_status || ''

  // 优先处理顶层 overall_status
  if (overallStatus === 'pass') {
    return { status: 'success', message: result.message || '预报预处理流程已全部完成' }
  }
  if (overallStatus === 'error') {
    return { status: 'failed', message: result.message || '预报预处理流程执行失败' }
  }
  // 处理 awaiting 状态
  const awaitingLabel = PREPROCESS_AWAITING_LABELS[overallStatus]
  if (awaitingLabel) {
    return { status: 'awaiting_confirmation', message: awaitingLabel }
  }

  // 顶层状态不明确时，按执行顺序倒序查找最新的非空、非跳过 step
  const stepKeys = ['step_c', 'step_b', 'step_a'] as const
  const success_status = ['success', 'ok', 'completed', 'pass'] as const

  for (const key of stepKeys) {
    const step = result[key]
    if (!step || step.status === 'skipped') continue
    const label = FORECAST_PREPROCESS_FULL_STEP_LABELS[key]
    const ok = success_status.includes(step.status)
    return {
      status: ok ? 'success' : 'failed',
      message: ok ? `预报预处理流程${label}步骤已完成` : `预报预处理流程${label}步骤失败`
    }
  }

  return { status: 'failed', message: result.message || '预报预处理流程正在执行未知状态' }
}

function transformTrainStart(result: any, toolLabel: string): { status: 'success' | 'failed' | 'awaiting_confirmation'; message: string } {
  if (!result) return { status: 'failed', message: `${toolLabel}工具执行失败，未返回结果` }

  const status: string = result.status ?? ''

  // 训练/预测进程已成功启动
  if (status === 'started') {
    const mode: string = result.mode === 'predict' ? '预测推理' : '训练'
    const model: string = result.model ? `（${result.model}）` : ''
    return { status: 'success', message: `${toolLabel} ${mode}进程已启动${model}` }
  }

  // 各阶段等待确认
  const awaitingLabel = TRAIN_AWAITING_LABELS[status]
  if (awaitingLabel) {
    return { status: 'awaiting_confirmation', message: awaitingLabel }
  }

  // 错误状态
  if (status === 'error') {
    const errMsg: string = result.error ?? result.message ?? `${toolLabel}执行失败`
    return { status: 'failed', message: String(errMsg) }
  }

  // 兜底：有 error 字段视为失败
  if (result.error) {
    return { status: 'failed', message: String(result.error) }
  }

  return { status: 'failed', message: result.message ?? `${toolLabel}返回未知状态` }
}

function transformSimpleStatus(result: any, label: string): { status: 'success' | 'failed'; message: string } {
  if (!result) return { status: 'failed', message: `${label}失败` }

  const success_status = ['success', 'ok', 'completed', 'pass']
  const ok = success_status.includes(result?.status) || success_status.includes(result?.overall_status) || !result.error
  return { status: ok ? 'success' : 'failed', message: `${label}${ok ? '成功' : '失败'}` }
}

function transformFileOperationResult(toolCall: ToolCallSnapshot): {
  status: 'success' | 'failed'
  modified: boolean
  message: string
  paths: string[]
  path?: string
} {
  if (!toolCall.result) {
    return {
      status: 'failed',
      modified: false,
      message: '文件操作失败，未返回结果',
      paths: [] as string[],
    }
  }
  const { name: toolName, result, inputPreview } = toolCall
  const toolState = (toolCall as any).state

  const base = {
    status: 'failed' as const,
    modified: false,
    message: '文件操作失败，未返回结果',
    paths: [] as string[],
  }

  if (!result) return base

  const toList = (paths: Array<string | undefined>) => paths.filter((p): p is string => Boolean(p))
  const inputPath = inputPreview?.path
  const resultPath = result.path
  const fallbackPath = resultPath || inputPath

  if (toolState === 'FAILED' || result.error || result.ok === false) {
    return {
      status: 'failed',
      modified: false,
      message: `${toolName === 'fs_read' ? '读取文件失败' : '文件操作失败'}${fallbackPath ? `: ${fallbackPath}` : ''}`,
      path: fallbackPath,
      paths: fallbackPath ? toList([fallbackPath]) : [],
    }
  }

  switch (toolName) {
    case 'fs_read': {
      const path = result.path || inputPreview?.path
      const truncated = result.truncated ? '（内容已截断）' : ''
      return {
        status: 'success',
        modified: false,
        message: `读取文件${path ? ` ${path}` : ''}成功${truncated}`,
        path,
        paths: path ? toList([path]) : [],
      }
    }

    case 'fs_glob': {
      const ok = Boolean(result.ok)
      const matches = Array.isArray(result.matches) ? result.matches : []
      const truncated = result.truncated ? '（结果已截断）' : ''
      return {
        status: ok ? 'success' : 'failed',
        modified: false,
        message: ok
          ? `匹配到 ${matches.length} 个文件${truncated}`
          : '文件匹配失败',
        paths: ok ? toList(matches.slice(0, 20)) : [],
      }
    }

    case 'fs_grep': {
      const ok = Boolean(result.ok)
      const matchCount = Array.isArray(result.matches) ? result.matches.length : 0
      const target = result.path || inputPreview?.path || '目标文件'
      const matchPaths = Array.isArray(result.matches)
        ? result.matches
          .map((match: any) => match?.path)
          .filter((item: any): item is string => typeof item === 'string' && item.length > 0)
        : []
      return {
        status: ok ? 'success' : 'failed',
        modified: false,
        message: ok ? `在 ${target} 中找到 ${matchCount} 处匹配` : '文件搜索失败',
        path: typeof target === 'string' ? target : undefined,
        paths: ok ? toList(matchPaths.length > 0 ? matchPaths : [typeof target === 'string' ? target : undefined]) : [],
      }
    }

    case 'fs_write': {
      const ok = Boolean(result.ok)
      const path = result.path || inputPreview?.path
      const bytes = typeof result.bytes === 'number' ? result.bytes : undefined
      return {
        status: ok ? 'success' : 'failed',
        modified: ok,
        message: ok
          ? `写入文件成功${path ? `: ${path}` : ''}${bytes != null ? `，写入 ${bytes} 字节` : ''}`
          : `写入文件失败${path ? `: ${path}` : ''}`,
        path,
        paths: ok ? toList([path]) : [],
      }
    }

    case 'fs_edit': {
      const ok = Boolean(result.ok)
      const path = result.path || inputPreview?.path
      const replacements = typeof result.replacements === 'number' ? result.replacements : 0
      const modified = ok && replacements > 0
      return {
        status: ok ? 'success' : 'failed',
        modified,
        message: ok
          ? `编辑文件${path ? ` ${path}` : ''}${replacements > 0 ? `，替换 ${replacements} 处` : '，未发生替换'}`
          : `编辑文件失败${path ? `: ${path}` : ''}`,
        path,
        paths: path ? toList([path]) : [],
      }
    }

    case 'fs_multi_edit': {
      const ok = Boolean(result.ok)
      const items = Array.isArray(result.results) ? result.results : []
      const modifiedItems = items.filter(
        (item: any) => item && (item.status === 'updated' || item.status === 'ok' || (typeof item.replacements === 'number' && item.replacements > 0)),
      )
      const failedItems = items.filter((item: any) => item && item.status === 'failed')
      const paths = toList(modifiedItems.map((item: any) => item.path))
      const modified = paths.length > 0
      return {
        status: ok ? 'success' : 'failed',
        modified,
        message: ok
          ? `批量编辑完成：${paths.length} 个文件更新${failedItems.length ? `，${failedItems.length} 个失败` : ''}`
          : '批量编辑失败',
        paths: modified ? paths : [],
      }
    }

    default: {
      const action = toolName.split('_')[1] || '未知操作'
      const ok = Boolean(result.ok)
      const path = result.path || inputPreview?.path
      return {
        status: ok ? 'success' : 'failed',
        modified: false,
        message: `文件操作(${action})已完成`,
        path,
        paths: path ? toList([path]) : [],
      }
    }
  }
}
