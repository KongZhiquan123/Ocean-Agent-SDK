/**
 * @file tool-result-transformer.ts
 *
 * @description 工具结果格式转换器，根据工具名称裁剪 result，只保留后端需要的信息
 * @author kongzhiquan
 * @date 2026-02-10
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-10 kongzhiquan: v1.0.0 初始版本
 *     - 新增 transformToolResult 集中格式转换器
 *     - 对 ocean_preprocess_full 工具结果做裁剪，只保留当前步骤进度信息
 *     - 未注册的工具名原样透传
 */
import type { ToolCallSnapshot } from "@shareai-lab/kode-sdk"

/**
 * 简单状态工具的标签映射
 */
const SIMPLE_TOOL_LABELS: Record<string, string> = {
  ocean_metrics: '计算下采样数据质量指标',
  ocean_generate_report: '生成预处理报告',
  ocean_visualize: '生成可视化对比图',
  ocean_sr_check_gpu: 'GPU 检测',
  ocean_sr_list_models: '列出可用超分模型',
  ocean_sr_generate_report: '生成超分训练报告',
  ocean_sr_visualize: '生成超分训练可视化图表',
  ocean_sr_train_status: '查询超分训练状态',
  ocean_sr_train: '启动超分训练',
}

const PREPROCESS_FULL_STEP_LABELS: Record<string, string> = {
  step_a: '数据检查',
  step_b: '张量验证',
  step_c: 'HR 数据转换',
  step_c2: 'LR 数据转换',
  step_d: '下采样',
  step_e: '可视化',
}

/**
 * 集中的工具结果转换器
 * 根据工具名称裁剪 result，只保留后端需要的信息
 * 未注册的工具名原样透传
 */
interface TransformedToolResult {
  status: 'success' | 'failed'
  message: string
  [key: string]: any
}

export function transformToolResult(toolCall: ToolCallSnapshot): TransformedToolResult {
  const { name: toolName, result } = toolCall
  if (toolName === 'ocean_preprocess_full') {
    return transformPreprocessFull(result)
  }
  const label = SIMPLE_TOOL_LABELS[toolName]
  if (label) {
    return transformSimpleStatus(result, label)
  }
  if (toolName.startsWith('bash_')) {
    return transformBashResult(result)
  }
  if (toolName.startsWith('fs_')) {
    return transformFileOperationResult(toolCall)
  }
  if (toolName === 'skills') {
    return transformSkillResult(result)
  }
  return result
}

function transformBashResult(result: any): { status: 'success' | 'failed'; message: string } {
  if (!result) return { status: 'failed', message: 'Bash 执行失败，未返回结果' }
  const ok = Boolean(result.code === 0)
  const message = `Bash 执行结果: ${result.output || '无输出'}`
  return { status: ok ? 'success' : 'failed', message }
}

function transformSkillResult(result: any): { status: 'success' | 'failed'; message: string } {
  if (!result) return { status: 'failed', message: '技能调用失败，未返回结果' }
  const ok = Boolean(result.ok)
  const message = ok ? '技能调用成功' : '技能调用失败'
  return { status: ok ? 'success' : 'failed', message }
}

function transformPreprocessFull(result: any): { status: 'success' | 'failed'; message: string } {
  if (!result) return { status: 'failed', message: '预处理流程执行失败' }

  // 按执行顺序倒序查找最新的非空 step
  const stepKeys = ['step_e', 'step_d', 'step_c2', 'step_c', 'step_b', 'step_a']
  for (const key of stepKeys) {
    if (result[key] != null) {
      const label = PREPROCESS_FULL_STEP_LABELS[key]
      const success_status = ['success', 'ok', 'completed', 'pass']
      const ok = success_status.includes(result[key].status) || success_status.includes(result[key].overall_status)
      const op = result[key].status === 'skipped' ? '跳过' : '正在执行'
      return { status: ok ? 'success' : 'failed', message: `${op}${label}步骤...` }
    }
  }

  return { status: 'failed', message: result.message || '预处理流程正在执行未知状态' }
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

  const base = {
    status: 'failed' as const,
    modified: false,
    message: '文件操作失败，未返回结果',
    paths: [] as string[],
  }

  if (!result) return base

  const toList = (paths: Array<string | undefined>) => paths.filter((p): p is string => Boolean(p))

  switch (toolName) {
    case 'fs_read': {
      const path = result.path || inputPreview?.path
      const truncated = result.truncated ? '（内容已截断）' : ''
      return {
        status: 'success',
        modified: false,
        message: `读取文件${path ? ` ${path}` : ''}成功${truncated}`,
        paths: [],
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
        paths: [],
      }
    }

    case 'fs_grep': {
      const ok = Boolean(result.ok)
      const matchCount = Array.isArray(result.matches) ? result.matches.length : 0
      const target = result.path || inputPreview?.path || '目标文件'
      return {
        status: ok ? 'success' : 'failed',
        modified: false,
        message: ok ? `在 ${target} 中找到 ${matchCount} 处匹配` : '文件搜索失败',
        paths: [],
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
        paths: modified ? toList([path]) : [],
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
      return {
        status: ok ? 'success' : 'failed',
        modified: false,
        message: `文件操作(${action})已完成`,
        paths: [],
      }
    }
  }
}