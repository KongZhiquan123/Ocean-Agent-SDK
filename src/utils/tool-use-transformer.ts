/**
 * @file tool-use-transformer.ts
 *
 * @description 工具调用格式转换器，对 tool:start 输入与文案做集中格式化，只保留后端需要的信息
 * @author kongzhiquan
 * @date 2026-02-10
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-10 kongzhiquan: v1.0.0 初始版本
 *     - 新增 transformToolUse 集中格式转换器
 *     - 对常见工具调用的文案进行统一
 *     - 未注册的工具名原样透传输入
 */
import type { ToolCallSnapshot } from '@shareai-lab/kode-sdk'
import { SIMPLE_TOOL_LABELS } from './constants'

/**
 * 集中的工具调用转换器
 * 格式化 tool:start 的文案与输入
 * 未注册的工具名原样透传 inputPreview
 */
export function transformToolUse(toolCall: ToolCallSnapshot): { message: string; input?: any } {
  const { name: toolName, inputPreview } = toolCall

  if (toolName === 'ocean_sr_preprocess_full' || toolName === 'ocean_forecast_preprocess_full') {
    return {
      message: toolName === 'ocean_sr_preprocess_full'
        ? '启动超分预处理流程...'
        : '启动预报预处理流程...',
      input: inputPreview,
    }
  }

  const label = SIMPLE_TOOL_LABELS[toolName]
  if (label) {
    return {
      message: `开始${label}...`,
      input: inputPreview,
    }
  }

  if (toolName.startsWith('bash_')) {
    const cmd = inputPreview?.cmd || '未知命令'
    return {
      message: `执行 Bash: ${cmd}`,
      input: inputPreview,
    }
  }

  if (toolName.startsWith('fs_')) {
    const action = toolName.split('_')[1] || '未知操作'
    return {
      message: `执行文件操作: ${action} ${inputPreview.path ? `，路径： ${inputPreview.path}` : ''}...`,
      input: inputPreview,
    }
  }

  return {
    message: `正在调用工具: ${toolName}...`,
    input: inputPreview,
  }
}
