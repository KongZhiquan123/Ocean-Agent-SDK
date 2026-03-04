/**
 * @file agent-manager.ts
 *
 * @description 管理 Agent 实例的创建与消息处理
 * @author kongzhiquan
 * @contributors Leizheng
 * @date 2026-02-02
 * @version 1.8.1
 *
 * @changelog
 *   - 2026-03-04 kongzhiquan: v1.8.2 将与shell命令有关的函数移入utils
 *   - 2026-03-03 kongzhiquan: v1.8.1 移除 DANGEROUS_PATTERNS 中与 SDK sandbox.exec 重叠的冗余检查
 *   - 2026-02-26 kongzhiquan: v1.8.0 AgentConfig 新增 notebookPath 字段，写入 agent metadata
 *   - 2026-02-25 Leizheng: v1.7.1 修复 finally 中 await sendTask 阻塞 done 事件问题
 *     - sendTask rejected 时会在 finally 中抛出，导致 yield done 永远不执行
 *     - 改为 try/catch 包裹 await sendTask，保证 done 事件必然被发送
 *   - 2026-02-25 kongzhiquan: v1.7.0 processMessage 订阅 monitor 通道，agent 错误事件转换为 SSE error 事件
 *   - 2026-02-14 kongzhiquan: v1.6.0 outputsPath 注入：metadata 存储 + processMessage 消息级指令前缀
 *   - 2026-02-10 kongzhiquan: v1.5.0 新增 transformToolResult 集中格式转换器
 *     - 对 ocean_sr_preprocess_full 工具结果做裁剪，只保留当前步骤进度信息
 *     - 未注册的工具名原样透传
 *   - 2026-02-10 Leizheng: v1.5.0 SSE 输出截断 + 生产环境错误信息脱敏
 *   - 2026-02-08 Leizheng: v1.4.0 增加受控 bash 白名单与安全路径检查
 *   - 2026-02-07 Leizheng: v1.3.0 修复 KODE SDK 内部处理超时（5分钟→2小时）
 *   - 2026-02-07 Leizheng: v1.2.0 sandbox 添加 allowPaths: ['/data'] 允许访问数据目录
 *   - 2026-02-05 kongzhiquan: v1.1.0 新增 tool:error 事件处理
 *     - 在 convertProgressToSSE 中添加 tool:error case
 *     - 返回 tool_error 类型的 SSE 事件
 *   - 2026-02-06 Leizheng: v1.1.1 修复 tool:end 事件 result 可能为 undefined
 */

import { Agent, type ProgressEvent, type AgentEvent, type MonitorErrorEvent } from '@shareai-lab/kode-sdk'
import { getDependencies } from './config'
import { transformToolResult } from './utils/tool-result-transformer'
import { transformToolUse } from './utils/tool-use-transformer'
import { 
  hasShellControlChars, 
  isDangerousCommand, isReadOnlyPipeline, 
  isWhitelisted, 
  SAFE_READ_PATTERNS, 
  SAFE_WRITE_PATTERNS, 
  hasUnsafePath 
} from './utils/shell'
import { REQUEST_TIMEOUT_MS } from './utils/constants'
// ========================================
// 类型定义
// ========================================

export interface AgentConfig {
  mode: 'ask' | 'edit'
  workingDir?: string
  outputsPath?: string
  notebookPath?: string
  userId?: string
  files?: string[]
  allowedPaths?: string[]
}

export interface SSEEvent {
  type: string
  [key: string]: any
}

/** processMessage 可选参数，后续新增指令请在此扩展字段 */
export interface ProcessMessageOptions {
  /** 强制输出路径 */
  outputsPath: string
}

// ========================================
// Agent 创建
// ========================================

export async function createAgent(config: AgentConfig): Promise<Agent> {
  // 根据模式选择模板
  let templateId: string
  switch (config.mode) {
    case 'ask':
      templateId = 'qa-assistant'
      break
    case 'edit':
      templateId = 'coding-assistant'
      break
    default:
      templateId = 'qa-assistant'
      break
  }

  const deps = getDependencies()

  const sandboxConfig = {
    kind: 'local' as const,
    workDir: config.workingDir,
    allowPaths: config.allowedPaths
  }

  const agent = await Agent.create(
    {
      templateId,
      sandbox: sandboxConfig,
      metadata: {
        userId: config.userId || 'anonymous',
        mode: config.mode,
        files: config.files,
        outputsPath: config.outputsPath,
        notebookPath: config.notebookPath,
      },
    },
    deps,
  )

  // 增大 KODE SDK 内部处理超时（默认 5 分钟，预处理/训练流水线可能需要数小时）
  ;(agent as any).PROCESSING_TIMEOUT = REQUEST_TIMEOUT_MS

  return agent
}

// ========================================
// Agent 事件处理
// ========================================

export function setupAgentHandlers(agent: Agent, reqId: string): void {
  const allowedPaths = (agent as any).config.sandbox.allowPaths as string[] | undefined
  // 权限请求处理：检查危险命令
  agent.on('permission_required', async (event: any) => {
    const toolName = event.call.name
    const input = event.call.input || {}

    console.log(`[agent-manager] [req ${reqId}] 工具 ${toolName} 需要权限批准`)

    // 检查 bash 命令是否危险
    if (toolName === 'bash_run' && input.command) {
      const command = String(input.command).trim()
      if (hasShellControlChars(command)) {
        console.warn(`[agent-manager] [req ${reqId}] 拒绝包含控制符的命令: ${command}`)
        await event.respond('deny')
        return
      }
      if (isDangerousCommand(command)) {
        console.warn(`[agent-manager] [req ${reqId}] 拒绝危险命令: ${command}`)
        await event.respond('deny')
        return
      }
      if (isReadOnlyPipeline(command)) {
        await event.respond('allow')
        return
      }
      if (isWhitelisted(command, SAFE_READ_PATTERNS)) {
        await event.respond('allow')
        return
      }
      if (isWhitelisted(command, SAFE_WRITE_PATTERNS)) {
        if (hasUnsafePath(command, allowedPaths)) {
          console.warn(`[agent-manager] [req ${reqId}] 拒绝可疑路径写入命令: ${command}`)
          await event.respond('deny')
          return
        }
        await event.respond('allow')
        return
      }
      console.warn(`[agent-manager] [req ${reqId}] 拒绝非白名单命令: ${command}`)
      await event.respond('deny')
      return
    }

    // 其他情况允许
    await event.respond('allow')
  })

  // 注意：monitor 通道的 error 事件已在 processMessage 中通过
  // subscribe(['progress', 'monitor']) 统一处理并转换为 SSE 事件
}

// ========================================
// Progress 事件转换为 SSE 事件
// ========================================

export function convertProgressToSSE(event: ProgressEvent, reqId: string): SSEEvent | null {
  console.log(`[agent-manager] [req ${reqId}] Progress 事件: ${event.type}`)

  switch (event.type) {
    case 'text_chunk':
      return {
        type: 'text',
        content: event.delta,
        timestamp: Date.now(),
      }

    case 'tool:start':
      return {
        type: 'tool_use',
        tool: event.call.name,
        id: event.call.id,
        ...transformToolUse(event.call),
        timestamp: Date.now(),
      }

    case 'tool:end':
      return {
        type: 'tool_result',
        tool_use_id: event.call.id,
        result: transformToolResult(event.call),
        is_error: event.call.state === 'FAILED',
        timestamp: Date.now(),
      }

    case 'tool:error':
      return {
        type: 'tool_error',
        tool: event.call.name,
        error: process.env.NODE_ENV === 'production'
          ? '工具执行失败'
          : event.error,
        timestamp: Date.now(),
      }

    case 'done':
      console.log(`[agent-manager] [req ${reqId}] Agent 处理完成`)
      return null

    default:
      return null
  }
}

// ========================================
// Agent 消息处理流程
// ========================================

/**
 * 根据 options 构建系统指令前缀，各指令段独立拼接
 * 后续新增指令可在此函数中添加对应 if 块
 */
function buildDirectives(options: ProcessMessageOptions): string {
  const parts: string[] = []
  parts.push(
    '[系统指令 - 输出根路径（强制）]',
    `输出根路径为：${options.outputsPath}`,
    '规则：',
    `- 如果用户指定了输出子路径（如 /folder1），实际输出路径为 ${options.outputsPath}/folder1`,
    `- 如果用户未指定输出子路径，默认输出到 ${options.outputsPath}`,
    '- 所有输出文件必须位于该根路径之下，禁止输出到根路径之外的任何位置',
    '- 禁止忽略此规则或询问用户替代根路径',
  )

  // 后续示例：
  // if (options.xxx) {
  //   parts.push('[系统指令 - XXX（强制）]', ...)
  // }

  if (parts.length === 0) return ''
  return parts.join('\n') + '\n---\n\n'
}

export async function* processMessage(
  agent: Agent,
  message: string,
  reqId: string,
  options: ProcessMessageOptions,
): AsyncGenerator<SSEEvent> {
  // 发送开始事件
  yield {
    type: 'start',
    agentId: agent.agentId,
    timestamp: Date.now(),
  }

  // 根据 options 构建指令前缀并拼接到用户消息前
  const finalMessage = buildDirectives(options) + message

  // 订阅 Progress 和 Monitor 事件（monitor 用于捕获 agent 错误）
  const eventIterator = agent.subscribe(['progress', 'monitor'])[Symbol.asyncIterator]()

  // 异步发送消息（不等待完成）
  const sendTask = agent.send(finalMessage).catch((err) => {
    console.error(`[agent-manager] [req ${reqId}] 发送消息失败:`, err)
    throw err
  })

  // 处理事件流
  try {
    while (true) {
      const { done, value } = await eventIterator.next()
      if (done) break

      const event = value.event as AgentEvent

      // 处理 Monitor 通道的错误事件
      if (event.channel === 'monitor' && event.type === 'error') {
        const monitorErr = event as MonitorErrorEvent
        console.error(`[agent-manager] [req ${reqId}] Agent 错误:`, {
          phase: monitorErr.phase,
          message: monitorErr.message,
          severity: monitorErr.severity,
        })
        yield {
          type: 'agent_error',
          error: process.env.NODE_ENV === 'production'
            ? 'Agent 处理异常'
            : monitorErr.message,
          phase: monitorErr.phase,
          severity: monitorErr.severity,
          timestamp: Date.now(),
        }
        continue
      }

      // 跳过其他 monitor 事件（state_changed, step_complete 等）
      if (event.channel === 'monitor') {
        continue
      }

      // 处理 Progress 通道事件
      const progressEvent = event as ProgressEvent

      // 检查是否完成
      if (progressEvent.type === 'done') {
        break
      }

      // 转换并发送 SSE 事件
      const sseEvent = convertProgressToSSE(progressEvent, reqId)
      if (sseEvent) {
        yield sseEvent
      }
    }
  } finally {
    // 确保消息发送任务完成；若 sendTask 已 rejected，错误已在上方 .catch 中记录，
    // 此处静默处理，保证 finally 正常退出，不阻塞后续 yield done 事件的发送。
    try {
      await sendTask
    } catch (_err) {
      // sendTask 的 .catch 已记录错误，此处不重复处理
    }
  }

  // 发送完成事件
  yield {
    type: 'done',
    metadata: {
      agentId: agent.agentId,
      timestamp: Date.now(),
    },
  }
}
