/**
 * @file agent-manager.ts
 *
 * @description 管理 Agent 实例的创建与消息处理
 * @author kongzhiquan
 * @contributors Leizheng
 * @date 2026-02-02
 * @version 1.5.0
 *
 * @changelog
 *   - 2026-02-10 kongzhiquan: v1.5.0 新增 transformToolResult 集中格式转换器
 *     - 对 ocean_preprocess_full 工具结果做裁剪，只保留当前步骤进度信息
 *     - 未注册的工具名原样透传
 *   - 2026-02-08 Leizheng: v1.4.0 增加受控 bash 白名单与安全路径检查
 *   - 2026-02-07 Leizheng: v1.3.0 修复 KODE SDK 内部处理超时（5分钟→2小时）
 *   - 2026-02-07 Leizheng: v1.2.0 sandbox 添加 allowPaths: ['/data'] 允许访问数据目录
 *   - 2026-02-05 kongzhiquan: v1.1.0 新增 tool:error 事件处理
 *     - 在 convertProgressToSSE 中添加 tool:error case
 *     - 返回 tool_error 类型的 SSE 事件
 *   - 2026-02-06 Leizheng: v1.1.1 修复 tool:end 事件 result 可能为 undefined
 */

import { Agent, type ProgressEvent } from '@shareai-lab/kode-sdk'
import { getDependencies } from './config'
import { transformToolResult } from './utils/tool-result-transformer'
import { transformToolUse } from './utils/tool-use-transformer'

// ========================================
// 类型定义
// ========================================

export interface AgentConfig {
  mode: 'ask' | 'edit'
  workingDir?: string
  outputsPath?: string
  userId?: string
  files?: string[]
}

export interface SSEEvent {
  type: string
  [key: string]: any
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
    allowPaths: ['/data', `${process.cwd()}/.skills`], // 允许访问数据目录和技能目录
  }

  const agent = await Agent.create(
    {
      templateId,
      sandbox: sandboxConfig,
      metadata: {
        userId: config.userId || 'anonymous',
        mode: config.mode,
        files: config.files,
      },
    },
    deps,
  )

  // 增大 KODE SDK 内部处理超时（默认 5 分钟，预处理/训练流水线可能需要数小时）
  ;(agent as any).PROCESSING_TIMEOUT = 2 * 60 * 60 * 1000 // 2 小时

  return agent
}

// ========================================
// Agent 事件处理
// ========================================

// ========================================
// 权限控制 - 危险命令黑名单
// ========================================

const DANGEROUS_PATTERNS = [
  /rm\s+(-[rRf]+\s+)*[\/~]/,         // rm -rf / 或 rm -rf ~
  /rm\s+(-[rRf]+\s+)*\.\./,          // rm -rf ..
  />\s*\/etc\//,                      // 重定向写入 /etc/
  />\s*\/usr\//,                      // 重定向写入 /usr/
  />\s*\/bin\//,                      // 重定向写入 /bin/
  /sudo\s+/,                          // sudo 命令
  /chmod\s+777/,                      // 危险权限
  /chown\s+root/,                     // 改变所有者为 root
  /mkfs/,                             // 格式化磁盘
  /dd\s+.*of=\/dev/,                  // 写入设备
  /:(){ :|:& };:/,                    // fork 炸弹
  />\s*\/dev\/(sda|hda|nvme)/,        // 写入磁盘设备
  /curl.*\|\s*(ba)?sh/,               // curl | bash 远程执行
  /wget.*\|\s*(ba)?sh/,               // wget | bash 远程执行
]

const SHELL_CONTROL_PATTERN = /[;&`<>]/ // 禁止命令拼接/重定向
const SUBSHELL_PATTERN = /\$\(/ // 禁止 $() 子命令
const DOLLAR_PATTERN = /\$/ // 禁止环境变量展开
const NEWLINE_PATTERN = /[\r\n]/ // 禁止换行注入

const SAFE_READ_PATTERNS: RegExp[] = [
  /^pwd$/,
  /^whoami$/,
  /^id$/,
  /^date$/,
  /^ls(\s+[-\w./]+)*$/,
  /^cat\s+[-\w./]+$/,
  /^head(\s+-n\s+\d+)?\s+[-\w./]+$/,
  /^tail(\s+-n\s+\d+)?\s+[-\w./]+$/,
  /^sed\s+-n\s+['"]?\d+(,\d+)?p['"]?\s+[-\w./]+$/,
  /^rg\s+['"][^'"]+['"](\s+[-\w./]+)*$/,
  /^(grep|egrep|fgrep)(\s+-[a-zA-Z]+)*\s+[^|]+\s+[-\w./]+$/,
  /^diff(\s+-[a-zA-Z]+)*\s+[-\w./]+\s+[-\w./]+$/,
  /^tree(\s+-[a-zA-Z]+)*(\s+[-\w./]+)?$/,
  /^wc(\s+-[clmw]+)?\s+[-\w./]+$/,
  /^stat\s+[-\w./]+$/,
  /^du(\s+-h)?\s+[-\w./]+$/,
  /^df(\s+-h)?$/,
]

const SAFE_WRITE_PATTERNS: RegExp[] = [
  /^mkdir(\s+-p)?\s+[-\w./]+$/,
  /^touch\s+[-\w./]+$/,
  /^cp(\s+-[a-zA-Z]+)?\s+[-\w./]+\s+[-\w./]+$/,
  /^mv(\s+-[a-zA-Z]+)?\s+[-\w./]+\s+[-\w./]+$/,
]

function isDangerousCommand(command: string): boolean {
  return DANGEROUS_PATTERNS.some(pattern => pattern.test(command))
}

function hasShellControlChars(command: string): boolean {
  return (
    SHELL_CONTROL_PATTERN.test(command) ||
    SUBSHELL_PATTERN.test(command) ||
    DOLLAR_PATTERN.test(command) ||
    NEWLINE_PATTERN.test(command)
  )
}

function isWhitelisted(command: string, patterns: RegExp[]): boolean {
  return patterns.some(pattern => pattern.test(command))
}

function splitPipeline(command: string): string[] {
  return command.split('|').map((part) => part.trim()).filter(Boolean)
}

function isReadOnlyPipeline(command: string): boolean {
  if (!command.includes('|')) return false
  const parts = splitPipeline(command)
  if (parts.length === 0) return false
  return parts.every(part => isWhitelisted(part, SAFE_READ_PATTERNS))
}

function isAllowedPathToken(token: string): boolean {
  const trimmed = token.replace(/^['"]|['"]$/g, '')
  if (!trimmed || trimmed.startsWith('-')) return true
  if (trimmed.includes('..')) return false
  if (trimmed.startsWith('~')) return false
  if (trimmed.startsWith('/')) {
    return trimmed.startsWith('/data') || trimmed.startsWith(process.cwd())
  }
  return true
}

function hasUnsafePath(command: string): boolean {
  const tokens = command.split(/\s+/)
  return tokens.some(token => !isAllowedPathToken(token))
}

export function setupAgentHandlers(agent: Agent, reqId: string): void {
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
        if (hasUnsafePath(command)) {
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

  // 错误处理
  agent.on('error', (event: any) => {
    console.error(`[agent-manager] [req ${reqId}] Agent 错误:`, {
      phase: event.phase,
      message: event.message,
      severity: event.severity,
    })
  })
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
        error: event.error,
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

export async function* processMessage(
  agent: Agent,
  message: string,
  reqId: string,
): AsyncGenerator<SSEEvent> {
  // 发送开始事件
  yield {
    type: 'start',
    agentId: agent.agentId,
    timestamp: Date.now(),
  }

  // 订阅 Progress 事件
  const progressIterator = agent.subscribe(['progress'])[Symbol.asyncIterator]()

  // 异步发送消息（不等待完成）
  const sendTask = agent.send(message).catch((err) => {
    console.error(`[agent-manager] [req ${reqId}] 发送消息失败:`, err)
    throw err
  })

  // 处理 Progress 事件流
  try {
    while (true) {
      const { done, value } = await progressIterator.next()
      if (done) break

      const event = value.event as ProgressEvent

      // 检查是否完成
      if (event.type === 'done') {
        break
      }

      // 转换并发送 SSE 事件
      const sseEvent = convertProgressToSSE(event, reqId)
      if (sseEvent) {
        yield sseEvent
      }
    }
  } finally {
    // 确保消息发送任务完成
    await sendTask
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
