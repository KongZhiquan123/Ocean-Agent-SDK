import type { SSEEvent } from '@/agent-manager'

export type HudStatus =
  | 'idle'
  | 'thinking'
  | 'running'
  | 'calling_tool'
  | 'reading_file'
  | 'writing_file'
  | 'running_command'
  | 'waiting'
  | 'done'
  | 'failed'

export interface HudRequestMeta {
  reqId: string
  agentId: string
  userId: string
  mode: 'ask' | 'edit'
  isNewSession: boolean
  message: string
  workingDir: string
  outputsPath: string
  notebookPath: string
}

export interface HudTimelineEvent {
  id: string
  timestamp: number
  agentId: string
  reqId: string
  role: string
  type: string
  summary: string
  status: HudStatus
  target?: string | null
  tool?: string | null
  command?: string | null
  path?: string | null
  isError?: boolean
}

export interface HudArtifact {
  id: string
  timestamp: number
  path: string
  kind: string
  source: string
}

export interface HudToolCall {
  id: string
  tool: string
  status: 'running' | 'success' | 'failed'
  inputPreview?: string
  resultPreview?: string
  startedAt: number
  endedAt?: number
}

export interface HudFileActivity {
  id: string
  timestamp: number
  action: 'read' | 'write' | 'edit' | 'create' | 'other'
  path: string
  source: string
}

export interface HudCommandActivity {
  id: string
  timestamp: number
  phase: 'start' | 'stdout' | 'stderr' | 'end'
  command: string
  snippet?: string
  exitCode?: number
}

export interface HudTextChunk {
  id: string
  timestamp: number
  content: string
}

export interface HudAgentState {
  agentId: string
  reqId: string
  role: string
  userId: string
  mode: 'ask' | 'edit'
  status: HudStatus
  isNewSession: boolean
  messagePreview: string
  workingDir: string
  outputsPath: string
  notebookPath: string
  currentAction: string | null
  currentTarget: string | null
  currentTool: string | null
  currentCommand: string | null
  lastText: string | null
  lastOutput: string | null
  lastHeartbeatAt: number | null
  heartbeatCount: number
  startedAt: number
  updatedAt: number
  completedAt: number | null
  error: { message: string; phase?: string; severity?: string } | null
  textStream: HudTextChunk[]
  toolHistory: HudToolCall[]
  files: HudFileActivity[]
  commands: HudCommandActivity[]
  artifacts: HudArtifact[]
  timeline: HudTimelineEvent[]
}

export interface HudStats {
  totalAgents: number
  activeAgents: number
  runningAgents: number
  waitingAgents: number
  failedAgents: number
  doneAgents: number
  connectedWatchers: number
  updatedAt: number
}

export interface HudSnapshot {
  generatedAt: number
  stats: HudStats
  agents: HudAgentState[]
  timeline: HudTimelineEvent[]
}

export interface HudBroadcastEvent {
  type: 'hud_update'
  generatedAt: number
  stats: HudStats
  agent: HudAgentState
  timelineEvent?: HudTimelineEvent
}

const MAX_TIMELINE = 400
const MAX_AGENT_TIMELINE = 60
const MAX_TOOL_HISTORY = 30
const MAX_FILE_HISTORY = 30
const MAX_COMMAND_HISTORY = 20
const MAX_ARTIFACTS = 30
const MAX_TEXT_STREAM = 50

type Listener = (payload: HudBroadcastEvent) => void

function limitPush<T>(items: T[], item: T, max: number): void {
  items.push(item)
  if (items.length > max) {
    items.splice(0, items.length - max)
  }
}

function truncateText(input: string, max = 180): string {
  const normalized = input.replace(/\s+/g, ' ').trim()
  if (normalized.length <= max) return normalized
  return normalized.slice(0, max - 1) + '…'
}

function previewValue(value: unknown, max = 220): string {
  if (value === undefined) return ''
  if (value === null) return 'null'
  if (typeof value === 'string') return truncateText(value, max)

  try {
    return truncateText(JSON.stringify(value), max)
  } catch {
    return truncateText(String(value), max)
  }
}

function uniqueId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

function deriveRole(mode: 'ask' | 'edit'): string {
  return mode === 'edit' ? 'Coding Agent' : 'Ask Agent'
}

function guessArtifactKind(pathValue: string): string {
  const lower = pathValue.toLowerCase()
  if (/\.(png|jpg|jpeg|gif|webp|svg)$/.test(lower)) return 'image'
  if (/\.(md|pdf|html)$/.test(lower)) return 'report'
  if (/\.(ipynb)$/.test(lower)) return 'notebook'
  if (/\.(pth|ckpt|pt|safetensors)$/.test(lower)) return 'model'
  if (/\.(npy|npz|nc|csv|json|yaml|yml)$/.test(lower)) return 'data'
  if (/\.(log|txt)$/.test(lower)) return 'log'
  return 'file'
}

function collectPathLikeValues(value: unknown, depth = 0, found = new Set<string>()): Set<string> {
  if (depth > 3 || value === null || value === undefined) return found

  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (
      trimmed &&
      trimmed.length < 512 &&
      (trimmed.startsWith('/') ||
        trimmed.startsWith('./') ||
        trimmed.startsWith('../') ||
        /[A-Za-z0-9._-]+\.[A-Za-z0-9]{1,8}$/.test(trimmed))
    ) {
      found.add(trimmed)
    }
    return found
  }

  if (Array.isArray(value)) {
    value.forEach(item => collectPathLikeValues(item, depth + 1, found))
    return found
  }

  if (typeof value === 'object') {
    for (const [key, nested] of Object.entries(value as Record<string, unknown>)) {
      if (/(path|paths|file|files|dir|output)/i.test(key)) {
        collectPathLikeValues(nested, depth + 1, found)
      }
    }
    return found
  }

  return found
}

function extractGenericTarget(input: Record<string, unknown> | undefined): string | null {
  if (!input) return null
  const candidates = [
    input.path,
    input.output_path,
    input.outputPath,
    input.log_dir,
    input.dataset_root,
    input.workingDir,
    input.notebookPath,
    input.command,
    input.pattern,
  ]
  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.trim()) {
      return truncateText(candidate, 180)
    }
  }
  return null
}

function deriveToolState(tool: string, input?: Record<string, unknown>): {
  status: HudStatus
  action: string
  target: string | null
  command: string | null
  fileAction?: HudFileActivity['action']
} {
  if (tool === 'bash_run') {
    const command = typeof input?.command === 'string' ? input.command : ''
    return {
      status: 'running_command',
      action: '正在运行命令',
      target: command || null,
      command: command || null,
    }
  }

  if (tool === 'bash_logs') {
    return {
      status: 'running_command',
      action: '正在查看命令日志',
      target: extractGenericTarget(input),
      command: null,
    }
  }

  if (tool === 'fs_read' || tool === 'fs_glob' || tool === 'fs_grep') {
    return {
      status: 'reading_file',
      action: '正在读取文件',
      target: extractGenericTarget(input),
      command: null,
      fileAction: 'read',
    }
  }

  if (tool === 'fs_write') {
    return {
      status: 'writing_file',
      action: '正在写入文件',
      target: extractGenericTarget(input),
      command: null,
      fileAction: 'write',
    }
  }

  if (tool === 'fs_edit' || tool === 'fs_multi_edit') {
    return {
      status: 'writing_file',
      action: '正在修改文件',
      target: extractGenericTarget(input),
      command: null,
      fileAction: 'edit',
    }
  }

  if (tool.startsWith('ocean_sr_') || tool.startsWith('ocean_forecast_')) {
    return {
      status: 'running',
      action: '正在执行海洋任务',
      target: extractGenericTarget(input) ?? tool,
      command: null,
    }
  }

  if (tool.startsWith('todo_')) {
    return {
      status: 'running',
      action: '正在更新任务列表',
      target: tool,
      command: null,
    }
  }

  return {
    status: 'calling_tool',
    action: `正在调用工具 ${tool}`,
    target: extractGenericTarget(input) ?? tool,
    command: null,
  }
}

function cloneAgentState(agent: HudAgentState): HudAgentState {
  return {
    ...agent,
    error: agent.error ? { ...agent.error } : null,
    textStream: agent.textStream.map(item => ({ ...item })),
    toolHistory: agent.toolHistory.map(item => ({ ...item })),
    files: agent.files.map(item => ({ ...item })),
    commands: agent.commands.map(item => ({ ...item })),
    artifacts: agent.artifacts.map(item => ({ ...item })),
    timeline: agent.timeline.map(item => ({ ...item })),
  }
}

class HudManager {
  private agents = new Map<string, HudAgentState>()
  private timeline: HudTimelineEvent[] = []
  private listeners = new Set<Listener>()

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  getSnapshot(): HudSnapshot {
    return {
      generatedAt: Date.now(),
      stats: this.getStats(),
      agents: Array.from(this.agents.values())
        .map(agent => cloneAgentState(agent))
        .sort((a, b) => b.updatedAt - a.updatedAt),
      timeline: this.timeline.map(item => ({ ...item })),
    }
  }

  registerRequest(meta: HudRequestMeta): void {
    const now = Date.now()
    const agent = this.ensureAgent(meta, now)
    agent.status = 'running'
    agent.currentAction = '收到新任务'
    agent.currentTarget = agent.messagePreview || null
    agent.updatedAt = now
    agent.completedAt = null
    agent.error = null

    const timelineEvent: HudTimelineEvent = {
      id: uniqueId('hud-request'),
      timestamp: now,
      agentId: agent.agentId,
      reqId: meta.reqId,
      role: agent.role,
      type: 'request_started',
      summary: `收到任务：${agent.messagePreview || '新请求'}`,
      status: agent.status,
      target: agent.currentTarget,
    }

    this.pushTimeline(agent, timelineEvent)
    this.broadcast(agent, timelineEvent)
  }

  recordSseEvent(meta: HudRequestMeta, event: SSEEvent): void {
    const now = Date.now()
    const agent = this.ensureAgent(meta, now)
    let timelineEvent: HudTimelineEvent | undefined

    switch (event.type) {
      case 'start':
        agent.status = 'running'
        agent.currentAction = '开始处理任务'
        agent.currentTarget = agent.messagePreview || null
        agent.updatedAt = now
        timelineEvent = {
          id: uniqueId('hud-start'),
          timestamp: now,
          agentId: agent.agentId,
          reqId: meta.reqId,
          role: agent.role,
          type: 'start',
          summary: 'Agent 已开始处理请求',
          status: agent.status,
          target: agent.currentTarget,
        }
        break

      case 'heartbeat':
        agent.lastHeartbeatAt = typeof event.timestamp === 'number' ? event.timestamp : now
        agent.heartbeatCount += 1
        agent.updatedAt = now
        if (agent.status !== 'failed' && agent.status !== 'done') {
          agent.status = agent.status === 'idle' ? 'running' : agent.status
        }
        if (!agent.currentAction) {
          agent.currentAction = '正在处理任务'
        }
        break

      case 'text': {
        const content = typeof event.content === 'string' ? event.content : ''
        agent.status = 'thinking'
        agent.currentAction = '正在生成文本'
        agent.currentTarget = null
        agent.lastText = content
        agent.lastOutput = truncateText(content, 180)
        agent.updatedAt = now
        limitPush(agent.textStream, {
          id: uniqueId('hud-text'),
          timestamp: now,
          content,
        }, MAX_TEXT_STREAM)
        timelineEvent = {
          id: uniqueId('hud-text-event'),
          timestamp: now,
          agentId: agent.agentId,
          reqId: meta.reqId,
          role: agent.role,
          type: 'text',
          summary: `输出文本：${truncateText(content, 120)}`,
          status: agent.status,
        }
        break
      }

      case 'tool_use': {
        const tool = typeof event.tool === 'string' ? event.tool : 'unknown_tool'
        const input = (event.input && typeof event.input === 'object') ? event.input as Record<string, unknown> : undefined
        const toolState = deriveToolState(tool, input)

        agent.status = toolState.status
        agent.currentAction = toolState.action
        agent.currentTarget = toolState.target
        agent.currentTool = tool
        agent.currentCommand = toolState.command
        agent.updatedAt = now

        limitPush(agent.toolHistory, {
          id: String(event.id ?? uniqueId('tool')),
          tool,
          status: 'running',
          inputPreview: previewValue(input),
          startedAt: now,
        }, MAX_TOOL_HISTORY)

        if (toolState.fileAction && toolState.target) {
          limitPush(agent.files, {
            id: uniqueId('file'),
            timestamp: now,
            action: toolState.fileAction,
            path: toolState.target,
            source: tool,
          }, MAX_FILE_HISTORY)
        }

        if (toolState.command) {
          limitPush(agent.commands, {
            id: uniqueId('cmd'),
            timestamp: now,
            phase: 'start',
            command: toolState.command,
          }, MAX_COMMAND_HISTORY)
        }

        timelineEvent = {
          id: uniqueId('hud-tool-start'),
          timestamp: now,
          agentId: agent.agentId,
          reqId: meta.reqId,
          role: agent.role,
          type: 'tool_use',
          summary: `${toolState.action}`,
          status: agent.status,
          target: agent.currentTarget,
          tool,
          command: toolState.command,
          path: toolState.fileAction ? toolState.target : null,
        }
        break
      }

      case 'tool_result': {
        const toolCallId = typeof event.tool_use_id === 'string' ? event.tool_use_id : ''
        const toolEntry = [...agent.toolHistory].reverse().find(item => item.id === toolCallId)
        const resultPreview = previewValue(event.result)
        const failed = Boolean(event.is_error)

        if (toolEntry) {
          toolEntry.status = failed ? 'failed' : 'success'
          toolEntry.resultPreview = resultPreview
          toolEntry.endedAt = now
        }

        agent.status = failed ? 'failed' : 'running'
        agent.currentAction = failed ? '工具执行失败' : '工具执行完成'
        agent.currentTarget = toolEntry?.tool ?? agent.currentTarget
        agent.lastOutput = resultPreview || agent.lastOutput
        agent.updatedAt = now
        if (failed) {
          agent.error = { message: resultPreview || '工具失败' }
        }

        const pathValues = Array.from(collectPathLikeValues(event.result))
        for (const pathValue of pathValues.slice(0, 6)) {
          limitPush(agent.artifacts, {
            id: uniqueId('artifact'),
            timestamp: now,
            path: pathValue,
            kind: guessArtifactKind(pathValue),
            source: toolEntry?.tool ?? 'tool_result',
          }, MAX_ARTIFACTS)
        }

        timelineEvent = {
          id: uniqueId('hud-tool-end'),
          timestamp: now,
          agentId: agent.agentId,
          reqId: meta.reqId,
          role: agent.role,
          type: 'tool_result',
          summary: failed
            ? `工具失败：${toolEntry?.tool ?? toolCallId}`
            : `工具完成：${toolEntry?.tool ?? toolCallId}`,
          status: agent.status,
          target: pathValues[0] ?? agent.currentTarget,
          tool: toolEntry?.tool ?? null,
          isError: failed,
        }
        break
      }

      case 'tool_error':
        agent.status = 'failed'
        agent.currentAction = '工具抛出异常'
        agent.currentTarget = typeof event.tool === 'string' ? event.tool : agent.currentTarget
        agent.error = {
          message: typeof event.error === 'string' ? event.error : '工具执行失败',
        }
        agent.updatedAt = now
        timelineEvent = {
          id: uniqueId('hud-tool-error'),
          timestamp: now,
          agentId: agent.agentId,
          reqId: meta.reqId,
          role: agent.role,
          type: 'tool_error',
          summary: `工具异常：${typeof event.tool === 'string' ? event.tool : 'unknown_tool'}`,
          status: agent.status,
          target: agent.currentTarget,
          tool: typeof event.tool === 'string' ? event.tool : null,
          isError: true,
        }
        break

      case 'agent_error':
        agent.status = 'failed'
        agent.currentAction = 'Agent 发生错误'
        agent.error = {
          message: typeof event.error === 'string' ? event.error : 'Agent 处理异常',
          phase: typeof event.phase === 'string' ? event.phase : undefined,
          severity: typeof event.severity === 'string' ? event.severity : undefined,
        }
        agent.updatedAt = now
        timelineEvent = {
          id: uniqueId('hud-agent-error'),
          timestamp: now,
          agentId: agent.agentId,
          reqId: meta.reqId,
          role: agent.role,
          type: 'agent_error',
          summary: `Agent 错误：${agent.error.message}`,
          status: agent.status,
          target: agent.currentTarget,
          isError: true,
        }
        break

      case 'error':
        agent.status = 'failed'
        agent.currentAction = '请求处理失败'
        agent.error = {
          message: typeof event.message === 'string' ? event.message : '服务端错误',
          phase: 'system',
          severity: 'error',
        }
        agent.updatedAt = now
        timelineEvent = {
          id: uniqueId('hud-fatal-error'),
          timestamp: now,
          agentId: agent.agentId,
          reqId: meta.reqId,
          role: agent.role,
          type: 'error',
          summary: `请求失败：${agent.error.message}`,
          status: agent.status,
          target: agent.currentTarget,
          isError: true,
        }
        break

      case 'done':
        agent.status = 'done'
        agent.currentAction = '处理完成'
        agent.currentTarget = null
        agent.currentTool = null
        agent.currentCommand = null
        agent.completedAt = now
        agent.updatedAt = now
        timelineEvent = {
          id: uniqueId('hud-done'),
          timestamp: now,
          agentId: agent.agentId,
          reqId: meta.reqId,
          role: agent.role,
          type: 'done',
          summary: '处理完成',
          status: agent.status,
        }
        break
    }

    if (timelineEvent) {
      this.pushTimeline(agent, timelineEvent)
    }

    this.broadcast(agent, timelineEvent)
  }

  private ensureAgent(meta: HudRequestMeta, now: number): HudAgentState {
    const existing = this.agents.get(meta.agentId)
    if (existing) {
      existing.reqId = meta.reqId
      existing.userId = meta.userId
      existing.mode = meta.mode
      existing.role = deriveRole(meta.mode)
      existing.isNewSession = meta.isNewSession
      existing.messagePreview = truncateText(meta.message, 180)
      existing.workingDir = meta.workingDir
      existing.outputsPath = meta.outputsPath
      existing.notebookPath = meta.notebookPath
      existing.updatedAt = now
      return existing
    }

    const created: HudAgentState = {
      agentId: meta.agentId,
      reqId: meta.reqId,
      role: deriveRole(meta.mode),
      userId: meta.userId,
      mode: meta.mode,
      status: 'idle',
      isNewSession: meta.isNewSession,
      messagePreview: truncateText(meta.message, 180),
      workingDir: meta.workingDir,
      outputsPath: meta.outputsPath,
      notebookPath: meta.notebookPath,
      currentAction: null,
      currentTarget: null,
      currentTool: null,
      currentCommand: null,
      lastText: null,
      lastOutput: null,
      lastHeartbeatAt: null,
      heartbeatCount: 0,
      startedAt: now,
      updatedAt: now,
      completedAt: null,
      error: null,
      textStream: [],
      toolHistory: [],
      files: [],
      commands: [],
      artifacts: [],
      timeline: [],
    }

    this.agents.set(meta.agentId, created)
    return created
  }

  private pushTimeline(agent: HudAgentState, event: HudTimelineEvent): void {
    limitPush(agent.timeline, event, MAX_AGENT_TIMELINE)
    limitPush(this.timeline, event, MAX_TIMELINE)
  }

  private getStats(): HudStats {
    const agents = Array.from(this.agents.values())
    const runningStatuses: HudStatus[] = [
      'thinking',
      'running',
      'calling_tool',
      'reading_file',
      'writing_file',
      'running_command',
    ]
    return {
      totalAgents: agents.length,
      activeAgents: agents.filter(agent => !['done', 'failed'].includes(agent.status)).length,
      runningAgents: agents.filter(agent => runningStatuses.includes(agent.status)).length,
      waitingAgents: agents.filter(agent => agent.status === 'waiting').length,
      failedAgents: agents.filter(agent => agent.status === 'failed').length,
      doneAgents: agents.filter(agent => agent.status === 'done').length,
      connectedWatchers: this.listeners.size,
      updatedAt: Date.now(),
    }
  }

  private broadcast(agent: HudAgentState, timelineEvent?: HudTimelineEvent): void {
    const payload: HudBroadcastEvent = {
      type: 'hud_update',
      generatedAt: Date.now(),
      stats: this.getStats(),
      agent: cloneAgentState(agent),
      timelineEvent: timelineEvent ? { ...timelineEvent } : undefined,
    }

    for (const listener of this.listeners) {
      listener(payload)
    }
  }
}

export const hudManager = new HudManager()
