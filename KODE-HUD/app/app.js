const state = {
  apiKey: localStorage.getItem('kodeHud.apiKey') || '',
  connectionState: 'disconnected',
  detailTab: 'live',
  selectedAgentId: null,
  agents: new Map(),
  timeline: [],
  stats: null,
  hudAbortController: null,
  launchAbortController: null,
  reconnectTimer: null,
}

const el = {
  apiKeyInput: document.querySelector('#apiKeyInput'),
  connectButton: document.querySelector('#connectButton'),
  disconnectButton: document.querySelector('#disconnectButton'),
  connectionState: document.querySelector('#connectionState'),
  lastUpdated: document.querySelector('#lastUpdated'),
  watcherCount: document.querySelector('#watcherCount'),
  globalStats: document.querySelector('#globalStats'),
  agentBoard: document.querySelector('#agentBoard'),
  timeline: document.querySelector('#timeline'),
  detailAgentMeta: document.querySelector('#detailAgentMeta'),
  detailSummary: document.querySelector('#detailSummary'),
  detailContent: document.querySelector('#detailContent'),
  detailTabs: [...document.querySelectorAll('.detail-tab')],
  launchForm: document.querySelector('#launchForm'),
  launchLog: document.querySelector('#launchLog'),
  messageInput: document.querySelector('#messageInput'),
  modeInput: document.querySelector('#modeInput'),
  userIdInput: document.querySelector('#userIdInput'),
  workingDirInput: document.querySelector('#workingDirInput'),
  outputsPathInput: document.querySelector('#outputsPathInput'),
  notebookPathInput: document.querySelector('#notebookPathInput'),
  statusFilter: document.querySelector('#statusFilter'),
  searchInput: document.querySelector('#searchInput'),
}

const STATUS_LABELS = {
  idle: 'Idle',
  thinking: 'Thinking',
  running: 'Running',
  calling_tool: 'Calling Tool',
  reading_file: 'Reading File',
  writing_file: 'Writing File',
  running_command: 'Running Command',
  waiting: 'Waiting',
  done: 'Done',
  failed: 'Failed',
}

function escapeHtml(input) {
  return String(input ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;')
}

function formatTime(timestamp) {
  if (!timestamp) return '-'
  return new Date(timestamp).toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function formatDuration(startedAt, completedAt = null) {
  if (!startedAt) return '-'
  const end = completedAt || Date.now()
  const totalSec = Math.max(0, Math.floor((end - startedAt) / 1000))
  const hours = Math.floor(totalSec / 3600)
  const minutes = Math.floor((totalSec % 3600) / 60)
  const seconds = totalSec % 60
  return [hours, minutes, seconds].map((part) => String(part).padStart(2, '0')).join(':')
}

function truncate(input, max = 140) {
  const value = String(input ?? '').replace(/\s+/g, ' ').trim()
  if (value.length <= max) return value
  return value.slice(0, max - 1) + '…'
}

function updateConnectionState(nextState, note = '') {
  state.connectionState = nextState
  const labels = {
    disconnected: '未连接',
    connecting: '连接中',
    connected: '已连接',
    error: note ? `连接失败：${note}` : '连接失败',
  }
  el.connectionState.textContent = labels[nextState] || nextState
}

function getHeaders() {
  return {
    'X-API-Key': state.apiKey,
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: {
      ...(options.headers || {}),
      ...getHeaders(),
    },
  })

  if (!response.ok) {
    let detail = response.statusText
    try {
      const payload = await response.json()
      detail = payload.message || payload.error || detail
    } catch {
      // ignore
    }
    throw new Error(detail)
  }

  return response.json()
}

async function readSseStream(response, onEvent) {
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    let boundary = buffer.indexOf('\n\n')

    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      boundary = buffer.indexOf('\n\n')

      const dataLines = rawEvent
        .split(/\r?\n/)
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.slice(5).trim())

      if (dataLines.length === 0) continue

      try {
        onEvent(JSON.parse(dataLines.join('\n')))
      } catch (error) {
        console.error('[KODE-HUD] failed to parse SSE event', error)
      }
    }
  }
}

function normalizeAgent(agent) {
  return {
    ...agent,
    toolHistory: agent.toolHistory || [],
    files: agent.files || [],
    commands: agent.commands || [],
    artifacts: agent.artifacts || [],
    textStream: agent.textStream || [],
    timeline: agent.timeline || [],
  }
}

function mergeAgent(agent) {
  const normalized = normalizeAgent(agent)
  state.agents.set(normalized.agentId, normalized)
  if (!state.selectedAgentId) {
    state.selectedAgentId = normalized.agentId
  }
}

function applySnapshot(snapshot) {
  state.agents = new Map()
  for (const agent of snapshot.agents || []) {
    mergeAgent(agent)
  }
  state.timeline = snapshot.timeline || []
  state.stats = snapshot.stats || null
  if (!state.selectedAgentId && snapshot.agents?.length) {
    state.selectedAgentId = snapshot.agents[0].agentId
  }
  render()
}

function applyHudUpdate(payload) {
  if (payload.agent) {
    mergeAgent(payload.agent)
  }

  if (payload.timelineEvent) {
    state.timeline.push(payload.timelineEvent)
    if (state.timeline.length > 300) {
      state.timeline.splice(0, state.timeline.length - 300)
    }
  }

  state.stats = payload.stats || state.stats
  el.lastUpdated.textContent = formatTime(payload.generatedAt)
  render()
}

async function connectHud() {
  if (!state.apiKey) {
    updateConnectionState('error', '缺少 API Key')
    return
  }

  disconnectHud({ silent: true })
  updateConnectionState('connecting')

  try {
    const snapshot = await fetchJson('/api/hud/state')
    applySnapshot(snapshot)
    updateConnectionState('connected')
  } catch (error) {
    updateConnectionState('error', error.message)
    return
  }

  const controller = new AbortController()
  state.hudAbortController = controller

  try {
    const response = await fetch('/api/hud/events', {
      method: 'GET',
      headers: getHeaders(),
      signal: controller.signal,
    })

    if (!response.ok) {
      throw new Error(`HUD stream error: ${response.status}`)
    }

    await readSseStream(response, (event) => {
      if (event.type === 'hud_update') {
        applyHudUpdate(event)
        return
      }
      if (event.type === 'hud_connected') {
        updateConnectionState('connected')
        return
      }
    })
  } catch (error) {
    if (controller.signal.aborted) return
    updateConnectionState('error', error.message)
    clearTimeout(state.reconnectTimer)
    state.reconnectTimer = setTimeout(connectHud, 2500)
  }
}

function disconnectHud({ silent = false } = {}) {
  if (state.hudAbortController) {
    state.hudAbortController.abort()
    state.hudAbortController = null
  }
  clearTimeout(state.reconnectTimer)
  state.reconnectTimer = null
  if (!silent) {
    updateConnectionState('disconnected')
  }
}

async function launchTask(event) {
  event.preventDefault()

  if (!state.apiKey) {
    updateLaunchLog('请先填写 API Key 并连接 HUD。')
    return
  }

  if (state.launchAbortController) {
    state.launchAbortController.abort()
  }

  const message = el.messageInput.value.trim()
  if (!message) {
    updateLaunchLog('消息不能为空。')
    return
  }

  const payload = {
    message,
    mode: el.modeInput.value,
    outputsPath: el.outputsPathInput.value.trim(),
    context: {
      userId: el.userIdInput.value.trim(),
      workingDir: el.workingDirInput.value.trim(),
      notebookPath: el.notebookPathInput.value.trim(),
      files: [],
    },
  }

  const controller = new AbortController()
  state.launchAbortController = controller
  updateLaunchLog('任务已提交，等待 SSE 返回 start 事件...')

  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getHeaders(),
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    })

    if (!response.ok) {
      throw new Error(`Launch failed: ${response.status}`)
    }

    let launchAgentId = null
    const textBuffer = []

    await readSseStream(response, (sse) => {
      if (sse.type === 'start') {
        launchAgentId = sse.agentId
        updateLaunchLog(`任务已启动。\nagentId: ${launchAgentId}`)
        return
      }

      if (sse.type === 'text' && typeof sse.content === 'string') {
        textBuffer.push(sse.content)
        updateLaunchLog(`任务进行中...\nagentId: ${launchAgentId || 'pending'}\n\n最近文本:\n${textBuffer.join('').slice(-1200)}`)
        return
      }

      if (sse.type === 'tool_error' || sse.type === 'agent_error' || sse.type === 'error') {
        updateLaunchLog(`任务出现错误。\nagentId: ${launchAgentId || 'pending'}\n${JSON.stringify(sse, null, 2)}`)
        return
      }

      if (sse.type === 'done') {
        updateLaunchLog(`任务已完成。\nagentId: ${launchAgentId || sse.metadata?.agentId || 'unknown'}`)
      }
    })
  } catch (error) {
    if (controller.signal.aborted) {
      updateLaunchLog('任务流已被中断。')
      return
    }
    updateLaunchLog(`任务发起失败：${error.message}`)
  } finally {
    state.launchAbortController = null
  }
}

function updateLaunchLog(text) {
  el.launchLog.textContent = text
}

function getVisibleAgents() {
  const search = el.searchInput.value.trim().toLowerCase()
  const statusFilter = el.statusFilter.value
  const agents = Array.from(state.agents.values())
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .filter((agent) => {
      if (statusFilter !== 'all' && agent.status !== statusFilter) {
        return false
      }

      if (!search) return true

      const haystack = [
        agent.agentId,
        agent.role,
        agent.currentAction,
        agent.currentTarget,
        agent.lastOutput,
        agent.lastText,
        agent.currentTool,
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase()

      return haystack.includes(search)
    })

  return agents
}

function renderStats() {
  const stats = state.stats || {
    totalAgents: 0,
    activeAgents: 0,
    runningAgents: 0,
    waitingAgents: 0,
    failedAgents: 0,
    doneAgents: 0,
    connectedWatchers: 0,
  }

  el.watcherCount.textContent = String(stats.connectedWatchers || 0)

  const tiles = [
    ['Total Agents', stats.totalAgents],
    ['Active', stats.activeAgents],
    ['Running', stats.runningAgents],
    ['Waiting', stats.waitingAgents],
    ['Failed', stats.failedAgents],
    ['Done', stats.doneAgents],
  ]

  el.globalStats.innerHTML = tiles
    .map(([label, value]) => `
      <div class="stat-tile">
        <span>${escapeHtml(label)}</span>
        <strong>${escapeHtml(value)}</strong>
      </div>
    `)
    .join('')
}

function renderBoard() {
  const agents = getVisibleAgents()

  if (agents.length === 0) {
    el.agentBoard.innerHTML = `
      <div class="empty-state">
        当前没有可展示的 Agent。<br />
        你可以从左侧 Quick Launch 发起任务，或连接现有的 Ocean-Agent-SDK 实例观察实时事件。
      </div>
    `
    return
  }

  el.agentBoard.innerHTML = agents
    .map((agent) => {
      const isActive = agent.agentId === state.selectedAgentId
      const heartbeat = agent.lastHeartbeatAt ? formatTime(agent.lastHeartbeatAt) : '-'
      return `
        <article class="agent-card ${isActive ? 'active' : ''}" data-agent-id="${escapeHtml(agent.agentId)}">
          <div class="agent-card-header">
            <div>
              <div class="agent-role">${escapeHtml(agent.role)}</div>
              <div class="agent-id">${escapeHtml(agent.agentId)}</div>
            </div>
            <span class="badge status-${escapeHtml(agent.status)}">${escapeHtml(STATUS_LABELS[agent.status] || agent.status)}</span>
          </div>

          <div class="agent-main">
            <h3>${escapeHtml(agent.currentAction || '等待事件')}</h3>
            <p class="agent-target">${escapeHtml(agent.currentTarget || agent.messagePreview || '暂无目标对象')}</p>
            <p class="agent-output">${escapeHtml(agent.lastOutput || agent.lastText || '暂无最新产出')}</p>
          </div>

          <div class="agent-card-row">
            <span>Last heartbeat: ${escapeHtml(heartbeat)}</span>
            <span>${escapeHtml(formatDuration(agent.startedAt, agent.completedAt))}</span>
          </div>
        </article>
      `
    })
    .join('')

  for (const card of el.agentBoard.querySelectorAll('.agent-card')) {
    card.addEventListener('click', () => {
      state.selectedAgentId = card.dataset.agentId
      render()
    })
  }
}

function renderTimeline() {
  const items = [...state.timeline].sort((a, b) => b.timestamp - a.timestamp).slice(0, 120)

  if (items.length === 0) {
    el.timeline.innerHTML = `<div class="empty-state">还没有实时事件。连接 HUD 后，新的 Agent 事件会在这里滚动出现。</div>`
    return
  }

  el.timeline.innerHTML = items
    .map((item) => `
      <div class="timeline-item">
        <div class="timeline-main">
          <strong>${escapeHtml(item.summary)}</strong>
          <span>${escapeHtml(item.role)} · ${escapeHtml(item.agentId)}${item.target ? ` · ${escapeHtml(item.target)}` : ''}</span>
        </div>
        <div class="mono">${escapeHtml(formatTime(item.timestamp))}</div>
      </div>
    `)
    .join('')
}

function renderDetail() {
  const selected = state.selectedAgentId ? state.agents.get(state.selectedAgentId) : null

  if (!selected) {
    el.detailAgentMeta.textContent = '未选中 Agent'
    el.detailSummary.innerHTML = `<div class="empty-state">点击中间任意 Agent 卡片，这里会显示它的详细执行过程。</div>`
    el.detailContent.innerHTML = ''
    return
  }

  el.detailAgentMeta.textContent = `${selected.role} · ${selected.agentId}`
  el.detailSummary.innerHTML = `
    <div class="detail-summary-grid">
      <div class="summary-block">
        <span>Status</span>
        <strong>${escapeHtml(STATUS_LABELS[selected.status] || selected.status)}</strong>
      </div>
      <div class="summary-block">
        <span>Action</span>
        <strong>${escapeHtml(selected.currentAction || '-')}</strong>
      </div>
      <div class="summary-block">
        <span>Target</span>
        <strong>${escapeHtml(selected.currentTarget || '-')}</strong>
      </div>
      <div class="summary-block">
        <span>Runtime</span>
        <strong>${escapeHtml(formatDuration(selected.startedAt, selected.completedAt))}</strong>
      </div>
      <div class="summary-block">
        <span>User</span>
        <strong>${escapeHtml(selected.userId)}</strong>
      </div>
      <div class="summary-block">
        <span>Message</span>
        <strong>${escapeHtml(selected.messagePreview || '-')}</strong>
      </div>
    </div>
  `

  const tab = state.detailTab
  let rows = []

  if (tab === 'live') {
    rows = [...selected.textStream]
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, 30)
      .map((entry) => ({
        title: formatTime(entry.timestamp),
        body: entry.content,
        meta: selected.currentTarget || selected.currentTool || '',
      }))
  } else if (tab === 'tools') {
    rows = [...selected.toolHistory]
      .sort((a, b) => (b.endedAt || b.startedAt) - (a.endedAt || a.startedAt))
      .map((entry) => ({
        title: `${entry.tool} · ${entry.status}`,
        body: entry.resultPreview || entry.inputPreview || '无预览',
        meta: `${formatTime(entry.startedAt)}${entry.endedAt ? ` → ${formatTime(entry.endedAt)}` : ''}`,
      }))
  } else if (tab === 'files') {
    rows = [...selected.files]
      .sort((a, b) => b.timestamp - a.timestamp)
      .map((entry) => ({
        title: `${entry.action.toUpperCase()} · ${entry.path}`,
        body: entry.source,
        meta: formatTime(entry.timestamp),
      }))
  } else if (tab === 'commands') {
    rows = [...selected.commands]
      .sort((a, b) => b.timestamp - a.timestamp)
      .map((entry) => ({
        title: `${entry.phase.toUpperCase()} · ${truncate(entry.command, 120)}`,
        body: entry.snippet || (entry.exitCode !== undefined ? `exitCode=${entry.exitCode}` : '命令事件'),
        meta: formatTime(entry.timestamp),
      }))
  } else if (tab === 'artifacts') {
    rows = [...selected.artifacts]
      .sort((a, b) => b.timestamp - a.timestamp)
      .map((entry) => ({
        title: `${entry.kind} · ${entry.path}`,
        body: entry.source,
        meta: formatTime(entry.timestamp),
      }))
  }

  if (rows.length === 0) {
    el.detailContent.innerHTML = `<div class="empty-state">当前标签下还没有数据。</div>`
    return
  }

  el.detailContent.innerHTML = rows
    .map((row) => `
      <div class="list-row">
        <div class="list-main">
          <strong>${escapeHtml(row.title)}</strong>
          <span>${escapeHtml(row.body)}</span>
        </div>
        <div class="mono">${escapeHtml(row.meta)}</div>
      </div>
    `)
    .join('')
}

function render() {
  renderStats()
  renderBoard()
  renderTimeline()
  renderDetail()
}

function bindEvents() {
  el.apiKeyInput.value = state.apiKey

  el.connectButton.addEventListener('click', () => {
    state.apiKey = el.apiKeyInput.value.trim()
    localStorage.setItem('kodeHud.apiKey', state.apiKey)
    connectHud()
  })

  el.disconnectButton.addEventListener('click', () => {
    disconnectHud()
  })

  el.launchForm.addEventListener('submit', launchTask)

  el.statusFilter.addEventListener('change', render)
  el.searchInput.addEventListener('input', render)

  for (const button of el.detailTabs) {
    button.addEventListener('click', () => {
      state.detailTab = button.dataset.tab
      for (const tab of el.detailTabs) {
        tab.classList.toggle('active', tab === button)
      }
      renderDetail()
    })
  }
}

bindEvents()
render()

if (state.apiKey) {
  connectHud()
}
