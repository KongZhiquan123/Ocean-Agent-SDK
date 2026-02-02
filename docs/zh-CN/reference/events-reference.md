# 事件参考

KODE SDK 所有事件的完整参考，按通道组织。

---

## 事件通道

| 通道 | 用途 | 订阅者 |
|------|------|--------|
| `progress` | 流式输出（文本、工具调用）| 用户界面 |
| `control` | 权限请求和决策 | 业务逻辑 |
| `monitor` | 系统可观测性 | 监控/日志 |

---

## Progress 事件

用于向用户流式输出的事件。

### ProgressTextChunkStartEvent

文本流开始时发出。

```typescript
interface ProgressTextChunkStartEvent {
  channel: 'progress';
  type: 'text_chunk_start';
  step: number;
  bookmark?: Bookmark;
}
```

### ProgressTextChunkEvent

流式传输时每个文本块发出。

```typescript
interface ProgressTextChunkEvent {
  channel: 'progress';
  type: 'text_chunk';
  step: number;
  delta: string;           // 文本块内容
  bookmark?: Bookmark;
}
```

### ProgressTextChunkEndEvent

文本流完成时发出。

```typescript
interface ProgressTextChunkEndEvent {
  channel: 'progress';
  type: 'text_chunk_end';
  step: number;
  text: string;            // 完整文本
  bookmark?: Bookmark;
}
```

### ProgressThinkChunkStartEvent

思考/推理流开始时发出。

```typescript
interface ProgressThinkChunkStartEvent {
  channel: 'progress';
  type: 'think_chunk_start';
  step: number;
  bookmark?: Bookmark;
}
```

### ProgressThinkChunkEvent

每个思考块发出。

```typescript
interface ProgressThinkChunkEvent {
  channel: 'progress';
  type: 'think_chunk';
  step: number;
  delta: string;           // 思考块内容
  bookmark?: Bookmark;
}
```

### ProgressThinkChunkEndEvent

思考流完成时发出。

```typescript
interface ProgressThinkChunkEndEvent {
  channel: 'progress';
  type: 'think_chunk_end';
  step: number;
  bookmark?: Bookmark;
}
```

### ProgressToolStartEvent

工具执行开始时发出。

```typescript
interface ProgressToolStartEvent {
  channel: 'progress';
  type: 'tool:start';
  call: ToolCallSnapshot;
  bookmark?: Bookmark;
}
```

### ProgressToolEndEvent

工具执行完成时发出。

```typescript
interface ProgressToolEndEvent {
  channel: 'progress';
  type: 'tool:end';
  call: ToolCallSnapshot;
  bookmark?: Bookmark;
}
```

### ProgressToolErrorEvent

工具执行失败时发出。

```typescript
interface ProgressToolErrorEvent {
  channel: 'progress';
  type: 'tool:error';
  call: ToolCallSnapshot;
  error: string;
  bookmark?: Bookmark;
}
```

### ProgressDoneEvent

处理完成时发出。

```typescript
interface ProgressDoneEvent {
  channel: 'progress';
  type: 'done';
  step: number;
  reason: 'completed' | 'interrupted';
  bookmark?: Bookmark;
}
```

---

## Control 事件

用于权限处理的事件。

### ControlPermissionRequiredEvent

工具调用需要审批时发出。

```typescript
interface ControlPermissionRequiredEvent {
  channel: 'control';
  type: 'permission_required';
  call: ToolCallSnapshot;
  respond(decision: 'allow' | 'deny', opts?: { note?: string }): Promise<void>;
  bookmark?: Bookmark;
}
```

**用法：**
```typescript
agent.on('permission_required', async (event) => {
  // 审查工具调用
  console.log('工具:', event.call.name);
  console.log('输入:', event.call.inputPreview);

  // 做出决策
  await event.respond('allow', { note: '管理员批准' });
});
```

### ControlPermissionDecidedEvent

权限决策完成时发出。

```typescript
interface ControlPermissionDecidedEvent {
  channel: 'control';
  type: 'permission_decided';
  callId: string;
  decision: 'allow' | 'deny';
  decidedBy: string;
  note?: string;
  bookmark?: Bookmark;
}
```

---

## Monitor 事件

用于系统可观测性的事件。

### MonitorStateChangedEvent

Agent 状态变化时发出。

```typescript
interface MonitorStateChangedEvent {
  channel: 'monitor';
  type: 'state_changed';
  state: AgentRuntimeState;   // 'READY' | 'WORKING' | 'PAUSED'
  bookmark?: Bookmark;
}
```

### MonitorStepCompleteEvent

处理步骤完成时发出。

```typescript
interface MonitorStepCompleteEvent {
  channel: 'monitor';
  type: 'step_complete';
  step: number;
  durationMs?: number;
  bookmark: Bookmark;
}
```

### MonitorErrorEvent

发生错误时发出。

```typescript
interface MonitorErrorEvent {
  channel: 'monitor';
  type: 'error';
  severity: 'info' | 'warn' | 'error';
  phase: 'model' | 'tool' | 'system' | 'lifecycle';
  message: string;
  detail?: any;
  bookmark?: Bookmark;
}
```

### MonitorTokenUsageEvent

Token 使用统计。

```typescript
interface MonitorTokenUsageEvent {
  channel: 'monitor';
  type: 'token_usage';
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  bookmark?: Bookmark;
}
```

### MonitorToolExecutedEvent

工具执行完成时发出。

```typescript
interface MonitorToolExecutedEvent {
  channel: 'monitor';
  type: 'tool_executed';
  call: ToolCallSnapshot;
  bookmark?: Bookmark;
}
```

### MonitorAgentResumedEvent

Agent 从存储恢复时发出。

```typescript
interface MonitorAgentResumedEvent {
  channel: 'monitor';
  type: 'agent_resumed';
  strategy: 'crash' | 'manual';
  sealed: ToolCallSnapshot[];    // 自动封口的工具调用
  bookmark?: Bookmark;
}
```

### MonitorBreakpointChangedEvent

断点状态变化时发出。

```typescript
interface MonitorBreakpointChangedEvent {
  channel: 'monitor';
  type: 'breakpoint_changed';
  previous: BreakpointState;
  current: BreakpointState;
  timestamp: number;
  bookmark?: Bookmark;
}
```

### MonitorTodoChangedEvent

Todo 列表变化时发出。

```typescript
interface MonitorTodoChangedEvent {
  channel: 'monitor';
  type: 'todo_changed';
  current: TodoItem[];
  previous: TodoItem[];
  bookmark?: Bookmark;
}
```

### MonitorTodoReminderEvent

Todo 提醒触发时发出。

```typescript
interface MonitorTodoReminderEvent {
  channel: 'monitor';
  type: 'todo_reminder';
  todos: TodoItem[];
  reason: string;
  bookmark?: Bookmark;
}
```

### MonitorFileChangedEvent

监听的文件变化时发出。

```typescript
interface MonitorFileChangedEvent {
  channel: 'monitor';
  type: 'file_changed';
  path: string;
  mtime: number;
  bookmark?: Bookmark;
}
```

### MonitorReminderSentEvent

向模型发送提醒时发出。

```typescript
interface MonitorReminderSentEvent {
  channel: 'monitor';
  type: 'reminder_sent';
  category: 'file' | 'todo' | 'security' | 'performance' | 'general';
  content: string;
  bookmark?: Bookmark;
}
```

### MonitorContextCompressionEvent

上下文压缩期间发出。

```typescript
interface MonitorContextCompressionEvent {
  channel: 'monitor';
  type: 'context_compression';
  phase: 'start' | 'end';
  summary?: string;
  ratio?: number;
  bookmark?: Bookmark;
}
```

### MonitorSchedulerTriggeredEvent

定时任务触发时发出。

```typescript
interface MonitorSchedulerTriggeredEvent {
  channel: 'monitor';
  type: 'scheduler_triggered';
  taskId: string;
  spec: string;
  kind: 'steps' | 'time' | 'cron';
  triggeredAt: number;
  bookmark?: Bookmark;
}
```

### MonitorToolManualUpdatedEvent

工具说明书更新时发出。

```typescript
interface MonitorToolManualUpdatedEvent {
  channel: 'monitor';
  type: 'tool_manual_updated';
  tools: string[];
  timestamp: number;
  bookmark?: Bookmark;
}
```

### MonitorSkillsMetadataUpdatedEvent

技能元数据更新时发出。

```typescript
interface MonitorSkillsMetadataUpdatedEvent {
  channel: 'monitor';
  type: 'skills_metadata_updated';
  skills: string[];
  timestamp: number;
  bookmark?: Bookmark;
}
```

### MonitorToolCustomEvent

工具发出的自定义事件。

```typescript
interface MonitorToolCustomEvent {
  channel: 'monitor';
  type: 'tool_custom_event';
  toolName: string;
  eventType: string;
  data?: any;
  timestamp: number;
  bookmark?: Bookmark;
}
```

---

## 订阅事件

### 使用 `agent.on()` (仅 Control/Monitor)

`agent.on()` 仅支持 Control 和 Monitor 事件。

```typescript
// Control 事件
agent.on('permission_required', async (event) => {
  console.log('需要权限:', event.call.name);
  await event.respond('allow');
});

agent.on('permission_decided', (event) => {
  console.log(`决定: ${event.decision} 由 ${event.decidedBy}`);
});

// Monitor 事件
agent.on('error', (event) => {
  console.error(`[${event.severity}] ${event.message}`);
});

agent.on('token_usage', (event) => {
  console.log(`Tokens: ${event.totalTokens}`);
});

agent.on('tool_executed', (event) => {
  console.log(`工具 ${event.call.name} 已执行`);
});

agent.on('state_changed', (event) => {
  console.log(`状态: ${event.state}`);
});
```

### 使用 `agent.subscribe()` (所有通道)

对于 Progress 事件，请使用 `agent.subscribe()`:

```typescript
for await (const envelope of agent.subscribe(['progress'])) {
  const { event } = envelope;

  switch (event.type) {
    case 'text_chunk':
      process.stdout.write(event.delta);
      break;
    case 'tool:start':
      console.log('工具:', event.call.name);
      break;
    case 'done':
      console.log('完成');
      break;
  }
}
```

### 使用 `stream()` 异步迭代器

```typescript
for await (const envelope of agent.stream('Hello')) {
  const { event } = envelope;

  switch (event.type) {
    case 'text_chunk':
      process.stdout.write(event.delta);
      break;
    case 'tool:start':
      console.log('工具:', event.call.name);
      break;
    case 'done':
      console.log('完成');
      break;
  }
}
```

---

## 事件类型联合

### ProgressEvent

```typescript
type ProgressEvent =
  | ProgressThinkChunkStartEvent
  | ProgressThinkChunkEvent
  | ProgressThinkChunkEndEvent
  | ProgressTextChunkStartEvent
  | ProgressTextChunkEvent
  | ProgressTextChunkEndEvent
  | ProgressToolStartEvent
  | ProgressToolEndEvent
  | ProgressToolErrorEvent
  | ProgressDoneEvent;
```

### ControlEvent

```typescript
type ControlEvent =
  | ControlPermissionRequiredEvent
  | ControlPermissionDecidedEvent;
```

### MonitorEvent

```typescript
type MonitorEvent =
  | MonitorStateChangedEvent
  | MonitorStepCompleteEvent
  | MonitorErrorEvent
  | MonitorTokenUsageEvent
  | MonitorToolExecutedEvent
  | MonitorAgentResumedEvent
  | MonitorTodoChangedEvent
  | MonitorTodoReminderEvent
  | MonitorFileChangedEvent
  | MonitorReminderSentEvent
  | MonitorContextCompressionEvent
  | MonitorSchedulerTriggeredEvent
  | MonitorBreakpointChangedEvent
  | MonitorToolManualUpdatedEvent
  | MonitorSkillsMetadataUpdatedEvent
  | MonitorToolCustomEvent;
```

---

## 参考资料

- [事件系统指南](../guides/events.md)
- [API 参考](./api.md)
- [类型参考](./types.md)
