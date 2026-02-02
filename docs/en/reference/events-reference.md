# Events Reference

Complete reference for all KODE SDK events organized by channel.

---

## Event Channels

| Channel | Purpose | Subscriber |
|---------|---------|------------|
| `progress` | Streaming output (text, tool calls) | User interface |
| `control` | Permission requests and decisions | Business logic |
| `monitor` | System observability | Monitoring/logging |

---

## Progress Events

Events for streaming output to users.

### ProgressTextChunkStartEvent

Emitted when text streaming begins.

```typescript
interface ProgressTextChunkStartEvent {
  channel: 'progress';
  type: 'text_chunk_start';
  step: number;
  bookmark?: Bookmark;
}
```

### ProgressTextChunkEvent

Emitted for each text chunk during streaming.

```typescript
interface ProgressTextChunkEvent {
  channel: 'progress';
  type: 'text_chunk';
  step: number;
  delta: string;           // Text chunk content
  bookmark?: Bookmark;
}
```

### ProgressTextChunkEndEvent

Emitted when text streaming completes.

```typescript
interface ProgressTextChunkEndEvent {
  channel: 'progress';
  type: 'text_chunk_end';
  step: number;
  text: string;            // Complete text
  bookmark?: Bookmark;
}
```

### ProgressThinkChunkStartEvent

Emitted when thinking/reasoning streaming begins.

```typescript
interface ProgressThinkChunkStartEvent {
  channel: 'progress';
  type: 'think_chunk_start';
  step: number;
  bookmark?: Bookmark;
}
```

### ProgressThinkChunkEvent

Emitted for each thinking chunk.

```typescript
interface ProgressThinkChunkEvent {
  channel: 'progress';
  type: 'think_chunk';
  step: number;
  delta: string;           // Thinking chunk content
  bookmark?: Bookmark;
}
```

### ProgressThinkChunkEndEvent

Emitted when thinking streaming completes.

```typescript
interface ProgressThinkChunkEndEvent {
  channel: 'progress';
  type: 'think_chunk_end';
  step: number;
  bookmark?: Bookmark;
}
```

### ProgressToolStartEvent

Emitted when tool execution starts.

```typescript
interface ProgressToolStartEvent {
  channel: 'progress';
  type: 'tool:start';
  call: ToolCallSnapshot;
  bookmark?: Bookmark;
}
```

### ProgressToolEndEvent

Emitted when tool execution completes.

```typescript
interface ProgressToolEndEvent {
  channel: 'progress';
  type: 'tool:end';
  call: ToolCallSnapshot;
  bookmark?: Bookmark;
}
```

### ProgressToolErrorEvent

Emitted when tool execution fails.

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

Emitted when processing completes.

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

## Control Events

Events for permission handling.

### ControlPermissionRequiredEvent

Emitted when a tool call requires approval.

```typescript
interface ControlPermissionRequiredEvent {
  channel: 'control';
  type: 'permission_required';
  call: ToolCallSnapshot;
  respond(decision: 'allow' | 'deny', opts?: { note?: string }): Promise<void>;
  bookmark?: Bookmark;
}
```

**Usage:**
```typescript
agent.on('permission_required', async (event) => {
  // Review the tool call
  console.log('Tool:', event.call.name);
  console.log('Input:', event.call.inputPreview);

  // Make decision
  await event.respond('allow', { note: 'Approved by admin' });
});
```

### ControlPermissionDecidedEvent

Emitted when a permission decision is made.

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

## Monitor Events

Events for system observability.

### MonitorStateChangedEvent

Emitted when Agent state changes.

```typescript
interface MonitorStateChangedEvent {
  channel: 'monitor';
  type: 'state_changed';
  state: AgentRuntimeState;   // 'READY' | 'WORKING' | 'PAUSED'
  bookmark?: Bookmark;
}
```

### MonitorStepCompleteEvent

Emitted when a processing step completes.

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

Emitted when an error occurs.

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

Emitted with token usage statistics.

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

Emitted when a tool execution completes.

```typescript
interface MonitorToolExecutedEvent {
  channel: 'monitor';
  type: 'tool_executed';
  call: ToolCallSnapshot;
  bookmark?: Bookmark;
}
```

### MonitorAgentResumedEvent

Emitted when an Agent resumes from storage.

```typescript
interface MonitorAgentResumedEvent {
  channel: 'monitor';
  type: 'agent_resumed';
  strategy: 'crash' | 'manual';
  sealed: ToolCallSnapshot[];    // Auto-sealed tool calls
  bookmark?: Bookmark;
}
```

### MonitorBreakpointChangedEvent

Emitted when breakpoint state changes.

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

Emitted when Todo list changes.

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

Emitted when a Todo reminder is triggered.

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

Emitted when a watched file changes.

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

Emitted when a reminder is sent to the model.

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

Emitted during context compression.

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

Emitted when a scheduled task triggers.

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

Emitted when tool manuals are updated.

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

Emitted when skills metadata is updated.

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

Custom events emitted by tools.

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

## Subscribing to Events

### Using `agent.on()` (Control/Monitor only)

`agent.on()` only supports Control and Monitor events.

```typescript
// Control events
agent.on('permission_required', async (event) => {
  console.log('Permission needed for:', event.call.name);
  await event.respond('allow');
});

agent.on('permission_decided', (event) => {
  console.log(`Decision: ${event.decision} by ${event.decidedBy}`);
});

// Monitor events
agent.on('error', (event) => {
  console.error(`[${event.severity}] ${event.message}`);
});

agent.on('token_usage', (event) => {
  console.log(`Tokens: ${event.totalTokens}`);
});

agent.on('tool_executed', (event) => {
  console.log(`Tool ${event.call.name} executed`);
});

agent.on('state_changed', (event) => {
  console.log(`State: ${event.state}`);
});
```

### Using `agent.subscribe()` (All channels)

For Progress events, use `agent.subscribe()`:

```typescript
for await (const envelope of agent.subscribe(['progress'])) {
  const { event } = envelope;

  switch (event.type) {
    case 'text_chunk':
      process.stdout.write(event.delta);
      break;
    case 'tool:start':
      console.log('Tool:', event.call.name);
      break;
    case 'done':
      console.log('Completed');
      break;
  }
}
```

### Using Async Iterator with `stream()`

```typescript
for await (const envelope of agent.stream('Hello')) {
  const { event } = envelope;

  switch (event.type) {
    case 'text_chunk':
      process.stdout.write(event.delta);
      break;
    case 'tool:start':
      console.log('Tool:', event.call.name);
      break;
    case 'done':
      console.log('Completed');
      break;
  }
}
```

---

## Event Type Unions

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

## References

- [Events Guide](../guides/events.md)
- [API Reference](./api.md)
- [Types Reference](./types.md)
