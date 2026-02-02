# Types Reference

This document provides a reference for all TypeScript types exported by KODE SDK.

---

## Message Types

### MessageRole

```typescript
type MessageRole = 'user' | 'assistant' | 'system';
```

### Message

```typescript
interface Message {
  role: MessageRole;
  content: ContentBlock[];
  metadata?: MessageMetadata;
}
```

### MessageMetadata

```typescript
interface MessageMetadata {
  content_blocks?: ContentBlock[];
  transport?: 'provider' | 'text' | 'omit';
}
```

---

## Content Blocks

### ContentBlock

Union type for all content block types.

```typescript
type ContentBlock =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } }
  | { type: 'tool_use'; id: string; name: string; input: any; meta?: Record<string, any> }
  | { type: 'tool_result'; tool_use_id: string; content: any; is_error?: boolean }
  | ReasoningContentBlock
  | ImageContentBlock
  | AudioContentBlock
  | FileContentBlock;
```

### ReasoningContentBlock

```typescript
type ReasoningContentBlock = {
  type: 'reasoning';
  reasoning: string;
  meta?: Record<string, any>;
};
```

### ImageContentBlock

```typescript
type ImageContentBlock = {
  type: 'image';
  url?: string;
  file_id?: string;
  base64?: string;
  mime_type?: string;
  meta?: Record<string, any>;
};
```

### AudioContentBlock

```typescript
type AudioContentBlock = {
  type: 'audio';
  url?: string;
  file_id?: string;
  base64?: string;
  mime_type?: string;
  meta?: Record<string, any>;
};
```

### FileContentBlock

```typescript
type FileContentBlock = {
  type: 'file';
  url?: string;
  file_id?: string;
  filename?: string;
  base64?: string;
  mime_type?: string;
  meta?: Record<string, any>;
};
```

---

## Agent State Types

### AgentRuntimeState

```typescript
type AgentRuntimeState = 'READY' | 'WORKING' | 'PAUSED';
```

| State | Description |
|-------|-------------|
| `READY` | Agent is idle and ready to receive messages |
| `WORKING` | Agent is processing a message |
| `PAUSED` | Agent is paused waiting for permission decision |

### BreakpointState

```typescript
type BreakpointState =
  | 'READY'
  | 'PRE_MODEL'
  | 'STREAMING_MODEL'
  | 'TOOL_PENDING'
  | 'AWAITING_APPROVAL'
  | 'PRE_TOOL'
  | 'TOOL_EXECUTING'
  | 'POST_TOOL';
```

### AgentStatus

```typescript
interface AgentStatus {
  agentId: string;
  state: AgentRuntimeState;
  stepCount: number;
  lastSfpIndex: number;
  lastBookmark?: Bookmark;
  cursor: number;
  breakpoint: BreakpointState;
}
```

### AgentInfo

```typescript
interface AgentInfo {
  agentId: string;
  templateId: string;
  createdAt: string;
  lineage: string[];
  configVersion: string;
  messageCount: number;
  lastSfpIndex: number;
  lastBookmark?: Bookmark;
  breakpoint?: BreakpointState;
  metadata?: Record<string, any>;
}
```

---

## Tool Call Types

### ToolCallState

```typescript
type ToolCallState =
  | 'PENDING'
  | 'APPROVAL_REQUIRED'
  | 'APPROVED'
  | 'EXECUTING'
  | 'COMPLETED'
  | 'FAILED'
  | 'DENIED'
  | 'SEALED';
```

| State | Description |
|-------|-------------|
| `PENDING` | Tool call received, not yet processed |
| `APPROVAL_REQUIRED` | Waiting for user approval |
| `APPROVED` | Approved, ready to execute |
| `EXECUTING` | Currently executing |
| `COMPLETED` | Execution completed successfully |
| `FAILED` | Execution failed |
| `DENIED` | User denied the tool call |
| `SEALED` | Auto-sealed during resume |

### ToolCallRecord

```typescript
interface ToolCallRecord {
  id: string;
  name: string;
  input: any;
  state: ToolCallState;
  approval: ToolCallApproval;
  result?: any;
  error?: string;
  isError?: boolean;
  startedAt?: number;
  completedAt?: number;
  durationMs?: number;
  createdAt: number;
  updatedAt: number;
  auditTrail: ToolCallAuditEntry[];
}
```

### ToolCallSnapshot

```typescript
type ToolCallSnapshot = Pick<
  ToolCallRecord,
  'id' | 'name' | 'state' | 'approval' | 'result' | 'error' | 'isError' | 'durationMs' | 'startedAt' | 'completedAt'
> & {
  inputPreview?: any;
  auditTrail?: ToolCallAuditEntry[];
};
```

### ToolCallApproval

```typescript
interface ToolCallApproval {
  required: boolean;
  decision?: 'allow' | 'deny';
  decidedBy?: string;
  decidedAt?: number;
  note?: string;
  meta?: Record<string, any>;
}
```

### ToolCallAuditEntry

```typescript
interface ToolCallAuditEntry {
  state: ToolCallState;
  timestamp: number;
  note?: string;
}
```

### ToolOutcome

```typescript
interface ToolOutcome {
  id: string;
  name: string;
  ok: boolean;
  content: any;
  durationMs?: number;
}
```

### ToolCall

```typescript
interface ToolCall {
  id: string;
  name: string;
  args: any;
  agentId: string;
}
```

### ToolContext

```typescript
interface ToolContext {
  agentId: string;
  sandbox: Sandbox;
  agent: any;
  services?: Record<string, any>;
  signal?: AbortSignal;
  emit?: (eventType: string, data?: any) => void;
}
```

---

## Event Types

### Bookmark

```typescript
interface Bookmark {
  seq: number;
  timestamp: number;
}
```

### AgentChannel

```typescript
type AgentChannel = 'progress' | 'control' | 'monitor';
```

### AgentEvent

```typescript
type AgentEvent = ProgressEvent | ControlEvent | MonitorEvent;
```

### AgentEventEnvelope

```typescript
interface AgentEventEnvelope<T extends AgentEvent = AgentEvent> {
  cursor: number;
  bookmark: Bookmark;
  event: T;
}
```

### Timeline

```typescript
interface Timeline {
  cursor: number;
  bookmark: Bookmark;
  event: AgentEvent;
}
```

---

## Snapshot Types

### SnapshotId

```typescript
type SnapshotId = string;
```

### Snapshot

```typescript
interface Snapshot {
  id: SnapshotId;
  messages: Message[];
  lastSfpIndex: number;
  lastBookmark: Bookmark;
  createdAt: string;
  metadata?: Record<string, any>;
}
```

---

## Hook Types

### HookDecision

```typescript
type HookDecision =
  | { decision: 'ask'; meta?: any }
  | { decision: 'deny'; reason?: string; toolResult?: any }
  | { result: any }
  | void;
```

### PostHookResult

```typescript
type PostHookResult =
  | void
  | { update: Partial<ToolOutcome> }
  | { replace: ToolOutcome };
```

---

## Configuration Types

### PermissionConfig

```typescript
interface PermissionConfig {
  mode: PermissionDecisionMode;
  requireApprovalTools?: string[];
  allowTools?: string[];
  denyTools?: string[];
  metadata?: Record<string, any>;
}
```

### PermissionDecisionMode

```typescript
type PermissionDecisionMode = 'auto' | 'approval' | 'readonly' | (string & {});
```

| Mode | Description |
|------|-------------|
| `auto` | Automatically allow all tool calls |
| `approval` | Require approval for all tool calls |
| `readonly` | Allow read-only tools, require approval for others |

### SubAgentConfig

```typescript
interface SubAgentConfig {
  templates?: string[];
  depth: number;
  inheritConfig?: boolean;
  overrides?: {
    permission?: PermissionConfig;
    todo?: TodoConfig;
  };
}
```

### TodoConfig

```typescript
interface TodoConfig {
  enabled: boolean;
  remindIntervalSteps?: number;
  storagePath?: string;
  reminderOnStart?: boolean;
}
```

### SandboxConfig

```typescript
interface SandboxConfig {
  kind: SandboxKind;
  workDir?: string;
  enforceBoundary?: boolean;
  allowPaths?: string[];
  watchFiles?: boolean;
  [key: string]: any;
}
```

### SandboxKind

```typescript
type SandboxKind = 'local' | 'docker' | 'remote';
```

---

## Resume Types

### ResumeStrategy

```typescript
type ResumeStrategy = 'crash' | 'manual';
```

| Strategy | Description |
|----------|-------------|
| `crash` | Auto-seal incomplete tools and emit `agent_resumed` event |
| `manual` | Leave incomplete tools as-is for manual handling |

---

## Reminder Types

### ReminderOptions

```typescript
interface ReminderOptions {
  skipStandardEnding?: boolean;
  priority?: 'low' | 'medium' | 'high';
  category?: 'file' | 'todo' | 'security' | 'performance' | 'general';
}
```

---

## E2B Types

### E2BSandboxOptions

```typescript
interface E2BSandboxOptions {
  apiKey?: string;
  template?: string;
  timeoutMs?: number;
  workDir?: string;
  envs?: Record<string, string>;
  metadata?: Record<string, string>;
  allowInternetAccess?: boolean;
  execTimeoutMs?: number;
  sandboxId?: string;
  domain?: string;
}
```

### E2BTemplateConfig

```typescript
interface E2BTemplateConfig {
  alias: string;
  base: 'python' | 'node' | 'debian' | 'ubuntu' | 'custom';
  baseVersion?: string;
  dockerfile?: string;
  aptPackages?: string[];
  pipPackages?: string[];
  npmPackages?: string[];
  buildCommands?: string[];
  workDir?: string;
  cpuCount?: number;
  memoryMB?: number;
}
```

---

## References

- [API Reference](./api.md)
- [Events Reference](./events-reference.md)
