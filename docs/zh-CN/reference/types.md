# 类型参考

本文档提供 KODE SDK 导出的所有 TypeScript 类型参考。

---

## 消息类型

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

## 内容块

### ContentBlock

所有内容块类型的联合类型。

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

## Agent 状态类型

### AgentRuntimeState

```typescript
type AgentRuntimeState = 'READY' | 'WORKING' | 'PAUSED';
```

| 状态 | 说明 |
|------|------|
| `READY` | Agent 空闲，准备接收消息 |
| `WORKING` | Agent 正在处理消息 |
| `PAUSED` | Agent 暂停，等待权限决策 |

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

## 工具调用类型

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

| 状态 | 说明 |
|------|------|
| `PENDING` | 收到工具调用，尚未处理 |
| `APPROVAL_REQUIRED` | 等待用户审批 |
| `APPROVED` | 已批准，准备执行 |
| `EXECUTING` | 正在执行 |
| `COMPLETED` | 执行成功完成 |
| `FAILED` | 执行失败 |
| `DENIED` | 用户拒绝了工具调用 |
| `SEALED` | Resume 时自动封口 |

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

## 事件类型

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

## 快照类型

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

## Hook 类型

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

## 配置类型

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

| 模式 | 说明 |
|------|------|
| `auto` | 自动允许所有工具调用 |
| `approval` | 所有工具调用都需要审批 |
| `readonly` | 允许只读工具，其他需要审批 |

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

## Resume 类型

### ResumeStrategy

```typescript
type ResumeStrategy = 'crash' | 'manual';
```

| 策略 | 说明 |
|------|------|
| `crash` | 自动封口未完成工具，发出 `agent_resumed` 事件 |
| `manual` | 保持未完成工具不变，手动处理 |

---

## 提醒类型

### ReminderOptions

```typescript
interface ReminderOptions {
  skipStandardEnding?: boolean;
  priority?: 'low' | 'medium' | 'high';
  category?: 'file' | 'todo' | 'security' | 'performance' | 'general';
}
```

---

## E2B 类型

### E2BSandboxOptions

```typescript
interface E2BSandboxOptions {
  apiKey?: string;              // E2B API Key
  template?: string;            // 模板 ID/别名，默认 'base'
  timeoutMs?: number;           // 沙箱超时，默认 300_000
  workDir?: string;             // 工作目录，默认 '/home/user'
  envs?: Record<string, string>; // 环境变量
  metadata?: Record<string, string>; // 自定义元数据
  allowInternetAccess?: boolean; // 允许联网，默认 true
  execTimeoutMs?: number;       // 命令超时，默认 120_000
  sandboxId?: string;           // 连接已有沙箱
  domain?: string;              // API 域名
}
```

### E2BTemplateConfig

```typescript
interface E2BTemplateConfig {
  alias: string;                // 模板别名
  base: 'python' | 'node' | 'debian' | 'ubuntu' | 'custom';
  baseVersion?: string;         // 版本号
  dockerfile?: string;          // 自定义 Dockerfile
  aptPackages?: string[];       // 系统包
  pipPackages?: string[];       // Python 包
  npmPackages?: string[];       // Node.js 包
  buildCommands?: string[];     // 构建命令
  workDir?: string;             // 工作目录
  cpuCount?: number;            // CPU 核数，默认 2
  memoryMB?: number;            // 内存 MB，默认 512
}
```

---

## 参考资料

- [API 参考](./api.md)
- [事件参考](./events-reference.md)
