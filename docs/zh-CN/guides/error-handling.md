# 错误处理指南

KODE SDK 实现了完整的错误处理机制，遵循三个核心原则：

1. **模型感知错误** - 所有错误信息对模型可见且可操作
2. **程序永不崩溃** - 多层错误捕获，确保系统稳定运行
3. **完整可观测性** - 所有错误触发事件，方便监控和调试

---

## 错误类型

| 错误类型 | 标识 | 可重试 | 典型场景 |
|---------|------|--------|---------|
| `validation` | `_validationError: true` | 否 | 参数类型错误、必填参数缺失 |
| `runtime` | `_thrownError: true` | 是 | 文件不存在、权限不足、网络错误 |
| `logical` | 工具返回 `{ok: false}` | 是 | 文件内容不匹配、命令执行失败 |
| `aborted` | 超时/中断 | 否 | 工具执行超时、用户中断 |
| `exception` | 未预期异常 | 是 | 系统异常、未知错误 |

---

## 错误流转

```
工具执行
  ├─ 参数验证失败 → {ok: false, error: ..., _validationError: true}
  ├─ 执行抛异常 → {ok: false, error: ..., _thrownError: true}
  ├─ 返回 {ok: false} → 保持原样（逻辑错误）
  └─ 正常返回 → 保持原样
     ↓
Agent 处理
  ├─ 识别错误类型：validation | runtime | logical | aborted | exception
  ├─ 判断可重试性：validation不可重试，其他可重试
  ├─ 生成智能建议：基于错误类型和工具名称
  ├─ 发出 tool:error 事件（ProgressEvent - 用户可见）
  └─ 发出 error 事件（MonitorEvent - 监控系统）
     ↓
返回给模型
  └─ {
       ok: false,
       error: "具体错误信息",
       errorType: "错误类型",
       retryable: true/false,
       recommendations: ["建议1", "建议2", ...]
     }
```

---

## 监听错误

### Progress 事件（用户层）

```typescript
// 监听工具错误用于 UI
agent.on('tool:error', (event) => {
  console.log('工具错误:', event.error);
  console.log('工具状态:', event.call.state);
  // 显示 UI 通知
});

// 使用流
for await (const envelope of agent.stream(input)) {
  if (envelope.event.type === 'tool:error') {
    showNotification({
      type: 'error',
      message: envelope.event.error,
    });
  }
}
```

### Monitor 事件（系统层）

```typescript
// 监听所有错误
agent.on('error', (event) => {
  if (event.phase === 'tool') {
    const { errorType, retryable } = event.detail || {};

    // 记录到日志系统
    logger.warn('Tool Error', {
      message: event.message,
      errorType,
      retryable,
      severity: event.severity,
      timestamp: Date.now(),
    });

    // 发送告警
    if (event.severity === 'error') {
      alerting.send('工具执行失败', event);
    }
  }
});
```

---

## 模型自我调整

### 场景：文件不存在

**工具返回：**
```json
{
  "ok": false,
  "error": "File not found: /src/utils/helper.ts",
  "errorType": "logical",
  "retryable": true,
  "recommendations": [
    "确认文件路径是否正确",
    "使用 fs_glob 搜索文件",
    "检查文件是否被外部修改"
  ]
}
```

**模型分析：**
1. `errorType: "logical"` - 不是参数问题，是文件确实不存在
2. `retryable: true` - 可以尝试其他方案
3. 建议提到"确认文件路径"

**模型调整策略：**
```
1. 使用 fs_glob("src/**/*.ts") 查找所有 ts 文件
2. 使用 fs_grep("helper", "src/**/*.ts") 搜索包含 helper 的文件
3. 找到正确的文件路径后继续操作
```

### 场景：参数验证错误

**工具返回：**
```json
{
  "ok": false,
  "error": "Invalid parameters: path is required",
  "errorType": "validation",
  "retryable": false,
  "recommendations": [
    "检查工具参数是否符合 schema 要求",
    "确认所有必填参数已提供",
    "检查参数类型是否正确"
  ]
}
```

**模型调整策略：**
```
1. 检查工具调用，发现缺少 path 参数
2. 补充必要的 path 参数
3. 重新调用工具
```

---

## 多层防护机制

```
第1层：工具执行层 (tool.ts)
  └─ try-catch 捕获所有异常 → {ok: false, _thrownError: true}

第2层：Agent调用层 (agent.ts)
  └─ try-catch 捕获调用异常 → errorType: 'exception'

第3层：参数验证层
  └─ safeParse 避免验证异常 → {ok: false, _validationError: true}

第4层：Hook执行层
  └─ Hook失败不影响主流程 → 记录错误继续执行
```

### 错误隔离原则

- 单个工具错误 ≠ Agent 崩溃
- Agent 错误 ≠ 系统崩溃
- 工具间完全隔离
- 所有错误可追踪

---

## 最佳实践

### 工具开发者

```typescript
// ✅ 推荐：使用 {ok: false} 返回预期的业务错误
if (!fileExists) {
  return {
    ok: false,
    error: '文件未找到',
    recommendations: ['检查文件路径', '使用 fs_glob 搜索文件'],
  };
}

// ❌ 避免：抛出异常表示业务错误
throw new Error('文件未找到');  // 应该只用于意外异常
```

### 应用开发者

```typescript
// 监听错误并做 UI 提示
agent.on('tool:error', (event) => {
  showNotification({
    type: 'error',
    message: event.error,
    action: event.call.state === 'FAILED' ? 'retry' : null,
  });
});

// 智能重试逻辑
if (result.status === 'paused' && result.permissionIds?.length) {
  // 有 pending 权限，等待用户决策
} else if (lastError?.retryable && retryCount < 3) {
  // 可重试错误，自动重试
  await agent.send('请根据建议调整后重试');
}
```

### 系统运维

```typescript
// 错误统计和分析
const errorStats = {
  validation: 0,
  runtime: 0,
  logical: 0,
  aborted: 0,
  exception: 0,
};

agent.on('error', (event) => {
  if (event.phase === 'tool') {
    const type = event.detail?.errorType || 'unknown';
    errorStats[type]++;

    // 定期分析错误模式
    if (errorStats.validation > 100) {
      alert('参数验证错误过多，请检查工具 schema 配置');
    }
  }
});
```

---

## 错误事件类型

### ProgressToolErrorEvent

```typescript
interface ProgressToolErrorEvent {
  channel: 'progress';
  type: 'tool:error';
  call: ToolCallSnapshot;  // 工具调用快照
  error: string;           // 错误信息
  bookmark?: Bookmark;
}
```

### MonitorErrorEvent

```typescript
interface MonitorErrorEvent {
  channel: 'monitor';
  type: 'error';
  severity: 'warn' | 'error';
  phase: 'model' | 'tool' | 'sandbox' | 'system';
  message: string;
  detail?: {
    errorType?: string;
    retryable?: boolean;
    [key: string]: any;
  };
}
```

---

## 总结

错误处理机制提供：

**模型智能感知**
- 错误类型明确（validation/runtime/logical/aborted/exception）
- 可重试性清晰（retryable: true/false）
- 建议具体可操作（根据工具和错误类型定制）

**系统稳定性**
- 工具层 try-catch 兜底
- Agent层 try-catch 保护
- 参数验证 safeParse
- Hook执行隔离

**完整可观测性**
- Progress 事件（tool:error）- 用户可见
- Monitor 事件（error）- 系统记录
- 工具记录（ToolCallRecord）- 完整审计
- 事件时间线（EventBus）- 可回溯

---

## 参考资料

- [事件系统指南](./events.md)
- [工具系统指南](./tools.md)
- [Resume/Fork 指南](./resume-fork.md)
