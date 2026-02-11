# SSE 事件格式文档

## 传输格式

所有事件通过 `POST /api/chat/stream` 以 `text/event-stream; charset=utf-8` 返回，每条消息格式：

```
data: <JSON>\n\n
```

---

## 1. `start`

会话开始，每次请求的第一个事件。

```jsonc
{
  "type": "start",
  "agentId": "agt-xxxxxxxx",   // Agent 实例 ID，多轮对话需回传
  "isNewSession": true,         // true=新建会话, false=复用已有会话
  "timestamp": 1707500000000
}
```
注意，`agentId` 是本次会话的唯一标识，后续事件需原样回传以关联会话状态。
如果 `isNewSession` 为 `false`，表示当前请求复用了已有会话，服务端会根据 `agentId` 恢复之前的会话状态，这对于多轮对话非常有用，可以避免重复执行前面的工具调用，同时保证上下文记忆。

## 2. `heartbeat`

每 2 秒发送一次，保持连接不被中间网关断开。

```jsonc
{
  "type": "heartbeat",
  "message": "processing",
  "count": 1,                   // 从 1 递增
  "timestamp": 1707500002000
}
```

## 3. `text`

AI 生成的文本增量，客户端需拼接所有 `content` 得到完整回复。

```jsonc
{
  "type": "text",
  "content": "这是一段增量文本...",
  "timestamp": 1707500003000
}
```

## 4. `tool_use`

工具调用开始。

```jsonc
{
  "type": "tool_use",
  "tool": "ocean_preprocess_full",
  "id": "call_xxxxxxxx",
  "message": "启动预处理流程...",
  "input": { ... },
  "timestamp": 1707500004000
}
```

| 字段 | 说明 |
|------|------|
| `tool` | 工具名称 |
| `id` | 调用唯一 ID，与后续 `tool_result.tool_use_id` 对应 |
| `message` | 人类可读的操作描述 |
| `input` | 工具输入参数预览，可能为 `undefined` |

## 5. `tool_result`

工具执行完毕。

```jsonc
{
  "type": "tool_result",
  "tool_use_id": "call_xxxxxxxx",
  "result": { "status": "success", "message": "..." },
  "is_error": false,
  "timestamp": 1707500005000
}
```

| 字段 | 说明 |
|------|------|
| `tool_use_id` | 对应 `tool_use` 事件的 `id` |
| `result` | 转换后的工具结果（见下方） |
| `is_error` | `true` 表示工具执行状态为 FAILED |

### `result` 通用结构

```jsonc
{
  "status": "success",   // "success" | "failed"
  "message": "..."
}
```
注意，`status` 字段表示工具执行状态，`is_error=false` 不代表工具执行成功，需结合 `status` 判断。
只有当异常被throw抛出时，才会触发 `tool_error` 事件，此时 `status` 应为 `failed`，并且 `is_error` 为 `true`。
而如果工具执行完成但业务逻辑失败（如模型调用成功但返回错误），则 `status` 也应为 `failed`，但 `is_error` 为 `false`。

### `result` 文件工具（`fs_*`）额外字段

```jsonc
{
  "status": "success",
  "message": "写入文件成功: /data/output.npy，写入 1024 字节",
  "modified": true,              // 文件是否被修改
  "paths": ["/data/output.npy"]  // 被修改的文件路径列表
}
```

注意，应该优先使用 `modified` 字段判断文件是否被修改，因为有些工具（如 `fs_read`）虽然执行成功但不修改文件。

## 6. `tool_error`

工具执行过程中抛出异常（区别于 `tool_result` 中 `is_error=true` 的正常失败返回）。

```jsonc
{
  "type": "tool_error",
  "tool": "bash_run",
  "error": "Command execution timeout",
  "timestamp": 1707500006000
}
```
注意，即使发生工具异常，`tool_result` 事件仍会正常返回，且 `is_error` 为 `true`，以便客户端统一处理工具结果。

## 7. `error`

服务端不可恢复的错误。

```jsonc
{
  "type": "error",
  "error": "INTERNAL_ERROR",
  "message": "Internal server error",
  "timestamp": 1707500007000
}
```

`error` 取值：`"INTERNAL_ERROR"` | `"REQUEST_TIMEOUT"`

## 8. `done`

本次请求处理完毕，是最后一个事件。

```jsonc
{
  "type": "done",
  "metadata": {
    "agentId": "agt-xxxxxxxx",
    "timestamp": 1707500008000
  }
}
```
