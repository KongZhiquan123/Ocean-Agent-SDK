# KODE-HUD 事件模型与后端缺口

## 1. 当前 Ocean-Agent-SDK 已有实时事件

当前项目已经有 SSE 流式接口：

- 接口：`POST /api/chat/stream`
- 事件文档：`docs/backend/sse-events.md`

现有事件包括：

- `start`
- `heartbeat`
- `text`
- `tool_use`
- `tool_result`
- `tool_error`
- `agent_error`
- `done`

这些事件足够支撑“流式回答”，但还不足够支撑“执行监控”。

## 2. 当前已有事件能解决什么

### 已能解决

- 任务开始没有
- Agent 是否还活着
- 文本回复是否在持续输出
- 工具有没有开始 / 结束
- 工具是否报错
- 本轮请求是否完成

### 还不能很好解决

- Agent 现在到底处于什么状态
- 它正在操作哪个文件
- 它正在跑哪个命令
- 它最近产出了哪个文件
- 它当前属于哪个角色
- 多 Agent 之间如何聚合显示

## 3. HUD 真正需要的事件层

为了做成 Mission Control，需要把“聊天事件”提升为“监控事件”。

建议新增以下事件。

## 4. 建议新增事件：status_update

用途：

- 驱动 Agent 卡片主状态
- 让前端不必自己猜测当前状态

示例：

```json
{
  "type": "status_update",
  "agentId": "agt-abc123",
  "role": "Coder",
  "status": "reading_file",
  "currentAction": "正在读取文件",
  "currentTarget": "src/server.ts",
  "timestamp": 1707500003000
}
```

## 5. 建议新增事件：file_activity

用途：

- 显式告诉前端 Agent 当前作用在哪个文件上

示例：

```json
{
  "type": "file_activity",
  "agentId": "agt-abc123",
  "role": "Coder",
  "action": "read",
  "path": "src/server.ts",
  "timestamp": 1707500004000
}
```

动作建议值：

- `read`
- `write`
- `edit`
- `create`
- `delete`

## 6. 建议新增事件：command_activity

用途：

- 显示正在运行什么命令
- 显示命令的最新输出

示例：

```json
{
  "type": "command_activity",
  "agentId": "agt-abc123",
  "role": "Coder",
  "phase": "start",
  "command": "npm run typecheck",
  "timestamp": 1707500005000
}
```

```json
{
  "type": "command_activity",
  "agentId": "agt-abc123",
  "role": "Coder",
  "phase": "stdout",
  "command": "npm run typecheck",
  "snippet": "src/server.ts: no errors found",
  "timestamp": 1707500006000
}
```

```json
{
  "type": "command_activity",
  "agentId": "agt-abc123",
  "role": "Coder",
  "phase": "end",
  "command": "npm run typecheck",
  "exitCode": 0,
  "timestamp": 1707500007000
}
```

## 7. 建议新增事件：artifact_created

用途：

- 告诉前端“刚刚产生了什么”

示例：

```json
{
  "type": "artifact_created",
  "agentId": "agt-abc123",
  "role": "Coder",
  "path": "/outputs/report.md",
  "kind": "report",
  "timestamp": 1707500008000
}
```

建议 `kind`：

- `file`
- `report`
- `image`
- `model`
- `notebook`
- `data`
- `log`

## 8. 建议新增事件：todo_update

用途：

- 让前端直接看到 Agent 当前子任务和进度变化

示例：

```json
{
  "type": "todo_update",
  "agentId": "agt-abc123",
  "role": "Planner",
  "todos": [
    { "content": "分析 SSE 结构", "status": "completed" },
    { "content": "定义 HUD 数据模型", "status": "in_progress" },
    { "content": "设计前端看板布局", "status": "pending" }
  ],
  "timestamp": 1707500009000
}
```

## 9. 建议新增事件：child_agent_update

用途：

- 未来支持多角色 / 多 Agent 看板

示例：

```json
{
  "type": "child_agent_update",
  "parentAgentId": "agt-parent-1",
  "agentId": "agt-child-1",
  "role": "Researcher",
  "status": "running",
  "timestamp": 1707500010000
}
```

## 10. 前端如何从现有事件推导基础状态

即使后端还没补齐监控事件，前端也可以先从现有事件做一版投影。

### 现有事件到状态映射

- `start` -> `running`
- `heartbeat` -> `running`
- `text` -> `thinking` 或 `responding`
- `tool_use` -> `calling_tool`
- `tool_result` -> `running`
- `tool_error` -> `failed`
- `agent_error` -> `failed`
- `done` -> `done`

### 现有事件到 currentAction 的粗映射

- `text` -> `正在生成文本`
- `tool_use` -> `正在调用工具`
- `tool_result` -> `工具执行完成`
- `heartbeat` -> `正在处理任务`

但这层映射不够细，因为看不出：

- 正在读文件
- 正在写文件
- 正在执行命令
- 正在跑训练

所以真正落 HUD，还是建议补监控事件。

## 11. 建议后端补充方式

优先建议不另开一套复杂传输协议，而是：

- 继续使用现有 SSE
- 在 SSE 中追加新的监控事件类型
- 前端统一按事件类型做状态归约

这样改动最小，也最贴合当前项目结构。

## 12. 建议新增的最小闭环

如果只做最有价值的最小补充，我建议先加这四个：

- `status_update`
- `file_activity`
- `command_activity`
- `artifact_created`

只要这四个补上，HUD 的价值就会立刻明显上来。

