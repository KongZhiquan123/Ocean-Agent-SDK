# KODE-HUD 实施路线

## 1. 总体策略

KODE-HUD 不建议一次性做成“大而全平台”，而应该按三阶段推进：

1. 单 Agent 实时面板
2. 多 Agent 看板
3. 完整 Mission Control

## 2. 阶段一：单 Agent HUD

目标：

- 用最少改动，让前端已经能“持续看到 Agent 正在做什么”

### 后端

- 复用现有 `/api/chat/stream`
- 保留当前 SSE 事件
- 新增最小的 `status_update`
- 在工具调用前后补 `currentAction`

### 前端

- 做一个单页 HUD
- 左侧文本流
- 中间工具流
- 右侧状态卡
- 底部事件流

### 阶段一产出

- 能看到 Agent 是否活着
- 能看到当前工具调用
- 能看到最近文本输出
- 能看到当前状态

## 3. 阶段二：多 Agent 看板

目标：

- 让多个角色或多个 Agent 可以同时显示

### 后端

- 为每个 Agent 输出 role
- 提供统一事件聚合流，或者在前端聚合多个 SSE
- 新增 `child_agent_update`

### 前端

- 多卡片看板
- 可按 role 过滤
- 可按 status 过滤
- 支持选中某个 Agent 看详情

### 阶段二产出

- 一个任务下多个 Agent 同时可见
- 用户能看出谁在工作、谁在等待、谁报错

## 4. 阶段三：完整 Mission Control

目标：

- 从“看执行”升级到“看协作”

### 建议能力

- 任务树视图
- 父子 Agent 关系
- 按任务回放事件
- 产出物聚合
- 审计视图
- 卡死检测和告警

## 5. 与当前代码的接入建议

### 当前最关键的接入点

- [src/server.ts](/home/lz/Ocean-Agent-SDK/src/server.ts)
  - SSE 输出入口
- [src/agent-manager.ts](/home/lz/Ocean-Agent-SDK/src/agent-manager.ts)
  - ProgressEvent 到 SSE 的转换入口
- [docs/backend/sse-events.md](/home/lz/Ocean-Agent-SDK/docs/backend/sse-events.md)
  - 事件文档维护位置

### 最小改法

1. 在 `convertProgressToSSE()` 周围追加更细粒度的状态投影
2. 在工具调用转换器里提取文件路径、命令、输出路径
3. 在返回 SSE 时把这些信息作为独立事件或扩展字段发给前端

## 6. 建议的前端状态模型

前端不要直接把原始 SSE 生硬渲染，而要维护一个 store。

建议 store 中每个 Agent 包含：

```ts
type HudAgentState = {
  agentId: string
  role: string
  status: string
  currentAction: string | null
  currentTarget: string | null
  lastText: string | null
  lastOutput: string | null
  lastHeartbeatAt: number | null
  startedAt: number | null
  updatedAt: number | null
  toolCalls: HudToolCall[]
  files: HudFileActivity[]
  commands: HudCommandActivity[]
  artifacts: HudArtifact[]
  timeline: HudEvent[]
}
```

## 7. 第一版前端页面建议

如果你接下来要开始真正做页面，我建议第一版就做下面这些模块：

- `GlobalStatusBar`
- `AgentBoard`
- `AgentCard`
- `AgentDetailPanel`
- `LiveEventTimeline`

这样能最快形成完整闭环。

## 8. 第一版视觉方向建议

方向上不要做成普通后台管理系统。

建议：

- 有明显“运行中”氛围
- 卡片有状态色
- 事件流有持续刷新感
- 字体和层级要像监控台，不像表单页面

重点不是花哨，而是：

- 一眼能看懂
- 一秒能定位
- 十分钟连续盯着也不累

## 9. 下一步最合理动作

如果要继续往实现推进，建议按下面顺序做：

1. 先定义前端 `Agent HUD store`
2. 再定义 SSE 到 store 的 reducer
3. 再补后端缺失事件
4. 最后做 UI 组件

这比“先画页面再想数据结构”更稳。

## 10. 当前建议结论

KODE-HUD 不应该从“页面长什么样”开始，而应该从“事件和状态如何定义”开始。

因为这个项目最核心的不是静态 UI，而是“实时运行态”。

