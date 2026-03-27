# KODE-HUD

KODE-HUD 是面向 Ocean-Agent-SDK 的 Agent 实时可视化监控界面设计文档集合。

它的目标不是做一个普通聊天窗口，而是做一个 `Agent Live Dashboard / Mission Control`：

- 实时看到每个 Agent 当前在做什么
- 实时看到每个 Agent 正在读什么文件、写什么文件、运行什么命令或脚本
- 实时看到每个 Agent 最近产出了什么文件、日志、报告或结果
- 在一个看板里同时观察多个角色或多个 Agent 的工作进度

## 文档索引

- [01-product-definition.md](/home/lz/Ocean-Agent-SDK/KODE-HUD/01-product-definition.md)
  - 产品定义、目标、范围、MVP
- [02-ui-information-architecture.md](/home/lz/Ocean-Agent-SDK/KODE-HUD/02-ui-information-architecture.md)
  - 页面结构、卡片字段、详情面板、交互方式
- [03-event-model-and-backend-gap.md](/home/lz/Ocean-Agent-SDK/KODE-HUD/03-event-model-and-backend-gap.md)
  - 当前 Ocean-Agent-SDK 已有事件、缺口、建议新增事件模型
- [04-implementation-roadmap.md](/home/lz/Ocean-Agent-SDK/KODE-HUD/04-implementation-roadmap.md)
  - 从当前 SDK 落地到 HUD 的实施路线

## 这一版文档回答的核心问题

1. KODE-HUD 到底是什么，不是什么
2. 页面上应该持续显示哪些信息
3. 当前 Ocean-Agent-SDK 已经具备哪些实时事件基础
4. 还需要补哪些事件，才能真正做到“看到 Agent 正在做什么”
5. 最合理的落地顺序应该是什么

## 当前结论

KODE-HUD 的核心不是“更好看的聊天 UI”，而是“Agent 执行过程可观测性”。

一句话定义：

`KODE-HUD 是一个让用户在网页上持续看到 Agent 当前状态、当前动作、当前操作对象、以及最新产出的实时监控面板。`

## 当前 Ocean-Agent-SDK 可直接复用的基础

当前项目已经有 SSE 流式能力，可作为 HUD 的第一阶段实时数据通道：

- 对话流接口：`POST /api/chat/stream`
- 已有事件：`start`、`heartbeat`、`text`、`tool_use`、`tool_result`、`tool_error`、`agent_error`、`done`

相关代码和文档位置：

- [README.md](/home/lz/Ocean-Agent-SDK/README.md)
- [src/server.ts](/home/lz/Ocean-Agent-SDK/src/server.ts)
- [src/agent-manager.ts](/home/lz/Ocean-Agent-SDK/src/agent-manager.ts)
- [docs/backend/sse-events.md](/home/lz/Ocean-Agent-SDK/docs/backend/sse-events.md)

## 建议落地顺序

第一阶段：

- 先做单 Agent HUD
- 直接消费现有 SSE
- 显示文本流、工具调用、心跳、完成状态
- 补最必要的 `currentAction` / `currentTarget`

第二阶段：

- 做多 Agent / 多角色看板
- 引入统一状态聚合层
- 新增文件活动、命令活动、产出物事件

第三阶段：

- 做完整的任务级 Mission Control
- 支持父子 Agent、角色编排、回放、过滤、审计

