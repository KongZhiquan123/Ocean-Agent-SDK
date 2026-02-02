# Resume / Fork 指南

长时运行的 Agent 必须具备"随时恢复、可分叉、可审计"的能力。KODE SDK 在内核层实现了统一的持久化协议（消息、工具调用、Todo、事件、断点、Lineage）。

---

## 关键概念

| 概念 | 说明 |
|------|------|
| **Metadata** | 序列化模板、工具描述符、权限、Todo、沙箱配置、断点、lineage 等信息 |
| **Safe-Fork-Point (SFP)** | 每次用户消息或工具结果都会形成可恢复节点，用于 snapshot/fork |
| **BreakpointState** | 标记当前执行阶段（`READY` → `PRE_MODEL` → ... → `POST_TOOL`） |
| **Auto-Seal** | 当崩溃发生在工具执行阶段，Resume 会自动封口并落下 `tool_result` |

---

## Resume 方式

### 方式一：显式配置

```typescript
import { Agent } from '@shareai-lab/kode-sdk';

const agent = await Agent.resume('agt-demo', {
  templateId: 'repo-assistant',
  modelConfig: {
    provider: 'anthropic',
    model: process.env.ANTHROPIC_MODEL_ID ?? 'claude-sonnet-4-20250514',
    apiKey: process.env.ANTHROPIC_API_KEY!,
  },
  sandbox: { kind: 'local', workDir: './workspace', enforceBoundary: true },
}, deps, {
  strategy: 'crash',  // 自动封口未完成工具
  autoRun: true,      // 恢复后继续处理队列
});
```

### 方式二：从 Store 恢复（推荐）

```typescript
const agent = await Agent.resumeFromStore('agt-demo', deps, {
  overrides: {
    modelConfig: {
      provider: 'anthropic',
      model: process.env.ANTHROPIC_MODEL_ID ?? 'claude-sonnet-4-20250514',
      apiKey: process.env.ANTHROPIC_API_KEY!,
    },
  },
});
```

### Resume 选项

| 选项 | 取值 | 说明 |
|------|------|------|
| `strategy` | `'manual'` \| `'crash'` | `crash` 会自动封口未完成工具 |
| `autoRun` | `boolean` | 恢复后立即继续处理消息队列 |
| `overrides` | `Partial<AgentConfig>` | 对 metadata 进行覆盖（模型升级、权限调整等） |

> **重要**：Resume 后**必须**重新绑定事件监听（Control/Monitor 回调不会自动恢复）。

---

## SDK vs 业务方的职责分界

| 能力 | SDK | 业务方 |
|------|-----|--------|
| 模板、工具、沙箱恢复 | 自动重建 | 无需处理 |
| 消息、工具记录、Todo、Lineage | 自动加载 | 无需处理 |
| FilePool 监听 | 自动恢复 | 无需处理 |
| Hooks | 自动重新注册 | 无需处理 |
| Control/Monitor 监听 | 不处理 | Resume 后需重新绑定 |
| 审批流程、告警 | 不处理 | 结合业务系统处理 |
| 依赖单例管理 | 不处理 | 确保 `store`/`registry` 全局复用 |

---

## 快照与分叉

### 创建快照

```typescript
// 在当前点创建快照
const bookmarkId = await agent.snapshot('pre-release-audit');
```

### 分叉 Agent

```typescript
// 从快照分叉
const forked = await agent.fork(bookmarkId);

// 从最新点分叉
const forked2 = await agent.fork();

// 使用分叉的 Agent
await forked.send('这是一个基于原对话分叉出的新任务。');
```

- `snapshot(label?)` 返回 `SnapshotId`（默认为 `sfp-{index}`）
- `fork(sel?)` 创建新 Agent：继承工具/权限/lineage，把消息复制到新 Store 命名空间
- 分叉后的 Agent 需要独立绑定事件

---

## 自动封口机制

当崩溃发生在以下阶段，Resume 会自动写入补偿性的 `tool_result`：

| 阶段 | 封口信息 | 推荐处理 |
|------|---------|---------|
| `PENDING` | 工具尚未执行 | 验证参数后重新触发 |
| `APPROVAL_REQUIRED` | 等待审批 | 再次触发审批或手动完成 |
| `APPROVED` | 准备执行 | 确认输入仍然有效后重试 |
| `EXECUTING` | 执行中断 | 检查副作用，必要时人工确认 |

封口会触发：

- `monitor.agent_resumed`：包含 `sealed` 列表与 `strategy`
- `progress.tool:end`：补上一条失败的 `tool_result`，附带 `recommendations`

---

## Resume 后重新绑定事件

```typescript
const agent = await Agent.resumeFromStore('agt-demo', deps);

// 重新绑定 Control/Monitor 事件监听
agent.on('tool_executed', (event) => {
  console.log('工具执行:', event.call.name);
});

agent.on('error', (event) => {
  console.error('错误:', event.message);
});

agent.on('permission_required', async (event) => {
  await event.respond('allow');
});

// 对于 Progress 事件，使用 subscribe()
const progressSubscription = (async () => {
  for await (const envelope of agent.subscribe(['progress'])) {
    if (envelope.event.type === 'text_chunk') {
      process.stdout.write(envelope.event.delta);
    }
    if (envelope.event.type === 'done') break;
  }
})();

// 继续处理
await agent.run();
await progressSubscription;
```

---

## 多实例 / Serverless 最佳实践

1. **依赖单例**：在模块级创建 `AgentDependencies`，避免多个实例写入同一 Store 目录

2. **事件重绑**：每次 `resume` 后立刻绑定事件

3. **并发控制**：同一个 AgentId 最好只在单实例中运行，可通过外部锁或队列保证

4. **持久化目录**：`JSONStore` 适用于单机/共享磁盘环境。分布式部署请实现自定义 Store（如 S3 + DynamoDB）

5. **可观测性**：监听 `monitor.state_changed` 与 `monitor.error`，在异常时迅速定位

---

## 故障排查

| 现象 | 排查方向 |
|------|---------|
| Resume 报 `AGENT_NOT_FOUND` | Store 目录缺失或未持久化。确认 `store.baseDir` 是否正确挂载 |
| Resume 报 `TEMPLATE_NOT_FOUND` | 启动时未注册模板；确保模板 ID 与 metadata 中一致 |
| 工具缺失 | ToolRegistry 未注册对应名称；内置工具需手动注册 |
| FilePool 未恢复 | 自定义 Sandbox 未实现 `watchFiles`；可关闭 watch 或补齐实现 |
| 事件监听失效 | Resume 后未重新调用 `agent.on(...)` 绑定 |

---

## 完整 Resume 示例

```typescript
import { Agent, createExtendedStore } from '@shareai-lab/kode-sdk';

async function resumeAgent(agentId: string) {
  const store = await createExtendedStore();
  const deps = createDependencies({ store });

  // 检查 Agent 是否存在
  const exists = await store.exists(agentId);
  if (!exists) {
    throw new Error(`Agent ${agentId} 不存在`);
  }

  // 从 store 恢复
  const agent = await Agent.resumeFromStore(agentId, deps, {
    strategy: 'crash',
    autoRun: false,
  });

  // 重新绑定 Monitor 事件监听（on() 仅支持 Control/Monitor 事件）
  agent.on('tool_executed', (e) => console.log('工具:', e.call.name));
  agent.on('agent_resumed', (e) => {
    if (e.sealed.length > 0) {
      console.log('自动封口的工具:', e.sealed);
    }
  });
  agent.on('error', (e) => console.error('错误:', e.message));

  // 对于 Progress 事件，使用 subscribe()
  const progressTask = (async () => {
    for await (const env of agent.subscribe(['progress'])) {
      if (env.event.type === 'text_chunk') {
        process.stdout.write(env.event.delta);
      }
      if (env.event.type === 'done') break;
    }
  })();

  // 继续处理
  await agent.run();

  return agent;
}
```

---

## 参考资料

- [事件系统指南](./events.md)
- [错误处理指南](./error-handling.md)
- [数据库指南](./database.md)
