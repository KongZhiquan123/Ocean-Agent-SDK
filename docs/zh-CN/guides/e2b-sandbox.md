# E2B 云端沙箱指南

KODE SDK 支持 [E2B](https://e2b.dev) 作为云端沙箱后端，为 AI Agent 代码执行提供完全隔离的远程 Linux 环境。

---

## 概述

| 特性 | 说明 |
|------|------|
| **完全隔离** | 每个沙箱运行在独立的微虚拟机中 |
| **快速启动** | ~150ms 创建完整 Linux 环境 |
| **开箱即用** | 预装 Python、Node.js 及常用系统工具 |
| **弹性扩展** | 每个 Agent 独立沙箱实例 |

### E2B vs 本地沙箱的选择

| 场景 | 推荐方案 |
|------|---------|
| 开发/测试 | 本地沙箱 |
| 生产环境执行不可信代码 | E2B |
| 多用户并发 Agent | E2B |
| 需要特定环境（GPU、特殊包） | E2B（自定义模板） |
| 离线/无网络环境 | 本地沙箱 |

---

## 前置条件

1. 在 [e2b.dev](https://e2b.dev) 注册账号
2. 从 Dashboard 获取 API Key
3. 设置环境变量：

```bash
export E2B_API_KEY=your-api-key
```

---

## 快速开始

### 创建并使用沙箱

```typescript
import { E2BSandbox } from '@shareai-lab/kode-sdk';

const sandbox = new E2BSandbox({
  apiKey: process.env.E2B_API_KEY,
  template: 'base',
  timeoutMs: 300_000, // 5 分钟
});
await sandbox.init();

// 执行命令
const result = await sandbox.exec('python3 -c "print(1+1)"');
console.log(result.stdout); // "2\n"

// 文件操作
await sandbox.fs.write('script.py', 'print("hello")');
const content = await sandbox.fs.read('script.py');

// 清理
await sandbox.dispose();
```

---

## 配置项

### E2BSandboxOptions

```typescript
interface E2BSandboxOptions {
  apiKey?: string;              // E2B API Key（或使用 E2B_API_KEY 环境变量）
  template?: string;            // 模板 ID/别名，默认 'base'
  timeoutMs?: number;           // 沙箱生命周期，默认 300_000（5分钟）
  workDir?: string;             // 工作目录，默认 '/home/user'
  envs?: Record<string, string>; // 环境变量
  metadata?: Record<string, string>; // 自定义元数据
  allowInternetAccess?: boolean; // 允许互联网访问，默认 true
  execTimeoutMs?: number;       // 命令超时，默认 120_000
  sandboxId?: string;           // 连接已有沙箱（用于恢复）
  domain?: string;              // E2B API 域名
}
```

### 环境变量

| 变量 | 说明 |
|------|------|
| `E2B_API_KEY` | E2B API 密钥（未提供 `apiKey` 参数时使用） |

---

## Agent 集成

### 在 Agent 中使用 E2B 沙箱

```typescript
import { Agent, E2BSandbox } from '@shareai-lab/kode-sdk';
import { createRuntime } from './shared/runtime';

const sandbox = new E2BSandbox({
  template: 'base',
  timeoutMs: 600_000,
});
await sandbox.init();

const deps = createRuntime(({ templates, registerBuiltin }) => {
  registerBuiltin('fs', 'bash', 'todo');
  templates.register({
    id: 'coder',
    systemPrompt: '你是一个编程助手。',
    tools: ['bash_run', 'fs_read', 'fs_write', 'todo_read', 'todo_write'],
  });
});

const agent = await Agent.create({ templateId: 'coder', sandbox }, deps);
await agent.send('写一个 Python 斐波那契脚本并运行');
```

### 沙箱生命周期绑定

- **Agent 启动**：在 `Agent.create()` 前调用 `sandbox.init()`
- **Agent 暂停**：沙箱保持运行（通过 `sandboxId` 重连）
- **Agent 恢复**：传入 `sandboxId` 重连已有沙箱
- **Agent 销毁**：调用 `sandbox.dispose()` 终止沙箱

### Resume / Fork 与 sandboxId 持久化

```typescript
// 首次运行 - 创建
const sandbox = new E2BSandbox({ template: 'base' });
await sandbox.init();
const id = sandbox.getSandboxId(); // 持久化此 ID

// 后续恢复
const restored = new E2BSandbox({ sandboxId: id });
await restored.init();
// 同一个沙箱环境可继续使用
```

---

## 自定义模板

### 使用 E2BTemplateBuilder

```typescript
import { E2BTemplateBuilder } from '@shareai-lab/kode-sdk';

await E2BTemplateBuilder.build({
  alias: 'data-analysis',
  base: 'python',
  baseVersion: '3.11',
  aptPackages: ['graphviz'],
  pipPackages: ['pandas', 'numpy', 'matplotlib'],
  workDir: '/workspace',
  cpuCount: 4,
  memoryMB: 2048,
});
```

### 可用基础镜像

| 类型 | 说明 |
|------|------|
| `python` | 含 pip 的 Python 环境 |
| `node` | 含 npm 的 Node.js 环境 |
| `debian` | Debian 基础镜像 |
| `ubuntu` | Ubuntu 基础镜像 |
| `custom` | 自定义 Dockerfile |

---

## 网络与端口

### 暴露端口

```typescript
const sandbox = new E2BSandbox({ template: 'node' });
await sandbox.init();

await sandbox.exec('npx serve -l 3000 &');
const url = sandbox.getHostUrl(3000);
console.log(`预览: ${url}`);
// https://3000-<sandboxId>.e2b.app
```

---

## 最佳实践

1. **超时管理**：合理设置 `timeoutMs` 控制费用，完成后及时调用 `dispose()`
2. **错误处理**：E2B 操作都是远程调用，需妥善处理网络异常
3. **数据导出**：沙箱销毁后数据丢失，重要结果需在 `dispose()` 前导出
4. **并发控制**：E2B 有账户级别的沙箱并发限制，AgentPool 场景需规划
5. **模板复用**：模板构建一次，多个沙箱复用，加速启动
