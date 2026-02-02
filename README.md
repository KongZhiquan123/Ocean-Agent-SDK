# Agent Service

基于 KODE SDK 实现的 AI Agent HTTP 服务，提供 RESTful API 接口供后端调用。

## 特性

- 🤖 基于 **KODE SDK** 实现，支持文件操作、命令执行等完整工具链
- 🔄 **SSE 流式响应**，实时返回 AI 生成的内容
- 🔐 **API Key 认证**，保护服务安全
- 📁 **工作目录隔离**，支持指定输出路径
- 🛠️ **双模式支持**：编程模式（edit）和问答模式（ask）
- 📦 **模块化架构**：配置、Agent 管理、服务器分离

## 项目结构

```
agent-trying/
├── src/
│   ├── config.ts          # 环境配置和依赖初始化
│   ├── agent-manager.ts   # Agent 创建和事件处理
│   └── server.ts          # HTTP 服务器（Express）
├── docs/                  # KODE SDK 文档
├── .env.example          # 环境变量示例
├── package.json          # 项目配置
├── README.md            # 本文件
├── COMPARISON.md        # 新旧版本对比
└── QUICKSTART.md        # 快速上手指南
```

## 快速开始

### 1. 安装依赖

```bash
npm install
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填写：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
ANTHROPIC_API_KEY=sk-ant-...
KODE_API_SECRET=your-secret-key
KODE_API_PORT=8787
```

### 3. 启动服务

```bash
# 启动服务
npm start

# 或开发模式（自动重启）
npm run dev
```

服务将在 `http://localhost:8787` 启动。

## API 接口

### 1. 健康检查

```http
GET /health
```

**响应：**
```json
{
  "status": "ok",
  "service": "kode-agent-service",
  "sdk": "kode-sdk",
  "timestamp": 1706889600000
}
```

### 2. 对话接口（SSE 流式）

```http
POST /api/chat/stream
Content-Type: application/json
X-API-Key: your-secret-key

{
  "message": "请帮我创建一个 hello.py 文件",
  "mode": "edit",
  "outputsPath": "/path/to/outputs",
  "context": {
    "userId": "user123",
    "workingDir": "/path/to/work",
    "files": ["file1.txt", "file2.py"]
  }
}
```

**请求参数：**
- `message` (string, 必需)：用户消息
- `mode` (string, 可选)：模式，`"edit"` 或 `"ask"`，默认 `"edit"`
  - `edit`：可以读写文件、执行命令（编程助手）
  - `ask`：只读模式，用于问答（问答助手）
- `outputsPath` (string, 可选)：输出文件路径
- `context` (object, 可选)：上下文信息
  - `userId` (string)：用户 ID
  - `workingDir` (string)：工作目录
  - `files` (string[])：相关文件列表

**响应（SSE 事件流）：**

服务器会通过 Server-Sent Events (SSE) 返回多个事件：

```
data: {"type":"start","agentId":"agt-abc123","timestamp":1706889600000}

data: {"type":"text","content":"我来帮你创建 hello.py 文件。","timestamp":1706889600000}

data: {"type":"tool_use","tool":"fs_write","id":"toolu_xyz","input":{"path":"hello.py","content":"print('Hello, World!')"},"timestamp":1706889600000}

data: {"type":"tool_result","tool_use_id":"toolu_xyz","result":"{\"ok\":true,\"path\":\"hello.py\"}","is_error":false,"timestamp":1706889600000}

data: {"type":"text","content":"文件已创建成功！","timestamp":1706889600000}

data: {"type":"done","metadata":{"agentId":"agt-abc123","timestamp":1706889600000}}
```

**事件类型：**
- `start`：开始处理
- `heartbeat`：心跳（每 2 秒）
- `text`：AI 生成的文本内容
- `tool_use`：工具调用开始
- `tool_result`：工具调用结果
- `done`：处理完成
- `error`：发生错误

## 模式对比

### Edit 模式（编程助手）

- 可以读写文件
- 可以执行 Shell 命令
- 可以管理 Todo 列表
- 适合代码生成、文件操作等任务

**可用工具：**
- `fs_read`, `fs_write`, `fs_edit`
- `fs_glob`, `fs_grep`
- `bash_run`
- `todo_read`, `todo_write`

### Ask 模式（问答助手）

- 只能读取文件
- 只能执行只读命令
- 不能修改任何内容
- 适合代码解释、问题回答等任务

**可用工具：**
- `fs_read`
- `fs_glob`, `fs_grep`
- `bash_run`（只读命令）

## 客户端示例

### cURL

```bash
curl -X POST http://localhost:8787/api/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "message": "请创建一个 Python 脚本来计算斐波那契数列",
    "mode": "edit",
    "outputsPath": "./outputs"
  }'
```

### JavaScript/TypeScript

```typescript
const response = await fetch('http://localhost:8787/api/chat/stream', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'your-secret-key',
  },
  body: JSON.stringify({
    message: '请帮我分析 main.py 文件',
    mode: 'ask',
  }),
})

const reader = response.body.getReader()
const decoder = new TextDecoder()

while (true) {
  const { done, value } = await reader.read()
  if (done) break

  const text = decoder.decode(value)
  const lines = text.split('\n')

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const event = JSON.parse(line.slice(6))
      console.log('Event:', event)

      if (event.type === 'text') {
        console.log('AI:', event.content)
      }
    }
  }
}
```

### Python

```python
import requests
import json

url = 'http://localhost:8787/api/chat/stream'
headers = {
    'Content-Type': 'application/json',
    'X-API-Key': 'your-secret-key'
}
data = {
    'message': '请创建一个 hello.txt 文件',
    'mode': 'edit',
    'outputsPath': './outputs'
}

with requests.post(url, headers=headers, json=data, stream=True) as response:
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                event = json.loads(line[6:])
                print('Event:', event)

                if event['type'] == 'text':
                    print('AI:', event['content'])
```

## 架构说明

### 模块化架构

项目采用模块化设计，职责清晰：

#### 1. **src/config.ts** - 配置和依赖管理
- 环境变量验证
- 创建和初始化所有依赖（Store、ToolRegistry、TemplateRegistry 等）
- 统一的依赖注入入口

#### 2. **src/agent-manager.ts** - Agent 生命周期管理
- Agent 创建和配置
- 事件处理（权限审批、错误处理）
- Progress 事件转换为 SSE 格式
- 消息处理流程（Generator 模式）

#### 3. **src/server.ts** - HTTP 服务器
- Express 应用
- 路由定义（健康检查、对话接口）
- 中间件（日志、认证）
- SSE 流式响应
- 简化的错误处理（避免深层嵌套）

### KODE SDK 核心组件

1. **Store (JSONStore)**
   - 持久化 Agent 的消息、工具调用记录等
   - 存储位置：`./.kode/`

2. **ToolRegistry**
   - 注册所有可用工具
   - 内置工具：文件系统、Bash、Todo 等

3. **AgentTemplateRegistry**
   - 定义 Agent 模板
   - 包含系统提示词、可用工具列表等

4. **SandboxFactory**
   - 创建沙箱环境
   - 隔离文件操作和命令执行

5. **ModelFactory**
   - 创建 LLM Provider
   - 当前使用 AnthropicProvider

### 事件系统

KODE SDK 使用三通道事件系统：

- **Progress**：数据面，UI 渲染（文本流、工具生命周期）
- **Control**：审批面，人工决策（权限请求）
- **Monitor**：治理面，审计告警（错误、状态变化）

本服务主要使用 Progress 通道进行流式输出。

## 开发建议

### 添加自定义工具

在 `src/config.ts` 中添加：

```typescript
import { defineTool } from '@shareai-lab/kode-sdk'

const myTool = defineTool({
  name: 'my_custom_tool',
  description: '我的自定义工具',
  params: {
    input: { type: 'string', description: '输入参数' },
  },
  async exec(args, ctx) {
    // 工具逻辑
    return { result: 'success' }
  },
})

// 在 createToolRegistry() 函数中注册
function createToolRegistry() {
  const registry = new ToolRegistry()

  // ... 其他工具

  // 注册自定义工具
  registry.register(myTool.name, () => myTool)

  return registry
}
```

### 修改系统提示词

在 `src/config.ts` 的 `createTemplateRegistry()` 函数中编辑 `systemPrompt` 字段。

### 添加权限控制

在 `src/agent-manager.ts` 的 `setupAgentHandlers()` 函数中自定义审批逻辑：

```typescript
export function setupAgentHandlers(agent: Agent, reqId: string): void {
  agent.on('permission_required', async (event: any) => {
    console.log(`工具 ${event.call.name} 需要权限批准`)

    // 自定义审批逻辑
    if (event.call.name === 'bash_run') {
      const cmd = event.call.args.cmd
      if (cmd.includes('rm -rf')) {
        await event.respond('deny', { note: '危险命令' })
        return
      }
    }

    await event.respond('allow')
  })

  // ... 其他处理
}
```

## 环境要求

- **Node.js**: >= 20.18.1
- **KODE SDK**: ^2.7.2
