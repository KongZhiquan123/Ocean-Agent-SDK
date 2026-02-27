# Agent Service

基于 KODE SDK 实现的 AI Agent HTTP 服务，提供 RESTful API 接口供后端调用。

## 项目结构

```
Ocean-Agent-SDK/
├── src/                   # 服务主代码（Agent、工具、服务端）
├── docs/                  # 多语言文档
├── scripts/               # 数据预处理、校验与训练脚本
├── work_ocean/            # 示例数据、坏例与规则
├── test-client.ts         # 客户端 SSE 调用示例
├── test-ocean-badcases.ts # 海洋预处理坏例测试用例
├── package.json           # 项目依赖与脚本
├── tsconfig.json          # TypeScript 编译配置
├── README.md              # 项目说明
└── CLAUDE.md              # Claude 模型使用说明
```

## 快速开始

### 1. 安装依赖

```bash
npm install
cd ./scripts/ocean_SR_training_masked
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 并填写：

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

编辑 `.env` 文件：

```env
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_BASE_URL=your-anthropic-endpoint
KODE_API_SECRET=your-secret-key
ANTHROPIC_MODEL_ID=your-model-id
KODE_API_PORT=8787
SKILLS_DIR=your-skills-directory
PYTHON3=your-python3-path
```
`.env` 文件中的变量会被自动加载到 `process.env`。
### 3. 启动服务

```bash
# 启动服务
npm run start

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
- `outputsPath` (string, 必需)：输出文件根目录，所有生成的文件都必须在此目录下
- `agentId` (string, 可选)：会话 ID，用于延续多轮对话
- `context` (object, 必需)：上下文信息
  - `userId` (string, 必需)：用户 ID
  - `workingDir` (string, 必需)：工作目录
  - `notebookPath` (string, 必需)：Jupyter Notebook 路径
  - `files` (string[], 可选)：相关文件列表

**响应（SSE 事件流）：**

服务器会通过 Server-Sent Events (SSE) 返回多个事件：

```
data: {"type":"start","agentId":"agt-abc123","timestamp":1706889600000}

data: {"type":"text","content":"我来帮你创建 hello.py 文件。","timestamp":1706889600000}

data: {"type":"tool_use","tool":"fs_write","id":"toolu_xyz","input":{"path":"hello.py","content":"print('Hello, World!')"},"timestamp":1706889600000}

data: {"type":"tool_result","tool_use_id":"toolu_xyz","result":"{\"ok\":true,\"path\":\"hello.py\"}","is_error":false,"timestamp":1706889600000}

data: {"type":"tool_error","tool":"fs_write","error":"Permission denied","timestamp":1706889600000}

data: {"type":"agent_error","error":"Agent 处理异常","phase":"execution","severity":"error","timestamp":1706889600000}

data: {"type":"text","content":"文件已创建成功！","timestamp":1706889600000}

data: {"type":"done","metadata":{"agentId":"agt-abc123","timestamp":1706889600000}}
```

**事件类型：**
- `start`：开始处理，返回 `agentId`
- `heartbeat`：心跳（每 2 秒）
- `text`：AI 生成的文本内容
- `tool_use`：工具调用开始，包含输入参数
- `tool_result`：工具调用结果
- `tool_error`：工具调用失败（如权限不足、执行错误）
- `agent_error`：Agent 内部错误（如监控报错）
- `done`：处理完成
- `error`：发生严重错误

## 模式对比

### Edit 模式（编程助手）

- 可以读写文件
- 可以执行 Shell 命令
- 可以管理 Todo 列表
- 适合代码生成、文件操作等任务

**可用工具：**

1. **通用工具**
   - 文件操作：`fs_read`, `fs_write`, `fs_edit`, `fs_glob`, `fs_grep`
   - 系统命令：`bash_run`
   - 任务管理：`todo_read`, `todo_write`
   - 技能管理：`skills`

2. **海洋超分数据预处理 (Ocean SR Data Preprocess)**
   - `ocean_inspect_data`: 数据检查
   - `ocean_validate_tensor`: 张量校验
   - `ocean_sr_preprocess_convert_npy`: 格式转换
   - `ocean_sr_preprocess_full`: 全流程处理
   - `ocean_sr_preprocess_downsample`: 下采样
   - `ocean_sr_preprocess_visualize`: 可视化
   - `ocean_sr_preprocess_metrics`: 指标计算
   - `ocean_sr_preprocess_report`: 报告生成

3. **海洋超分训练 (Ocean SR Training)**
   - `ocean_sr_gpu_check`: GPU 检查
   - `ocean_sr_model_list`: 模型列表
   - `ocean_sr_train_start`: 模型训练
   - `ocean_sr_train_status`: 训练状态查询
   - `ocean_sr_train_report`: 训练报告生成
   - `ocean_sr_train_visualize`: 训练可视化

4. **海洋时序预测数据预处理 (Ocean Forecast Data Preprocess)**
   - `ocean_inspect_data`: 数据检查（共用）
   - `ocean_forecast_preprocess_full`: 全流程处理
   - `ocean_forecast_preprocess_visualize`: 可视化
   - `ocean_forecast_preprocess_report`: 报告生成
   - `ocean_forecast_preprocess_stats`: 统计分析

### Ask 模式（问答助手）

- 只能读取文件
- 只能执行只读命令
- 不能修改任何内容
- 适合代码解释、问题回答等任务

**可用工具：**
- `fs_read`
- `fs_glob`, `fs_grep`
- `bash_run`（只读命令）
- `ocean_inspect_data`（只读数据检查）

## 客户端示例

### cURL

```bash
curl -X POST http://localhost:8787/api/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "message": "请创建一个 Python 脚本来计算斐波那契数列",
    "mode": "edit",
    "outputsPath": "./test_outputs",
    "context": {
      "userId": "test-user",
      "workingDir": "./work_ocean",
      "notebookPath": "./work_ocean/work_ocean.ipynb"
    }
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
    outputsPath: './test_outputs',
    context: {
       userId: 'test-user',
       workingDir: './work_ocean',
       notebookPath: './work_ocean/work_ocean.ipynb',
    }
  }),
})

const reader = response.body.getReader()
const decoder = new TextDecoder()

while (true) {
  const { done, value } = await reader.read()
  if (done) break

  const text = decoder.decode(value, { stream: true })
  const lines = text.split('\n')

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const jsonStr = line.slice(6)
      if (!jsonStr.trim()) continue
      
      try {
        const event = JSON.parse(jsonStr)
        
        switch (event.type) {
          case 'start':
            console.log('开始处理，Agent ID:', event.agentId)
            break
          case 'text':
            process.stdout.write(event.content)
            break
          case 'tool_use':
            console.log(`\n[调用工具] ${event.tool}`)
            break
          case 'tool_result':
            console.log(`[工具结果] ${event.is_error ? '失败' : '成功'}`)
            break
          case 'tool_error':
            console.error(`[工具错误] ${event.tool}: ${event.error}`)
            break
          case 'done':
            console.log('\n处理完成')
            break
        }
      } catch (e) {
        console.error('解析错误:', e)
      }
    }
  }
}
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
**工具中的任何文件操作和命令执行，请使用 `ctx.sandbox`提供的沙箱环境。**

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

### 自定义skill
在 `skills/` 目录下创建新的 skill 文件夹，添加 `metadata.json` 和`SKILL.md` 文件：

```
skills/
  my_skill/
    metadata.json
    SKILL.md
```
详情可见 [技能开发指南](./docs/zh-CN/guides/skills.md)

**注意，SKILL.md必须使用LF换行符，否则会导致YAML FORMATTER解析失败！**

## 环境要求

- **Node.js**: >= 20.18.1
- **KODE SDK**: ^2.7.2

## 🔧 常见问题

### Q1: 服务启动失败

**检查清单：**
- ✅ 是否安装了依赖？运行 `npm install`
- ✅ 是否配置了 `.env` 文件？
- ✅ `ANTHROPIC_API_KEY` 是否有效？
- ✅ 端口 8787 是否被占用？

### Q2: API 请求返回 401

确保在请求头中添加了正确的 `X-API-Key`：

```bash
-H "X-API-Key: your-secret-key"
```

密钥应该与 `.env` 文件中的 `KODE_API_SECRET` 一致。

### Q3: 文件没有生成

- ✅ 确保使用了 `edit` 模式（不是 `ask`）
- ✅ 查看服务器日志，确认工具是否执行成功

### Q4: CLAUDE.md文件会被kode-sdk传给Anthropic Claude模型吗？
- 不会，`CLAUDE.md`文件仅作为本项目的使用说明文档，不会被传递给Anthropic Claude模型。
- `CLAUDE.md`文件仅仅用于本地使用Claude Code进行开发时参考使用。