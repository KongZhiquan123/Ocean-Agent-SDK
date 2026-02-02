# 快速上手指南

本指南将帮助你在 5 分钟内运行起基于 KODE SDK 的 Agent 服务。

## 📦 步骤 1: 安装依赖

```bash
npm install
```

## 🔑 步骤 2: 配置环境变量

创建 `.env` 文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填写你的 Anthropic API Key：

```env
ANTHROPIC_API_KEY=sk-ant-api03-...
KODE_API_SECRET=my-secret-key-123
KODE_API_PORT=8787
```

> 💡 如何获取 Anthropic API Key：访问 https://console.anthropic.com/

## 🚀 步骤 3: 启动服务

```bash
# 启动服务
npm run start

# 或开发模式（自动重启）
npm run dev
```

你应该看到：

```
[config] 已注册工具: [
  'fs_read',    'fs_write',
  'fs_edit',    'fs_glob',
  'fs_grep',    'fs_multi_edit',
  'bash_run',   'bash_logs',
  'bash_kill',  'todo_read',
  'todo_write'
]
[config] 依赖初始化完成
[server] 启动中，端口=8787, NODE_ENV=undefined
[server] 服务已启动在 http://localhost:8787
```

## ✅ 步骤 4: 测试服务

### 方式 1: 使用测试客户端

```bash
# 运行自动化测试
npm run test:client

# 或使用自定义消息
npm run test:client -- "请创建一个 hello.py 文件"
```

### 方式 2: 使用 cURL

```bash
# 健康检查
curl http://localhost:8787/health

# 发送对话请求
curl -X POST http://localhost:8787/api/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: my-secret-key-123" \
  -d '{
    "message": "请创建一个 hello.py 文件，打印 Hello World",
    "mode": "edit",
    "outputsPath": "./outputs"
  }'
```

### 方式 3: 使用 JavaScript

```javascript
const response = await fetch('http://localhost:8787/api/chat/stream', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'my-secret-key-123',
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
  console.log(text)
}
```

## 📖 步骤 5: 了解更多

### 查看文档

- 📚 [README.md](./README.md) - 完整的 API 文档和使用指南
- 🔄 [COMPARISON.md](./COMPARISON.md) - 与原版的详细对比
- 🛠️ [KODE SDK 文档](./docs/zh-CN/) - SDK 的完整文档

### 常用命令

```bash
# 启动服务
npm start

# 开发模式（自动重启）
npm run dev

# 运行测试客户端
npm run test:client

# 使用自定义消息测试
# 先设置环境变量，然后运行测试客户端
MODE=ask npm run test:client -- "什么是 KODE SDK?"
```

### 输出目录

默认情况下，Agent 生成的文件会保存在 `./outputs/` 目录下。你可以通过 `outputsPath` 参数自定义。

```bash
# 创建输出目录
mkdir -p outputs
```

### 查看 Agent 数据

Agent 的持久化数据存储在 `./.kode/` 目录：

```bash
# 查看存储的 Agent
ls -la .kode/
```

## 🎯 示例场景

### 场景 1: 代码生成

```bash
curl -X POST http://localhost:8787/api/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: my-secret-key-123" \
  -d '{
    "message": "创建一个计算斐波那契数列的 Python 脚本",
    "mode": "edit",
    "outputsPath": "./outputs"
  }'
```

### 场景 2: 代码分析

```bash
curl -X POST http://localhost:8787/api/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: my-secret-key-123" \
  -d '{
    "message": "分析 package.json 文件，列出所有依赖",
    "mode": "ask"
  }'
```

### 场景 3: 批量文件处理

```bash
curl -X POST http://localhost:8787/api/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: my-secret-key-123" \
  -d '{
    "message": "找到所有 .ts 文件并添加注释",
    "mode": "edit",
    "outputsPath": "./outputs",
    "context": {
      "workingDir": "/path/to/project"
    }
  }'
```

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
- ✅ 检查 `outputsPath` 目录是否存在
- ✅ 查看服务器日志，确认工具是否执行成功

### Q4: 如何修改工作目录？

在请求中添加`context.workingDir`：

```json
{
  "message": "创建文件",
  "outputsPath": "/custom/path",
  "context": {
    "workingDir": "/another/path"
  }
}
```
agent会在 `context.workingDir` 的隔离沙箱中工作，以避免对其他目录的影响。

## 🎓 下一步

### 学习 KODE SDK

```bash
# 查看中文文档
ls docs/zh-CN/

# 快速上手
cat docs/zh-CN/getting-started/quickstart.md

# 了解核心概念
cat docs/zh-CN/getting-started/concepts.md
```

### 自定义工具

在 `kode-agent-service.ts` 中添加自定义工具：

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

// 注册到 ToolRegistry
toolRegistry.register(myTool.name, () => myTool)

// 在模板中启用
templateRegistry.register({
  id: 'my-template',
  tools: [..., 'my_custom_tool'],
})
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

## 🆘 获取帮助

- 📖 查看 [README.md](./README.md) 获取完整文档
- 🐛 遇到问题？查看 [KODE SDK GitHub](https://github.com/shareai-lab/kode-sdk)
- 后续将删除对Windows的支持，以避免兼容性问题。
---