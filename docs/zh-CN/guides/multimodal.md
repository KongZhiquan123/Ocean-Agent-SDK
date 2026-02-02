# 多模态内容指南

KODE SDK 支持多模态输入，包括图像、音频和文件（PDF）。本指南介绍如何向 LLM 模型发送多模态内容以及管理多模态历史记录。

---

## 支持的内容类型

| 类型 | Block 类型 | 支持的 Provider |
|------|------------|-----------------|
| 图片 | `image` | Anthropic, OpenAI, Gemini, GLM, Minimax |
| PDF 文件 | `file` | Anthropic, OpenAI (Responses API), Gemini |
| 音频 | `audio` | OpenAI, Gemini |

---

## 发送多模态内容

### 图片输入

使用 `ContentBlock[]` 配合 `agent.send()` 发送图片：

```typescript
import { Agent, ContentBlock } from '@shareai-lab/kode-sdk';
import * as fs from 'fs';

// 读取图片为 base64
const imageBuffer = fs.readFileSync('./image.png');
const base64 = imageBuffer.toString('base64');

// 构建内容块
const content: ContentBlock[] = [
  { type: 'text', text: '这张图片中有哪些动物？' },
  { type: 'image', base64, mime_type: 'image/png' }
];

// 发送给 agent
const response = await agent.send(content);
```

### 基于 URL 的图片

也可以使用 URL 代替 base64：

```typescript
const content: ContentBlock[] = [
  { type: 'text', text: '描述这张图片。' },
  { type: 'image', url: 'https://example.com/image.jpg' }
];

const response = await agent.send(content);
```

### PDF 文件输入

```typescript
const pdfBuffer = fs.readFileSync('./document.pdf');
const base64 = pdfBuffer.toString('base64');

const content: ContentBlock[] = [
  { type: 'text', text: '从这个 PDF 中提取主要内容。' },
  { type: 'file', base64, mime_type: 'application/pdf', filename: 'document.pdf' }
];

const response = await agent.send(content);
```

---

## 多模态配置

### Agent 配置

创建 Agent 时配置多模态行为：

```typescript
const agent = await Agent.create({
  templateId: 'multimodal-assistant',
  // 在对话历史中保留多模态内容
  multimodalContinuation: 'history',
  // 压缩上下文时保留最近 3 条多模态消息
  multimodalRetention: { keepRecent: 3 },
}, deps);
```

| 选项 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `multimodalContinuation` | `'history'` | `'history'` | 在对话历史中保留多模态内容 |
| `multimodalRetention.keepRecent` | `number` | `3` | 上下文压缩时保留的最近多模态消息数量 |

### Provider 配置

在模型配置中配置多模态选项：

```typescript
const provider = new AnthropicProvider(
  process.env.ANTHROPIC_API_KEY!,
  'claude-sonnet-4-20250514',
  undefined, // baseUrl
  undefined, // proxyUrl
  {
    multimodal: {
      mode: 'url+base64',           // 同时允许 URL 和 base64
      maxBase64Bytes: 20_000_000,   // base64 最大 20MB
      allowMimeTypes: [             // 允许的 MIME 类型
        'image/jpeg',
        'image/png',
        'image/gif',
        'image/webp',
        'application/pdf',
      ],
    },
  }
);
```

| 选项 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `mode` | `'url'` \| `'url+base64'` | `'url'` | URL 处理模式 |
| `maxBase64Bytes` | `number` | `20000000` | base64 内容最大尺寸 |
| `allowMimeTypes` | `string[]` | 常见图片 + PDF 类型 | 允许的 MIME 类型 |

---

## 支持的 MIME 类型

### 图片

| MIME 类型 | 扩展名 | 备注 |
|-----------|--------|------|
| `image/jpeg` | `.jpg`, `.jpeg` | 所有 Provider |
| `image/png` | `.png` | 所有 Provider |
| `image/webp` | `.webp` | 所有 Provider |
| `image/gif` | `.gif` | Gemini 不支持 |

### 文档

| MIME 类型 | 扩展名 | 备注 |
|-----------|--------|------|
| `application/pdf` | `.pdf` | Anthropic, OpenAI (Responses API), Gemini |

---

## Provider 特定说明

### Anthropic

- 支持图片和 PDF 文件
- 使用 `files-api-2025-04-14` beta 进行文件上传
- Base64 图片直接嵌入消息

```typescript
const provider = new AnthropicProvider(apiKey, model, baseUrl, proxyUrl, {
  beta: {
    filesApi: true,  // 启用 Files API
  },
  multimodal: {
    mode: 'url+base64',
  },
});
```

### OpenAI

- 图片：Chat Completions API 支持
- PDF/文件：需要 Responses API（`openaiApi: 'responses'`）

```typescript
const provider = new OpenAIProvider(apiKey, model, baseUrl, proxyUrl, {
  api: 'responses',  // PDF 支持必需
  multimodal: {
    mode: 'url+base64',
  },
});
```

### Gemini

- 支持图片和 PDF 文件
- 不支持 GIF 格式
- 使用 `mediaResolution` 选项控制图片质量

```typescript
const provider = new GeminiProvider(apiKey, model, baseUrl, proxyUrl, {
  mediaResolution: 'high',  // 'low' | 'medium' | 'high'
  multimodal: {
    mode: 'url+base64',
  },
});
```

---

## 最佳实践

### 1. 使用适当的图片尺寸

大图片会增加 token 使用量和延迟。发送前请调整图片大小：

```typescript
// 建议：保持图片在 1MB 以下以获得最佳性能
const maxBytes = 1024 * 1024; // 1MB

function validateImageSize(base64: string): boolean {
  const bytes = Math.ceil(base64.length * 3 / 4);
  return bytes <= maxBytes;
}
```

### 2. 处理多模态上下文保留

对于包含大量图片的长对话，配置保留策略以避免上下文溢出：

```typescript
const agent = await Agent.create({
  templateId: 'vision-assistant',
  multimodalRetention: { keepRecent: 2 },  // 仅保留最近 2 张图片
  context: {
    maxTokens: 100_000,
    compressToTokens: 60_000,
  },
}, deps);
```

### 3. 验证 MIME 类型

发送前始终验证 MIME 类型：

```typescript
const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

function getImageMimeType(filename: string): string {
  const ext = filename.toLowerCase().split('.').pop();
  const mimeMap: Record<string, string> = {
    jpg: 'image/jpeg',
    jpeg: 'image/jpeg',
    png: 'image/png',
    webp: 'image/webp',
  };
  const mimeType = mimeMap[ext!];
  if (!mimeType || !ALLOWED_IMAGE_TYPES.includes(mimeType)) {
    throw new Error(`不支持的图片类型: ${ext}`);
  }
  return mimeType;
}
```

---

## 错误处理

常见多模态错误：

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `MultimodalValidationError: Base64 is not allowed` | `mode` 仅设置为 `'url'` | 设置 `mode: 'url+base64'` |
| `MultimodalValidationError: base64 payload too large` | 超过 `maxBase64Bytes` | 调整图片大小或增加限制 |
| `MultimodalValidationError: mime_type not allowed` | MIME 类型不在允许列表中 | 添加到 `allowMimeTypes` |
| `MultimodalValidationError: Missing url/file_id/base64` | 未提供内容源 | 提供 `url`、`file_id` 或 `base64` |

---

## 完整示例

```typescript
import { Agent, AnthropicProvider, JSONStore, ContentBlock } from '@shareai-lab/kode-sdk';
import * as fs from 'fs';

async function analyzeImage() {
  const provider = new AnthropicProvider(
    process.env.ANTHROPIC_API_KEY!,
    'claude-sonnet-4-20250514',
    undefined,
    undefined,
    {
      multimodal: {
        mode: 'url+base64',
        maxBase64Bytes: 10_000_000,
      },
    }
  );

  const store = new JSONStore('./.kode');

  const agent = await Agent.create({
    templateId: 'vision-assistant',
    multimodalContinuation: 'history',
    multimodalRetention: { keepRecent: 3 },
  }, {
    store,
    templateRegistry,
    toolRegistry,
    sandboxFactory,
    modelFactory: () => provider,
  });

  // 读取并发送图片
  const imageBuffer = fs.readFileSync('./photo.jpg');
  const base64 = imageBuffer.toString('base64');

  const content: ContentBlock[] = [
    { type: 'text', text: '这张照片中有哪些物体？' },
    { type: 'image', base64, mime_type: 'image/jpeg' }
  ];

  for await (const envelope of agent.subscribe(['progress'])) {
    if (envelope.event.type === 'text_chunk') {
      process.stdout.write(envelope.event.delta);
    }
    if (envelope.event.type === 'done') break;
  }

  await agent.send(content);
}
```

---

## 参考资料

- [Provider 指南](./providers.md) - Provider 特定配置
- [事件指南](./events.md) - Progress 事件处理
- [API 参考](../reference/api.md) - ContentBlock 类型
