# Multimodal Content Guide

KODE SDK supports multimodal input including images, audio, and files (PDF). This guide covers how to send multimodal content to LLM models and manage multimodal history.

---

## Supported Content Types

| Type | Block Type | Supported Providers |
|------|------------|---------------------|
| Images | `image` | Anthropic, OpenAI, Gemini, GLM, Minimax |
| PDF Files | `file` | Anthropic, OpenAI (Responses API), Gemini |
| Audio | `audio` | OpenAI, Gemini |

---

## Sending Multimodal Content

### Image Input

Send images using `ContentBlock[]` with `agent.send()`:

```typescript
import { Agent, ContentBlock } from '@shareai-lab/kode-sdk';
import * as fs from 'fs';

// Read image as base64
const imageBuffer = fs.readFileSync('./image.png');
const base64 = imageBuffer.toString('base64');

// Build content blocks
const content: ContentBlock[] = [
  { type: 'text', text: 'What animals are in this image?' },
  { type: 'image', base64, mime_type: 'image/png' }
];

// Send to agent
const response = await agent.send(content);
```

### URL-based Images

You can also use URLs instead of base64:

```typescript
const content: ContentBlock[] = [
  { type: 'text', text: 'Describe this image.' },
  { type: 'image', url: 'https://example.com/image.jpg' }
];

const response = await agent.send(content);
```

### PDF File Input

```typescript
const pdfBuffer = fs.readFileSync('./document.pdf');
const base64 = pdfBuffer.toString('base64');

const content: ContentBlock[] = [
  { type: 'text', text: 'Extract the main topics from this PDF.' },
  { type: 'file', base64, mime_type: 'application/pdf', filename: 'document.pdf' }
];

const response = await agent.send(content);
```

---

## Multimodal Configuration

### Agent Configuration

Configure multimodal behavior when creating an Agent:

```typescript
const agent = await Agent.create({
  templateId: 'multimodal-assistant',
  // Keep multimodal content in conversation history
  multimodalContinuation: 'history',
  // Keep recent 3 messages with multimodal content when compressing context
  multimodalRetention: { keepRecent: 3 },
}, deps);
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `multimodalContinuation` | `'history'` | `'history'` | Preserve multimodal content in conversation history |
| `multimodalRetention.keepRecent` | `number` | `3` | Number of recent multimodal messages to keep during context compression |

### Provider Configuration

Configure multimodal options in the model configuration:

```typescript
const provider = new AnthropicProvider(
  process.env.ANTHROPIC_API_KEY!,
  'claude-sonnet-4-20250514',
  undefined, // baseUrl
  undefined, // proxyUrl
  {
    multimodal: {
      mode: 'url+base64',           // Allow both URL and base64
      maxBase64Bytes: 20_000_000,   // 20MB max for base64
      allowMimeTypes: [             // Allowed MIME types
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

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mode` | `'url'` \| `'url+base64'` | `'url'` | URL handling mode |
| `maxBase64Bytes` | `number` | `20000000` | Maximum size for base64 content |
| `allowMimeTypes` | `string[]` | Common image + PDF types | Allowed MIME types |

---

## Supported MIME Types

### Images

| MIME Type | Extension | Notes |
|-----------|-----------|-------|
| `image/jpeg` | `.jpg`, `.jpeg` | All providers |
| `image/png` | `.png` | All providers |
| `image/webp` | `.webp` | All providers |
| `image/gif` | `.gif` | Not supported by Gemini |

### Documents

| MIME Type | Extension | Notes |
|-----------|-----------|-------|
| `application/pdf` | `.pdf` | Anthropic, OpenAI (Responses API), Gemini |

---

## Provider-Specific Notes

### Anthropic

- Supports images and PDF files
- Use `files-api-2025-04-14` beta for file uploads
- Base64 images embedded directly in messages

```typescript
const provider = new AnthropicProvider(apiKey, model, baseUrl, proxyUrl, {
  beta: {
    filesApi: true,  // Enable Files API
  },
  multimodal: {
    mode: 'url+base64',
  },
});
```

### OpenAI

- Images: Supported in Chat Completions API
- PDF/Files: Requires Responses API (`openaiApi: 'responses'`)

```typescript
const provider = new OpenAIProvider(apiKey, model, baseUrl, proxyUrl, {
  api: 'responses',  // Required for PDF support
  multimodal: {
    mode: 'url+base64',
  },
});
```

### Gemini

- Supports images and PDF files
- GIF format not supported
- Use `mediaResolution` option for image quality

```typescript
const provider = new GeminiProvider(apiKey, model, baseUrl, proxyUrl, {
  mediaResolution: 'high',  // 'low' | 'medium' | 'high'
  multimodal: {
    mode: 'url+base64',
  },
});
```

---

## Best Practices

### 1. Use Appropriate Image Sizes

Large images increase token usage and latency. Resize images before sending:

```typescript
// Recommendation: Keep images under 1MB for optimal performance
const maxBytes = 1024 * 1024; // 1MB

function validateImageSize(base64: string): boolean {
  const bytes = Math.ceil(base64.length * 3 / 4);
  return bytes <= maxBytes;
}
```

### 2. Handle Multimodal Context Retention

For long conversations with many images, configure retention to avoid context overflow:

```typescript
const agent = await Agent.create({
  templateId: 'vision-assistant',
  multimodalRetention: { keepRecent: 2 },  // Keep only recent 2 images
  context: {
    maxTokens: 100_000,
    compressToTokens: 60_000,
  },
}, deps);
```

### 3. Validate MIME Types

Always validate MIME types before sending:

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
    throw new Error(`Unsupported image type: ${ext}`);
  }
  return mimeType;
}
```

---

## Error Handling

Common multimodal errors:

| Error | Cause | Solution |
|-------|-------|----------|
| `MultimodalValidationError: Base64 is not allowed` | `mode` set to `'url'` only | Set `mode: 'url+base64'` |
| `MultimodalValidationError: base64 payload too large` | Exceeds `maxBase64Bytes` | Resize image or increase limit |
| `MultimodalValidationError: mime_type not allowed` | MIME type not in allowlist | Add to `allowMimeTypes` |
| `MultimodalValidationError: Missing url/file_id/base64` | No content source provided | Provide `url`, `file_id`, or `base64` |

---

## Complete Example

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

  // Read and send image
  const imageBuffer = fs.readFileSync('./photo.jpg');
  const base64 = imageBuffer.toString('base64');

  const content: ContentBlock[] = [
    { type: 'text', text: 'What objects are in this photo?' },
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

## References

- [Provider Guide](./providers.md) - Provider-specific configuration
- [Events Guide](./events.md) - Progress event handling
- [API Reference](../reference/api.md) - ContentBlock types
