# Core Concepts

## What is KODE SDK?

KODE SDK is an **Agent Runtime Kernel** — it manages the complete lifecycle of AI agents including state persistence, crash recovery, and tool execution.

Think of it like **V8 for JavaScript**, but for AI agents:

```
+------------------+     +------------------+
|       V8         |     |    KODE SDK      |
|  JS Runtime      |     |  Agent Runtime   |
+------------------+     +------------------+
        |                        |
        v                        v
+------------------+     +------------------+
|    Express.js    |     |   Your App       |
|  Web Framework   |     | (CLI/Desktop/Web)|
+------------------+     +------------------+
```

**KODE SDK provides:**
- Agent lifecycle management (create, run, pause, resume, fork)
- State persistence with crash recovery (WAL-protected)
- Tool execution with permission governance
- Three-channel event system for observability

**KODE SDK does NOT provide:**
- HTTP routing or API framework
- User authentication or authorization
- Multi-tenancy or resource isolation
- Horizontal scaling (you architect that layer)

> For deep dive into architecture, see [Architecture Guide](../advanced/architecture.md)

---

## Agent

The central entity that manages conversations with LLM models.

```typescript
// Setup dependencies
const templates = new AgentTemplateRegistry();
templates.register({
  id: 'assistant',
  systemPrompt: 'You are a helpful assistant.',
  tools: ['fs_read', 'fs_write'],  // Optional: tool names
});

// Create agent
const agent = await Agent.create(
  { templateId: 'assistant' },
  { store, templateRegistry: templates, toolRegistry: tools, sandboxFactory, modelFactory }
);
```

Key capabilities:
- **Send messages**: `agent.send('...')` or `agent.send(contentBlocks)`
- **Subscribe to events**: `agent.subscribe(['progress'])` or `agent.on('event_type', callback)`
- **Resume from store**: `Agent.resume(agentId, config, deps)` or `Agent.resumeFromStore(agentId, deps)`
- **Fork conversation**: `agent.fork()`

## Three-Channel Event System

KODE SDK separates events into three channels for clean architecture:

### Progress Channel

Real-time streaming data for UI display. Use `subscribe()`:

```typescript
for await (const envelope of agent.subscribe(['progress'])) {
  switch (envelope.event.type) {
    case 'text_chunk':      // Text chunk from model
      process.stdout.write(envelope.event.delta);
      break;
    case 'tool:start':      // Tool execution started
    case 'tool:end':        // Tool execution completed
    case 'done':            // Response complete
  }
}
```

### Control Channel

Approval requests that need human/system decision. Use `on()`:

```typescript
agent.on('permission_required', async (event) => {
  // Approve or reject tool execution
  await event.respond('allow');  // or event.respond('deny', { note: 'reason' })
});
```

### Monitor Channel

Audit and observability events. Use `on()`:

```typescript
agent.on('tool_executed', (event) => {
  console.log('Tool:', event.call.name, 'Duration:', event.call.durationMs);
});

agent.on('token_usage', (event) => {
  console.log('Tokens:', event.totalTokens);
});

agent.on('error', (event) => {
  console.error('Error:', event.message);
});
```

## Tools

Tools extend Agent capabilities. KODE provides built-in tools and supports custom tools.

### Built-in Tools

| Category | Tools |
|----------|-------|
| File System | `fs_read`, `fs_write`, `fs_edit`, `fs_glob`, `fs_grep` |
| Shell | `bash_run`, `bash_logs`, `bash_kill` |
| Task Management | `todo_read`, `todo_write` |

### Custom Tools

```typescript
import { defineTool } from '@shareai-lab/kode-sdk';

const weatherTool = defineTool({
  name: 'get_weather',
  description: 'Get weather for a city',
  params: {
    city: { type: 'string', description: 'City name' }
  },
  attributes: { readonly: true },
  async exec(args, ctx) {
    return { temp: 22, condition: 'sunny' };
  }
});
```

## Store

Persistence backend for Agent state.

| Store Type | Use Case |
|------------|----------|
| `JSONStore` | Development, single instance |
| `SqliteStore` | Production, single machine |
| `PostgresStore` | Production, multi-instance |

```typescript
// JSONStore (default)
const store = new JSONStore('./.kode');

// SQLite
const store = new SqliteStore('./agents.db', './data');

// PostgreSQL
const store = new PostgresStore(connectionConfig, './data');

// Factory function
const store = createExtendedStore({
  type: 'sqlite',
  dbPath: './agents.db',
  fileStoreBaseDir: './data'
});
```

## Sandbox

Isolated execution environment for tools.

```typescript
const agent = await Agent.create(
  {
    templateId: 'assistant',
    sandbox: {
      kind: 'local',
      workDir: './workspace',
      enforceBoundary: true,  // Restrict file access to workDir
    }
  },
  deps
);
```

## Provider

Model provider adapters. KODE uses Anthropic-style messages internally.

```typescript
// Anthropic
const provider = new AnthropicProvider(apiKey, modelId);

// OpenAI
const provider = new OpenAIProvider(apiKey, modelId);

// Gemini
const provider = new GeminiProvider(apiKey, modelId);
```

## Resume & Fork

### Resume

Recover from crash or continue later:

```typescript
// Resume existing agent
const agent = await Agent.resume(agentId, config, deps);

// Resume or create new
const exists = await store.exists(agentId);
const agent = exists
  ? await Agent.resume(agentId, config, deps)
  : await Agent.create(config, deps);
```

### Fork

Branch conversation at a checkpoint:

```typescript
// Create snapshot
const snapshotId = await agent.snapshot('before-risky-operation');

// Fork from snapshot
const forkedAgent = await agent.fork(snapshotId);

// Each agent continues independently
await forkedAgent.send('Try alternative approach');
```

## Multimodal Content

KODE SDK supports multimodal input including images, PDF files, and audio:

```typescript
import { ContentBlock } from '@shareai-lab/kode-sdk';

// Send image with text
const content: ContentBlock[] = [
  { type: 'text', text: 'What is in this image?' },
  { type: 'image', base64: imageBase64, mime_type: 'image/png' }
];

await agent.send(content);
```

Configure multimodal behavior:

```typescript
const agent = await Agent.create({
  templateId: 'vision-assistant',
  multimodalContinuation: 'history',      // Keep multimodal in history
  multimodalRetention: { keepRecent: 3 }, // Keep recent 3 multimodal messages
}, deps);
```

## Extended Thinking

Enable models to "think" through complex problems with extended thinking:

```typescript
const agent = await Agent.create({
  templateId: 'reasoning-assistant',
  exposeThinking: true,   // Emit thinking events to Progress channel
  retainThinking: true,   // Persist thinking in message history
}, deps);

// Listen for thinking events
for await (const envelope of agent.subscribe(['progress'])) {
  if (envelope.event.type === 'think_chunk') {
    console.log('[Thinking]', envelope.event.delta);
  }
}
```

## Next Steps

- [Events Guide](../guides/events.md) - Deep dive into event system
- [Tools Guide](../guides/tools.md) - Built-in and custom tools
- [Database Guide](../guides/database.md) - Persistence options
- [Multimodal Guide](../guides/multimodal.md) - Images, PDFs, and audio
- [Thinking Guide](../guides/thinking.md) - Extended thinking and reasoning
