# E2B Cloud Sandbox Guide

KODE SDK supports [E2B](https://e2b.dev) as a cloud sandbox backend, providing fully isolated remote Linux environments for AI Agent code execution.

---

## Overview

| Feature | Description |
|---------|-------------|
| **Isolation** | Each sandbox runs in an isolated micro-VM |
| **Startup** | ~150ms to create a full Linux environment |
| **Pre-installed** | Python, Node.js, common system tools |
| **Scalable** | Each Agent gets its own sandbox instance |

### When to Use E2B vs Local Sandbox

| Scenario | Recommended |
|----------|-------------|
| Development/Testing | Local Sandbox |
| Production with untrusted code | E2B |
| Multi-user concurrent agents | E2B |
| Need specific environment (GPU, packages) | E2B (Custom Template) |
| Offline / No internet | Local Sandbox |

---

## Prerequisites

1. Sign up at [e2b.dev](https://e2b.dev)
2. Get your API Key from the Dashboard
3. Set environment variable:

```bash
export E2B_API_KEY=your-api-key
```

---

## Quick Start

### Create and Use a Sandbox

```typescript
import { E2BSandbox } from '@shareai-lab/kode-sdk';

const sandbox = new E2BSandbox({
  apiKey: process.env.E2B_API_KEY,
  template: 'base',
  timeoutMs: 300_000, // 5 minutes
});
await sandbox.init();

// Execute commands
const result = await sandbox.exec('python3 -c "print(1+1)"');
console.log(result.stdout); // "2\n"

// File operations
await sandbox.fs.write('script.py', 'print("hello")');
const content = await sandbox.fs.read('script.py');

// Cleanup
await sandbox.dispose();
```

---

## Configuration

### E2BSandboxOptions

```typescript
interface E2BSandboxOptions {
  apiKey?: string;              // E2B API Key (or use E2B_API_KEY env)
  template?: string;            // Template ID/alias, default 'base'
  timeoutMs?: number;           // Sandbox lifetime, default 300_000 (5min)
  workDir?: string;             // Working directory, default '/home/user'
  envs?: Record<string, string>; // Environment variables
  metadata?: Record<string, string>; // Custom metadata
  allowInternetAccess?: boolean; // Allow internet, default true
  execTimeoutMs?: number;       // Command timeout, default 120_000
  sandboxId?: string;           // Connect to existing sandbox (resume)
  domain?: string;              // E2B API domain
}
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `E2B_API_KEY` | Your E2B API key (used if `apiKey` not provided) |

---

## Agent Integration

### Using E2B Sandbox with Agent

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
    systemPrompt: 'You are a coding assistant.',
    tools: ['bash_run', 'fs_read', 'fs_write', 'todo_read', 'todo_write'],
  });
});

const agent = await Agent.create({ templateId: 'coder', sandbox }, deps);
await agent.send('Write and run a Python fibonacci script');
```

### Sandbox Lifecycle Binding

- **Agent Start**: Call `sandbox.init()` before `Agent.create()`
- **Agent Pause**: Sandbox persists (use `sandboxId` to reconnect)
- **Agent Resume**: Pass `sandboxId` to reconnect to existing sandbox
- **Agent Destroy**: Call `sandbox.dispose()` to terminate

### Resume / Fork with Persistent sandboxId

```typescript
// First run - create
const sandbox = new E2BSandbox({ template: 'base' });
await sandbox.init();
const id = sandbox.getSandboxId(); // persist this

// Later - resume
const restored = new E2BSandbox({ sandboxId: id });
await restored.init();
// Same sandbox environment is available
```

---

## Custom Templates

### Using E2BTemplateBuilder

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

### Available Base Images

| Base | Description |
|------|-------------|
| `python` | Python with pip |
| `node` | Node.js with npm |
| `debian` | Debian base |
| `ubuntu` | Ubuntu base |
| `custom` | Custom Dockerfile |

---

## Network & Ports

### Exposing Ports

```typescript
const sandbox = new E2BSandbox({ template: 'node' });
await sandbox.init();

await sandbox.exec('npx serve -l 3000 &');
const url = sandbox.getHostUrl(3000);
console.log(`Preview: ${url}`);
// https://3000-<sandboxId>.e2b.app
```

---

## Best Practices

1. **Timeout Management**: Set appropriate `timeoutMs` to control costs. Call `dispose()` when done.
2. **Error Handling**: E2B operations are remote calls; handle network errors gracefully.
3. **Data Export**: Sandbox data is lost after `dispose()`. Export important results first.
4. **Concurrency**: E2B has account-level sandbox limits. Plan accordingly with AgentPool.
5. **Template Reuse**: Build templates once, reuse across sandboxes for faster startup.
