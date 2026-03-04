# Custom Tools & Configuration Reference

## Adding Custom Tools

1. Define tool in `src/config.ts` using `defineTool()` from KODE SDK
2. Register in `createToolRegistry()` function
3. Add tool name to appropriate template in `createTemplateRegistry()`
4. **Critical**: Use `ctx.sandbox` for all file operations and command execution to ensure proper isolation.

Example:
```typescript
import { defineTool } from '@shareai-lab/kode-sdk'

const myTool = defineTool({
  name: 'my_custom_tool',
  description: 'My custom tool',
  params: {
    input: { type: 'string', description: 'Input parameter' },
  },
  async exec(args, ctx) {
    // Use ctx.sandbox for file operations
    const result = await ctx.sandbox.read('/path/to/file')
    return { result: 'success' }
  },
})

// In createToolRegistry():
registry.register(myTool.name, () => myTool)
```

## Modifying System Prompts

Edit the `systemPrompt` field in `createTemplateRegistry()` within `src/config.ts`. The prompts are template-specific:
- `coding-assistant`: Edit mode prompt (includes ocean preprocessing instructions)
- `qa-assistant`: Ask mode prompt (read-only focus)

## Permission Control

Dangerous command filtering is implemented in `src/agent-manager.ts` via the `permission_required` event handler. The blacklist includes:
- `rm -rf /` or `rm -rf ~`
- `sudo` commands
- Writing to system directories (/etc, /usr, /bin)
- Disk operations (mkfs, dd)
- Fork bombs
- Remote execution patterns (curl | bash)

To customize, modify the `DANGEROUS_PATTERNS` array or add custom logic in `setupAgentHandlers()`.
