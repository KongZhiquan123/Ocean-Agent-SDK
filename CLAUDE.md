# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI Agent HTTP service built on top of the KODE SDK (@shareai-lab/kode-sdk). It provides a RESTful API with SSE (Server-Sent Events) streaming for AI-powered coding assistance and question answering. The service specializes in ocean data preprocessing for super-resolution scenarios, converting NC (NetCDF) format to NPY format.

**Tech Stack**: Node.js (>=20.18.1), TypeScript, Express, KODE SDK v2.7.2

## Development Commands

```bash
# Install dependencies
npm install

# Start the server (production)
npm run start

# Start in development mode (auto-restart on changes)
npm run dev

# Test the client
npm run test:client
```

## Environment Configuration

Required environment variables in `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...           # Required: Anthropic API key
ANTHROPIC_BASE_URL=https://yunwu.ai    # Required: API endpoint
ANTHROPIC_MODEL_ID=claude-sonnet-4-5-20250929  # Required: Model ID
KODE_API_SECRET=your-secret-key        # Required: API authentication
KODE_API_PORT=8787                     # Optional: Server port (default: 8787)
SKILLS_DIR=./.skills                   # Optional: Skills directory
```

## Architecture

### Core Modules

The codebase follows a modular architecture with clear separation of concerns:

**1. `src/config.ts` - Configuration & Dependency Injection**
- Validates environment variables
- Initializes all KODE SDK dependencies (Store, ToolRegistry, TemplateRegistry, SandboxFactory, ModelFactory)
- Registers built-in tools (fs, bash, todo) and custom ocean preprocessing tools
- Defines agent templates for "edit" and "ask" modes
- Uses singleton pattern via `getDependencies()` function

**2. `src/agent-manager.ts` - Agent Lifecycle Management**
- Creates agents based on mode (ask/edit) and configuration
- Sets up event handlers for permission requests and errors
- Implements dangerous command blacklist for bash_run tool (blocks rm -rf /, sudo, etc.)
- Converts KODE SDK ProgressEvents to SSE format
- Provides async generator `processMessage()` for streaming responses

**3. `src/server.ts` - HTTP Server**
- Express-based REST API with SSE streaming
- Main endpoint: `POST /api/chat/stream` (requires X-API-Key header)
- Health check: `GET /health`
- Implements request logging, authentication middleware, and error handling
- Manages SSE connections with heartbeat (2s interval) and timeout (10 min)
- Integrates with conversation manager for multi-turn sessions

**4. `src/conversation-manager.ts` - Multi-turn Conversation Support**
- Maintains Agent instance pool using agentId as key
- Automatic session expiration (30 min timeout) and cleanup (5 min interval)
- LRU eviction when max sessions (100) reached
- Enables conversation continuity by reusing Agent instances

**5. `src/tools/ocean-preprocess/` - Custom Ocean Data Tools**
- Four specialized tools for NC→NPY conversion pipeline
- Uses Python scripts via sandbox for data processing
- Implements interactive confirmation workflow for mask/coordinate detection

### KODE SDK Integration

The service uses KODE SDK's three-channel event system:
- **Progress**: Data plane for UI rendering (text streams, tool lifecycle) - converted to SSE events
- **Control**: Approval plane for human decisions (permission requests) - handled by dangerous command filter
- **Monitor**: Governance plane for audit/alerts (errors, state changes) - logged to console

**Key SDK Components**:
- `JSONStore`: Persists agent messages and tool calls to `./.kode/`
- `ToolRegistry`: Manages available tools per agent template
- `AgentTemplateRegistry`: Defines system prompts and tool sets for "coding-assistant" and "qa-assistant"
- `SandboxFactory`: Creates isolated environments for file operations and command execution
- `AnthropicProvider`: LLM provider for Claude models

### Agent Modes

**Edit Mode** (`coding-assistant` template):
- Full read/write file access (fs_read, fs_write, fs_edit)
- Command execution (bash_run)
- Todo management (todo_read, todo_write)
- Skills system (skills tool)
- Ocean preprocessing tools (ocean_inspect_data, ocean_validate_tensor, ocean_convert_npy, ocean_preprocess_full)

**Ask Mode** (`qa-assistant` template):
- Read-only file access (fs_read)
- File search (fs_glob, fs_grep)
- Read-only commands (bash_run with restrictions)

### Ocean Data Preprocessing Workflow

The service includes a specialized skill for ocean data preprocessing (`.skills/ocean-preprocess/`):

**Pipeline**: NC files → [Step A: Inspect] → [Step B: Validate] → [Step C: Convert] → NPY files

**Key Principles**:
- No data normalization or transformation - only format conversion
- Preserves original data structure completely
- Interactive confirmation for mask/coordinate variable detection
- Tensor shape conventions: Dynamic `[T,H,W]` or `[T,D,H,W]`, Static `[H,W]`

**Tools**:
1. `ocean_inspect_data`: Analyzes NC files, classifies variables (dynamic/static/mask)
2. `ocean_validate_tensor`: Validates tensor shapes and generates var_names config
3. `ocean_convert_npy`: Converts to NPY format with directory structure (hr/, static/)
4. `ocean_preprocess_full`: One-click execution of full A→B→C pipeline with interactive confirmation

**Important**: The `ocean_preprocess_full` tool implements a two-phase workflow:
- Phase 1: Returns `awaiting_confirmation` status with suspected masks/coordinates
- Agent must present these to user and ask for confirmation
- Phase 2: Re-invoke with user-confirmed `mask_vars` and `static_vars` to execute full pipeline

### Skills System

Skills are loaded from `SKILLS_DIR` (default: `./.skills/`) with whitelist filtering:
- Current whitelist: `['ocean-preprocess']`
- Each skill has `metadata.json` and `SKILL.md` (must use LF line endings)
- Skills are loaded via the `skills` tool with actions: "list" or "load"
- See `docs/zh-CN/guides/skills.md` for skill development guide

## API Usage

### Chat Stream Endpoint

```http
POST /api/chat/stream
Content-Type: application/json
X-API-Key: your-secret-key

{
  "message": "请帮我创建一个 hello.py 文件",
  "mode": "edit",
  "context": {
    "userId": "user123",
    "workingDir": "/path/to/work"
  },
  "agentId": "agt-abc123"  // Optional: for multi-turn conversations
}
```

**SSE Event Types**:
- `start`: Processing begins (includes `agentId` and `isNewSession`)
- `heartbeat`: Keep-alive every 2 seconds
- `text`: AI-generated text content
- `tool_use`: Tool invocation started
- `tool_result`: Tool execution result
- `done`: Processing complete
- `error`: Error occurred

### Multi-turn Conversations

To continue a conversation, include the `agentId` from the previous response's `start` event. The conversation manager will reuse the Agent instance if it hasn't expired (30 min timeout).

## Adding Custom Tools

1. Define tool in `src/config.ts` using `defineTool()` from KODE SDK
2. Register in `createToolRegistry()` function
3. Add tool name to appropriate template in `createTemplateRegistry()`
4. **Critical**: Use `ctx.sandbox` for all file operations and command execution to ensure proper isolation

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
    const result = await ctx.sandbox.readFile('/path/to/file')
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

## Important Notes

- **Skills YAML**: SKILL.md files must use LF line endings (not CRLF) to avoid YAML parser errors
- **Sandbox Usage**: Always use `ctx.sandbox` in custom tools for file/command operations
- **Session Management**: Agent instances are automatically cleaned up after 30 min of inactivity
- **Python Detection**: The `python-manager.ts` utility scans common Python installation paths (pyenv, conda, system) for ocean preprocessing tools
- **Data Storage**: KODE SDK stores agent state in `./.kode/` directory (configured via `KODE_STORE_PATH`)

## Author Documentation before Modifications or Generations
When generating or modifying code files, always add/update a standardized header comment at the top following this format:
```typescript
/**
 * @file filename.ext
 *
 * @description [Brief description]
 * @author kongzhiquan
 * @date YYYY-MM-DD
 * @version x.x.x
 *
 * @changelog
 *   - YYYY-MM-DD kongzhiquan: version description
 */
```
If the file already has a header, update the "Changelog" section and append `kongzhiquan`(your name should be here) to the "@contributors" list if not already present. If '@contributors' is not present, add it below the author line.

## Python Command Execution
If you need to use Bash tools to run Python, please use the executable file path /home/lz/miniconda3/envs/pytorch/bin/python to ensure the correct Python environment is used.

## Typescript Typechecking
If you want to typecheck the entire project, you can simply run:
```bash
npm run typecheck
```
This will typecheck all files in the project according to the tsconfig.json settings.