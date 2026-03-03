# Architecture Reference

## Core Modules

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

## KODE SDK Integration

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

## Skills System

Skills are loaded from `SKILLS_DIR` (default: `./.skills/`) with whitelist filtering:
- Current whitelist: `['ocean-preprocess']`
- Each skill has `metadata.json` and `SKILL.md` (must use LF line endings)
- Skills are loaded via the `skills` tool with actions: "list" or "load"
- See `docs/zh-CN/guides/skills.md` for skill development guide
