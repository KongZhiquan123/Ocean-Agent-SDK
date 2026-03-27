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
KODE_MODEL_PROVIDER=openai             # Required: anthropic | openai | gemini
OPENAI_API_KEY=sk-...                  # Required when provider=openai
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_ID=gpt-5.4
OPENAI_API_MODE=chat                   # Recommended for OpenAI-compatible gateways
KODE_API_SECRET=your-secret-key        # Required: API authentication
KODE_API_PORT=8787                     # Optional: Server port (default: 8787)
SKILLS_DIR=./.skills                   # Optional: Skills directory
```

Notes:
- `provider=openai` supports OpenAI-compatible services such as DeepSeek, GLM, Qwen, Minimax, OpenRouter, and compatible gateways.
- `provider=anthropic` uses `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` / `ANTHROPIC_MODEL_ID`.
- `provider=gemini` uses `GOOGLE_API_KEY` / `GEMINI_MODEL_ID`.

## Architecture

> Detailed module description, KODE SDK integration, and Skills system can be found in docs/claude-references/architecture.md

**Key entry points**: `src/config.ts` (DI + tool registration), `src/agent-manager.ts` (agent lifecycle + SSE), `src/server.ts` (HTTP + SSE), `src/conversation-manager.ts` (session pool)

**Agent Modes**:
- **Edit** (`coding-assistant`): full fs read/write, bash, todo, skills, ocean SR/forecast data+training tools
- **Ask** (`qa-assistant`): read-only fs, glob/grep, ocean_inspect_data

## API & Custom Tools

> For API interface details, see docs/claude-references/api-usage.md
> For custom tool development, system prompt modifications, and permission control, see docs/claude-references/custom-tools.md

- Main endpoint: `POST /api/chat/stream` (X-API-Key header required, SSE response)
- Health check: `GET /health`

## Important Notes

- **Skills YAML**: SKILL.md files must use LF line endings (not CRLF) to avoid YAML parser errors
- **Sandbox Usage**: Always use `ctx.sandbox` in custom tools for file/command operations
- **Session Management**: Conversation sessions are resumed from `./.kode/` on demand; this repo no longer uses the old in-memory 30-minute expiry/LRU pool design
- **Data Storage**: KODE SDK stores agent state in `./.kode/` directory (configured via `KODE_STORE_PATH`)

## Interaction Rules (MUST follow)

1. **Ask before acting when requirements are unclear**: If the user's description is missing either
   "expected behavior" or "actual behavior", ask specific clarifying questions first. Do NOT assume
   and proceed. Bad: "Can you be more specific?" Good: "Is the expected output format X or Y?"

2. **Diagnose before fixing**: When receiving a bug report, first analyze the root cause and confirm
   with the user before writing any code. Never see an error log and jump straight to editing code.

3. **Cross-layer change check**: When modifying ANY of the following layers, proactively check whether
   other layers need synchronized updates:
   - Python tool scripts (`scripts/`) ↔ TS tool definitions (`src/config.ts`) ↔ SKILL.md docs (`.skills/`) ↔ test client (`test-client.ts`)
   - API/SSE interface (`src/server.ts`) ↔ conversation manager (`src/conversation-manager.ts`) ↔ test client ↔ README
   - Model code (`scripts/ocean-forecast-training/models/`) ↔ training configs (`generate_config.py`) ↔ skill reference docs

4. **Challenge over compliance**: If the user's proposed approach has a better alternative, or the user
   may have missed an impact area, you MUST point it out. Do not stay silent to avoid interrupting momentum.

## Author Documentation before Modifications or Generations
When generating or modifying code files, always add/update a standardized header comment at the top following this format:
```typescript
/**
 * @file filename.ext
 *
 * @description [Brief description]
 * @author Leizheng
 * @date YYYY-MM-DD
 * @version x.x.x
 *
 * @changelog
 *   - YYYY-MM-DD Leizheng: version description
 */
```
If the file already has a header, update the "Changelog" section and append `Leizheng`(your name should be here) to the "@contributors" list if not already present. If '@contributors' is not present, add it below the author line.

## Python Command Execution
If you need to use Bash tools to run Python, please use the executable file path /home/lz/miniconda3/envs/pytorch/bin/python to ensure the correct Python environment is used.

## Typescript Typechecking
If you want to typecheck the entire project, you can simply run:
```bash
npm run typecheck
```
This will typecheck all files in the project according to the tsconfig.json settings.
