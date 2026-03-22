# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI Agent HTTP service built on top of the KODE SDK (@shareai-lab/kode-sdk). It provides a RESTful API with SSE (Server-Sent Events) streaming for AI-powered ocean data processing and model training workflows. The service covers two main scenarios:

- **Super-Resolution (SR)**: Data preprocessing (NC→NPY format conversion, downsampling, visualization) and model training
- **Ocean Forecasting**: Time-series data preprocessing and model training (FNO2d, UNet2d, SwinTransformerV2, Transformer)

The agent exposes **24 custom tools** across 4 functional modules (SR data preprocess, SR training, Forecast data preprocess, Forecast training), plus built-in file system, Bash, and task management tools.

**Tech Stack**: Node.js (>=20.18.1), TypeScript, Express, KODE SDK ^2.7.2; Python ML stack: PyTorch 2.0+, NumPy, xarray, netCDF4, timm, wandb

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
PYTHON3=python                         # Optional: Python executable path for tools, superior to PATH lookup
NODE_ENV=development                   # Optional: Set to 'production' for production mode
```

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
- **Session Management**: Agent instances are automatically cleaned up after 30 min of inactivity
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