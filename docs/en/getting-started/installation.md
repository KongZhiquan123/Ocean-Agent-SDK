# Installation

## Requirements

- **Node.js**: >= 18.0.0
- **npm** or **pnpm** or **yarn**

## Install

```bash
npm install @shareai-lab/kode-sdk
```

Or with pnpm/yarn:

```bash
pnpm add @shareai-lab/kode-sdk
yarn add @shareai-lab/kode-sdk
```

## Environment Variables

KODE SDK uses environment variables for API keys and model configuration.

### Anthropic (Default)

<!-- tabs:start -->
#### **Linux / macOS**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export ANTHROPIC_MODEL_ID=claude-sonnet-4-20250514  # optional
export ANTHROPIC_BASE_URL=https://api.anthropic.com  # optional
```

#### **Windows (PowerShell)**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-..."
$env:ANTHROPIC_MODEL_ID="claude-sonnet-4-20250514"  # optional
$env:ANTHROPIC_BASE_URL="https://api.anthropic.com"  # optional
```

#### **Windows (CMD)**
```cmd
set ANTHROPIC_API_KEY=sk-ant-...
set ANTHROPIC_MODEL_ID=claude-sonnet-4-20250514
```
<!-- tabs:end -->

### OpenAI

<!-- tabs:start -->
#### **Linux / macOS**
```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL_ID=gpt-4o  # optional
```

#### **Windows (PowerShell)**
```powershell
$env:OPENAI_API_KEY="sk-..."
$env:OPENAI_MODEL_ID="gpt-4o"  # optional
```
<!-- tabs:end -->

### Google Gemini

<!-- tabs:start -->
#### **Linux / macOS**
```bash
export GOOGLE_API_KEY=...
export GEMINI_MODEL_ID=gemini-2.0-flash  # optional
```

#### **Windows (PowerShell)**
```powershell
$env:GOOGLE_API_KEY="..."
$env:GEMINI_MODEL_ID="gemini-2.0-flash"  # optional
```
<!-- tabs:end -->

## Using .env File

Create a `.env` file in your project root:

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL_ID=claude-sonnet-4-20250514
```

Load it in your code:

```typescript
import 'dotenv/config';
// or
import { config } from 'dotenv';
config();
```

## Verify Installation

```typescript
import { Agent, AnthropicProvider, JSONStore } from '@shareai-lab/kode-sdk';

console.log('KODE SDK installed successfully!');
```

## Next Steps

- [Quickstart](./quickstart.md) - Build your first Agent
- [Concepts](./concepts.md) - Understand core concepts
