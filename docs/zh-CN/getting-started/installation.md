# 安装配置

## 环境要求

- **Node.js**: >= 18.0.0
- **npm** 或 **pnpm** 或 **yarn**

## 安装

```bash
npm install @shareai-lab/kode-sdk
```

或使用 pnpm/yarn：

```bash
pnpm add @shareai-lab/kode-sdk
yarn add @shareai-lab/kode-sdk
```

## 环境变量配置

KODE SDK 使用环境变量配置 API 密钥和模型。

### Anthropic（默认）

<!-- tabs:start -->
#### **Linux / macOS**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export ANTHROPIC_MODEL_ID=claude-sonnet-4-20250514  # 可选
export ANTHROPIC_BASE_URL=https://api.anthropic.com  # 可选
```

#### **Windows (PowerShell)**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-..."
$env:ANTHROPIC_MODEL_ID="claude-sonnet-4-20250514"  # 可选
$env:ANTHROPIC_BASE_URL="https://api.anthropic.com"  # 可选
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
export OPENAI_MODEL_ID=gpt-4o  # 可选
```

#### **Windows (PowerShell)**
```powershell
$env:OPENAI_API_KEY="sk-..."
$env:OPENAI_MODEL_ID="gpt-4o"  # 可选
```
<!-- tabs:end -->

### Google Gemini

<!-- tabs:start -->
#### **Linux / macOS**
```bash
export GOOGLE_API_KEY=...
export GEMINI_MODEL_ID=gemini-2.0-flash  # 可选
```

#### **Windows (PowerShell)**
```powershell
$env:GOOGLE_API_KEY="..."
$env:GEMINI_MODEL_ID="gemini-2.0-flash"  # 可选
```
<!-- tabs:end -->

## 使用 .env 文件

在项目根目录创建 `.env` 文件：

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL_ID=claude-sonnet-4-20250514
```

在代码中加载：

```typescript
import 'dotenv/config';
// 或
import { config } from 'dotenv';
config();
```

## 验证安装

```typescript
import { Agent, AnthropicProvider, JSONStore } from '@shareai-lab/kode-sdk';

console.log('KODE SDK 安装成功！');
```

## 下一步

- [快速上手](./quickstart.md) - 创建第一个 Agent
- [核心概念](./concepts.md) - 理解核心概念
