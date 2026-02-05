/**
 * @file config.ts
 *
 * @description 配置文件，包含环境变量加载、配置验证和依赖初始化
 * @author kongzhiquan
 * @date 2026-02-02
 * @version 1.1.0
 *
 * @changelog
 *   - 2026-02-05 kongzhiquan: v1.1.0 优化工具加载与模板配置
 *     - 将 loadAllTools() 函数改为常量 allTools，避免重复创建
 *     - ask 模式移除 bash_run，添加 ocean_inspect_data（只读数据检查）
 */

import dotenv from 'dotenv'
dotenv.config({ override: true })
import {
  JSONStore,
  AgentTemplateRegistry,
  ToolRegistry,
  SandboxFactory,
  AnthropicProvider,
  builtin,
  SkillsManager,
  createSkillsTool,
  type AgentDependencies,
} from '@shareai-lab/kode-sdk'

// 导入海洋数据预处理工具
import { oceanPreprocessTools } from './tools/ocean-preprocess'
import { findFirstPythonPath } from './utils/python-manager'

// ========================================
// 环境变量配置
// ========================================

export const config = {
  port: Number(process.env.KODE_API_PORT ?? '8787'),
  apiSecret: process.env.KODE_API_SECRET,
  anthropicApiKey: process.env.ANTHROPIC_API_KEY,
  anthropicModelId: process.env.ANTHROPIC_MODEL_ID ?? 'claude-sonnet-4-5-20250929',
  anthropicBaseUrl: process.env.ANTHROPIC_BASE_URL ?? 'https://yunwu.ai',
  kodeStorePath: process.env.KODE_STORE_PATH ?? './.kode',
  skillsDir: process.env.SKILLS_DIR ?? './.skills',
} as const

console.log('[config] 环境变量 ANTHROPIC_MODEL_ID:', process.env.ANTHROPIC_MODEL_ID)
console.log('[config] 最终使用模型:', config.anthropicModelId)

const skillsWhiteList= ['ocean-preprocess']

// ========================================
// 配置验证
// ========================================

export function validateConfig(): void {
  const errors: string[] = []
  const warnings: string[] = []

  // 必需配置
  if (!config.anthropicApiKey) {
    errors.push('未设置 ANTHROPIC_API_KEY 环境变量')
  }

  if (!config.anthropicModelId) {
    errors.push('未设置 ANTHROPIC_MODEL_ID 环境变量')
  }

  if (!config.anthropicBaseUrl) {
    errors.push('未设置 ANTHROPIC_BASE_URL 环境变量')
  }

  // 警告配置
  if (!config.apiSecret) {
    warnings.push('未设置 KODE_API_SECRET，服务将拒绝所有未认证请求')
  }

  // 输出警告
  warnings.forEach(w => console.warn(`[config] 警告：${w}`))

  // 有错误则抛出
  if (errors.length > 0) {
    throw new Error(`配置错误：\n${errors.map(e => `  - ${e}`).join('\n')}`)
  }

  console.log('[config] 配置验证通过')
}

// ========================================
// 依赖初始化
// ========================================

function createStore() {
  return new JSONStore(config.kodeStorePath)
}

// 创建 SkillsManager
const skillsManager = new SkillsManager(config.skillsDir, skillsWhiteList)
const allTools = [...builtin.fs(), ...builtin.bash(), ...builtin.todo(), ...oceanPreprocessTools, createSkillsTool(skillsManager)]

function createToolRegistry() {
  const registry = new ToolRegistry()
  // 注册所有工具
  allTools.forEach(tool => registry.register(tool.name, () => tool))
  console.log('[config] 已注册工具:', registry.list())
  return registry
}

function createTemplateRegistry() {
  const registry = new AgentTemplateRegistry()

  // 编程助手模板（edit 模式）
  registry.register({
    id: 'coding-assistant',
    name: '编程助手',
    desc: '可以读写文件、执行命令的编程助手',
    systemPrompt: `# 角色
你是专业的海洋科学 AI 助手，支持数据预处理、模型训练、推理预测、可视化分析等任务。使用用户的语言回复。

# 核心原则

1. **禁止自动决策**：所有关键参数必须由用户确认后才能执行
2. **警告优先**：遇到任何异常或警告，必须暂停并询问用户
3. **错误不自动重试**：遇到错误时展示给用户，等待指示
4. **输出持久化**：结果必须保存到文件，不要只在聊天中显示

# 工具使用

- Python 环境：${findFirstPythonPath() ?? '系统无可用的 Python 环境'}
- 避免危险命令（rm -rf /、sudo 等）

# Skills 系统（重要）

Skills 提供特定任务的详细指导。**执行专业任务前必须先加载对应 Skill**。

**查看可用技能**：
\`\`\`
skills {"action": "list"}
\`\`\`

**加载技能**（获取详细流程指导）：
\`\`\`
skills {"action": "load", "skill_name": "技能名称"}
\`\`\`

**已有技能**：
- ocean-preprocess: 海洋数据预处理（超分辨率、格式转换等）

**工作流程**：
1. 用户提出任务需求
2. 判断是否需要加载 Skill
3. 如需要，先加载 Skill 获取详细指导
4. 按 Skill 中的流程执行任务

# 通用安全原则

- 分阶段确认：复杂任务分步骤，每步确认后再继续
- Token 验证：部分工具有 confirmation_token 机制，必须正确使用
- 参数校验：用户提供的路径、参数在执行前确认
- 结果汇报：任务完成后向用户展示关键结果和输出路径`,
    tools: allTools.map(t => t.name),
  })

  // 问答助手模板（ask 模式）
  registry.register({
    id: 'qa-assistant',
    name: '问答助手',
    desc: '只读助手，专注于回答问题',
    systemPrompt: `# 角色与能力
你是一个乐于助人的 AI 助手。你可以读取文件和执行只读命令来回答用户的问题。
语言：如果用户使用中文，请用中文回复；否则使用用户的语言。

# 核心原则
1. 你只能读取信息，不能修改任何文件。
2. 专注于回答用户的问题，提供准确、有帮助的信息。
3. 如果需要查看代码或文件内容，使用 fs_read 工具。`,
    tools: ['fs_read', 'fs_glob', 'fs_grep', 'ocean_inspect_data'],
  })

  return registry
}

function createSandboxFactory() {
  return new SandboxFactory()
}

function createModelFactory() {
  return () => new AnthropicProvider(config.anthropicApiKey, config.anthropicModelId, config.anthropicBaseUrl)
}

// ========================================
// 导出依赖
// ========================================

let dependencies: AgentDependencies | null = null

export function getDependencies(): AgentDependencies {
  if (!dependencies) {
    dependencies = {
      store: createStore(),
      templateRegistry: createTemplateRegistry(),
      toolRegistry: createToolRegistry(),
      sandboxFactory: createSandboxFactory(),
      modelFactory: createModelFactory(),
    }
    console.log('[config] 依赖初始化完成')
  }
  return dependencies
}
