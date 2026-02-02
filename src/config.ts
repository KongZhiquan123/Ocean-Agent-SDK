// src/config.ts
// 配置和依赖初始化

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

function createToolRegistry() {
  const registry = new ToolRegistry()

  // 注册文件系统工具
  for (const tool of builtin.fs()) {
    registry.register(tool.name, () => tool)
  }

  // 注册 Bash 工具
  for (const tool of builtin.bash()) {
    registry.register(tool.name, () => tool)
  }

  // 注册 Todo 工具
  for (const tool of builtin.todo()) {
    registry.register(tool.name, () => tool)
  }

  // 注册海洋数据预处理工具
  for (const tool of oceanPreprocessTools) {
    registry.register(tool.name, () => tool)
  }

  // 注册 Skills 工具
  const skillsTool = createSkillsTool(skillsManager)
  registry.register('skills', () => skillsTool)

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
    systemPrompt: `# 角色与能力
你是一个专业的 AI 开发者助手。你可以访问文件系统和专用工具来设计、编码和执行解决方案。
语言：如果用户使用中文，请用中文回复；否则使用用户的语言。

# 核心原则：持久化输出
1. 所有工作必须是持久化的、可复现的、可审查的。
2. 不要只在聊天中显示代码/内容。你必须将所有输出实际保存到指定的文件路径。

# 执行流程
1. 分析请求。
2. 设计解决方案。
3. 调用适当的工具写入文件/执行代码。
4. 向用户确认哪些文件已更新/创建。

# Skills 使用
- 查看可用技能：skills 工具，参数 {"action": "list"}
- 加载特定技能：skills 工具，参数 {"action": "load", "skill_name": "ocean-preprocess"}
- 当用户需要进行海洋数据预处理时，先加载 ocean-preprocess 技能获取完整流程指导。

# 海洋数据预处理注意事项
当使用 ocean_preprocess_full 工具时：
1. **首次调用**：仅提供必需参数（nc_folder, output_base, research_vars）
2. **如果返回 awaiting_confirmation 状态**：
   - 工具会在 message 中展示 suspected_masks 和 suspected_coordinates
   - 你必须向用户展示这些信息，并询问确认：
     * "检测到疑似掩码变量: [列表]，是否正确？"
     * "检测到疑似坐标变量: [列表]，哪些需要保存为静态数据？"
   - 等待用户回复
3. **用户确认后**：使用用户确认的变量列表重新调用 ocean_preprocess_full，这次提供 mask_vars 和 static_vars 参数
4. **如果用户不确定**：建议使用默认的 ROMS 配置（工具会自动使用默认值）
5. **第二次调用时**：工具会跳过确认步骤，直接执行完整的 A→B→C 流程`,
    tools: [
      'fs_read',
      'fs_write',
      'fs_edit',
      'fs_glob',
      'fs_grep',
      'bash_run',
      'todo_read',
      'todo_write',
      'skills',
      'ocean_inspect_data',
      'ocean_validate_tensor',
      'ocean_convert_npy',
      'ocean_preprocess_full',
    ],
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
    tools: ['fs_read', 'fs_glob', 'fs_grep', 'bash_run'],
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
