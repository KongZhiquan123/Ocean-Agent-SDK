/**
 * @file validate.ts
 * @description Step B: 张量约定验证工具
 *              调用 Python 脚本验证变量形状
 *
 * @author leizheng
 * @contributors kongzhiquan
 * @date 2026-02-02
 * @version 2.1.0
 *
 * @changelog
 *   - 2026-02-25 kongzhiquan: v2.2.0 tempDir 改为基于 output_base 的 .ocean_preprocess_temp
 *     - 新增 output_base 参数，用于指定临时目录的基础路径
 *   - 2026-02-05 kongzhiquan: v2.1.0 移除 try-catch，统一由上层处理错误
 *     - 删除无用参数 inspect_result_path（改为自动从临时目录读取）
 *     - 错误时直接 throw Error 而非返回 status: 'error'
 *   - 2026-02-02 leizheng: v2.0.0 重构为调用独立 Python 脚本
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findFirstPythonPath } from '@/utils/python-manager'
import path from 'node:path'

function shellEscapeDouble(str: string): string {
  return str.replace(/[\\"$`!]/g, '\\$&')
}

// ========================================
// 类型定义
// ========================================

export interface ValidateResult {
  status: 'pass' | 'error' | 'pending'
  research_vars: string[]
  tensor_convention: Record<string, any>
  var_names_config: {
    dynamic: string[]
    static: string[]
    research: string[]
    mask: string[]
  }
  warnings: string[]
  errors: string[]
  message: string
}

// ========================================
// 工具定义
// ========================================

export const oceanValidateTensorTool = defineTool({
  name: 'ocean_validate_tensor',
  description: `Step B: 进行张量约定验证

验证变量的张量形状是否符合约定，生成 var_names 配置。

**防错规则**：
- B1: 动态变量必须是 [D, H, W] 或 [T, D, H, W] 形状
- B2: 静态变量必须是 [H, W] 形状
- B3: 研究变量必须在数据中存在
- B4: 掩码变量形状必须是 2D

**输入**：Step A 的结果文件路径 + 用户确认的研究变量列表
**返回**：验证结果、var_names配置、张量约定信息`,

  params: {
    research_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '用户确认的研究变量列表，如 ["uo", "vo"]'
    },
    mask_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '掩码变量列表',
      required: false,
      default: ['mask_rho', 'mask_u', 'mask_v', 'mask_psi']
    },
    output_base: {
      type: 'string',
      description: '输出基础目录，用于存放临时文件'
    }
  },

  attributes: {
    readonly: true,
    noEffect: true
  },

  async exec(args, ctx) {
    const {
      research_vars,
      mask_vars = ['mask_rho', 'mask_u', 'mask_v', 'mask_psi'],
      output_base
    } = args

    // 1. 检查 Python 环境
    const pythonPath = findFirstPythonPath()
    if (!pythonPath) {
      throw new Error('未找到可用的Python解释器，请安装Python或配置PYTHON/PYENV')
    }

    // 2. 准备路径
    const pythonCmd = `"${shellEscapeDouble(pythonPath)}"`
    const tempDir = path.resolve(output_base, '.ocean_preprocess_temp')
    const configPath = path.join(tempDir, 'validate_config.json')
    const outputPath = path.join(tempDir, 'validate_result.json')

    // Python 脚本路径
    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean_preprocess/validate_tensor.py')

    // 3. 准备配置
    const config = {
      inspect_result_path: path.join(tempDir, 'inspect_result.json'),
      research_vars,
      mask_vars
    }

    // 4. 写入配置
    await ctx.sandbox.fs.write(configPath, JSON.stringify(config, null, 2))

    // 5. 执行 Python 脚本
    const result = await ctx.sandbox.exec(
      `${pythonCmd} "${shellEscapeDouble(scriptPath)}" --config "${shellEscapeDouble(configPath)}" --output "${shellEscapeDouble(outputPath)}"`,
      { timeoutMs: 60000 }
    )

    if (result.code !== 0) {
      throw new Error(`Python执行失败: ${result.stderr}`)
    }

    // 6. 读取结果
    const jsonContent = await ctx.sandbox.fs.read(outputPath)
    const validateResult: ValidateResult = JSON.parse(jsonContent)
    if (validateResult.status === 'error') {
      throw new Error(`张量约定验证失败: ${validateResult.errors.join('; ')}`)
    }

    return validateResult
  }
})
