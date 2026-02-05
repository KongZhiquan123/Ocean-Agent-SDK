/**
 * @file validate.ts
 * @description Step B: 张量约定验证工具
 *              调用 Python 脚本验证变量形状
 *
 * @author leizheng
 * @date 2026-02-02
 * @version 2.0.0
 *
 * @changelog
 *   - 2026-02-02 leizheng: v2.0.0 重构为调用独立 Python 脚本
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findFirstPythonPath } from '@/utils/python-manager'
import path from 'node:path'

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
- B1: 动态变量必须是 [T, H, W] 或 [T, D, H, W] 形状
- B2: 静态变量必须是 [H, W] 形状
- B3: 研究变量必须在数据中存在
- B4: 掩码变量形状必须是 2D

**输入**：Step A 的结果文件路径 + 用户确认的研究变量列表
**返回**：验证结果、var_names配置、张量约定信息`,

  params: {
    inspect_result_path: {
      type: 'string',
      description: 'Step A 生成的 inspect_result.json 文件路径'
    },
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
    }
  },

  attributes: {
    readonly: true,
    noEffect: true
  },

  async exec(args, ctx) {
    const {
      inspect_result_path,
      research_vars,
      mask_vars = ['mask_rho', 'mask_u', 'mask_v', 'mask_psi']
    } = args

    // 1. 检查 Python 环境
    const pythonPath = findFirstPythonPath()
    if (!pythonPath) {
      const errorMsg = '未找到可用的Python解释器，请安装Python或配置PYTHON/PYENV'
      return {
        status: 'error',
        errors: [errorMsg],
        message: '张量验证失败'
      }
    }

    // 2. 准备路径
    const pythonCmd = `"${pythonPath}"`
    const tempDir = path.resolve(ctx.sandbox.workDir, 'ocean_preprocess_temp')
    const configPath = path.join(tempDir, 'validate_config.json')
    const outputPath = path.join(tempDir, 'validate_result.json')

    // Python 脚本路径
    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean_preprocess/validate_tensor.py')

    // 3. 准备配置
    const config = {
      inspect_result_path,
      research_vars,
      mask_vars
    }

    try {
      // 4. 写入配置
      await ctx.sandbox.fs.write(configPath, JSON.stringify(config, null, 2))

      // 5. 执行 Python 脚本
      const result = await ctx.sandbox.exec(
        `${pythonCmd} "${scriptPath}" --config "${configPath}" --output "${outputPath}"`,
        { timeoutMs: 60000 }
      )

      if (result.code !== 0) {
        return {
          status: 'error',
          errors: [`Python执行失败: ${result.stderr}`],
          message: '张量验证失败'
        }
      }

      // 6. 读取结果
      const jsonContent = await ctx.sandbox.fs.read(outputPath)
      const validateResult: ValidateResult = JSON.parse(jsonContent)

      return validateResult

    } catch (error: any) {
      return {
        status: 'error',
        errors: [error.message],
        message: '张量验证执行异常'
      }
    }
  }
})
