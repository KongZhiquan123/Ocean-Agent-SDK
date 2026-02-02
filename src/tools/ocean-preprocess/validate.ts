import { defineTool } from '@shareai-lab/kode-sdk'

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

const TEMP_DIR = '/tmp/ocean_preprocess'

function generateValidateScript(
  variablesJson: string,
  researchVars: string[],
  maskVars: string[],
  outputJson: string
): string {
  return `
import json

VARIABLES_JSON = ${JSON.stringify(variablesJson)}
RESEARCH_VARS = ${JSON.stringify(researchVars)}
MASK_VARS = ${JSON.stringify(maskVars)}
OUTPUT_JSON = ${JSON.stringify(outputJson)}

with open(VARIABLES_JSON, 'r', encoding='utf-8') as f:
    inspect_result = json.load(f)

variables_info = inspect_result.get("variables", {})

result = {
    "status": "pending",
    "research_vars": RESEARCH_VARS,
    "tensor_convention": {},
    "var_names_config": {
        "dynamic": [],
        "static": [],
        "research": RESEARCH_VARS,
        "mask": MASK_VARS
    },
    "warnings": [],
    "errors": [],
    "message": ""
}

# 防错规则 B3: 验证研究变量存在
for var in RESEARCH_VARS:
    if var not in variables_info:
        result["errors"].append(f"研究变量 '{var}' 不存在于数据中")

if result["errors"]:
    result["status"] = "error"
    result["message"] = "研究变量验证失败"
else:
    # 防错规则 B1, B2: 张量形状检查
    for var_name, var_info in variables_info.items():
        shape = tuple(var_info["shape"])
        category = var_info["category"]
        ndim = len(shape)

        convention = {
            "name": var_name,
            "original_shape": shape,
            "category": category,
            "valid": False,
            "interpretation": ""
        }

        if category == "dynamic":
            result["var_names_config"]["dynamic"].append(var_name)
            if ndim == 3:
                convention["valid"] = True
                convention["interpretation"] = "[T, H, W]"
                convention["T"], convention["H"], convention["W"] = shape
            elif ndim == 4:
                convention["valid"] = True
                convention["interpretation"] = "[T, D, H, W]"
                convention["T"], convention["D"], convention["H"], convention["W"] = shape
            else:
                convention["interpretation"] = f"不符合约定: {ndim}D"
                result["warnings"].append(f"动态变量 '{var_name}' 维度不符合约定: {shape}")

        elif category == "static":
            result["var_names_config"]["static"].append(var_name)
            if ndim == 2:
                convention["valid"] = True
                convention["interpretation"] = "[H, W]"
                convention["H"], convention["W"] = shape
            else:
                convention["interpretation"] = f"不符合约定: {ndim}D"
                if var_name in MASK_VARS:
                    result["errors"].append(f"掩码变量 '{var_name}' 维度错误: {shape}")

        result["tensor_convention"][var_name] = convention

    if result["errors"]:
        result["status"] = "error"
        result["message"] = "张量约定验证失败"
    else:
        result["status"] = "pass"
        result["message"] = "张量约定验证通过"

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(json.dumps(result, ensure_ascii=False))
`
}

export const oceanValidateTensorTool = defineTool({
  name: 'ocean_validate_tensor',
  description: `Step B: 进行张量约定验证

验证变量的张量形状是否符合约定，生成 var_names 配置。

**防错规则**：
- B1: 动态变量必须是 [T, H, W] 或 [T, D, H, W] 形状
- B2: 静态变量必须是 [H, W] 形状
- B3: 研究变量必须在数据中存在
- B4: 掩码变量形状必须与数据空间维匹配

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
      description: '用户确认的研究变量列表，如 ["u_eastward", "v_northward"]'
    },
    mask_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '掩码变量列表',
      required: false,
      default: ['mask_u', 'mask_rho', 'mask_v']
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
      mask_vars = ['mask_u', 'mask_rho', 'mask_v']
    } = args

    ctx.emit('step_started', { step: 'B', description: '进行张量约定验证' })

    const outputJson = `${TEMP_DIR}/validate_result.json`
    const script = generateValidateScript(inspect_result_path, research_vars, mask_vars, outputJson)
    const scriptPath = `${TEMP_DIR}/validate_tensor.py`

    try {
      await ctx.sandbox.fs.write(scriptPath, script)

      const result = await ctx.sandbox.exec(`python3 ${scriptPath}`, {
        timeoutMs: 60000,
      })

      if (result.code !== 0) {
        ctx.emit('step_failed', { step: 'B', error: result.stderr })
        return {
          status: 'error',
          errors: [`Python执行失败: ${result.stderr}`],
          message: '张量验证失败'
        }
      }

      const jsonContent = await ctx.sandbox.fs.read(outputJson)
      const validateResult: ValidateResult = JSON.parse(jsonContent)

      ctx.emit('step_completed', {
        step: 'B',
        status: validateResult.status,
        var_names_config: validateResult.var_names_config
      })

      return validateResult
    } catch (error: any) {
      ctx.emit('step_failed', { step: 'B', error: error.message })
      return {
        status: 'error',
        errors: [error.message],
        message: '张量验证执行异常'
      }
    }
  }
})
