/**
 * @file workflow.ts
 * @description 预测数据预处理工作流 - 4 阶段分步确认逻辑
 *              函数式实现，替代原 workflow-state.ts 中的 ForecastWorkflow 类。
 *              Token 机制：SHA-256 签名覆盖关键参数字段，参数变更后 token 自动失效。
 *
 * @author kongzhiquan
 * @contributors Leizheng
 * @date 2026-03-12
 * @version 2.1.1
 *
 * @changelog
 *   - 2026-03-12 kongzhiquan: v2.1.1 将 hasStageX/buildStageX 重命名为语义化函数名
 *   - 2026-03-12 kongzhiquan: v2.1.0 Token 从 UUID 改为 SHA-256 签名
 *     - 签名覆盖 dyn_vars, stat_vars, mask_vars, train/valid/test_ratio,
 *       h_slice, w_slice, crop_lon_range, crop_lat_range
 *     - resolveStage() 不再需要 sessionToken 参数
 *     - full.ts 不再需要存储/读取 _execution_token
 *   - 2026-03-12 kongzhiquan: v2.0.0 重构为函数式，移除 ForecastWorkflow 类
 *     - 暴露 resolveStage() 作为唯一控制流入口
 *     - 阶段判断和提示构建内聚在同一文件
 */

import crypto from 'crypto'

// ============================================================
// Constants
// ============================================================

const TOKEN_SALT = 'ocean-forecast-preprocess-v1'

// ============================================================
// Types
// ============================================================

export interface ForecastPreprocessWorkflowParams {
  nc_folder: string
  output_base: string
  dyn_vars?: string[]
  stat_vars?: string[]
  mask_vars?: string[]
  train_ratio?: number
  valid_ratio?: number
  test_ratio?: number
  h_slice?: string
  w_slice?: string
  crop_lon_range?: number[]
  crop_lat_range?: number[]
  user_confirmed?: boolean
  confirmation_token?: string
  [key: string]: any
}

export type ForecastPreprocessWorkflowState =
  | 'awaiting_variable_selection'
  | 'awaiting_static_selection'
  | 'awaiting_parameters'
  | 'awaiting_execution'
  | 'token_invalid'

export interface ForecastPreprocessStagePromptResult {
  status: ForecastPreprocessWorkflowState
  message: string
  canExecute: boolean
  data?: Record<string, unknown>
}

// ============================================================
// Stage check predicates
// ============================================================

function hasDynamicVarsSelected(p: ForecastPreprocessWorkflowParams): boolean {
  return !!p.dyn_vars?.length
}

function hasStaticAndMaskVarsSelected(p: ForecastPreprocessWorkflowParams): boolean {
  return p.stat_vars !== undefined && p.mask_vars !== undefined
}

function hasSplitRatiosConfigured(p: ForecastPreprocessWorkflowParams): boolean {
  return (
    p.train_ratio !== undefined &&
    p.valid_ratio !== undefined &&
    p.test_ratio !== undefined
  )
}

function hasAllRequiredParams(p: ForecastPreprocessWorkflowParams): boolean {
  return (
    hasDynamicVarsSelected(p) &&
    hasStaticAndMaskVarsSelected(p) &&
    hasSplitRatiosConfigured(p)
  )
}

// ============================================================
// Token generation / validation
// ============================================================

/**
 * 生成 SHA-256 确认 Token。
 * 签名覆盖所有用户确认的关键参数，任何变更都会使 token 失效。
 */
export function generateConfirmationToken(params: ForecastPreprocessWorkflowParams): string {
  const tokenData = {
    nc_folder: params.nc_folder,
    output_base: params.output_base,
    dyn_vars: params.dyn_vars ? [...params.dyn_vars].sort().join(',') : '',
    stat_vars: params.stat_vars ? [...params.stat_vars].sort().join(',') : '',
    mask_vars: params.mask_vars ? [...params.mask_vars].sort().join(',') : '',
    train_ratio: params.train_ratio,
    valid_ratio: params.valid_ratio,
    test_ratio: params.test_ratio,
    h_slice: params.h_slice ?? null,
    w_slice: params.w_slice ?? null,
    crop_lon_range: params.crop_lon_range ? params.crop_lon_range.join(',') : '',
    crop_lat_range: params.crop_lat_range ? params.crop_lat_range.join(',') : '',
  }
  const dataStr = JSON.stringify(tokenData) + TOKEN_SALT
  return crypto.createHash('sha256').update(dataStr).digest('hex').substring(0, 16)
}

function validateConfirmationToken(params: ForecastPreprocessWorkflowParams): boolean {
  if (!params.confirmation_token) return false
  return params.confirmation_token === generateConfirmationToken(params)
}

// ============================================================
// Prompt builders
// ============================================================

function buildVariableSelectionPrompt(inspectResult: any, params: ForecastPreprocessWorkflowParams): ForecastPreprocessStagePromptResult {
  const dynCandidates: string[] = inspectResult?.dynamic_vars_candidates || []
  const variables = inspectResult?.variables || {}

  const varLines = dynCandidates.map((name: string) => {
    const info = variables[name]
    if (!info) return `  - ${name}`
    const dims = info.dims?.join(',') || '?'
    const shape = info.shape?.join('×') || '?'
    return `  - ${name}: 形状 (${shape}), 维度 [${dims}], ${info.dtype || '?'}`
  }).join('\n') || '  无'

  return {
    status: 'awaiting_variable_selection',
    message: `数据分析完成！

================================================================================
                     ⚠️ 请选择研究变量（必须）
================================================================================

【数据概况】
- 数据目录: ${params.nc_folder}
- 文件数量: ${inspectResult?.file_count || '?'} 个

【动态变量候选】（有时间维度，可作为预测目标）
${varLines}

【疑似静态/坐标变量】
${(inspectResult?.suspected_coordinates || []).map((v: string) => `  - ${v}`).join('\n') || '  无'}

【疑似掩码变量】
${(inspectResult?.suspected_masks || []).map((v: string) => `  - ${v}`).join('\n') || '  无'}

================================================================================

**请回答以下问题：**

1️⃣ **您要预测哪些变量？**
   可选: ${dynCandidates.join(', ') || '无'}
   （请从上面的动态变量候选中选择）

================================================================================

⚠️ Agent 注意：**禁止自动推断研究变量！**
必须等待用户明确指定后，再使用 dyn_vars 参数重新调用。`,
    canExecute: false,
    data: {
      dynamic_vars_candidates: dynCandidates,
      suspected_coordinates: inspectResult?.suspected_coordinates,
      suspected_masks: inspectResult?.suspected_masks
    }
  }
}

function buildStaticAndMaskSelectionPrompt(inspectResult: any, params: ForecastPreprocessWorkflowParams): ForecastPreprocessStagePromptResult {
  return {
    status: 'awaiting_static_selection',
    message: `研究变量已确认：${params.dyn_vars?.join(', ')}

================================================================================
                  ⚠️ 请选择静态变量和掩码变量
================================================================================

【疑似静态/坐标变量】（建议保存用于可视化和后处理）
${(inspectResult?.suspected_coordinates || []).map((v: string) => `  - ${v}`).join('\n') || '  无检测到'}

【疑似掩码变量】（用于区分海洋/陆地区域）
${(inspectResult?.suspected_masks || []).map((v: string) => `  - ${v}`).join('\n') || '  无检测到'}

================================================================================

**请回答以下问题：**

2️⃣ **需要保存哪些静态变量？**
   可选: ${(inspectResult?.suspected_coordinates || []).join(', ') || '无'}
   （如果不需要，请指定 stat_vars: []）

3️⃣ **使用哪些掩码变量？**
   可选: ${(inspectResult?.suspected_masks || []).join(', ') || '无'}
   （如果数据没有掩码，请指定 mask_vars: []）

================================================================================

⚠️ Agent 注意：**禁止自动决定静态变量和掩码变量！**
必须等待用户明确指定后，再使用 stat_vars 和 mask_vars 参数重新调用。`,
    canExecute: false,
    data: {
      dyn_vars_confirmed: params.dyn_vars,
      suspected_coordinates: inspectResult?.suspected_coordinates,
      suspected_masks: inspectResult?.suspected_masks
    }
  }
}

function buildProcessingParametersPrompt(inspectResult: any, params: ForecastPreprocessWorkflowParams): ForecastPreprocessStagePromptResult {
  const firstVar = params.dyn_vars?.[0]
  const varInfo = inspectResult?.variables?.[firstVar]
  const dataShape = varInfo?.shape || []
  const H = dataShape.length >= 2 ? dataShape[dataShape.length - 2] : '?'
  const W = dataShape.length >= 1 ? dataShape[dataShape.length - 1] : '?'

  return {
    status: 'awaiting_parameters',
    message: `变量选择已确认：
- 研究变量: ${params.dyn_vars?.join(', ')}
- 静态变量: ${params.stat_vars?.length ? params.stat_vars.join(', ') : '无'}
- 掩码变量: ${params.mask_vars?.length ? params.mask_vars.join(', ') : '无'}

================================================================================
                    ⚠️ 请确认处理参数
================================================================================

【当前数据形状】
- 空间尺寸: H=${H}, W=${W}
- 文件数量: ${inspectResult?.file_count || '?'} 个

================================================================================

**请回答以下问题：**

4️⃣ **数据集划分比例？**（三者之和必须为 1.0，按时间顺序划分，不打乱）
   - train_ratio: 训练集比例（如 0.7）
   - valid_ratio: 验证集比例（如 0.15）
   - test_ratio: 测试集比例（如 0.15）

5️⃣ **空间裁剪？**（可选，不需要可跳过）
   - 当前尺寸: ${H} × ${W}
   - h_slice: H 方向裁剪，如 "0:512"
   - w_slice: W 方向裁剪，如 "0:512"
   - crop_lon_range: 经度裁剪范围，如 [110, 130]
   - crop_lat_range: 纬度裁剪范围，如 [20, 40]

================================================================================

⚠️ Agent 注意：**禁止自动决定处理参数！**
必须等待用户明确指定后，再传入相应参数重新调用。`,
    canExecute: false,
    data: {
      dyn_vars_confirmed: params.dyn_vars,
      stat_vars_confirmed: params.stat_vars,
      mask_vars_confirmed: params.mask_vars,
      data_shape: { H, W },
      file_count: inspectResult?.file_count
    }
  }
}

function buildExecutionConfirmationPrompt(
  inspectResult: any,
  params: ForecastPreprocessWorkflowParams,
  token: string
): ForecastPreprocessStagePromptResult {
  const firstVar = params.dyn_vars?.[0]
  const varInfo = inspectResult?.variables?.[firstVar]
  const dataShape = varInfo?.shape || []
  const originalH = dataShape.length >= 2 ? dataShape[dataShape.length - 2] : '?'
  const originalW = dataShape.length >= 1 ? dataShape[dataShape.length - 1] : '?'

  const cropLines = (params.h_slice || params.w_slice || params.crop_lon_range || params.crop_lat_range)
    ? [
        params.h_slice        ? `- H 裁剪: ${params.h_slice}` : '',
        params.w_slice        ? `- W 裁剪: ${params.w_slice}` : '',
        params.crop_lon_range ? `- 经度范围: [${params.crop_lon_range.join(', ')}]` : '',
        params.crop_lat_range ? `- 纬度范围: [${params.crop_lat_range.join(', ')}]` : ''
      ].filter(Boolean).join('\n')
    : '- 不裁剪'

  return {
    status: 'awaiting_execution',
    message: `所有参数已确认，请检查后确认执行：

================================================================================
                         📋 处理参数汇总
================================================================================

【数据信息】
- 数据目录: ${params.nc_folder}
- 文件数量: ${inspectResult?.file_count || '?'} 个
- 输出目录: ${params.output_base}

【变量配置】
- 研究变量: ${params.dyn_vars?.join(', ')}
- 静态变量: ${params.stat_vars?.length ? params.stat_vars.join(', ') : '无'}
- 掩码变量: ${params.mask_vars?.length ? params.mask_vars.join(', ') : '无'}

【处理模式】
- 模式: 预测数据预处理（无下采样）
- 时间排序: 严格时间升序（按 NC 文件内时间变量）

【空间裁剪】
- 原始尺寸: ${originalH} × ${originalW}
${cropLines}

【数据集划分】
- 训练集: ${((params.train_ratio || 0) * 100).toFixed(0)}%
- 验证集: ${((params.valid_ratio || 0) * 100).toFixed(0)}%
- 测试集: ${((params.test_ratio || 0) * 100).toFixed(0)}%

================================================================================

⚠️ **请确认以上参数无误后，回复"确认执行"**

如需修改任何参数，请直接告诉我要修改的内容。

================================================================================

🔐 **执行确认 Token**: ${token}
（Agent 必须将上面一段话发送给用户等待确认，同时必须在下次调用时携带此 token 和 user_confirmed=true）`,
    canExecute: false,
    data: {
      confirmation_token: token,
      summary: {
        dyn_vars: params.dyn_vars,
        stat_vars: params.stat_vars,
        mask_vars: params.mask_vars,
        train_ratio: params.train_ratio,
        valid_ratio: params.valid_ratio,
        test_ratio: params.test_ratio,
        h_slice: params.h_slice,
        w_slice: params.w_slice,
        crop_lon_range: params.crop_lon_range,
        crop_lat_range: params.crop_lat_range
      }
    }
  }
}

function buildTokenInvalidPrompt(
  params: ForecastPreprocessWorkflowParams,
  mode: 'missing_token' | 'token_mismatch'
): ForecastPreprocessStagePromptResult {
  if (mode === 'missing_token') {
    return {
      status: 'token_invalid',
      message: `⚠️ 检测到跳步行为！

您设置了 user_confirmed=true，但未提供 confirmation_token。

必须：
1. 先调用工具（不带 user_confirmed），进入 awaiting_execution 阶段
2. 从返回结果中获取 confirmation_token
3. 用户确认后，再次调用并携带 user_confirmed=true 和 confirmation_token`,
      canExecute: false,
      data: { error_type: 'token_invalid' }
    }
  }

  return {
    status: 'token_invalid',
    message: `⚠️ Token 验证失败！

提供的 confirmation_token 与当前参数不匹配。

可能原因：
1. Token 生成后参数被修改（如 dyn_vars, 比例, 裁剪范围等）
2. Token 被错误地复制或截断

【当前参数快照】
- nc_folder: ${params.nc_folder}
- output_base: ${params.output_base}
- dyn_vars: ${params.dyn_vars?.join(', ')}
- stat_vars: ${params.stat_vars?.join(', ') ?? '未设置'}
- mask_vars: ${params.mask_vars?.join(', ') ?? '未设置'}
- train_ratio: ${params.train_ratio}
- valid_ratio: ${params.valid_ratio}
- test_ratio: ${params.test_ratio}
- h_slice: ${params.h_slice ?? '无'}
- w_slice: ${params.w_slice ?? '无'}
- crop_lon_range: ${params.crop_lon_range ? `[${params.crop_lon_range.join(', ')}]` : '无'}
- crop_lat_range: ${params.crop_lat_range ? `[${params.crop_lat_range.join(', ')}]` : '无'}

【解决方法】请重新调用工具（不带 user_confirmed），获取新的 confirmation_token。`,
    canExecute: false,
    data: {
      error_type: 'token_invalid',
      provided_token: params.confirmation_token,
      expected_token: generateConfirmationToken(params),
    }
  }
}

// ============================================================
// Main entry point
// ============================================================

/**
 * 根据参数确定当前所处阶段并返回对应提示。
 *
 * @param params        当前有效参数（session 合并后的 effectiveArgs）
 * @param inspectResult Step A 的数据检查结果
 * @returns ForecastPreprocessStageResult（仍在某阶段）或 null（所有阶段通过，继续执行）
 */
export function resolveStage(
  params: ForecastPreprocessWorkflowParams,
  inspectResult: any,
): ForecastPreprocessStagePromptResult | null {
  // 所有阶段完成 + user_confirmed=true：验证 token
  if (params.user_confirmed === true && hasAllRequiredParams(params)) {
    if (!params.confirmation_token) {
      return buildTokenInvalidPrompt(params, 'missing_token')
    }

    if (!validateConfirmationToken(params)) {
      return buildTokenInvalidPrompt(params, 'token_mismatch')
    }

    // 所有检查通过，返回 null 表示可继续执行
    return null
  }

  if (!hasDynamicVarsSelected(params)) return buildVariableSelectionPrompt(inspectResult, params)
  if (!hasStaticAndMaskVarsSelected(params)) return buildStaticAndMaskSelectionPrompt(inspectResult, params)
  if (!hasSplitRatiosConfigured(params)) return buildProcessingParametersPrompt(inspectResult, params)

  // Stage 4：所有参数就绪，生成 SHA-256 token 并等待用户确认
  const token = generateConfirmationToken(params)
  return buildExecutionConfirmationPrompt(inspectResult, params, token)
}
