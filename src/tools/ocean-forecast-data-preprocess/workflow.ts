/**
 * @file workflow.ts
 * @description 预测数据预处理工作流 - 4 阶段分步确认逻辑
 *              函数式实现，替代原 workflow-state.ts 中的 ForecastWorkflow 类。
 *              Token 机制：Stage 4 生成 UUID，由 full.ts 写入 session JSON；下次调用验证。
 *
 * @author kongzhiquan
 * @date 2026-03-12
 * @version 2.0.0
 *
 * @changelog
 *   - 2026-03-12 kongzhiquan: v2.0.0 重构为函数式，移除 ForecastWorkflow 类
 *     - 删除 SHA-256 Token，改用 UUID（由 resolveStage 生成，full.ts 写入 session）
 *     - 暴露 resolveStage() 作为唯一控制流入口
 *     - 阶段判断和提示构建内聚在同一文件
 */

import crypto from 'crypto'

// ============================================================
// Types
// ============================================================

export interface ForecastPreprocessParams {
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

export type ForecastPreprocessStageStatus = 
  | 'awaiting_variable_selection' 
  | 'awaiting_static_selection' 
  | 'awaiting_parameters' 
  | 'awaiting_execution' 
  | 'token_invalid'

export interface ForecastPreprocessStageResult {
  status: ForecastPreprocessStageStatus
  message: string
  canExecute: boolean
  data?: Record<string, unknown>
}

// ============================================================
// Stage check predicates
// ============================================================

function hasStage1(p: ForecastPreprocessParams): boolean {
  return !!p.dyn_vars?.length
}

function hasStage2(p: ForecastPreprocessParams): boolean {
  return p.stat_vars !== undefined && p.mask_vars !== undefined
}

function hasStage3(p: ForecastPreprocessParams): boolean {
  return (
    p.train_ratio !== undefined &&
    p.valid_ratio !== undefined &&
    p.test_ratio !== undefined
  )
}

function hasAllStages(p: ForecastPreprocessParams): boolean {
  return hasStage1(p) && hasStage2(p) && hasStage3(p)
}

// ============================================================
// Prompt builders
// ============================================================

function buildStage1Prompt(inspectResult: any, params: ForecastPreprocessParams): ForecastPreprocessStageResult {
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

function buildStage2Prompt(inspectResult: any, params: ForecastPreprocessParams): ForecastPreprocessStageResult {
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

function buildStage3Prompt(inspectResult: any, params: ForecastPreprocessParams): ForecastPreprocessStageResult {
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

function buildStage4Prompt(
  inspectResult: any,
  params: ForecastPreprocessParams,
  token: string
): ForecastPreprocessStageResult {
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

// ============================================================
// Main entry point
// ============================================================

/**
 * 根据参数确定当前所处阶段并返回对应提示。
 *
 * @param params        当前有效参数（session 合并后的 effectiveArgs）
 * @param inspectResult Step A 的数据检查结果
 * @param sessionToken  上次 Stage 4 写入 session 的 UUID token
 * @returns ForecastStageResult（仍在某阶段）或 null（所有阶段通过，继续执行）
 */
export function resolveStage(
  params: ForecastPreprocessParams,
  inspectResult: any,
  sessionToken?: string
): ForecastPreprocessStageResult | null {
  // 所有阶段完成 + user_confirmed=true：验证 token
  if (params.user_confirmed === true && hasAllStages(params)) {
    if (!params.confirmation_token) {
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

    if (params.confirmation_token !== sessionToken) {
      return {
        status: 'token_invalid',
        message: `⚠️ Token 验证失败！

提供的 confirmation_token 与 session 中记录的不匹配。

可能原因：
1. Token 来自之前的调用，但参数已修改
2. Token 被错误地复制或截断

【解决方法】请重新调用工具（不带 user_confirmed），获取新的 confirmation_token。`,
        canExecute: false,
        data: { error_type: 'token_invalid', provided_token: params.confirmation_token }
      }
    }

    // 所有检查通过，返回 null 表示可继续执行
    return null
  }

  if (!hasStage1(params)) return buildStage1Prompt(inspectResult, params)
  if (!hasStage2(params)) return buildStage2Prompt(inspectResult, params)
  if (!hasStage3(params)) return buildStage3Prompt(inspectResult, params)

  // Stage 4：所有参数就绪，生成 UUID token 并等待用户确认
  const token = crypto.randomUUID()
  return buildStage4Prompt(inspectResult, params, token)
}
