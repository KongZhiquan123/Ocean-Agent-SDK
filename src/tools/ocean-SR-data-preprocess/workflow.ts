/**
 * @file workflow.ts
 * @description 超分辨率数据预处理工作流 - 5 阶段分步确认逻辑（含阶段 2.5 区域裁剪）
 *              函数式实现，替代原 workflow-state.ts 中的 PreprocessWorkflow 类。
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
 *     - 签名覆盖 dyn/stat/mask、区域裁剪、下采样、切分比例、空间裁剪、模式参数
 *     - resolveStage() 不再需要 sessionToken 参数
 *   - 2026-03-12 kongzhiquan: v2.0.0 重构为函数式，移除 PreprocessWorkflow 类
 *     - 删除 SHA-256 Token，改用 UUID（由 resolveStage 生成，full.ts 写入 session）
 *     - 暴露 resolveStage() 作为唯一控制流入口
 *     - 阶段判断和提示构建内聚在同一文件
 */

import crypto from 'crypto'

// ============================================================
// Constants
// ============================================================

const TOKEN_SALT = 'ocean-sr-preprocess-v1'

// ============================================================
// Types
// ============================================================

export interface SrPreprocessWorkflowParams {
  nc_folder: string
  output_base: string

  // 阶段1: 研究变量
  dyn_vars?: string[]

  // 阶段2: 静态/掩码变量
  stat_vars?: string[]
  mask_vars?: string[]

  // 阶段2.5: 区域裁剪
  /** undefined=未回答, true=启用裁剪, false=不启用裁剪 */
  enable_region_crop?: boolean
  crop_lon_range?: [number, number]
  crop_lat_range?: [number, number]
  crop_mode?: 'one_step' | 'two_step'

  // 阶段3: 处理参数
  scale?: number
  downsample_method?: string
  train_ratio?: number
  valid_ratio?: number
  test_ratio?: number
  h_slice?: string
  w_slice?: string

  // 阶段4: 最终确认
  user_confirmed?: boolean
  confirmation_token?: string

  // 粗网格模式
  lr_nc_folder?: string

  [key: string]: any
}

export type SrPreprocessWorkflowState =
  | 'awaiting_variable_selection'
  | 'awaiting_static_selection'
  | 'awaiting_region_selection'
  | 'awaiting_parameters'
  | 'awaiting_execution'
  | 'token_invalid'

export interface SrPreprocessStagePromptResult {
  status: SrPreprocessWorkflowState
  message: string
  canExecute: boolean
  data?: Record<string, unknown>
}
// ============================================================
// Stage check predicates
// ============================================================

function hasDynamicVarsSelected(p: SrPreprocessWorkflowParams): boolean {
  return !!p.dyn_vars?.length
}

function hasStaticAndMaskVarsSelected(p: SrPreprocessWorkflowParams): boolean {
  return p.stat_vars !== undefined && p.mask_vars !== undefined
}

/**
 * 阶段2.5：区域裁剪决定
 * - false  → 明确不裁剪，通过
 * - true   → 必须同时提供有效的经纬度范围才算通过
 * - undefined → 未回答，未通过
 */
function hasRegionCropDecisionConfigured(p: SrPreprocessWorkflowParams): boolean {
  if (p.enable_region_crop === false) return true
  if (p.enable_region_crop === true) {
    return !!(
      p.crop_lon_range && p.crop_lon_range.length === 2 &&
      p.crop_lat_range && p.crop_lat_range.length === 2
    )
  }
  return false
}

function hasProcessingParamsConfigured(p: SrPreprocessWorkflowParams): boolean {
  const isNumericalModelMode = !!p.lr_nc_folder
  const hasSplitRatios = (
    p.train_ratio !== undefined &&
    p.valid_ratio !== undefined &&
    p.test_ratio !== undefined
  )
  if (!hasSplitRatios) return false
  if (!isNumericalModelMode) {
    return !!(p.scale && p.scale > 1 && p.downsample_method)
  }
  return true
}

function hasAllRequiredParams(p: SrPreprocessWorkflowParams): boolean {
  return (
    hasDynamicVarsSelected(p) &&
    hasStaticAndMaskVarsSelected(p) &&
    hasRegionCropDecisionConfigured(p) &&
    hasProcessingParamsConfigured(p)
  )
}

// ============================================================
// Token generation / validation
// ============================================================

/**
 * 生成 SHA-256 确认 Token。
 * 签名覆盖所有用户确认的关键参数，任何变更都会使 token 失效。
 */
export function generateConfirmationToken(params: SrPreprocessWorkflowParams): string {
  const tokenData = {
    nc_folder: params.nc_folder,
    output_base: params.output_base,
    dyn_vars: params.dyn_vars ? [...params.dyn_vars].sort().join(',') : '',
    stat_vars: params.stat_vars ? [...params.stat_vars].sort().join(',') : '',
    mask_vars: params.mask_vars ? [...params.mask_vars].sort().join(',') : '',
    enable_region_crop: params.enable_region_crop ?? null,
    crop_lon_range: params.crop_lon_range ? params.crop_lon_range.join(',') : '',
    crop_lat_range: params.crop_lat_range ? params.crop_lat_range.join(',') : '',
    crop_mode: params.crop_mode ?? null,
    lr_nc_folder: params.lr_nc_folder ?? null,
    scale: params.scale ?? null,
    downsample_method: params.downsample_method ?? null,
    train_ratio: params.train_ratio,
    valid_ratio: params.valid_ratio,
    test_ratio: params.test_ratio,
    h_slice: params.h_slice ?? null,
    w_slice: params.w_slice ?? null,
  }
  const dataStr = JSON.stringify(tokenData) + TOKEN_SALT
  return crypto.createHash('sha256').update(dataStr).digest('hex').substring(0, 16)
}

function validateConfirmationToken(params: SrPreprocessWorkflowParams): boolean {
  if (!params.confirmation_token) return false
  return params.confirmation_token === generateConfirmationToken(params)
}

// ============================================================
// Prompt builders
// ============================================================

function buildVariableSelectionPrompt(inspectResult: any, params: SrPreprocessWorkflowParams): SrPreprocessStagePromptResult {
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

【动态变量候选】（有时间维度，可作为研究目标）
${varLines}

【疑似静态/坐标变量】
${(inspectResult?.suspected_coordinates || []).map((v: string) => `  - ${v}`).join('\n') || '  无'}

【疑似掩码变量】
${(inspectResult?.suspected_masks || []).map((v: string) => `  - ${v}`).join('\n') || '  无'}

================================================================================

**请回答以下问题：**

1️⃣ **您要研究哪些变量？**
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

function buildStaticAndMaskSelectionPrompt(inspectResult: any, params: SrPreprocessWorkflowParams): SrPreprocessStagePromptResult {
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
   （如果不需要，请回复"不需要"或指定 stat_vars: []）

3️⃣ **使用哪些掩码变量？**
   可选: ${(inspectResult?.suspected_masks || []).join(', ') || '无'}
   （如果数据没有掩码，请回复"无掩码"或指定 mask_vars: []）

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

function buildRegionSelectionPrompt(inspectResult: any, params: SrPreprocessWorkflowParams): SrPreprocessStagePromptResult {
  const statistics = inspectResult?.statistics || {}

  let lonVarName: string | undefined
  let latVarName: string | undefined
  let dataLonMin: number | undefined
  let dataLonMax: number | undefined
  let dataLatMin: number | undefined
  let dataLatMax: number | undefined

  for (const [varName, stats] of Object.entries(statistics)) {
    const s = stats as any
    const lower = varName.toLowerCase()
    if (lower.includes('lon') || lower === 'x') {
      lonVarName = varName; dataLonMin = s.min; dataLonMax = s.max
    }
    if (lower.includes('lat') || lower === 'y') {
      latVarName = varName; dataLatMin = s.min; dataLatMax = s.max
    }
  }

  const firstVar = params.dyn_vars?.[0]
  const varInfo = inspectResult?.variables?.[firstVar]
  const dataShape = varInfo?.shape || []
  const H = typeof dataShape[dataShape.length - 2] === 'number' ? dataShape[dataShape.length - 2] : '?'
  const W = typeof dataShape[dataShape.length - 1] === 'number' ? dataShape[dataShape.length - 1] : '?'

  const lonRangeStr = (dataLonMin !== undefined && dataLonMax !== undefined)
    ? `[${dataLonMin.toFixed(4)}, ${dataLonMax.toFixed(4)}]`
    : '未知（请确认经度变量名）'
  const latRangeStr = (dataLatMin !== undefined && dataLatMax !== undefined)
    ? `[${dataLatMin.toFixed(4)}, ${dataLatMax.toFixed(4)}]`
    : '未知（请确认纬度变量名）'

  // 验证已提供的裁剪范围
  let rangeValidationMsg = ''
  if (params.enable_region_crop === true && params.crop_lon_range && params.crop_lat_range) {
    const [uLonMin, uLonMax] = params.crop_lon_range
    const [uLatMin, uLatMax] = params.crop_lat_range
    const errors: string[] = []
    if (dataLonMin !== undefined && dataLonMax !== undefined) {
      if (uLonMin < dataLonMin || uLonMax > dataLonMax)
        errors.push(`  ❌ 经度越界: 指定 [${uLonMin}, ${uLonMax}]，数据范围 [${dataLonMin.toFixed(4)}, ${dataLonMax.toFixed(4)}]`)
      if (uLonMin >= uLonMax)
        errors.push(`  ❌ 经度无效: 最小值 ${uLonMin} 必须小于最大值 ${uLonMax}`)
    }
    if (dataLatMin !== undefined && dataLatMax !== undefined) {
      if (uLatMin < dataLatMin || uLatMax > dataLatMax)
        errors.push(`  ❌ 纬度越界: 指定 [${uLatMin}, ${uLatMax}]，数据范围 [${dataLatMin.toFixed(4)}, ${dataLatMax.toFixed(4)}]`)
      if (uLatMin >= uLatMax)
        errors.push(`  ❌ 纬度无效: 最小值 ${uLatMin} 必须小于最大值 ${uLatMax}`)
    }
    if (errors.length > 0) {
      rangeValidationMsg = `\n================================================================================
                         ⚠️ 裁剪范围验证失败
================================================================================\n\n${errors.join('\n')}\n\n请重新指定有效的裁剪范围。\n`
    }
  }

  const alreadyEnabledCrop = params.enable_region_crop === true

  return {
    status: 'awaiting_region_selection',
    message: `变量选择已确认：
- 研究变量: ${params.dyn_vars?.join(', ')}
- 静态变量: ${params.stat_vars?.length ? params.stat_vars.join(', ') : '无'}
- 掩码变量: ${params.mask_vars?.length ? params.mask_vars.join(', ') : '无'}
${rangeValidationMsg}
================================================================================
                    ⚠️ ${alreadyEnabledCrop ? '请确认区域裁剪参数' : '是否需要区域裁剪？'}
================================================================================

【数据空间范围】
- 经度变量: ${lonVarName || '未检测到'}
- 纬度变量: ${latVarName || '未检测到'}
- 经度范围: ${lonRangeStr}
- 纬度范围: ${latRangeStr}
- 空间尺寸: ${H} × ${W}

================================================================================

**请回答以下问题：**

${alreadyEnabledCrop ? '' : `🔹 **是否需要先裁剪到特定区域？**
   - 如果需要，请回复"需要裁剪"或"是"，并提供经纬度范围
   - 如果不需要，请回复"不需要裁剪"或"否"

`}🗺️ **裁剪区域（如果需要裁剪）：**
   - crop_lon_range: [经度最小值, 经度最大值]，如 [100, 120]
   - crop_lat_range: [纬度最小值, 纬度最大值]，如 [20, 40]
   - 注意: 裁剪范围必须在数据范围内

📐 **裁剪模式：**
   - "one_step": 一步到位，直接计算能被 scale 整除的区域（不保存 raw）
   - "two_step": 两步裁剪，先保存到 raw/，再裁剪到 hr/（默认，推荐）

================================================================================

⚠️ Agent 注意：
- 如果用户说"不需要裁剪"，设置 enable_region_crop: false
- 如果用户说"需要裁剪"并提供了范围，设置 enable_region_crop: true 和对应的范围
- **禁止自动决定是否裁剪或裁剪范围！**`,
    canExecute: false,
    data: {
      dyn_vars_confirmed: params.dyn_vars,
      stat_vars_confirmed: params.stat_vars,
      mask_vars_confirmed: params.mask_vars,
      lon_var_name: lonVarName,
      lat_var_name: latVarName,
      data_lon_range: dataLonMin !== undefined ? [dataLonMin, dataLonMax] : null,
      data_lat_range: dataLatMin !== undefined ? [dataLatMin, dataLatMax] : null,
      data_shape: { H, W }
    }
  }
}

function buildProcessingParametersPrompt(inspectResult: any, params: SrPreprocessWorkflowParams): SrPreprocessStagePromptResult {
  const isNumericalModelMode = !!params.lr_nc_folder
  const firstVar = params.dyn_vars?.[0]
  const varInfo = inspectResult?.variables?.[firstVar]
  const dataShape = varInfo?.shape || []
  const H = typeof dataShape[dataShape.length - 2] === 'number' ? dataShape[dataShape.length - 2] : 0
  const W = typeof dataShape[dataShape.length - 1] === 'number' ? dataShape[dataShape.length - 1] : 0

  let cropRecommendation = ''
  const scale = params.scale
  if (scale && scale > 1 && H > 0 && W > 0) {
    const hRemainder = H % scale
    const wRemainder = W % scale
    if (hRemainder !== 0 || wRemainder !== 0) {
      const recommendedH = Math.floor(H / scale) * scale
      const recommendedW = Math.floor(W / scale) * scale
      cropRecommendation = `
   ⚠️ **当前尺寸 ${H}×${W} 不能被 ${scale} 整除！**
   - H 余数: ${hRemainder} (${H} % ${scale} = ${hRemainder})
   - W 余数: ${wRemainder} (${W} % ${scale} = ${wRemainder})

   **建议裁剪参数：**
   - h_slice: "0:${recommendedH}" (裁剪后 H=${recommendedH})
   - w_slice: "0:${recommendedW}" (裁剪后 W=${recommendedW})`
    } else {
      cropRecommendation = `\n   ✅ 当前尺寸 ${H}×${W} 可以被 ${scale} 整除，无需裁剪`
    }
  }

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
- 空间尺寸: H=${H || '?'}, W=${W || '?'}
- 文件数量: ${inspectResult?.file_count || '?'} 个

================================================================================

**请回答以下问题：**

4️⃣ **超分数据来源方式？**
   - **下采样模式**：从 HR 数据下采样生成 LR 数据
   - **粗网格模式**：HR 和 LR 数据来自不同精度的数值模型

${!isNumericalModelMode ? `5️⃣ **下采样参数？**（下采样模式必须）
   - scale: 下采样倍数（如 4 表示缩小到 1/4）
   - downsample_method: 插值方法
     • area（推荐）：区域平均，最接近真实低分辨率
     • cubic：三次插值，较平滑
     • linear：双线性插值
     • nearest：最近邻插值，保留原始值
     • lanczos：Lanczos 插值，高质量
` : ''}6️⃣ **数据集划分比例？**（三者之和必须为 1.0）
   - train_ratio: 训练集比例（如 0.7）
   - valid_ratio: 验证集比例（如 0.15）
   - test_ratio: 测试集比例（如 0.15）

7️⃣ **数据裁剪？**【必须确认】
   - 当前尺寸: ${H || '?'} × ${W || '?'}
${cropRecommendation || '   - 请指定 h_slice 和 w_slice，或回复"不裁剪"'}

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
  params: SrPreprocessWorkflowParams,
  token: string
): SrPreprocessStagePromptResult {
  const isNumericalModelMode = !!params.lr_nc_folder
  const firstVar = params.dyn_vars?.[0]
  const varInfo = inspectResult?.variables?.[firstVar]
  const dataShape = varInfo?.shape || []
  const originalH = dataShape.length >= 2 ? dataShape[dataShape.length - 2] : '?'
  const originalW = dataShape.length >= 1 ? dataShape[dataShape.length - 1] : '?'

  let finalH: number | string = originalH
  let finalW: number | string = originalW
  if (params.h_slice && typeof originalH === 'number') {
    const parts = params.h_slice.split(':').map(Number)
    finalH = parts[1] - parts[0]
  }
  if (params.w_slice && typeof originalW === 'number') {
    const parts = params.w_slice.split(':').map(Number)
    finalW = parts[1] - parts[0]
  }

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

【区域裁剪】
${params.enable_region_crop
  ? `- 启用区域裁剪: 是
- 经度范围: [${params.crop_lon_range?.[0]}, ${params.crop_lon_range?.[1]}]
- 纬度范围: [${params.crop_lat_range?.[0]}, ${params.crop_lat_range?.[1]}]
- 裁剪模式: ${params.crop_mode === 'one_step' ? '一步到位（不保存 raw）' : '两步裁剪（保存 raw）'}`
  : '- 启用区域裁剪: 否'}

【处理参数】
- 模式: ${isNumericalModelMode ? '粗网格模式（数值模型）' : '下采样模式'}
${!isNumericalModelMode
  ? `- 下采样倍数: ${params.scale}x\n- 插值方法: ${params.downsample_method}`
  : `- LR 数据目录: ${params.lr_nc_folder}`}

【数据裁剪】
- 原始尺寸: ${originalH} × ${originalW}
${params.h_slice || params.w_slice
  ? `- 裁剪后尺寸: ${finalH} × ${finalW}
- H 裁剪: ${params.h_slice || '不裁剪'}
- W 裁剪: ${params.w_slice || '不裁剪'}`
  : '- 不裁剪'}

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
        enable_region_crop: params.enable_region_crop,
        crop_lon_range: params.crop_lon_range,
        crop_lat_range: params.crop_lat_range,
        crop_mode: params.crop_mode,
        scale: params.scale,
        downsample_method: params.downsample_method,
        train_ratio: params.train_ratio,
        valid_ratio: params.valid_ratio,
        test_ratio: params.test_ratio,
        h_slice: params.h_slice,
        w_slice: params.w_slice
      }
    }
  }
}

function buildTokenInvalidPrompt(
  params: SrPreprocessWorkflowParams,
  mode: 'missing_token' | 'token_mismatch'
): SrPreprocessStagePromptResult {
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
1. Token 生成后参数被修改（如变量、区域裁剪、比例、下采样参数等）
2. Token 被错误地复制或截断

【当前参数快照】
- nc_folder: ${params.nc_folder}
- output_base: ${params.output_base}
- dyn_vars: ${params.dyn_vars?.join(', ')}
- stat_vars: ${params.stat_vars?.join(', ') ?? '未设置'}
- mask_vars: ${params.mask_vars?.join(', ') ?? '未设置'}
- enable_region_crop: ${params.enable_region_crop}
- crop_lon_range: ${params.crop_lon_range ? `[${params.crop_lon_range.join(', ')}]` : '无'}
- crop_lat_range: ${params.crop_lat_range ? `[${params.crop_lat_range.join(', ')}]` : '无'}
- crop_mode: ${params.crop_mode ?? '无'}
- lr_nc_folder: ${params.lr_nc_folder ?? '无'}
- scale: ${params.scale ?? '无'}
- downsample_method: ${params.downsample_method ?? '无'}
- train_ratio: ${params.train_ratio}
- valid_ratio: ${params.valid_ratio}
- test_ratio: ${params.test_ratio}
- h_slice: ${params.h_slice ?? '无'}
- w_slice: ${params.w_slice ?? '无'}

【解决方法】请重新调用工具（不带 user_confirmed），获取新的 confirmation_token。`,
    canExecute: false,
    data: {
      error_type: 'token_invalid',
      provided_token: params.confirmation_token,
      expected_token: generateConfirmationToken(params)
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
 * @returns SrStageResult（仍在某阶段）或 null（所有阶段通过，继续执行）
 */
export function resolveStage(
  params: SrPreprocessWorkflowParams,
  inspectResult: any,
): SrPreprocessStagePromptResult | null {
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
  if (!hasRegionCropDecisionConfigured(params)) return buildRegionSelectionPrompt(inspectResult, params)
  if (!hasProcessingParamsConfigured(params)) return buildProcessingParametersPrompt(inspectResult, params)

  // Stage 4：所有参数就绪，生成 SHA-256 token 并等待用户确认
  const token = generateConfirmationToken(params)
  return buildExecutionConfirmationPrompt(inspectResult, params, token)
}
