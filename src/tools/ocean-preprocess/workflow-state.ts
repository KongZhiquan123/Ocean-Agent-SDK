/**
 * @file workflow-state.ts
 * @description 预处理工作流状态机 - 实现分阶段强制确认逻辑
 *              核心思想：根据已有参数倒推当前阶段，防止跳步
 *
 * @author kongzhiquan
 * @date 2026-02-05
 * @version 3.2.0
 *
 * @changelog
 *   - 2026-02-05 kongzhiquan: v3.2.0 新增阶段2.5区域裁剪
 *     - 新增 AWAITING_REGION_SELECTION 状态
 *     - WorkflowParams 添加区域裁剪参数（enable_region_crop, crop_lon_range, crop_lat_range, crop_mode）
 *     - determineCurrentState() 插入阶段2.5判断逻辑
 *     - Token 生成包含区域裁剪参数
 *     - 新增 buildRegionSelectionPrompt() 提示构建
 *   - 2026-02-05 kongzhiquan: v3.0.3 新增执行确认 Token 机制
 *     - 防止 Agent 跳过 awaiting_execution 阶段直接执行
 *     - 在 awaiting_execution 阶段生成 confirmation_token
 *     - user_confirmed=true 时必须携带正确的 token 才能执行
 *   - 2026-02-05 kongzhiquan: v3.0.2 初始版本
 *     - 从 full.ts 抽取状态机逻辑
 *     - 状态倒推防跳步机制
 */

import * as crypto from 'crypto'

/**
 * 工作流状态常量
 * 按顺序定义，每个阶段必须完成才能进入下一阶段
 */
export const WorkflowState = {
  /** 阶段1: 等待用户选择研究变量 */
  AWAITING_VARIABLE_SELECTION: 'awaiting_variable_selection',
  /** 阶段2: 等待用户选择静态/掩码变量 */
  AWAITING_STATIC_SELECTION: 'awaiting_static_selection',
  /** 阶段2.5: 等待用户确认区域裁剪 */
  AWAITING_REGION_SELECTION: 'awaiting_region_selection',
  /** 阶段3: 等待用户确认处理参数 */
  AWAITING_PARAMETERS: 'awaiting_parameters',
  /** 阶段4: 等待用户最终确认执行 */
  AWAITING_EXECUTION: 'awaiting_execution',
  /** 阶段5: 执行完成 */
  PASS: 'pass',
  /** 错误状态 */
  ERROR: 'error',
  /** Token 验证失败 - Agent 试图跳过确认阶段 */
  TOKEN_INVALID: 'token_invalid'
} as const

export type WorkflowStateType = typeof WorkflowState[keyof typeof WorkflowState]

/**
 * 工作流参数接口
 */
export interface WorkflowParams {
  // 基础参数
  nc_folder: string
  output_base: string

  // 阶段1: 研究变量
  dyn_vars?: string[]

  // 阶段2: 静态/掩码变量
  stat_vars?: string[]
  mask_vars?: string[]

  // 阶段2.5: 区域裁剪（新增）
  /** undefined=未回答, true=启用裁剪, false=不启用裁剪 */
  enable_region_crop?: boolean
  /** 经度裁剪范围 [min, max] */
  crop_lon_range?: [number, number]
  /** 纬度裁剪范围 [min, max] */
  crop_lat_range?: [number, number]
  /** 裁剪模式: one_step=一步到位, two_step=两步裁剪(保存raw) */
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
  /** 执行确认 Token - 必须从 awaiting_execution 阶段获取 */
  confirmation_token?: string

  // 粗网格模式
  lr_nc_folder?: string

  // 其他参数
  [key: string]: any
}

/**
 * 阶段检查结果
 */
export interface StageCheckResult {
  /** 当前所处阶段 */
  currentState: WorkflowStateType
  /** 缺失的参数 */
  missingParams: string[]
  /** 是否可以继续执行 */
  canProceed: boolean
  /** 阶段描述 */
  stageDescription: string
  /** Token 验证失败的错误信息 */
  tokenError?: string
}

/**
 * 预处理工作流状态机
 *
 * 核心逻辑：根据已有参数倒推当前阶段，严格防止跳步
 * - 即使用户传了后续阶段的参数，如果前置阶段未完成，也会被忽略
 * - 每个阶段必须显式确认才能进入下一阶段
 * - user_confirmed=true 时必须携带正确的 confirmation_token
 */
export class PreprocessWorkflow {
  private params: WorkflowParams
  private isNumericalModelMode: boolean

  /** Token 生成的盐值（或者可以配置为环境变量） */
  private static readonly TOKEN_SALT = 'ocean-preprocess-v3'

  constructor(params: WorkflowParams) {
    this.params = params
    this.isNumericalModelMode = !!params.lr_nc_folder
  }

  /**
   * 生成执行确认 Token
   * 基于关键参数生成 hash，确保参数未被篡改
   */
  generateConfirmationToken(): string {
    const { params } = this
    const tokenData = {
      nc_folder: params.nc_folder,
      output_base: params.output_base,
      dyn_vars: params.dyn_vars?.sort().join(','),
      stat_vars: params.stat_vars?.sort().join(','),
      mask_vars: params.mask_vars?.sort().join(','),
      // 区域裁剪参数（新增）
      enable_region_crop: params.enable_region_crop,
      crop_lon_range: params.crop_lon_range?.join(','),
      crop_lat_range: params.crop_lat_range?.join(','),
      crop_mode: params.crop_mode,
      // 处理参数
      scale: params.scale,
      downsample_method: params.downsample_method,
      train_ratio: params.train_ratio,
      valid_ratio: params.valid_ratio,
      test_ratio: params.test_ratio,
      h_slice: params.h_slice,
      w_slice: params.w_slice,
      lr_nc_folder: params.lr_nc_folder
    }

    const dataStr = JSON.stringify(tokenData) + PreprocessWorkflow.TOKEN_SALT
    return crypto.createHash('sha256').update(dataStr).digest('hex').substring(0, 16)
  }

  /**
   * 验证执行确认 Token
   */
  validateConfirmationToken(): boolean {
    if (!this.params.confirmation_token) {
      return false
    }
    const expectedToken = this.generateConfirmationToken()
    return this.params.confirmation_token === expectedToken
  }

  /**
   * 核心方法：根据参数倒推当前阶段
   *
   * 这是"防止跳步"的硬逻辑：
   * - 从最后阶段往前检查
   * - 只有满足所有前置条件才能进入该阶段
   * - user_confirmed=true 时必须验证 token
   */
  determineCurrentState(): StageCheckResult {
    const { params } = this

    // ========== 阶段5: 执行完成 ==========
    // 只有 user_confirmed 为 true，且所有前置参数都有，且 token 正确，才算完成
    if (params.user_confirmed === true) {
      // 验证所有前置条件
      if (this.hasAllRequiredParams()) {
        // 🔐 关键：验证 confirmation_token
        if (!params.confirmation_token) {
          return {
            currentState: WorkflowState.TOKEN_INVALID,
            missingParams: ['confirmation_token'],
            canProceed: false,
            stageDescription: 'Token 缺失',
            tokenError: `⚠️ 检测到跳步行为！

您设置了 user_confirmed=true，但未提供 confirmation_token。

这表明您可能试图跳过 awaiting_execution 阶段直接执行。
为了确保用户已经看到并确认了所有参数，必须：

1. 先调用工具（不带 user_confirmed），进入 awaiting_execution 阶段
2. 从返回结果中获取 confirmation_token
3. 用户确认后，再次调用并携带 user_confirmed=true 和 confirmation_token

【安全提示】
confirmation_token 是基于所有参数生成的签名，用于：
- 确保用户看到了完整的参数汇总
- 防止 Agent 自动跳过确认步骤
- 防止参数在确认后被篡改`
          }
        }

        if (!this.validateConfirmationToken()) {
          return {
            currentState: WorkflowState.TOKEN_INVALID,
            missingParams: [],
            canProceed: false,
            stageDescription: 'Token 验证失败',
            tokenError: `⚠️ Token 验证失败！

提供的 confirmation_token 与当前参数不匹配。

可能的原因：
1. Token 是从之前的调用中获取的，但参数已经修改
2. Token 被错误地复制或截断
3. 试图使用伪造的 token

【解决方法】
请重新调用工具（不带 user_confirmed），获取新的 confirmation_token，
然后让用户确认参数后再执行。

【当前 Token】: ${params.confirmation_token}
【期望 Token】: ${this.generateConfirmationToken()}`
          }
        }

        // Token 验证通过
        return {
          currentState: WorkflowState.PASS,
          missingParams: [],
          canProceed: true,
          stageDescription: '所有参数已确认，Token 验证通过，可以执行'
        }
      }
      // 如果 user_confirmed=true 但缺少参数，说明有问题
      // 回退到缺失参数的阶段（下面的逻辑会处理）
    }

    // ========== 阶段4: 等待执行确认 ==========
    // 前提：必须有 dyn_vars + stat_vars + mask_vars + 区域裁剪决定 + 处理参数
    if (this.hasVariableParams() && this.hasRegionCropDecision() && this.hasProcessingParams()) {
      return {
        currentState: WorkflowState.AWAITING_EXECUTION,
        missingParams: ['user_confirmed', 'confirmation_token'],
        canProceed: false,
        stageDescription: '所有参数就绪，等待用户最终确认执行'
      }
    }

    // ========== 阶段3: 等待处理参数 ==========
    // 前提：必须有 dyn_vars + stat_vars + mask_vars + 区域裁剪决定
    if (this.hasVariableParams() && this.hasRegionCropDecision()) {
      const missingProcessingParams = this.getMissingProcessingParams()
      return {
        currentState: WorkflowState.AWAITING_PARAMETERS,
        missingParams: missingProcessingParams,
        canProceed: false,
        stageDescription: '变量和区域裁剪已确认，等待处理参数'
      }
    }

    // ========== 阶段2.5: 等待区域裁剪确认 ==========
    // 前提：必须有 dyn_vars + stat_vars + mask_vars，但 enable_region_crop 未明确
    if (this.hasVariableParams() && !this.hasRegionCropDecision()) {
      return {
        currentState: WorkflowState.AWAITING_REGION_SELECTION,
        missingParams: ['enable_region_crop'],
        canProceed: false,
        stageDescription: '变量已确认，等待区域裁剪决定'
      }
    }

    // ========== 阶段2: 等待静态/掩码变量 ==========
    // 前提：必须有 dyn_vars
    if (params.dyn_vars && params.dyn_vars.length > 0) {
      const missingVarParams: string[] = []
      if (params.stat_vars === undefined) missingVarParams.push('stat_vars')
      if (params.mask_vars === undefined) missingVarParams.push('mask_vars')

      return {
        currentState: WorkflowState.AWAITING_STATIC_SELECTION,
        missingParams: missingVarParams,
        canProceed: false,
        stageDescription: '研究变量已确认，等待静态/掩码变量选择'
      }
    }

    // ========== 阶段1: 等待研究变量 ==========
    // 默认状态
    return {
      currentState: WorkflowState.AWAITING_VARIABLE_SELECTION,
      missingParams: ['dyn_vars'],
      canProceed: false,
      stageDescription: '等待用户选择研究变量'
    }
  }

  /**
   * 检查是否有所有变量参数（阶段1+2完成）
   */
  private hasVariableParams(): boolean {
    const { params } = this
    return !!(
      params.dyn_vars && params.dyn_vars.length > 0 &&
      params.stat_vars !== undefined &&  // 允许空数组 []
      params.mask_vars !== undefined      // 允许空数组 []
    )
  }

  /**
   * 检查是否已决定区域裁剪（阶段2.5完成）
   * enable_region_crop 必须有明确值（true 或 false）
   */
  private hasRegionCropDecision(): boolean {
    const { enable_region_crop, crop_lon_range, crop_lat_range } = this.params

    // 明确设置为 false = 不需要裁剪，可以跳过
    if (enable_region_crop === false) return true

    // 明确设置为 true，必须有有效的裁剪范围
    if (enable_region_crop === true) {
      return !!(
        crop_lon_range && crop_lon_range.length === 2 &&
        crop_lat_range && crop_lat_range.length === 2
      )
    }

    // undefined = 还未回答，需要询问
    return false
  }

  /**
   * 检查是否有所有处理参数（阶段3完成）
   */
  private hasProcessingParams(): boolean {
    const { params } = this

    // 数据集划分比例必须有
    const hasSplitRatios = (
      params.train_ratio !== undefined &&
      params.valid_ratio !== undefined &&
      params.test_ratio !== undefined
    )

    if (!hasSplitRatios) return false

    // 下采样参数（非粗网格模式下必须）
    if (!this.isNumericalModelMode) {
      return !!(params.scale && params.scale > 1 && params.downsample_method)
    }

    return true
  }

  /**
   * 获取缺失的处理参数
   */
  private getMissingProcessingParams(): string[] {
    const { params } = this
    const missing: string[] = []

    if (params.train_ratio === undefined) missing.push('train_ratio')
    if (params.valid_ratio === undefined) missing.push('valid_ratio')
    if (params.test_ratio === undefined) missing.push('test_ratio')

    if (!this.isNumericalModelMode) {
      if (!params.scale || params.scale <= 1) missing.push('scale')
      if (!params.downsample_method) missing.push('downsample_method')
    }

    return missing
  }

  /**
   * 检查是否有所有必需参数（可以执行）
   */
  private hasAllRequiredParams(): boolean {
    return this.hasVariableParams() && this.hasRegionCropDecision() && this.hasProcessingParams()
  }

  /**
   * 获取当前阶段的用户提示信息
   */
  getStagePrompt(inspectResult?: any): StagePromptResult {
    const stateCheck = this.determineCurrentState()

    switch (stateCheck.currentState) {
      case WorkflowState.AWAITING_VARIABLE_SELECTION:
        return this.buildVariableSelectionPrompt(inspectResult)

      case WorkflowState.AWAITING_STATIC_SELECTION:
        return this.buildStaticSelectionPrompt(inspectResult)

      case WorkflowState.AWAITING_REGION_SELECTION:
        return this.buildRegionSelectionPrompt(inspectResult)

      case WorkflowState.AWAITING_PARAMETERS:
        return this.buildParametersPrompt(inspectResult)

      case WorkflowState.AWAITING_EXECUTION:
        return this.buildExecutionPrompt(inspectResult)

      case WorkflowState.TOKEN_INVALID:
        return {
          status: WorkflowState.TOKEN_INVALID,
          message: stateCheck.tokenError || 'Token 验证失败',
          canExecute: false,
          data: {
            error_type: 'token_invalid',
            expected_token: this.generateConfirmationToken(),
            provided_token: this.params.confirmation_token
          }
        }

      case WorkflowState.PASS:
        return {
          status: WorkflowState.PASS,
          message: '所有参数已确认，Token 验证通过，开始执行预处理流程...',
          canExecute: true
        }

      default:
        return {
          status: WorkflowState.ERROR,
          message: '未知状态',
          canExecute: false
        }
    }
  }

  /**
   * 构建阶段1提示：研究变量选择
   */
  private buildVariableSelectionPrompt(inspectResult?: any): StagePromptResult {
    const dynCandidates = inspectResult?.dynamic_vars_candidates || []
    const variables = inspectResult?.variables || {}

    // 格式化变量信息
    const formatVarInfo = (vars: Record<string, any>, candidates: string[]) => {
      const lines: string[] = []
      for (const name of candidates) {
        const info = vars[name]
        if (info) {
          const dims = info.dims?.join(',') || '?'
          const shape = info.shape?.join('×') || '?'
          const dtype = info.dtype || '?'
          lines.push(`  - ${name}: 形状 (${shape}), 维度 [${dims}], ${dtype}`)
        }
      }
      return lines.join('\n') || '  无'
    }

    return {
      status: WorkflowState.AWAITING_VARIABLE_SELECTION,
      message: `数据分析完成！

================================================================================
                         ⚠️ 请选择研究变量（必须）
================================================================================

【数据概况】
- 数据目录: ${this.params.nc_folder}
- 文件数量: ${inspectResult?.file_count || '?'} 个

【动态变量候选】（有时间维度，可作为研究目标）
${formatVarInfo(variables, dynCandidates)}

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

  /**
   * 构建阶段2提示：静态/掩码变量选择
   */
  private buildStaticSelectionPrompt(inspectResult?: any): StagePromptResult {
    return {
      status: WorkflowState.AWAITING_STATIC_SELECTION,
      message: `研究变量已确认：${this.params.dyn_vars?.join(', ')}

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
        dyn_vars_confirmed: this.params.dyn_vars,
        suspected_coordinates: inspectResult?.suspected_coordinates,
        suspected_masks: inspectResult?.suspected_masks
      }
    }
  }

  /**
   * 构建阶段2.5提示：区域裁剪确认
   */
  private buildRegionSelectionPrompt(inspectResult?: any): StagePromptResult {
    const { params } = this

    // 从 inspectResult 获取数据的经纬度范围
    const statistics = inspectResult?.statistics || {}

    // 查找经纬度变量及其范围
    let lonVarName: string | undefined
    let latVarName: string | undefined
    let dataLonMin: number | undefined
    let dataLonMax: number | undefined
    let dataLatMin: number | undefined
    let dataLatMax: number | undefined

    // 从 statistics 中查找经纬度信息
    for (const [varName, stats] of Object.entries(statistics)) {
      const s = stats as any
      const lowerName = varName.toLowerCase()
      if (lowerName.includes('lon') || lowerName === 'x') {
        lonVarName = varName
        dataLonMin = s.min
        dataLonMax = s.max
      }
      if (lowerName.includes('lat') || lowerName === 'y') {
        latVarName = varName
        dataLatMin = s.min
        dataLatMax = s.max
      }
    }

    // 获取数据形状
    const firstVar = params.dyn_vars?.[0]
    const varInfo = inspectResult?.variables?.[firstVar]
    const dataShape = varInfo?.shape || []
    const H = typeof dataShape[dataShape.length - 2] === 'number' ? dataShape[dataShape.length - 2] : '?'
    const W = typeof dataShape[dataShape.length - 1] === 'number' ? dataShape[dataShape.length - 1] : '?'

    // 格式化经纬度范围显示
    const lonRangeStr = (dataLonMin !== undefined && dataLonMax !== undefined)
      ? `[${dataLonMin.toFixed(4)}, ${dataLonMax.toFixed(4)}]`
      : '未知（请确认经度变量名是否正确）'
    const latRangeStr = (dataLatMin !== undefined && dataLatMax !== undefined)
      ? `[${dataLatMin.toFixed(4)}, ${dataLatMax.toFixed(4)}]`
      : '未知（请确认纬度变量名是否正确）'

    // 如果用户已提供裁剪范围，验证是否在数据边界内
    let rangeValidationMsg = ''
    if (params.enable_region_crop === true && params.crop_lon_range && params.crop_lat_range) {
      const [userLonMin, userLonMax] = params.crop_lon_range
      const [userLatMin, userLatMax] = params.crop_lat_range
      const errors: string[] = []

      if (dataLonMin !== undefined && dataLonMax !== undefined) {
        if (userLonMin < dataLonMin || userLonMax > dataLonMax) {
          errors.push(`  ❌ 经度范围越界: 您指定 [${userLonMin}, ${userLonMax}]，但数据范围是 [${dataLonMin.toFixed(4)}, ${dataLonMax.toFixed(4)}]`)
        }
        if (userLonMin >= userLonMax) {
          errors.push(`  ❌ 经度范围无效: 最小值 ${userLonMin} 必须小于最大值 ${userLonMax}`)
        }
      }
      if (dataLatMin !== undefined && dataLatMax !== undefined) {
        if (userLatMin < dataLatMin || userLatMax > dataLatMax) {
          errors.push(`  ❌ 纬度范围越界: 您指定 [${userLatMin}, ${userLatMax}]，但数据范围是 [${dataLatMin.toFixed(4)}, ${dataLatMax.toFixed(4)}]`)
        }
        if (userLatMin >= userLatMax) {
          errors.push(`  ❌ 纬度范围无效: 最小值 ${userLatMin} 必须小于最大值 ${userLatMax}`)
        }
      }

      if (errors.length > 0) {
        rangeValidationMsg = `
================================================================================
                         ⚠️ 裁剪范围验证失败
================================================================================

${errors.join('\n')}

请重新指定有效的裁剪范围。

`
      }
    }

    // 根据用户是否已表态，显示不同的提示
    const alreadyEnabledCrop = params.enable_region_crop === true

    return {
      status: WorkflowState.AWAITING_REGION_SELECTION,
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
- 如果用户不清楚范围，请告知上面显示的数据经纬度范围
- 如果用户指定的范围超出数据边界，请告知有效范围并请求重新输入
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

  /**
   * 构建阶段3提示：处理参数确认
   */
  private buildParametersPrompt(inspectResult?: any): StagePromptResult {
    // 计算数据形状
    const firstVar = this.params.dyn_vars?.[0]
    const varInfo = inspectResult?.variables?.[firstVar]
    const dataShape = varInfo?.shape || []
    const H = typeof dataShape[dataShape.length - 2] === 'number' ? dataShape[dataShape.length - 2] : 0
    const W = typeof dataShape[dataShape.length - 1] === 'number' ? dataShape[dataShape.length - 1] : 0

    // 计算推荐裁剪值
    let cropRecommendation = ''
    const scale = this.params.scale
    if (scale && scale > 1 && H > 0 && W > 0) {
      const hRemainder = H % scale
      const wRemainder = W % scale
      const needsCrop = hRemainder !== 0 || wRemainder !== 0

      if (needsCrop) {
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
        cropRecommendation = `
   ✅ 当前尺寸 ${H}×${W} 可以被 ${scale} 整除，无需裁剪`
      }
    }

    return {
      status: WorkflowState.AWAITING_PARAMETERS,
      message: `变量选择已确认：
- 研究变量: ${this.params.dyn_vars?.join(', ')}
- 静态变量: ${this.params.stat_vars?.length ? this.params.stat_vars.join(', ') : '无'}
- 掩码变量: ${this.params.mask_vars?.length ? this.params.mask_vars.join(', ') : '无'}

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

${!this.isNumericalModelMode ? `5️⃣ **下采样参数？**（下采样模式必须）
   - scale: 下采样倍数（如 4 表示缩小到 1/4）
   - downsample_method: 插值方法
     • area（推荐）：区域平均，最接近真实低分辨率
     • cubic：三次插值，较平滑
     • linear：双线性插值
     • nearest：最近邻插值，保留原始值
     • lanczos：Lanczos 插值，高质量
` : ''}
6️⃣ **数据集划分比例？**（三者之和必须为 1.0）
   - train_ratio: 训练集比例（如 0.7）
   - valid_ratio: 验证集比例（如 0.15）
   - test_ratio: 测试集比例（如 0.15）

7️⃣ **数据裁剪？**【必须确认】
   - 当前尺寸: ${H || '?'} × ${W || '?'}
${cropRecommendation || `   - 请指定 h_slice 和 w_slice，或回复"不裁剪"`}

================================================================================

⚠️ Agent 注意：**禁止自动决定处理参数！**
必须等待用户明确指定后，再传入相应参数重新调用。`,
      canExecute: false,
      data: {
        dyn_vars_confirmed: this.params.dyn_vars,
        stat_vars_confirmed: this.params.stat_vars,
        mask_vars_confirmed: this.params.mask_vars,
        data_shape: { H, W },
        file_count: inspectResult?.file_count
      }
    }
  }

  /**
   * 构建阶段4提示：执行前最终确认
   * 生成 confirmation_token 供下次调用验证
   */
  private buildExecutionPrompt(inspectResult?: any): StagePromptResult {
    const { params } = this

    // 生成确认 Token
    const confirmationToken = this.generateConfirmationToken()

    // 计算裁剪后的尺寸
    const firstVar = params.dyn_vars?.[0]
    const varInfo = inspectResult?.variables?.[firstVar]
    const dataShape = varInfo?.shape || []
    const originalH = dataShape.length >= 2 ? dataShape[dataShape.length - 2] : '?'
    const originalW = dataShape.length >= 1 ? dataShape[dataShape.length - 1] : '?'

    // 解析裁剪后尺寸
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
      status: WorkflowState.AWAITING_EXECUTION,
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
${params.enable_region_crop ? `- 启用区域裁剪: 是
- 经度范围: [${params.crop_lon_range?.[0]}, ${params.crop_lon_range?.[1]}]
- 纬度范围: [${params.crop_lat_range?.[0]}, ${params.crop_lat_range?.[1]}]
- 裁剪模式: ${params.crop_mode === 'one_step' ? '一步到位（不保存 raw）' : '两步裁剪（保存 raw）'}` : '- 启用区域裁剪: 否'}

【处理参数】
- 模式: ${this.isNumericalModelMode ? '粗网格模式（数值模型）' : '下采样模式'}
${!this.isNumericalModelMode ? `- 下采样倍数: ${params.scale}x
- 插值方法: ${params.downsample_method}` : `- LR 数据目录: ${params.lr_nc_folder}`}

【数据裁剪】
- 原始尺寸: ${originalH} × ${originalW}
${params.h_slice || params.w_slice ? `- 裁剪后尺寸: ${finalH} × ${finalW}
- H 裁剪: ${params.h_slice || '不裁剪'}
- W 裁剪: ${params.w_slice || '不裁剪'}` : '- 不裁剪'}

【数据集划分】
- 训练集: ${((params.train_ratio || 0) * 100).toFixed(0)}%
- 验证集: ${((params.valid_ratio || 0) * 100).toFixed(0)}%
- 测试集: ${((params.test_ratio || 0) * 100).toFixed(0)}%

================================================================================

⚠️ **请确认以上参数无误后，回复"确认执行"**

如需修改任何参数，请直接告诉我要修改的内容。

================================================================================

🔐 **执行确认 Token**: ${confirmationToken}
（Agent 必须将上面一段话发送给用户等待确认，同时必须在下次调用时携带此 token 和 user_confirmed=true）`,
      canExecute: false,
      data: {
        confirmation_token: confirmationToken,
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
}

/**
 * 阶段提示结果
 */
export interface StagePromptResult {
  status: WorkflowStateType
  message: string
  canExecute: boolean
  data?: any
}
