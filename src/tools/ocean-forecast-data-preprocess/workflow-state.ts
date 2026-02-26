/**
 * @file workflow-state.ts
 * @description 预测数据预处理工作流状态机 - 4 阶段强制确认逻辑
 *              基于 ocean-preprocess/workflow-state.ts 简化而来
 *              移除超分辨率专用参数（scale、downsample_method、lr_nc_folder 等）
 *              移除区域裁剪阶段，简化为 4 阶段流程
 *
 * @author Leizheng
 * @date 2026-02-25
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-25 Leizheng: v1.0.0 初始版本
 *     - 4 阶段：变量选择 → 静态/掩码变量 → 处理参数 → 执行确认
 *     - 移除超分专用参数（scale、downsample_method、lr_nc_folder 等）
 *     - 移除区域裁剪阶段（AWAITING_REGION_SELECTION）
 *     - Token 盐值: 'ocean-forecast-preprocess-v1'
 */

import * as crypto from 'crypto'

/**
 * 工作流状态常量（4 阶段，无区域裁剪）
 */
export const WorkflowState = {
  /** 阶段1: 等待用户选择研究变量 */
  AWAITING_VARIABLE_SELECTION: 'awaiting_variable_selection',
  /** 阶段2: 等待用户选择静态/掩码变量 */
  AWAITING_STATIC_SELECTION: 'awaiting_static_selection',
  /** 阶段3: 等待用户确认处理参数 */
  AWAITING_PARAMETERS: 'awaiting_parameters',
  /** 阶段4: 等待用户最终确认执行 */
  AWAITING_EXECUTION: 'awaiting_execution',
  /** 通过：可以执行 */
  PASS: 'pass',
  /** 错误状态 */
  ERROR: 'error',
  /** Token 验证失败 */
  TOKEN_INVALID: 'token_invalid'
} as const

export type WorkflowStateType = typeof WorkflowState[keyof typeof WorkflowState]

/**
 * 工作流参数接口（预测专用，无超分参数）
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

  // 阶段3: 处理参数
  train_ratio?: number
  valid_ratio?: number
  test_ratio?: number
  h_slice?: string
  w_slice?: string

  // 阶段4: 最终确认
  user_confirmed?: boolean
  confirmation_token?: string

  [key: string]: any
}

/**
 * 阶段检查结果
 */
export interface StageCheckResult {
  currentState: WorkflowStateType
  missingParams: string[]
  canProceed: boolean
  stageDescription: string
  tokenError?: string
}

/**
 * 预测数据预处理工作流状态机
 */
export class ForecastWorkflow {
  private params: WorkflowParams

  private static readonly TOKEN_SALT = 'ocean-forecast-preprocess-v1'

  constructor(params: WorkflowParams) {
    this.params = params
  }

  /**
   * 生成执行确认 Token
   */
  generateConfirmationToken(): string {
    const { params } = this
    const tokenData = {
      nc_folder: params.nc_folder,
      output_base: params.output_base,
      dyn_vars: params.dyn_vars?.slice().sort().join(','),
      stat_vars: params.stat_vars?.slice().sort().join(','),
      mask_vars: params.mask_vars?.slice().sort().join(','),
      train_ratio: params.train_ratio,
      valid_ratio: params.valid_ratio,
      test_ratio: params.test_ratio,
      h_slice: params.h_slice,
      w_slice: params.w_slice
    }
    const dataStr = JSON.stringify(tokenData) + ForecastWorkflow.TOKEN_SALT
    return crypto.createHash('sha256').update(dataStr).digest('hex').substring(0, 16)
  }

  /**
   * 验证执行确认 Token
   */
  validateConfirmationToken(): boolean {
    if (!this.params.confirmation_token) return false
    return this.params.confirmation_token === this.generateConfirmationToken()
  }

  /**
   * 根据参数倒推当前阶段
   */
  determineCurrentState(): StageCheckResult {
    const { params } = this

    // ========== 阶段5: PASS ==========
    if (params.user_confirmed === true && this.hasAllRequiredParams()) {
      if (!params.confirmation_token) {
        return {
          currentState: WorkflowState.TOKEN_INVALID,
          missingParams: ['confirmation_token'],
          canProceed: false,
          stageDescription: 'Token 缺失',
          tokenError: `⚠️ 检测到跳步行为！

您设置了 user_confirmed=true，但未提供 confirmation_token。

必须：
1. 先调用工具（不带 user_confirmed），进入 awaiting_execution 阶段
2. 从返回结果中获取 confirmation_token
3. 用户确认后，再次调用并携带 user_confirmed=true 和 confirmation_token`
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

【当前 Token】: ${params.confirmation_token}
【期望 Token】: ${this.generateConfirmationToken()}

请重新调用工具（不带 user_confirmed），获取新的 confirmation_token。`
        }
      }
      return {
        currentState: WorkflowState.PASS,
        missingParams: [],
        canProceed: true,
        stageDescription: '所有参数已确认，Token 验证通过，可以执行'
      }
    }

    // ========== 阶段4: AWAITING_EXECUTION ==========
    if (this.hasVariableParams() && this.hasProcessingParams()) {
      return {
        currentState: WorkflowState.AWAITING_EXECUTION,
        missingParams: ['user_confirmed', 'confirmation_token'],
        canProceed: false,
        stageDescription: '所有参数就绪，等待用户最终确认执行'
      }
    }

    // ========== 阶段3: AWAITING_PARAMETERS ==========
    if (this.hasVariableParams()) {
      return {
        currentState: WorkflowState.AWAITING_PARAMETERS,
        missingParams: this.getMissingProcessingParams(),
        canProceed: false,
        stageDescription: '变量已确认，等待处理参数'
      }
    }

    // ========== 阶段2: AWAITING_STATIC_SELECTION ==========
    if (params.dyn_vars && params.dyn_vars.length > 0) {
      const missing: string[] = []
      if (params.stat_vars === undefined) missing.push('stat_vars')
      if (params.mask_vars === undefined) missing.push('mask_vars')
      return {
        currentState: WorkflowState.AWAITING_STATIC_SELECTION,
        missingParams: missing,
        canProceed: false,
        stageDescription: '研究变量已确认，等待静态/掩码变量选择'
      }
    }

    // ========== 阶段1: AWAITING_VARIABLE_SELECTION ==========
    return {
      currentState: WorkflowState.AWAITING_VARIABLE_SELECTION,
      missingParams: ['dyn_vars'],
      canProceed: false,
      stageDescription: '等待用户选择研究变量'
    }
  }

  private hasVariableParams(): boolean {
    const { params } = this
    return !!(
      params.dyn_vars && params.dyn_vars.length > 0 &&
      params.stat_vars !== undefined &&
      params.mask_vars !== undefined
    )
  }

  private hasProcessingParams(): boolean {
    const { params } = this
    return (
      params.train_ratio !== undefined &&
      params.valid_ratio !== undefined &&
      params.test_ratio !== undefined
    )
  }

  private getMissingProcessingParams(): string[] {
    const { params } = this
    const missing: string[] = []
    if (params.train_ratio === undefined) missing.push('train_ratio')
    if (params.valid_ratio === undefined) missing.push('valid_ratio')
    if (params.test_ratio === undefined) missing.push('test_ratio')
    return missing
  }

  private hasAllRequiredParams(): boolean {
    return this.hasVariableParams() && this.hasProcessingParams()
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
        return { status: WorkflowState.ERROR, message: '未知状态', canExecute: false }
    }
  }

  private buildVariableSelectionPrompt(inspectResult?: any): StagePromptResult {
    const dynCandidates = inspectResult?.dynamic_vars_candidates || []
    const variables = inspectResult?.variables || {}

    const varLines = dynCandidates.map((name: string) => {
      const info = variables[name]
      if (!info) return `  - ${name}`
      const dims = info.dims?.join(',') || '?'
      const shape = info.shape?.join('×') || '?'
      return `  - ${name}: 形状 (${shape}), 维度 [${dims}], ${info.dtype || '?'}`
    }).join('\n') || '  无'

    return {
      status: WorkflowState.AWAITING_VARIABLE_SELECTION,
      message: `数据分析完成！

================================================================================
                     ⚠️ 请选择研究变量（必须）
================================================================================

【数据概况】
- 数据目录: ${this.params.nc_folder}
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
   （如果不需要，请指定 stat_vars: []）

3️⃣ **使用哪些掩码变量？**
   可选: ${(inspectResult?.suspected_masks || []).join(', ') || '无'}
   （如果数据没有掩码，请指定 mask_vars: []）

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

  private buildParametersPrompt(inspectResult?: any): StagePromptResult {
    const firstVar = this.params.dyn_vars?.[0]
    const varInfo = inspectResult?.variables?.[firstVar]
    const dataShape = varInfo?.shape || []
    const H = dataShape.length >= 2 ? dataShape[dataShape.length - 2] : '?'
    const W = dataShape.length >= 1 ? dataShape[dataShape.length - 1] : '?'

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

  private buildExecutionPrompt(inspectResult?: any): StagePromptResult {
    const { params } = this
    const confirmationToken = this.generateConfirmationToken()

    const firstVar = params.dyn_vars?.[0]
    const varInfo = inspectResult?.variables?.[firstVar]
    const dataShape = varInfo?.shape || []
    const originalH = dataShape.length >= 2 ? dataShape[dataShape.length - 2] : '?'
    const originalW = dataShape.length >= 1 ? dataShape[dataShape.length - 1] : '?'

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

【处理模式】
- 模式: 预测数据预处理（无下采样）
- 时间排序: 严格时间升序（按 NC 文件内时间变量）

【空间裁剪】
- 原始尺寸: ${originalH} × ${originalW}
${params.h_slice || params.w_slice
    ? `- H 裁剪: ${params.h_slice || '不裁剪'}
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

🔐 **执行确认 Token**: ${confirmationToken}
（Agent 必须将上面一段话发送给用户等待确认，同时必须在下次调用时携带此 token 和 user_confirmed=true）`,
      canExecute: false,
      data: {
        confirmation_token: confirmationToken,
        summary: {
          dyn_vars: params.dyn_vars,
          stat_vars: params.stat_vars,
          mask_vars: params.mask_vars,
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
