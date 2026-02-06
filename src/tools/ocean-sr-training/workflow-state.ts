/**
 * @file workflow-state.ts
 * @description 训练工作流状态机 - 实现分阶段强制确认逻辑
 *              核心思想：根据已有参数倒推当前阶段，防止跳步
 *
 * @author Leizheng
 * @date 2026-02-06
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-06 Leizheng: v1.0.0 初始版本
 *     - 4 阶段状态机: 数据确认 → 模型选择 → 参数确认 → 执行确认
 *     - Token 防跳步机制
 *     - 自动检测数据目录的 dyn_vars / scale / shape
 *     - GPU 信息集成到参数确认阶段
 */

import * as crypto from 'crypto'

/**
 * 训练工作流状态常量
 */
export const TrainingState = {
  /** 阶段1: 等待确认数据目录和输出目录 */
  AWAITING_DATA_CONFIRMATION: 'awaiting_data_confirmation',
  /** 阶段2: 等待选择训练模型 */
  AWAITING_MODEL_SELECTION: 'awaiting_model_selection',
  /** 阶段3: 等待确认训练参数（含 GPU 选择） */
  AWAITING_PARAMETERS: 'awaiting_parameters',
  /** 阶段4: 等待用户最终确认执行 */
  AWAITING_EXECUTION: 'awaiting_execution',
  /** 阶段5: 所有确认通过，可以执行 */
  PASS: 'pass',
  /** 错误状态 */
  ERROR: 'error',
  /** Token 验证失败 */
  TOKEN_INVALID: 'token_invalid'
} as const

export type TrainingStateType = typeof TrainingState[keyof typeof TrainingState]

/**
 * 训练工作流参数接口
 */
export interface TrainingWorkflowParams {
  // ====== 阶段1: 数据确认 ======
  dataset_root?: string
  log_dir?: string

  // ====== 阶段2: 模型选择 ======
  model_name?: string

  // ====== 阶段3: 训练参数 ======
  dyn_vars?: string[]
  scale?: number
  mode?: string
  epochs?: number
  lr?: number
  batch_size?: number
  eval_batch_size?: number
  device_ids?: number[]
  distribute?: boolean
  distribute_mode?: string
  patience?: number
  eval_freq?: number
  normalize?: boolean
  normalizer_type?: string
  optimizer?: string
  weight_decay?: number
  scheduler?: string
  scheduler_step_size?: number
  scheduler_gamma?: number
  seed?: number
  wandb?: boolean
  ckpt_path?: string

  // ====== 阶段4: 执行确认 ======
  user_confirmed?: boolean
  confirmation_token?: string

  // 其他
  [key: string]: any
}

/**
 * 阶段检查结果
 */
export interface TrainingStageCheckResult {
  currentState: TrainingStateType
  missingParams: string[]
  canProceed: boolean
  stageDescription: string
  tokenError?: string
}

/**
 * 阶段提示结果
 */
export interface TrainingStagePromptResult {
  status: TrainingStateType
  message: string
  canExecute: boolean
  data?: any
}

/**
 * 数据集验证信息（从 validate_dataset.py 获取）
 */
export interface DatasetValidationInfo {
  status: string
  dataset_root: string
  dyn_vars: string[]
  scale: number | null
  hr_shape: number[] | null
  lr_shape: number[] | null
  splits: Record<string, { hr_count: number; lr_count: number }>
  has_static: boolean
  static_vars: string[]
  total_samples: { hr: number; lr: number }
  warnings: string[]
  errors: string[]
}

/**
 * GPU 信息
 */
export interface GpuInfo {
  cuda_available: boolean
  gpu_count: number
  gpus: Array<{
    id: number
    name: string
    total_memory_gb: number
    free_memory_gb: number
    used_memory_gb: number
  }>
  error?: string
}

/**
 * 模型信息
 */
export interface ModelInfo {
  name: string
  category: string
  trainer: string
  description: string
}

/**
 * 训练工作流状态机
 *
 * 核心逻辑：根据参数倒推当前阶段，严格防止跳步
 * - 阶段1: 确认数据目录（dataset_root + log_dir）→ 自动检测参数
 * - 阶段2: 选择模型（model_name）
 * - 阶段3: 确认训练参数（epochs, lr, batch_size, GPU等）
 * - 阶段4: 最终执行确认（token 机制）
 */
export class TrainingWorkflow {
  private params: TrainingWorkflowParams

  private static readonly TOKEN_SALT = 'ocean-sr-training-v1'

  constructor(params: TrainingWorkflowParams) {
    this.params = params
  }

  /**
   * 生成执行确认 Token
   */
  generateConfirmationToken(): string {
    const { params } = this
    const tokenData = {
      dataset_root: params.dataset_root,
      log_dir: params.log_dir,
      model_name: params.model_name,
      dyn_vars: params.dyn_vars?.sort().join(','),
      scale: params.scale,
      mode: params.mode,
      epochs: params.epochs,
      lr: params.lr,
      batch_size: params.batch_size,
      device_ids: params.device_ids?.join(','),
      distribute: params.distribute,
      distribute_mode: params.distribute_mode,
    }

    const dataStr = JSON.stringify(tokenData) + TrainingWorkflow.TOKEN_SALT
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
   * 核心方法：根据参数倒推当前阶段
   */
  determineCurrentState(): TrainingStageCheckResult {
    const { params } = this

    // ========== 阶段5: PASS ==========
    if (params.user_confirmed === true && this.hasAllRequiredParams()) {
      if (!params.confirmation_token) {
        return {
          currentState: TrainingState.TOKEN_INVALID,
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
          currentState: TrainingState.TOKEN_INVALID,
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

      return {
        currentState: TrainingState.PASS,
        missingParams: [],
        canProceed: true,
        stageDescription: '所有参数已确认，Token 验证通过，可以执行训练'
      }
    }

    // ========== 阶段4: 等待执行确认 ==========
    if (this.hasDataParams() && this.hasModelParam() && this.hasTrainingParams()) {
      return {
        currentState: TrainingState.AWAITING_EXECUTION,
        missingParams: ['user_confirmed', 'confirmation_token'],
        canProceed: false,
        stageDescription: '所有参数就绪，等待用户最终确认执行'
      }
    }

    // ========== 阶段3: 等待训练参数 ==========
    if (this.hasDataParams() && this.hasModelParam()) {
      const missing = this.getMissingTrainingParams()
      return {
        currentState: TrainingState.AWAITING_PARAMETERS,
        missingParams: missing,
        canProceed: false,
        stageDescription: '模型已选择，等待确认训练参数'
      }
    }

    // ========== 阶段2: 等待模型选择 ==========
    if (this.hasDataParams()) {
      return {
        currentState: TrainingState.AWAITING_MODEL_SELECTION,
        missingParams: ['model_name'],
        canProceed: false,
        stageDescription: '数据已确认，等待选择模型'
      }
    }

    // ========== 阶段1: 等待数据确认 ==========
    const missingData: string[] = []
    if (!params.dataset_root) missingData.push('dataset_root')
    if (!params.log_dir) missingData.push('log_dir')

    return {
      currentState: TrainingState.AWAITING_DATA_CONFIRMATION,
      missingParams: missingData,
      canProceed: false,
      stageDescription: '等待确认数据目录和输出目录'
    }
  }

  // ============================
  // 阶段检查辅助方法
  // ============================

  private hasDataParams(): boolean {
    return !!(this.params.dataset_root && this.params.log_dir)
  }

  private hasModelParam(): boolean {
    return !!this.params.model_name
  }

  private hasTrainingParams(): boolean {
    const { params } = this
    return !!(
      params.dyn_vars && params.dyn_vars.length > 0 &&
      params.scale !== undefined && params.scale > 0 &&
      params.epochs !== undefined && params.epochs > 0 &&
      params.lr !== undefined && params.lr > 0 &&
      params.batch_size !== undefined && params.batch_size > 0 &&
      params.device_ids && params.device_ids.length > 0
    )
  }

  private getMissingTrainingParams(): string[] {
    const { params } = this
    const missing: string[] = []
    if (!params.dyn_vars || params.dyn_vars.length === 0) missing.push('dyn_vars')
    if (params.scale === undefined || params.scale <= 0) missing.push('scale')
    if (params.epochs === undefined || params.epochs <= 0) missing.push('epochs')
    if (params.lr === undefined || params.lr <= 0) missing.push('lr')
    if (params.batch_size === undefined || params.batch_size <= 0) missing.push('batch_size')
    if (!params.device_ids || params.device_ids.length === 0) missing.push('device_ids')
    return missing
  }

  private hasAllRequiredParams(): boolean {
    return this.hasDataParams() && this.hasModelParam() && this.hasTrainingParams()
  }

  // ============================
  // 阶段提示构建
  // ============================

  /**
   * 获取当前阶段的用户提示信息
   */
  getStagePrompt(context?: {
    datasetInfo?: DatasetValidationInfo
    gpuInfo?: GpuInfo
    modelList?: ModelInfo[]
  }): TrainingStagePromptResult {
    const stateCheck = this.determineCurrentState()

    switch (stateCheck.currentState) {
      case TrainingState.AWAITING_DATA_CONFIRMATION:
        return this.buildDataConfirmationPrompt(context?.datasetInfo)

      case TrainingState.AWAITING_MODEL_SELECTION:
        return this.buildModelSelectionPrompt(context?.datasetInfo, context?.modelList)

      case TrainingState.AWAITING_PARAMETERS:
        return this.buildParametersPrompt(context?.datasetInfo, context?.gpuInfo)

      case TrainingState.AWAITING_EXECUTION:
        return this.buildExecutionPrompt(context?.datasetInfo, context?.gpuInfo)

      case TrainingState.TOKEN_INVALID:
        return {
          status: TrainingState.TOKEN_INVALID,
          message: stateCheck.tokenError || 'Token 验证失败',
          canExecute: false,
          data: {
            error_type: 'token_invalid',
            expected_token: this.generateConfirmationToken(),
            provided_token: this.params.confirmation_token
          }
        }

      case TrainingState.PASS:
        return {
          status: TrainingState.PASS,
          message: '所有参数已确认，Token 验证通过，开始执行训练...',
          canExecute: true
        }

      default:
        return {
          status: TrainingState.ERROR,
          message: '未知状态',
          canExecute: false
        }
    }
  }

  /**
   * 阶段1: 数据确认提示
   */
  private buildDataConfirmationPrompt(datasetInfo?: DatasetValidationInfo): TrainingStagePromptResult {
    if (!datasetInfo) {
      return {
        status: TrainingState.AWAITING_DATA_CONFIRMATION,
        message: `================================================================================
                    ⚠️ 请确认数据目录和输出目录
================================================================================

**请提供以下信息：**

1️⃣ **dataset_root**: 预处理数据根目录（ocean-preprocess 输出目录）
   - 该目录应包含 train/valid/test 子目录
   - 每个子目录下应有 hr/ 和 lr/ 数据

2️⃣ **log_dir**: 训练日志输出目录
   - 训练日志、模型权重、配置文件将保存于此

================================================================================

⚠️ Agent 注意：**禁止自动猜测数据目录！**
必须等待用户明确指定后再继续。`,
        canExecute: false,
        data: { missing: ['dataset_root', 'log_dir'] }
      }
    }

    // 数据目录已提供，展示检测结果
    const hasErrors = datasetInfo.errors.length > 0

    if (hasErrors) {
      return {
        status: TrainingState.AWAITING_DATA_CONFIRMATION,
        message: `================================================================================
                    ❌ 数据目录验证失败
================================================================================

【数据目录】${datasetInfo.dataset_root}

【错误】
${datasetInfo.errors.map(e => `  ❌ ${e}`).join('\n')}

${datasetInfo.warnings.length > 0 ? `【警告】\n${datasetInfo.warnings.map(w => `  ⚠️ ${w}`).join('\n')}` : ''}

================================================================================

**请检查数据目录是否正确，可能需要先运行数据预处理（ocean-preprocess）**

================================================================================

⚠️ Agent 注意：数据验证失败，不能继续。请告知用户错误信息并等待新的路径。`,
        canExecute: false,
        data: {
          dataset_root: datasetInfo.dataset_root,
          errors: datasetInfo.errors,
          warnings: datasetInfo.warnings
        }
      }
    }

    // 格式化 split 信息
    const splitLines = Object.entries(datasetInfo.splits).map(([split, info]) => {
      return `  - ${split}: HR ${info.hr_count} 个, LR ${info.lr_count} 个`
    }).join('\n')

    return {
      status: TrainingState.AWAITING_DATA_CONFIRMATION,
      message: `数据目录验证通过！

================================================================================
                    📊 数据集信息
================================================================================

【基本信息】
- 数据目录: ${datasetInfo.dataset_root}
- 日志目录: ${this.params.log_dir}

【检测到的变量】
- 动态变量: ${datasetInfo.dyn_vars.join(', ')}
- 静态变量: ${datasetInfo.has_static ? datasetInfo.static_vars.join(', ') : '无'}

【数据形状】
- HR 尺寸: ${datasetInfo.hr_shape ? datasetInfo.hr_shape.join(' × ') : '未检测到'}
- LR 尺寸: ${datasetInfo.lr_shape ? datasetInfo.lr_shape.join(' × ') : '未检测到'}
- 推算 scale: ${datasetInfo.scale ?? '未能推算（缺少 LR 数据）'}

【数据集划分】
${splitLines}
- 总样本: HR ${datasetInfo.total_samples.hr} 个, LR ${datasetInfo.total_samples.lr} 个

${datasetInfo.warnings.length > 0 ? `【警告】\n${datasetInfo.warnings.map(w => `  ⚠️ ${w}`).join('\n')}\n` : ''}
================================================================================

数据验证通过，请继续选择模型。
Agent 可以进入下一阶段（阶段2：模型选择）。`,
      canExecute: false,
      data: {
        dataset_root: datasetInfo.dataset_root,
        log_dir: this.params.log_dir,
        detected_dyn_vars: datasetInfo.dyn_vars,
        detected_scale: datasetInfo.scale,
        hr_shape: datasetInfo.hr_shape,
        lr_shape: datasetInfo.lr_shape,
        splits: datasetInfo.splits,
        total_samples: datasetInfo.total_samples,
        warnings: datasetInfo.warnings
      }
    }
  }

  /**
   * 阶段2: 模型选择提示
   */
  private buildModelSelectionPrompt(
    datasetInfo?: DatasetValidationInfo,
    modelList?: ModelInfo[]
  ): TrainingStagePromptResult {
    // 格式化模型列表
    let modelListStr = '（模型列表加载失败，请调用 ocean_sr_list_models 查看）'
    if (modelList && modelList.length > 0) {
      const standardModels = modelList.filter(m => m.category === 'standard')
      const diffusionModels = modelList.filter(m => m.category === 'diffusion')

      const formatGroup = (models: ModelInfo[]) =>
        models.map(m => `  - ${m.name}: ${m.description}`).join('\n')

      modelListStr = `【标准模型】（BaseTrainer）
${formatGroup(standardModels)}

【扩散模型】（DDPMTrainer / ResshiftTrainer）
${formatGroup(diffusionModels)}`
    }

    return {
      status: TrainingState.AWAITING_MODEL_SELECTION,
      message: `数据确认完成：
- 数据目录: ${this.params.dataset_root}
- 日志目录: ${this.params.log_dir}
${datasetInfo ? `- 检测到变量: ${datasetInfo.dyn_vars.join(', ')}
- 推算 scale: ${datasetInfo.scale ?? '未知'}
- HR 尺寸: ${datasetInfo.hr_shape?.join(' × ') ?? '未知'}` : ''}

================================================================================
                    ⚠️ 请选择训练模型
================================================================================

${modelListStr}

================================================================================

**请回答以下问题：**

🔹 **选择哪个模型进行训练？**
   请从上面的模型列表中选择一个模型名称。

   💡 推荐：
   - 初次尝试推荐 SwinIR（平衡性能和效果）
   - 追求精度推荐 FNO2d 或 HiNOTE
   - 需要不确定性估计推荐扩散模型（DDPM / SR3）

================================================================================

⚠️ Agent 注意：**禁止自动选择模型！**
必须等待用户明确指定后再继续。`,
      canExecute: false,
      data: {
        dataset_root: this.params.dataset_root,
        log_dir: this.params.log_dir,
        detected_dyn_vars: datasetInfo?.dyn_vars,
        detected_scale: datasetInfo?.scale,
        model_list: modelList
      }
    }
  }

  /**
   * 阶段3: 训练参数确认提示
   */
  private buildParametersPrompt(
    datasetInfo?: DatasetValidationInfo,
    gpuInfo?: GpuInfo
  ): TrainingStagePromptResult {
    const { params } = this

    // GPU 信息
    let gpuStr = 'GPU 信息未获取'
    if (gpuInfo) {
      if (!gpuInfo.cuda_available) {
        gpuStr = '⚠️ 未检测到可用 GPU！训练需要 GPU 支持。'
      } else {
        gpuStr = gpuInfo.gpus.map(g =>
          `  - GPU ${g.id}: ${g.name} (总 ${g.total_memory_gb}GB / 空闲 ${g.free_memory_gb}GB / 已用 ${g.used_memory_gb}GB)`
        ).join('\n')
      }
    }

    // 自动检测到的值
    const detectedVars = datasetInfo?.dyn_vars || []
    const detectedScale = datasetInfo?.scale

    // 当前已填参数（有默认值的显示默认值）
    const currentEpochs = params.epochs ?? 500
    const currentLr = params.lr ?? 0.001
    const currentBatchSize = params.batch_size ?? 32
    const currentEvalBatchSize = params.eval_batch_size ?? 32
    const currentDeviceIds = params.device_ids ?? [0]
    const currentDistribute = params.distribute ?? false
    const currentDistributeMode = params.distribute_mode ?? 'DDP'
    const currentPatience = params.patience ?? 10
    const currentEvalFreq = params.eval_freq ?? 5
    const currentNormalize = params.normalize ?? true
    const currentNormalizerType = params.normalizer_type ?? 'PGN'
    const currentOptimizer = params.optimizer ?? 'AdamW'
    const currentWeightDecay = params.weight_decay ?? 0.001
    const currentScheduler = params.scheduler ?? 'StepLR'
    const currentSchedulerStepSize = params.scheduler_step_size ?? 300
    const currentSchedulerGamma = params.scheduler_gamma ?? 0.5
    const currentSeed = params.seed ?? 42

    return {
      status: TrainingState.AWAITING_PARAMETERS,
      message: `模型已选择：${params.model_name}

================================================================================
                    ⚠️ 请确认训练参数
================================================================================

【数据参数】（从数据目录自动检测，请确认）
- dyn_vars: ${detectedVars.length > 0 ? detectedVars.join(', ') : '❓ 未检测到，请手动指定'}${params.dyn_vars ? ` ✅ 当前: ${params.dyn_vars.join(', ')}` : ''}
- scale: ${detectedScale ?? '❓ 未检测到，请手动指定'}${params.scale !== undefined ? ` ✅ 当前: ${params.scale}` : ''}

【训练核心参数】
- epochs: ${currentEpochs}（训练轮数）
- lr: ${currentLr}（学习率）
- batch_size: ${currentBatchSize}（训练 batch size）
- eval_batch_size: ${currentEvalBatchSize}（评估 batch size）
- patience: ${currentPatience}（早停耐心值）
- eval_freq: ${currentEvalFreq}（评估频率，每 N 个 epoch）

【优化器参数】
- optimizer: ${currentOptimizer}（可选: AdamW, Adam, SGD）
- weight_decay: ${currentWeightDecay}
- scheduler: ${currentScheduler}（可选: StepLR, MultiStepLR, OneCycleLR）
- scheduler_step_size: ${currentSchedulerStepSize}
- scheduler_gamma: ${currentSchedulerGamma}

【归一化参数】
- normalize: ${currentNormalize}
- normalizer_type: ${currentNormalizerType}（可选: PGN, GN）

【GPU 配置】
${gpuStr}

- device_ids: [${currentDeviceIds.join(', ')}]（选择使用的 GPU）
- distribute: ${currentDistribute}（是否多卡训练）
- distribute_mode: ${currentDistributeMode}（多卡模式: DP / DDP）

${gpuInfo && gpuInfo.gpu_count > 1 ? `💡 检测到 ${gpuInfo.gpu_count} 张 GPU，建议使用多卡 DDP 训练以加速。` : ''}

【其他参数】
- seed: ${currentSeed}（随机种子）
- wandb: ${params.wandb ?? false}（是否启用 WandB）
${params.ckpt_path ? `- ckpt_path: ${params.ckpt_path}（恢复训练检查点）` : ''}

================================================================================

**请确认或修改上述参数。**
- 以上参数均有默认值，如果都可以接受，直接回复"确认"
- 如果需要修改，请指明要修改的参数和新值
- **必须确认的参数**: dyn_vars, scale, device_ids

================================================================================

⚠️ Agent 注意：
- dyn_vars 和 scale 如果自动检测到了，需向用户展示并确认
- device_ids 必须由用户确认使用哪些 GPU
- **禁止自动决定训练参数！**必须等待用户确认后再继续。`,
      canExecute: false,
      data: {
        model_name: params.model_name,
        detected_dyn_vars: detectedVars,
        detected_scale: detectedScale,
        current_params: {
          dyn_vars: params.dyn_vars,
          scale: params.scale,
          epochs: currentEpochs,
          lr: currentLr,
          batch_size: currentBatchSize,
          eval_batch_size: currentEvalBatchSize,
          device_ids: currentDeviceIds,
          distribute: currentDistribute,
          distribute_mode: currentDistributeMode,
          patience: currentPatience,
          eval_freq: currentEvalFreq,
          normalize: currentNormalize,
          normalizer_type: currentNormalizerType,
          optimizer: currentOptimizer,
          weight_decay: currentWeightDecay,
          scheduler: currentScheduler,
          scheduler_step_size: currentSchedulerStepSize,
          scheduler_gamma: currentSchedulerGamma,
          seed: currentSeed,
          wandb: params.wandb ?? false,
          ckpt_path: params.ckpt_path,
        },
        gpu_info: gpuInfo
      }
    }
  }

  /**
   * 阶段4: 执行确认提示
   */
  private buildExecutionPrompt(
    datasetInfo?: DatasetValidationInfo,
    gpuInfo?: GpuInfo
  ): TrainingStagePromptResult {
    const { params } = this

    const confirmationToken = this.generateConfirmationToken()

    // GPU 模式描述
    const deviceIds = params.device_ids || [0]
    const distribute = params.distribute ?? false
    const distributeMode = params.distribute_mode ?? 'DDP'
    let gpuModeStr: string
    if (deviceIds.length === 1) {
      gpuModeStr = `单卡 (GPU ${deviceIds[0]})`
    } else if (distribute && distributeMode === 'DDP') {
      gpuModeStr = `多卡 DDP (GPU ${deviceIds.join(', ')})`
    } else {
      gpuModeStr = `多卡 DP (GPU ${deviceIds.join(', ')})`
    }

    // GPU 名称
    let gpuNames = ''
    if (gpuInfo) {
      const selectedGpus = gpuInfo.gpus.filter(g => deviceIds.includes(g.id))
      gpuNames = selectedGpus.map(g => `${g.name} (${g.free_memory_gb}GB 可用)`).join(', ')
    }

    return {
      status: TrainingState.AWAITING_EXECUTION,
      message: `所有参数已确认，请检查后确认执行：

================================================================================
                         📋 训练参数汇总
================================================================================

【数据信息】
- 数据目录: ${params.dataset_root}
- 日志目录: ${params.log_dir}
- 动态变量: ${params.dyn_vars?.join(', ')}
- Scale: ${params.scale}x
${datasetInfo ? `- HR 尺寸: ${datasetInfo.hr_shape?.join(' × ') ?? '?'}
- 总样本: HR ${datasetInfo.total_samples.hr} / LR ${datasetInfo.total_samples.lr}` : ''}

【模型配置】
- 模型: ${params.model_name}
- 模式: ${params.mode ?? 'train'}

【训练参数】
- Epochs: ${params.epochs}
- 学习率: ${params.lr}
- Batch Size: ${params.batch_size}
- 评估 Batch Size: ${params.eval_batch_size ?? 32}
- 早停耐心值: ${params.patience ?? 10}
- 评估频率: 每 ${params.eval_freq ?? 5} 个 epoch

【优化器】
- 优化器: ${params.optimizer ?? 'AdamW'}
- 权重衰减: ${params.weight_decay ?? 0.001}
- 调度器: ${params.scheduler ?? 'StepLR'}

【GPU 配置】
- 运行模式: ${gpuModeStr}
${gpuNames ? `- GPU: ${gpuNames}` : ''}

【其他】
- 归一化: ${params.normalize ?? true} (${params.normalizer_type ?? 'PGN'})
- 随机种子: ${params.seed ?? 42}
- WandB: ${params.wandb ?? false}
${params.ckpt_path ? `- 检查点恢复: ${params.ckpt_path}` : ''}

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
          dataset_root: params.dataset_root,
          log_dir: params.log_dir,
          model_name: params.model_name,
          dyn_vars: params.dyn_vars,
          scale: params.scale,
          mode: params.mode,
          epochs: params.epochs,
          lr: params.lr,
          batch_size: params.batch_size,
          eval_batch_size: params.eval_batch_size,
          device_ids: params.device_ids,
          distribute: params.distribute,
          distribute_mode: params.distribute_mode,
          patience: params.patience,
          eval_freq: params.eval_freq,
          optimizer: params.optimizer,
          weight_decay: params.weight_decay,
          scheduler: params.scheduler,
          normalize: params.normalize,
          normalizer_type: params.normalizer_type,
          seed: params.seed,
          wandb: params.wandb,
          ckpt_path: params.ckpt_path,
        }
      }
    }
  }
}
