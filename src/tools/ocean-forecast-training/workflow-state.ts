/**
 * @file workflow-state.ts
 * @description Ocean forecast training workflow state machine - 4-stage confirmation logic
 * @author Leizheng
 * @date 2026-02-26
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-26 Leizheng: v1.0.0 initial version for ocean forecast training
 */

import * as crypto from 'crypto'

/* ------------------------------------------------------------------ */
/*  Model sets                                                         */
/* ------------------------------------------------------------------ */

/** FFT-based models where AMP can cause numerical issues */
const AMP_AUTO_DISABLE_MODELS = new Set(['FNO2d', 'M2NO2d'])

/** Heavy models that benefit from gradient checkpointing by default */
const HEAVY_MODELS = new Set([
  'GalerkinTransformer',
  'SwinTransformerV2',
  'SwinMLP',
])

/* ------------------------------------------------------------------ */
/*  Helper functions                                                    */
/* ------------------------------------------------------------------ */

function resolveGradientCheckpointing(params: ForecastWorkflowParams): boolean {
  if (params.gradient_checkpointing !== undefined) {
    return Boolean(params.gradient_checkpointing)
  }
  const heavyModel = params.model_name
    ? HEAVY_MODELS.has(params.model_name)
    : false
  return heavyModel
}

function resolveUseAmp(params: ForecastWorkflowParams): boolean {
  if (params.use_amp !== undefined) {
    return Boolean(params.use_amp)
  }
  if (params.model_name && AMP_AUTO_DISABLE_MODELS.has(params.model_name)) {
    return false
  }
  return true
}

/**
 * Filter out undefined values from an object to prevent undefined
 * from overwriting session-confirmed values during merge.
 */
function filterDefined<T extends Record<string, any>>(
  obj: T
): Partial<T> {
  return Object.fromEntries(
    Object.entries(obj).filter(([, v]) => v !== undefined)
  ) as Partial<T>
}

/* ------------------------------------------------------------------ */
/*  State constants                                                     */
/* ------------------------------------------------------------------ */

export const ForecastTrainingState = {
  /** Stage 1: Awaiting confirmation of data directory and output directory */
  AWAITING_DATA_CONFIRMATION: 'awaiting_data_confirmation',
  /** Stage 2: Awaiting model selection */
  AWAITING_MODEL_SELECTION: 'awaiting_model_selection',
  /** Stage 3: Awaiting training parameter confirmation (incl. GPU) */
  AWAITING_PARAMETERS: 'awaiting_parameters',
  /** Stage 4: Awaiting final user confirmation to execute */
  AWAITING_EXECUTION: 'awaiting_execution',
  /** Stage 5: All confirmations passed, ready to execute */
  PASS: 'pass',
  /** Error state */
  ERROR: 'error',
  /** Token validation failed */
  TOKEN_INVALID: 'token_invalid',
} as const

export type ForecastTrainingStateType =
  (typeof ForecastTrainingState)[keyof typeof ForecastTrainingState]

/* ------------------------------------------------------------------ */
/*  Interfaces                                                          */
/* ------------------------------------------------------------------ */

/**
 * Forecast training workflow parameters
 */
export interface ForecastWorkflowParams {
  // ====== Stage 1: Data confirmation ======
  dataset_root?: string
  log_dir?: string

  // ====== Stage 2: Model selection ======
  model_name?: string

  // ====== Stage 3: Training parameters ======
  dyn_vars?: string[]
  mode?: string
  epochs?: number
  lr?: number
  batch_size?: number
  eval_batch_size?: number
  device_ids?: number[]
  distribute?: boolean
  distribute_mode?: string
  master_port?: number
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

  // ====== Forecast-specific parameters ======
  in_t?: number
  out_t?: number
  stride?: number

  // ====== OOM protection parameters ======
  use_amp?: boolean
  gradient_checkpointing?: boolean

  // ====== Stage 4: Execution confirmation ======
  user_confirmed?: boolean
  confirmation_token?: string

  // Other
  [key: string]: any
}

/**
 * Stage check result
 */
export interface ForecastStageCheckResult {
  currentState: ForecastTrainingStateType
  missingParams: string[]
  canProceed: boolean
  stageDescription: string
  tokenError?: string
}

/**
 * Stage prompt result
 */
export interface ForecastStagePromptResult {
  status: ForecastTrainingStateType
  message: string
  canExecute: boolean
  data?: any
}

/**
 * Forecast dataset validation info
 */
export interface ForecastDatasetInfo {
  status: string
  dataset_root: string
  dyn_vars: string[]
  spatial_shape: [number, number] | null // [H, W]
  splits: Record<string, number> // { train: N, valid: N, test: N }
  total_timesteps: number
  time_range: { start: string; end: string } | null
  has_static: boolean
  static_vars: string[]
  warnings: string[]
  errors: string[]
}

/**
 * GPU info
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
 * Model info
 */
export interface ModelInfo {
  name: string
  category: string
  trainer: string
  description: string
  supported?: boolean
  notes?: string
}

/* ------------------------------------------------------------------ */
/*  Workflow state machine                                              */
/* ------------------------------------------------------------------ */

/**
 * Forecast training workflow state machine
 *
 * Core logic: determine current stage from existing params, strictly prevent skipping.
 * - Stage 1: Confirm data directory (dataset_root + log_dir) -> auto-detect params
 * - Stage 2: Select model (model_name)
 * - Stage 3: Confirm training params (in_t, out_t, stride, epochs, lr, batch_size, GPU, etc.)
 * - Stage 4: Final execution confirmation (token mechanism)
 */
export class ForecastTrainingWorkflow {
  private params: ForecastWorkflowParams

  /**
   * Get merged complete parameters (user-supplied > session > defaults).
   * Used at execution stage to ensure user-confirmed params are not lost
   * due to missing fields in subsequent stateless calls.
   */
  getParams(): Readonly<ForecastWorkflowParams> {
    return this.params
  }

  private static readonly TOKEN_SALT = 'ocean-forecast-training-v1'

  constructor(
    args: ForecastWorkflowParams,
    sessionOverrides?: ForecastWorkflowParams,
    _datasetInfo?: ForecastDatasetInfo,
    _gpuInfo?: GpuInfo
  ) {
    // Filter out undefined values from args to prevent overwriting session-confirmed params
    const definedArgs = filterDefined(args)

    // Merge order: system defaults -> session confirmed values -> explicitly supplied values
    this.params = {
      mode: 'train',
      in_t: 7,
      out_t: 1,
      stride: 1,
      epochs: 500,
      lr: 0.001,
      batch_size: 4,
      eval_batch_size: 4,
      distribute: false,
      distribute_mode: 'DDP',
      patience: 10,
      eval_freq: 5,
      normalize: true,
      normalizer_type: 'PGN',
      optimizer: 'AdamW',
      weight_decay: 0.001,
      scheduler: 'StepLR',
      scheduler_step_size: 300,
      scheduler_gamma: 0.5,
      seed: 42,
      wandb: false,
      gradient_checkpointing: true,
      user_confirmed: false,
      ...(sessionOverrides ?? {}),
      ...definedArgs,
    }
  }

  /**
   * Generate execution confirmation token.
   * Signature: hash(dataset_root + model_name + in_t + out_t + batch_size)
   */
  generateConfirmationToken(): string {
    const { params } = this
    const tokenData = {
      dataset_root: params.dataset_root,
      model_name: params.model_name,
      in_t: params.in_t,
      out_t: params.out_t,
      batch_size: params.batch_size,
    }

    const dataStr =
      JSON.stringify(tokenData) + ForecastTrainingWorkflow.TOKEN_SALT
    return crypto
      .createHash('sha256')
      .update(dataStr)
      .digest('hex')
      .substring(0, 16)
  }

  /**
   * Validate execution confirmation token
   */
  validateConfirmationToken(): boolean {
    if (!this.params.confirmation_token) return false
    return this.params.confirmation_token === this.generateConfirmationToken()
  }

  /**
   * Core method: determine current stage from parameters
   */
  determineCurrentState(): ForecastStageCheckResult {
    const { params } = this

    // ========== Stage 5: PASS ==========
    // user_confirmed=true + all required params present + token valid
    if (params.user_confirmed === true && this.hasAllRequiredParams()) {
      if (!this.validateConfirmationToken()) {
        return {
          currentState: ForecastTrainingState.TOKEN_INVALID,
          missingParams: [],
          canProceed: false,
          stageDescription: 'Parameters modified after confirmation, token validation failed',
          tokenError: `Token mismatch: parameters were modified after user confirmation (e.g. device_ids, batch_size), please re-confirm. Provided token: ${params.confirmation_token ?? '(empty)'}, expected token: ${this.generateConfirmationToken()}`,
        }
      }
      return {
        currentState: ForecastTrainingState.PASS,
        missingParams: [],
        canProceed: true,
        stageDescription:
          'All parameters confirmed, token validated, ready to execute training',
      }
    }

    // ========== Stage 4: Awaiting execution confirmation ==========
    if (
      this.hasDataParams() &&
      this.hasModelParam() &&
      this.hasTrainingParams()
    ) {
      return {
        currentState: ForecastTrainingState.AWAITING_EXECUTION,
        missingParams: ['user_confirmed', 'confirmation_token'],
        canProceed: false,
        stageDescription:
          'All parameters ready, awaiting final user confirmation',
      }
    }

    // ========== Stage 3: Awaiting training parameters ==========
    if (this.hasDataParams() && this.hasModelParam()) {
      const missing = this.getMissingTrainingParams()
      return {
        currentState: ForecastTrainingState.AWAITING_PARAMETERS,
        missingParams: missing,
        canProceed: false,
        stageDescription:
          'Model selected, awaiting training parameter confirmation',
      }
    }

    // ========== Stage 2: Awaiting model selection ==========
    if (this.hasDataParams()) {
      return {
        currentState: ForecastTrainingState.AWAITING_MODEL_SELECTION,
        missingParams: ['model_name'],
        canProceed: false,
        stageDescription: 'Data confirmed, awaiting model selection',
      }
    }

    // ========== Stage 1: Awaiting data confirmation ==========
    const missingData: string[] = []
    if (!params.dataset_root) missingData.push('dataset_root')
    if (!params.log_dir) missingData.push('log_dir')

    return {
      currentState: ForecastTrainingState.AWAITING_DATA_CONFIRMATION,
      missingParams: missingData,
      canProceed: false,
      stageDescription: 'Awaiting data directory and output directory confirmation',
    }
  }

  // ============================
  // Stage check helpers
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
      params.dyn_vars &&
      params.dyn_vars.length > 0 &&
      params.epochs !== undefined &&
      params.epochs > 0 &&
      params.lr !== undefined &&
      params.lr > 0 &&
      params.batch_size !== undefined &&
      params.batch_size > 0 &&
      params.device_ids &&
      params.device_ids.length > 0
    )
  }

  private getMissingTrainingParams(): string[] {
    const { params } = this
    const missing: string[] = []
    if (!params.dyn_vars || params.dyn_vars.length === 0)
      missing.push('dyn_vars')
    if (params.epochs === undefined || params.epochs <= 0)
      missing.push('epochs')
    if (params.lr === undefined || params.lr <= 0) missing.push('lr')
    if (params.batch_size === undefined || params.batch_size <= 0)
      missing.push('batch_size')
    if (!params.device_ids || params.device_ids.length === 0)
      missing.push('device_ids')
    return missing
  }

  private hasAllRequiredParams(): boolean {
    return (
      this.hasDataParams() && this.hasModelParam() && this.hasTrainingParams()
    )
  }

  // ============================
  // Stage prompt builders
  // ============================

  /**
   * Get prompt information for the current stage
   */
  getStagePrompt(context?: {
    datasetInfo?: ForecastDatasetInfo
    gpuInfo?: GpuInfo
    modelList?: ModelInfo[]
  }): ForecastStagePromptResult {
    const stateCheck = this.determineCurrentState()

    switch (stateCheck.currentState) {
      case ForecastTrainingState.AWAITING_DATA_CONFIRMATION:
        return this.buildDataConfirmationPrompt(context?.datasetInfo)

      case ForecastTrainingState.AWAITING_MODEL_SELECTION:
        return this.buildModelSelectionPrompt(
          context?.datasetInfo,
          context?.modelList
        )

      case ForecastTrainingState.AWAITING_PARAMETERS:
        return this.buildParametersPrompt(
          context?.datasetInfo,
          context?.gpuInfo
        )

      case ForecastTrainingState.AWAITING_EXECUTION:
        return this.buildExecutionPrompt(context?.datasetInfo, context?.gpuInfo)

      case ForecastTrainingState.TOKEN_INVALID:
        return {
          status: ForecastTrainingState.TOKEN_INVALID,
          message: `================================================================================
                    Token validation failed - parameters modified
================================================================================

${stateCheck.tokenError || 'Token validation failed'}

Parameters were modified after user confirmation (possibly Agent auto-adjusted
device_ids, batch_size, etc.), causing the token to become invalid.

【Current Parameter Snapshot】
- dataset_root: ${this.params.dataset_root}
- log_dir: ${this.params.log_dir}
- model_name: ${this.params.model_name}
- dyn_vars: ${this.params.dyn_vars?.join(', ')}
- in_t: ${this.params.in_t}
- out_t: ${this.params.out_t}
- stride: ${this.params.stride}
- epochs: ${this.params.epochs}
- lr: ${this.params.lr}
- batch_size: ${this.params.batch_size}
- device_ids: [${this.params.device_ids?.join(', ')}]
- distribute: ${this.params.distribute}
- use_amp: ${resolveUseAmp(this.params)} (not included in token signature)
- gradient_checkpointing: ${this.params.gradient_checkpointing}

================================================================================

Please re-present the above parameters to the user, obtain confirmation,
and re-invoke with the new token.

================================================================================`,
          canExecute: false,
          data: {
            error_type: 'token_invalid',
            expected_token: this.generateConfirmationToken(),
            provided_token: this.params.confirmation_token,
            current_params: {
              dataset_root: this.params.dataset_root,
              log_dir: this.params.log_dir,
              model_name: this.params.model_name,
              dyn_vars: this.params.dyn_vars,
              in_t: this.params.in_t,
              out_t: this.params.out_t,
              stride: this.params.stride,
              epochs: this.params.epochs,
              lr: this.params.lr,
              batch_size: this.params.batch_size,
              device_ids: this.params.device_ids,
              distribute: this.params.distribute,
              distribute_mode: this.params.distribute_mode,
              use_amp: resolveUseAmp(this.params),
              gradient_checkpointing: this.params.gradient_checkpointing,
            },
          },
        }

      case ForecastTrainingState.PASS:
        return {
          status: ForecastTrainingState.PASS,
          message:
            'All parameters confirmed, token validated, starting training...',
          canExecute: true,
        }

      default:
        return {
          status: ForecastTrainingState.ERROR,
          message: 'Unknown state',
          canExecute: false,
        }
    }
  }

  /**
   * Stage 1: Data confirmation prompt
   */
  private buildDataConfirmationPrompt(
    datasetInfo?: ForecastDatasetInfo
  ): ForecastStagePromptResult {
    if (!datasetInfo) {
      return {
        status: ForecastTrainingState.AWAITING_DATA_CONFIRMATION,
        message: `================================================================================
                    Please confirm data directory and output directory
================================================================================

**Please provide the following information:**

1. **dataset_root**: Preprocessed data root directory (ocean-forecast-data-preprocess output)
   - Should contain hr/ subdirectory with .npy time-step files
   - Optionally contains static/ subdirectory

2. **log_dir**: Training log output directory
   - Training logs, model weights, and config files will be saved here

================================================================================

Agent note: **Do NOT auto-guess data directories!**
Must wait for user to explicitly specify before proceeding.`,
        canExecute: false,
        data: { missing: ['dataset_root', 'log_dir'] },
      }
    }

    // Dataset info provided, show validation results
    const hasErrors = datasetInfo.errors.length > 0

    if (hasErrors) {
      return {
        status: ForecastTrainingState.AWAITING_DATA_CONFIRMATION,
        message: `================================================================================
                    Data directory validation failed
================================================================================

【Data Directory】${datasetInfo.dataset_root}

【Errors】
${datasetInfo.errors.map((e) => `  - ${e}`).join('\n')}

${datasetInfo.warnings.length > 0 ? `【Warnings】\n${datasetInfo.warnings.map((w) => `  - ${w}`).join('\n')}` : ''}

================================================================================

**Please check if the data directory is correct. You may need to run
data preprocessing first (ocean-forecast-data-preprocess).**

================================================================================

Agent note: Data validation failed, cannot proceed.
Inform the user of errors and wait for a new path.`,
        canExecute: false,
        data: {
          dataset_root: datasetInfo.dataset_root,
          errors: datasetInfo.errors,
          warnings: datasetInfo.warnings,
        },
      }
    }

    // Format split info
    const splitLines = Object.entries(datasetInfo.splits)
      .map(([split, count]) => `  - ${split}: ${count} samples`)
      .join('\n')

    // Format time range
    const timeRangeStr = datasetInfo.time_range
      ? `${datasetInfo.time_range.start} ~ ${datasetInfo.time_range.end}`
      : 'Not detected'

    return {
      status: ForecastTrainingState.AWAITING_DATA_CONFIRMATION,
      message: `Data directory validated successfully!

================================================================================
                    Dataset Information
================================================================================

【Basic Info】
- Data directory: ${datasetInfo.dataset_root}
- Log directory: ${this.params.log_dir}

【Detected Variables】
- Dynamic variables: ${datasetInfo.dyn_vars.join(', ')}
- Static variables: ${datasetInfo.has_static ? datasetInfo.static_vars.join(', ') : 'None'}

【Spatial Shape】
- Shape (H x W): ${datasetInfo.spatial_shape ? datasetInfo.spatial_shape.join(' x ') : 'Not detected'}

【Temporal Info】
- Total timesteps: ${datasetInfo.total_timesteps}
- Time range: ${timeRangeStr}

【Dataset Splits】
${splitLines}

${datasetInfo.warnings.length > 0 ? `【Warnings】\n${datasetInfo.warnings.map((w) => `  - ${w}`).join('\n')}\n` : ''}
================================================================================

Data validation passed, please proceed with model selection.
Agent may proceed to Stage 2 (model selection).`,
      canExecute: false,
      data: {
        dataset_root: datasetInfo.dataset_root,
        log_dir: this.params.log_dir,
        detected_dyn_vars: datasetInfo.dyn_vars,
        spatial_shape: datasetInfo.spatial_shape,
        total_timesteps: datasetInfo.total_timesteps,
        time_range: datasetInfo.time_range,
        splits: datasetInfo.splits,
        warnings: datasetInfo.warnings,
      },
    }
  }

  /**
   * Stage 2: Model selection prompt
   */
  private buildModelSelectionPrompt(
    datasetInfo?: ForecastDatasetInfo,
    modelList?: ModelInfo[]
  ): ForecastStagePromptResult {
    // Format model list
    let modelListStr =
      '(Model list failed to load, please call ocean_forecast_list_models to view)'
    if (modelList && modelList.length > 0) {
      const supportedModels = modelList.filter((m) => m.supported !== false)
      const unsupportedModels = modelList.filter((m) => m.supported === false)

      const formatGroup = (models: ModelInfo[]) =>
        models
          .map((m) => {
            const note = m.notes ? ' (' + m.notes + ')' : ''
            return '  - ' + m.name + ': ' + m.description + note
          })
          .join('\n')

      // Group by category
      const categories = new Map<string, ModelInfo[]>()
      for (const m of supportedModels) {
        const cat = m.category || 'other'
        if (!categories.has(cat)) categories.set(cat, [])
        categories.get(cat)!.push(m)
      }

      const sections: string[] = []
      for (const [cat, models] of categories) {
        sections.push(`【${cat}】`)
        sections.push(formatGroup(models))
        sections.push('')
      }

      modelListStr = sections.join('\n')

      if (unsupportedModels.length > 0) {
        modelListStr = [
          modelListStr,
          '【Not integrated / experimental】',
          formatGroup(unsupportedModels),
          '',
          'These models are not yet integrated into the training pipeline.',
        ].join('\n')
      }
    }

    return {
      status: ForecastTrainingState.AWAITING_MODEL_SELECTION,
      message: `Data confirmed:
- Data directory: ${this.params.dataset_root}
- Log directory: ${this.params.log_dir}
${
  datasetInfo
    ? `- Detected variables: ${datasetInfo.dyn_vars.join(', ')}
- Spatial shape: ${datasetInfo.spatial_shape?.join(' x ') ?? 'Unknown'}
- Timesteps: ${datasetInfo.total_timesteps}`
    : ''
}

================================================================================
                    Please select a training model
================================================================================

${modelListStr}

================================================================================

**Please answer the following:**

- **Which model to use for training?**
  Select a model name from the list above.

================================================================================

Agent note: **Do NOT auto-select a model!**
Must wait for user to explicitly specify before proceeding.`,
      canExecute: false,
      data: {
        dataset_root: this.params.dataset_root,
        log_dir: this.params.log_dir,
        detected_dyn_vars: datasetInfo?.dyn_vars,
        spatial_shape: datasetInfo?.spatial_shape,
        model_list: modelList,
      },
    }
  }

  /**
   * Stage 3: Training parameter confirmation prompt
   */
  private buildParametersPrompt(
    datasetInfo?: ForecastDatasetInfo,
    gpuInfo?: GpuInfo
  ): ForecastStagePromptResult {
    const { params } = this

    // GPU info
    let gpuStr = 'GPU info not available'
    if (gpuInfo) {
      if (!gpuInfo.cuda_available) {
        gpuStr = 'No GPU detected! Training requires GPU support.'
      } else {
        gpuStr = gpuInfo.gpus
          .map(
            (g) =>
              `  - GPU ${g.id}: ${g.name} (total ${g.total_memory_gb}GB / free ${g.free_memory_gb}GB / used ${g.used_memory_gb}GB)`
          )
          .join('\n')
      }
    }

    // Auto-detected values
    const detectedVars = datasetInfo?.dyn_vars || []

    // Current param values (show defaults where applicable)
    const currentInT = params.in_t ?? 7
    const currentOutT = params.out_t ?? 1
    const currentStride = params.stride ?? 1
    const currentEpochs = params.epochs ?? 500
    const currentLr = params.lr ?? 0.001
    const currentBatchSize = params.batch_size ?? 4
    const currentEvalBatchSize = params.eval_batch_size ?? 4
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

    // OOM protection
    const currentUseAmp = resolveUseAmp(params)
    const currentGradientCheckpointing = resolveGradientCheckpointing(params)

    return {
      status: ForecastTrainingState.AWAITING_PARAMETERS,
      message: `Model selected: ${params.model_name}

================================================================================
                    Please confirm training parameters
================================================================================

【Data Parameters】(auto-detected from data directory, please confirm)
- dyn_vars: ${detectedVars.length > 0 ? detectedVars.join(', ') : '? Not detected, please specify manually'}${params.dyn_vars ? ` -> Current: ${params.dyn_vars.join(', ')}` : ''}

【Forecast-specific Parameters】
- in_t: ${currentInT} (number of input timesteps)
- out_t: ${currentOutT} (number of output/prediction timesteps)
- stride: ${currentStride} (sliding window stride for sample generation)

【Core Training Parameters】
- epochs: ${currentEpochs} (training epochs)
- lr: ${currentLr} (learning rate)
- batch_size: ${currentBatchSize} (training batch size)
- eval_batch_size: ${currentEvalBatchSize} (evaluation batch size)
- patience: ${currentPatience} (early stopping patience)
- eval_freq: ${currentEvalFreq} (evaluation frequency, every N epochs)

【Optimizer Parameters】
- optimizer: ${currentOptimizer} (options: AdamW, Adam, SGD)
- weight_decay: ${currentWeightDecay}
- scheduler: ${currentScheduler} (options: StepLR, MultiStepLR, OneCycleLR)
- scheduler_step_size: ${currentSchedulerStepSize}
- scheduler_gamma: ${currentSchedulerGamma}

【Normalization Parameters】
- normalize: ${currentNormalize}
- normalizer_type: ${currentNormalizerType} (options: PGN, GN)

【GPU Configuration】
${gpuStr}

- device_ids: [${currentDeviceIds.join(', ')}] (GPUs to use)
- distribute: ${currentDistribute} (multi-GPU training)
- distribute_mode: ${currentDistributeMode} (mode: DP / DDP)
- master_port: ${currentDistribute && currentDistributeMode === 'DDP' ? (params.master_port ?? 'auto') : 'N/A'} (DDP communication port)

${currentDistribute && currentDeviceIds.length <= 1 ? 'Warning: device_ids has only 1 GPU, cannot use DDP/DP, will fallback to single GPU.' : ''}

${gpuInfo && gpuInfo.gpu_count > 1 ? `Tip: ${gpuInfo.gpu_count} GPUs detected, consider using multi-GPU DDP training for speedup.` : ''}

【Other Parameters】
- seed: ${currentSeed} (random seed)
- wandb: ${params.wandb ?? false} (enable WandB logging)
${params.ckpt_path ? `- ckpt_path: ${params.ckpt_path} (resume from checkpoint)` : ''}

【OOM Protection Parameters】
- use_amp: ${currentUseAmp} (AMP mixed precision, ~40-50% VRAM reduction; FFT models default off)
- gradient_checkpointing: ${currentGradientCheckpointing} (gradient checkpointing, ~60% activation memory reduction)

Tip: If running out of VRAM, try use_amp=true. FFT models may have cuFFT size issues.
     The system will auto-estimate VRAM and reduce batch_size if needed before training.

================================================================================

**Please confirm or modify the above parameters.**
- All parameters have defaults; if acceptable, reply "confirm"
- To modify, specify the parameter name and new value
- **Must confirm**: dyn_vars, device_ids

================================================================================

Agent note:
- If dyn_vars was auto-detected, present to user and ask for confirmation
- device_ids must be confirmed by the user
- **Do NOT auto-decide training parameters!** Wait for user confirmation.`,
      canExecute: false,
      data: {
        model_name: params.model_name,
        detected_dyn_vars: detectedVars,
        current_params: {
          dyn_vars: params.dyn_vars,
          in_t: currentInT,
          out_t: currentOutT,
          stride: currentStride,
          epochs: currentEpochs,
          lr: currentLr,
          batch_size: currentBatchSize,
          eval_batch_size: currentEvalBatchSize,
          device_ids: currentDeviceIds,
          distribute: currentDistribute,
          distribute_mode: currentDistributeMode,
          master_port: params.master_port,
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
          use_amp: currentUseAmp,
          gradient_checkpointing: currentGradientCheckpointing,
        },
        gpu_info: gpuInfo,
      },
    }
  }

  /**
   * Stage 4: Execution confirmation prompt
   */
  private buildExecutionPrompt(
    datasetInfo?: ForecastDatasetInfo,
    gpuInfo?: GpuInfo
  ): ForecastStagePromptResult {
    const { params } = this

    const confirmationToken = this.generateConfirmationToken()
    const effectiveUseAmp = resolveUseAmp(params)
    const effectiveGradientCheckpointing = resolveGradientCheckpointing(params)

    // GPU mode description
    const deviceIds = params.device_ids || [0]
    const distribute = params.distribute ?? false
    const distributeMode = params.distribute_mode ?? 'DDP'
    let gpuModeStr: string
    if (deviceIds.length === 1) {
      gpuModeStr = `Single GPU (GPU ${deviceIds[0]})`
    } else if (distribute && distributeMode === 'DDP') {
      gpuModeStr = `Multi-GPU DDP (GPU ${deviceIds.join(', ')})`
    } else {
      gpuModeStr = `Multi-GPU DP (GPU ${deviceIds.join(', ')})`
    }

    // GPU names
    let gpuNames = ''
    if (gpuInfo) {
      const selectedGpus = gpuInfo.gpus.filter((g) =>
        deviceIds.includes(g.id)
      )
      gpuNames = selectedGpus
        .map((g) => `${g.name} (${g.free_memory_gb}GB free)`)
        .join(', ')
    }

    return {
      status: ForecastTrainingState.AWAITING_EXECUTION,
      message: `All parameters confirmed. Please review and confirm execution:

================================================================================
                         Training Parameter Summary
================================================================================

【Data Info】
- Data directory: ${params.dataset_root}
- Log directory: ${params.log_dir}
- Dynamic variables: ${params.dyn_vars?.join(', ')}
${
  datasetInfo
    ? `- Spatial shape: ${datasetInfo.spatial_shape?.join(' x ') ?? '?'}
- Total timesteps: ${datasetInfo.total_timesteps}
- Time range: ${datasetInfo.time_range ? `${datasetInfo.time_range.start} ~ ${datasetInfo.time_range.end}` : '?'}`
    : ''
}

【Model Configuration】
- Model: ${params.model_name}
- Mode: ${params.mode ?? 'train'}

【Forecast Parameters】
- Input timesteps (in_t): ${params.in_t}
- Output timesteps (out_t): ${params.out_t}
- Stride: ${params.stride}

【Training Parameters】
- Epochs: ${params.epochs}
- Learning rate: ${params.lr}
- Batch size: ${params.batch_size}
- Eval batch size: ${params.eval_batch_size ?? 4}
- Early stopping patience: ${params.patience ?? 10}
- Eval frequency: every ${params.eval_freq ?? 5} epochs

【Optimizer】
- Optimizer: ${params.optimizer ?? 'AdamW'}
- Weight decay: ${params.weight_decay ?? 0.001}
- Scheduler: ${params.scheduler ?? 'StepLR'}

【GPU Configuration】
- Mode: ${gpuModeStr}
${gpuNames ? `- GPU: ${gpuNames}` : ''}
${distribute && distributeMode === 'DDP' ? `- master_port: ${params.master_port ?? 'auto'}` : ''}

【Other】
- Normalization: ${params.normalize ?? true} (${params.normalizer_type ?? 'PGN'})
- Random seed: ${params.seed ?? 42}
- WandB: ${params.wandb ?? false}
${params.ckpt_path ? `- Checkpoint resume: ${params.ckpt_path}` : ''}

【OOM Protection】
- AMP mixed precision: ${effectiveUseAmp}
- Gradient checkpointing: ${effectiveGradientCheckpointing}
- VRAM estimation: auto (auto-reduce batch_size if estimated > 85%)

================================================================================

Please confirm the above parameters are correct, then reply "confirm execution".

To modify any parameter, tell me what you want to change.

================================================================================

Execution Confirmation Token: ${confirmationToken}
(Agent must present the above summary to the user for confirmation,
and include this token with user_confirmed=true in the next invocation.)`,
      canExecute: false,
      data: {
        confirmation_token: confirmationToken,
        summary: {
          dataset_root: params.dataset_root,
          log_dir: params.log_dir,
          model_name: params.model_name,
          dyn_vars: params.dyn_vars,
          mode: params.mode,
          in_t: params.in_t,
          out_t: params.out_t,
          stride: params.stride,
          epochs: params.epochs,
          lr: params.lr,
          batch_size: params.batch_size,
          eval_batch_size: params.eval_batch_size,
          device_ids: params.device_ids,
          distribute: params.distribute,
          distribute_mode: params.distribute_mode,
          master_port: params.master_port,
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
          use_amp: effectiveUseAmp,
          gradient_checkpointing: effectiveGradientCheckpointing,
        },
      },
    }
  }
}
