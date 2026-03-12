/**
 * @file workflow-state.ts
 * @description Ocean forecast training workflow - 4-stage confirmation logic
 *              函数式实现，替代原 ForecastTrainingWorkflow 类。
 *              Token 机制：SHA-256 签名覆盖 13 个关键字段，参数变更后 token 失效。
 *
 * @author Leizheng
 * @contributors kongzhiquan
 * @date 2026-02-26
 * @version 2.1.0
 *
 * @changelog
 *   - 2026-03-12 kongzhiquan: v2.1.0 新增 missing_token 分支
 *     - resolveStage 中 user_confirmed=true 时先校验 confirmation_token 是否缺失
 *     - buildTokenInvalidPrompt 支持 missing_token / token_mismatch 两种提示
 *   - 2026-03-12 kongzhiquan: v2.0.0 重构为函数式，移除 ForecastTrainingWorkflow 类
 *     - 暴露 mergeParams() + resolveStage() 作为唯一公开 API
 *     - 阶段判断和提示构建内聚在同一文件
 *     - 保留 SHA-256 Token 签名机制，防止确认后参数被篡改
 *   - 2026-02-26 Leizheng: v1.1.0 expand token signature from 5→13 fields, protect device_ids etc.
 *   - 2026-02-26 Leizheng: v1.0.0 initial version for ocean forecast training
 */

import crypto from 'crypto'

// ============================================================
// Constants
// ============================================================

/** FFT-based models where AMP can cause numerical issues */
const AMP_AUTO_DISABLE_MODELS = new Set(['FNO2d', 'M2NO2d'])

/** Heavy models that benefit from gradient checkpointing by default */
const HEAVY_MODELS = new Set([
  'GalerkinTransformer',
  'SwinTransformerV2',
  'SwinMLP',
])

const TOKEN_SALT = 'ocean-forecast-training-v2'

// ============================================================
// Types
// ============================================================

export type ForecastTrainingStateType =
  | 'awaiting_data_confirmation'
  | 'awaiting_model_selection'
  | 'awaiting_parameters'
  | 'awaiting_execution'
  | 'token_invalid'

export interface ForecastTrainingWorkflowParams {
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

export interface ForecastTrainingStagePromptResult {
  status: ForecastTrainingStateType
  message: string
  canExecute: boolean
  data?: any
}

export interface ForecastDatasetInfo {
  status: string
  dataset_root: string
  dyn_vars: string[]
  spatial_shape: [number, number] | null
  splits: Record<string, number>
  total_timesteps: number
  time_range: { start: string; end: string } | null
  has_static: boolean
  static_vars: string[]
  warnings: string[]
  errors: string[]
}

export interface ForecastGpuInfo {
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

export interface ForecastModelInfo {
  name: string
  category: string
  trainer: string
  description: string
  supported?: boolean
  notes?: string
}

// ============================================================
// Helper functions
// ============================================================

function resolveGradientCheckpointing(params: ForecastTrainingWorkflowParams): boolean {
  if (params.gradient_checkpointing !== undefined) {
    return Boolean(params.gradient_checkpointing)
  }
  return params.model_name ? HEAVY_MODELS.has(params.model_name) : false
}

function resolveUseAmp(params: ForecastTrainingWorkflowParams): boolean {
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
function filterDefined<T extends Record<string, any>>(obj: T): Partial<T> {
  return Object.fromEntries(
    Object.entries(obj).filter(([, v]) => v !== undefined)
  ) as Partial<T>
}

// ============================================================
// Stage check predicates
// ============================================================

function hasDataParams(p: ForecastTrainingWorkflowParams): boolean {
  return !!(p.dataset_root && p.log_dir)
}

function hasModelParam(p: ForecastTrainingWorkflowParams): boolean {
  return !!p.model_name
}

function hasTrainingParams(p: ForecastTrainingWorkflowParams): boolean {
  return !!(
    p.dyn_vars &&
    p.dyn_vars.length > 0 &&
    p.epochs !== undefined &&
    p.epochs > 0 &&
    p.lr !== undefined &&
    p.lr > 0 &&
    p.batch_size !== undefined &&
    p.batch_size > 0 &&
    p.device_ids &&
    p.device_ids.length > 0
  )
}

function hasAllRequiredParams(p: ForecastTrainingWorkflowParams): boolean {
  return hasDataParams(p) && hasModelParam(p) && hasTrainingParams(p)
}

function getMissingTrainingParams(p: ForecastTrainingWorkflowParams): string[] {
  const missing: string[] = []
  if (!p.dyn_vars || p.dyn_vars.length === 0) missing.push('dyn_vars')
  if (p.epochs === undefined || p.epochs <= 0) missing.push('epochs')
  if (p.lr === undefined || p.lr <= 0) missing.push('lr')
  if (p.batch_size === undefined || p.batch_size <= 0) missing.push('batch_size')
  if (!p.device_ids || p.device_ids.length === 0) missing.push('device_ids')
  return missing
}

// ============================================================
// Token generation / validation
// ============================================================

/**
 * Generate SHA-256 confirmation token.
 * Signature covers 13 fields: dataset_root, model_name, in_t, out_t,
 * batch_size, dyn_vars, stride, epochs, lr, device_ids, distribute,
 * distribute_mode, gradient_checkpointing.
 */
export function generateConfirmationToken(params: ForecastTrainingWorkflowParams): string {
  const tokenData = {
    dataset_root: params.dataset_root,
    model_name: params.model_name,
    in_t: params.in_t,
    out_t: params.out_t,
    batch_size: params.batch_size,
    dyn_vars: params.dyn_vars ? [...params.dyn_vars].sort().join(',') : '',
    stride: params.stride,
    epochs: params.epochs,
    lr: params.lr,
    device_ids: params.device_ids ? params.device_ids.join(',') : '',
    distribute: params.distribute,
    distribute_mode: params.distribute_mode,
    gradient_checkpointing: params.gradient_checkpointing,
  }
  const dataStr = JSON.stringify(tokenData) + TOKEN_SALT
  return crypto
    .createHash('sha256')
    .update(dataStr)
    .digest('hex')
    .substring(0, 16)
}

function validateConfirmationToken(params: ForecastTrainingWorkflowParams): boolean {
  if (!params.confirmation_token) return false
  return params.confirmation_token === generateConfirmationToken(params)
}

// ============================================================
// Prompt builders
// ============================================================

function buildDataConfirmationPrompt(
  params: ForecastTrainingWorkflowParams,
  datasetInfo?: ForecastDatasetInfo
): ForecastTrainingStagePromptResult {
  if (!datasetInfo) {
    return {
      status: 'awaiting_data_confirmation',
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

  const hasErrors = datasetInfo.errors.length > 0
  if (hasErrors) {
    return {
      status: 'awaiting_data_confirmation',
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

  const splitLines = Object.entries(datasetInfo.splits)
    .map(([split, count]) => `  - ${split}: ${count} samples`)
    .join('\n')

  const timeRangeStr = datasetInfo.time_range
    ? `${datasetInfo.time_range.start} ~ ${datasetInfo.time_range.end}`
    : 'Not detected'

  return {
    status: 'awaiting_data_confirmation',
    message: `Data directory validated successfully!

================================================================================
                    Dataset Information
================================================================================

【Basic Info】
- Data directory: ${datasetInfo.dataset_root}
- Log directory: ${params.log_dir}

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
      log_dir: params.log_dir,
      detected_dyn_vars: datasetInfo.dyn_vars,
      spatial_shape: datasetInfo.spatial_shape,
      total_timesteps: datasetInfo.total_timesteps,
      time_range: datasetInfo.time_range,
      splits: datasetInfo.splits,
      warnings: datasetInfo.warnings,
    },
  }
}

function buildModelSelectionPrompt(
  params: ForecastTrainingWorkflowParams,
  datasetInfo?: ForecastDatasetInfo,
  modelList?: ForecastModelInfo[]
): ForecastTrainingStagePromptResult {
  let modelListStr =
    '(Model list failed to load, please call ocean_forecast_list_models to view)'
  if (modelList && modelList.length > 0) {
    const supportedModels = modelList.filter((m) => m.supported !== false)
    const unsupportedModels = modelList.filter((m) => m.supported === false)

    const formatGroup = (models: ForecastModelInfo[]) =>
      models
        .map((m) => {
          const note = m.notes ? ' (' + m.notes + ')' : ''
          return '  - ' + m.name + ': ' + m.description + note
        })
        .join('\n')

    const categories = new Map<string, ForecastModelInfo[]>()
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
    status: 'awaiting_model_selection',
    message: `Data confirmed:
- Data directory: ${params.dataset_root}
- Log directory: ${params.log_dir}
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
      dataset_root: params.dataset_root,
      log_dir: params.log_dir,
      detected_dyn_vars: datasetInfo?.dyn_vars,
      spatial_shape: datasetInfo?.spatial_shape,
      model_list: modelList,
    },
  }
}

function buildParametersPrompt(
  params: ForecastTrainingWorkflowParams,
  datasetInfo?: ForecastDatasetInfo,
  gpuInfo?: ForecastGpuInfo
): ForecastTrainingStagePromptResult {
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

  const detectedVars = datasetInfo?.dyn_vars || []

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

  const currentUseAmp = resolveUseAmp(params)
  const currentGradientCheckpointing = resolveGradientCheckpointing(params)

  return {
    status: 'awaiting_parameters',
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

function buildExecutionPrompt(
  params: ForecastTrainingWorkflowParams,
  datasetInfo?: ForecastDatasetInfo,
  gpuInfo?: ForecastGpuInfo
): ForecastTrainingStagePromptResult {
  const confirmationToken = generateConfirmationToken(params)
  const effectiveUseAmp = resolveUseAmp(params)
  const effectiveGradientCheckpointing = resolveGradientCheckpointing(params)

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

  let gpuNames = ''
  if (gpuInfo) {
    const selectedGpus = gpuInfo.gpus.filter((g) => deviceIds.includes(g.id))
    gpuNames = selectedGpus
      .map((g) => `${g.name} (${g.free_memory_gb}GB free)`)
      .join(', ')
  }

  return {
    status: 'awaiting_execution',
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

function buildTokenInvalidPrompt(
  params: ForecastTrainingWorkflowParams,
  mode: 'missing_token' | 'token_mismatch'
): ForecastTrainingStagePromptResult {
  if (mode === 'missing_token') {
    return {
      status: 'token_invalid',
      message: `================================================================================
                    Token validation failed - missing confirmation token
================================================================================

Detected step-skipping behavior:
user_confirmed=true was provided, but confirmation_token is missing.

Required flow:
1. Call tool without user_confirmed to enter awaiting_execution stage
2. Get confirmation_token from the stage result
3. After user confirms, call again with user_confirmed=true and confirmation_token

================================================================================`,
      canExecute: false,
      data: {
        error_type: 'token_invalid',
        reason: 'missing_confirmation_token',
      },
    }
  }

  return {
    status: 'token_invalid',
    message: `================================================================================
                    Token validation failed - parameters modified
================================================================================

Parameters were modified after user confirmation (possibly Agent auto-adjusted
device_ids, batch_size, etc.), causing the token to become invalid.

【Current Parameter Snapshot】
- dataset_root: ${params.dataset_root}
- log_dir: ${params.log_dir}
- model_name: ${params.model_name}
- dyn_vars: ${params.dyn_vars?.join(', ')}
- in_t: ${params.in_t}
- out_t: ${params.out_t}
- stride: ${params.stride}
- epochs: ${params.epochs}
- lr: ${params.lr}
- batch_size: ${params.batch_size}
- device_ids: [${params.device_ids?.join(', ')}]
- distribute: ${params.distribute}
- use_amp: ${resolveUseAmp(params)} (not included in token signature)
- gradient_checkpointing: ${params.gradient_checkpointing}

================================================================================

Please re-present the above parameters to the user, obtain confirmation,
and re-invoke with the new token.

================================================================================`,
    canExecute: false,
    data: {
      error_type: 'token_invalid',
      expected_token: generateConfirmationToken(params),
      provided_token: params.confirmation_token,
      current_params: {
        dataset_root: params.dataset_root,
        log_dir: params.log_dir,
        model_name: params.model_name,
        dyn_vars: params.dyn_vars,
        in_t: params.in_t,
        out_t: params.out_t,
        stride: params.stride,
        epochs: params.epochs,
        lr: params.lr,
        batch_size: params.batch_size,
        device_ids: params.device_ids,
        distribute: params.distribute,
        distribute_mode: params.distribute_mode,
        use_amp: resolveUseAmp(params),
        gradient_checkpointing: params.gradient_checkpointing,
      },
    },
  }
}

// ============================================================
// Public API
// ============================================================

/**
 * 合并参数：system defaults → session confirmed → explicitly supplied（过滤 undefined）。
 * 返回完整的合并后参数，可直接用于阶段判断和最终执行。
 */
export function mergeParams(
  args: ForecastTrainingWorkflowParams,
  sessionOverrides?: ForecastTrainingWorkflowParams
): ForecastTrainingWorkflowParams {
  const definedArgs = filterDefined(args)

  return {
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
 * 根据参数确定当前所处阶段并返回对应提示。
 *
 * @param params   合并后的完整参数（mergeParams 的输出）
 * @param context  各阶段所需的上下文数据
 * @returns ForecastStagePromptResult（仍在某阶段）或 null（所有阶段通过，可继续执行）
 */
export function resolveStage(
  params: ForecastTrainingWorkflowParams,
  context?: {
    datasetInfo?: ForecastDatasetInfo
    gpuInfo?: ForecastGpuInfo
    modelList?: ForecastModelInfo[]
  }
): ForecastTrainingStagePromptResult | null {
  // ========== PASS check: user_confirmed + all params + token valid ==========
  if (params.user_confirmed === true && hasAllRequiredParams(params)) {
    if (!params.confirmation_token) {
      return buildTokenInvalidPrompt(params, 'missing_token')
    }
    if (!validateConfirmationToken(params)) {
      return buildTokenInvalidPrompt(params, 'token_mismatch')
    }
    // All checks passed → return null to signal execution
    return null
  }

  // ========== Stage 1: Data confirmation ==========
  if (!hasDataParams(params)) {
    return buildDataConfirmationPrompt(params, context?.datasetInfo)
  }

  // ========== Stage 2: Model selection ==========
  if (!hasModelParam(params)) {
    return buildModelSelectionPrompt(params, context?.datasetInfo, context?.modelList)
  }

  // ========== Stage 3: Training parameters ==========
  if (!hasTrainingParams(params)) {
    return buildParametersPrompt(params, context?.datasetInfo, context?.gpuInfo)
  }

  // ========== Stage 4: Awaiting execution confirmation ==========
  return buildExecutionPrompt(params, context?.datasetInfo, context?.gpuInfo)
}
