/**
 * @file workflow-state.ts
 * @description Ocean forecast training workflow - 4-stage confirmation logic
 *              函数式实现，替代原 ForecastTrainingWorkflow 类。
 *              Token 机制：SHA-256 签名覆盖 13 个关键字段，参数变更后 token 失效。
 *
 * @author Leizheng
 * @contributors kongzhiquan
 * @date 2026-02-26
 * @version 2.1.2
 *
 * @changelog
 *   - 2026-03-12 kongzhiquan: v2.1.1 所有提示词改为中文
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

export type ForecastTrainingWorkflowState =
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
  status: ForecastTrainingWorkflowState
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
            ⚠️ 请确认数据目录和输出目录
================================================================================

**请提供以下信息：**

1️⃣ **dataset_root**：预处理后的数据根目录（ocean-forecast-data-preprocess 的输出）
   - 应包含 hr/ 子目录，存放 .npy 格式的时间步文件
   - 可选包含 static/ 子目录

2️⃣ **log_dir**：训练日志输出目录
   - 训练日志、模型权重和配置文件将保存于此

================================================================================

⚠️ Agent 注意：**禁止自动猜测数据目录！**
必须等待用户明确指定后再继续。`,
      canExecute: false,
      data: { missing: ['dataset_root', 'log_dir'] },
    }
  }

  const hasErrors = datasetInfo.errors.length > 0
  if (hasErrors) {
    return {
      status: 'awaiting_data_confirmation',
      message: `================================================================================
            ❌ 数据目录验证失败
================================================================================

【数据目录】${datasetInfo.dataset_root}

【错误】
${datasetInfo.errors.map((e) => `  ❌ ${e}`).join('\n')}

${datasetInfo.warnings.length > 0 ? `【警告】\n${datasetInfo.warnings.map((w) => `  ⚠️ ${w}`).join('\n')}` : ''}

================================================================================

**请检查数据目录是否正确。可能需要先运行数据预处理（ocean-forecast-data-preprocess）。**

================================================================================

⚠️ Agent 注意：数据验证失败，无法继续。
请将错误信息告知用户，并等待用户提供新路径。`,
      canExecute: false,
      data: {
        dataset_root: datasetInfo.dataset_root,
        errors: datasetInfo.errors,
        warnings: datasetInfo.warnings,
      },
    }
  }

  const splitLines = Object.entries(datasetInfo.splits)
    .map(([split, count]) => `  - ${split}: ${count} 个样本`)
    .join('\n')

  const timeRangeStr = datasetInfo.time_range
    ? `${datasetInfo.time_range.start} ~ ${datasetInfo.time_range.end}`
    : '未检测到'

  return {
    status: 'awaiting_data_confirmation',
    message: `数据目录验证通过！

================================================================================
            📊 数据集信息
================================================================================

【基本信息】
- 数据目录：${datasetInfo.dataset_root}
- 日志目录：${params.log_dir}

【检测到的变量】
- 动态变量：${datasetInfo.dyn_vars.join(', ')}
- 静态变量：${datasetInfo.has_static ? datasetInfo.static_vars.join(', ') : '无'}

【空间尺寸】
- 形状（H x W）：${datasetInfo.spatial_shape ? datasetInfo.spatial_shape.join(' x ') : '未检测到'}

【时间信息】
- 总时间步数：${datasetInfo.total_timesteps}
- 时间范围：${timeRangeStr}

【数据集划分】
${splitLines}

${datasetInfo.warnings.length > 0 ? `【警告】\n${datasetInfo.warnings.map((w) => `  ⚠️ ${w}`).join('\n')}\n` : ''}
================================================================================

数据验证通过，请继续进行模型选择。
Agent 可以进入下一阶段（阶段2：模型选择）。`,
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
    '（模型列表加载失败，请调用 ocean_forecast_list_models 查看）'
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
        '【未集成 / 实验性】',
        formatGroup(unsupportedModels),
        '',
        '这些模型尚未集成到训练流水线中。',
      ].join('\n')
    }
  }

  return {
    status: 'awaiting_model_selection',
    message: `数据确认完成：
- 数据目录：${params.dataset_root}
- 日志目录：${params.log_dir}
${
  datasetInfo
    ? `- 检测到的变量：${datasetInfo.dyn_vars.join(', ')}
- 空间尺寸：${datasetInfo.spatial_shape?.join(' x ') ?? '未知'}
- 时间步数：${datasetInfo.total_timesteps}`
    : ''
}

================================================================================
                    ⚠️ 请选择训练模型
================================================================================

${modelListStr}

================================================================================

**请回答以下问题：**

🔹 **选择哪个模型进行训练？**
  请从上面的模型列表中选择一个模型名称。

  💡 推荐：
  - 初次尝试推荐 SwinTransformerV2（平衡性能和效果）
  - 时间建模可先试 OneForecast / FNO2d
  - 显存敏感场景可优先考虑轻量模型

================================================================================

⚠️ Agent 注意：**禁止自动选择模型！**
必须等待用户明确指定后再继续。`,
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
  let gpuStr = 'GPU 信息不可用'
  if (gpuInfo) {
    if (!gpuInfo.cuda_available) {
      gpuStr = '⚠️ 未检测到可用 GPU！训练需要 GPU 支持。'
    } else {
      gpuStr = gpuInfo.gpus
        .map(
          (g) =>
            `  - GPU ${g.id}: ${g.name} (总显存 ${g.total_memory_gb}GB / 空闲 ${g.free_memory_gb}GB / 已用 ${g.used_memory_gb}GB)`
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
    message: `模型已选择：${params.model_name}

================================================================================
            ⚠️ 请确认训练参数
================================================================================

【数据参数】（自动从数据目录检测，请确认）
- dyn_vars: ${detectedVars.length > 0 ? detectedVars.join(', ') : '❓ 未检测到，请手动指定'}${params.dyn_vars ? ` ✅ 当前: ${params.dyn_vars.join(', ')}` : ''}

【预测专用参数】
- in_t: ${currentInT} （输入时间步数）
- out_t: ${currentOutT} （输出/预测时间步数）
- stride: ${currentStride} （样本生成的滑动窗口步长）

【核心训练参数】
- epochs: ${currentEpochs} （训练轮数）
- lr: ${currentLr} （学习率）
- batch_size: ${currentBatchSize} （训练批大小）
- eval_batch_size: ${currentEvalBatchSize} （评估批大小）
- patience: ${currentPatience} （早停耐心值）
- eval_freq: ${currentEvalFreq} （评估频率，每 N 轮评估一次）

【优化器参数】
- optimizer: ${currentOptimizer} （可选：AdamW、Adam、SGD）
- weight_decay: ${currentWeightDecay}
- scheduler: ${currentScheduler} （可选：StepLR、MultiStepLR、OneCycleLR）
- scheduler_step_size: ${currentSchedulerStepSize}
- scheduler_gamma: ${currentSchedulerGamma}

【归一化参数】
- normalize: ${currentNormalize}
- normalizer_type: ${currentNormalizerType} （可选：PGN、GN）

【GPU 配置】
${gpuStr}

- device_ids: [${currentDeviceIds.join(', ')}] （使用的 GPU）
- distribute: ${currentDistribute} （多 GPU 训练）
- distribute_mode: ${currentDistributeMode} （模式：DP / DDP）
- master_port: ${currentDistribute && currentDistributeMode === 'DDP' ? (params.master_port ?? '自动') : 'N/A'} （DDP 通信端口）

${currentDistribute && currentDeviceIds.length <= 1 ? '⚠️ device_ids 只有 1 张 GPU 时无法使用 DDP/DP，将自动降级为单卡。' : ''}

${gpuInfo && gpuInfo.gpu_count > 1 ? `💡 检测到 ${gpuInfo.gpu_count} 张 GPU，建议使用多卡 DDP 训练以加速。` : ''}

【其他参数】
- seed: ${currentSeed} （随机种子）
- wandb: ${params.wandb ?? false} （启用 WandB 日志）
${params.ckpt_path ? `- ckpt_path: ${params.ckpt_path} （从检查点恢复）` : ''}

【OOM 保护参数】
- use_amp: ${currentUseAmp} （AMP 混合精度，约降低 40-50% 显存；FFT 模型默认关闭）
- gradient_checkpointing: ${currentGradientCheckpointing} （梯度检查点，约减少 60% 激活内存）

💡 显存不足时可尝试 use_amp=true。FFT 模型可能存在 cuFFT size 问题。
      系统将在训练前自动估算显存，并在需要时自动降低 batch_size。

================================================================================

**请确认或修改以上参数。**
- 所有参数均有默认值，如无异议请回复"确认"
- 如需修改，请指定参数名称和新值
- **必须确认**：dyn_vars、device_ids

================================================================================

⚠️ Agent 注意：
- 如果 dyn_vars 是自动检测的，请向用户展示并请求确认
- device_ids 必须由用户确认
- **禁止自动决定训练参数！**必须等待用户确认后再继续。`,
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
    gpuModeStr = `单 GPU（GPU ${deviceIds[0]}）`
  } else if (distribute && distributeMode === 'DDP') {
    gpuModeStr = `多 GPU DDP（GPU ${deviceIds.join(', ')}）`
  } else {
    gpuModeStr = `多 GPU DP（GPU ${deviceIds.join(', ')}）`
  }

  let gpuNames = ''
  if (gpuInfo) {
    const selectedGpus = gpuInfo.gpus.filter((g) => deviceIds.includes(g.id))
    gpuNames = selectedGpus
      .map((g) => `${g.name}（空闲 ${g.free_memory_gb}GB）`)
      .join(', ')
  }

  return {
    status: 'awaiting_execution',
    message: `所有参数已确认，请检查后确认执行：

================================================================================
               📋 训练参数汇总
================================================================================

【数据信息】
- 数据目录：${params.dataset_root}
- 日志目录：${params.log_dir}
- 动态变量：${params.dyn_vars?.join(', ')}
${
  datasetInfo
    ? `- 空间尺寸：${datasetInfo.spatial_shape?.join(' x ') ?? '?'}
- 总时间步数：${datasetInfo.total_timesteps}
- 时间范围：${datasetInfo.time_range ? `${datasetInfo.time_range.start} ~ ${datasetInfo.time_range.end}` : '?'}`
    : ''
}

【模型配置】
- 模型：${params.model_name}
- 模式：${params.mode ?? 'train'}

【预测参数】
- 输入时间步数（in_t）：${params.in_t}
- 输出时间步数（out_t）：${params.out_t}
- 步长（stride）：${params.stride}

【训练参数】
- 训练轮数（epochs）：${params.epochs}
- 学习率（lr）：${params.lr}
- 训练批大小（batch_size）：${params.batch_size}
- 评估批大小（eval_batch_size）：${params.eval_batch_size ?? 4}
- 早停耐心值（patience）：${params.patience ?? 10}
- 评估频率：每 ${params.eval_freq ?? 5} 轮

【优化器】
- 优化器：${params.optimizer ?? 'AdamW'}
- 权重衰减：${params.weight_decay ?? 0.001}
- 调度器：${params.scheduler ?? 'StepLR'}

【GPU 配置】
- 模式：${gpuModeStr}
${gpuNames ? `- GPU：${gpuNames}` : ''}
${distribute && distributeMode === 'DDP' ? `- master_port：${params.master_port ?? '自动'}` : ''}

【其他】
- 归一化：${params.normalize ?? true}（${params.normalizer_type ?? 'PGN'}）
- 随机种子：${params.seed ?? 42}
- WandB：${params.wandb ?? false}
${params.ckpt_path ? `- 从检查点恢复：${params.ckpt_path}` : ''}

【OOM 保护】
- AMP 混合精度：${effectiveUseAmp}
- 梯度检查点：${effectiveGradientCheckpointing}
- 显存估算：自动（估算超过 85% 时自动降低 batch_size）

================================================================================

⚠️ **请确认以上参数无误后，回复"确认执行"**

如需修改任何参数，请告知具体修改内容。

================================================================================

🔐 **执行确认 Token**：${confirmationToken}
（Agent 必须将上述汇总展示给用户确认，并在下次调用时附带此 token 和 user_confirmed=true。）`,
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
            ⚠️ Token 校验失败 — 缺少确认令牌
================================================================================

检测到跳步行为：
提供了 user_confirmed=true，但 confirmation_token 缺失。

必须按以下流程执行：
1. 不携带 user_confirmed 调用工具，进入 awaiting_execution 阶段
2. 从阶段结果中获取 confirmation_token
3. 用户确认后，携带 user_confirmed=true 和 confirmation_token 再次调用

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
            ⚠️ Token 校验失败 — 参数已被修改
================================================================================

用户确认后参数被修改（可能是 Agent 自动调整了 device_ids、batch_size 等），
导致 Token 失效。

【当前参数快照】
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
- use_amp: ${resolveUseAmp(params)}（不计入 Token 签名）
- gradient_checkpointing: ${params.gradient_checkpointing}

================================================================================

请将上述参数重新展示给用户，获取确认后携带新 Token 再次调用。

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
