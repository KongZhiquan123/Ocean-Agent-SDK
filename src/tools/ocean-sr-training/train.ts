/**
 * @file train.ts
 *
 * @description 海洋超分辨率模型训练工具
 *              集成状态机实现分阶段确认流程
 *              支持后台执行和实时日志流
 * @author Leizheng
 * @contributors kongzhiquan
 * @date 2026-02-09
 * @version 4.2.7
 *
 * @changelog
 *   - 2026-02-09 Leizheng: v4.2.7 FFT 模型 AMP 默认策略修复 + 模型支持性预检
 *   - 2026-02-09 Leizheng: v4.2.6 默认 batch_size 下调为 4 + 默认开启 gradient_checkpointing
 *   - 2026-02-09 Leizheng: v4.2.5 训练前输出尺寸预检 + 默认 batch_size 下调
 *     - 训练启动前检查模型输出尺寸是否与 HR 匹配
 *     - batch_size/eval_batch_size 默认降为 16
 *   - 2026-02-09 Leizheng: v4.2.4 FFT + AMP 推荐 patch_size 提示
 *   - 2026-02-09 Leizheng: v4.2.3 FFT 模型 AMP 默认策略
 *     - FFT 模型默认关闭 AMP（用户显式开启则尊重并提示风险）
 *   - 2026-02-09 Leizheng: v4.2.2 FFT 提示策略调整 + 显存预警
 *     - FFT 预检扩展到所有模型，改为提示不拦截
 *     - 新增 GPU 低空闲显存提示（不拦截）
 *   - 2026-02-09 Leizheng: v4.2.1 FFT + AMP 兼容性预检扩展
 *     - 适配 HiNOTE 等 FFT 模型，统一前置拦截
 *     - 提示与错误信息统一为 FFT 维度要求
 *   - 2026-02-09 Leizheng: v4.2.0 FNO + AMP 兼容性预检
 *     - 训练前检测 FFT 输入尺寸是否为 2 的幂
 *     - 在阶段提示中追加明确告警，避免运行时 cuFFT 崩溃
 *   - 2026-02-08 Leizheng: v4.1.0 修复参数传递 + 显式白名单
 *     - configParams 改用显式白名单，避免 restParams 泄漏状态机内部字段
 *     - 显式传递 ckpt_path（之前被解构后从 restParams 中丢失）
 *   - 2026-02-07 kongzhiquan: v4.0.0 OOM 自动防护 + 事件驱动启动监控
 *     - 显存预估改为自动循环调参（AMP→减batch_size→报错），不可跳过
 *     - 移除 skip_memory_check 参数，use_amp 默认改为 true
 *     - 启动后等待 training_start 事件（最长 5 分钟），捕获早期崩溃
 *     - 启动阶段崩溃时直接返回 error_summary + suggestions
 *   - 2026-02-07 kongzhiquan: v3.0.0 后台执行模式
 *     - 使用 TrainingProcessManager 启动后台训练进程
 *     - 训练启动后立即返回 process_id，不再阻塞等待
 *     - 支持实时日志流（通过 ocean_sr_train_status 工具查询）
 *     - 服务器关闭时自动清理训练进程
 *   - 2026-02-07 Leizheng: v3.0.0 OOM 防护三件套
 *     - 新增 use_amp / gradient_checkpointing / patch_size 参数
 *     - 训练前自动运行显存预估，OOM 提前拦截并给出建议
 *     - 新参数通过 generate_config.py 写入 YAML 配置
 *   - 2026-02-07 Leizheng: v2.3.0 按模型选择性复制代码到用户输出目录执行，保持 SDK 源码不被修改
 *   - 2026-02-07 Leizheng: v2.2.0 使用 findPythonWithModule('torch') 自动查找带 PyTorch 的 Python
 *   - 2026-02-06 Leizheng: v2.1.0 指向 masked 版本训练框架
 *     - trainingDir 改为 scripts/ocean_SR_training_masked
 *   - 2026-02-06 Leizheng: v2.0.0 集成训练工作流状态机
 *     - 4 阶段确认: 数据 → 模型 → 参数(GPU) → 执行
 *     - 自动检测 dyn_vars / scale / shape
 *     - GPU 信息集成到参数确认阶段
 *     - Token 防跳步机制
 *   - 2026-02-06 Leizheng: v1.0.0 初始版本
 *     - 支持单卡/多卡(DP/DDP)训练
 *     - 自动生成 YAML 配置文件
 *     - 支持 train/test 两种模式
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findPythonWithModule, findFirstPythonPath } from '@/utils/python-manager'
import { trainingProcessManager } from '@/utils/training-process-manager'
import path from 'node:path'
import {
  TrainingWorkflow,
  TrainingState,
  type DatasetValidationInfo,
  type GpuInfo,
  type ModelInfo,
} from './workflow-state'

/**
 * 将 JSON 字符串转义为可以安全嵌入 shell 单引号的形式
 * 策略：替换单引号为 '\'' (结束单引号 + 转义单引号 + 开始单引号)
 */
function shellSafeJson(json: string): string {
  return json.replace(/'/g, "'\\''")
}

function extractTaggedJson(output: string, tag: string): Record<string, unknown> | null {
  const pattern = new RegExp(`__${tag}__([\\s\\S]*?)__${tag}__`)
  const match = output.match(pattern)
  if (!match) return null
  try {
    return JSON.parse(match[1])
  } catch {
    return null
  }
}

// FFT/频域/复数变换相关模型：默认关闭 AMP，允许用户显式 override（强提示）
const AMP_AUTO_DISABLE_MODELS = new Set([
  'FNO2d',
  'HiNOTE',
  'MWT2d',
  'M2NO2d',
  'MG-DDPM',
])
const DIFFUSION_MODELS = new Set(['DDPM', 'SR3', 'MG-DDPM', 'ReMiG', 'ResShift', 'Resshift'])

function isFftSensitiveModel(modelName?: string): boolean {
  return Boolean(modelName && AMP_AUTO_DISABLE_MODELS.has(modelName))
}

function isPowerOfTwo(value: number): boolean {
  return Number.isInteger(value) && value > 0 && (value & (value - 1)) === 0
}

function countPowerOfTwoFactor(value: number): number {
  let v = Math.abs(Math.trunc(value))
  let count = 0
  while (v > 0 && v % 2 === 0) {
    v = v / 2
    count += 1
  }
  return count
}

function getModelDivisor(modelName?: string): number {
  if (!modelName) return 1
  if (modelName === 'Resshift' || modelName === 'ResShift') return 8
  if (DIFFUSION_MODELS.has(modelName)) return 32
  if (modelName === 'UNet2d') return 16
  return 1
}

function getMaxHrDim(datasetInfo?: DatasetValidationInfo): number | null {
  const hrDims = getSpatialDims(datasetInfo?.hr_shape)
  if (!hrDims) return null
  return Math.min(hrDims[0], hrDims[1])
}

function buildFftPatchRecommendations(params: {
  model_name?: string
  scale?: number
  patch_size?: number | null
  datasetInfo?: DatasetValidationInfo
}): {
  patch_sizes: number[]
  lr_sizes: number[]
  divisor: number
  max_dim?: number
  reason?: string
} {
  const scale = params.scale ?? params.datasetInfo?.scale ?? null
  const divisor = getModelDivisor(params.model_name)
  if (!scale || !Number.isFinite(scale) || scale <= 0) {
    return {
      patch_sizes: [],
      lr_sizes: [],
      divisor,
      reason: '缺少有效的 scale',
    }
  }

  const maxDim = getMaxHrDim(params.datasetInfo)
  if (maxDim !== null && maxDim < scale) {
    return {
      patch_sizes: [],
      lr_sizes: [],
      divisor,
      max_dim: maxDim,
      reason: `HR 尺寸过小（${maxDim} < scale ${scale}）`,
    }
  }

  let minK = 0
  if (divisor > 1 && isPowerOfTwo(divisor)) {
    const divisorPow = Math.log2(divisor)
    const scalePow = countPowerOfTwoFactor(scale)
    minK = Math.max(0, Math.ceil(divisorPow - scalePow))
  }

  let maxK = minK + 8
  if (maxDim !== null) {
    const ratio = maxDim / scale
    if (ratio >= 1) {
      maxK = Math.floor(Math.log2(ratio))
    } else {
      maxK = minK - 1
    }
  }

  if (maxK < minK) {
    return {
      patch_sizes: [],
      lr_sizes: [],
      divisor,
      max_dim: maxDim ?? undefined,
      reason: 'HR 尺寸不足以满足 2 的幂与整除要求',
    }
  }

  const candidates: number[] = []
  for (let k = minK; k <= maxK; k += 1) {
    const lrSize = Math.pow(2, k)
    const patchSize = scale * lrSize
    if (!Number.isFinite(patchSize)) continue
    if (divisor > 1 && patchSize % divisor !== 0) continue
    if (maxDim !== null && patchSize > maxDim) break
    candidates.push(patchSize)
  }

  if (candidates.length === 0) {
    return {
      patch_sizes: [],
      lr_sizes: [],
      divisor,
      max_dim: maxDim ?? undefined,
      reason: '未找到满足约束的 patch_size',
    }
  }

  const target = params.patch_size ?? (maxDim ? Math.min(Math.floor(maxDim / 2), 256) : candidates[0])
  const sorted = candidates
    .slice()
    .sort((a, b) => {
      const da = Math.abs(a - target)
      const db = Math.abs(b - target)
      if (da !== db) return da - db
      return a - b
    })
    .slice(0, 3)

  return {
    patch_sizes: sorted,
    lr_sizes: sorted.map(size => size / scale),
    divisor,
    max_dim: maxDim ?? undefined,
  }
}

function getSpatialDims(shape?: number[] | null): [number, number] | null {
  if (!shape || shape.length < 2) return null
  const height = shape[shape.length - 2]
  const width = shape[shape.length - 1]
  if (!Number.isFinite(height) || !Number.isFinite(width)) return null
  return [height, width]
}

function resolveFftInputDims(params: {
  datasetInfo?: DatasetValidationInfo
  scale?: number
  patch_size?: number | null
}): { height: number; width: number; source: string } | null {
  const { datasetInfo, scale, patch_size } = params
  const resolvedScale = scale ?? datasetInfo?.scale ?? null

  if (patch_size !== undefined && patch_size !== null && resolvedScale && resolvedScale > 0) {
    const lrSize = patch_size / resolvedScale
    if (Number.isInteger(lrSize)) {
      return {
        height: lrSize,
        width: lrSize,
        source: `patch_size(${patch_size})/scale(${resolvedScale})`,
      }
    }
  }

  const lrDims = getSpatialDims(datasetInfo?.lr_shape)
  if (lrDims) {
    return {
      height: lrDims[0],
      width: lrDims[1],
      source: 'lr_shape',
    }
  }

  const hrDims = getSpatialDims(datasetInfo?.hr_shape)
  if (hrDims && resolvedScale && resolvedScale > 0) {
    const hrHeight = hrDims[0]
    const hrWidth = hrDims[1]
    if (hrHeight % resolvedScale === 0 && hrWidth % resolvedScale === 0) {
      return {
        height: hrHeight / resolvedScale,
        width: hrWidth / resolvedScale,
        source: `hr_shape/scale(${resolvedScale})`,
      }
    }
  }

  return null
}

function buildFftAmpIncompatibility(params: {
  model_name?: string
  use_amp?: boolean
  datasetInfo?: DatasetValidationInfo
  scale?: number
  patch_size?: number | null
}): {
  message: string
  details: {
    lr_height: number
    lr_width: number
    source: string
    divisor?: number
    max_dim?: number
    recommended_patch_sizes?: number[]
    recommended_lr_sizes?: number[]
  }
} | null {
  const { model_name, use_amp, datasetInfo, scale, patch_size } = params
  if (!isFftSensitiveModel(model_name)) return null
  if (use_amp !== true) return null
  if (!datasetInfo || datasetInfo.status !== 'ok') return null

  const dims = resolveFftInputDims({ datasetInfo, scale, patch_size })
  if (!dims) return null

  const { height, width, source } = dims
  if (isPowerOfTwo(height) && isPowerOfTwo(width)) return null

  const recommendations = buildFftPatchRecommendations({
    model_name,
    scale,
    patch_size,
    datasetInfo,
  })
  const recPatchSizes = recommendations.patch_sizes
  const recLrSizes = recommendations.lr_sizes
  const recDetail =
    recPatchSizes.length > 0
      ? `\n✅ 推荐 patch_size（满足 LR=2^k${recommendations.divisor > 1 ? `，且可被 ${recommendations.divisor} 整除` : ''}${recommendations.max_dim ? `，且 ≤ ${recommendations.max_dim}` : ''}）：${recPatchSizes.join(', ')}\n   对应 LR 尺寸：${recLrSizes.join(', ')}`
      : recommendations.reason
        ? `\n⚠️ 无法给出推荐 patch_size：${recommendations.reason}`
        : ''

  const message = `================================================================================
⚠️ FFT + AMP 兼容性提醒（避免 cuFFT 崩溃）
================================================================================

检测到模型为 **${model_name}** 且 **use_amp=true**。
cuFFT 在半精度下要求 FFT 输入尺寸为 **2 的幂**。

当前 LR 尺寸：${height} × ${width}（来源：${source}）

✅ 解决方案（需用户确认）：
1) 设置 use_amp=false（最简单，FNO/HiNOTE 等 FFT 模型推荐）
2) 重新预处理为 2 的幂尺寸（如 64/128）
3) 设置 patch_size，使 LR 尺寸为 2 的幂（并满足 scale/model_divisor）
${recDetail}

请修改参数后再继续确认执行。
================================================================================`

  return {
    message,
    details: {
      lr_height: height,
      lr_width: width,
      source,
      divisor: recommendations.divisor,
      max_dim: recommendations.max_dim,
      recommended_patch_sizes: recPatchSizes.length > 0 ? recPatchSizes : undefined,
      recommended_lr_sizes: recLrSizes.length > 0 ? recLrSizes : undefined,
    },
  }
}

function buildOomPreWarning(params: {
  gpuInfo?: GpuInfo
  device_ids?: number[]
}): {
  message: string
  details: {
    low_gpus: Array<{
      id: number
      free_gb: number
      total_gb: number
      free_ratio: number
    }>
  }
} | null {
  const { gpuInfo, device_ids } = params
  if (!gpuInfo || !gpuInfo.cuda_available) return null
  if (!device_ids || device_ids.length === 0) return null

  const LOW_FREE_GB = 2
  const LOW_FREE_RATIO = 0.1

  const selected = gpuInfo.gpus.filter((gpu) => device_ids.includes(gpu.id))
  if (selected.length === 0) return null

  const lowGpus = selected.filter((gpu) => {
    const freeRatio = gpu.total_memory_gb > 0 ? gpu.free_memory_gb / gpu.total_memory_gb : 0
    return gpu.free_memory_gb < LOW_FREE_GB || freeRatio < LOW_FREE_RATIO
  })

  if (lowGpus.length === 0) return null

  const detailLines = lowGpus
    .map((gpu) => `- GPU ${gpu.id}: ${gpu.free_memory_gb}GB / ${gpu.total_memory_gb}GB 空闲`)
    .join('\n')

  return {
    message: `================================================================================
⚠️ 显存预警（可能 OOM）
================================================================================

检测到所选 GPU 空闲显存偏低：
${detailLines}

建议：减小 batch_size，设置 patch_size，或更换空闲 GPU。
================================================================================`,
    details: {
      low_gpus: lowGpus.map((gpu) => ({
        id: gpu.id,
        free_gb: gpu.free_memory_gb,
        total_gb: gpu.total_memory_gb,
        free_ratio: gpu.total_memory_gb > 0 ? gpu.free_memory_gb / gpu.total_memory_gb : 0,
      })),
    },
  }
}

async function validateDataset(
  datasetRoot: string,
  pythonPath: string,
  trainingDir: string,
  ctx: { sandbox: { exec: (cmd: string, options?: { timeoutMs?: number }) => Promise<{ code: number; stdout: string; stderr: string }> } },
): Promise<DatasetValidationInfo> {
  const validateScript = path.join(trainingDir, 'validate_dataset.py')
  const validateResult = await ctx.sandbox.exec(
    `"${pythonPath}" "${validateScript}" --dataset_root "${datasetRoot}"`,
    { timeoutMs: 60000 }
  )
  if (validateResult.code === 0) {
    return JSON.parse(validateResult.stdout)
  }

  return {
    status: 'error',
    dataset_root: datasetRoot,
    dyn_vars: [],
    scale: null,
    hr_shape: null,
    lr_shape: null,
    splits: {},
    has_static: false,
    static_vars: [],
    total_samples: { hr: 0, lr: 0 },
    warnings: [],
    errors: [`验证脚本执行失败: ${validateResult.stderr}`]
  }
}

export const oceanSrTrainTool = defineTool({
  name: 'ocean_sr_train',
  description: `执行海洋超分辨率模型训练或测试。

**分阶段确认流程**（每阶段必须等待用户确认）：
1. 确认数据目录和输出目录（自动检测变量和 scale）
2. 选择训练模型
3. 确认训练参数（包括 GPU 选择）
4. 最终确认执行

**首次调用**：只传 dataset_root 和 log_dir，工具会自动检测数据并展示信息
**逐步补充参数**：每次调用补充该阶段需要的参数，直到所有阶段通过
**最终执行**：传入 user_confirmed=true 和 confirmation_token 后启动后台训练

**后台执行模式**：
- 训练启动后立即返回 process_id，不会阻塞等待训练完成
- 使用 ocean_sr_train_status 工具查询训练状态和实时日志
- 服务器关闭时会自动终止训练进程

**训练模式 (mode=train)**：执行完整训练流程，包含验证和早停
**测试模式 (mode=test)**：加载最佳模型，在测试集上评估

**GPU 模式**：
- 单卡：device_ids 长度为 1
- 多卡 DP：distribute=true, distribute_mode="DP"
- 多卡 DDP（推荐）：distribute=true, distribute_mode="DDP"`,

  params: {
    dataset_root: {
      type: 'string',
      description: '预处理数据根目录（ocean-preprocess 输出目录）',
      required: false
    },
    log_dir: {
      type: 'string',
      description: '训练日志输出目录',
      required: false
    },
    model_name: {
      type: 'string',
      description: '模型名称（如 SwinIR, FNO2d, DDPM 等）',
      required: false
    },
    dyn_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '动态变量列表（如 ["temp", "salt"]）。如不提供，将从数据目录自动检测并要求确认。',
      required: false
    },
    scale: {
      type: 'number',
      description: '超分辨率倍数。如不提供，将从数据目录自动推算并要求确认。',
      required: false
    },
    mode: {
      type: 'string',
      description: '运行模式: "train" 或 "test"',
      required: false,
      default: 'train'
    },
    epochs: {
      type: 'number',
      description: '训练轮数',
      required: false,
      default: 500
    },
    lr: {
      type: 'number',
      description: '学习率',
      required: false,
      default: 0.001
    },
    batch_size: {
      type: 'number',
      description: '训练 batch size',
      required: false,
      default: 4
    },
    eval_batch_size: {
      type: 'number',
      description: '评估 batch size',
      required: false,
      default: 4
    },
    device_ids: {
      type: 'array',
      items: { type: 'number' },
      description: '使用的 GPU 列表（如 [0, 1, 2, 3]）。必须由用户确认。若启用多卡训练，至少需要两个 GPU。',
      required: false
    },
    distribute: {
      type: 'boolean',
      description: '是否启用多卡训练',
      required: false,
      default: false
    },
    distribute_mode: {
      type: 'string',
      description: '多卡模式: "DP" 或 "DDP"',
      required: false,
      default: 'DDP'
    },
    patience: {
      type: 'number',
      description: '早停耐心值',
      required: false,
      default: 10
    },
    eval_freq: {
      type: 'number',
      description: '评估频率（每 N 个 epoch）',
      required: false,
      default: 5
    },
    normalize: {
      type: 'boolean',
      description: '是否归一化',
      required: false,
      default: true
    },
    normalizer_type: {
      type: 'string',
      description: '归一化类型: "PGN" 或 "GN"',
      required: false,
      default: 'PGN'
    },
    optimizer: {
      type: 'string',
      description: '优化器: "AdamW", "Adam", "SGD"',
      required: false,
      default: 'AdamW'
    },
    weight_decay: {
      type: 'number',
      description: '权重衰减',
      required: false,
      default: 0.001
    },
    scheduler: {
      type: 'string',
      description: '学习率调度器: "StepLR", "MultiStepLR", "OneCycleLR"',
      required: false,
      default: 'StepLR'
    },
    scheduler_step_size: {
      type: 'number',
      description: '调度器步长',
      required: false,
      default: 300
    },
    scheduler_gamma: {
      type: 'number',
      description: '调度器衰减率',
      required: false,
      default: 0.5
    },
    seed: {
      type: 'number',
      description: '随机种子',
      required: false,
      default: 42
    },
    wandb: {
      type: 'boolean',
      description: '是否启用 WandB 日志',
      required: false,
      default: false
    },
    use_amp: {
      type: 'boolean',
      description: '是否启用 AMP 混合精度训练（减少约 40-50% 显存；默认：非 FFT 开启 / FFT 频域模型关闭）',
      required: false
    },
    gradient_checkpointing: {
      type: 'boolean',
      description: '是否启用 Gradient Checkpointing（减少约 60% 激活显存，增加约 30% 计算时间，默认开启）',
      required: false,
      default: true
    },
    patch_size: {
      type: 'number',
      description: 'HR Patch 裁剪尺寸（如 64, 128），设置后训练时随机裁剪小区域而非全图训练。必须能被 scale 整除。',
      required: false
    },
    ckpt_path: {
      type: 'string',
      description: '恢复训练的检查点路径',
      required: false
    },
    user_confirmed: {
      type: 'boolean',
      description: '【必须】用户确认标志。必须在展示参数汇总并获得用户明确确认后，才能设置为 true。禁止自动设置！',
      required: false,
      default: false
    },
    confirmation_token: {
      type: 'string',
      description: '执行确认 Token。必须从 awaiting_execution 阶段的返回值中获取。',
      required: false
    }
  },

  async exec(args, ctx) {
    // 训练工具需要 torch，优先查找安装了 torch 的 Python
    const pythonPath = findPythonWithModule('torch') || findFirstPythonPath()
    if (!pythonPath) {
      throw new Error('未找到可用的 Python 解释器（需要安装 torch）')
    }

    const trainingDir = path.resolve(process.cwd(), 'scripts/ocean_SR_training_masked')
    args.log_dir = args.log_dir ? path.resolve(ctx.sandbox.workDir, args.log_dir) : undefined
    args.dataset_root = args.dataset_root ? path.resolve(ctx.sandbox.workDir, args.dataset_root) : undefined

    const userSpecifiedUseAmp = Object.prototype.hasOwnProperty.call(args, 'use_amp')
    const fftSensitive = isFftSensitiveModel(args.model_name as string | undefined)
    let autoDisabledAmp = false

    if (!userSpecifiedUseAmp && args.model_name) {
      if (fftSensitive) {
        args.use_amp = false
        autoDisabledAmp = true
      } else {
        args.use_amp = true
      }
    }

    // ===== 1. 构建工作流参数 =====
    const workflowParams = { ...args }
    const workflow = new TrainingWorkflow(workflowParams)
    const stateCheck = workflow.determineCurrentState()

    // ===== 2. 如果未到 PASS 阶段，收集上下文信息并返回提示 =====
    if (stateCheck.currentState !== TrainingState.PASS) {
      const context: {
        datasetInfo?: DatasetValidationInfo
        gpuInfo?: GpuInfo
        modelList?: ModelInfo[]
      } = {}

      // 如果有 dataset_root，验证数据目录
      if (args.dataset_root) {
        context.datasetInfo = await validateDataset(args.dataset_root, pythonPath, trainingDir, ctx)
      }

      // 阶段2+需要模型列表
      if (stateCheck.currentState === TrainingState.AWAITING_MODEL_SELECTION) {
        const listScript = path.join(trainingDir, 'list_models.py')
        const listResult = await ctx.sandbox.exec(
          `"${pythonPath}" "${listScript}"`,
          { timeoutMs: 30000 }
        )
        if (listResult.code === 0) {
          const parsed = JSON.parse(listResult.stdout)
          context.modelList = parsed.models
        }
      }

      // 阶段3+需要 GPU 信息
      if (
        stateCheck.currentState === TrainingState.AWAITING_PARAMETERS ||
        stateCheck.currentState === TrainingState.AWAITING_EXECUTION
      ) {
        const gpuScript = path.join(trainingDir, 'check_gpu.py')
        const gpuResult = await ctx.sandbox.exec(
          `"${pythonPath}" "${gpuScript}"`,
          { timeoutMs: 30000 }
        )
        if (gpuResult.code === 0) {
          context.gpuInfo = JSON.parse(gpuResult.stdout)
        }
      }

      const prompt = workflow.getStagePrompt(context)
      const fnoAmpWarning = buildFftAmpIncompatibility({
        model_name: args.model_name,
        use_amp: args.use_amp ?? true,
        datasetInfo: context.datasetInfo,
        scale: args.scale,
        patch_size: args.patch_size ?? null,
      })
      if (fnoAmpWarning) {
        prompt.message = `${prompt.message}\n\n${fnoAmpWarning.message}`
        prompt.data = {
          ...(prompt.data ?? {}),
          fft_amp_warning: fnoAmpWarning.details,
          fno_amp_warning: fnoAmpWarning.details,
        }
      }

      if (autoDisabledAmp) {
        prompt.message = `${prompt.message}\n\n检测到模型 ${args.model_name} 属于 FFT 类，已默认关闭 AMP（use_amp=false）。如需强制开启，请明确设置 use_amp=true。`
        prompt.data = {
          ...(prompt.data ?? {}),
          amp_auto_disabled: { model: args.model_name },
        }
      }

      if (
        args.model_name &&
        AMP_AUTO_DISABLE_MODELS.has(args.model_name as string) &&
        userSpecifiedUseAmp &&
        args.use_amp === true
      ) {
        prompt.message = `${prompt.message}\n\n你已手动启用 AMP（强烈不建议）。FFT 类模型可能触发 cuFFT 尺寸限制错误，如仍要继续请确认。`
        prompt.data = {
          ...(prompt.data ?? {}),
          amp_user_override: { model: args.model_name },
        }
      }

      const oomWarning = buildOomPreWarning({
        gpuInfo: context.gpuInfo,
        device_ids: args.device_ids,
      })
      if (oomWarning) {
        prompt.message = `${prompt.message}\n\n${oomWarning.message}`
        prompt.data = {
          ...(prompt.data ?? {}),
          oom_warning: oomWarning.details,
        }
      }
      return {
        status: prompt.status,
        message: prompt.message,
        canExecute: prompt.canExecute,
        ...prompt.data
      }
    }

    // ===== 3. PASS 阶段：执行训练 =====
    const {
      dataset_root,
      log_dir,
      model_name,
      dyn_vars,
      scale,
      mode = 'train',
      device_ids = [0],
      distribute = false,
      distribute_mode = 'DDP',
      ckpt_path,
    } = args

    if (!log_dir) {
      return {
        status: 'error',
        error: '未指定训练日志输出目录 (log_dir)',
        suggestion: '请在参数中提供 log_dir'
      }
    }

    // ===== 3.0 模型支持性检查（若模型未接入，提前阻断） =====
    let modelSupportInfo: ModelInfo | undefined
    const listScript = path.join(trainingDir, 'list_models.py')
    const listResult = await ctx.sandbox.exec(
      '"' + pythonPath + '" "' + listScript + '"',
      { timeoutMs: 30000 }
    )
    if (listResult.code === 0) {
      try {
        const parsed = JSON.parse(listResult.stdout)
        if (Array.isArray(parsed.models)) {
          modelSupportInfo = parsed.models.find((m: ModelInfo) => m.name === model_name)
        }
      } catch {
        modelSupportInfo = undefined
      }
    }
    if (modelSupportInfo && modelSupportInfo.supported === false) {
      return {
        status: 'error',
        error: '模型 ' + model_name + ' 未接入训练流程',
        reason: modelSupportInfo.notes ?? modelSupportInfo.description,
        suggestion: '请改用已接入的模型，或补齐模型注册 / trainer / 配置模板后再试'
      }
    }
    if (listResult.code === 0 && !modelSupportInfo) {
      return {
        status: 'error',
        error: '未知模型: ' + model_name,
        suggestion: '请从模型列表中选择，或确认模型名称是否拼写正确'
      }
    }

    // ===== 3.1 FFT + AMP 兼容性预检（提示，不拦截） =====
    let fftAmpWarningAtStart: ReturnType<typeof buildFftAmpIncompatibility> | null = null
    if (isFftSensitiveModel(model_name) && (args.use_amp ?? true)) {
      const datasetInfo = await validateDataset(dataset_root, pythonPath, trainingDir, ctx)
      fftAmpWarningAtStart = buildFftAmpIncompatibility({
        model_name,
        use_amp: args.use_amp ?? true,
        datasetInfo,
        scale,
        patch_size: args.patch_size ?? null,
      })
    }

    // ===== 3a. 准备训练工作空间（只复制所选模型相关代码） =====
    // 训练在副本上执行，保持 Agent SDK 源码不被修改；
    // Agent 运行时如需调整代码，可直接修改副本而不影响 SDK；
    // 切换模型时会自动清理旧模型代码并替换为新模型代码
    const workspaceDir = path.resolve(log_dir, '_ocean_sr_code')
    const prepareScript = path.join(trainingDir, 'prepare_workspace.py')
    const prepareResult = await ctx.sandbox.exec(
      `"${pythonPath}" "${prepareScript}" --source_dir "${trainingDir}" --target_dir "${workspaceDir}" --model_name "${model_name}"`,
      { timeoutMs: 60000 }
    )
    if (prepareResult.code !== 0) {
      return {
        status: 'error',
        error: `工作空间准备失败: ${prepareResult.stderr}`,
        reason: '无法将训练代码复制到输出目录',
        suggestion: `请检查输出目录 ${log_dir} 是否存在且有写入权限`
      }
    }
    const prepareInfo = JSON.parse(prepareResult.stdout)

    const generateScript = path.join(workspaceDir, 'generate_config.py')

    // ===== 3b. 生成配置文件 =====
    // 显式白名单：只传递 generate_config.py 需要的参数，避免泄漏状态机内部字段
    const configParams: Record<string, unknown> = {
      model_name,
      dataset_root,
      dyn_vars,
      scale,
      log_dir,
      device: device_ids[0],
      device_ids,
      distribute,
      distribute_mode,
      ckpt_path,
      epochs: args.epochs,
      lr: args.lr,
      batch_size: args.batch_size,
      eval_batch_size: args.eval_batch_size,
      patience: args.patience,
      eval_freq: args.eval_freq,
      normalize: args.normalize,
      normalizer_type: args.normalizer_type,
      optimizer: args.optimizer,
      weight_decay: args.weight_decay,
      scheduler: args.scheduler,
      scheduler_step_size: args.scheduler_step_size,
      scheduler_gamma: args.scheduler_gamma,
      seed: args.seed,
      wandb: args.wandb,
      use_amp: args.use_amp,
      gradient_checkpointing: args.gradient_checkpointing,
      patch_size: args.patch_size,
    }

    const configPath = path.join(workspaceDir, `${model_name}_config.yaml`)

    const paramsJson = JSON.stringify(configParams)
    const genResult = await ctx.sandbox.exec(
      `"${pythonPath}" "${generateScript}" --params '${shellSafeJson(paramsJson)}' --output "${configPath}"`,
      { timeoutMs: 60000 }
    )

    if (genResult.code !== 0) {
      return {
        status: 'error',
        error: `配置生成失败: ${genResult.stderr}`,
        reason: '参数可能不兼容所选模型，或数据目录不可访问',
        suggestion: '请检查 dataset_root 路径是否正确，以及 model_name 是否在支持列表中'
      }
    }

    const genInfo = JSON.parse(genResult.stdout)

    // ===== 3b.1 训练前模型输出尺寸预检 =====
    if (mode === 'train') {
      const shapeCheckScript = path.join(workspaceDir, 'check_output_shape.py')
      const deviceId = device_ids[0] ?? 0
      const shapeResult = await ctx.sandbox.exec(
        `"${pythonPath}" "${shapeCheckScript}" --config "${configPath}" --device ${deviceId}`,
        { timeoutMs: 120000 }
      )

      if (shapeResult.code !== 0) {
        return {
          status: 'error',
          error: `输出尺寸预检失败: ${shapeResult.stderr || shapeResult.stdout}`,
          reason: '无法完成模型输出尺寸检查',
          suggestion: '请检查模型配置或数据目录是否可用'
        }
      }

      const shapeInfo = extractTaggedJson(shapeResult.stdout, 'shape_check')
      if (shapeInfo && shapeInfo.status === 'error') {
        return {
          status: 'error',
          error: shapeInfo.error ?? '模型输出尺寸与 HR 不匹配',
          reason: shapeInfo.reason ?? '模型输出尺寸与目标尺寸不一致',
          details: shapeInfo.details,
          suggestion:
            '请检查 scale/upsample_factor 配置，或调整 patch_size/模型参数使输出与 HR 对齐'
        }
      }
    }

    // ===== 3c. 自动显存预估 + 自动调参（不可跳过） =====
    if (mode === 'train') {
      const estimateScript = path.join(workspaceDir, 'estimate_memory.py')
      const cudaDevice = device_ids[0]
      let currentBatchSize = (configParams.batch_size as number) ?? 4
      let currentAmp = (configParams.use_amp as boolean) ?? true
      const MAX_ATTEMPTS = 5

      for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
        // 每次调参后重新生成配置
        if (attempt > 0) {
          configParams.batch_size = currentBatchSize
          configParams.use_amp = currentAmp
          const regenJson = JSON.stringify(configParams)
          const regenResult = await ctx.sandbox.exec(
            `"${pythonPath}" "${generateScript}" --params '${shellSafeJson(regenJson)}' --output "${configPath}"`,
            { timeoutMs: 60000 }
          )
          if (regenResult.code !== 0) break
        }

        const estimateResult = await ctx.sandbox.exec(
          `cd "${workspaceDir}" && CUDA_VISIBLE_DEVICES=${cudaDevice} "${pythonPath}" "${estimateScript}" --config "${configPath}" --device 0`,
          { timeoutMs: 120000 }
        )
        if (estimateResult.code !== 0) break

        try {
          const mem = JSON.parse(estimateResult.stdout)
          if (mem.status === 'success' && mem.utilization_pct <= 85) {
            // 通过 → 跳出循环
            break
          }

          // OOM 或 >85%：依次降级
          if (!currentAmp) {
            currentAmp = true
          } else if (currentBatchSize > 1) {
            currentBatchSize = Math.max(1, Math.floor(currentBatchSize / 2))
          } else {
            // 所有手段耗尽
            return {
              status: 'error',
              error: 'GPU 显存不足，已尝试所有自动优化手段仍无法适配',
              memory_estimate: mem,
              applied_optimizations: { use_amp: currentAmp, batch_size: currentBatchSize },
              recommendations: mem.recommendations,
              suggestion: '请使用更大显存的 GPU，或手动设置更小的 patch_size'
            }
          }
        } catch {
          // 解析失败不阻止训练
          break
        }
      }
    }

    // ===== 3d. 构建运行命令 =====
    // 注：代码快照由 Python 的 main.py / main_ddp.py 在训练开始前自动保存到 saving_path/code/
    let cmdPath: string
    let cmdArgs: string[]
    let cmdEnv: Record<string, string> = {}

    if (distribute && distribute_mode === 'DDP' && device_ids.length > 1) {
      const nproc = device_ids.length
      const cudaDevices = device_ids.join(',')
      const mainDdp = path.join(workspaceDir, 'main_ddp.py')
      cmdPath = pythonPath
      cmdArgs = ['-m', 'torch.distributed.run', `--nproc_per_node=${nproc}`, mainDdp, '--mode', mode, '--config', configPath]
      cmdEnv = { CUDA_VISIBLE_DEVICES: cudaDevices }
    } else {
      const cudaDevice = String(device_ids[0])
      const mainPy = path.join(workspaceDir, 'main.py')
      cmdPath = pythonPath
      cmdArgs = [mainPy, '--mode', mode, '--config', configPath]
      cmdEnv = { CUDA_VISIBLE_DEVICES: cudaDevice }
    }

    // ===== 3e. 启动后台训练进程 =====
    const processInfo = trainingProcessManager.startProcess({
      cmd: cmdPath,
      args: cmdArgs,
      cwd: workspaceDir,
      logDir: log_dir,
      env: cmdEnv,
      metadata: {
        modelName: model_name,
        datasetRoot: dataset_root,
        logDir: log_dir,
        configPath: genInfo.config_path,
        workspaceDir: workspaceDir,
      },
    })

    // ===== 3f. 等待训练启动成功（事件驱动） =====
    const STARTUP_TIMEOUT_MS = 300000  // 5 分钟（数据加载可能很久）
    const startupResult = await trainingProcessManager.waitForEvent(
      processInfo.id, 'training_start', STARTUP_TIMEOUT_MS
    )

    if (startupResult.processStatus === 'failed' || startupResult.processStatus === 'killed') {
      // 启动阶段崩溃 → 直接返回错误
      const failedInfo = trainingProcessManager.getProcess(processInfo.id)
      return {
        status: 'error',
        error: '训练在启动阶段崩溃（数据加载/模型构建失败）',
        process_id: processInfo.id,
        error_summary: failedInfo?.errorSummary ?? null,
        error_log_tail: trainingProcessManager.readLogs(processInfo.id, { tail: 50 })?.content,
        suggestions: failedInfo?.errorSummary?.suggestions ?? [],
      }
    }

    if (startupResult.found) {
      // training_start 事件收到 → 训练正常运行中
      return {
        status: 'started',
        message: '训练已启动并正常运行中。使用 ocean_sr_train_status 工具监控进度。',
        process_id: processInfo.id,
        pid: processInfo.pid,
        mode,
        model: model_name,
        config_path: genInfo.config_path,
        log_dir,
        log_file: processInfo.logFile,
        distribute: distribute && device_ids.length > 1,
        distribute_mode: distribute ? distribute_mode : 'single',
        device_ids,
        workspace_dir: workspaceDir,
        workspace_info: prepareInfo,
        amp_auto_disabled: autoDisabledAmp ? { model: args.model_name } : undefined,
        warnings: fftAmpWarningAtStart
          ? [`FFT + AMP 可能不兼容：LR ${fftAmpWarningAtStart.details.lr_height}×${fftAmpWarningAtStart.details.lr_width} 不是 2 的幂，建议 use_amp=false 或调整尺寸。`]
          : undefined,
        fft_amp_warning: fftAmpWarningAtStart?.details,
        next_steps: [
          `调用 ocean_sr_train_status({ action: "wait", process_id: "${processInfo.id}", timeout: 120 }) 等待训练状态变化`,
          `调用 ocean_sr_train_status({ process_id: "${processInfo.id}" }) 查看训练状态`,
          `调用 ocean_sr_train_status({ action: "logs", process_id: "${processInfo.id}", tail: 50 }) 查看最新日志`,
          `调用 ocean_sr_train_status({ action: "kill", process_id: "${processInfo.id}" }) 终止训练`,
        ],
      }
    }

    // 超时 → 进程仍在运行但还没开始训练（超大数据集加载中）
    return {
      status: 'started',
      message: '训练进程已启动，仍在初始化中（可能数据量较大）。使用 ocean_sr_train_status 监控。',
      process_id: processInfo.id,
      pid: processInfo.pid,
      mode,
      model: model_name,
      config_path: genInfo.config_path,
      log_dir,
      log_file: processInfo.logFile,
      distribute: distribute && device_ids.length > 1,
      distribute_mode: distribute ? distribute_mode : 'single',
      device_ids,
      workspace_dir: workspaceDir,
      workspace_info: prepareInfo,
      amp_auto_disabled: autoDisabledAmp ? { model: args.model_name } : undefined,
      warnings: fftAmpWarningAtStart
        ? [`FFT + AMP 可能不兼容：LR ${fftAmpWarningAtStart.details.lr_height}×${fftAmpWarningAtStart.details.lr_width} 不是 2 的幂，建议 use_amp=false 或调整尺寸。`]
        : undefined,
      fft_amp_warning: fftAmpWarningAtStart?.details,
      next_steps: [
        `调用 ocean_sr_train_status({ action: "wait", process_id: "${processInfo.id}", timeout: 120 }) 等待训练状态变化`,
        `调用 ocean_sr_train_status({ process_id: "${processInfo.id}" }) 查看训练状态`,
        `调用 ocean_sr_train_status({ action: "logs", process_id: "${processInfo.id}", tail: 50 }) 查看最新日志`,
      ],
    }
  }
})
