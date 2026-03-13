/**
 * @file train-validators.ts
 *
 * @description 训练前验证与警告函数
 *              包含 FFT+AMP 兼容性检查、OOM 预警、
 *              数据集验证、超参数推荐等函数
 * @author kongzhiquan
 * @date 2026-03-13
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-03-13 kongzhiquan: v1.0.0 从 train.ts 第 306-509 行提取
 */

import path from 'node:path'
import type { SrDatasetInfo, SrGpuInfo } from './workflow'
import { shellEscapeDouble, extractTaggedJson } from '@/utils/shell'
import {
  isFftSensitiveModel,
  isPowerOfTwo,
  buildFftPatchRecommendations,
  resolveFftInputDims,
} from './train-model-config'

export function buildFftAmpIncompatibility(params: {
  model_name?: string
  use_amp?: boolean
  datasetInfo?: SrDatasetInfo
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

export function buildOomPreWarning(params: {
  gpuInfo?: SrGpuInfo
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

export async function validateDataset(
  datasetRoot: string,
  pythonPath: string,
  trainingDir: string,
  ctx: { sandbox: { exec: (cmd: string, options?: { timeoutMs?: number }) => Promise<{ code: number; stdout: string; stderr: string }> } },
): Promise<SrDatasetInfo> {
  const validateScript = path.join(trainingDir, 'validate_dataset.py')
  const validateResult = await ctx.sandbox.exec(
    `"${shellEscapeDouble(pythonPath)}" "${shellEscapeDouble(validateScript)}" --dataset_root "${shellEscapeDouble(datasetRoot)}"`,
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

/**
 * 调用 recommend_hyperparams.py 获取超参数推荐。
 * 失败时返回 null，不抛出异常（不影响主流程）。
 */
export async function runHyperparamRecommendation(
  args: {
    dataset_root?: string
    model_name?: string
    scale?: number
    dyn_vars?: string[]
    device_ids?: number[]
  },
  pythonPath: string,
  trainingDir: string,
  ctx: { sandbox: { exec: (cmd: string, options?: { timeoutMs?: number }) => Promise<{ code: number; stdout: string; stderr: string }> } },
): Promise<Record<string, unknown> | null> {
  if (!args.dataset_root || !args.model_name || !args.scale || !args.dyn_vars?.length) {
    return null
  }
  try {
    const recommendScript = path.join(trainingDir, 'recommend_hyperparams.py')
    const deviceId = Number(args.device_ids?.[0] ?? 0)
    const cmd = [
      `cd "${shellEscapeDouble(trainingDir)}"`,
      `&&`,
      `CUDA_VISIBLE_DEVICES=${deviceId}`,
      `"${shellEscapeDouble(pythonPath)}"`,
      `"${shellEscapeDouble(recommendScript)}"`,
      `--dataset_root "${shellEscapeDouble(args.dataset_root)}"`,
      `--model_name "${shellEscapeDouble(args.model_name)}"`,
      `--scale ${args.scale}`,
      `--dyn_vars "${shellEscapeDouble(args.dyn_vars.join(','))}"`,
      `--device 0`,
    ].join(' ')
    const result = await ctx.sandbox.exec(cmd, { timeoutMs: 180000 })
    if (result.code !== 0) return null
    return extractTaggedJson(result.stdout, 'recommend')
  } catch {
    return null
  }
}
