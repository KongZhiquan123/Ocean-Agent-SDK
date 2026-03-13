/**
 * @file train-model-config.ts
 *
 * @description 模型常量与形状工具函数
 *              包含 FFT/AMP 敏感模型集合、Diffusion 模型集合、
 *              以及 patch_size/LR 尺寸相关纯函数
 * @author kongzhiquan
 * @date 2026-03-13
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-03-13 kongzhiquan: v1.0.0 从 train.ts 第 106-304 行提取
 */

import type { SrDatasetInfo } from './workflow'

// FFT/频域/复数变换相关模型：默认关闭 AMP，允许用户显式 override（强提示）
export const FFT_AMP_SENSITIVE_MODELS = new Set([
  'FNO2d',
  'HiNOTE',
  'MWT2d',
  'M2NO2d',
  'MG-DDPM',
])
export const AMP_DEFAULT_OFF_MODELS = new Set([...FFT_AMP_SENSITIVE_MODELS, 'SRNO'])
export const DIFFUSION_MODELS = new Set(['DDPM', 'SR3', 'MG-DDPM', 'ReMiG', 'ResShift', 'Resshift'])

export function isFftSensitiveModel(modelName?: string): boolean {
  return Boolean(modelName && FFT_AMP_SENSITIVE_MODELS.has(modelName))
}

export function isAmpDefaultOffModel(modelName?: string): boolean {
  return Boolean(modelName && AMP_DEFAULT_OFF_MODELS.has(modelName))
}

export function isPowerOfTwo(value: number): boolean {
  return Number.isInteger(value) && value > 0 && (value & (value - 1)) === 0
}

export function countPowerOfTwoFactor(value: number): number {
  let v = Math.abs(Math.trunc(value))
  let count = 0
  while (v > 0 && v % 2 === 0) {
    v = v / 2
    count += 1
  }
  return count
}

export function getModelDivisor(modelName?: string): number {
  if (!modelName) return 1
  // ResShift: downsample 2^3=8, Swin window_size=8, divisor=8*8=64
  if (modelName === 'Resshift' || modelName === 'ResShift') return 64
  if (DIFFUSION_MODELS.has(modelName)) return 32
  if (modelName === 'UNet2d') return 16
  return 1
}

export function getMaxHrDim(datasetInfo?: SrDatasetInfo): number | null {
  const hrDims = getSpatialDims(datasetInfo?.hr_shape)
  if (!hrDims) return null
  return Math.min(hrDims[0], hrDims[1])
}

export function buildFftPatchRecommendations(params: {
  model_name?: string
  scale?: number
  patch_size?: number | null
  datasetInfo?: SrDatasetInfo
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

export function getSpatialDims(shape?: number[] | null): [number, number] | null {
  if (!shape || shape.length < 2) return null
  const height = shape[shape.length - 2]
  const width = shape[shape.length - 1]
  if (!Number.isFinite(height) || !Number.isFinite(width)) return null
  return [height, width]
}

export function resolveFftInputDims(params: {
  datasetInfo?: SrDatasetInfo
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
