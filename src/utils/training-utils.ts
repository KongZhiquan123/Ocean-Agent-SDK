/**
 * @file training-utils.ts
 *
 * @description 训练工具公共函数：会话参数持久化、超参数推荐格式化
 * @author kongzhiquan
 * @date 2026-03-04
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-03-04 kongzhiquan: 初始版本，从 ocean-SR-training/train.ts 和 ocean-forecast-training/train.ts 提取公共逻辑
 */
import { readFile, mkdir, writeFile } from 'fs/promises'
import path from 'node:path'
import type { EnhancedToolContext } from '@shareai-lab/kode-sdk'

// ========================================
// 训练会话参数持久化
// ========================================

/**
 * 从 targetDir 读取保存的训练会话参数。
 * 不存在或解析失败时返回 null。
 *
 * @param targetDir  目标目录
 * @param filename  会话缓存文件名（如 .ocean_sr_session.json）
 * @param ctx  工具执行上下文（需包含 ctx.sandbox.fs.read）
 */
export async function loadSessionParams<T>(
  targetDir: string,
  filename: string,
  ctx: EnhancedToolContext
): Promise<T | null> {
  try {
    if (!ctx.sandbox.fs.isInside(targetDir)) {
      throw new Error('Target directory is not inside sandbox')
    }
    const sessionPath = path.join(targetDir, filename)
    const content = await readFile(sessionPath, 'utf-8')
    const parsed = JSON.parse(content)
    return (parsed?.params as T) ?? null
  } catch {
    return null
  }
}

/**
 * 将当前全量参数写入 targetDir 的会话缓存文件。
 * 写入失败不抛出异常，不影响主流程。
 *
 * @param targetDir  目标目录
 * @param filename  会话缓存文件名（如 .ocean_sr_session.json）
 * @param params  要持久化的参数对象
 * @param ctx  工具执行上下文（需包含 ctx.sandbox.fs.write）
 */
export async function saveSessionParams<T>(
  targetDir: string,
  filename: string,
  params: T,
  ctx: EnhancedToolContext,
): Promise<void> {
  try {
    if (!ctx.sandbox.fs.isInside(targetDir)) {
      throw new Error('Target directory is not inside sandbox')
    }
    await mkdir(targetDir, { recursive: true })
    const sessionPath = path.join(targetDir, filename)
    await writeFile(sessionPath, JSON.stringify({ savedAt: Date.now(), params }))
  } catch {
    // 写入失败不影响主流程
  }
}

// ========================================
// 超参数推荐结果格式化
// ========================================

export interface FormatRecommendationOptions {
  /**
   * 数据集形状字段名。
   * SR 任务传 'hr_shape'（HR 图像形状），Forecast 任务传 'spatial_shape'（空间分辨率）。
   * 默认 'hr_shape'。
   */
  datasetShapeKey?: string
  /**
   * 数据集形状的显示标签。
   * SR 任务传 'HR'，Forecast 任务传 '空间分辨率'。
   * 默认 'HR'。
   */
  datasetShapeLabel?: string
}

/**
 * 将超参数推荐结果格式化为用户可读的消息段落。
 * 失败时返回空字符串。
 */
export function formatRecommendationMessage(
  rec: Record<string, unknown>,
  options: FormatRecommendationOptions = {},
): string {
  const { datasetShapeKey = 'hr_shape', datasetShapeLabel = 'HR' } = options

  const recommendations = rec.recommendations as Record<string, unknown> | undefined
  const reasoning = rec.reasoning as Record<string, unknown> | undefined
  const datasetInfo = rec.dataset_info as Record<string, unknown> | undefined
  const gpuInfo = rec.gpu_info as Record<string, unknown> | undefined
  const spectral = rec.spectral_analysis as Record<string, unknown> | undefined
  const modelNotes = rec.model_notes as Record<string, unknown> | undefined

  if (!recommendations) return ''

  const lines: string[] = [
    '================================================================================',
    '                    💡 超参数推荐（基于实测显存 + 数据分析）',
    '================================================================================',
  ]

  // 数据集 & GPU 基本信息
  if (datasetInfo || gpuInfo) {
    lines.push('\n【分析基础】')
    if (datasetInfo) {
      const shapeStr = (datasetInfo[datasetShapeKey] as number[])?.join(' × ') ?? '?'
      lines.push(`- 训练集：${datasetInfo.n_train} 个样本，${datasetShapeLabel} ${shapeStr}，${datasetInfo.n_vars} 个变量`)
    }
    if (gpuInfo && gpuInfo.name) {
      lines.push(`- GPU：${gpuInfo.name}（${gpuInfo.total_gb ?? '?'} GB）`)
    }
  }

  // 推荐参数
  lines.push('\n【推荐参数】')
  if (recommendations.batch_size !== undefined)
    lines.push(`- batch_size:        ${recommendations.batch_size}`)
  if (recommendations.eval_batch_size !== undefined)
    lines.push(`- eval_batch_size:   ${recommendations.eval_batch_size}`)
  if (recommendations.epochs !== undefined)
    lines.push(`- epochs:            ${recommendations.epochs}`)
  if (recommendations.lr !== undefined)
    lines.push(`- lr:                ${(recommendations.lr as number).toExponential(2)}`)
  if (recommendations.gradient_checkpointing !== undefined)
    lines.push(`- gradient_checkpointing: ${recommendations.gradient_checkpointing}`)

  // 推荐理由
  if (reasoning && Object.keys(reasoning).length > 0) {
    lines.push('\n【推荐理由】')
    for (const [key, val] of Object.entries(reasoning)) {
      lines.push(`- ${key}: ${val}`)
    }
  }

  // 频谱分析
  if (spectral) {
    lines.push('\n【数据频谱分析（仅供参考，不自动修改模型结构）】')
    lines.push(`- 频率特征：${spectral.freq_desc}（k90 ≈ ${spectral.k90_mean}，max_k = ${spectral.max_k}）`)
  }

  // 模型特定提示
  if (modelNotes) {
    lines.push('\n【模型结构参数参考】')
    for (const [, note] of Object.entries(modelNotes)) {
      lines.push(`- ${note}`)
    }
  }

  lines.push('\n⚠️ Agent 注意：以上为系统推荐值，请告知用户并询问是否采用或调整，再继续执行确认。')
  lines.push('================================================================================')

  return lines.join('\n')
}
