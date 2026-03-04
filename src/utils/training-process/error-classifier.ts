/**
 * @file error-classifier.ts
 *
 * @description 训练进程管理器 —— 失败类型分类
 * @author kongzhiquan
 * @date 2026-03-04
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-03-04 kongzhiquan: v1.0.0 从 training-process-manager.ts 拆分
 */

import { ErrorSummary, ManagedProcess } from './types'

const FAILURE_PATTERNS: Array<{
  pattern: RegExp
  type: string
  suggestion: string
}> = [
  {
    pattern: /CUDA out of memory/i,
    type: 'CUDA_OOM',
    suggestion: '启用 use_amp=true，减小 batch_size，或设置 patch_size',
  },
  {
    pattern: /EADDRINUSE|address already in use|server socket has failed to listen/i,
    type: 'DDP_PORT_IN_USE',
    suggestion: 'DDP 端口被占用，请指定 master_port 或结束占用端口的训练进程',
  },
  {
    pattern: /Default process group has not been initialized|init_process_group/i,
    type: 'DDP_NOT_INITIALIZED',
    suggestion: '未初始化分布式进程组：单卡请关闭 distribute，或使用 torchrun 启动 DDP',
  },
  {
    pattern: /ChildFailedError|elastic.*multiprocessing.*errors/i,
    type: 'DDP_CHILD_FAILED',
    suggestion: '某个 GPU rank 崩溃导致 DDP 训练中止，查看 stderr 获取具体 rank 和错误信息',
  },
  {
    pattern: /cuFFT|cufft/i,
    type: 'FFT_ERROR',
    suggestion: 'FFT/频域报错：建议关闭 AMP，或调整 patch_size/输入尺寸为 2 的幂',
  },
  {
    pattern: /\bNaN\b|\bnan\b/i,
    type: 'NUMERICAL_NAN',
    suggestion: '出现 NaN：建议降低学习率、关闭 AMP、检查归一化与数据异常值',
  },
  {
    pattern: /NCCL.*timeout|NCCL.*error/i,
    type: 'NCCL_ERROR',
    suggestion: '检查多卡连接或改用单卡训练',
  },
  {
    pattern: /size mismatch|shape.*mismatch/i,
    type: 'SHAPE_ERROR',
    suggestion: '检查数据尺寸与模型配置是否匹配',
  },
  {
    pattern: /FileNotFoundError|No such file/i,
    type: 'FILE_NOT_FOUND',
    suggestion: '检查数据路径和配置文件路径',
  },
  {
    pattern: /ModuleNotFoundError|ImportError/i,
    type: 'IMPORT_ERROR',
    suggestion: '缺少 Python 依赖，检查 conda 环境',
  },
  {
    pattern: /KeyError/i,
    type: 'CONFIG_ERROR',
    suggestion: '配置文件缺少必要字段',
  },
  {
    pattern: /RuntimeError.*expected.*got/i,
    type: 'DTYPE_ERROR',
    suggestion: '数据类型不匹配',
  },
]

export function classifyFailure(managed: ManagedProcess): ErrorSummary {
  const stderrText = managed.stderrRingBuffer.join('\n')
  const structuredError = managed.lastTrainingError ?? undefined

  // 从结构化错误中提取信息
  let errorMessage = '训练进程异常退出'
  if (structuredError) {
    errorMessage = `${structuredError.error_type}: ${structuredError.error_message}`
  }

  // 匹配失败类型
  let failureType = 'UNKNOWN'
  const suggestions: string[] = []

  // 先检查结构化错误，再检查 stderr
  const textToSearch = structuredError
    ? `${structuredError.error_type}: ${structuredError.error_message}\n${structuredError.traceback ?? ''}\n${stderrText}`
    : stderrText

  for (const fp of FAILURE_PATTERNS) {
    if (fp.pattern.test(textToSearch)) {
      failureType = fp.type
      suggestions.push(fp.suggestion)
      break
    }
  }

  // 如果没有匹配到已知模式，但有结构化错误
  if (failureType === 'UNKNOWN' && structuredError) {
    failureType = String(structuredError.error_type ?? 'UNKNOWN')
  }

  // 对 DDP_CHILD_FAILED 类型，尝试从 stderr 提取崩溃 rank 信息
  if (failureType === 'DDP_CHILD_FAILED') {
    const rankMatch = stderrText.match(/rank\s*:\s*(\d+)\s*\(local_rank:\s*(\d+)\)/)
    if (rankMatch) {
      errorMessage = `DDP rank ${rankMatch[1]} (local_rank: ${rankMatch[2]}) 崩溃导致训练中止`
    }
    if (structuredError?.error_message) {
      errorMessage += `; 原始错误: ${structuredError.error_message}`
    }
  }

  return {
    failureType,
    errorMessage,
    lastStderrLines: managed.stderrRingBuffer.slice(-20),
    suggestions,
    structuredError: structuredError as Record<string, unknown> | undefined,
  }
}
