/**
 * @file types.ts
 *
 * @description 训练进程管理器 —— 公共类型定义
 * @author kongzhiquan
 * @date 2026-03-04
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-03-04 kongzhiquan: v1.0.0 从 training-process-manager.ts 拆分
 */

import { ChildProcess } from 'child_process'
import { WriteStream } from 'fs'

export interface ErrorSummary {
  failureType: string
  errorMessage: string
  lastStderrLines: string[]
  suggestions: string[]
  structuredError?: Record<string, unknown>
}

export interface TrainingProgress {
  // Training 模式（epoch 粒度）
  currentEpoch?: number
  totalEpochs?: number
  epochsPerHour?: number | null
  latestMetrics?: Record<string, number>
  // Predict 模式（sample 粒度）
  currentSample?: number
  totalSamples?: number
  samplesPerMinute?: number | null
  currentFilename?: string
  // 共用
  estimatedRemainingSeconds: number | null
}

export interface RuntimeStats {
  sampledAt: string
  uptimeSeconds: number
  cpuPercent: number | null
  memoryRssMB: number | null
  ioReadMB: number | null
  ioWriteMB: number | null
  gpu?: Array<{
    id: number
    utilizationPct: number | null
    memoryUsedMB: number | null
    memoryTotalMB: number | null
    temperatureC: number | null
    powerW: number | null
  }>
}

export interface TrainingNotification {
  id: string
  type: 'training_start' | 'training_end' | 'training_error' | 'process_exit' | 'predict_start' | 'predict_end'
  timestamp: string
  processId: string
  payload?: Record<string, unknown>
}

export interface TrainingProcessInfo {
  id: string
  cmd: string
  args: string[]
  cwd: string
  startTime: number
  endTime?: number
  exitCode?: number
  status: 'running' | 'completed' | 'failed' | 'killed'
  logFile: string
  errorLogFile: string
  pid?: number
  // 训练相关元数据
  metadata?: {
    modelName?: string
    datasetRoot?: string
    logDir?: string
    configPath?: string
    workspaceDir?: string
    deviceIds?: number[]
    mode?: string
  }
  // 错误摘要（失败时填充）
  errorSummary?: ErrorSummary
  // 训练进度（运行时填充）
  progress?: TrainingProgress
  // 运行时资源监控（运行中更新）
  runtimeStats?: RuntimeStats
}

export interface ManagedProcess {
  info: TrainingProcessInfo
  process: ChildProcess
  logStream: WriteStream
  errorLogStream: WriteStream
  // 环形缓冲区：最后 100 行 stderr
  stderrRingBuffer: string[]
  // 来自 Python 的结构化错误事件
  lastTrainingError: Record<string, unknown> | null
  // 来自 Python 的训练结束事件（含 best_epoch / final_metrics）
  lastTrainingEnd: Record<string, unknown> | null
  // stdout 缓冲区（处理跨 chunk 的 __event__ 标记）
  stdoutBuffer: string
  // stderr 事件缓冲区（Python logging.info → StreamHandler → stderr 也可能包含 __event__）
  stderrEventBuffer: string
  // 已接收的事件类型集合（去重，O(1) 查找）
  receivedEvents: Set<string>
  // 最近一次 epoch 信息
  lastEpochInfo: {
    epoch: number
    timestamp: string
    metrics: Record<string, number>
  } | null
  // 训练元信息（来自 training_start 事件）
  trainingMeta: {
    totalEpochs: number
    startTimestamp: string
  } | null
  // Predict 模式元信息（来自 predict_start 事件）
  predictMeta: {
    totalSamples: number
    outputDir: string
    startTimestamp: string
  } | null
  // 最近一次 predict_progress 信息
  lastPredictProgress: {
    current: number
    total: number
    filename: string
    timestamp: string
  } | null
  // 日志滚动锁
  rotatingLog: boolean
  rotatingErrorLog: boolean
  runtimeStats: RuntimeStats | null
  runtimeSampleTimer: ReturnType<typeof setInterval> | null
  notifications: TrainingNotification[]
}
