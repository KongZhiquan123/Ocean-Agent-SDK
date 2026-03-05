/**
 * @file manager.ts
 *
 * @description 训练进程管理器 —— TrainingProcessManager 主类
 * @author kongzhiquan
 * @date 2026-03-04
 * @version 3.3.0
 *
 * @changelog
 *   - 2026-03-05 kongzhiquan: v3.3.0 startProcess/readLogs 改为异步，消除同步 fs 阻塞事件循环
 *   - 2026-03-04 kongzhiquan: v3.2.0 拆分为模块化文件，委托调用自由函数
 *   - 2026-02-25 Leizheng: v3.2.0 修复三个事件系统 Bug
 *   - 2026-02-11 Leizheng: v3.1.0 Predict 模式进度追踪
 *   - 2026-02-11 kongzhiquan: v3.0.0 精简重构
 *   - 2026-02-08 Leizheng: v2.2.0 增加训练日志滚动，限制日志文件大小
 *   - 2026-02-07 Leizheng: v2.1.0 修复事件捕获通道
 *   - 2026-02-07 Leizheng: v2.0.0 训练错误实时反馈增强
 *   - 2026-02-07 kongzhiquan: v1.2.0 增强错误处理
 *   - 2026-02-07 kongzhiquan: v1.1.0 优化日志输出
 *   - 2026-02-07 kongzhiquan: v1.0.0 初始版本
 */

import { spawn } from 'child_process'
import { createWriteStream } from 'fs'
import fs from 'fs/promises'
import path from 'path'

import { MAX_COMPLETED_PROCESSES, RING_BUFFER_SIZE, RUNTIME_SAMPLE_INTERVAL_MS, TAIL_READ_BYTES } from './constants'
import { classifyFailure } from './error-classifier'
import { parseEventMarkers, createNotification, pushNotification } from './event-parser'
import { attachStreamErrorHandler, safeWrite, writeLog } from './log-helpers'
import { sampleRuntimeStats } from './runtime-sampler'
import { ManagedProcess, TrainingNotification, TrainingProcessInfo, TrainingProgress } from './types'

export class TrainingProcessManager {
  private processes: Map<string, ManagedProcess> = new Map()
  private isShuttingDown = false

  /**
   * 从 managed process 计算进度（训练模式或 predict 模式）
   */
  private computeProgress(managed: ManagedProcess): TrainingProgress | undefined {
    // ---- Predict 模式 ----
    if (managed.predictMeta) {
      const total = managed.predictMeta.totalSamples
      if (!managed.lastPredictProgress) {
        return {
          currentSample: 0,
          totalSamples: total,
          estimatedRemainingSeconds: null,
        }
      }
      const current = managed.lastPredictProgress.current
      const startTime = new Date(managed.predictMeta.startTimestamp).getTime()
      const progressTime = new Date(managed.lastPredictProgress.timestamp).getTime()
      const elapsed = progressTime - startTime
      const avgPerSample = current > 0 ? elapsed / current : 0
      const remaining = avgPerSample * (total - current)
      return {
        currentSample: current,
        totalSamples: total,
        estimatedRemainingSeconds: current > 0 ? Math.round(remaining / 1000) : null,
        samplesPerMinute: avgPerSample > 0 ? Math.round((60000 / avgPerSample) * 10) / 10 : null,
        currentFilename: managed.lastPredictProgress.filename,
      }
    }

    // ---- Training 模式 ----
    if (!managed.lastEpochInfo || !managed.trainingMeta || managed.trainingMeta.totalEpochs <= 0) {
      return undefined
    }
    const current = managed.lastEpochInfo.epoch
    const total = managed.trainingMeta.totalEpochs
    const startTime = new Date(managed.trainingMeta.startTimestamp).getTime()
    const epochTime = new Date(managed.lastEpochInfo.timestamp).getTime()
    const elapsed = epochTime - startTime
    // epoch 是 0-indexed，completedEpochs 是已完成轮数
    const completedEpochs = current + 1
    const avgPerEpoch = elapsed / completedEpochs
    const remaining = avgPerEpoch * (total - completedEpochs)
    const estimatedRemainingSeconds = Math.round(remaining / 1000)
    const epochsPerHour = avgPerEpoch > 0 ? Math.round(3600000 / avgPerEpoch) : null
    return {
      currentEpoch: current,
      totalEpochs: total,
      estimatedRemainingSeconds,
      epochsPerHour,
      latestMetrics: managed.lastEpochInfo.metrics ?? {},
    }
  }

  /**
   * 启动一个后台训练进程
   */
  async startProcess(options: {
    cmd: string
    args: string[]
    cwd: string
    logDir: string
    env?: Record<string, string>
    metadata?: TrainingProcessInfo['metadata']
  }): Promise<TrainingProcessInfo> {
    const { cmd, args, cwd, logDir, env, metadata } = options

    const id = `train-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

    await fs.mkdir(logDir, { recursive: true })

    const logFile = path.join(logDir, `${id}.log`)
    const errorLogFile = path.join(logDir, `${id}.error.log`)

    const logStream = createWriteStream(logFile, { flags: 'a' })
    const errorLogStream = createWriteStream(errorLogFile, { flags: 'a' })

    attachStreamErrorHandler(logStream, id, 'log')
    attachStreamErrorHandler(errorLogStream, id, 'error')

    const processLabel = metadata?.mode === 'predict' ? 'Predict' : 'Training'
    const startHeader = `
================================================================================
${processLabel} Process Started
ID: ${id}
Command: ${cmd} ${args.join(' ')}
Working Directory: ${cwd}
Start Time: ${new Date().toISOString()}
================================================================================

`
    safeWrite(logStream, startHeader)

    const processEnv = {
      ...process.env,
      ...env,
      PYTHONUNBUFFERED: '1',
    }

    const childProcess = spawn(cmd, args, {
      cwd,
      env: processEnv,
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: false,
    })

    const info: TrainingProcessInfo = {
      id,
      cmd,
      args,
      cwd,
      startTime: Date.now(),
      status: 'running',
      logFile,
      errorLogFile,
      pid: childProcess.pid,
      metadata,
    }

    const managed: ManagedProcess = {
      info,
      process: childProcess,
      logStream,
      errorLogStream,
      stderrRingBuffer: [],
      lastTrainingError: null,
      lastTrainingEnd: null,
      stdoutBuffer: '',
      stderrEventBuffer: '',
      receivedEvents: new Set(),
      lastEpochInfo: null,
      trainingMeta: null,
      predictMeta: null,
      lastPredictProgress: null,
      rotatingLog: false,
      rotatingErrorLog: false,
      runtimeStats: null,
      runtimeSampleTimer: null,
      notifications: [],
    }

    managed.runtimeSampleTimer = setInterval(() => {
      if (managed.info.status === 'running') {
        sampleRuntimeStats(managed).catch(() => {
          // ignore sampling errors
        })
      }
    }, RUNTIME_SAMPLE_INTERVAL_MS)

    if (childProcess.stdout) {
      childProcess.stdout.on('data', (data: Buffer) => {
        const text = data.toString()
        writeLog(managed, 'log', text).catch(() => {})
        managed.stdoutBuffer += text
        parseEventMarkers(managed, 'stdout')
      })
    }

    if (childProcess.stderr) {
      childProcess.stderr.on('data', (data: Buffer) => {
        const text = data.toString()
        writeLog(managed, 'log', text).catch(() => {})
        writeLog(managed, 'error', text).catch(() => {})

        const lines = text.split('\n').filter((l) => l.trim())
        managed.stderrRingBuffer.push(...lines)
        if (managed.stderrRingBuffer.length > RING_BUFFER_SIZE) {
          managed.stderrRingBuffer.splice(
            0,
            managed.stderrRingBuffer.length - RING_BUFFER_SIZE,
          )
        }

        if (text.includes('__event__')) {
          managed.stderrEventBuffer += text
          parseEventMarkers(managed, 'stderr')
        }
      })
    }

    childProcess.on('exit', (code, signal) => {
      if (managed.runtimeSampleTimer) {
        clearInterval(managed.runtimeSampleTimer)
        managed.runtimeSampleTimer = null
      }

      managed.info.endTime = Date.now()
      managed.info.exitCode = code ?? undefined

      if (signal === 'SIGTERM' || signal === 'SIGKILL') {
        managed.info.status = 'killed'
      } else if (code === 0) {
        managed.info.status = 'completed'
      } else {
        managed.info.status = 'failed'
        managed.info.errorSummary = classifyFailure(managed)
      }

      pushNotification(
        managed,
        createNotification(managed, 'process_exit', {
          status: managed.info.status,
          exitCode: managed.info.exitCode,
          errorSummary: managed.info.errorSummary ?? undefined,
        }),
      )

      const endLabel = managed.info.metadata?.mode === 'predict' ? 'Predict' : 'Training'
      const endFooter = `
================================================================================
${endLabel} Process ${managed.info.status.toUpperCase()}
Exit Code: ${code}
Signal: ${signal || 'none'}
End Time: ${new Date().toISOString()}
Duration: ${((managed.info.endTime - managed.info.startTime) / 1000).toFixed(1)}s
================================================================================
`
      writeLog(managed, 'log', endFooter).catch(() => {})
      this.evictOldProcesses()

      logStream.end()
      errorLogStream.end()
    })

    childProcess.on('error', (err) => {
      if (managed.runtimeSampleTimer) {
        clearInterval(managed.runtimeSampleTimer)
        managed.runtimeSampleTimer = null
      }

      managed.info.status = 'failed'
      managed.info.endTime = Date.now()
      managed.info.errorSummary = classifyFailure(managed)
      pushNotification(
        managed,
        createNotification(managed, 'process_exit', {
          status: managed.info.status,
          exitCode: managed.info.exitCode,
          errorSummary: managed.info.errorSummary ?? undefined,
          errorMessage: err.message,
        }),
      )
      writeLog(managed, 'error', `Process error: ${err.message}\n`).catch(() => {})
      writeLog(managed, 'log', `[ERROR] Process error: ${err.message}\n`).catch(() => {})
      this.evictOldProcesses()
      logStream.end()
      errorLogStream.end()
    })

    this.processes.set(id, managed)

    console.log(`[TrainingProcessManager] Started process ${id} (PID: ${childProcess.pid})`)

    return info
  }

  /**
   * 获取进程信息（含实时进度）
   */
  getProcess(id: string): TrainingProcessInfo | undefined {
    const managed = this.processes.get(id)
    if (!managed) return undefined

    if (managed.info.status === 'running') {
      managed.info.progress = this.computeProgress(managed)
      managed.info.runtimeStats = managed.runtimeStats ?? undefined
    }

    return managed.info
  }

  /**
   * 获取所有进程信息
   */
  getAllProcesses(): TrainingProcessInfo[] {
    return Array.from(this.processes.values()).map((m) => {
      if (m.info.status === 'running') {
        m.info.progress = this.computeProgress(m)
        m.info.runtimeStats = m.runtimeStats ?? undefined
      }
      return m.info
    })
  }

  /**
   * 获取正在运行的进程
   */
  getRunningProcesses(): TrainingProcessInfo[] {
    return this.getAllProcesses().filter((p) => p.status === 'running')
  }

  /**
   * 读取进程日志
   * - tail 模式：从文件尾部读取最后 N 行
   * - offset 模式：从指定字节偏移量开始增量读取
   */
  async readLogs(
    id: string,
    options: { tail?: number; offset?: number },
  ): Promise<{ content: string; size: number; offset: number } | undefined> {
    const managed = this.processes.get(id)
    if (!managed) return undefined

    const { logFile } = managed.info

    try {
      let stat: Awaited<ReturnType<typeof fs.stat>>
      try {
        stat = await fs.stat(logFile)
      } catch {
        return { content: '', size: 0, offset: 0 }
      }

      const size = stat.size

      if (options.tail) {
        const readBytes = Math.min(size, TAIL_READ_BYTES)
        const tailOffset = Math.max(0, size - readBytes)
        const buffer = Buffer.alloc(readBytes)
        const fh = await fs.open(logFile, 'r')
        try {
          await fh.read(buffer, 0, readBytes, tailOffset)
        } finally {
          await fh.close()
        }
        const lines = buffer.toString('utf-8').split('\n')
        if (tailOffset > 0) lines.shift()
        return { content: lines.slice(-options.tail).join('\n'), size, offset: size }
      }

      if (options.offset !== undefined) {
        const safeOffset = Math.max(0, options.offset)
        if (safeOffset >= size) {
          return { content: '', size, offset: size }
        }
        const readBytes = size - safeOffset
        const buffer = Buffer.alloc(readBytes)
        const fh = await fs.open(logFile, 'r')
        try {
          await fh.read(buffer, 0, readBytes, safeOffset)
        } finally {
          await fh.close()
        }
        return { content: buffer.toString('utf-8'), size, offset: size }
      }

      return { content: '', size, offset: size }
    } catch (err) {
      console.error(`[TrainingProcessManager] Failed to read logs for ${id}:`, err)
      return { content: `Error reading logs: ${(err as Error).message}`, size: 0, offset: 0 }
    }
  }

  /**
   * 终止进程
   */
  killProcess(id: string, signal: NodeJS.Signals = 'SIGTERM'): boolean {
    const managed = this.processes.get(id)
    if (!managed || managed.info.status !== 'running') {
      return false
    }

    try {
      managed.process.kill(signal)
      console.log(
        `[TrainingProcessManager] Sent ${signal} to process ${id} (PID: ${managed.info.pid})`,
      )
      return true
    } catch (err) {
      console.error(`[TrainingProcessManager] Failed to kill process ${id}:`, err)
      return false
    }
  }

  /**
   * 统一的长轮询方法：等待进程状态变化或通知事件
   */
  async waitForChange(
    id: string,
    opts: { mode?: 'status' | 'notification'; timeoutMs?: number; pollIntervalMs?: number } = {},
  ): Promise<{
    processInfo?: TrainingProcessInfo
    notification?: TrainingNotification
    processStatus: string
    found: boolean
  }> {
    const { mode = 'status', timeoutMs = 120000, pollIntervalMs = 1000 } = opts
    const start = Date.now()
    const initial = this.getProcess(id)
    if (!initial) return { found: false, processStatus: 'unknown' }

    if (initial.status !== 'running' && mode === 'status') {
      return { found: true, processInfo: initial, processStatus: initial.status }
    }

    const initialStatus = initial.status

    while (Date.now() - start < timeoutMs) {
      const managed = this.processes.get(id)
      if (!managed) return { found: false, processStatus: 'unknown' }

      if (mode === 'notification' && managed.notifications.length > 0) {
        const notification = managed.notifications.shift()
        if (notification) {
          return {
            found: true,
            notification,
            processInfo: managed.info,
            processStatus: managed.info.status,
          }
        }
      }

      if (mode === 'status' && managed.info.status !== initialStatus) {
        const info = this.getProcess(id)
        return { found: true, processInfo: info, processStatus: info?.status ?? 'unknown' }
      }

      if (mode === 'notification' && managed.info.status !== 'running') {
        return { found: false, processInfo: managed.info, processStatus: managed.info.status }
      }

      await new Promise((r) => setTimeout(r, pollIntervalMs))
    }

    const final = this.getProcess(id)
    return { found: false, processInfo: final, processStatus: final?.status ?? 'timeout' }
  }

  /**
   * 等待特定事件出现
   */
  async waitForEvent(
    id: string,
    eventType: string,
    timeoutMs = 300000,
  ): Promise<{ found: boolean; processStatus: string; event?: Record<string, unknown> }> {
    const start = Date.now()
    while (Date.now() - start < timeoutMs) {
      const managed = this.processes.get(id)
      if (!managed) return { found: false, processStatus: 'unknown' }

      if (managed.receivedEvents.has(eventType)) {
        return { found: true, processStatus: managed.info.status }
      }

      if (managed.info.status !== 'running') {
        return { found: false, processStatus: managed.info.status }
      }

      await new Promise((r) => setTimeout(r, 1000))
    }
    return { found: false, processStatus: this.getProcess(id)?.status ?? 'timeout' }
  }

  /**
   * 自动淘汰超量的已完成进程（保留最近 MAX_COMPLETED_PROCESSES 条）
   */
  private evictOldProcesses(): void {
    const completed: Array<{ id: string; endTime: number }> = []
    for (const [id, managed] of this.processes) {
      if (managed.info.status !== 'running') {
        completed.push({ id, endTime: managed.info.endTime ?? 0 })
      }
    }
    if (completed.length <= MAX_COMPLETED_PROCESSES) return

    completed.sort((a, b) => b.endTime - a.endTime)
    const toRemove = completed.slice(MAX_COMPLETED_PROCESSES)
    for (const item of toRemove) {
      this.processes.delete(item.id)
    }
    if (toRemove.length > 0) {
      console.log(`[TrainingProcessManager] Evicted ${toRemove.length} old completed process(es)`)
    }
  }

  /**
   * 关闭所有进程（服务器关闭时调用）
   */
  async shutdown(timeoutMs = 10000): Promise<void> {
    if (this.isShuttingDown) return
    this.isShuttingDown = true

    const running = this.getRunningProcesses()
    if (running.length === 0) {
      console.log('[TrainingProcessManager] No running processes to shutdown')
      return
    }

    console.log(`[TrainingProcessManager] Shutting down ${running.length} running process(es)...`)

    for (const proc of running) {
      this.killProcess(proc.id, 'SIGTERM')
    }

    const startTime = Date.now()
    while (Date.now() - startTime < timeoutMs) {
      const stillRunning = this.getRunningProcesses()
      if (stillRunning.length === 0) {
        console.log('[TrainingProcessManager] All processes terminated gracefully')
        return
      }
      await new Promise((resolve) => setTimeout(resolve, 500))
    }

    const stillRunning = this.getRunningProcesses()
    if (stillRunning.length > 0) {
      console.log(
        `[TrainingProcessManager] Force killing ${stillRunning.length} process(es) after timeout`,
      )
      for (const proc of stillRunning) {
        this.killProcess(proc.id, 'SIGKILL')
      }
    }
  }
}
