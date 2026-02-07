/**
 * @file training-process-manager.ts
 *
 * @description 训练进程管理器
 *              - 后台启动训练进程（使用 child_process.spawn）
 *              - 实时日志流（stdout/stderr 写入日志文件）
 *              - 进程生命周期管理（服务器关闭时清理所有进程）
 *              - stderr 环形缓冲区 + stdout 事件解析
 *              - 失败分类 + 错误摘要
 *              - waitForStatusChange / waitForEvent 长轮询
 * @author kongzhiquan
 * @date 2026-02-07
 * @version 2.1.0
 *
 * @changelog
 *   - 2026-02-07 kongzhiquan: v2.1.0 修复事件捕获通道
 *     - 新增 stderrEventBuffer + stderr 侧 __event__ 解析（兜底 Python logging → stderr 的事件）
 *     - FAILURE_PATTERNS 新增 ChildFailedError / torch.distributed.elastic 等 DDP 失败模式
 *     - parseStdoutEvents → parseEventMarkers，同时用于 stdout 和 stderr
 *   - 2026-02-07 kongzhiquan: v2.0.0 训练错误实时反馈增强
 *     - 新增 stderr 环形缓冲区（最后 100 行）
 *     - 新增 stdout __event__ 标记解析（training_error / epoch_train / training_start 等）
 *     - 新增 classifyFailure() 失败分类，填充 errorSummary
 *     - 新增 waitForStatusChange() 长轮询方法
 *     - 新增 waitForEvent() 等待特定事件方法
 *     - TrainingProcessInfo 新增 errorSummary / progress 字段
 *     - ManagedProcess 新增 stderrRingBuffer / receivedEvents / lastTrainingError / lastEpochInfo / trainingMeta
 *   - 2026-02-07 kongzhiquan: v1.2.0 增强错误处理
 *     - 移除 EventEmitter 继承，避免未处理的 error 事件导致崩溃
 *     - 日志流写入添加错误处理
 *     - 文件读取操作添加 try-catch
 *   - 2026-02-07 kongzhiquan: v1.1.0 优化日志输出
 *     - stderr 中的正常日志（INFO/DEBUG）直接写入，不加前缀
 *     - 只有真正的错误（ERROR/WARNING/Traceback 等）才加 [STDERR] 前缀
 *   - 2026-02-07 kongzhiquan: v1.0.0 初始版本
 */

import { spawn, ChildProcess } from 'child_process'
import {
  createWriteStream,
  existsSync,
  mkdirSync,
  readFileSync,
  statSync,
  openSync,
  readSync,
  closeSync,
  WriteStream,
} from 'fs'
import path from 'path'

export interface ErrorSummary {
  failureType: string
  errorMessage: string
  lastStderrLines: string[]
  suggestions: string[]
  structuredError?: Record<string, unknown>
}

export interface TrainingProgress {
  currentEpoch: number
  totalEpochs: number
  estimatedRemainingSeconds: number | null
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
  }
  // 错误摘要（失败时填充）
  errorSummary?: ErrorSummary
  // 训练进度（运行时填充）
  progress?: TrainingProgress
}

interface ManagedProcess {
  info: TrainingProcessInfo
  process: ChildProcess
  logStream: WriteStream
  errorLogStream: WriteStream
  // 环形缓冲区：最后 100 行 stderr
  stderrRingBuffer: string[]
  // 来自 Python 的结构化错误事件
  lastTrainingError: Record<string, unknown> | null
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
}

const RING_BUFFER_SIZE = 100

class TrainingProcessManager {
  private processes: Map<string, ManagedProcess> = new Map()
  private isShuttingDown = false

  // 用于判断 stderr 内容是否为真正的错误
  private static readonly ERROR_PATTERNS = [
    /\bERROR\b/i,
    /\bWARNING\b/i,
    /\bWARN\b/i,
    /\bCRITICAL\b/i,
    /\bFATAL\b/i,
    /\bException\b/,
    /\bError\b/,
    /\bTraceback\b/,
    /\bFailed\b/i,
    /CUDA out of memory/i,
    /RuntimeError/,
    /ValueError/,
    /TypeError/,
    /KeyError/,
    /IndexError/,
    /FileNotFoundError/,
    /ModuleNotFoundError/,
    /ImportError/,
    /AssertionError/,
    /AttributeError/,
    /NameError/,
    /ZeroDivisionError/,
    /MemoryError/,
    /OSError/,
    /IOError/,
  ]

  // 失败分类模式
  private static readonly FAILURE_PATTERNS: Array<{
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
      pattern: /ChildFailedError|elastic.*multiprocessing.*errors/i,
      type: 'DDP_CHILD_FAILED',
      suggestion: '某个 GPU rank 崩溃导致 DDP 训练中止，查看 stderr 获取具体 rank 和错误信息',
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

  /**
   * 判断文本是否包含错误信息
   */
  private isErrorContent(text: string): boolean {
    return TrainingProcessManager.ERROR_PATTERNS.some((pattern) => pattern.test(text))
  }

  /**
   * 安全写入日志流
   */
  private safeWrite(stream: WriteStream, data: string): void {
    try {
      if (!stream.destroyed) {
        stream.write(data)
      }
    } catch (err) {
      console.error('[TrainingProcessManager] Failed to write to log stream:', err)
    }
  }

  /**
   * 对失败进程进行错误分类
   */
  private classifyFailure(managed: ManagedProcess): ErrorSummary {
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

    for (const fp of TrainingProcessManager.FAILURE_PATTERNS) {
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
      // 尝试从结构化错误中获取更多信息（base.py 可能已捕获实际错误）
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

  /**
   * 从 managed process 计算训练进度
   */
  private computeProgress(managed: ManagedProcess): TrainingProgress | undefined {
    if (!managed.lastEpochInfo || !managed.trainingMeta) {
      return undefined
    }
    const current = managed.lastEpochInfo.epoch
    const total = managed.trainingMeta.totalEpochs
    const startTime = new Date(managed.trainingMeta.startTimestamp).getTime()
    const epochTime = new Date(managed.lastEpochInfo.timestamp).getTime()
    const elapsed = epochTime - startTime
    const avgPerEpoch = current > 0 ? elapsed / current : 0
    const remaining = avgPerEpoch * (total - current)
    return {
      currentEpoch: current,
      totalEpochs: total,
      estimatedRemainingSeconds: current > 0 ? Math.round(remaining / 1000) : null,
    }
  }

  /**
   * 解析缓冲区中的 __event__ 标记
   * 处理完整的事件后将其从缓冲区移除，保留不完整的尾部
   * @param managed 管理的进程
   * @param source 来源：'stdout' 或 'stderr'
   */
  private parseEventMarkers(managed: ManagedProcess, source: 'stdout' | 'stderr'): void {
    const bufferKey = source === 'stdout' ? 'stdoutBuffer' : 'stderrEventBuffer'
    const EVENT_START = '__event__'
    const EVENT_END = '__event__'

    let searchFrom = 0
    while (true) {
      const startIdx = managed[bufferKey].indexOf(EVENT_START, searchFrom)
      if (startIdx === -1) break

      const contentStart = startIdx + EVENT_START.length
      const endIdx = managed[bufferKey].indexOf(EVENT_END, contentStart)
      if (endIdx === -1) {
        // 不完整的事件，等下一个 chunk
        break
      }

      const jsonStr = managed[bufferKey].substring(contentStart, endIdx)
      searchFrom = endIdx + EVENT_END.length

      try {
        const event = JSON.parse(jsonStr) as Record<string, unknown>
        if (typeof event.event === 'string') {
          managed.receivedEvents.add(event.event)

          if (event.event === 'training_error') {
            managed.lastTrainingError = event
          }

          if (event.event === 'epoch_train' || event.event === 'epoch_valid') {
            managed.lastEpochInfo = {
              epoch: event.epoch as number,
              timestamp: event.timestamp as string,
              metrics: (event.metrics as Record<string, number>) ?? {},
            }
          }

          if (event.event === 'training_start') {
            managed.trainingMeta = {
              totalEpochs: event.total_epochs as number,
              startTimestamp: event.timestamp as string,
            }
          }
        }
      } catch {
        /* ignore parse errors */
      }
    }

    // 保留从最后一个未完成的 __event__ 开始的内容
    const lastStart = managed[bufferKey].lastIndexOf(EVENT_START, searchFrom > 0 ? searchFrom - 1 : managed[bufferKey].length)
    if (lastStart >= searchFrom) {
      // 有不完整的事件标记
      managed[bufferKey] = managed[bufferKey].substring(lastStart)
    } else {
      // 所有事件都已处理
      managed[bufferKey] = ''
    }
  }

  /**
   * 启动一个后台训练进程
   */
  startProcess(options: {
    cmd: string
    args: string[]
    cwd: string
    logDir: string
    env?: Record<string, string>
    metadata?: TrainingProcessInfo['metadata']
  }): TrainingProcessInfo {
    const { cmd, args, cwd, logDir, env, metadata } = options

    // 生成唯一 ID
    const id = `train-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

    // 确保日志目录存在
    if (!existsSync(logDir)) {
      mkdirSync(logDir, { recursive: true })
    }

    const logFile = path.join(logDir, `${id}.log`)
    const errorLogFile = path.join(logDir, `${id}.error.log`)

    // 创建日志流
    const logStream = createWriteStream(logFile, { flags: 'a' })
    const errorLogStream = createWriteStream(errorLogFile, { flags: 'a' })

    // 处理日志流错误，避免未捕获异常
    logStream.on('error', (err) => {
      console.error(`[TrainingProcessManager] Log stream error for ${id}:`, err)
    })
    errorLogStream.on('error', (err) => {
      console.error(`[TrainingProcessManager] Error log stream error for ${id}:`, err)
    })

    // 写入启动信息
    const startHeader = `
================================================================================
Training Process Started
ID: ${id}
Command: ${cmd} ${args.join(' ')}
Working Directory: ${cwd}
Start Time: ${new Date().toISOString()}
================================================================================

`
    this.safeWrite(logStream, startHeader)

    // 合并环境变量
    const processEnv = {
      ...process.env,
      ...env,
      // 确保 Python 输出不缓冲
      PYTHONUNBUFFERED: '1',
    }

    // 启动进程
    const childProcess = spawn(cmd, args, {
      cwd,
      env: processEnv,
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: false, // 不分离，确保父进程退出时子进程也退出
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
      stdoutBuffer: '',
      stderrEventBuffer: '',
      receivedEvents: new Set(),
      lastEpochInfo: null,
      trainingMeta: null,
    }

    // 管道 stdout 到日志文件 + 解析事件
    if (childProcess.stdout) {
      childProcess.stdout.on('data', (data: Buffer) => {
        const text = data.toString()
        this.safeWrite(logStream, text)
        // 追加到缓冲区并解析事件
        managed.stdoutBuffer += text
        this.parseEventMarkers(managed, 'stdout')
      })
    }

    // 管道 stderr 到错误日志文件（同时也写入主日志）+ 环形缓冲区 + 事件解析
    // 只有真正的错误才加 [STDERR] 前缀
    if (childProcess.stderr) {
      childProcess.stderr.on('data', (data: Buffer) => {
        const text = data.toString()
        // 判断是否为真正的错误内容
        if (this.isErrorContent(text)) {
          this.safeWrite(logStream, `[STDERR] ${text}`)
        } else {
          this.safeWrite(logStream, text)
        }
        this.safeWrite(errorLogStream, text)

        // 维护环形缓冲区
        const lines = text.split('\n').filter((l) => l.trim())
        managed.stderrRingBuffer.push(...lines)
        if (managed.stderrRingBuffer.length > RING_BUFFER_SIZE) {
          managed.stderrRingBuffer.splice(
            0,
            managed.stderrRingBuffer.length - RING_BUFFER_SIZE,
          )
        }

        // 兜底：解析 stderr 中的 __event__ 标记
        // Python logging.info → StreamHandler → stderr，事件可能通过此通道到达
        if (text.includes('__event__')) {
          managed.stderrEventBuffer += text
          this.parseEventMarkers(managed, 'stderr')
        }
      })
    }

    // 监听进程退出
    childProcess.on('exit', (code, signal) => {
      managed.info.endTime = Date.now()
      managed.info.exitCode = code ?? undefined

      if (signal === 'SIGTERM' || signal === 'SIGKILL') {
        managed.info.status = 'killed'
      } else if (code === 0) {
        managed.info.status = 'completed'
      } else {
        managed.info.status = 'failed'
        // 失败时进行错误分类
        managed.info.errorSummary = this.classifyFailure(managed)
      }

      // 写入结束信息
      const endFooter = `
================================================================================
Training Process ${managed.info.status.toUpperCase()}
Exit Code: ${code}
Signal: ${signal || 'none'}
End Time: ${new Date().toISOString()}
Duration: ${((managed.info.endTime - managed.info.startTime) / 1000).toFixed(1)}s
================================================================================
`
      this.safeWrite(logStream, endFooter)

      // 关闭日志流
      logStream.end()
      errorLogStream.end()
    })

    childProcess.on('error', (err) => {
      managed.info.status = 'failed'
      managed.info.endTime = Date.now()
      managed.info.errorSummary = this.classifyFailure(managed)
      this.safeWrite(errorLogStream, `Process error: ${err.message}\n`)
      this.safeWrite(logStream, `[ERROR] Process error: ${err.message}\n`)
      logStream.end()
      errorLogStream.end()
    })

    // 存储进程信息
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

    // 动态计算进度
    if (managed.info.status === 'running') {
      managed.info.progress = this.computeProgress(managed)
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
   * 读取进程日志（支持增量读取）
   */
  readLogs(
    id: string,
    options?: {
      tail?: number // 只读取最后 N 行
      offset?: number // 从字节偏移量开始读取
    },
  ): { content: string; size: number; offset: number } | undefined {
    const managed = this.processes.get(id)
    if (!managed) return undefined

    const { logFile } = managed.info

    try {
      if (!existsSync(logFile)) {
        return { content: '', size: 0, offset: 0 }
      }

      const stat = statSync(logFile)
      const size = stat.size

      if (options?.tail) {
        // 读取最后 N 行
        const content = readFileSync(logFile, 'utf-8')
        const lines = content.split('\n')
        const tailLines = lines.slice(-options.tail).join('\n')
        return { content: tailLines, size, offset: size }
      }

      if (options?.offset !== undefined && options.offset < size) {
        // 增量读取
        const buffer = Buffer.alloc(size - options.offset)
        const fd = openSync(logFile, 'r')
        try {
          readSync(fd, buffer, 0, buffer.length, options.offset)
        } finally {
          closeSync(fd)
        }
        return { content: buffer.toString('utf-8'), size, offset: size }
      }

      // 读取全部
      const content = readFileSync(logFile, 'utf-8')
      return { content, size, offset: size }
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
   * 等待进程状态变化（长轮询）
   */
  async waitForStatusChange(
    id: string,
    opts: { timeoutMs?: number; pollIntervalMs?: number } = {},
  ): Promise<TrainingProcessInfo | undefined> {
    const { timeoutMs = 60000, pollIntervalMs = 2000 } = opts
    const start = Date.now()
    const initial = this.getProcess(id)
    if (!initial) return undefined
    // 如果进程已结束，直接返回（不会再变）
    if (initial.status !== 'running') return initial
    const initialStatus = initial.status
    while (Date.now() - start < timeoutMs) {
      const current = this.getProcess(id)
      if (!current) return undefined
      if (current.status !== initialStatus) return current
      await new Promise((r) => setTimeout(r, pollIntervalMs))
    }
    return this.getProcess(id)
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

      // 检查是否收到目标事件
      if (managed.receivedEvents.has(eventType)) {
        return { found: true, processStatus: managed.info.status }
      }

      // 进程已退出 → 不用再等了
      if (managed.info.status !== 'running') {
        return { found: false, processStatus: managed.info.status }
      }

      await new Promise((r) => setTimeout(r, 1000))
    }
    return { found: false, processStatus: this.getProcess(id)?.status ?? 'timeout' }
  }

  /**
   * 清理已完成的进程记录（保留日志文件）
   */
  cleanupCompleted(): number {
    let count = 0
    for (const [id, managed] of this.processes) {
      if (managed.info.status !== 'running') {
        this.processes.delete(id)
        count++
      }
    }
    return count
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

    // 先发送 SIGTERM
    for (const proc of running) {
      this.killProcess(proc.id, 'SIGTERM')
    }

    // 等待进程退出
    const startTime = Date.now()
    while (Date.now() - startTime < timeoutMs) {
      const stillRunning = this.getRunningProcesses()
      if (stillRunning.length === 0) {
        console.log('[TrainingProcessManager] All processes terminated gracefully')
        return
      }
      await new Promise((resolve) => setTimeout(resolve, 500))
    }

    // 超时后强制杀死
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

// 单例导出
export const trainingProcessManager = new TrainingProcessManager()
