/**
 * @file training-process-manager.ts
 *
 * @description 训练进程管理器
 *              - 后台启动训练进程（使用 child_process.spawn）
 *              - 实时日志流（stdout/stderr 写入日志文件）
 *              - 进程生命周期管理（服务器关闭时清理所有进程）
 * @author kongzhiquan
 * @date 2026-02-07
 * @version 1.2.0
 *
 * @changelog
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
}

interface ManagedProcess {
  info: TrainingProcessInfo
  process: ChildProcess
  logStream: WriteStream
  errorLogStream: WriteStream
}

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

    // 管道 stdout 到日志文件
    if (childProcess.stdout) {
      childProcess.stdout.on('data', (data: Buffer) => {
        this.safeWrite(logStream, data.toString())
      })
    }

    // 管道 stderr 到错误日志文件（同时也写入主日志）
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
      })
    }

    // 监听进程退出
    childProcess.on('exit', (code, signal) => {
      const managed = this.processes.get(id)
      if (managed) {
        managed.info.endTime = Date.now()
        managed.info.exitCode = code ?? undefined

        if (signal === 'SIGTERM' || signal === 'SIGKILL') {
          managed.info.status = 'killed'
        } else if (code === 0) {
          managed.info.status = 'completed'
        } else {
          managed.info.status = 'failed'
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
      }
    })

    childProcess.on('error', (err) => {
      const managed = this.processes.get(id)
      if (managed) {
        managed.info.status = 'failed'
        managed.info.endTime = Date.now()
        this.safeWrite(errorLogStream, `Process error: ${err.message}\n`)
        this.safeWrite(logStream, `[ERROR] Process error: ${err.message}\n`)
        logStream.end()
        errorLogStream.end()
      }
    })

    // 存储进程信息
    this.processes.set(id, {
      info,
      process: childProcess,
      logStream,
      errorLogStream,
    })

    console.log(`[TrainingProcessManager] Started process ${id} (PID: ${childProcess.pid})`)

    return info
  }

  /**
   * 获取进程信息
   */
  getProcess(id: string): TrainingProcessInfo | undefined {
    return this.processes.get(id)?.info
  }

  /**
   * 获取所有进程信息
   */
  getAllProcesses(): TrainingProcessInfo[] {
    return Array.from(this.processes.values()).map((p) => p.info)
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
      console.log(`[TrainingProcessManager] Sent ${signal} to process ${id} (PID: ${managed.info.pid})`)
      return true
    } catch (err) {
      console.error(`[TrainingProcessManager] Failed to kill process ${id}:`, err)
      return false
    }
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
