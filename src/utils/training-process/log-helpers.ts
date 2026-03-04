/**
 * @file log-helpers.ts
 *
 * @description 训练进程管理器 —— 日志写入与滚动辅助函数
 * @author kongzhiquan
 * @date 2026-03-04
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-03-04 kongzhiquan: v1.0.0 从 training-process-manager.ts 拆分
 */

import { createWriteStream, existsSync, renameSync, statSync, unlinkSync, WriteStream } from 'fs'
import { MAX_LOG_BYTES, MAX_LOG_ROTATIONS } from './constants'
import { ManagedProcess } from './types'

export function safeWrite(stream: WriteStream, data: string): void {
  try {
    if (!stream.destroyed) {
      stream.write(data)
    }
  } catch (err) {
    console.error('[TrainingProcessManager] Failed to write to log stream:', err)
  }
}

export function attachStreamErrorHandler(stream: WriteStream, id: string, kind: 'log' | 'error'): void {
  stream.on('error', (err) => {
    const label = kind === 'log' ? 'Log' : 'Error log'
    console.error(`[TrainingProcessManager] ${label} stream error for ${id}:`, err)
  })
}

export function rotateLogIfNeeded(managed: ManagedProcess, kind: 'log' | 'error'): void {
  if (MAX_LOG_ROTATIONS <= 0 || MAX_LOG_BYTES <= 0) return
  const logFile = kind === 'log' ? managed.info.logFile : managed.info.errorLogFile
  const rotatingKey = kind === 'log' ? 'rotatingLog' : 'rotatingErrorLog'
  if (managed[rotatingKey]) return

  try {
    if (!existsSync(logFile)) return
    const size = statSync(logFile).size
    if (size < MAX_LOG_BYTES) return

    managed[rotatingKey] = true

    const stream = kind === 'log' ? managed.logStream : managed.errorLogStream
    stream.end()

    for (let i = MAX_LOG_ROTATIONS; i >= 1; i--) {
      const src = i === 1 ? logFile : `${logFile}.${i - 1}`
      const dest = `${logFile}.${i}`
      if (i === MAX_LOG_ROTATIONS && existsSync(dest)) {
        unlinkSync(dest)
      }
      if (existsSync(src)) {
        renameSync(src, dest)
      }
    }

    const newStream = createWriteStream(logFile, { flags: 'a' })
    attachStreamErrorHandler(newStream, managed.info.id, kind)
    if (kind === 'log') {
      managed.logStream = newStream
    } else {
      managed.errorLogStream = newStream
    }
    safeWrite(newStream, `\n[Log rotated at ${new Date().toISOString()}]\n`)
  } catch (err) {
    console.error(`[TrainingProcessManager] Failed to rotate ${kind} log:`, err)
  } finally {
    managed[rotatingKey] = false
  }
}

export function writeLog(managed: ManagedProcess, kind: 'log' | 'error', data: string): void {
  rotateLogIfNeeded(managed, kind)
  const stream = kind === 'log' ? managed.logStream : managed.errorLogStream
  safeWrite(stream, data)
}
