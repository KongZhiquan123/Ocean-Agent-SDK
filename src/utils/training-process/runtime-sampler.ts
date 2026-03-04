/**
 * @file runtime-sampler.ts
 *
 * @description 训练进程管理器 —— 运行时资源采样（CPU/内存/IO/GPU）
 * @author kongzhiquan
 * @date 2026-03-04
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-03-04 kongzhiquan: v1.0.0 从 training-process-manager.ts 拆分
 */

import { execFileSync } from 'child_process'
import { readFileSync } from 'fs'
import { PROCESS_STATS_TIMEOUT_MS } from './constants'
import { ManagedProcess, RuntimeStats } from './types'

export function sampleRuntimeStats(managed: ManagedProcess): void {
  const now = Date.now()

  const stats: RuntimeStats = {
    sampledAt: new Date(now).toISOString(),
    uptimeSeconds: Math.round((now - managed.info.startTime) / 1000),
    cpuPercent: null,
    memoryRssMB: null,
    ioReadMB: null,
    ioWriteMB: null,
  }

  const pid = managed.info.pid
  if (pid) {
    try {
      const output = execFileSync(
        'ps',
        ['-p', String(pid), '-o', '%cpu,%mem,rss'],
        { encoding: 'utf-8', timeout: PROCESS_STATS_TIMEOUT_MS },
      ).trim()
      const lines = output.split(/\r?\n/)
      if (lines.length >= 2) {
        const parts = lines[1].trim().split(/\s+/)
        if (parts.length >= 3) {
          const cpu = Number(parts[0])
          const rssKb = Number(parts[2])
          stats.cpuPercent = Number.isFinite(cpu) ? cpu : null
          stats.memoryRssMB = Number.isFinite(rssKb) ? Math.round((rssKb / 1024) * 10) / 10 : null
        }
      }
    } catch {
      // ignore ps failures
    }

    try {
      const ioText = readFileSync(`/proc/${pid}/io`, 'utf-8')
      const readMatch = ioText.match(/read_bytes:\s*(\d+)/)
      const writeMatch = ioText.match(/write_bytes:\s*(\d+)/)
      if (readMatch) {
        const readBytes = Number(readMatch[1])
        if (Number.isFinite(readBytes)) {
          stats.ioReadMB = Math.round((readBytes / 1024 / 1024) * 10) / 10
        }
      }
      if (writeMatch) {
        const writeBytes = Number(writeMatch[1])
        if (Number.isFinite(writeBytes)) {
          stats.ioWriteMB = Math.round((writeBytes / 1024 / 1024) * 10) / 10
        }
      }
    } catch {
      // ignore non-linux or permission errors
    }
  }

  try {
    const output = execFileSync(
      'nvidia-smi',
      [
        '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
        '--format=csv,noheader,nounits',
      ],
      { encoding: 'utf-8', timeout: PROCESS_STATS_TIMEOUT_MS },
    ).trim()
    if (output) {
      const deviceSet = managed.info.metadata?.deviceIds
        ? new Set(managed.info.metadata.deviceIds)
        : null
      const gpuStats: NonNullable<RuntimeStats['gpu']> = []
      for (const line of output.split(/\r?\n/)) {
        const parts = line.split(',').map((p) => p.trim())
        if (parts.length < 6) continue
        const id = Number(parts[0])
        if (deviceSet && !deviceSet.has(id)) continue
        const toNum = (val: string) => {
          const num = Number(val)
          return Number.isFinite(num) ? num : null
        }
        gpuStats.push({
          id,
          utilizationPct: toNum(parts[1]),
          memoryUsedMB: toNum(parts[2]),
          memoryTotalMB: toNum(parts[3]),
          temperatureC: toNum(parts[4]),
          powerW: toNum(parts[5]),
        })
      }
      if (gpuStats.length > 0) {
        stats.gpu = gpuStats
      }
    }
  } catch {
    // nvidia-smi not available or failed
  }

  managed.runtimeStats = stats
  managed.info.runtimeStats = stats
}
