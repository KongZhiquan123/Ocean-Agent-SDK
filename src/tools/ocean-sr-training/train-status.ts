/**
 * @file train-status.ts
 *
 * @description 训练状态查询工具
 *              - 查询训练进程状态（含进度和错误详情）
 *              - 获取实时日志（支持增量读取）
 *              - 终止训练进程
 *              - 列出所有训练进程
 *              - 等待状态变化（wait 模式）
 * @author Leizheng
 * @date 2026-02-07
 * @version 2.0.0
 *
 * @changelog
 *   - 2026-02-07 Leizheng: v2.0.0 训练错误实时反馈增强
 *     - 新增 action="wait" 长轮询模式，等待训练状态变化
 *     - 新增 timeout 参数（wait 模式超时秒数）
 *     - status 查询增强：running 时附加 progress，failed 时附加 error_summary
 *     - failed 时自动读取 error.log 最后 30 行作为 fallback
 *   - 2026-02-07 Leizheng: v1.0.0 初始版本
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { trainingProcessManager } from '@/utils/training-process-manager'

export const oceanSrTrainStatusTool = defineTool({
  name: 'ocean_sr_train_status',
  description: `查询训练进程状态、获取日志或终止训练。

**查询状态**：传入 process_id 获取训练进程的当前状态（含进度和错误详情）
**获取日志**：传入 process_id 和 tail 参数获取最新日志
**增量日志**：传入 process_id 和 offset 参数获取自上次读取后的新日志
**终止训练**：传入 action="kill" 和 process_id 终止训练进程
**列出所有**：传入 action="list" 列出所有训练进程
**等待变化**：传入 action="wait" 和 process_id，等待训练状态变化（长轮询）`,

  params: {
    action: {
      type: 'string',
      description: '操作类型: "status"(默认), "logs", "kill", "list", "wait"',
      required: false,
      default: 'status',
    },
    process_id: {
      type: 'string',
      description: '训练进程 ID（从 ocean_sr_train 返回值获取）',
      required: false,
    },
    tail: {
      type: 'number',
      description: '获取最后 N 行日志（默认 100）',
      required: false,
      default: 100,
    },
    offset: {
      type: 'number',
      description: '日志字节偏移量（用于增量读取，从上次返回的 offset 值开始）',
      required: false,
    },
    timeout: {
      type: 'number',
      description: 'wait 模式的超时秒数（默认 120）',
      required: false,
      default: 120,
    },
  },

  async exec(args) {
    const { action = 'status', process_id, tail = 100, offset, timeout = 120 } = args

    // 列出所有进程
    if (action === 'list') {
      const all = trainingProcessManager.getAllProcesses()
      const running = all.filter((p) => p.status === 'running')
      const completed = all.filter((p) => p.status !== 'running')

      return {
        status: 'ok',
        total: all.length,
        running: running.length,
        completed: completed.length,
        processes: all.map((p) => ({
          id: p.id,
          status: p.status,
          pid: p.pid,
          model: p.metadata?.modelName,
          startTime: new Date(p.startTime).toISOString(),
          endTime: p.endTime ? new Date(p.endTime).toISOString() : undefined,
          exitCode: p.exitCode,
          logFile: p.logFile,
          progress: p.progress,
          errorSummary: p.errorSummary
            ? {
                failureType: p.errorSummary.failureType,
                errorMessage: p.errorSummary.errorMessage,
                suggestions: p.errorSummary.suggestions,
              }
            : undefined,
        })),
      }
    }

    // 其他操作需要 process_id
    if (!process_id) {
      return {
        status: 'error',
        error: '缺少 process_id 参数',
        suggestion: '请提供训练进程 ID，可通过 action="list" 查看所有进程',
      }
    }

    // wait 模式：等待状态变化
    if (action === 'wait') {
      const timeoutMs = (timeout ?? 120) * 1000
      const result = await trainingProcessManager.waitForStatusChange(process_id, { timeoutMs })
      if (!result) {
        return {
          status: 'error',
          error: `未找到进程: ${process_id}`,
          suggestion: '进程可能已被清理或 ID 不正确',
        }
      }

      // 返回完整状态信息（与 status 查询一致）
      return buildStatusResponse(process_id, result)
    }

    const processInfo = trainingProcessManager.getProcess(process_id)
    if (!processInfo) {
      return {
        status: 'error',
        error: `未找到进程: ${process_id}`,
        suggestion: '进程可能已被清理或 ID 不正确，请使用 action="list" 查看所有进程',
      }
    }

    // 终止进程
    if (action === 'kill') {
      if (processInfo.status !== 'running') {
        return {
          status: 'error',
          error: `进程已结束，状态: ${processInfo.status}`,
          exitCode: processInfo.exitCode,
        }
      }

      const killed = trainingProcessManager.killProcess(process_id)
      if (killed) {
        return {
          status: 'ok',
          message: `已发送终止信号到进程 ${process_id} (PID: ${processInfo.pid})`,
          process_id,
          pid: processInfo.pid,
        }
      } else {
        return {
          status: 'error',
          error: '终止进程失败',
          process_id,
        }
      }
    }

    // 获取日志
    if (action === 'logs') {
      const logsResult = trainingProcessManager.readLogs(process_id, {
        tail: offset === undefined ? tail : undefined,
        offset,
      })

      if (!logsResult) {
        return {
          status: 'error',
          error: '无法读取日志',
          process_id,
        }
      }

      return {
        status: 'ok',
        process_id,
        process_status: processInfo.status,
        exitCode: processInfo.exitCode,
        logs: logsResult.content,
        log_size: logsResult.size,
        offset: logsResult.offset,
        next_offset: logsResult.offset,
        tip:
          processInfo.status === 'running'
            ? `进程仍在运行，可使用 offset=${logsResult.offset} 获取后续日志`
            : '进程已结束',
      }
    }

    // 默认：查询状态
    return buildStatusResponse(process_id, processInfo)
  },
})

/**
 * 构建状态查询响应（status 和 wait 共用）
 */
function buildStatusResponse(
  process_id: string,
  processInfo: ReturnType<typeof trainingProcessManager.getProcess> & object,
) {
  const duration = processInfo.endTime
    ? processInfo.endTime - processInfo.startTime
    : Date.now() - processInfo.startTime

  // 基础信息
  const response: Record<string, unknown> = {
    status: 'ok',
    process_id,
    process_status: processInfo.status,
    pid: processInfo.pid,
    exitCode: processInfo.exitCode,
    startTime: new Date(processInfo.startTime).toISOString(),
    endTime: processInfo.endTime ? new Date(processInfo.endTime).toISOString() : undefined,
    duration_seconds: Math.round(duration / 1000),
    duration_human: formatDuration(duration),
    metadata: processInfo.metadata,
    logFile: processInfo.logFile,
    errorLogFile: processInfo.errorLogFile,
  }

  // running 时附加进度信息
  if (processInfo.status === 'running' && processInfo.progress) {
    response.progress = processInfo.progress
  }

  // failed 时附加错误详情
  if (processInfo.status === 'failed') {
    if (processInfo.errorSummary) {
      response.error_summary = processInfo.errorSummary
      response.suggestions = processInfo.errorSummary.suggestions
    } else {
      // fallback：读取 error.log 最后 30 行
      const errorLogs = trainingProcessManager.readLogs(process_id, { tail: 30 })
      if (errorLogs) {
        response.error_log_tail = errorLogs.content
      }
    }
  }

  // 操作提示
  response.actions =
    processInfo.status === 'running'
      ? [
          `查看日志: ocean_sr_train_status({ action: "logs", process_id: "${process_id}", tail: 50 })`,
          `等待完成: ocean_sr_train_status({ action: "wait", process_id: "${process_id}", timeout: 120 })`,
          `终止训练: ocean_sr_train_status({ action: "kill", process_id: "${process_id}" })`,
        ]
      : [
          `查看完整日志: ocean_sr_train_status({ action: "logs", process_id: "${process_id}", tail: 200 })`,
        ]

  return response
}

function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000)
  const minutes = Math.floor(seconds / 60)
  const hours = Math.floor(minutes / 60)
  const days = Math.floor(hours / 24)

  if (days > 0) {
    return `${days}d ${hours % 24}h ${minutes % 60}m`
  }
  if (hours > 0) {
    return `${hours}h ${minutes % 60}m ${seconds % 60}s`
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`
  }
  return `${seconds}s`
}
