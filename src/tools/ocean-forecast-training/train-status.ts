/**
 * @file train-status.ts
 *
 * @description Ocean forecast training status query tool.
 *              Reuses shared trainingProcessManager.
 * @author Leizheng
 * @date 2026-02-26
 * @version 1.3.0
 *
 * @changelog
 *   - 2026-02-27 Leizheng: v1.3.0 strengthen auto-predict note to explicitly warn against calling ocean_forecast_train predict mode
 *   - 2026-02-27 Leizheng: v1.2.0 add auto-predict note + predict visualization action to training completion
 *   - 2026-02-26 Leizheng: v1.1.0 completed state returns progress + visualization/report actions
 *   - 2026-02-26 Leizheng: v1.0.0 initial version for ocean forecast training
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { trainingProcessManager } from '@/utils/training-process-manager'

export const oceanForecastTrainStatusTool = defineTool({
  name: 'ocean_forecast_train_status',
  description: `查询海洋预测训练进程状态、获取日志或终止训练。

**查询状态**：传入 process_id 获取训练进程的当前状态（含进度和错误详情）
**获取日志**：传入 process_id 和 tail 参数获取最新日志
**增量日志**：传入 process_id 和 offset 参数获取自上次读取后的新日志
**终止训练**：传入 action="kill" 和 process_id 终止训练进程
**列出所有**：传入 action="list" 列出所有训练进程
**等待变化**：传入 action="wait" 和 process_id，等待训练状态变化（长轮询）
**推送通知**：传入 action="watch" 和 process_id，等待训练启动/报错/结束事件（长轮询）`,

  params: {
    action: {
      type: 'string',
      description: '操作类型: "status"(默认), "logs", "kill", "list", "wait", "watch"',
      required: false,
      default: 'status',
    },
    process_id: {
      type: 'string',
      description: '训练进程 ID（从 ocean_forecast_train 返回值获取）',
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
      description: 'wait/watch 模式的超时秒数（默认 120）',
      required: false,
      default: 120,
    },
  },

  async exec(args) {
    const { action = 'status', process_id, tail = 100, offset, timeout = 120 } = args

    // List all processes
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
          runtime_stats: p.status === 'running' ? p.runtimeStats : undefined,
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

    if (!process_id) {
      return {
        status: 'error',
        error: '缺少 process_id 参数',
        suggestion: '请提供训练进程 ID，可通过 action="list" 查看所有进程',
      }
    }

    // wait mode
    if (action === 'wait') {
      const timeoutMs = (timeout ?? 120) * 1000
      const result = await trainingProcessManager.waitForChange(process_id, { mode: 'status', timeoutMs })
      if (!result.processInfo) {
        return {
          status: 'error',
          error: `未找到进程: ${process_id}`,
          suggestion: '进程可能已被清理或 ID 不正确',
        }
      }
      return buildStatusResponse(process_id, result.processInfo)
    }

    // watch mode
    if (action === 'watch') {
      const timeoutMs = (timeout ?? 120) * 1000
      const result = await trainingProcessManager.waitForChange(process_id, { mode: 'notification', timeoutMs })
      if (result.processStatus === 'unknown') {
        return {
          status: 'error',
          error: `未找到进程: ${process_id}`,
          suggestion: '进程可能已被清理或 ID 不正确',
        }
      }

      const watchResponse: Record<string, unknown> = {
        status: 'ok',
        process_id,
        process_status: result.processStatus,
        pushed: result.found,
        notification: result.notification,
        message: result.found
          ? getNotificationMessage(result.notification?.type)
          : '等待超时：暂无新的关键事件（启动成功/报错/结束）',
        error_summary:
          result.notification?.type === 'process_exit' && result.processInfo?.status === 'failed'
            ? result.processInfo.errorSummary
            : undefined,
      }

      // Append actions when process has finished so Agent knows the next step.
      if (result.processStatus === 'completed' || result.processStatus === 'failed') {
        const logDir = result.processInfo?.metadata?.logDir as string | undefined
        const isPredict = result.processInfo?.metadata?.mode === 'predict'
        if (result.processStatus === 'completed') {
          if (isPredict) {
            watchResponse.actions = [
              `生成预测可视化: ocean_forecast_train_visualize({ log_dir: "${logDir ?? '...'}", mode: "predict" })`,
            ]
          } else {
            watchResponse.actions = [
              `生成训练可视化: ocean_forecast_train_visualize({ log_dir: "${logDir ?? '...'}", mode: "train" })`,
              `生成预测可视化: ocean_forecast_train_visualize({ log_dir: "${logDir ?? '...'}", mode: "predict" })`,
              `生成训练报告: ocean_forecast_train_report({ log_dir: "${logDir ?? '...'}" })`,
            ]
            watchResponse.note = '训练结束时已自动运行推理（auto-predict），predictions/ 目录已包含测试集预测结果。请直接调用上面的 visualize 和 report 工具生成图表与报告。【重要】不要再调用 ocean_forecast_train({ mode: "predict" })，否则会重复推理。'
          }
        }
      }

      return watchResponse
    }

    const processInfo = trainingProcessManager.getProcess(process_id)
    if (!processInfo) {
      return {
        status: 'error',
        error: `未找到进程: ${process_id}`,
        suggestion: '进程可能已被清理或 ID 不正确，请使用 action="list" 查看所有进程',
      }
    }

    // Kill
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

    // Logs
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

      const logsResponse: Record<string, unknown> = {
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

      // Append next-step actions for completed/failed processes so Agent
      // can discover the visualization / report workflow from logs queries too.
      if (processInfo.status !== 'running') {
        const logDir = processInfo.metadata?.logDir as string | undefined
        const isPredict = processInfo.metadata?.mode === 'predict'
        if (processInfo.status === 'completed') {
          if (isPredict) {
            logsResponse.actions = [
              `生成预测可视化: ocean_forecast_train_visualize({ log_dir: "${logDir ?? '...'}", mode: "predict" })`,
            ]
          } else {
            logsResponse.actions = [
              `生成训练可视化: ocean_forecast_train_visualize({ log_dir: "${logDir ?? '...'}", mode: "train" })`,
              `生成预测可视化: ocean_forecast_train_visualize({ log_dir: "${logDir ?? '...'}", mode: "predict" })`,
              `生成训练报告: ocean_forecast_train_report({ log_dir: "${logDir ?? '...'}" })`,
            ]
            logsResponse.note = '训练结束时已自动运行推理（auto-predict），predictions/ 目录已包含测试集预测结果。请直接调用上面的 visualize 和 report 工具生成图表与报告。【重要】不要再调用 ocean_forecast_train({ mode: "predict" })，否则会重复推理。'
          }
        }
      }

      return logsResponse
    }

    // Default: status query
    return buildStatusResponse(process_id, processInfo)
  },
})

function buildStatusResponse(
  process_id: string,
  processInfo: ReturnType<typeof trainingProcessManager.getProcess> & object,
) {
  const duration = processInfo.endTime
    ? processInfo.endTime - processInfo.startTime
    : Date.now() - processInfo.startTime

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

  // Show progress for both running and completed states (final progress snapshot)
  if (processInfo.progress) {
    response.progress = processInfo.progress
  }

  if (processInfo.status === 'running' && processInfo.runtimeStats) {
    response.runtime_stats = processInfo.runtimeStats
  }

  if (processInfo.status === 'failed') {
    if (processInfo.errorSummary) {
      response.error_summary = processInfo.errorSummary
      response.suggestions = processInfo.errorSummary.suggestions
    } else {
      const errorLogs = trainingProcessManager.readLogs(process_id, { tail: 30 })
      if (errorLogs) {
        response.error_log_tail = errorLogs.content
      }
    }
  }

  const isPredict = processInfo.metadata?.mode === 'predict'
  const actionLabel = isPredict ? '推理' : '训练'
  const logDir = processInfo.metadata?.logDir as string | undefined

  if (processInfo.status === 'running') {
    response.actions = [
      `查看日志: ocean_forecast_train_status({ action: "logs", process_id: "${process_id}", tail: 50 })`,
      `等待完成: ocean_forecast_train_status({ action: "wait", process_id: "${process_id}", timeout: 120 })`,
      `等待推送: ocean_forecast_train_status({ action: "watch", process_id: "${process_id}", timeout: 300 })`,
      `终止${actionLabel}: ocean_forecast_train_status({ action: "kill", process_id: "${process_id}" })`,
    ]
  } else if (processInfo.status === 'completed') {
    const actions = [
      `查看完整日志: ocean_forecast_train_status({ action: "logs", process_id: "${process_id}", tail: 200 })`,
    ]
    if (isPredict) {
      actions.push(
        `生成预测可视化: ocean_forecast_train_visualize({ log_dir: "${logDir ?? '...'}", mode: "predict" })`,
      )
    } else {
      actions.push(
        `生成训练可视化: ocean_forecast_train_visualize({ log_dir: "${logDir ?? '...'}", mode: "train" })`,
        `生成预测可视化: ocean_forecast_train_visualize({ log_dir: "${logDir ?? '...'}", mode: "predict" })`,
        `生成训练报告: ocean_forecast_train_report({ log_dir: "${logDir ?? '...'}" })`,
      )
      response.note = '训练结束时已自动运行推理（auto-predict），predictions/ 目录已包含测试集预测结果。请直接调用上面的 visualize 和 report 工具生成图表与报告。【重要】不要再调用 ocean_forecast_train({ mode: "predict" })，否则会重复推理。'
    }
    response.actions = actions
  } else {
    // failed / killed
    response.actions = [
      `查看完整日志: ocean_forecast_train_status({ action: "logs", process_id: "${process_id}", tail: 200 })`,
    ]
  }

  return response
}

function getNotificationMessage(type?: string): string {
  switch (type) {
    case 'training_start':
      return '训练启动成功，已开始执行。'
    case 'training_end':
      return '训练已完成，最终指标见 notification.payload。建议接下来生成可视化图表和训练报告。'
    case 'predict_start':
      return '推理启动成功，已开始执行。'
    case 'training_error':
      return '过程中捕获到错误事件。'
    case 'predict_end':
      return '推理已完成。建议接下来生成预测可视化图表。'
    case 'process_exit':
      return '进程已结束。'
    default:
      return '收到关键事件。'
  }
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
