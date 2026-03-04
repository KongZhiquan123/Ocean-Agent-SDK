/**
 * @file event-parser.ts
 *
 * @description 训练进程管理器 —— __event__ 标记解析与通知管理
 * @author kongzhiquan
 * @date 2026-03-04
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-03-04 kongzhiquan: v1.0.0 从 training-process-manager.ts 拆分
 */

import { MAX_NOTIFICATION_QUEUE } from './constants'
import { ManagedProcess, TrainingNotification } from './types'

export function createNotification(
  managed: ManagedProcess,
  type: TrainingNotification['type'],
  payload?: Record<string, unknown>,
): TrainingNotification {
  return {
    id: `notify-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    type,
    timestamp: new Date().toISOString(),
    processId: managed.info.id,
    payload,
  }
}

export function pushNotification(managed: ManagedProcess, notification: TrainingNotification): void {
  managed.notifications.push(notification)
  if (managed.notifications.length > MAX_NOTIFICATION_QUEUE) {
    managed.notifications.splice(0, managed.notifications.length - MAX_NOTIFICATION_QUEUE)
  }
}

/**
 * 解析缓冲区中的 __event__ 标记
 * 处理完整的事件后将其从缓冲区移除，保留不完整的尾部
 */
export function parseEventMarkers(managed: ManagedProcess, source: 'stdout' | 'stderr'): void {
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
        const eventType = event.event
        const isFirst = !managed.receivedEvents.has(eventType)
        managed.receivedEvents.add(eventType)

        if (eventType === 'training_error') {
          managed.lastTrainingError = event
          if (isFirst) {
            pushNotification(managed, createNotification(managed, 'training_error', event))
          }
        }

        if (eventType === 'epoch_train' || eventType === 'epoch_valid') {
          managed.lastEpochInfo = {
            epoch: event.epoch as number,
            timestamp: event.timestamp as string,
            metrics: (event.metrics as Record<string, number>) ?? {},
          }
        }

        if (eventType === 'training_start') {
          managed.trainingMeta = {
            totalEpochs: event.total_epochs as number,
            startTimestamp: event.timestamp as string,
          }
          if (isFirst) {
            pushNotification(managed, createNotification(managed, 'training_start', event))
          }
        }

        if (eventType === 'predict_start') {
          managed.predictMeta = {
            totalSamples: event.n_samples as number,
            outputDir: event.output_dir as string,
            startTimestamp: event.timestamp as string,
          }
          if (isFirst) {
            pushNotification(managed, createNotification(managed, 'predict_start', event))
          }
        }

        if (eventType === 'predict_progress') {
          managed.lastPredictProgress = {
            current: event.current as number,
            total: event.total as number,
            filename: event.filename as string,
            timestamp: event.timestamp as string,
          }
        }

        if (eventType === 'predict_end') {
          if (isFirst) {
            pushNotification(managed, createNotification(managed, 'predict_end', event))
          }
        }

        if (eventType === 'training_end') {
          managed.lastTrainingEnd = event
          if (isFirst) {
            pushNotification(managed, createNotification(managed, 'training_end', event))
          }
        }
      }
    } catch {
      /* ignore parse errors */
    }
  }

  // 保留从下一个未完整事件标记开始的内容
  // 注意：EVENT_START 和 EVENT_END 是同一字符串（'__event__'），使用 indexOf 向前查找
  // 而非 lastIndexOf 向后查找，以防止不完整事件在已完整事件之后被错误清除
  const nextStart = managed[bufferKey].indexOf(EVENT_START, searchFrom)
  if (nextStart !== -1) {
    managed[bufferKey] = managed[bufferKey].substring(nextStart)
  } else {
    managed[bufferKey] = ''
  }
}
