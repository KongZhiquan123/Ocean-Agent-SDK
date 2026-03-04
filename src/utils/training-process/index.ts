/**
 * @file index.ts
 *
 * @description 训练进程管理器 —— 桶文件，re-export 公共类型并构建单例
 * @author kongzhiquan
 * @date 2026-03-04
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-03-04 kongzhiquan: v1.0.0 从 training-process-manager.ts 拆分
 */

export type {
  ErrorSummary,
  TrainingProgress,
  RuntimeStats,
  TrainingNotification,
  TrainingProcessInfo,
  ManagedProcess,
} from './types'

export { TrainingProcessManager } from './manager'

import { TrainingProcessManager } from './manager'

export const trainingProcessManager = new TrainingProcessManager()
