// ---- 训练进程管理器常量 ----
export const MAX_LOG_BYTES = Number(process.env.TRAIN_LOG_MAX_BYTES ?? 50 * 1024 * 1024)
export const MAX_LOG_ROTATIONS = Number(process.env.TRAIN_LOG_MAX_ROTATIONS ?? 5)
export const MAX_NOTIFICATION_QUEUE = Number(process.env.TRAIN_NOTIFICATION_QUEUE_SIZE ?? 50)
export const RUNTIME_SAMPLE_INTERVAL_MS = Number(process.env.TRAIN_RUNTIME_SAMPLE_INTERVAL_MS ?? 5000)
export const PROCESS_STATS_TIMEOUT_MS = Number(process.env.TRAIN_PROCESS_STATS_TIMEOUT_MS ?? 2000)
/** tail 模式最多从文件尾部读取的字节数（避免 OOM） */
export const TAIL_READ_BYTES = 52488 as const // 约 50KB，通常足够读取最近的日志内容
export const RING_BUFFER_SIZE = 100 as const
/** 内存中保留的最大已完成进程数 */
export const MAX_COMPLETED_PROCESSES = 50 as const