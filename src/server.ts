/**
 * server.ts
 *
 * Description: HTTP 服务器（使用 Node.js HTTP 模块 + express）
 *              支持 SSE 流式对话和多轮会话管理
 * Author: leizheng, kongzhiquan
 * Time: 2026-02-02
 * Version: 2.3.0
 *
 * Changelog:
 *   - 2026-02-14 kongzhiquan: v2.3.0 传递 outputsPath 到 processMessage 以注入 Agent 指令
 *   - 2026-02-10 Leizheng: v2.2.0 速率限制 + 请求体大小限制 + 优雅关闭修复
 *   - 2026-02-07 kongzhiquan: v2.1.0 服务器关闭时清理训练进程
 *   - 2026-02-03 kongzhiquan: v2.0.0 适配持久化会话管理器
 *   - 2026-02-02 leizheng: v1.2.0 简化为直接使用 agentId 复用会话
 *   - 2026-02-02 leizheng: v1.1.0 添加多轮对话支持
 *   - 2026-02-01 kongzhiquan: v1.0.0 初始版本
 */

import express, { Request, Response, NextFunction } from 'express'
import { randomUUID } from 'crypto'
import { config, validateConfig } from './config'
import {
  createAgent,
  setupAgentHandlers,
  processMessage,
  type AgentConfig,
  type SSEEvent,
} from './agent-manager'
import { conversationManager } from './conversation-manager'
import { trainingProcessManager } from './utils/training-process-manager'
import fs from 'fs/promises'
import path from 'path'
// ========================================
// 初始化
// ========================================

validateConfig()

const app = express()
app.use(express.json({ limit: '2mb' }))

console.log(
  `[server] 启动中，端口=${config.port}, NODE_ENV=${process.env.NODE_ENV}`,
)

// ========================================
// 简单的内存速率限制器
// ========================================

const RATE_LIMIT_WINDOW_MS = 60 * 1000 // 1 分钟窗口
const RATE_LIMIT_MAX_REQUESTS = 30      // 每窗口最多 30 请求

const rateLimitMap = new Map<string, { count: number; resetAt: number }>()

function getRateLimitKey(req: Request): string {
  const forwarded = req.headers['x-forwarded-for']
  if (typeof forwarded === 'string') return forwarded.split(',')[0].trim()
  return req.ip || 'unknown'
}

// 定期清理过期条目（每 5 分钟），防止 Map 无限增长
setInterval(() => {
  const now = Date.now()
  for (const [key, entry] of rateLimitMap) {
    if (now > entry.resetAt) rateLimitMap.delete(key)
  }
}, 5 * 60 * 1000).unref()

function rateLimitMiddleware(req: Request, res: Response, next: NextFunction): void {
  const key = getRateLimitKey(req)
  const now = Date.now()
  let entry = rateLimitMap.get(key)

  if (!entry || now > entry.resetAt) {
    entry = { count: 0, resetAt: now + RATE_LIMIT_WINDOW_MS }
    rateLimitMap.set(key, entry)
  }

  entry.count++
  if (entry.count > RATE_LIMIT_MAX_REQUESTS) {
    const retryAfter = Math.ceil((entry.resetAt - now) / 1000)
    res.setHeader('Retry-After', String(retryAfter))
    sendError(res, 429, 'RATE_LIMITED', `Rate limit exceeded. Retry after ${retryAfter}s`)
    return
  }

  next()
}

// ========================================
// SSE 工具函数
// ========================================

function encodeSseEvent(event: SSEEvent): string {
  return `data: ${JSON.stringify(event)}\n\n`
}

function sendSSE(res: Response, event: SSEEvent): boolean {
  if (res.writableEnded) return false
  res.write(encodeSseEvent(event))
  return true
}

// ========================================
// 错误响应工具
// ========================================

function sendError(res: Response, status: number, error: string, message: string): void {
  if (!res.headersSent) {
    res.status(status).json({ error, message })
  }
}

// ========================================
// 中间件：请求日志
// ========================================

app.use((req: Request, res: Response, next: NextFunction) => {
  const reqId = randomUUID().slice(0, 8)
  const now = new Date().toISOString()
  const ip = req.headers['x-forwarded-for'] || req.headers['x-real-ip'] || req.ip

  console.log(`[server] [${now}] [req ${reqId}] ${req.method} ${req.path} from ${ip}`)

  // 将 reqId 存储在 res.locals 中供后续使用
  res.locals.reqId = reqId
  next()
})

// ========================================
// 中间件：API Key 认证
// ========================================

function requireAuth(req: Request, res: Response, next: NextFunction): void {
  const apiKey = req.headers['x-api-key'] as string

  if (!config.apiSecret || apiKey !== config.apiSecret) {
    const reqId = res.locals.reqId
    console.warn(`[server] [req ${reqId}] 未授权请求：无效的 X-API-Key`)
    sendError(res, 401, 'UNAUTHORIZED', 'Invalid or missing X-API-Key')
    return
  }

  next()
}

// ========================================
// 路由：健康检查
// @modified 2026-02-02 leizheng: 添加会话统计信息
// ========================================

app.get('/health', (req: Request, res: Response) => {
  const stats = conversationManager.getStats()
  res.json({
    status: 'ok',
    service: 'kode-agent-service',
    sdk: 'kode-sdk',
    timestamp: Date.now(),
    conversations: stats,
  })
})

// ========================================
// 路由：对话接口（SSE 流式）
// @modified 2026-02-03 kongzhiquan: 持久化读取
// @modified 2026-02-02 leizheng: 使用 agentId 复用会话
// ========================================

app.post('/api/chat/stream', rateLimitMiddleware, requireAuth, async (req: Request, res: Response) => {
  const reqId = res.locals.reqId
  const { message, mode = 'edit', context = {}, agentId: inputAgentId } = req.body
  // 参数验证
  const validateField = (value: any, fieldName: string) => {
    if (!value || typeof value !== 'string') {
      console.warn(`[server] [req ${reqId}] 缺少或无效的 "${fieldName}" 字段`)
      sendError(res, 400, 'BAD_REQUEST', `Field "${fieldName}" must be a non-empty string`)
      return false
    }
    return true
  }
  if (!validateField(message, 'message')) return
  if (!validateField(context?.userId, 'context.userId')) return
  if (!validateField(context?.workingDir, 'context.workingDir')) return
  if (!validateField(req.body.outputsPath, 'outputsPath')) return

  const userId = context.userId
  const workingDir = path.resolve(context.workingDir)
  const outputsPath = path.resolve(req.body.outputsPath)
  const files = Array.isArray(context.files) ? context.files : []

  console.log(
    `[server] [req ${reqId}] message="${message.slice(0, 80)}" mode=${mode} userId=${userId} agentId=${inputAgentId || 'new'}`,
  )

  // 设置 SSE 响应头
  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8')
  res.setHeader('Cache-Control', 'no-cache, no-transform')
  res.setHeader('Connection', 'keep-alive')
  res.flushHeaders()

  let agent = undefined
  let isNewSession = false

  // 如果输出目录和工作目录不存在，创建它
  try {
    await fs.mkdir(outputsPath, { recursive: true })
    await fs.mkdir(workingDir, { recursive: true })
  } catch (err) {
    console.error(`[server] [req ${reqId}] 创建目录失败:`, err)
    sendError(res, 500, 'INTERNAL_ERROR', 'Failed to create working or outputs directory')
    res.end()
    return
  }

  try {
    // 尝试加载已有会话
    if (inputAgentId && conversationManager.hasSession(inputAgentId)) {
      agent = await conversationManager.getAgent(inputAgentId)
      if (agent) {
        setupAgentHandlers(agent, reqId)
        console.log(`[server] [req ${reqId}] 加载会话: ${inputAgentId}`)
      }
    }
    
    // 如果没有可用会话，创建新的
    if (!agent) {
      const agentConfig: AgentConfig = { mode, workingDir, outputsPath, userId, files }
      agent = await createAgent(agentConfig)
      setupAgentHandlers(agent, reqId, [outputsPath, workingDir, '/data']) // 传递允许访问的路径列表

      // 注册到会话管理器
      conversationManager.registerSession(agent)
      isNewSession = true
      console.log(`[server] [req ${reqId}] 创建新会话: ${agent.agentId}`)
    }
  } catch (err: any) {
    console.error(`[server] [req ${reqId}] Agent 创建/加载失败:`, err)
    sendSSE(res, {
      type: 'error',
      error: 'INTERNAL_ERROR',
      message: 'Failed to create or load agent',
      timestamp: Date.now(),
    })
    res.end()
    return
  }

  // 心跳定时器
  let heartbeatCount = 0
  let clientDisconnected = false
  const heartbeatInterval = setInterval(() => {
    if (!res.writableEnded && !clientDisconnected) {
      heartbeatCount++
      sendSSE(res, {
        type: 'heartbeat',
        message: 'processing',
        count: heartbeatCount,
        timestamp: Date.now(),
      })
    }
  }, 2000)

  // 请求超时（2 小时，训练任务可能持续很长时间）
  const REQUEST_TIMEOUT = 2 * 60 * 60 * 1000
  const timeoutTimer = setTimeout(() => {
    if (!res.writableEnded && !clientDisconnected) {
      console.warn(`[server] [req ${reqId}] 请求超时`)
      sendSSE(res, {
        type: 'error',
        error: 'REQUEST_TIMEOUT',
        message: 'Request timeout after 2 hours',
        timestamp: Date.now(),
      })
      cleanup()
      res.end()
    }
  }, REQUEST_TIMEOUT)

  // 监听客户端断开连接
  const cleanup = () => {
    if (!clientDisconnected) {
      clientDisconnected = true
      clearInterval(heartbeatInterval)
      clearTimeout(timeoutTimer)
      console.log(`[server] [req ${reqId}] 客户端断开连接，清理资源`)
      // 中断 agent 处理
      if (agent && typeof agent.interrupt === 'function') {
        agent.interrupt({ note: 'Client disconnected' }).catch((err: any) => {
          console.error(`[server] [req ${reqId}] 中断 Agent 失败:`, err)
        })
      }
    }
  }

  req.on('aborted', cleanup)
  req.on('error', cleanup)
  res.on('close', cleanup)
  res.on('finish', cleanup)

  // 处理消息并流式返回
  try {
    for await (const event of processMessage(agent, message, reqId, { outputsPath })) {
      if (res.writableEnded || clientDisconnected) {
        console.log(`[server] [req ${reqId}] 检测到连接已断开，停止处理`)
        break
      }

      // 在 start 事件中标记是否新会话
      if (event.type === 'start') {
        sendSSE(res, {
          ...event,
          isNewSession,
        })
      } else {
        sendSSE(res, event)
      }
    }
  } catch (err: any) {
    console.error(`[server] [req ${reqId}] 处理消息失败:`, err)

    if (!res.writableEnded && !clientDisconnected) {
      sendSSE(res, {
        type: 'error',
        error: 'INTERNAL_ERROR',
        message: process.env.NODE_ENV === 'development'
          ? String(err?.message ?? err)
          : 'Internal server error',
        timestamp: Date.now(),
      })
    }
  } finally {
    cleanup()
    console.log(
      `[server] [req ${reqId}] 流已完成，agentId: ${agent.agentId}, 心跳: ${heartbeatCount}`,
    )
    if (!res.writableEnded) {
      res.end()
    }
  }
})

// ========================================
// 错误处理中间件
// ========================================

app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  const reqId = res.locals.reqId || 'unknown'
  console.error(`[server] [req ${reqId}] 未处理的错误:`, err)
  sendError(res, 500, 'INTERNAL_ERROR', 'Internal server error')
})

// ========================================
// 404 处理
// ========================================

app.use((req: Request, res: Response) => {
  sendError(res, 404, 'NOT_FOUND', 'Not found')
})

// ========================================
// 启动服务器
// ========================================

const server = app.listen(config.port, () => {
  console.log(`[server] 服务已启动在 http://localhost:${config.port}`)
})

// 优雅关闭（防止重复执行）
let isShuttingDown = false
async function gracefulShutdown(signal: string): Promise<void> {
  if (isShuttingDown) return
  isShuttingDown = true

  console.log(`[server] 收到 ${signal} 信号，开始关闭...`)

  // 1. 先停止接受新连接
  server.close(() => {
    console.log('[server] HTTP 服务器已停止接受新连接')
  })

  // 2. 清理会话管理器
  conversationManager.shutdown()

  // 3. 等待训练进程关闭
  try {
    await trainingProcessManager.shutdown()
  } catch (err) {
    console.error('[server] 训练进程关闭失败:', err)
  }

  console.log('[server] 服务器已关闭')
  process.exit(0)
}

// 硬超时保护：如果优雅关闭超过 15 秒，强制退出
function scheduleForceExit(): void {
  setTimeout(() => {
    console.error('[server] 优雅关闭超时，强制退出')
    process.exit(1)
  }, 15000).unref()
}

process.on('SIGTERM', () => {
  scheduleForceExit()
  gracefulShutdown('SIGTERM')
})

process.on('SIGINT', () => {
  scheduleForceExit()
  gracefulShutdown('SIGINT')
})

// 全局错误处理
process.on('uncaughtException', (err) => {
  console.error('[server] 未捕获的异常:', err)
  process.exit(1)
})

process.on('unhandledRejection', (reason, promise) => {
  console.error('[server] 未处理的 Promise 拒绝:', reason)
})
