/**
 * server.ts
 *
 * Description: HTTP 服务器（使用 Node.js HTTP 模块 + express）
 *              支持 SSE 流式对话和多轮会话管理
 * Author: leizheng
 * Time: 2026-02-02
 * Version: 1.2.0
 *
 * Changelog:
 *   - 2026-02-02 leizheng: v1.2.0 简化为直接使用 agentId 复用会话
 *   - 2026-02-02 leizheng: v1.1.0 添加多轮对话支持
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

// ========================================
// 初始化
// ========================================

validateConfig()

const app = express()
app.use(express.json())

console.log(
  `[server] 启动中，端口=${config.port}, NODE_ENV=${process.env.NODE_ENV}`,
)

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
// @modified 2026-02-02 leizheng: 使用 agentId 复用会话
// ========================================

app.post('/api/chat/stream', requireAuth, async (req: Request, res: Response) => {
  const reqId = res.locals.reqId
  const { message, mode = 'edit', outputsPath, context = {}, agentId: inputAgentId } = req.body

  // 参数验证
  if (!message || typeof message !== 'string') {
    console.warn(`[server] [req ${reqId}] 缺少或无效的 "message" 字段`)
    sendError(res, 400, 'BAD_REQUEST', 'Field "message" must be a non-empty string')
    return
  }
  if (!context.userId) {
    console.warn(`[server] [req ${reqId}] 缺少 "context.userId" 字段`)
    sendError(res, 400, 'BAD_REQUEST', 'Field "context.userId" is required')
    return
  }
  if (!context.workingDir) {
    console.warn(`[server] [req ${reqId}] 缺少 "context.workingDir" 字段`)
    sendError(res, 400, 'BAD_REQUEST', 'Field "context.workingDir" is required')
    return
  }
  const userId = context.userId
  const workingDir = context.workingDir
  const files = Array.isArray(context.files) ? context.files : []

  console.log(
    `[server] [req ${reqId}] message="${message.slice(0, 80)}" mode=${mode} userId=${userId} agentId=${inputAgentId || 'new'}`,
  )

  // 设置 SSE 响应头
  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8')
  res.setHeader('Cache-Control', 'no-cache, no-transform')
  res.setHeader('Connection', 'keep-alive')
  res.flushHeaders()

  // ========================================
  // 多轮对话支持 - 直接使用 agentId
  // @author leizheng
  // @date 2026-02-02
  // ========================================
  let agent
  let isNewSession = false

  try {
    // 尝试复用已有会话
    if (inputAgentId && conversationManager.hasSession(inputAgentId)) {
      agent = conversationManager.getAgent(inputAgentId)
      console.log(`[server] [req ${reqId}] 复用已有会话: ${inputAgentId}`)
    }

    // 如果没有可用会话，创建新的
    if (!agent) {
      const agentConfig: AgentConfig = { mode, workingDir, outputsPath, userId, files }
      agent = await createAgent(agentConfig)
      setupAgentHandlers(agent, reqId)

      // 注册到会话管理器
      conversationManager.registerSession(agent, userId, workingDir)
      isNewSession = true
      console.log(`[server] [req ${reqId}] 创建新会话: ${agent.agentId}`)
    }
  } catch (err: any) {
    console.error(`[server] [req ${reqId}] Agent 创建失败:`, err)
    sendSSE(res, {
      type: 'error',
      error: 'INTERNAL_ERROR',
      message: 'Failed to create agent',
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

  // 请求超时（10 分钟）
  const REQUEST_TIMEOUT = 10 * 60 * 1000
  const timeoutTimer = setTimeout(() => {
    if (!res.writableEnded && !clientDisconnected) {
      console.warn(`[server] [req ${reqId}] 请求超时`)
      sendSSE(res, {
        type: 'error',
        error: 'REQUEST_TIMEOUT',
        message: 'Request timeout after 10 minutes',
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

      // 注意：不再中断 agent，因为会话可能还要继续使用
      // 会话管理器会在过期后自动清理
    }
  }


  req.on('aborted', cleanup)
  req.on('error', cleanup)
  res.on('close', cleanup)
  res.on('finish', cleanup)

  // 处理消息并流式返回
  try {
    for await (const event of processMessage(agent, message, reqId)) {
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

  if (!res.headersSent) {
    sendError(res, 500, 'INTERNAL_ERROR', 'Internal server error')
  }
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

// 关闭
process.on('SIGTERM', () => {
  console.log('[server] 收到 SIGTERM 信号，开始关闭...')
  conversationManager.shutdown()
  server.close(() => {
    console.log('[server] 服务器已关闭')
    process.exit(0)
  })
})

process.on('SIGINT', () => {
  console.log('[server] 收到 SIGINT 信号，开始关闭...')
  conversationManager.shutdown()
  server.close(() => {
    console.log('[server] 服务器已关闭')
    process.exit(0)
  })
})

// 全局错误处理
process.on('uncaughtException', (err) => {
  console.error('[server] 未捕获的异常:', err)
  process.exit(1)
})

process.on('unhandledRejection', (reason, promise) => {
  console.error('[server] 未处理的 Promise 拒绝:', reason)
})
