/**
 * conversation-manager.ts
 *
 * Description: 会话管理器 - 实现多轮对话支持
 *              维护 Agent 实例池，支持会话复用和自动过期清理
 *              直接使用 agentId 作为会话标识
 * Author: leizheng
 * Time: 2026-02-02
 * Version: 1.1.0
 *
 * Changelog:
 *   - 2026-02-02 leizheng: v1.1.0 简化为直接使用 agentId
 *   - 2026-02-02 leizheng: v1.0.0 初始版本
 */

import { Agent } from '@shareai-lab/kode-sdk'

// ========================================
// 类型定义
// ========================================

interface SessionEntry {
  agent: Agent
  agentId: string
  userId: string
  workingDir: string
  createdAt: number
  lastActiveAt: number
  messageCount: number
}

interface ConversationManagerConfig {
  sessionTimeoutMs?: number
  cleanupIntervalMs?: number
  maxSessions?: number
}

// ========================================
// 会话管理器类
// @author leizheng
// @date 2026-02-02
// ========================================

class ConversationManager {
  private sessions: Map<string, SessionEntry> = new Map()
  private sessionTimeoutMs: number
  private maxSessions: number
  private cleanupTimer: NodeJS.Timeout | null = null

  constructor(config: ConversationManagerConfig = {}) {
    this.sessionTimeoutMs = config.sessionTimeoutMs || 30 * 60 * 1000
    this.maxSessions = config.maxSessions || 100

    const cleanupIntervalMs = config.cleanupIntervalMs || 5 * 60 * 1000
    this.startCleanupTimer(cleanupIntervalMs)

    console.log('[ConversationManager] 初始化完成', {
      sessionTimeoutMs: this.sessionTimeoutMs,
      maxSessions: this.maxSessions,
    })
  }

  /**
   * 检查 agentId 对应的会话是否存在且未过期
   */
  hasSession(agentId: string): boolean {
    const entry = this.sessions.get(agentId)
    if (!entry) return false

    const now = Date.now()
    if (now - entry.lastActiveAt > this.sessionTimeoutMs) {
      this.removeSession(agentId)
      return false
    }
    return true
  }

  /**
   * 获取 agentId 对应的 Agent 实例
   */
  getAgent(agentId: string): Agent | null {
    const entry = this.sessions.get(agentId)
    if (!entry) return null

    const now = Date.now()
    if (now - entry.lastActiveAt > this.sessionTimeoutMs) {
      this.removeSession(agentId)
      return null
    }

    entry.lastActiveAt = now
    entry.messageCount++
    console.log(`[ConversationManager] 复用会话 ${agentId}, 消息计数: ${entry.messageCount}`)
    return entry.agent
  }

  /**
   * 注册新的 Agent 会话（使用 agent.agentId 作为 key）
   */
  registerSession(
    agent: Agent,
    userId: string,
    workingDir: string
  ): void {
    if (this.sessions.size >= this.maxSessions) {
      this.evictOldestSession()
    }

    const agentId = agent.agentId
    const now = Date.now()
    this.sessions.set(agentId, {
      agent,
      agentId,
      userId,
      workingDir,
      createdAt: now,
      lastActiveAt: now,
      messageCount: 1,
    })

    console.log(`[ConversationManager] 注册新会话 ${agentId}, 当前会话数: ${this.sessions.size}`)
  }

  /**
   * 移除指定会话
   */
  removeSession(agentId: string): boolean {
    const deleted = this.sessions.delete(agentId)
    if (deleted) {
      console.log(`[ConversationManager] 移除会话 ${agentId}, 剩余: ${this.sessions.size}`)
    }
    return deleted
  }

  /**
   * 获取会话统计信息
   */
  getStats() {
    return {
      totalSessions: this.sessions.size,
      maxSessions: this.maxSessions,
    }
  }

  private startCleanupTimer(intervalMs: number): void {
    this.cleanupTimer = setInterval(() => {
      this.cleanupExpiredSessions()
    }, intervalMs)
  }

  private cleanupExpiredSessions(): void {
    const now = Date.now()
    const expiredIds: string[] = []

    for (const [id, entry] of this.sessions.entries()) {
      if (now - entry.lastActiveAt > this.sessionTimeoutMs) {
        expiredIds.push(id)
      }
    }

    for (const id of expiredIds) {
      this.removeSession(id)
    }

    if (expiredIds.length > 0) {
      console.log(`[ConversationManager] 清理了 ${expiredIds.length} 个过期会话`)
    }
  }

  private evictOldestSession(): void {
    let oldestId: string | null = null
    let oldestTime = Infinity

    for (const [id, entry] of this.sessions.entries()) {
      if (entry.lastActiveAt < oldestTime) {
        oldestTime = entry.lastActiveAt
        oldestId = id
      }
    }

    if (oldestId) {
      console.log(`[ConversationManager] 驱逐最旧会话 ${oldestId}`)
      this.removeSession(oldestId)
    }
  }

  shutdown(): void {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer)
      this.cleanupTimer = null
    }
    this.sessions.clear()
    console.log('[ConversationManager] 已关闭')
  }
}

// ========================================
// 单例导出
// ========================================

export const conversationManager = new ConversationManager({
  sessionTimeoutMs: 30 * 60 * 1000,
  cleanupIntervalMs: 5 * 60 * 1000,
  maxSessions: 100,
})

export { ConversationManager, ConversationManagerConfig, SessionEntry }
