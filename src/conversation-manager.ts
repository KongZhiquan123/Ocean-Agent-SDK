/**
 * @file conversation-manager.ts
 *
 * @description 会话管理器 - 实现多轮对话支持
 *              从 .kode 文件夹持久化读取会话记录，按需加载 Agent 实例
 * @author Leizheng, kongzhiquan
 * @date 2026-03-05
 * @version 3.1.0
 *
 * @changelog
 *   - 2026-03-05 kongzhiquan: v3.0.0 全面异步化文件操作
 *   - 2026-02-10 Leizheng: v2.1.0 agentId 格式验证 + 路径遍历防护
 *   - 2026-02-03 kongzhiquan: v2.0.0 重构为从磁盘持久化读取，移除内存缓存和过期清理
 *   - 2026-02-02 leizheng: v1.1.0 简化为直接使用 agentId
 *   - 2026-02-02 leizheng: v1.0.0 初始版本
 */

import { Agent } from '@shareai-lab/kode-sdk'
import { getDependencies } from './config'
import { access, readdir, rm } from 'fs/promises'
import path from 'path'
import { REQUEST_TIMEOUT_MS } from './utils/constants'

// ========================================
// 类型定义
// ========================================

interface ConversationManagerConfig {
  storePath?: string  // .kode 文件夹路径
}

// agentId 格式：agt- 前缀 + 字母数字/下划线/连字符（防路径遍历）
const AGENT_ID_PATTERN = /^agt-[a-zA-Z0-9_-]+$/

// ========================================
// 会话管理器类
// @author leizheng, kongzhiquan
// @date 2026-03-05
// ========================================

class ConversationManager {
  private storePath: string

  constructor(config: ConversationManagerConfig = {}) {
    this.storePath = path.resolve(config.storePath || './.kode')
    console.log('[ConversationManager] 初始化完成', {
      storePath: this.storePath,
    })
  }

  /**
   * 验证 agentId 格式，防止路径遍历攻击
   */
  private isValidAgentId(agentId: string): boolean {
    if (!agentId || typeof agentId !== 'string') return false
    if (!AGENT_ID_PATTERN.test(agentId)) return false
    // 防御性检查：resolve 后路径必须在 storePath 内
    const resolved = path.resolve(this.storePath, agentId)
    return resolved.startsWith(this.storePath + path.sep)
  }

  /**
   * 检查 agentId 对应的会话是否存在于磁盘
   */
  async hasSession(agentId: string): Promise<boolean> {
    if (!this.isValidAgentId(agentId)) return false
    const metaPath = path.join(this.storePath, agentId, 'meta.json')
    try {
      await access(metaPath)
      return true
    } catch {
      return false
    }
  }

  /**
   * 从磁盘加载 agentId 对应的 Agent 实例
   */
  async getAgent(agentId: string): Promise<Agent | null> {
    if (!this.isValidAgentId(agentId)) {
      console.warn(`[ConversationManager] 非法 agentId 格式: ${agentId}`)
      return null
    }
    if (!(await this.hasSession(agentId))) {
      console.log(`[ConversationManager] 会话不存在: ${agentId}`)
      return null
    }

    try {
      const deps = getDependencies()

      // 使用 KODE SDK 的 resumeFromStore 方法从磁盘恢复 Agent
      const agent = await Agent.resumeFromStore(agentId, deps, {
        autoRun: false,  // 不自动运行，等待新消息
      })

      // 增大 KODE SDK 内部处理超时（默认 5 分钟，预处理/训练流水线可能需要数小时）
      ;(agent as any).PROCESSING_TIMEOUT = REQUEST_TIMEOUT_MS

      console.log(`[ConversationManager] 从磁盘加载会话 ${agentId}`)
      return agent
    } catch (error) {
      console.error(`[ConversationManager] 加载会话失败 ${agentId}:`, error)
      return null
    }
  }

  /**
   * 注册新的 Agent 会话（实际上不需要做任何事，因为 KODE SDK 已经持久化了），如果后续涉及到数据库交互，可以在这里处理
   */
  async registerSession(agent: Agent): Promise<void> {
    console.log(`[ConversationManager] 注册新会话 ${agent.agentId}`)
  }

  /**
   * 获取所有会话的 agentId 列表
   */
  async listSessions(): Promise<string[]> {
    try {
      const entries = await readdir(this.storePath, { withFileTypes: true })
      return entries
        .filter(e => e.isDirectory() && AGENT_ID_PATTERN.test(e.name))
        .map(e => e.name)
    } catch (error: any) {
      if (error.code !== 'ENOENT') {
        console.error('[ConversationManager] 列出会话失败:', error)
      }
      return []
    }
  }

  /**
   * 获取会话统计信息
   */
  async getStats(): Promise<{ totalSessions: number; storePath: string }> {
    const sessions = await this.listSessions()
    return {
      totalSessions: sessions.length,
      storePath: this.storePath,
    }
  }

  /**
   * 删除指定会话（从磁盘异步删除）
   */
  async removeSession(agentId: string): Promise<boolean> {
    if (!this.isValidAgentId(agentId)) {
      console.warn(`[ConversationManager] 拒绝删除非法 agentId: ${agentId}`)
      return false
    }
    const agentDir = path.join(this.storePath, agentId)
    try {
      await rm(agentDir, { recursive: true, force: true })
      console.log(`[ConversationManager] 删除会话 ${agentId}`)
      return true
    } catch (error) {
      console.error(`[ConversationManager] 删除会话失败 ${agentId}:`, error)
      return false
    }
  }

  async shutdown(): Promise<void> {
    /**
     * 因后续将是与数据库交互，这里预留关闭连接的接口，直接与磁盘交互无需处理
     */
    console.log('[ConversationManager] 已关闭')
  }
}

// ========================================
// 单例导出
// ========================================

export const conversationManager = new ConversationManager({
  storePath: process.env.KODE_STORE_PATH || './.kode',
})

export { ConversationManager, ConversationManagerConfig }
