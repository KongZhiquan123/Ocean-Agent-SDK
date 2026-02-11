/**
 * conversation-manager.ts
 *
 * Description: 会话管理器 - 实现多轮对话支持
 *              从 .kode 文件夹持久化读取会话记录，按需加载 Agent 实例
 *              不再维护内存 Map，节省内存资源
 * Author: leizheng, kongzhiquan
 * Time: 2026-02-02
 * Version: 2.1.0
 *
 * Changelog:
 *   - 2026-02-10 Leizheng: v2.1.0 agentId 格式验证 + 路径遍历防护
 *   - 2026-02-03 kongzhiquan: v2.0.0 重构为从磁盘持久化读取，移除内存缓存和过期清理
 *   - 2026-02-02 leizheng: v1.1.0 简化为直接使用 agentId
 *   - 2026-02-02 leizheng: v1.0.0 初始版本
 */

import { Agent } from '@shareai-lab/kode-sdk'
import { getDependencies } from './config'
import * as fs from 'fs'
import * as path from 'path'

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
// @date 2026-02-02
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
  hasSession(agentId: string): boolean {
    if (!this.isValidAgentId(agentId)) return false
    const agentDir = path.join(this.storePath, agentId)
    const metaPath = path.join(agentDir, 'meta.json')

    try {
      return fs.existsSync(metaPath)
    } catch (error) {
      console.error(`[ConversationManager] 检查会话失败 ${agentId}:`, error)
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
    if (!this.hasSession(agentId)) {
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
      ;(agent as any).PROCESSING_TIMEOUT = 2 * 60 * 60 * 1000 // 2 小时

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
  registerSession(agent: Agent): void {
    const agentId = agent.agentId
    console.log(`[ConversationManager] 注册新会话 ${agentId}`)
  }

  /**
   * 获取所有会话的 agentId 列表
   */
  listSessions(): string[] {
    try {
      if (!fs.existsSync(this.storePath)) {
        return []
      }

      const entries = fs.readdirSync(this.storePath, { withFileTypes: true })
      return entries
        .filter(entry => entry.isDirectory() && AGENT_ID_PATTERN.test(entry.name))
        .map(entry => entry.name)
    } catch (error) {
      console.error('[ConversationManager] 列出会话失败:', error)
      return []
    }
  }

  /**
   * 获取会话统计信息
   */
  getStats() {
    const sessions = this.listSessions()
    return {
      totalSessions: sessions.length,
      storePath: this.storePath,
    }
  }

  /**
   * 删除指定会话（从磁盘删除）
   */
  removeSession(agentId: string): boolean {
    if (!this.isValidAgentId(agentId)) {
      console.warn(`[ConversationManager] 拒绝删除非法 agentId: ${agentId}`)
      return false
    }
    const agentDir = path.join(this.storePath, agentId)

    try {
      if (fs.existsSync(agentDir)) {
        fs.rmSync(agentDir, { recursive: true, force: true })
        console.log(`[ConversationManager] 删除会话 ${agentId}`)
        return true
      }
      return false
    } catch (error) {
      console.error(`[ConversationManager] 删除会话失败 ${agentId}:`, error)
      return false
    }
  }

  shutdown(): void {
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
