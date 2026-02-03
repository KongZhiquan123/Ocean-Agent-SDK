/**
 * test-client.ts
 *
 * Description: 测试 kode-agent-service 的交互式客户端
 *              支持多轮对话、海洋数据预处理交互测试
 * Author: leizheng
 * Time: 2026-02-02
 * Version: 1.2.0
 *
 * Changelog:
 *   - 2026-02-03 leizheng: v1.2.0 海洋预处理测试增加输出目录输入
 *   - 2026-02-02 leizheng: v1.1.0 使用 agentId 支持多轮对话
 */

import 'dotenv/config'
import * as readline from 'readline'

const API_URL = process.env.KODE_API_URL || 'http://localhost:8787'
const API_KEY = process.env.KODE_API_SECRET || 'secret-key'

interface SSEEvent {
  type: string
  [key: string]: any
}

// 创建 readline 接口用于用户输入
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
})

function prompt(question: string): Promise<string> {
  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      resolve(answer)
    })
  })
}

// 会话 ID，用于多轮对话（直接使用 agentId）
let agentId: string | null = null

// 是否显示完整工具结果（可通过参数控制）
let showFullToolResult = true

async function chat(message: string, mode: 'ask' | 'edit' = 'edit'): Promise<string> {
  console.log('\n' + '='.repeat(60))
  console.log(`用户: ${message}`)
  console.log('='.repeat(60) + '\n')

  try {
    const body: any = {
      message,
      mode,
      context: {
        userId: 'test-user',
        workingDir: './work_ocean',
      },
    }

    // 如果有 agentId，传递给服务端保持上下文
    if (agentId) {
      body.agentId = agentId
    }

    const response = await fetch(`${API_URL}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY,
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error(`请求失败: ${response.status} ${response.statusText}`)
      console.error('错误详情:', errorText)
      return ''
    }

    if (!response.body) {
      console.error('响应体为空')
      return ''
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()

    let buffer = ''
    let fullText = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const event: SSEEvent = JSON.parse(line.slice(6))

            switch (event.type) {
              case 'start':
                console.log(`[开始] Agent ID: ${event.agentId}`)
                // 保存 agentId 用于多轮对话
                if (event.agentId) {
                  agentId = event.agentId
                }
                break

              case 'text':
                process.stdout.write(event.content)
                fullText += event.content
                break

              case 'tool_use':
                console.log(`\n[工具调用] ${event.tool}`)
                console.log('输入参数:', JSON.stringify(event.input, null, 2))
                break

              case 'tool_result':
                console.log(`[工具结果] ${event.is_error ? '失败' : '成功'}`)
                // 显示完整结果（格式化 JSON）
                if (showFullToolResult) {
                  const resultStr = JSON.stringify(event.result, null, 2)
                  if (resultStr.length > 2000) {
                    console.log('结果（前 2000 字符）:')
                    console.log(resultStr.slice(0, 2000) + '\n... [截断]')
                  } else {
                    console.log('结果:', resultStr)
                  }
                } else {
                  // 简短显示
                  console.log('结果:', JSON.stringify(event.result).slice(0, 200))
                }
                break

              case 'done':
                console.log('\n\n[完成] 处理结束')
                break

              case 'heartbeat':
                // 静默处理心跳
                break

              case 'error':
                console.error(`\n[错误] ${event.error}: ${event.message}`)
                break

              default:
                console.log(`[${event.type}]`, event)
            }
          } catch (err) {
            console.error('解析事件失败:', line, err)
          }
        }
      }
    }

    return fullText
  } catch (err: any) {
    console.error('请求错误:', err.message)
    return ''
  }
}

async function testHealth(): Promise<boolean> {
  console.log('测试健康检查接口...')
  try {
    const response = await fetch(`${API_URL}/health`)
    const data = await response.json()
    console.log('健康检查结果:', data)
    return true
  } catch (err: any) {
    console.error('健康检查失败:', err.message)
    return false
  }
}

// 交互式对话模式
async function interactiveMode() {
  console.log('\n' + '='.repeat(60))
  console.log('交互式对话模式 (输入 "exit" 退出, "reset" 重置会话)')
  console.log('='.repeat(60))

  while (true) {
    const userInput = await prompt('\n你: ')

    if (userInput.toLowerCase() === 'exit') {
      console.log('再见！')
      break
    }

    if (userInput.toLowerCase() === 'reset') {
      agentId = null
      console.log('会话已重置')
      continue
    }

    if (!userInput.trim()) {
      continue
    }

    await chat(userInput, 'edit')
  }

  rl.close()
}

// 测试海洋数据预处理的交互流程
async function testOceanPreprocess() {
  console.log('\n' + '='.repeat(60))
  console.log('海洋数据预处理交互式测试')
  console.log('='.repeat(60))

  // 重置会话
  agentId = null
  showFullToolResult = true

  // ========== 收集必要的路径信息 ==========
  console.log('\n请提供数据路径信息：')

  const dynFolder = await prompt('动态数据文件夹路径: ')
  if (!dynFolder.trim()) {
    console.log('错误: 必须提供动态数据文件夹路径')
    rl.close()
    return
  }

  const staticFile = await prompt('静态文件路径 (无则回车): ')
  const fileFilter = await prompt('文件名过滤关键字 (无则回车): ')
  const researchVars = await prompt('研究变量 (逗号分隔，如 uo,vo): ')

  if (!researchVars.trim()) {
    console.log('错误: 必须指定研究变量')
    rl.close()
    return
  }

  const outputDir = await prompt('输出目录路径: ')
  if (!outputDir.trim()) {
    console.log('错误: 必须指定输出目录')
    rl.close()
    return
  }

  // ========== 发送请求，Agent 会分析并展示疑似变量让用户确认 ==========
  console.log('\n' + '='.repeat(60))
  console.log('开始处理，Agent 会分析数据并展示变量信息供你确认...')
  console.log('='.repeat(60))

  let msg = `请帮我预处理海洋数据：
- 动态数据目录: ${dynFolder}
- 研究变量: ${researchVars}
- 输出目录: ${outputDir}`

  if (staticFile.trim()) {
    msg += `\n- 静态文件: ${staticFile}`
  }
  if (fileFilter.trim()) {
    msg += `\n- 文件过滤器: ${fileFilter}`
  }

  await chat(msg, 'edit')

  // ========== 交互式确认循环 ==========
  // Agent 会返回 awaiting_confirmation 并展示疑似变量
  // 用户在这里进行确认
  console.log('\n' + '-'.repeat(60))
  console.log('请根据 Agent 展示的变量信息进行确认')
  console.log('输入 done 结束测试')
  console.log('-'.repeat(60))

  while (true) {
    const userInput = await prompt('\n你: ')
    if (userInput.toLowerCase() === 'done') {
      break
    }
    if (userInput.trim()) {
      await chat(userInput, 'edit')
    }
  }

  console.log('\n' + '='.repeat(60))
  console.log('海洋数据预处理测试完成！')
  console.log('='.repeat(60))
  rl.close()
}

// 原来的自动化测试（保留）
async function runAutomatedTests() {
  console.log('\n' + '='.repeat(60))
  console.log('运行自动化测试套件')
  console.log('='.repeat(60))

  // 使用简短输出模式
  showFullToolResult = false

  // 测试 0: agent 有什么 skills
  console.log('\n--- 测试 0: 查看 skills ---')
  await chat('你有什么skills，加载它', 'edit')
  await new Promise((resolve) => setTimeout(resolve, 2000))

  // 测试 1: 问答模式
  console.log('\n--- 测试 1: 问答模式 ---')
  await chat('请介绍一下 KODE SDK 是什么？', 'ask')
  await new Promise((resolve) => setTimeout(resolve, 2000))

  // 测试 2: 编程模式 - 创建文件
  console.log('\n--- 测试 2: 创建文件 ---')
  await chat('请创建一个 hello.py 文件，打印 "Hello, KODE SDK!"', 'edit')
  await new Promise((resolve) => setTimeout(resolve, 2000))

  // 测试 3: 编程模式 - 执行命令
  console.log('\n--- 测试 3: 列出文件 ---')
  await chat('请列出当前目录下的所有文件', 'edit')
  await new Promise((resolve) => setTimeout(resolve, 2000))

  console.log('\n' + '='.repeat(60))
  console.log('自动化测试完成！')
  console.log('提示: 海洋数据预处理测试请使用 -o 参数进行交互式测试')
  console.log('='.repeat(60))
  rl.close()
}

// 主程序
async function main() {
  console.log('KODE Agent Service 交互式测试客户端')
  console.log(`API URL: ${API_URL}`)
  console.log(`API Key: ${API_KEY.slice(0, 10)}...`)

  // 测试健康检查
  const healthy = await testHealth()
  if (!healthy) {
    console.error('\n服务不可用，请确保服务已启动')
    rl.close()
    process.exit(1)
  }

  console.log('\n服务正常！')
  console.log('\n选择测试模式:')
  console.log('  1. 交互式对话')
  console.log('  2. 海洋数据预处理测试（交互式）')
  console.log('  3. 单条消息测试')
  console.log('  4. 自动化测试套件（原测试）')

  const choice = await prompt('\n请选择 (1/2/3/4): ')

  switch (choice) {
    case '1':
      await interactiveMode()
      break
    case '2':
      await testOceanPreprocess()
      break
    case '3':
      const msg = await prompt('请输入消息: ')
      await chat(msg, 'edit')
      rl.close()
      break
    case '4':
      await runAutomatedTests()
      break
    default:
      console.log('无效选择，进入交互模式')
      await interactiveMode()
  }
}

// 命令行参数处理
const args = process.argv.slice(2)
if (args.includes('--interactive') || args.includes('-i')) {
  testHealth().then((healthy) => {
    if (healthy) {
      interactiveMode()
    } else {
      console.error('服务不可用')
      process.exit(1)
    }
  })
} else if (args.includes('--ocean') || args.includes('-o')) {
  testHealth().then((healthy) => {
    if (healthy) {
      testOceanPreprocess()
    } else {
      console.error('服务不可用')
      process.exit(1)
    }
  })
} else if (args.includes('--auto') || args.includes('-a')) {
  testHealth().then((healthy) => {
    if (healthy) {
      runAutomatedTests()
    } else {
      console.error('服务不可用')
      process.exit(1)
    }
  })
} else if (args.length > 0 && !args[0].startsWith('-')) {
  const message = args.join(' ')
  testHealth().then((healthy) => {
    if (healthy) {
      chat(message, 'edit').then(() => process.exit(0))
    } else {
      process.exit(1)
    }
  })
} else {
  main()
}
