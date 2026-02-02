// test-client.ts
// 测试 kode-agent-service 的简单客户端

import 'dotenv/config'

const API_URL = process.env.KODE_API_URL || 'http://localhost:8787'
const API_KEY = process.env.KODE_API_SECRET || 'secret-key'

interface SSEEvent {
  type: string
  [key: string]: any
}

async function testChat(message: string, mode: 'ask' | 'edit' = 'edit') {
  console.log('\n=================================')
  console.log(`测试消息: ${message}`)
  console.log(`模式: ${mode}`)
  console.log('=================================\n')

  try {
    const response = await fetch(`${API_URL}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY,
      },
      body: JSON.stringify({
        message,
        mode,
        context: {
          userId: 'test-user',
          workingDir: './work_ocean',
        },
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error(`请求失败: ${response.status} ${response.statusText}`)
      console.error('错误详情:', errorText)
      return
    }

    if (!response.body) {
      console.error('响应体为空')
      return
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
                break

              case 'text':
                console.log(event.content)
                fullText += event.content
                break

              case 'tool_use':
                console.log(
                  `\n[工具调用] ${event.tool}`,
                  JSON.stringify(event.input, null, 2),
                )
                break

              case 'tool_result':
                console.log(
                  `[工具结果] ${event.is_error ? '失败' : '成功'}:`,
                  JSON.stringify(event.result).slice(0, 200),
                )
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

    console.log('\n\n完整输出:')
    console.log('-'.repeat(50))
    console.log(fullText)
    console.log('-'.repeat(50))
  } catch (err: any) {
    console.error('请求错误:', err.message)
  }
}

async function testHealth() {
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

// 主程序
async function main() {
  console.log('KODE Agent Service 测试客户端')
  console.log(`API URL: ${API_URL}`)
  console.log(`API Key: ${API_KEY.slice(0, 10)}...`)

  // 测试健康检查
  const healthy = await testHealth()
  if (!healthy) {
    console.error('\n服务不可用，请确保服务已启动')
    process.exit(1)
  }

  console.log('\n服务正常，开始测试对话...\n')
  // //测试 0： agent有什么skills
  // await testChat('你有什么skills，加载它', 'edit')

  // // 测试 1: 问答模式
  // await testChat('请介绍一下 KODE SDK 是什么？', 'ask')

  // //等待一下
  // await new Promise((resolve) => setTimeout(resolve, 2000))

  // //测试 2: 编程模式 - 创建文件
  // await testChat('请创建一个 hello.py 文件，打印 "Hello, KODE SDK!"', 'edit')

  // // 等待一下
  // await new Promise((resolve) => setTimeout(resolve, 2000))

  // // 测试 3: 编程模式 - 执行命令
  // await testChat('请列出当前目录下的所有文件', 'edit')

  await testChat('请帮我对海洋超分数据的nc文件做预处理，研究变量为流场的u分量，静态文件在xxx_static.nc中', 'edit')
  console.log('\n所有测试完成！')
}

// 命令行参数
const args = process.argv.slice(2)
if (args.length > 0) {
  const message = args.join(' ')
  const mode = process.env.MODE === 'ask' ? 'ask' : 'edit'
  testChat(message, mode).then(() => process.exit(0))
} else {
  main().then(() => process.exit(0))
}
