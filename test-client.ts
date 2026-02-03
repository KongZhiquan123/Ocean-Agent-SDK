/**
 * test-client.ts
 *
 * Description: 测试 kode-agent-service 的交互式客户端
 *              支持多轮对话、海洋数据预处理完整流程测试
 * Author: leizheng
 * Time: 2026-02-02
 * Version: 2.0.0
 *
 * Changelog:
 *   - 2026-02-03 leizheng: v2.0.0 完善海洋预处理完整流程测试
 *     - 新增裁剪参数输入
 *     - 新增下采样测试
 *     - 新增可视化测试
 *     - 新增指标检测测试
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

// 测试海洋数据预处理的完整流程
async function testOceanPreprocess() {
  console.log('\n' + '='.repeat(60))
  console.log('海洋数据预处理完整流程测试 v2.0')
  console.log('='.repeat(60))

  // 重置会话
  agentId = null
  showFullToolResult = true

  // ========== Step 1: 收集必要的路径信息 ==========
  console.log('\n【Step 1】收集数据路径信息')
  console.log('-'.repeat(40))

  const dynFolder = await prompt('动态数据文件夹路径: ')
  if (!dynFolder.trim()) {
    console.log('错误: 必须提供动态数据文件夹路径')
    rl.close()
    return
  }

  const staticFile = await prompt('静态文件路径 (无则回车): ')
  const fileFilter = await prompt('文件名过滤关键字 (无则回车): ')
  const researchVars = await prompt('研究变量 (逗号分隔，如 chl,no3): ')

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

  // ========== Step 2: 收集裁剪和划分参数 ==========
  console.log('\n【Step 2】收集裁剪和划分参数')
  console.log('-'.repeat(40))

  const scale = await prompt('下采样倍数 (如 4): ')
  const hSlice = await prompt('H 方向裁剪 (如 0:680，无则回车): ')
  const wSlice = await prompt('W 方向裁剪 (如 0:1440，无则回车): ')

  const trainRatio = await prompt('训练集比例 (如 0.7): ')
  const validRatio = await prompt('验证集比例 (如 0.15): ')
  const testRatio = await prompt('测试集比例 (如 0.15): ')

  // ========== Step 3: 发送 NC→NPY 转换请求 ==========
  console.log('\n【Step 3】NC → NPY 转换（含裁剪和划分）')
  console.log('-'.repeat(40))
  console.log('Agent 会分析数据并展示变量信息供你确认...')

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
  if (scale.trim()) {
    msg += `\n- 下采样倍数: ${scale}`
  }
  if (hSlice.trim()) {
    msg += `\n- H 方向裁剪: ${hSlice}`
  }
  if (wSlice.trim()) {
    msg += `\n- W 方向裁剪: ${wSlice}`
  }
  if (trainRatio.trim() && validRatio.trim() && testRatio.trim()) {
    msg += `\n- 数据集划分: train=${trainRatio}, valid=${validRatio}, test=${testRatio}`
  }

  await chat(msg, 'edit')

  // ========== 交互式确认循环 ==========
  console.log('\n' + '-'.repeat(60))
  console.log('请根据 Agent 展示的变量信息进行确认')
  console.log('输入 "next" 进入下一步，输入 "done" 结束测试')
  console.log('-'.repeat(60))

  while (true) {
    const userInput = await prompt('\n你: ')
    if (userInput.toLowerCase() === 'done') {
      console.log('\n测试结束')
      rl.close()
      return
    }
    if (userInput.toLowerCase() === 'next') {
      break
    }
    if (userInput.trim()) {
      await chat(userInput, 'edit')
    }
  }

  // ========== Step 4: 下采样 ==========
  console.log('\n【Step 4】HR → LR 下采样')
  console.log('-'.repeat(40))

  const doDownsample = await prompt('是否执行下采样？(y/n): ')
  if (doDownsample.toLowerCase() === 'y') {
    const method = await prompt('插值方法 (area/bicubic/nearest，默认 area): ')

    let downsampleMsg = `请对 ${outputDir} 目录执行下采样：
- 下采样倍数: ${scale || '4'}
- 插值方法: ${method || 'area'}`

    await chat(downsampleMsg, 'edit')

    // 等待用户确认
    while (true) {
      const userInput = await prompt('\n你 (输入 next 继续): ')
      if (userInput.toLowerCase() === 'next') break
      if (userInput.toLowerCase() === 'done') {
        rl.close()
        return
      }
      if (userInput.trim()) {
        await chat(userInput, 'edit')
      }
    }
  }

  // ========== Step 5: 可视化检查 ==========
  console.log('\n【Step 5】可视化检查')
  console.log('-'.repeat(40))

  const doVisualize = await prompt('是否生成可视化对比图？(y/n): ')
  if (doVisualize.toLowerCase() === 'y') {
    await chat(`请对 ${outputDir} 目录生成 HR vs LR 可视化对比图`, 'edit')

    // 等待用户确认
    while (true) {
      const userInput = await prompt('\n你 (输入 next 继续): ')
      if (userInput.toLowerCase() === 'next') break
      if (userInput.toLowerCase() === 'done') {
        rl.close()
        return
      }
      if (userInput.trim()) {
        await chat(userInput, 'edit')
      }
    }
  }

  // ========== Step 6: 质量指标检测 ==========
  console.log('\n【Step 6】质量指标检测')
  console.log('-'.repeat(40))

  const doMetrics = await prompt('是否计算质量指标？(y/n): ')
  if (doMetrics.toLowerCase() === 'y') {
    await chat(`请对 ${outputDir} 目录计算下采样质量指标（SSIM、Relative L2 等），下采样倍数为 ${scale || '4'}`, 'edit')

    // 等待用户确认
    while (true) {
      const userInput = await prompt('\n你 (输入 done 结束): ')
      if (userInput.toLowerCase() === 'done') break
      if (userInput.trim()) {
        await chat(userInput, 'edit')
      }
    }
  }

  // ========== 完成 ==========
  console.log('\n' + '='.repeat(60))
  console.log('海洋数据预处理完整流程测试完成！')
  console.log('='.repeat(60))
  console.log('\n输出目录结构:')
  console.log(`${outputDir}/`)
  console.log('├── train/')
  console.log('│   ├── hr/')
  console.log('│   └── lr/')
  console.log('├── valid/')
  console.log('│   ├── hr/')
  console.log('│   └── lr/')
  console.log('├── test/')
  console.log('│   ├── hr/')
  console.log('│   └── lr/')
  console.log('├── static_variables/')
  console.log('├── visualisation_data_process/')
  console.log('└── metrics_result.json')
  console.log('='.repeat(60))

  rl.close()
}

// 快速测试海洋预处理工具（自动化）
async function testOceanToolsQuick() {
  console.log('\n' + '='.repeat(60))
  console.log('海洋预处理工具快速测试（自动化）')
  console.log('='.repeat(60))

  // 重置会话
  agentId = null
  showFullToolResult = false

  // 测试数据目录
  const testDir = '/tmp/test_ocean_tools'

  // 测试 1: 加载 skill
  console.log('\n--- 测试 1: 加载 ocean-preprocess skill ---')
  await chat('加载 ocean-preprocess skill', 'edit')
  await new Promise((resolve) => setTimeout(resolve, 2000))

  // 测试 2: 查看可用工具
  console.log('\n--- 测试 2: 查看可用工具 ---')
  await chat('ocean-preprocess skill 有哪些工具可用？', 'ask')
  await new Promise((resolve) => setTimeout(resolve, 2000))

  // 测试 3: 检查测试数据
  console.log('\n--- 测试 3: 检查测试数据 ---')
  await chat(`检查 /tmp/test_split_output 目录下有什么文件`, 'edit')
  await new Promise((resolve) => setTimeout(resolve, 2000))

  console.log('\n' + '='.repeat(60))
  console.log('快速测试完成！')
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
  console.log('KODE Agent Service 交互式测试客户端 v2.0')
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
  console.log('  2. 海洋数据预处理完整流程测试（交互式）')
  console.log('  3. 海洋预处理工具快速测试（自动化）')
  console.log('  4. 单条消息测试')
  console.log('  5. 自动化测试套件（原测试）')

  const choice = await prompt('\n请选择 (1/2/3/4/5): ')

  switch (choice) {
    case '1':
      await interactiveMode()
      break
    case '2':
      await testOceanPreprocess()
      break
    case '3':
      await testOceanToolsQuick()
      break
    case '4':
      const msg = await prompt('请输入消息: ')
      await chat(msg, 'edit')
      rl.close()
      break
    case '5':
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
} else if (args.includes('--ocean-quick') || args.includes('-q')) {
  testHealth().then((healthy) => {
    if (healthy) {
      testOceanToolsQuick()
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
