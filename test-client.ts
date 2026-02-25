/**
 * @file test-client.ts
 *
 * @description 测试 kode-agent-service 的交互式客户端
 *              支持多轮对话、海洋数据预处理完整流程测试、超分训练测试
 * @author leizheng
 * @contributors kongzhiquan, Leizheng
 * @date 2026-02-02
 * @version 3.4.0
 *
 * @changelog
 *   - 2026-02-07 Leizheng: v3.4.0 训练测试更新 OOM 防护流程
 *     - testOceanSRTraining() 工作流更新为 9 步（含显存预估 + 报告生成）
 *     - 新增 OOM 防护参数提示（use_amp, gradient_checkpointing, patch_size）
 *   - 2026-02-06 Leizheng: v3.3.0 新增超分训练测试入口
 *     - 新增 testOceanSRTraining() 交互式训练测试流程
 *     - 菜单新增选项 6、命令行参数 --train / -t
 *   - 2026-02-05 kongzhiquan: v3.2.0 合并功能更新
 *     - 新增 tool_error 事件处理，显示工具执行错误
 *     - 简化 tool_result 显示逻辑
 *     - 阶段 2.5 新增经纬度范围验证功能说明
 *     - 系统会自动检测数据的经纬度范围并展示给用户
 *     - 用户指定的裁剪范围如果超出数据边界会提示错误
 *   - 2026-02-04 kongzhiquan: v3.0.0 适配 4 阶段强制确认流程
 *     - 新增 4 阶段交互式测试流程引导
 *     - 新增阶段状态显示和流程提示
 *     - 更新测试说明，对应 SKILL.md v3.0.0
 *   - 2026-02-04 leizheng: v2.2.0 改为自然语言交互模式
 *     - 用户输入自然语言 prompt，Agent 引导对话
 *     - 符合 SKILL.md 中的交互流程设计
 *   - 2026-02-04 leizheng: v2.1.0 支持粗网格模式
 *     - 新增粗网格模式测试（HR/LR 来自不同精度模型）
 *     - 修复插值方法名称（bicubic → cubic）
 *     - 更新可视化说明（支持经纬度坐标）
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

const API_PORT = process.env.KODE_API_PORT || '8787'
const API_URL = process.env.KODE_API_URL || `http://localhost:${API_PORT}`
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
      outputsPath: './test_outputs',
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
                console.log(`[工具结果] ${event.is_error || !(event.result.status === 'success') ? '失败' : '成功'}`)
                const resultStr = JSON.stringify(event.result ?? null, null, 2) ?? 'null'
                // 显示完整结果（格式化 JSON）
                const sliceLen = showFullToolResult ? resultStr.length : Math.min(2000, resultStr.length)
                console.log('结果:', resultStr.slice(0, sliceLen) + (resultStr.length > 2000 && !showFullToolResult ? '\n... [截断]' : ''))
                break
              
              case 'tool_error':
                console.log(`\n[工具错误] ${event.tool}`)
                console.log('错误详情:', event.error)
                break

              case 'agent_error':
                console.error(`\n[Agent 错误] ${event.error}`)
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

// 测试海洋数据预处理的完整流程（v3.1 五阶段）
async function testOceanPreprocess() {
  console.log('\n' + '='.repeat(60))
  console.log('海洋数据预处理测试 v3.1（5 阶段强制确认流程）')
  console.log('='.repeat(60))

  console.log(`
┌─────────────────────────────────────────────────────────────┐
│  v3.1 工作流程（5 阶段强制确认）                            │
├─────────────────────────────────────────────────────────────┤
│  阶段 0: 启动分析                                           │
│    → 提供 NC 数据目录和输出目录                             │
│                                                             │
│  阶段 1: 研究变量选择 (awaiting_variable_selection)         │
│    → Agent 展示检测到的动态变量候选                         │
│    → 你选择要研究的变量（如 uo, vo）                        │
│                                                             │
│  阶段 2: 静态/掩码变量选择 (awaiting_static_selection)      │
│    → Agent 展示静态变量、掩码变量、坐标变量候选             │
│    → 你逐一确认：静态变量、掩码变量、经纬度变量             │
│                                                             │
│  阶段 2.5: 区域裁剪确认 (awaiting_region_selection)【新增】 │
│    → Agent 展示数据的经纬度范围（自动检测 min/max）         │
│    → 确认是否裁剪、经纬度范围、裁剪模式                     │
│    → 如果指定范围超出数据边界，会提示错误并要求重新输入     │
│    → 裁剪模式: one_step（一步到位）/ two_step（两步裁剪）   │
│                                                             │
│  阶段 3: 处理参数确认 (awaiting_parameters)                 │
│    → Agent 展示数据尺寸和推荐裁剪                           │
│    → 你确认：scale、插值方法、数据集划分、裁剪参数          │
│                                                             │
│  阶段 4: 执行确认 (awaiting_execution)                      │
│    → Agent 展示完整执行预览                                 │
│    → 你回复"确认"开始执行                                   │
│                                                             │
│  完成后: Agent 自动生成预处理报告                           │
└─────────────────────────────────────────────────────────────┘

【v3.1 新增目录结构】
  dataset_root/
  ├── train/
  │   ├── raw/    ← 区域裁剪后的原始数据（两步裁剪模式）
  │   ├── hr/     ← 尺寸裁剪后的高分辨率数据
  │   └── lr/     ← 下采样后的低分辨率数据
  ├── static_variables/
  │   ├── raw/    ← 区域裁剪后的静态变量
  │   ├── hr/     ← 尺寸裁剪后的静态变量
  │   └── lr/     ← 下采样后的静态变量（经纬度用linear，掩码用nearest）
  └── ...
`)

  console.log('请用自然语言描述您的预处理需求，例如：')
  console.log('  - "我的数据在 /data/ocean，输出到 /output，帮我做超分预处理"')
  console.log('  - "HR 数据在 /data/hr，LR 在 /data/lr，转成超分训练格式"')
  console.log('  - "/data/cmems 里有 chl 和 no3，4 倍下采样到 /output"')
  console.log('  - "我想先裁剪到南海区域，经度100-120，纬度5-25，然后再下采样"')
  console.log('  - "我不确定数据的经纬度范围，帮我分析一下"  ← 系统会自动显示范围')
  console.log('\nAgent 会引导您完成 5 个阶段的参数确认。')
  console.log('-'.repeat(60))

  // 重置会话
  agentId = null
  showFullToolResult = true

  // 用户输入初始 prompt
  const userPrompt = await prompt('\n你: ')
  if (!userPrompt.trim()) {
    console.log('错误: 请输入预处理需求')
    rl.close()
    return
  }

  // 发送初始请求
  await chat(userPrompt, 'edit')

  // 进入交互式对话循环
  console.log('\n' + '-'.repeat(60))
  console.log('继续与 Agent 对话，按阶段确认参数：')
  console.log('  - 阶段 1: 选择研究变量（如"uo, vo"）')
  console.log('  - 阶段 2: 确认静态/掩码/坐标变量')
  console.log('  - 阶段 2.5: 确认区域裁剪参数（如果启用）')
  console.log('  - 阶段 3: 确认 scale、插值方法、划分比例、裁剪')
  console.log('  - 阶段 4: 回复"确认"执行')
  console.log('')
  console.log('命令: "done" 结束测试, "reset" 重置会话, "status" 查看当前阶段')
  console.log('-'.repeat(60))

  let currentStage = 0

  while (true) {
    const userInput = await prompt('\n你: ')

    if (userInput.toLowerCase() === 'done') {
      console.log('\n测试结束')
      break
    }

    if (userInput.toLowerCase() === 'reset') {
      agentId = null
      currentStage = 0
      console.log('会话已重置，请重新描述您的需求')
      const newPrompt = await prompt('\n你: ')
      if (newPrompt.trim()) {
        await chat(newPrompt, 'edit')
      }
      continue
    }

    if (userInput.toLowerCase() === 'status') {
      console.log(`\n当前阶段: ${currentStage}`)
      console.log('阶段说明:')
      console.log('  0 = 等待启动（提供数据目录）')
      console.log('  1 = awaiting_variable_selection（选择研究变量）')
      console.log('  2 = awaiting_static_selection（选择静态/掩码变量）')
      console.log('  2.5 = awaiting_region_selection（确认区域裁剪参数）')
      console.log('  3 = awaiting_parameters（确认处理参数）')
      console.log('  4 = awaiting_execution（确认执行）')
      console.log('  5 = 执行完成，生成报告')
      continue
    }

    if (userInput.trim()) {
      await chat(userInput, 'edit')
    }
  }

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

// 测试海洋超分辨率训练流程
async function testOceanSRTraining() {
  console.log('\n' + '='.repeat(60))
  console.log('海洋超分辨率训练测试 v2.0（含 OOM 防护 + 显存预估）')
  console.log('='.repeat(60))

  console.log(`
┌─────────────────────────────────────────────────────────────┐
│  训练工作流程（v3.0.0，9 步）                                │
├─────────────────────────────────────────────────────────────┤
│  步骤 1: 确认数据目录和输出目录                              │
│    → 提供 ocean-preprocess 预处理后的数据目录                │
│    → 提供训练日志输出目录                                    │
│                                                             │
│  步骤 2: 选择模型                                            │
│    → Agent 展示可用模型列表（标准模型 / 扩散模型）           │
│    → 你选择要训练的模型                                      │
│                                                             │
│  步骤 3: 确认训练参数                                        │
│    → epochs, lr, batch_size                                 │
│    → GPU 选择：查看可用显卡，选择用哪些卡                    │
│    → 多卡模式选择（DP / DDP）                                │
│    → OOM 防护参数: use_amp, gradient_checkpointing,         │
│      patch_size                                             │
│                                                             │
│  步骤 4: 参数汇总确认                                        │
│    → Agent 展示完整参数列表（含 OOM 防护参数）               │
│    → 你回复"确认"开始                                        │
│                                                             │
│  步骤 5: 显存预估（自动，不可跳过）                           │
│    → dry-run forward+backward 测量峰值显存                  │
│    → 若 OOM → 自动降级（AMP → 减半 batch_size）           │
│                                                             │
│  步骤 6: 执行训练                                            │
│    → 训练进度输出                                            │
│    → 陆地掩码自动处理（NaN → 0，mask 排除陆地格点）         │
│                                                             │
│  步骤 7: 查看结果                                            │
│    → 训练日志、最佳模型路径、测试指标                        │
│                                                             │
│  步骤 8: 生成训练报告                                        │
│    → Agent 读取报告，补充分析与建议                          │
│                                                             │
│  步骤 9: 完成                                                │
│    → 向用户展示报告路径和关键结果                            │
└─────────────────────────────────────────────────────────────┘

【数据目录要求】（ocean-preprocess 预处理输出）
  dataset_root/
  ├── train/hr/{var}/*.npy
  ├── train/lr/{var}/*.npy
  ├── valid/hr/{var}/*.npy
  ├── valid/lr/{var}/*.npy
  ├── test/hr/{var}/*.npy
  └── test/lr/{var}/*.npy
`)

  console.log('请用自然语言描述您的训练需求，例如：')
  console.log('  - "数据在 /data/output/demo14，帮我训练超分模型"')
  console.log('  - "用 SwinIR 训练，数据在 /output，日志输出到 /logs"')
  console.log('  - "查看有哪些可用模型"')
  console.log('  - "看看当前 GPU 情况"')
  console.log('  - "显存不够，帮我开启 AMP 混合精度"')
  console.log('  - "用 patch_size=128 裁剪训练"')
  console.log('\nAgent 会引导您完成模型选择、参数确认、显存预估、训练执行。')
  console.log('-'.repeat(60))

  // 重置会话
  agentId = null
  showFullToolResult = true

  // 用户输入初始 prompt
  const userPrompt = await prompt('\n你: ')
  if (!userPrompt.trim()) {
    console.log('错误: 请输入训练需求')
    rl.close()
    return
  }

  // 发送初始请求
  await chat(userPrompt, 'edit')

  // 进入交互式对话循环
  console.log('\n' + '-'.repeat(60))
  console.log('继续与 Agent 对话：')
  console.log('  - 选择模型（如"SwinIR"、"FNO2d"）')
  console.log('  - 确认参数（epochs, lr, batch_size, GPU 等；默认 batch_size=4）')
  console.log('  - OOM 防护（"开启 AMP"、"用 patch_size=128"、"开启梯度检查点"）')
  console.log('  - FFT/频域模型默认 use_amp=false，可手动覆盖并查看风险提示')
  console.log('  - 回复"确认"开始训练')
  console.log('')
  console.log('命令: "done" 结束测试, "reset" 重置会话')
  console.log('-'.repeat(60))

  while (true) {
    const userInput = await prompt('\n你: ')

    if (userInput.toLowerCase() === 'done') {
      console.log('\n测试结束')
      break
    }

    if (userInput.toLowerCase() === 'reset') {
      agentId = null
      console.log('会话已重置，请重新描述您的需求')
      const newPrompt = await prompt('\n你: ')
      if (newPrompt.trim()) {
        await chat(newPrompt, 'edit')
      }
      continue
    }

    if (userInput.trim()) {
      await chat(userInput, 'edit')
    }
  }

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
  console.log('KODE Agent Service 交互式测试客户端 v3.4.0')
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
  console.log('  2. 海洋数据预处理测试（v3.1 五阶段流程，含区域裁剪）')
  console.log('  3. 海洋预处理工具快速测试（自动化）')
  console.log('  4. 单条消息测试')
  console.log('  5. 自动化测试套件（原测试）')
  console.log('  6. 海洋超分辨率训练测试')

  const choice = await prompt('\n请选择 (1/2/3/4/5/6): ')

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
    case '6':
      await testOceanSRTraining()
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
} else if (args.includes('--train') || args.includes('-t')) {
  testHealth().then((healthy) => {
    if (healthy) {
      testOceanSRTraining()
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
