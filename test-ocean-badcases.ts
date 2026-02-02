/**
 * test-ocean-badcases.ts
 *
 * Author: leizheng
 * Time: 2026-02-02
 * Description: 海洋数据预处理自动化测试
 *              - 核心防错规则测试（NaN/Inf/维度/变量缺失）
 *              - 不同问法/表达方式测试（口语化/英文夹杂/简短/专业/错别字）
 *              - 模糊请求追问测试（缺少变量/输出目录/文件路径）
 *              - 成功后输出结构展示
 * Version: 2.0.0
 *
 * Changelog:
 *   - 2026-02-02 leizheng: v2.0.0 重构测试框架，新增表达方式和模糊请求测试
 *   - 2026-02-02 leizheng: v1.0.0 初始版本，核心 bad case 测试
 */

import 'dotenv/config'
import * as fs from 'fs'
import * as path from 'path'

const API_URL = process.env.KODE_API_URL || 'http://localhost:8787'
const API_KEY = process.env.KODE_API_SECRET || 'secret-key'
const BAD_CASES_DIR = '/home/lz/Ocean-Agent-SDK/work_ocean/bad_cases'
const REPORT_DIR = '/home/lz/Ocean-Agent-SDK/test_reports'
const OUTPUT_DIR = '/tmp/ocean_badcase_test'

// ============================================================
// 测试用例配置
// ============================================================

type TestType = 'data_quality' | 'expression_variant' | 'vague_request'

interface TestCase {
  id: string
  type: TestType
  file: string
  description: string
  expectedResult: 'error' | 'pass' | 'clarify'  // clarify = 期望追问
  dynVars: string[]
  // 初始请求消息（不同问法）
  initialMessage: string
  // 模拟用户确认回答（多轮对话）
  userConfirmations: string[]
}

// ============================================================
// 1. 数据质量测试用例（核心防错）
// ============================================================
const DATA_QUALITY_CASES: TestCase[] = [
  {
    id: 'DQ-001',
    type: 'data_quality',
    file: 'bad_nan_in_data.nc',
    description: 'NaN 数据检测',
    expectedResult: 'error',
    dynVars: ['uo', 'vo'],
    initialMessage: `请帮我预处理海洋数据：
- 数据目录: ${BAD_CASES_DIR}
- 文件匹配: bad_nan_in_data.nc
- 研究变量: uo, vo
- 输出目录: ${OUTPUT_DIR}/nan_test`,
    userConfirmations: ['好的，用默认的配置', '继续处理']
  },
  {
    id: 'DQ-002',
    type: 'data_quality',
    file: 'bad_inf_in_data.nc',
    description: 'Inf 数据检测',
    expectedResult: 'error',
    dynVars: ['uo', 'vo'],
    initialMessage: `帮我处理一下这个海洋数据文件 bad_inf_in_data.nc，在 ${BAD_CASES_DIR} 目录下，变量是 uo 和 vo，输出到 ${OUTPUT_DIR}/inf_test`,
    userConfirmations: ['确认', '开始']
  },
  {
    id: 'DQ-003',
    type: 'data_quality',
    file: 'bad_coords_nan.nc',
    description: '坐标 NaN 检测',
    expectedResult: 'error',
    dynVars: ['uo', 'vo'],
    initialMessage: `我有个 nc 文件需要预处理，路径是 ${BAD_CASES_DIR}/bad_coords_nan.nc，动态变量选 uo vo，输出到 ${OUTPUT_DIR}/coords_nan`,
    userConfirmations: ['行', 'go']
  },
  {
    id: 'DQ-004',
    type: 'data_quality',
    file: 'bad_zero_dimension.nc',
    description: '零长度维度检测',
    expectedResult: 'error',
    dynVars: ['uo', 'vo'],
    initialMessage: `预处理 ${BAD_CASES_DIR}/bad_zero_dimension.nc 这个文件，uo vo 是动态变量，结果放 ${OUTPUT_DIR}/zero_dim`,
    userConfirmations: ['ok', '继续']
  },
  {
    id: 'DQ-005',
    type: 'data_quality',
    file: 'bad_no_time_dim.nc',
    description: '非标准维度(3D)检测',
    expectedResult: 'error',
    dynVars: ['uo', 'vo'],
    initialMessage: `处理海洋流速数据 bad_no_time_dim.nc，目录 ${BAD_CASES_DIR}，研究 uo vo 变量，输出 ${OUTPUT_DIR}/no_time`,
    userConfirmations: ['嗯嗯好的', '开始处理']
  },
  {
    id: 'DQ-006',
    type: 'data_quality',
    file: 'bad_extra_dim.nc',
    description: '5D数据检测',
    expectedResult: 'error',
    dynVars: ['uo', 'vo'],
    initialMessage: `${BAD_CASES_DIR} 下有个 bad_extra_dim.nc，帮我预处理成 npy，变量 uo vo，输出到 ${OUTPUT_DIR}/extra_dim`,
    userConfirmations: ['确认配置', "let's go"]
  },
  {
    id: 'DQ-007',
    type: 'data_quality',
    file: 'bad_missing_variable.nc',
    description: '缺少变量检测',
    expectedResult: 'error',
    dynVars: ['uo', 'vo'],
    initialMessage: `请预处理 ${BAD_CASES_DIR}/bad_missing_variable.nc，动态变量 uo vo，输出 ${OUTPUT_DIR}/missing_var`,
    userConfirmations: ['用默认的就行', '开始']
  },
  {
    id: 'DQ-008',
    type: 'data_quality',
    file: 'good_normal_data.nc',
    description: '正常数据（应该成功）',
    expectedResult: 'pass',
    dynVars: ['uo', 'vo'],
    initialMessage: `帮我预处理海洋数据 ${BAD_CASES_DIR}/good_normal_data.nc，动态变量是 uo 和 vo，输出到 ${OUTPUT_DIR}/good_data`,
    userConfirmations: ['确认，继续', '处理吧']
  }
]

// ============================================================
// 2. 不同问法/表达方式测试
// ============================================================
const EXPRESSION_VARIANT_CASES: TestCase[] = [
  {
    id: 'EV-001',
    type: 'expression_variant',
    file: 'good_normal_data.nc',
    description: '口语化请求',
    expectedResult: 'pass',
    dynVars: ['uo', 'vo'],
    initialMessage: `嗨，我这有个海洋数据文件要处理一下，在 ${BAD_CASES_DIR}/good_normal_data.nc，想要分析 uo 和 vo 这两个变量，处理完放到 ${OUTPUT_DIR}/casual_test 吧`,
    userConfirmations: ['行行行', '搞起']
  },
  {
    id: 'EV-002',
    type: 'expression_variant',
    file: 'good_normal_data.nc',
    description: '英文夹杂请求',
    expectedResult: 'pass',
    dynVars: ['uo', 'vo'],
    initialMessage: `help me preprocess 海洋数据 ${BAD_CASES_DIR}/good_normal_data.nc，variables 是 uo vo，output 到 ${OUTPUT_DIR}/mixed_lang`,
    userConfirmations: ['ok fine', 'go ahead']
  },
  {
    id: 'EV-003',
    type: 'expression_variant',
    file: 'good_normal_data.nc',
    description: '简短命令式',
    expectedResult: 'pass',
    dynVars: ['uo', 'vo'],
    initialMessage: `预处理 ${BAD_CASES_DIR}/good_normal_data.nc uo vo ${OUTPUT_DIR}/short_cmd`,
    userConfirmations: ['y', 'y']
  },
  {
    id: 'EV-004',
    type: 'expression_variant',
    file: 'good_normal_data.nc',
    description: '详细专业请求',
    expectedResult: 'pass',
    dynVars: ['uo', 'vo'],
    initialMessage: `我需要对 NetCDF 格式的海洋环流数据进行预处理。源数据文件位于 ${BAD_CASES_DIR}/good_normal_data.nc，包含东向流速(uo)和北向流速(vo)两个动态变量。请将其转换为 NumPy 数组格式，输出至 ${OUTPUT_DIR}/professional 目录。`,
    userConfirmations: ['配置无误，请执行', '确认开始处理']
  },
  {
    id: 'EV-005',
    type: 'expression_variant',
    file: 'good_normal_data.nc',
    description: '带错别字请求',
    expectedResult: 'pass',
    dynVars: ['uo', 'vo'],
    initialMessage: `帮我与处理下海样数据 ${BAD_CASES_DIR}/good_normal_data.nc，边量 uo vo，输入到 ${OUTPUT_DIR}/typo_test`,
    userConfirmations: ['对的', '开始']
  }
]

// ============================================================
// 3. 模糊请求测试（期望追问）
// ============================================================
const VAGUE_REQUEST_CASES: TestCase[] = [
  {
    id: 'VR-001',
    type: 'vague_request',
    file: 'good_normal_data.nc',
    description: '没有指定变量',
    expectedResult: 'clarify',
    dynVars: [],
    initialMessage: `帮我预处理 ${BAD_CASES_DIR}/good_normal_data.nc 这个海洋数据文件`,
    userConfirmations: ['uo 和 vo', '确认', '开始']
  },
  {
    id: 'VR-002',
    type: 'vague_request',
    file: 'good_normal_data.nc',
    description: '没有指定输出目录',
    expectedResult: 'clarify',
    dynVars: ['uo', 'vo'],
    initialMessage: `处理 ${BAD_CASES_DIR}/good_normal_data.nc，变量是 uo vo`,
    userConfirmations: [`${OUTPUT_DIR}/vague_output`, '好的', '处理']
  },
  {
    id: 'VR-003',
    type: 'vague_request',
    file: '',
    description: '只说要处理海洋数据',
    expectedResult: 'clarify',
    dynVars: [],
    initialMessage: `我想处理一些海洋数据`,
    userConfirmations: [
      `文件在 ${BAD_CASES_DIR}/good_normal_data.nc`,
      'uo vo',
      `${OUTPUT_DIR}/super_vague`,
      '确认',
      '开始'
    ]
  },
  {
    id: 'VR-004',
    type: 'vague_request',
    file: 'good_normal_data.nc',
    description: '不确定哪些是动态变量',
    expectedResult: 'clarify',
    dynVars: [],
    initialMessage: `预处理 ${BAD_CASES_DIR}/good_normal_data.nc，我不太确定哪些是动态变量，你帮我看看`,
    userConfirmations: ['用 uo 和 vo', '确认', '处理']
  }
]

// 合并所有测试用例
const ALL_TEST_CASES: TestCase[] = [
  ...DATA_QUALITY_CASES,
  ...EXPRESSION_VARIANT_CASES,
  ...VAGUE_REQUEST_CASES
]

// ============================================================
// SSE 客户端
// ============================================================

interface ConversationTurn {
  turnNumber: number
  role: 'user' | 'assistant'
  content: string
  toolCalls?: { tool: string; input: any; result?: any }[]
}

async function chat(
  message: string,
  agentId: string | null,
  workingDir: string
): Promise<{ agentId: string | null; response: string; toolResults: any[]; toolCalls: any[] }> {
  const body: any = {
    message,
    mode: 'edit',
    context: { userId: 'badcase-tester', workingDir }
  }
  if (agentId) body.agentId = agentId

  const response = await fetch(`${API_URL}/api/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY
    },
    body: JSON.stringify(body)
  })

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`)
  }

  const reader = response.body!.getReader()
  const decoder = new TextDecoder()

  let buffer = ''
  let fullText = ''
  let newAgentId: string | null = agentId
  const toolResults: any[] = []
  const toolCalls: { tool: string; input: any; result?: any }[] = []

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const event = JSON.parse(line.slice(6))
          switch (event.type) {
            case 'start':
              if (event.agentId) newAgentId = event.agentId
              break
            case 'text':
              fullText += event.content
              break
            case 'tool_use':
              toolCalls.push({ tool: event.tool, input: event.input })
              break
            case 'tool_result':
              toolResults.push(event.result)
              if (toolCalls.length > 0) {
                toolCalls[toolCalls.length - 1].result = event.result
              }
              break
          }
        } catch {}
      }
    }
  }

  return { agentId: newAgentId, response: fullText, toolResults, toolCalls }
}

// ============================================================
// 输出结构打印
// ============================================================

function printOutputStructure(outputDir: string): void {
  console.log(`\n📂 输出结构: ${outputDir}`)
  console.log('─'.repeat(50))

  if (!fs.existsSync(outputDir)) {
    console.log('   (目录不存在)')
    return
  }

  function walkDir(dir: string, prefix: string = ''): void {
    const items = fs.readdirSync(dir).sort()
    items.forEach((item, index) => {
      const itemPath = path.join(dir, item)
      const isLast = index === items.length - 1
      const connector = isLast ? '└── ' : '├── '
      const stats = fs.statSync(itemPath)

      if (stats.isDirectory()) {
        console.log(`   ${prefix}${connector}📁 ${item}/`)
        walkDir(itemPath, prefix + (isLast ? '    ' : '│   '))
      } else {
        const size = stats.size
        const sizeStr = size > 1024 * 1024
          ? `${(size / 1024 / 1024).toFixed(1)}MB`
          : size > 1024
            ? `${(size / 1024).toFixed(1)}KB`
            : `${size}B`
        console.log(`   ${prefix}${connector}📄 ${item} (${sizeStr})`)
      }
    })
  }

  walkDir(outputDir)
  console.log('─'.repeat(50))
}

// ============================================================
// 测试执行
// ============================================================

interface TestResult {
  id: string
  caseName: string
  type: TestType
  description: string
  status: 'PASS' | 'FAIL' | 'ERROR'
  expectedResult: string
  actualResult: string
  conversation: ConversationTurn[]
  duration: number
  outputDir?: string
  errorDetail?: string
}

async function runTestCase(testCase: TestCase): Promise<TestResult> {
  const startTime = Date.now()
  const outputDir = testCase.initialMessage.match(/输出[到至]?\s*(\S+)/)?.[1] ||
    testCase.initialMessage.match(/output[到至]?\s*(\S+)/i)?.[1] ||
    `${OUTPUT_DIR}/${testCase.id}`

  const result: TestResult = {
    id: testCase.id,
    caseName: testCase.file || '(无文件)',
    type: testCase.type,
    description: testCase.description,
    status: 'ERROR',
    expectedResult: testCase.expectedResult,
    actualResult: 'unknown',
    conversation: [],
    duration: 0,
    outputDir
  }

  // 类型标签
  const typeLabels: Record<TestType, string> = {
    data_quality: '🔬 数据质量',
    expression_variant: '💬 表达方式',
    vague_request: '❓ 模糊请求'
  }

  console.log(`\n${'═'.repeat(70)}`)
  console.log(`📋 [${testCase.id}] ${testCase.description}`)
  console.log(`   类型: ${typeLabels[testCase.type]}`)
  console.log(`   文件: ${testCase.file || '(待指定)'}`)
  console.log(`   期望: ${testCase.expectedResult === 'clarify' ? '追问用户' : testCase.expectedResult}`)
  console.log(`${'═'.repeat(70)}`)

  try {
    let agentId: string | null = null
    let turnNumber = 0
    let lastResponse = ''
    let allToolResults: any[] = []
    let didClarify = false

    // ========== 第一轮对话 ==========
    turnNumber++
    console.log(`\n┌─ 第 ${turnNumber} 轮对话 ${'─'.repeat(50)}`)
    console.log(`│ 👤 用户: ${testCase.initialMessage.replace(/\n/g, '\n│         ')}`)

    const turn1 = await chat(testCase.initialMessage, agentId, BAD_CASES_DIR)
    agentId = turn1.agentId
    lastResponse = turn1.response
    allToolResults.push(...turn1.toolResults)

    result.conversation.push({
      turnNumber,
      role: 'user',
      content: testCase.initialMessage
    })
    result.conversation.push({
      turnNumber,
      role: 'assistant',
      content: turn1.response,
      toolCalls: turn1.toolCalls
    })

    // 显示 Agent 回复
    console.log(`│`)
    console.log(`│ 🤖 Agent:`)
    const lines1 = turn1.response.split('\n').slice(0, 20)
    lines1.forEach(line => console.log(`│    ${line}`))
    if (turn1.response.split('\n').length > 20) {
      console.log(`│    ... (更多内容省略)`)
    }

    // 显示工具调用
    if (turn1.toolCalls?.length) {
      console.log(`│`)
      console.log(`│ 🔧 工具调用:`)
      turn1.toolCalls.forEach(tc => {
        console.log(`│    - ${tc.tool}`)
        if (tc.result?.status) console.log(`│      状态: ${tc.result.status}`)
        if (tc.result?.overall_status) console.log(`│      整体状态: ${tc.result.overall_status}`)
        if (tc.result?.errors?.length) {
          console.log(`│      ❌ 错误: ${tc.result.errors.slice(0, 2).join('; ')}`)
        }
      })
    }
    console.log(`└${'─'.repeat(65)}`)

    // 检查是否在追问
    const isAsking = checkIfAsking(turn1.response)
    if (isAsking) {
      didClarify = true
      console.log(`\n   💡 Agent 正在追问用户...`)
    }

    // ========== 后续对话轮次 ==========
    for (const confirmMsg of testCase.userConfirmations) {
      // 判断是否需要继续对话
      const needMore = checkNeedMoreTurns(lastResponse)
      if (!needMore) break

      turnNumber++
      console.log(`\n┌─ 第 ${turnNumber} 轮对话 ${'─'.repeat(50)}`)
      console.log(`│ 👤 用户: ${confirmMsg}`)

      const turnN = await chat(confirmMsg, agentId, BAD_CASES_DIR)
      agentId = turnN.agentId
      lastResponse = turnN.response
      allToolResults.push(...turnN.toolResults)

      result.conversation.push({
        turnNumber,
        role: 'user',
        content: confirmMsg
      })
      result.conversation.push({
        turnNumber,
        role: 'assistant',
        content: turnN.response,
        toolCalls: turnN.toolCalls
      })

      console.log(`│`)
      console.log(`│ 🤖 Agent:`)
      const linesN = turnN.response.split('\n').slice(0, 15)
      linesN.forEach(line => console.log(`│    ${line}`))
      if (turnN.response.split('\n').length > 15) {
        console.log(`│    ... (更多内容省略)`)
      }

      if (turnN.toolCalls?.length) {
        console.log(`│`)
        console.log(`│ 🔧 工具调用:`)
        turnN.toolCalls.forEach(tc => {
          console.log(`│    - ${tc.tool}`)
          if (tc.result?.status) console.log(`│      状态: ${tc.result.status}`)
          if (tc.result?.overall_status) console.log(`│      整体状态: ${tc.result.overall_status}`)
          if (tc.result?.errors?.length) {
            console.log(`│      ❌ 错误: ${tc.result.errors.slice(0, 2).join('; ')}`)
          }
        })
      }
      console.log(`└${'─'.repeat(65)}`)

      // 检查这轮是否在追问
      if (checkIfAsking(turnN.response)) {
        didClarify = true
        console.log(`\n   💡 Agent 正在追问用户...`)
      }

      await new Promise(r => setTimeout(r, 500))
    }

    // ========== 分析结果 ==========
    result.actualResult = analyzeResult(lastResponse, allToolResults, didClarify)

    // 判断测试是否通过
    if (testCase.expectedResult === 'clarify') {
      result.status = didClarify ? 'PASS' : 'FAIL'
    } else if (testCase.expectedResult === 'error') {
      result.status = result.actualResult === 'error' ? 'PASS' : 'FAIL'
    } else {
      result.status = result.actualResult === 'pass' ? 'PASS' : 'FAIL'
    }

    // ========== 成功时打印输出结构 ==========
    if (result.actualResult === 'pass' && testCase.expectedResult === 'pass') {
      printOutputStructure(outputDir)
    }

  } catch (err: any) {
    result.status = 'ERROR'
    result.errorDetail = err.message
    result.actualResult = 'exception'
  }

  result.duration = Date.now() - startTime

  // 显示结果
  const icon = result.status === 'PASS' ? '✅' : result.status === 'FAIL' ? '❌' : '⚠️'
  console.log(`\n${icon} 结果: ${result.status}`)
  console.log(`   期望: ${result.expectedResult}, 实际: ${result.actualResult}`)
  console.log(`   轮次: ${Math.ceil(result.conversation.length / 2)}`)
  console.log(`   耗时: ${(result.duration / 1000).toFixed(1)}s`)
  if (result.errorDetail) {
    console.log(`   错误: ${result.errorDetail}`)
  }

  return result
}

function checkIfAsking(response: string): boolean {
  const askPatterns = [
    /请问/,
    /请告诉我/,
    /请指定/,
    /请提供/,
    /需要.*确认/,
    /哪些.*变量/,
    /什么.*变量/,
    /输出.*目录/,
    /保存.*位置/,
    /请选择/,
    /是否/,
    /\?$/m,
    /？$/m
  ]
  return askPatterns.some(p => p.test(response))
}

function checkNeedMoreTurns(response: string): boolean {
  // 继续对话的条件
  const continuePatterns = [
    /awaiting_confirmation/i,
    /请确认/,
    /疑似/,
    /是否继续/,
    /请问/,
    /请指定/,
    /请提供/,
    /请选择/,
    /\?$/m,
    /？$/m
  ]

  // 完成的标志
  const donePatterns = [
    /预处理完成/,
    /处理完成/,
    /已完成/,
    /成功/,
    /失败.*无法继续/,
    /错误.*终止/
  ]

  if (donePatterns.some(p => p.test(response))) {
    return false
  }

  return continuePatterns.some(p => p.test(response))
}

function analyzeResult(response: string, toolResults: any[], didClarify: boolean): string {
  // 检查工具结果
  for (const tr of toolResults) {
    if (tr?.status === 'error' || tr?.overall_status === 'error') {
      return 'error'
    }
    if (tr?.errors?.length > 0) {
      return 'error'
    }
  }

  // 检查文本
  const lower = response.toLowerCase()
  if (lower.includes('❌') || lower.includes('失败') || lower.includes('无法')) {
    return 'error'
  }
  if (lower.includes('✅') && (lower.includes('完成') || lower.includes('成功'))) {
    return 'pass'
  }
  if (lower.includes('预处理完成') || lower.includes('处理完成')) {
    return 'pass'
  }

  // 如果有追问行为
  if (didClarify) {
    return 'clarify'
  }

  return 'unknown'
}

// ============================================================
// 报告生成
// ============================================================

function generateReport(results: TestResult[]): string {
  const passed = results.filter(r => r.status === 'PASS').length
  const failed = results.filter(r => r.status === 'FAIL').length
  const errors = results.filter(r => r.status === 'ERROR').length

  // 按类型分组
  const byType = {
    data_quality: results.filter(r => r.type === 'data_quality'),
    expression_variant: results.filter(r => r.type === 'expression_variant'),
    vague_request: results.filter(r => r.type === 'vague_request')
  }

  let report = `# 海洋数据预处理测试报告

生成时间: ${new Date().toLocaleString('zh-CN')}

## 总体概览

| 指标 | 数量 |
|------|------|
| 总计 | ${results.length} |
| ✅ 通过 | ${passed} |
| ❌ 失败 | ${failed} |
| ⚠️ 错误 | ${errors} |
| **通过率** | **${((passed / results.length) * 100).toFixed(0)}%** |

## 分类统计

| 测试类型 | 总数 | 通过 | 失败 | 通过率 |
|----------|------|------|------|--------|
| 🔬 数据质量 | ${byType.data_quality.length} | ${byType.data_quality.filter(r => r.status === 'PASS').length} | ${byType.data_quality.filter(r => r.status !== 'PASS').length} | ${((byType.data_quality.filter(r => r.status === 'PASS').length / byType.data_quality.length) * 100).toFixed(0)}% |
| 💬 表达方式 | ${byType.expression_variant.length} | ${byType.expression_variant.filter(r => r.status === 'PASS').length} | ${byType.expression_variant.filter(r => r.status !== 'PASS').length} | ${((byType.expression_variant.filter(r => r.status === 'PASS').length / byType.expression_variant.length) * 100).toFixed(0)}% |
| ❓ 模糊请求 | ${byType.vague_request.length} | ${byType.vague_request.filter(r => r.status === 'PASS').length} | ${byType.vague_request.filter(r => r.status !== 'PASS').length} | ${((byType.vague_request.filter(r => r.status === 'PASS').length / byType.vague_request.length) * 100).toFixed(0)}% |

## 详细结果

`

  const typeNames: Record<TestType, string> = {
    data_quality: '🔬 数据质量测试',
    expression_variant: '💬 表达方式测试',
    vague_request: '❓ 模糊请求测试'
  }

  for (const type of ['data_quality', 'expression_variant', 'vague_request'] as TestType[]) {
    const cases = byType[type]
    report += `### ${typeNames[type]}\n\n`

    for (const r of cases) {
      const icon = r.status === 'PASS' ? '✅' : r.status === 'FAIL' ? '❌' : '⚠️'
      report += `#### ${icon} [${r.id}] ${r.description}

- **文件**: ${r.caseName}
- **期望**: ${r.expectedResult}
- **实际**: ${r.actualResult}
- **状态**: ${r.status}
- **轮次**: ${Math.ceil(r.conversation.length / 2)}
- **耗时**: ${(r.duration / 1000).toFixed(1)}s
${r.errorDetail ? `- **错误**: ${r.errorDetail}` : ''}

<details>
<summary>对话记录</summary>

`
      let currentTurn = 0
      for (const turn of r.conversation) {
        if (turn.turnNumber !== currentTurn) {
          currentTurn = turn.turnNumber
          report += `**第 ${currentTurn} 轮**\n\n`
        }
        const role = turn.role === 'user' ? '👤 用户' : '🤖 Agent'
        const content = turn.content.slice(0, 300) + (turn.content.length > 300 ? '...' : '')
        report += `- ${role}: ${content.replace(/\n/g, ' ')}\n`
      }
      report += `\n</details>\n\n---\n\n`
    }
  }

  return report
}

// ============================================================
// 主程序
// ============================================================

async function main() {
  console.log('╔' + '═'.repeat(68) + '╗')
  console.log('║' + ' '.repeat(12) + '海洋数据预处理 自动化测试' + ' '.repeat(28) + '║')
  console.log('╚' + '═'.repeat(68) + '╝')
  console.log(`\nAPI: ${API_URL}`)
  console.log(`测试用例: ${ALL_TEST_CASES.length} 个`)
  console.log(`  - 🔬 数据质量: ${DATA_QUALITY_CASES.length} 个`)
  console.log(`  - 💬 表达方式: ${EXPRESSION_VARIANT_CASES.length} 个`)
  console.log(`  - ❓ 模糊请求: ${VAGUE_REQUEST_CASES.length} 个`)

  // 健康检查
  try {
    const res = await fetch(`${API_URL}/health`)
    if (!res.ok) throw new Error('Health check failed')
    console.log('\n服务状态: ✅ 正常\n')
  } catch {
    console.error('\n服务状态: ❌ 不可用')
    process.exit(1)
  }

  // 准备目录
  fs.mkdirSync(REPORT_DIR, { recursive: true })
  fs.mkdirSync(OUTPUT_DIR, { recursive: true })

  // 运行测试
  const results: TestResult[] = []
  for (const testCase of ALL_TEST_CASES) {
    const result = await runTestCase(testCase)
    results.push(result)
    await new Promise(r => setTimeout(r, 2000))
  }

  // 生成报告
  const report = generateReport(results)
  const reportPath = path.join(REPORT_DIR, `test_report_${Date.now()}.md`)
  fs.writeFileSync(reportPath, report)

  // 打印汇总
  console.log('\n' + '═'.repeat(70))
  console.log('测试完成')
  console.log('═'.repeat(70))

  const passed = results.filter(r => r.status === 'PASS').length
  const failed = results.filter(r => r.status === 'FAIL').length

  console.log(`\n✅ 通过: ${passed}`)
  console.log(`❌ 失败: ${failed}`)
  console.log(`📊 通过率: ${((passed / results.length) * 100).toFixed(0)}%`)
  console.log(`\n📄 报告: ${reportPath}`)

  if (failed > 0) {
    console.log('\n失败用例:')
    results.filter(r => r.status !== 'PASS').forEach(r => {
      console.log(`  - [${r.id}] ${r.description}: 期望 ${r.expectedResult}, 实际 ${r.actualResult}`)
    })
  }

  process.exit(failed > 0 ? 1 : 0)
}

main().catch(err => {
  console.error('测试失败:', err)
  process.exit(1)
})
