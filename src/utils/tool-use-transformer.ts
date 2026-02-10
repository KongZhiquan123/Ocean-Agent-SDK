/**
 * @file tool-use-transformer.ts
 *
 * @description 工具调用格式转换器，对 tool:start 输入与文案做集中格式化，只保留后端需要的信息
 * @author kongzhiquan
 * @date 2026-02-10
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-10 kongzhiquan: v1.0.0 初始版本
 *     - 新增 transformToolUse 集中格式转换器
 *     - 对常见工具调用的文案进行统一
 *     - 未注册的工具名原样透传输入
 */
import type { ToolCallSnapshot } from '@shareai-lab/kode-sdk'

/**
 * 简单状态工具的标签映射
 */
const SIMPLE_TOOL_LABELS: Record<string, string> = {
  ocean_metrics: '计算下采样数据质量指标',
  ocean_generate_report: '生成预处理报告',
  ocean_visualize: '生成可视化对比图',
  ocean_sr_check_gpu: 'GPU 检测',
  ocean_sr_list_models: '列出可用超分模型',
  ocean_sr_generate_report: '生成超分训练报告',
  ocean_sr_visualize: '生成超分训练可视化图表',
  ocean_sr_train_status: '查询超分训练状态',
  ocean_sr_train: '启动超分训练',
}

/**
 * 集中的工具调用转换器
 * 格式化 tool:start 的文案与输入
 * 未注册的工具名原样透传 inputPreview
 */
export function transformToolUse(toolCall: ToolCallSnapshot): { message: string; input?: any } {
  const { name: toolName, inputPreview } = toolCall

  if (toolName === 'ocean_preprocess_full') {
    return {
      message: '启动预处理流程...',
      input: inputPreview,
    }
  }

  const label = SIMPLE_TOOL_LABELS[toolName]
  if (label) {
    return {
      message: `开始${label}...`,
      input: inputPreview,
    }
  }

  if (toolName.startsWith('bash_')) {
    const cmd = inputPreview?.cmd || '未知命令'
    return {
      message: `执行 Bash: ${cmd}`,
      input: inputPreview,
    }
  }

  if (toolName.startsWith('fs_')) {
    const action = toolName.split('_')[1] || '未知操作'
    const path = inputPreview?.path || '未知路径'
    return {
      message: `执行文件操作: ${action}，路径: ${path}`,
      input: inputPreview,
    }
  }

  return {
    message: `正在调用工具: ${toolName}...`,
    input: inputPreview,
  }
}
