/**
 * @file index.ts
 * @description 海洋数据预处理工具集导出
 *
 * @author leizheng
 * @contributors kongzhiquan
 * @date 2026-02-04
 * @version 3.0.3
 *
 * @changelog
 *   - 2026-02-05 kongzhiquan: v3.0.3 新增执行确认 Token 机制
 *     - workflow-state.ts 新增 confirmation_token 验证
 *     - 防止 Agent 跳过 awaiting_execution 阶段
 *   - 2026-02-05 kongzhiquan: v3.0.2 新增状态机模块
 *     - full.ts, report.ts, visualize.ts 统一为 v3.0.0
 *     - visualize 新增统计分布图（均值/方差时序、直方图）
 *     - report 新增可视化图片嵌入（对比图 + 统计图）
 *     - 4 阶段强制确认流程完整实现
 *   - 2026-02-04 leizheng: v2.9.0 分阶段强制确认流程
 *     - full.ts 实现4阶段强制停止点
 *     - 研究变量/静态变量/参数/执行 分别确认
 *   - 2026-02-04 leizheng: v2.8.0 配合 SKILL.md v2.8.0
 *   - 2026-02-04 kongzhiquan: v2.6.0 新增报告生成工具
 *   - 2026-02-03 leizheng: v2.4.0 新增下采样、可视化、指标检测工具
 */

import { oceanInspectDataTool } from './inspect'
import { oceanValidateTensorTool } from './validate'
import { oceanSrPreprocessConvertNpyTool } from './convert'
import { oceanSrPreprocessFullTool } from './full'
import { oceanSrPreprocessDownsampleTool } from './downsample'
import { oceanSrPreprocessVisualizeTool } from './visualize'
import { oceanSrPreprocessMetricsTool } from './metrics-tool'
import { oceanSrPreprocessReportTool } from './report'

// 导出状态机相关类型和类
export {
  PreprocessWorkflow,
  WorkflowState,
  type WorkflowParams,
  type WorkflowStateType,
  type StageCheckResult,
  type StagePromptResult
} from './workflow-state'

export const oceanSrPreprocessTools = [
  oceanInspectDataTool,
  oceanValidateTensorTool,
  oceanSrPreprocessConvertNpyTool,
  oceanSrPreprocessFullTool,
  oceanSrPreprocessDownsampleTool,
  oceanSrPreprocessVisualizeTool,
  oceanSrPreprocessMetricsTool,
  oceanSrPreprocessReportTool
]

export {
  oceanInspectDataTool,
  oceanValidateTensorTool,
  oceanSrPreprocessConvertNpyTool,
  oceanSrPreprocessFullTool,
  oceanSrPreprocessDownsampleTool,
  oceanSrPreprocessVisualizeTool,
  oceanSrPreprocessMetricsTool,
  oceanSrPreprocessReportTool
}

export default oceanSrPreprocessTools
