/**
 * @file index.ts
 * @description 海洋预报数据预处理工具集导出
 *
 * @author Leizheng
 * @date 2026-02-25
 * @version 1.1.0
 *
 * @changelog
 *   - 2026-02-26 Leizheng: v1.1.0 补全模块入口
 *     - 新增 oceanInspectDataTool 具名导出（SR/Forecast 共用，通过 SR 模块注册，此处仅导出供引用）
 *     - 新增 oceanForecastStatsTool 工具注册与导出
 *   - 2026-02-25 Leizheng: v1.0.0 初始版本
 *     - 包含 ocean_forecast_preprocess_full、ocean_forecast_visualize、ocean_forecast_generate_report
 *     - 导出 ForecastWorkflow 状态机相关类型
 */

import { oceanForecastPreprocessFullTool } from './full'
import { oceanForecastVisualizeTool } from './visualize'
import { oceanForecastReportTool } from './report'
import { oceanForecastStatsTool } from './stats'

export {
  ForecastWorkflow,
  WorkflowState,
  type WorkflowParams,
  type WorkflowStateType,
  type StageCheckResult,
  type StagePromptResult
} from './workflow-state'

// ocean_inspect_data 是 SR/Forecast 共用工具，已通过 SR 模块（oceanPreprocessTools）注册到全局
// 此处仅做具名导出，供模块消费者直接引用，不再二次注册
export { oceanInspectDataTool } from './inspect'

export const oceanForecastPreprocessTools = [
  oceanForecastPreprocessFullTool,
  oceanForecastVisualizeTool,
  oceanForecastReportTool,
  oceanForecastStatsTool
]

export {
  oceanForecastPreprocessFullTool,
  oceanForecastVisualizeTool,
  oceanForecastReportTool,
  oceanForecastStatsTool
}

export default oceanForecastPreprocessTools
