/**
 * @file index.ts
 *
 * @description 海洋时序预测训练工具集导出
 * @author Leizheng
 * @date 2026-02-26
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-26 Leizheng: v1.0.0 初始版本
 */

import { oceanForecastCheckGpuTool } from './check-gpu'
import { oceanForecastListModelsTool } from './list-models'
import { oceanForecastTrainTool } from './train'
import { oceanForecastTrainStatusTool } from './train-status'
import { oceanForecastReportTool } from './report'
import { oceanForecastVisualizeTool } from './visualize'

export const oceanForecastTrainingTools = [
  oceanForecastCheckGpuTool,
  oceanForecastListModelsTool,
  oceanForecastTrainTool,
  oceanForecastTrainStatusTool,
  oceanForecastReportTool,
  oceanForecastVisualizeTool
]

export {
  oceanForecastCheckGpuTool,
  oceanForecastListModelsTool,
  oceanForecastTrainTool,
  oceanForecastTrainStatusTool,
  oceanForecastReportTool,
  oceanForecastVisualizeTool
}

export default oceanForecastTrainingTools
