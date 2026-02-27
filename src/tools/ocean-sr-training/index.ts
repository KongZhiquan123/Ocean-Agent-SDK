/**
 * @file index.ts
 *
 * @description 海洋超分辨率训练工具集导出
 * @author Leizheng
 * @contributors Leizheng, kongzhiquan
 * @date 2026-02-06
 * @version 2.2.0
 *
 * @changelog
 *   - 2026-02-07 kongzhiquan: v2.2.0 新增训练可视化工具
 *   - 2026-02-07 kongzhiquan: v2.1.0 新增训练状态查询工具
 *   - 2026-02-06 Leizheng: v2.0.0 新增训练报告生成工具
 *   - 2026-02-06 Leizheng: v1.0.0 初始版本
 */

import { oceanSrGpuCheckTool } from './check-gpu'
import { oceanSrModelListTool } from './list-models'
import { oceanSrTrainStartTool } from './train'
import { oceanSrTrainStatusTool } from './train-status'
import { oceanSrTrainReportTool } from './report'
import { oceanSrTrainVisualizeTool } from './visualize'

export const oceanSrTrainingTools = [
  oceanSrGpuCheckTool,
  oceanSrModelListTool,
  oceanSrTrainStartTool,
  oceanSrTrainStatusTool,
  oceanSrTrainReportTool,
  oceanSrTrainVisualizeTool
]

export {
  oceanSrGpuCheckTool,
  oceanSrModelListTool,
  oceanSrTrainStartTool,
  oceanSrTrainStatusTool,
  oceanSrTrainReportTool,
  oceanSrTrainVisualizeTool
}

export default oceanSrTrainingTools
