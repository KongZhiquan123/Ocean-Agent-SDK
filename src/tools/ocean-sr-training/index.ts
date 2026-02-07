/**
 * @file index.ts
 *
 * @description 海洋超分辨率训练工具集导出
 * @author Leizheng
 * @contributors Leizheng, kongzhiquan
 * @date 2026-02-06
 * @version 2.1.0
 *
 * @changelog
 *   - 2026-02-07 kongzhiquan: v2.1.0 新增训练状态查询工具
 *   - 2026-02-06 Leizheng: v2.0.0 新增训练报告生成工具
 *   - 2026-02-06 Leizheng: v1.0.0 初始版本
 */

import { oceanSrCheckGpuTool } from './check-gpu'
import { oceanSrListModelsTool } from './list-models'
import { oceanSrTrainTool } from './train'
import { oceanSrTrainStatusTool } from './train-status'
import { oceanSrReportTool } from './report'

export const oceanSrTrainingTools = [
  oceanSrCheckGpuTool,
  oceanSrListModelsTool,
  oceanSrTrainTool,
  oceanSrTrainStatusTool,
  oceanSrReportTool
]

export {
  oceanSrCheckGpuTool,
  oceanSrListModelsTool,
  oceanSrTrainTool,
  oceanSrTrainStatusTool,
  oceanSrReportTool
}

export default oceanSrTrainingTools
