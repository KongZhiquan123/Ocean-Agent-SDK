/**
 * @file index.ts
 *
 * @description 海洋超分辨率训练工具集导出
 * @author Leizheng
 * @date 2026-02-06
 * @version 2.0.0
 *
 * @changelog
 *   - 2026-02-06 Leizheng: v2.0.0 新增训练报告生成工具
 *   - 2026-02-06 Leizheng: v1.0.0 初始版本
 */

import { oceanSrCheckGpuTool } from './check-gpu'
import { oceanSrListModelsTool } from './list-models'
import { oceanSrTrainTool } from './train'
import { oceanSrReportTool } from './report'

export const oceanSrTrainingTools = [
  oceanSrCheckGpuTool,
  oceanSrListModelsTool,
  oceanSrTrainTool,
  oceanSrReportTool
]

export {
  oceanSrCheckGpuTool,
  oceanSrListModelsTool,
  oceanSrTrainTool,
  oceanSrReportTool
}

export default oceanSrTrainingTools
