/**
 * @file index.ts
 * @description 海洋数据预处理工具集导出
 *
 * @author leizheng
 * @contributors kongzhiquan
 * @date 2026-02-03
 * @version 2.9.0
 *
 * @changelog
 *   - 2026-02-04 kongzhiquan: v3.0.0 增加 oceanReportTool 用于生成预处理报告
 *   - 2026-02-04 leizheng: v2.9.0 分阶段强制确认流程
 *     - full.ts 实现4阶段强制停止点
 *     - 研究变量/静态变量/参数/执行 分别确认
 *   - 2026-02-04 leizheng: v2.8.0 配合 SKILL.md v2.8.0
 */

import { oceanInspectDataTool } from './inspect'
import { oceanValidateTensorTool } from './validate'
import { oceanConvertNpyTool } from './convert'
import { oceanPreprocessFullTool } from './full'
import { oceanDownsampleTool } from './downsample'
import { oceanVisualizeTool } from './visualize'
import { oceanMetricsTool } from './metrics-tool'
import { oceanReportTool } from './report'

export const oceanPreprocessTools = [
  oceanInspectDataTool,
  oceanValidateTensorTool,
  oceanConvertNpyTool,
  oceanPreprocessFullTool,
  oceanDownsampleTool,
  oceanVisualizeTool,
  oceanMetricsTool,
  oceanReportTool
]

export {
  oceanInspectDataTool,
  oceanValidateTensorTool,
  oceanConvertNpyTool,
  oceanPreprocessFullTool,
  oceanDownsampleTool,
  oceanVisualizeTool,
  oceanMetricsTool,
  oceanReportTool
}

export default oceanPreprocessTools
