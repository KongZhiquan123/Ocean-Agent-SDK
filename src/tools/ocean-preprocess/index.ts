/**
 * @file index.ts
 * @description 海洋数据预处理工具集导出
 *
 * @author leizheng
 * @date 2026-02-03
 * @version 2.4.0
 *
 * @changelog
 *   - 2026-02-03 leizheng: v2.4.0 新增下采样、可视化、指标检测工具
 */

import { oceanInspectDataTool } from './inspect'
import { oceanValidateTensorTool } from './validate'
import { oceanConvertNpyTool } from './convert'
import { oceanPreprocessFullTool } from './full'
import { oceanDownsampleTool } from './downsample'
import { oceanVisualizeTool } from './visualize'
import { oceanMetricsTool } from './metrics-tool'

export const oceanPreprocessTools = [
  oceanInspectDataTool,
  oceanValidateTensorTool,
  oceanConvertNpyTool,
  oceanPreprocessFullTool,
  oceanDownsampleTool,
  oceanVisualizeTool,
  oceanMetricsTool
]

export {
  oceanInspectDataTool,
  oceanValidateTensorTool,
  oceanConvertNpyTool,
  oceanPreprocessFullTool,
  oceanDownsampleTool,
  oceanVisualizeTool,
  oceanMetricsTool
}

export default oceanPreprocessTools
