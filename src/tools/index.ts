/**
 * @author kongzhiquan
 * @contributors Leizheng
 * 工具索引文件
 * 导出所有自定义工具
 *
 * @changelog
 *   - 2026-02-25 Leizheng: 新增 oceanForecastPreprocessTools
 */

import {oceanPreprocessTools} from './ocean-SR-data-preprocess'
import {oceanSrTrainingTools} from './ocean-sr-training'
import {oceanForecastPreprocessTools} from './ocean-forecast-data-preprocess'

export default [
  ...oceanPreprocessTools,
  ...oceanSrTrainingTools,
  ...oceanForecastPreprocessTools
] as const