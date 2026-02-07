/**
 * @author kongzhiquan
 * 工具索引文件
 * 导出所有自定义工具
 */

import {oceanPreprocessTools} from './ocean-preprocess' 
import {oceanSrTrainingTools} from './ocean-sr-training'

export default [
  ...oceanPreprocessTools,
  ...oceanSrTrainingTools
] as const