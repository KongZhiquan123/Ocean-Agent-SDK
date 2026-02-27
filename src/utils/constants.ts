/**
 * 简单状态工具的标签映射
 */
export const SIMPLE_TOOL_LABELS: Record<string, string> = {
  ocean_sr_preprocess_metrics: '计算下采样数据质量指标',
  ocean_sr_preprocess_report: '生成预处理报告',
  ocean_sr_preprocess_visualize: '生成预处理后的可视化对比图',
  ocean_sr_preprocess_downsample: '下采样数据生成',
  ocean_sr_preprocess_convert_npy: '转换数据为 NPY 格式',
  ocean_sr_check_gpu: 'GPU 检测',
  ocean_sr_list_models: '列出可用超分模型',
  ocean_sr_train_report: '生成超分训练报告',
  ocean_sr_train_visualize: '生成超分训练可视化图表',
  ocean_sr_train_status: '查询超分训练状态',
  ocean_sr_train_start: '启动超分训练',
  ocean_forecast_preprocess_visualize: '生成预报数据可视化',
  ocean_forecast_preprocess_report: '生成预报数据预处理报告',
  ocean_forecast_preprocess_stats: '预报数据统计分析',
  ocean_forecast_check_gpu: '预报训练 GPU 检测',
  ocean_forecast_list_models: '列出可用预报模型',
  ocean_forecast_train_start: '启动预报训练',
  ocean_forecast_train_status: '查询预报训练状态',
  ocean_forecast_train_visualize: '生成预报训练可视化图表',
  ocean_forecast_train_report: '生成预报训练报告',
  ocean_inspect_data: '数据检查',
  ocean_validate_tensor: '张量约定验证'
}