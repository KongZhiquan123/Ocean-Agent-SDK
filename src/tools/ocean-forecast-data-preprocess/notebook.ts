/**
 * @file notebook.ts
 *
 * @description 海洋预报数据预处理 Jupyter Notebook 生成模块
 *              在预处理流水线完成后，生成可复现的 Notebook
 *              Notebook 使用 subprocess 调用 Python 脚本（保持与预处理框架一致）
 *
 * @author kongzhiquan
 * @date 2026-02-26
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-26 kongzhiquan: v1.0.0 初始版本
 *     - 参考 ocean-SR-data-preprocess/notebook.ts 架构
 *     - 适配预报数据预处理流程（无下采样、无 hr/lr 层级）
 *     - Step A: 数据检查 → Step B: forecast_preprocess.py → Step C: 可视化
 */

import { toPyRepr, mdCell, codeCell } from '@/utils/notebook'
import type { NotebookCell, Notebook } from '@/utils/notebook'

export type { NotebookCell, Notebook }
export { saveOrAppendNotebook } from '@/utils/notebook'

export interface ForecastNotebookParams {
  outputBase: string
  ncFolder: string
  staticFile?: string
  dynVars: string[]
  statVars: string[]
  maskVars: string[]
  lonVar?: string
  latVar?: string
  trainRatio: number
  validRatio: number
  testRatio: number
  hSlice?: string
  wSlice?: string
  allowNan: boolean
  dynFilePattern: string
  useDateFilename: boolean
  dateFormat: string
  timeVar?: string
  chunkSize: number
  maxFiles?: number
  skipVisualize: boolean
  pythonPath: string
  ncFiles?: string[]
}


// ========================================
// Cell 生成函数
// ========================================

function generateTitleCell(params: ForecastNotebookParams): NotebookCell {
  const timestamp = new Date().toISOString().replace('T', ' ').slice(0, 19)

  return mdCell(
    `# 海洋预报数据预处理 Notebook\n` +
    `\n` +
    `本 Notebook 记录了完整的 NC → NPY 预报数据预处理流程。\n` +
    `运行此 Notebook 可复现与工具调用完全相同的预处理结果。\n` +
    `\n` +
    `**与超分辨率预处理的区别**：\n` +
    `- 无下采样步骤（无 hr/lr 目录层级）\n` +
    `- 输出直接存储在 \`split/var_name/\` 目录下\n` +
    `- 数据按 NC 文件内时间变量**严格升序**排列\n` +
    `- 生成 \`time_index.json\` 记录完整时间戳溯源\n` +
    `\n` +
    `| 项目 | 值 |\n` +
    `|------|----|\n` +
    `| 模式 | 预报数据预处理 |\n` +
    `| 生成时间 | ${timestamp} |\n` +
    `| 数据源 | \`${params.ncFolder}\` |\n` +
    `| 输出目录 | \`${params.outputBase}\` |`
  )
}

function generateSetupCell(params: ForecastNotebookParams): NotebookCell {
  const lines: string[] = [
    'import subprocess',
    'import sys',
    'import os',
    'import json',
    '',
    '# ====== 路径配置 ======',
    `OUTPUT_BASE = ${toPyRepr(params.outputBase)}`,
    `NC_FOLDER = ${toPyRepr(params.ncFolder)}`,
    `STATIC_FILE = ${toPyRepr(params.staticFile)}`,
    `SCRIPT_DIR = os.path.join(OUTPUT_BASE, "_ocean_forecast_preprocess_code")`,
    `TEMP_DIR = os.path.join(OUTPUT_BASE, ".ocean_forecast_temp")`,
    `PYTHON_PATH = ${toPyRepr(params.pythonPath)}`,
    'os.makedirs(TEMP_DIR, exist_ok=True)',
    '',
    '# ====== 变量配置（用户确认值）======',
    `DYN_VARS = ${toPyRepr(params.dynVars)}`,
    `STAT_VARS = ${toPyRepr(params.statVars)}`,
    `MASK_VARS = ${toPyRepr(params.maskVars)}`,
    `LON_VAR = ${toPyRepr(params.lonVar)}`,
    `LAT_VAR = ${toPyRepr(params.latVar)}`,
    '',
    '# ====== 处理参数 ======',
    `TRAIN_RATIO = ${toPyRepr(params.trainRatio)}`,
    `VALID_RATIO = ${toPyRepr(params.validRatio)}`,
    `TEST_RATIO = ${toPyRepr(params.testRatio)}`,
    `H_SLICE = ${toPyRepr(params.hSlice)}`,
    `W_SLICE = ${toPyRepr(params.wSlice)}`,
    `ALLOW_NAN = ${toPyRepr(params.allowNan)}`,
    `DYN_FILE_PATTERN = ${toPyRepr(params.dynFilePattern)}`,
    `CHUNK_SIZE = ${toPyRepr(params.chunkSize)}`,
    '',
    '# ====== 日期文件名 ======',
    `USE_DATE_FILENAME = ${toPyRepr(params.useDateFilename)}`,
    `DATE_FORMAT = ${toPyRepr(params.dateFormat)}`,
    `TIME_VAR = ${toPyRepr(params.timeVar)}`,
    '',
    '# ====== 其他 ======',
    `MAX_FILES = ${toPyRepr(params.maxFiles)}`,
  ]

  if (params.ncFiles && params.ncFiles.length > 0) {
    lines.push(
      '',
      '# ====== 指定文件列表 ======',
      `NC_FILES = ${toPyRepr(params.ncFiles)}`,
    )
  } else {
    lines.push(
      '',
      'NC_FILES = None',
    )
  }

  return codeCell(lines.join('\n'))
}

function generateStepACells(): NotebookCell[] {
  return [
    mdCell(
      `## Step A: 数据检查与变量分类\n` +
      `\n` +
      `扫描 NC 文件目录，分析变量信息并自动分类为动态/静态/掩码类型。`
    ),
    codeCell(
      `inspect_config = {\n` +
      `    "nc_folder": NC_FOLDER,\n` +
      `    "static_file": STATIC_FILE,\n` +
      `    "dyn_file_pattern": DYN_FILE_PATTERN,\n` +
      `}\n` +
      `if NC_FILES is not None:\n` +
      `    inspect_config["nc_files"] = NC_FILES\n` +
      `\n` +
      `config_path = os.path.join(TEMP_DIR, "inspect_config.json")\n` +
      `output_path = os.path.join(TEMP_DIR, "inspect_result.json")\n` +
      `with open(config_path, "w", encoding="utf-8") as f:\n` +
      `    json.dump(inspect_config, f, ensure_ascii=False)\n` +
      `\n` +
      `result = subprocess.run(\n` +
      `    [PYTHON_PATH, os.path.join(SCRIPT_DIR, "inspect_data.py"),\n` +
      `     "--config", config_path, "--output", output_path],\n` +
      `    capture_output=True, text=True\n` +
      `)\n` +
      `if result.returncode == 0:\n` +
      `    with open(output_path, "r", encoding="utf-8") as f:\n` +
      `        inspect_result = json.load(f)\n` +
      `    print(f"状态: {inspect_result.get('status')}")\n` +
      `    print(f"文件数量: {inspect_result.get('file_count')}")\n` +
      `    print(f"动态变量候选: {inspect_result.get('dynamic_vars_candidates')}")\n` +
      `    print(f"疑似掩码变量: {inspect_result.get('suspected_masks')}")\n` +
      `    print(f"疑似坐标变量: {inspect_result.get('suspected_coordinates')}")\n` +
      `else:\n` +
      `    print("Step A 失败:", result.stderr)\n` +
      `    raise RuntimeError("数据检查失败，请检查 NC_FOLDER 路径是否正确")`
    )
  ]
}

function generateStepBCells(): NotebookCell[] {
  return [
    mdCell(
      `## Step B: NC → NPY 转换（预报模式）\n` +
      `\n` +
      `将 NC 文件中的动态变量按时间严格升序转换为 NPY 格式，\n` +
      `并划分为训练集/验证集/测试集。\n` +
      `\n` +
      `- 输出结构: \`{split}/{var_name}/{date}.npy\`\n` +
      `- 生成 \`time_index.json\` 记录时间戳溯源\n` +
      `- 生成 \`var_names.json\` 记录变量配置`
    ),
    codeCell(
      `forecast_config = {\n` +
      `    "nc_folder": NC_FOLDER,\n` +
      `    "output_base": OUTPUT_BASE,\n` +
      `    "dyn_vars": DYN_VARS,\n` +
      `    "stat_vars": STAT_VARS,\n` +
      `    "mask_vars": MASK_VARS,\n` +
      `    "lon_var": LON_VAR,\n` +
      `    "lat_var": LAT_VAR,\n` +
      `    "train_ratio": TRAIN_RATIO,\n` +
      `    "valid_ratio": VALID_RATIO,\n` +
      `    "test_ratio": TEST_RATIO,\n` +
      `    "h_slice": H_SLICE,\n` +
      `    "w_slice": W_SLICE,\n` +
      `    "dyn_file_pattern": DYN_FILE_PATTERN,\n` +
      `    "chunk_size": CHUNK_SIZE,\n` +
      `    "use_date_filename": USE_DATE_FILENAME,\n` +
      `    "date_format": DATE_FORMAT,\n` +
      `    "time_var": TIME_VAR,\n` +
      `    "max_files": MAX_FILES,\n` +
      `    "run_validation": True,\n` +
      `    "allow_nan": ALLOW_NAN,\n` +
      `}\n` +
      `if NC_FILES is not None:\n` +
      `    forecast_config["nc_files"] = NC_FILES\n` +
      `\n` +
      `config_path = os.path.join(TEMP_DIR, "forecast_config.json")\n` +
      `output_path = os.path.join(TEMP_DIR, "forecast_result.json")\n` +
      `with open(config_path, "w", encoding="utf-8") as f:\n` +
      `    json.dump(forecast_config, f, ensure_ascii=False)\n` +
      `\n` +
      `result = subprocess.run(\n` +
      `    [PYTHON_PATH, os.path.join(SCRIPT_DIR, "forecast_preprocess.py"),\n` +
      `     "--config", config_path, "--output", output_path],\n` +
      `    capture_output=True, text=True\n` +
      `)\n` +
      `if result.returncode == 0:\n` +
      `    with open(output_path, "r", encoding="utf-8") as f:\n` +
      `        forecast_result = json.load(f)\n` +
      `    print(f"转换状态: {forecast_result.get('status')}")\n` +
      `    print(f"消息: {forecast_result.get('message')}")\n` +
      `    if forecast_result.get("splits"):\n` +
      `        for split_name, split_info in forecast_result["splits"].items():\n` +
      `            print(f"  {split_name}: {split_info.get('total_files', '?')} 个文件")\n` +
      `    if forecast_result.get("warnings"):\n` +
      `        for w in forecast_result["warnings"]:\n` +
      `            print(f"  警告: {w}")\n` +
      `else:\n` +
      `    print("Step B 失败:", result.stderr)\n` +
      `    raise RuntimeError("预报数据转换失败")`
    )
  ]
}

function generateStepCCells(): NotebookCell[] {
  return [
    mdCell(
      `## Step C: 可视化检查\n` +
      `\n` +
      `生成样本帧空间分布图和时序统计图。\n` +
      `\n` +
      `- \`{var}_frames.png\` - 样本帧空间分布\n` +
      `- \`{var}_timeseries.png\` - 时序统计曲线`
    ),
    codeCell(
      `vis_out_dir = os.path.join(OUTPUT_BASE, "visualisation_forecast")\n` +
      `\n` +
      `result = subprocess.run(\n` +
      `    [PYTHON_PATH, os.path.join(SCRIPT_DIR, "forecast_visualize.py"),\n` +
      `     "--dataset_root", OUTPUT_BASE,\n` +
      `     "--splits", "train", "valid", "test",\n` +
      `     "--out_dir", vis_out_dir],\n` +
      `    capture_output=True, text=True\n` +
      `)\n` +
      `if result.returncode == 0:\n` +
      `    print("可视化完成")\n` +
      `    print(f"图片保存在: {vis_out_dir}")\n` +
      `else:\n` +
      `    print("Step C 失败:", result.stderr)\n` +
      `    raise RuntimeError("可视化失败")`
    )
  ]
}

function generateCompletionCell(): NotebookCell {
  return mdCell(
    `## 预处理完成\n` +
    `\n` +
    `所有步骤已执行完毕。输出目录结构：\n` +
    `\n` +
    `\`\`\`\n` +
    `output_base/\n` +
    `├── train/{var_name}/*.npy     # 训练集\n` +
    `├── valid/{var_name}/*.npy     # 验证集\n` +
    `├── test/{var_name}/*.npy      # 测试集\n` +
    `├── static_variables/*.npy     # 静态变量\n` +
    `├── time_index.json            # 时间戳溯源\n` +
    `├── var_names.json             # 变量配置\n` +
    `├── preprocess_manifest.json   # 预处理清单\n` +
    `├── visualisation_forecast/    # 可视化图片\n` +
    `└── _ocean_forecast_preprocess_code/  # 预处理脚本\n` +
    `\`\`\``
  )
}


// ========================================
// 主导出函数
// ========================================

/**
 * 根据预报预处理参数生成所有 Notebook cells
 */
export function generateForecastPreprocessCells(params: ForecastNotebookParams): NotebookCell[] {
  const cells: NotebookCell[] = []

  // 标题 + 环境设置
  cells.push(generateTitleCell(params))
  cells.push(generateSetupCell(params))

  // Step A: 数据检查
  cells.push(...generateStepACells())

  // Step B: NC → NPY 转换（预报模式）
  cells.push(...generateStepBCells())

  // Step C: 可视化（未跳过）
  if (!params.skipVisualize) {
    cells.push(...generateStepCCells())
  }

  // 完成说明
  cells.push(generateCompletionCell())

  return cells
}
