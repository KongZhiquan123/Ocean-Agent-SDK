/**
 * @file notebook.ts
 *
 * @description 海洋数据预处理 Jupyter Notebook 生成模块
 *              在预处理流水线完成后，生成可复现的 Notebook
 *              Notebook 使用 subprocess 调用 Python 脚本（保持与预处理框架一致）
 *
 * @author kongzhiquan
 * @date 2026-02-25
 * @version 2.0.0
 *
 * @changelog
 *   - 2026-02-25 kongzhiquan: v1.1.0 公共工具迁移至 @/utils/notebook
 *   - 2026-02-25 kongzhiquan: v1.0.0 初始版本
 */

// ========================================
// 公共工具从 utils 导入
// ========================================
import { toPyRepr, mdCell, codeCell } from '@/utils/notebook'
import type { NotebookCell, Notebook } from '@/utils/notebook'

export type { NotebookCell, Notebook }
export { saveOrAppendNotebook } from '@/utils/notebook'

export interface NotebookParams {
  outputBase: string
  ncFolder: string
  staticFile?: string
  dynVars: string[]
  statVars: string[]
  maskVars: string[]
  lonVar?: string
  latVar?: string
  primaryMaskVar?: string
  trainRatio: number
  validRatio: number
  testRatio: number
  scale?: number
  downsampleMethod?: string
  hSlice?: string
  wSlice?: string
  workers: number
  allowNan: boolean
  dynFilePattern: string
  enableRegionCrop: boolean
  cropLonRange?: [number, number]
  cropLatRange?: [number, number]
  cropMode: string
  useDateFilename: boolean
  dateFormat: string
  timeVar?: string
  isNumericalModelMode: boolean
  lrNcFolder?: string
  lrStaticFile?: string
  lrDynFilePattern?: string
  maxFiles?: number
  skipDownsample: boolean
  skipVisualize: boolean
  pythonPath: string
}


// ========================================
// Cell 生成函数
// ========================================

function generateTitleCell(params: NotebookParams): NotebookCell {
  const mode = params.isNumericalModelMode ? '粗网格模式（数值模型）' : '下采样模式'
  const timestamp = new Date().toISOString().replace('T', ' ').slice(0, 19)

  return mdCell(
    `# 海洋数据预处理 Notebook\n` +
    `\n` +
    `本 Notebook 记录了完整的 NC → NPY 数据预处理流程。\n` +
    `运行此 Notebook 可复现与工具调用完全相同的预处理结果。\n` +
    `\n` +
    `| 项目 | 值 |\n` +
    `|------|----|\n` +
    `| 模式 | ${mode} |\n` +
    `| 生成时间 | ${timestamp} |\n` +
    `| 数据源 | \`${params.ncFolder}\` |\n` +
    `| 输出目录 | \`${params.outputBase}\` |`
  )
}

function generateSetupCell(params: NotebookParams): NotebookCell {
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
    `SCRIPT_DIR = os.path.join(OUTPUT_BASE, "_ocean_preprocess_code")`,
    `TEMP_DIR = os.path.join(OUTPUT_BASE, ".ocean_preprocess_temp")`,
    `PYTHON_PATH = ${toPyRepr(params.pythonPath)}`,
    'os.makedirs(TEMP_DIR, exist_ok=True)',
    '',
    '# ====== 变量配置（用户确认值）======',
    `DYN_VARS = ${toPyRepr(params.dynVars)}`,
    `STAT_VARS = ${toPyRepr(params.statVars)}`,
    `MASK_VARS = ${toPyRepr(params.maskVars)}`,
    `LON_VAR = ${toPyRepr(params.lonVar)}`,
    `LAT_VAR = ${toPyRepr(params.latVar)}`,
    `PRIMARY_MASK_VAR = ${toPyRepr(params.primaryMaskVar)}`,
    '',
    '# ====== 处理参数 ======',
    `TRAIN_RATIO = ${toPyRepr(params.trainRatio)}`,
    `VALID_RATIO = ${toPyRepr(params.validRatio)}`,
    `TEST_RATIO = ${toPyRepr(params.testRatio)}`,
    `SCALE = ${toPyRepr(params.scale)}`,
    `DOWNSAMPLE_METHOD = ${toPyRepr(params.downsampleMethod)}`,
    `H_SLICE = ${toPyRepr(params.hSlice)}`,
    `W_SLICE = ${toPyRepr(params.wSlice)}`,
    `WORKERS = ${toPyRepr(params.workers)}`,
    `ALLOW_NAN = ${toPyRepr(params.allowNan)}`,
    `DYN_FILE_PATTERN = ${toPyRepr(params.dynFilePattern)}`,
    '',
    '# ====== 区域裁剪 ======',
    `ENABLE_REGION_CROP = ${toPyRepr(params.enableRegionCrop)}`,
    `CROP_LON_RANGE = ${toPyRepr(params.cropLonRange)}`,
    `CROP_LAT_RANGE = ${toPyRepr(params.cropLatRange)}`,
    `CROP_MODE = ${toPyRepr(params.cropMode)}`,
    '',
    '# ====== 日期文件名 ======',
    `USE_DATE_FILENAME = ${toPyRepr(params.useDateFilename)}`,
    `DATE_FORMAT = ${toPyRepr(params.dateFormat)}`,
    `TIME_VAR = ${toPyRepr(params.timeVar)}`,
    '',
    '# ====== 其他 ======',
    `MAX_FILES = ${toPyRepr(params.maxFiles)}`,
  ]

  // 粗网格模式特有参数
  if (params.isNumericalModelMode) {
    lines.push(
      '',
      '# ====== 粗网格模式参数 ======',
      `LR_NC_FOLDER = ${toPyRepr(params.lrNcFolder)}`,
      `LR_STATIC_FILE = ${toPyRepr(params.lrStaticFile)}`,
      `LR_DYN_FILE_PATTERN = ${toPyRepr(params.lrDynFilePattern)}`,
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
      `## Step B: 张量约定验证\n` +
      `\n` +
      `验证变量的张量形状是否符合约定：\n` +
      `- 动态变量: \`[D, H, W]\` 或 \`[T, D, H, W]\`\n` +
      `- 静态/掩码变量: \`[H, W]\``
    ),
    codeCell(
      `validate_config = {\n` +
      `    "inspect_result_path": os.path.join(TEMP_DIR, "inspect_result.json"),\n` +
      `    "research_vars": DYN_VARS,\n` +
      `    "mask_vars": MASK_VARS,\n` +
      `}\n` +
      `\n` +
      `config_path = os.path.join(TEMP_DIR, "validate_config.json")\n` +
      `output_path = os.path.join(TEMP_DIR, "validate_result.json")\n` +
      `with open(config_path, "w", encoding="utf-8") as f:\n` +
      `    json.dump(validate_config, f, ensure_ascii=False)\n` +
      `\n` +
      `result = subprocess.run(\n` +
      `    [PYTHON_PATH, os.path.join(SCRIPT_DIR, "validate_tensor.py"),\n` +
      `     "--config", config_path, "--output", output_path],\n` +
      `    capture_output=True, text=True\n` +
      `)\n` +
      `if result.returncode == 0:\n` +
      `    with open(output_path, "r", encoding="utf-8") as f:\n` +
      `        validate_result = json.load(f)\n` +
      `    print(f"验证状态: {validate_result.get('status')}")\n` +
      `    print(f"消息: {validate_result.get('message')}")\n` +
      `    if validate_result.get("warnings"):\n` +
      `        for w in validate_result["warnings"]:\n` +
      `            print(f"  警告: {w}")\n` +
      `    if validate_result.get("errors"):\n` +
      `        for e in validate_result["errors"]:\n` +
      `            print(f"  错误: {e}")\n` +
      `else:\n` +
      `    print("Step B 失败:", result.stderr)\n` +
      `    raise RuntimeError("张量验证失败")`
    )
  ]
}

function generateStepCCells(): NotebookCell[] {
  return [
    mdCell(
      `## Step C: NC → NPY 转换 (HR)\n` +
      `\n` +
      `将 NC 文件中的动态变量按时间顺序转换为 NPY 格式，\n` +
      `并划分为训练集/验证集/测试集。`
    ),
    codeCell(
      `convert_config = {\n` +
      `    "nc_folder": NC_FOLDER,\n` +
      `    "output_base": OUTPUT_BASE,\n` +
      `    "dyn_vars": DYN_VARS,\n` +
      `    "static_file": STATIC_FILE,\n` +
      `    "dyn_file_pattern": DYN_FILE_PATTERN,\n` +
      `    "stat_vars": STAT_VARS,\n` +
      `    "mask_vars": MASK_VARS,\n` +
      `    "lon_var": LON_VAR,\n` +
      `    "lat_var": LAT_VAR,\n` +
      `    "run_validation": True,\n` +
      `    "allow_nan": ALLOW_NAN,\n` +
      `    "mask_src_var": PRIMARY_MASK_VAR,\n` +
      `    "mask_derive_op": "identity",\n` +
      `    "train_ratio": TRAIN_RATIO,\n` +
      `    "valid_ratio": VALID_RATIO,\n` +
      `    "test_ratio": TEST_RATIO,\n` +
      `    "h_slice": H_SLICE,\n` +
      `    "w_slice": W_SLICE,\n` +
      `    "scale": SCALE,\n` +
      `    "workers": WORKERS,\n` +
      `    "output_subdir": "hr",\n` +
      `    "enable_region_crop": ENABLE_REGION_CROP,\n` +
      `    "crop_lon_range": CROP_LON_RANGE,\n` +
      `    "crop_lat_range": CROP_LAT_RANGE,\n` +
      `    "crop_mode": CROP_MODE,\n` +
      `    "use_date_filename": USE_DATE_FILENAME,\n` +
      `    "date_format": DATE_FORMAT,\n` +
      `    "time_var": TIME_VAR,\n` +
      `    "max_files": MAX_FILES,\n` +
      `}\n` +
      `\n` +
      `config_path = os.path.join(TEMP_DIR, "convert_hr_config.json")\n` +
      `output_path = os.path.join(TEMP_DIR, "convert_hr_result.json")\n` +
      `with open(config_path, "w", encoding="utf-8") as f:\n` +
      `    json.dump(convert_config, f, ensure_ascii=False)\n` +
      `\n` +
      `result = subprocess.run(\n` +
      `    [PYTHON_PATH, os.path.join(SCRIPT_DIR, "convert_npy.py"),\n` +
      `     "--config", config_path, "--output", output_path],\n` +
      `    capture_output=True, text=True\n` +
      `)\n` +
      `if result.returncode == 0:\n` +
      `    with open(output_path, "r", encoding="utf-8") as f:\n` +
      `        convert_result = json.load(f)\n` +
      `    print(f"转换状态: {convert_result.get('status')}")\n` +
      `    print(f"输出目录: {convert_result.get('output_dir')}")\n` +
      `    if convert_result.get("warnings"):\n` +
      `        for w in convert_result["warnings"]:\n` +
      `            print(f"  警告: {w}")\n` +
      `else:\n` +
      `    print("Step C 失败:", result.stderr)\n` +
      `    raise RuntimeError("HR 数据转换失败")`
    )
  ]
}

function generateStepC2Cells(): NotebookCell[] {
  return [
    mdCell(
      `## Step C2: NC → NPY 转换 (LR)\n` +
      `\n` +
      `粗网格模式：将低分辨率 NC 数据转换为 NPY 格式。`
    ),
    codeCell(
      `lr_convert_config = {\n` +
      `    "nc_folder": LR_NC_FOLDER,\n` +
      `    "output_base": OUTPUT_BASE,\n` +
      `    "dyn_vars": DYN_VARS,\n` +
      `    "static_file": LR_STATIC_FILE,\n` +
      `    "dyn_file_pattern": LR_DYN_FILE_PATTERN,\n` +
      `    "stat_vars": STAT_VARS,\n` +
      `    "mask_vars": MASK_VARS,\n` +
      `    "lon_var": LON_VAR,\n` +
      `    "lat_var": LAT_VAR,\n` +
      `    "run_validation": True,\n` +
      `    "allow_nan": ALLOW_NAN,\n` +
      `    "mask_src_var": PRIMARY_MASK_VAR,\n` +
      `    "mask_derive_op": "identity",\n` +
      `    "train_ratio": TRAIN_RATIO,\n` +
      `    "valid_ratio": VALID_RATIO,\n` +
      `    "test_ratio": TEST_RATIO,\n` +
      `    "h_slice": H_SLICE,\n` +
      `    "w_slice": W_SLICE,\n` +
      `    "workers": WORKERS,\n` +
      `    "output_subdir": "lr",\n` +
      `    "use_date_filename": USE_DATE_FILENAME,\n` +
      `    "date_format": DATE_FORMAT,\n` +
      `    "time_var": TIME_VAR,\n` +
      `    "max_files": MAX_FILES,\n` +
      `}\n` +
      `\n` +
      `config_path = os.path.join(TEMP_DIR, "convert_lr_config.json")\n` +
      `output_path = os.path.join(TEMP_DIR, "convert_lr_result.json")\n` +
      `with open(config_path, "w", encoding="utf-8") as f:\n` +
      `    json.dump(lr_convert_config, f, ensure_ascii=False)\n` +
      `\n` +
      `result = subprocess.run(\n` +
      `    [PYTHON_PATH, os.path.join(SCRIPT_DIR, "convert_npy.py"),\n` +
      `     "--config", config_path, "--output", output_path],\n` +
      `    capture_output=True, text=True\n` +
      `)\n` +
      `if result.returncode == 0:\n` +
      `    with open(output_path, "r", encoding="utf-8") as f:\n` +
      `        lr_convert_result = json.load(f)\n` +
      `    print(f"LR 转换状态: {lr_convert_result.get('status')}")\n` +
      `    print(f"LR 输出目录: {lr_convert_result.get('output_dir')}")\n` +
      `    if lr_convert_result.get("warnings"):\n` +
      `        for w in lr_convert_result["warnings"]:\n` +
      `            print(f"  警告: {w}")\n` +
      `else:\n` +
      `    print("Step C2 失败:", result.stderr)\n` +
      `    raise RuntimeError("LR 数据转换失败")`
    )
  ]
}

function generateStepDCells(): NotebookCell[] {
  return [
    mdCell(
      `## Step D: 下采样 (HR → LR)\n` +
      `\n` +
      `对高分辨率数据进行下采样，生成低分辨率数据。`
    ),
    codeCell(
      `output_path = os.path.join(TEMP_DIR, "downsample_result.json")\n` +
      `\n` +
      `result = subprocess.run(\n` +
      `    [PYTHON_PATH, os.path.join(SCRIPT_DIR, "downsample.py"),\n` +
      `     "--dataset_root", OUTPUT_BASE,\n` +
      `     "--scale", str(SCALE),\n` +
      `     "--method", DOWNSAMPLE_METHOD,\n` +
      `     "--splits", "train", "valid", "test",\n` +
      `     "--include_static",\n` +
      `     "--output", output_path],\n` +
      `    capture_output=True, text=True\n` +
      `)\n` +
      `if result.returncode == 0:\n` +
      `    with open(output_path, "r", encoding="utf-8") as f:\n` +
      `        ds_result = json.load(f)\n` +
      `    total_files = 0\n` +
      `    for split_result in ds_result.get("splits", {}).values():\n` +
      `        total_files += len(split_result)\n` +
      `    total_files += len(ds_result.get("static_variables", []))\n` +
      `    print(f"下采样完成，共处理 {total_files} 个文件")\n` +
      `else:\n` +
      `    print("Step D 失败:", result.stderr)\n` +
      `    raise RuntimeError("下采样失败")`
    )
  ]
}

function generateStepECells(): NotebookCell[] {
  return [
    mdCell(
      `## Step E: 可视化检查\n` +
      `\n` +
      `生成 HR vs LR 空间对比图、统计分布图和全局汇总图。`
    ),
    codeCell(
      `result = subprocess.run(\n` +
      `    [PYTHON_PATH, os.path.join(SCRIPT_DIR, "visualize_check.py"),\n` +
      `     "--dataset_root", OUTPUT_BASE],\n` +
      `    capture_output=True, text=True\n` +
      `)\n` +
      `if result.returncode == 0:\n` +
      `    print("可视化完成")\n` +
      `    print(f"图片保存在: {os.path.join(OUTPUT_BASE, 'visualisation_data_process')}")\n` +
      `else:\n` +
      `    print("Step E 失败:", result.stderr)\n` +
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
    `├── train/hr/          # 训练集高分辨率数据\n` +
    `├── train/lr/          # 训练集低分辨率数据\n` +
    `├── valid/hr/, lr/     # 验证集\n` +
    `├── test/hr/, lr/      # 测试集\n` +
    `├── static_variables/  # 静态变量\n` +
    `├── visualisation_data_process/  # 可视化对比图\n` +
    `└── _ocean_preprocess_code/      # 预处理脚本\n` +
    `\`\`\``
  )
}

// ========================================
// 主导出函数
// ========================================

/**
 * 根据预处理参数生成所有 Notebook cells
 */
export function generatePreprocessCells(params: NotebookParams): NotebookCell[] {
  const cells: NotebookCell[] = []

  // 标题 + 环境设置
  cells.push(generateTitleCell(params))
  cells.push(generateSetupCell(params))

  // Step A: 数据检查
  cells.push(...generateStepACells())

  // Step B: 张量验证
  cells.push(...generateStepBCells())

  // Step C: HR 数据转换
  cells.push(...generateStepCCells())

  // Step C2: LR 数据转换（仅粗网格模式）
  if (params.isNumericalModelMode) {
    cells.push(...generateStepC2Cells())
  }

  // Step D: 下采样（仅下采样模式且未跳过）
  if (!params.isNumericalModelMode && !params.skipDownsample) {
    cells.push(...generateStepDCells())
  }

  // Step E: 可视化（未跳过）
  if (!params.skipVisualize) {
    cells.push(...generateStepECells())
  }

  // 完成说明
  cells.push(generateCompletionCell())

  return cells
}
