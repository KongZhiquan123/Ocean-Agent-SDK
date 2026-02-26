/**
 * @file full.ts
 * @description 完整的海洋预测数据预处理流程工具
 *              串联 Step A（数据检查）→ Step B（NC→NPY 转换）→ Step C（可视化）
 *              无下采样步骤，数据按时间严格排序
 *
 * @author Leizheng
 * @date 2026-02-25
 * @version 1.1.0
 *
 * @changelog
 *   - 2026-02-26 Leizheng: v1.1.0 将 static_file 作为 grid_file 传给 forecast_preprocess.py
 *     - 支持 ROMS 等模式的独立网格文件（坐标/掩码变量不在数据文件中的情况）
 *   - 2026-02-26 Leizheng: v1.0.1 修复 Step A 调用缺少 output_base 参数导致 inspect 工具崩溃
 *   - 2026-02-25 Leizheng: v1.0.0 初始版本
 *     - 复用 ocean_inspect_data 工具（Step A）
 *     - 调用 forecast_preprocess.py 执行转换（Step B）
 *     - 4 阶段强制确认状态机（无 scale/downsample/lr 参数）
 *     - 输出目录结构: {split}/{var_name}/{date}.npy（无 hr/lr 层级）
 *     - 严格时间排序，生成 time_index.json
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import os from 'node:os'
import path from 'node:path'
import { findFirstPythonPath } from '@/utils/python-manager'
import { oceanInspectDataTool } from '../ocean-SR-data-preprocess/inspect'
import { oceanForecastVisualizeTool } from './visualize'
import { ForecastWorkflow, WorkflowState } from './workflow-state'

const DEFAULT_WORKERS = Math.max(1, Math.min(8, os.cpus().length || 1))

function shellEscapeDouble(str: string): string {
  return str.replace(/[\\"$`!]/g, '\\$&')
}

export const oceanForecastPreprocessFullTool = defineTool({
  name: 'ocean_forecast_preprocess_full',
  description: `运行完整的海洋预测数据预处理流程 (A → B → C)

将 NC 格式的海洋数值模式输出转换为 NPY 格式，用于深度学习预测模型训练。

**与超分辨率预处理的区别**：
- 无下采样步骤（无 hr/lr 目录层级）
- 输出直接存储在 \`split/var_name/\` 目录下
- 数据按 NC 文件内时间变量**严格升序**排列
- 生成 \`time_index.json\` 记录完整时间戳溯源

**自动执行步骤**：
1. Step A: 数据检查（检测变量类型，等待用户确认）
2. Step B: NC → NPY 转换（时间排序 + 切分 + 保存）
3. Step C: 可视化检查（可选，生成样本帧和时序图）

**确认流程（4 阶段，防止 Agent 跳步）**：
- 阶段 1: 选择预测变量（dyn_vars）
- 阶段 2: 确认静态/掩码变量（stat_vars, mask_vars）
- 阶段 3: 确认处理参数（train/valid/test 比例，裁剪）
- 阶段 4: 最终执行确认（user_confirmed + confirmation_token）

**输出目录结构**：
- output_base/train/{var_name}/*.npy - 训练集
- output_base/valid/{var_name}/*.npy - 验证集
- output_base/test/{var_name}/*.npy  - 测试集
- output_base/static_variables/*.npy - 静态变量
- output_base/time_index.json        - 时间戳溯源
- output_base/var_names.json         - 变量配置
- output_base/preprocess_manifest.json
- output_base/visualisation_forecast/ - 可视化图片（可选）
- output_base/preprocessing_report.md - 报告（需 Agent 生成）

**完成后**：调用 ocean_forecast_generate_report 生成报告`,

  params: {
    nc_folder: {
      type: 'string',
      description: 'NC 文件所在目录，也可以直接传入单个 .nc 文件路径'
    },
    nc_files: {
      type: 'array',
      items: { type: 'string' },
      description: '可选：明确指定要处理的文件列表（支持通配符如 "ocean_avg_*.nc"）',
      required: false
    },
    output_base: {
      type: 'string',
      description: '输出基础目录'
    },
    dyn_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '动态研究变量列表。未提供时工具先分析数据并返回 awaiting_variable_selection 状态。Agent 禁止猜测！',
      required: false
    },
    static_file: {
      type: 'string',
      description: '静态 NC 文件路径（可选）',
      required: false
    },
    dyn_file_pattern: {
      type: 'string',
      description: '动态文件的 glob 匹配模式，如 "*.nc" 或 "*avg*.nc"',
      required: false,
      default: '*.nc'
    },
    mask_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '掩码变量列表（从 Step A 的 suspected_masks 中选择）',
      required: false
    },
    stat_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '静态变量列表（从 Step A 的 suspected_coordinates 中选择）',
      required: false
    },
    lon_var: {
      type: 'string',
      description: '经度变量名（可选，用于可视化）',
      required: false
    },
    lat_var: {
      type: 'string',
      description: '纬度变量名（可选，用于可视化）',
      required: false
    },
    run_validation: {
      type: 'boolean',
      description: '是否执行后置验证 (Rule 1/2/3)',
      required: false,
      default: true
    },
    allow_nan: {
      type: 'boolean',
      description: '是否允许 NaN/Inf 值存在（有掩码变量时自动允许）',
      required: false,
      default: false
    },
    user_confirmed: {
      type: 'boolean',
      description: '【必须】用户确认标志。必须在展示 Step A 分析结果并获得用户明确确认后才能设置为 true。禁止自动设置！',
      required: false,
      default: false
    },
    confirmation_token: {
      type: 'string',
      description: '执行确认 Token（从 awaiting_execution 阶段获取，与 user_confirmed=true 配合使用）',
      required: false
    },
    train_ratio: {
      type: 'number',
      description: '【必须由用户指定】训练集比例（按时间顺序取前 N%），如 0.7',
      required: false
    },
    valid_ratio: {
      type: 'number',
      description: '【必须由用户指定】验证集比例，如 0.15',
      required: false
    },
    test_ratio: {
      type: 'number',
      description: '【必须由用户指定】测试集比例，如 0.15',
      required: false
    },
    h_slice: {
      type: 'string',
      description: 'H 方向裁剪切片，如 "0:512"（可选）',
      required: false
    },
    w_slice: {
      type: 'string',
      description: 'W 方向裁剪切片，如 "0:1024"（可选）',
      required: false
    },
    chunk_size: {
      type: 'number',
      description: '批量处理文件数（控制内存使用，默认 200）',
      required: false,
      default: 200
    },
    use_date_filename: {
      type: 'boolean',
      description: '是否使用日期作为文件名（如 20200101.npy 而非 000000.npy），默认开启',
      required: false,
      default: true
    },
    date_format: {
      type: 'string',
      description: '日期格式: "auto"（自动检测）, "YYYYMMDD", "YYYYMMDDHH", "YYYYMMDDHHmm"',
      required: false,
      default: 'auto'
    },
    time_var: {
      type: 'string',
      description: '时间变量名（默认自动检测 time/ocean_time 等）',
      required: false
    },
    max_files: {
      type: 'number',
      description: '可选：限制处理的最大 NC 文件数量（按排序后取前 N 个）',
      required: false
    },
    skip_visualize: {
      type: 'boolean',
      description: '是否跳过可视化步骤（默认 false）',
      required: false,
      default: false
    }
  },

  attributes: {
    readonly: false,
    noEffect: false
  },

  async exec(args, ctx) {
    const {
      nc_folder,
      nc_files,
      output_base,
      dyn_vars,
      static_file,
      dyn_file_pattern = '*.nc',
      mask_vars,
      stat_vars,
      lon_var,
      lat_var,
      run_validation = true,
      allow_nan = false,
      user_confirmed = false,
      confirmation_token,
      train_ratio,
      valid_ratio,
      test_ratio,
      h_slice,
      w_slice,
      chunk_size = 200,
      use_date_filename = true,
      date_format = 'auto',
      time_var,
      max_files,
      skip_visualize = false
    } = args

    // 智能路径处理：支持目录或单个文件
    let actualNcFolder = nc_folder.trim()
    let actualNcFiles = nc_files as string[] | undefined
    let actualFilePattern = dyn_file_pattern

    if (actualNcFolder.endsWith('.nc') || actualNcFolder.endsWith('.NC')) {
      const filePath = actualNcFolder
      const lastSlash = filePath.lastIndexOf('/')
      if (lastSlash === -1) {
        actualNcFolder = '.'
        actualNcFiles = [filePath]
      } else {
        actualNcFolder = filePath.substring(0, lastSlash)
        actualNcFiles = [filePath.substring(lastSlash + 1)]
      }
    }

    const result = {
      step_a: null as any,
      step_b: null as any,
      step_c: null as any,
      overall_status: 'pending' as string,
      message: '',
      mode: 'forecast'
    }

    // ========== Step A: 数据检查 ==========
    const stepAResult = await oceanInspectDataTool.exec({
      nc_folder: actualNcFolder,
      nc_files: actualNcFiles,
      static_file,
      dyn_file_pattern: actualFilePattern,
      output_base
    }, ctx)

    result.step_a = stepAResult
    const dynCandidates = stepAResult.dynamic_vars_candidates || []

    // ========== 状态机判断 ==========
    const workflow = new ForecastWorkflow({
      nc_folder: actualNcFolder,
      output_base,
      dyn_vars: dyn_vars as string[] | undefined,
      stat_vars: stat_vars as string[] | undefined,
      mask_vars: mask_vars as string[] | undefined,
      train_ratio: train_ratio as number | undefined,
      valid_ratio: valid_ratio as number | undefined,
      test_ratio: test_ratio as number | undefined,
      h_slice: h_slice as string | undefined,
      w_slice: w_slice as string | undefined,
      user_confirmed: user_confirmed as boolean,
      confirmation_token: confirmation_token as string | undefined
    })

    const stateCheck = workflow.determineCurrentState()

    if (stateCheck.currentState !== WorkflowState.PASS) {
      const prompt = workflow.getStagePrompt(stepAResult)
      result.step_a = {
        status: stepAResult.status,
        nc_folder: stepAResult.nc_folder,
        file_count: stepAResult.file_count,
        dynamic_vars_candidates: dynCandidates,
        suspected_masks: stepAResult.suspected_masks,
        suspected_coordinates: stepAResult.suspected_coordinates
      }
      result.overall_status = prompt.status
      result.message = prompt.message
      return result
    }

    // ========== 状态机通过，验证研究变量 ==========
    const missingVars = (dyn_vars as string[]).filter((v: string) => !dynCandidates.includes(v))
    if (missingVars.length > 0) {
      throw new Error(`研究变量不在动态变量候选列表中：${missingVars.join(', ')}\n可用的动态变量候选：${dynCandidates.join(', ')}`)
    }

    const totalRatio = (train_ratio as number) + (valid_ratio as number) + (test_ratio as number)
    if (totalRatio < 0.99 || totalRatio > 1.01) {
      throw new Error(`数据集划分比例之和必须等于 1.0，目前为 ${totalRatio}`)
    }

    // ========== 准备变量配置 ==========
    const detectedMaskVars = stepAResult.suspected_masks || []
    const finalMaskVars = mask_vars !== undefined
      ? (mask_vars as string[])
      : (detectedMaskVars.length > 0 ? detectedMaskVars : [])
    const detectedCoordVars = stepAResult.suspected_coordinates || []
    const finalStaticVars = stat_vars !== undefined
      ? (stat_vars as string[])
      : (detectedCoordVars.length > 0 ? [...detectedCoordVars, ...detectedMaskVars] : [])

    // 经纬度变量
    const detectedLonVar = finalStaticVars.find((v: string) =>
      v.toLowerCase().includes('lon') && !v.toLowerCase().includes('mask')
    )
    const detectedLatVar = finalStaticVars.find((v: string) =>
      v.toLowerCase().includes('lat') && !v.toLowerCase().includes('mask')
    )
    const finalLonVar = (lon_var as string | undefined) || detectedLonVar
    const finalLatVar = (lat_var as string | undefined) || detectedLatVar

    // 有掩码变量时自动允许 NaN
    const effectiveAllowNan = allow_nan || (finalMaskVars && finalMaskVars.length > 0)

    // ========== Step B: 执行 forecast_preprocess.py ==========
    const pythonPath = findFirstPythonPath()
    if (!pythonPath) {
      throw new Error('未找到可用的Python解释器，请安装Python或配置PYTHON/PYENV')
    }

    const pythonCmd = `"${shellEscapeDouble(pythonPath)}"`
    const tempDir = path.resolve(ctx.sandbox.workDir, 'ocean_forecast_temp')
    const configPath = path.join(tempDir, 'forecast_config.json')
    const outputPath = path.join(tempDir, 'forecast_result.json')
    const scriptPath = path.resolve(process.cwd(), 'scripts/ocean-forecast-data-preprocess/forecast_preprocess.py')

    // 构建 Python 配置
    const forecastConfig: Record<string, any> = {
      nc_folder: actualNcFolder,
      output_base,
      dyn_vars,
      stat_vars: finalStaticVars,
      mask_vars: finalMaskVars,
      lon_var: finalLonVar || null,
      lat_var: finalLatVar || null,
      train_ratio,
      valid_ratio,
      test_ratio,
      h_slice: h_slice || null,
      w_slice: w_slice || null,
      dyn_file_pattern: actualFilePattern,
      chunk_size,
      use_date_filename,
      date_format,
      time_var: time_var || null,
      max_files: max_files || null,
      run_validation,
      allow_nan: effectiveAllowNan,
      grid_file: static_file || null
    }

    if (actualNcFiles && actualNcFiles.length > 0) {
      forecastConfig.nc_files = actualNcFiles
    }

    await ctx.sandbox.exec(`mkdir -p "${shellEscapeDouble(tempDir)}"`)
    await ctx.sandbox.fs.write(configPath, JSON.stringify(forecastConfig, null, 2))

    const pyResult = await ctx.sandbox.exec(
      `${pythonCmd} "${shellEscapeDouble(scriptPath)}" --config "${shellEscapeDouble(configPath)}" --output "${shellEscapeDouble(outputPath)}"`,
      { timeoutMs: 3600000 }  // 1 小时超时
    )

    if (pyResult.code !== 0) {
      throw new Error(`forecast_preprocess.py 执行失败:\n${pyResult.stderr}`)
    }

    const stepBJson = await ctx.sandbox.fs.read(outputPath)
    const stepBResult = JSON.parse(stepBJson)
    result.step_b = stepBResult

    if (stepBResult.status === 'error') {
      result.overall_status = 'error'
      result.message = `Step B 失败: ${stepBResult.errors?.join('; ')}`
      return result
    }

    // ========== Step C: 可视化（可选） ==========
    if (!skip_visualize) {
      const stepCResult = await oceanForecastVisualizeTool.exec({
        dataset_root: output_base,
        splits: ['train', 'valid', 'test']
      }, ctx)
      result.step_c = stepCResult
    } else {
      result.step_c = { status: 'skipped', reason: 'skip_visualize=true' }
    }

    // ========== 最终状态 ==========
    result.overall_status = 'pass'
    result.message = `预处理完成！${stepBResult.message || ''} 请调用 ocean_forecast_generate_report 工具生成预处理报告。`

    return result
  }
})
