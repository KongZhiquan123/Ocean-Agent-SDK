/**
 * @file full.ts
 * @description 完整的海洋数据预处理流程工具
 *              串联 Step A -> B -> C -> (C2) -> D -> E 步骤
 *
 * @author leizheng
 * @contributors kongzhiquan
 * @date 2026-02-02
 * @version 3.5.1
 *
 * @changelog
 *   - 2026-02-25 Leizheng: v3.5.1 修复 Step C2 (数值模型 LR 转换) 缺少 scale 参数
 *     - Step C 正确传递了 scale，Step C2 遗漏，导致 LR 数据元数据缺失 scale 信息
 *   - 2026-02-07 kongzhiquan: v3.5.0 新增 max_files 参数，限制处理的最大 NC 文件数量
 *   - 2026-02-06 Leizheng: v3.4.1 有 mask 变量时自动允许 NaN
 *     - 陆地掩码区域的 NaN 是海洋数据的正常特征，不应阻断转换
 *   - 2026-02-05 kongzhiquan: v3.4.0 日期文件名功能
 *     - 新增 use_date_filename, date_format, time_var 参数
 *     - 支持从 NC 文件提取时间戳作为 NPY 文件名
 *     - 自动检测日期格式（日/小时/分钟级数据）
 *   - 2026-02-05 kongzhiquan: v3.3.0 合并重构与区域裁剪功能
 *     - 移除冗余的 try-catch 和 status 检查
 *     - 将验证逻辑下沉到子工具中
 *     - 简化 full.ts 为纯粹的流程编排器
 *     - 新增执行确认 Token 机制，防止 Agent 跳过确认阶段
 *     - 整合状态机架构，阶段判断逻辑移至 workflow-state.ts
 *     - 支持阶段2.5区域裁剪确认
 *     - 新增 enable_region_crop, crop_lon_range, crop_lat_range, crop_mode 参数
 *   - 2026-02-04 leizheng: v2.9.0 分阶段强制确认流程
 *     - 阶段1: awaiting_variable_selection - 研究变量选择
 *     - 阶段2: awaiting_static_selection - 静态/掩码变量选择
 *     - 阶段3: awaiting_parameters - 处理参数确认
 *     - 阶段4: awaiting_execution - 执行前最终确认
 *     - 每个阶段都必须等用户确认后才能继续
 *   - 2026-02-04 leizheng: v2.8.1 研究变量选择强制化
 *     - 新增 lr_nc_folder/lr_static_file/lr_dyn_file_pattern 参数
 *     - 新增 Step C2: 粗网格数据转换到 lr/ 目录
 *     - 粗网格模式下自动跳过下采样（Step D）
 *   - 2026-02-03 leizheng: v2.5.0 集成下采样和可视化
 *     - 新增 Step D: HR → LR 下采样
 *     - 新增 Step E: 可视化检查
 *     - 新增 downsample_method 参数
 *     - 新增 skip_downsample/skip_visualize 参数
 *   - 2026-02-03 leizheng: v2.4.0 裁剪与多线程
 *     - 新增 h_slice/w_slice 参数，在转换时直接裁剪
 *     - 新增 scale 参数，验证裁剪后尺寸能否被整除
 *     - 新增 workers 参数，多线程并行处理（默认 8，且不超过 CPU 核数）
 *   - 2026-02-03 leizheng: v2.3.2 修复确认流程被绕过问题
 *     - 添加 user_confirmed 参数，必须显式设置为 true 才能继续处理
 *     - 防止 AI Agent 自行决定跳过确认步骤
 *   - 2026-02-03 leizheng: v2.3.1 修复无掩码数据集分析失败
 *     - 掩码/静态变量改为可选，缺失时发出警告而非报错
 *     - 修复 primaryMaskVar 空数组时的错误
 *   - 2026-02-03 leizheng: v2.3.0 路径灵活处理
 *     - 支持 nc_files 参数明确指定文件列表
 *     - 支持单个文件路径自动转换为目录模式
 *     - 逐文件检测时间维度，识别静态文件混入
 *   - 2026-02-03 leizheng: v2.2.0 P0 安全修复
 *     - 移除硬编码默认值（lon_rho, lat_rho, mask_rho 等）
 *     - 添加路径验证（检测文件路径 vs 目录路径）
 *     - 掩码/静态变量必须从数据检测或用户指定
 *   - 2026-02-02 leizheng: v2.1.0 增加 P0 特性
 *     - allow_nan: NaN/Inf 采样检测
 *     - lon_range/lat_range: 坐标范围验证
 *   - 2026-02-02 leizheng: v2.0.0 适配新的 Python 脚本架构
 *     - 支持 dyn_file_pattern glob 模式
 *     - 集成后置验证结果
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import os from 'node:os'
import { oceanInspectDataTool } from './inspect'
import { oceanValidateTensorTool } from './validate'
import { oceanConvertNpyTool } from './convert'
import { oceanDownsampleTool } from './downsample'
import { oceanVisualizeTool } from './visualize'
import { PreprocessWorkflow, WorkflowState } from './workflow-state'

const DEFAULT_WORKERS = Math.max(1, Math.min(8, os.cpus().length || 1))

export const oceanPreprocessFullTool = defineTool({
  name: 'ocean_preprocess_full',
  description: `运行完整的超分辨率数据预处理流程 (A -> B -> C -> (C2) -> D -> E)

**支持两种模式**：

1. **下采样模式**（默认）：
   - 用户提供高分辨率 (HR) 数据
   - 自动下采样生成低分辨率 (LR) 数据
   - 需要指定 scale 和 downsample_method

2. **粗网格模式**（数值模型）：
   - 用户分别提供 HR 和 LR 数据（来自不同精度的数值模型）
   - HR 数据来自细网格模型运行
   - LR 数据来自粗网格模型运行
   - 通过 lr_nc_folder 参数启用此模式
   - 此模式下自动跳过下采样步骤

自动执行所有步骤：
1. Step A: 查看数据并定义变量
2. Step B: 进行张量约定验证
3. Step C: 转换 HR 数据为 NPY 格式（含后置验证 Rule 1/2/3）
4. Step C2: [粗网格模式] 转换 LR 数据为 NPY 格式
5. Step D: [下采样模式] HR → LR 下采样
6. Step E: 可视化检查（生成 HR vs LR 对比图）

**重要**：如果 Step A 检测到疑似变量但未提供 mask_vars/stat_vars，会返回 awaiting_confirmation 状态，此时需要用户确认后重新调用。

**注意**：研究变量、数据集划分比例必须由用户明确指定

**⚠️ 完成后必须生成报告**：
- 预处理完成后，Agent 必须调用 ocean_generate_report 工具生成报告
- 报告会包含一个分析占位符，Agent 必须读取报告并填写专业分析
- 分析应基于质量指标、验证结果等数据，提供具体的、有针对性的建议

**输出目录结构**：
- output_base/train/hr/*.npy - 训练集高分辨率数据
- output_base/train/lr/*.npy - 训练集低分辨率数据
- output_base/valid/hr/*.npy, valid/lr/*.npy - 验证集
- output_base/test/hr/*.npy, test/lr/*.npy - 测试集
- output_base/static_variables/*.npy - 静态变量
- output_base/visualisation_data_process/*.png - 可视化对比图
- output_base/preprocess_manifest.json - 数据溯源清单
- output_base/preprocessing_report.md - 预处理报告（需 Agent 填写分析）

**后置验证**：
- Rule 1: 输出完整性与形状约定
- Rule 2: 掩码不可变性检查
- Rule 3: 排序确定性检查

**返回**：各步骤结果、整体状态（awaiting_confirmation | pass | error）`,

  params: {
    nc_folder: {
      type: 'string',
      description: 'NC文件所在目录，也可以直接传入单个 .nc 文件路径'
    },
    nc_files: {
      type: 'array',
      items: { type: 'string' },
      description: '可选：明确指定要处理的文件列表（支持简单通配符如 "ocean_avg_*.nc"）',
      required: false
    },
    output_base: {
      type: 'string',
      description: '输出基础目录'
    },
    dyn_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '动态研究变量列表。【重要】如果不提供，工具会先分析数据并返回 awaiting_variable_selection 状态，要求用户选择。Agent 禁止猜测！',
      required: false
    },
    static_file: {
      type: 'string',
      description: '静态NC文件路径（可选）',
      required: false
    },
    dyn_file_pattern: {
      type: 'string',
      description: '动态文件的 glob 匹配模式，如 "*.nc" 或 "*avg*.nc"（当 nc_files 未指定时使用）',
      required: false,
      default: '*.nc'
    },
    mask_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '掩码变量列表（建议从 Step A 的 suspected_masks 中选择）',
      required: false
    },
    stat_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '静态变量列表（建议从 Step A 的 suspected_coordinates 中选择）',
      required: false
    },
    lon_var: {
      type: 'string',
      description: '经度参考变量名（必须由用户指定或从数据检测，禁止硬编码默认值）',
      required: false
    },
    lat_var: {
      type: 'string',
      description: '纬度参考变量名（必须由用户指定或从数据检测，禁止硬编码默认值）',
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
      description: '是否允许 NaN/Inf 值存在（默认 false，检测到会报错）',
      required: false,
      default: false
    },
    lon_range: {
      type: 'array',
      items: { type: 'number' },
      description: '经度有效范围 [min, max]，如 [-180, 180]',
      required: false
    },
    lat_range: {
      type: 'array',
      items: { type: 'number' },
      description: '纬度有效范围 [min, max]，如 [-90, 90]',
      required: false
    },
    user_confirmed: {
      type: 'boolean',
      description: '【必须】用户确认标志。必须在展示 Step A 分析结果并获得用户明确确认后，才能设置为 true。禁止自动设置！',
      required: false,
      default: false
    },
    train_ratio: {
      type: 'number',
      description: '【必须由用户指定】训练集比例（按时间顺序取前 N%），如 0.7。Agent 禁止自动设置！',
      required: false
    },
    valid_ratio: {
      type: 'number',
      description: '【必须由用户指定】验证集比例（按时间顺序取中间 N%），如 0.15。Agent 禁止自动设置！',
      required: false
    },
    test_ratio: {
      type: 'number',
      description: '【必须由用户指定】测试集比例（按时间顺序取最后 N%），如 0.15。Agent 禁止自动设置！',
      required: false
    },
    h_slice: {
      type: 'string',
      description: '【必须由用户指定】H 方向裁剪切片，如 "0:680"。确保裁剪后尺寸能被 scale 整除',
      required: false
    },
    w_slice: {
      type: 'string',
      description: '【必须由用户指定】W 方向裁剪切片，如 "0:1440"。确保裁剪后尺寸能被 scale 整除',
      required: false
    },
    scale: {
      type: 'number',
      description: '【必须由用户指定】下采样倍数（用于验证裁剪后尺寸能否被整除）',
      required: false
    },
    workers: {
      type: 'number',
      description: '并行线程数（默认 8，且不超过 CPU 核数）',
      required: false,
      default: DEFAULT_WORKERS
    },
    downsample_method: {
      type: 'string',
      description: '【必须由用户指定】下采样插值方法：area（推荐）、cubic、nearest、linear、lanczos',
      required: false
    },
    skip_downsample: {
      type: 'boolean',
      description: '是否跳过下采样步骤（默认 false，即执行下采样）',
      required: false,
      default: false
    },
    skip_visualize: {
      type: 'boolean',
      description: '是否跳过可视化步骤（默认 false，即生成可视化）',
      required: false,
      default: false
    },
    // ========== 粗网格模式参数 ==========
    lr_nc_folder: {
      type: 'string',
      description: '【粗网格模式】低分辨率 NC 文件所在目录。提供此参数将启用粗网格模式，自动跳过下采样步骤。',
      required: false
    },
    lr_static_file: {
      type: 'string',
      description: '【粗网格模式】低分辨率静态 NC 文件路径（可选）',
      required: false
    },
    lr_dyn_file_pattern: {
      type: 'string',
      description: '【粗网格模式】低分辨率动态文件的 glob 匹配模式（默认与 dyn_file_pattern 相同）',
      required: false
    },
    // ========== 区域裁剪参数 ==========
    enable_region_crop: {
      type: 'boolean',
      description: '是否启用区域裁剪（先裁剪到特定经纬度区域再进行下采样）',
      required: false,
      default: false
    },
    crop_lon_range: {
      type: 'array',
      items: { type: 'number' },
      description: '区域裁剪的经度范围 [min, max]，如 [100, 120]',
      required: false
    },
    crop_lat_range: {
      type: 'array',
      items: { type: 'number' },
      description: '区域裁剪的纬度范围 [min, max]，如 [20, 40]',
      required: false
    },
    crop_mode: {
      type: 'string',
      description: `区域裁剪模式:
- "one_step": 一步到位，直接计算能被 scale 整除的裁剪区域，不保存 raw
- "two_step": 两步裁剪，先保存裁剪后的原始数据到 raw/，再做尺寸调整保存到 hr/`,
      required: false,
      default: 'two_step'
    },
    // ========== 日期文件名参数 ==========
    use_date_filename: {
      type: 'boolean',
      description: '是否使用日期作为文件名（如 20200101.npy 而非 000000.npy），默认开启',
      required: false,
      default: true
    },
    date_format: {
      type: 'string',
      description: '日期格式: "auto"（自动检测）, "YYYYMMDD", "YYYYMMDDHH", "YYYYMMDDHHmm", "YYYY-MM-DD"',
      required: false,
      default: 'auto'
    },
    time_var: {
      type: 'string',
      description: '时间变量名（默认自动检测 time/ocean_time 等）',
      required: false
    },
    // ========== 文件数限制参数 ==========
    max_files: {
      type: 'number',
      description: '可选：限制处理的最大 NC 文件数量（按排序后取前 N 个）',
      required: false
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
      lon_range,
      lat_range,
      user_confirmed = false,
      confirmation_token,
      train_ratio,
      valid_ratio,
      test_ratio,
      h_slice,
      w_slice,
      scale,
      workers = DEFAULT_WORKERS,
      downsample_method,
      skip_downsample = false,
      skip_visualize = false,
      lr_nc_folder,
      lr_static_file,
      lr_dyn_file_pattern,
      // 区域裁剪参数
      enable_region_crop = false,
      crop_lon_range,
      crop_lat_range,
      crop_mode = 'two_step',
      // 日期文件名参数
      use_date_filename = true,
      date_format = 'auto',
      time_var,
      // 文件数限制参数
      max_files
    } = args

    // 检测是否为粗网格模式（数值模型模式）
    const isNumericalModelMode = !!lr_nc_folder

    // 智能路径处理：支持目录或单个文件
    let actualNcFolder = nc_folder.trim()
    let actualNcFiles = nc_files
    let actualFilePattern = dyn_file_pattern

    // 检测是否为单个 NC 文件路径
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
      step_c2: null as any,
      step_d: null as any,
      step_e: null as any,
      overall_status: 'pending' as string,
      message: '',
      validation_summary: null as any,
      mode: isNumericalModelMode ? 'numerical_model' : 'downsample'
    }

    // ========== Step A: 数据检查 ==========
    const stepAResult = await oceanInspectDataTool.exec({
      nc_folder: actualNcFolder,
      nc_files: actualNcFiles,
      static_file,
      dyn_file_pattern: actualFilePattern
    }, ctx)

    result.step_a = stepAResult
    const dynCandidates = stepAResult.dynamic_vars_candidates || []

    // ========== 状态机判断 ==========
    const workflow = new PreprocessWorkflow({
      nc_folder: actualNcFolder,
      output_base,
      dyn_vars,
      stat_vars,
      mask_vars,
      enable_region_crop,
      crop_lon_range: crop_lon_range as [number, number] | undefined,
      crop_lat_range: crop_lat_range as [number, number] | undefined,
      crop_mode: crop_mode as 'one_step' | 'two_step' | undefined,
      scale,
      downsample_method,
      train_ratio,
      valid_ratio,
      test_ratio,
      h_slice,
      w_slice,
      lr_nc_folder,
      user_confirmed,
      confirmation_token
    })

    const stateCheck = workflow.determineCurrentState()

    // 如果状态机判断未通过，返回相应的阶段提示
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
    const missingVars = dyn_vars!.filter((v: string) => !dynCandidates.includes(v))
    if (missingVars.length > 0) {
      throw new Error(`研究变量不在动态变量候选列表中：${missingVars.join(', ')}
可用的动态变量候选：${dynCandidates.join(', ')}`)
    }
    const totalRatio = train_ratio! + valid_ratio! + test_ratio!
    if (totalRatio < 0.99 || totalRatio > 1.01) {
      throw new Error(`数据集划分比例之和必须等于 1.0，目前为 ${totalRatio}`)
    }

    // ========== 准备变量配置 ==========
    const detectedMaskVars = stepAResult.suspected_masks || []
    const finalMaskVars = mask_vars || (detectedMaskVars.length > 0 ? detectedMaskVars : [])
    const detectedCoordVars = stepAResult.suspected_coordinates || []
    const finalStaticVars = stat_vars || (detectedCoordVars.length > 0
      ? [...detectedCoordVars, ...detectedMaskVars]
      : [])

    // 主掩码变量选择
    let primaryMaskVar: string | undefined
    if (finalMaskVars.length === 1) {
      primaryMaskVar = finalMaskVars[0]
    } else if (finalMaskVars.length > 1) {
      const rhoMask = finalMaskVars.find((m: string) => m.includes('rho'))
      primaryMaskVar = rhoMask || finalMaskVars[0]
    }

    // 经纬度变量
    const detectedLonVar = finalStaticVars.find((v: string) =>
      v.toLowerCase().includes('lon') && !v.toLowerCase().includes('mask')
    )
    const detectedLatVar = finalStaticVars.find((v: string) =>
      v.toLowerCase().includes('lat') && !v.toLowerCase().includes('mask')
    )
    const finalLonVar = lon_var || detectedLonVar
    const finalLatVar = lat_var || detectedLatVar

    // ========== Step B: 张量验证 ==========
    const stepBResult = await oceanValidateTensorTool.exec({
      research_vars: dyn_vars,
      mask_vars: finalMaskVars
    }, ctx)

    result.step_b = stepBResult

    // ========== Step C: HR 数据转换 ==========
    // 有掩码变量时自动允许 NaN（陆地区域的 NaN 是海洋数据的正常特征）
    const effectiveAllowNan = allow_nan || (finalMaskVars && finalMaskVars.length > 0)

    const normalizedWorkers = Math.max(1, Math.min(workers, os.cpus().length || 1))

    const stepCResult = await oceanConvertNpyTool.exec({
      nc_folder: actualNcFolder,
      nc_files: actualNcFiles,
      output_base,
      dyn_vars,
      static_file,
      dyn_file_pattern: actualFilePattern,
      stat_vars: finalStaticVars,
      mask_vars: finalMaskVars,
      lon_var: finalLonVar,
      lat_var: finalLatVar,
      run_validation,
      allow_nan: effectiveAllowNan,
      lon_range,
      lat_range,
      mask_src_var: primaryMaskVar,
      mask_derive_op: 'identity',
      heuristic_check_var: dyn_vars?.[0],
      land_threshold_abs: 1e-12,
      heuristic_sample_size: 2000,
      require_sorted: true,
      train_ratio,
      valid_ratio,
      test_ratio,
      h_slice,
      w_slice,
      scale,
      workers: normalizedWorkers,
      // 区域裁剪参数
      enable_region_crop,
      crop_lon_range,
      crop_lat_range,
      crop_mode,
      // 日期文件名参数
      use_date_filename,
      date_format,
      time_var,
      // 文件数限制
      max_files
    }, ctx)

    result.step_c = stepCResult

    // ========== Step C2: 粗网格模式下转换 LR 数据 ==========
    if (isNumericalModelMode) {
      let actualLrNcFolder = lr_nc_folder!.trim()
      let actualLrFilePattern = lr_dyn_file_pattern || actualFilePattern

      if (actualLrNcFolder.endsWith('.nc') || actualLrNcFolder.endsWith('.NC')) {
        const filePath = actualLrNcFolder
        const lastSlash = filePath.lastIndexOf('/')
        if (lastSlash === -1) {
          actualLrNcFolder = '.'
          actualLrFilePattern = filePath
        } else {
          actualLrNcFolder = filePath.substring(0, lastSlash)
          actualLrFilePattern = filePath.substring(lastSlash + 1)
        }
      }

      const stepC2Result = await oceanConvertNpyTool.exec({
        nc_folder: actualLrNcFolder,
        output_base,
        dyn_vars,
        static_file: lr_static_file || static_file,
        dyn_file_pattern: actualLrFilePattern,
        stat_vars: finalStaticVars,
        mask_vars: finalMaskVars,
        lon_var: finalLonVar,
        lat_var: finalLatVar,
        run_validation,
        allow_nan: effectiveAllowNan,
        lon_range,
        lat_range,
        mask_src_var: primaryMaskVar,
        mask_derive_op: 'identity',
        heuristic_check_var: dyn_vars?.[0],
        land_threshold_abs: 1e-12,
        heuristic_sample_size: 2000,
        require_sorted: true,
        train_ratio,
        valid_ratio,
        test_ratio,
        h_slice,
        w_slice,
        scale,
        workers: normalizedWorkers,
        output_subdir: 'lr',
        // 日期文件名参数
        use_date_filename,
        date_format,
        time_var,
        // 文件数限制
        max_files
      }, ctx)

      result.step_c2 = stepC2Result
    }

    // ========== Step D: 下采样 ==========
    if (isNumericalModelMode) {
      result.step_d = { status: 'skipped', reason: '粗网格模式（数值模型）下自动跳过下采样' }
    } else if (!skip_downsample) {
      const stepDResult = await oceanDownsampleTool.exec({
        dataset_root: output_base,
        scale: scale,
        method: downsample_method,
        splits: ['train', 'valid', 'test'],
        include_static: true
      }, ctx)

      result.step_d = stepDResult
    } else {
      result.step_d = { status: 'skipped', reason: 'skip_downsample=true' }
    }

    // ========== Step E: 可视化 ==========
    if (!skip_visualize) {
      const stepEResult = await oceanVisualizeTool.exec({
        dataset_root: output_base,
        splits: ['train', 'valid', 'test']
      }, ctx)
      result.step_e = stepEResult
    } else {
      result.step_e = { status: 'skipped', reason: 'skip_visualize=true' }
    }

    // ========== 最终状态 ==========
    result.overall_status = 'pass'
    result.message = '预处理完成，所有检查通过，请调用ocean_metrics工具生成质量指标，随后调用ocean_generate_report生成预处理报告。'
    result.validation_summary = stepCResult.post_validation

    return result
  }
})
