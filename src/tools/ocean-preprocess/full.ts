/**
 * @file full.ts
 * @description 完整的海洋数据预处理流程工具
 *              串联 Step A -> B -> C -> D -> E 五个步骤
 *
 * @author leizheng
 * @contributers kongzhiquan
 * @date 2026-02-02
 * @version 2.5.0
 *
 * @changelog
 *   - 2026-02-03 leizheng: v2.5.0 集成下采样和可视化
 *     - 新增 Step D: HR → LR 下采样
 *     - 新增 Step E: 可视化检查
 *     - 新增 downsample_method 参数
 *     - 新增 skip_downsample/skip_visualize 参数
 *   - 2026-02-03 leizheng: v2.4.0 裁剪与多线程
 *     - 新增 h_slice/w_slice 参数，在转换时直接裁剪
 *     - 新增 scale 参数，验证裁剪后尺寸能否被整除
 *     - 新增 workers 参数，多线程并行处理（默认 32）
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

import path from 'path'
import { defineTool } from '@shareai-lab/kode-sdk'
import { oceanInspectDataTool } from './inspect'
import { oceanValidateTensorTool } from './validate'
import { oceanConvertNpyTool } from './convert'
import { oceanDownsampleTool } from './downsample'
import { oceanVisualizeTool } from './visualize'

export const oceanPreprocessFullTool = defineTool({
  name: 'ocean_preprocess_full',
  description: `运行完整的超分辨率数据预处理流程 (A -> B -> C -> D -> E)

自动执行所有五个步骤：
1. Step A: 查看数据并定义变量
2. Step B: 进行张量约定验证
3. Step C: 转换为NPY格式存储（含后置验证 Rule 1/2/3）
4. Step D: HR → LR 下采样
5. Step E: 可视化检查（生成 HR vs LR 对比图）

**重要**：如果 Step A 检测到疑似变量但未提供 mask_vars/stat_vars，会返回 awaiting_confirmation 状态，此时需要用户确认后重新调用。

**注意**：研究变量、数据集划分比例、下采样倍数必须由用户明确指定

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
      description: 'NC文件所在目录'
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
      description: '动态研究变量列表（必须由用户指定）'
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
      // P0 修复：移除硬编码默认值 'lon_rho'
    },
    lat_var: {
      type: 'string',
      description: '纬度参考变量名（必须由用户指定或从数据检测，禁止硬编码默认值）',
      required: false
      // P0 修复：移除硬编码默认值 'lat_rho'
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
      // 注意：无默认值，必须由用户提供
    },
    valid_ratio: {
      type: 'number',
      description: '【必须由用户指定】验证集比例（按时间顺序取中间 N%），如 0.15。Agent 禁止自动设置！',
      required: false
      // 注意：无默认值，必须由用户提供
    },
    test_ratio: {
      type: 'number',
      description: '【必须由用户指定】测试集比例（按时间顺序取最后 N%），如 0.15。Agent 禁止自动设置！',
      required: false
      // 注意：无默认值，必须由用户提供
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
      description: '并行线程数（默认 32）',
      required: false,
      default: 32
    },
    downsample_method: {
      type: 'string',
      description: '【必须由用户指定】下采样插值方法：area（推荐）、cubic、nearest、linear、lanczos',
      required: false
      // 注意：无默认值，必须由用户提供
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
      train_ratio,   // 无默认值，必须由用户提供
      valid_ratio,   // 无默认值，必须由用户提供
      test_ratio,    // 无默认值，必须由用户提供
      h_slice,       // 裁剪参数
      w_slice,       // 裁剪参数
      scale,         // 下采样倍数
      workers = 32,  // 并行线程数
      downsample_method,   // 下采样插值方法，无默认值
      skip_downsample = false,     // 是否跳过下采样
      skip_visualize = false       // 是否跳过可视化
    } = args

    // 智能路径处理：支持目录或单个文件
    let actualNcFolder = nc_folder.trim()
    let actualNcFiles = nc_files
    let actualFilePattern = dyn_file_pattern

    // 检测是否为单个 NC 文件路径
    if (actualNcFolder.endsWith('.nc') || actualNcFolder.endsWith('.NC')) {
      // 用户提供的是单个文件，自动转换为目录 + nc_files 模式
      const filePath = actualNcFolder
      const lastSlash = filePath.lastIndexOf('/')
      if (lastSlash === -1) {
        actualNcFolder = '.'
        actualNcFiles = [filePath]
      } else {
        actualNcFolder = filePath.substring(0, lastSlash)
        actualNcFiles = [filePath.substring(lastSlash + 1)]
      }

      ctx.emit('info', {
        type: 'single_file_mode',
        message: `检测到单个文件路径，自动转换为目录模式`,
        original_path: filePath,
        nc_folder: actualNcFolder,
        nc_files: actualNcFiles
      })
    }

    ctx.emit('pipeline_started', {
      nc_folder: actualNcFolder,
      nc_files: actualNcFiles,
      output_base,
      dyn_vars
    })

    const result = {
      step_a: null as any,
      step_b: null as any,
      step_c: null as any,
      step_d: null as any,  // 下采样结果
      step_e: null as any,  // 可视化结果
      overall_status: 'pending' as string,
      message: '',
      validation_summary: null as any
    }

    // Step A
    ctx.emit('step_started', { step: 'A', description: '查看数据并定义变量' })

    const stepAResult = await oceanInspectDataTool.exec({
      nc_folder: actualNcFolder,
      nc_files: actualNcFiles,
      static_file,
      dyn_file_pattern: actualFilePattern
    }, ctx)

    result.step_a = stepAResult

    if (stepAResult.status === 'error') {
      result.overall_status = 'error'
      result.message = 'Step A 失败'
      ctx.emit('pipeline_failed', { step: 'A', result })
      return result
    }

    // 检查是否找到动态数据文件
    if (stepAResult.file_count === 0) {
      result.overall_status = 'error'
      result.message = `未找到匹配的动态数据文件！
- 搜索目录: ${actualNcFolder}
- 文件匹配模式: "${actualFilePattern}"
请检查：
1. nc_folder 路径是否正确
2. dyn_file_pattern 是否匹配你的文件名`
      ctx.emit('pipeline_failed', { step: 'A', error: '未找到动态数据文件' })
      return result
    }

    // 检查是否找到任何动态变量候选
    const dynCandidates = stepAResult.dynamic_vars_candidates || []
    if (dynCandidates.length === 0) {
      result.overall_status = 'error'
      result.message = `数据文件中没有找到任何动态变量（带时间维度的变量）！

这通常意味着您可能提供了静态文件而非动态数据文件。

【文件信息】
- 搜索目录: ${nc_folder}
- 找到文件数: ${stepAResult.file_count}
- 文件列表: ${(stepAResult.file_list || []).slice(0, 3).join(', ')}${(stepAResult.file_list || []).length > 3 ? '...' : ''}

【检测到的变量】（都没有时间维度）
${Object.keys(stepAResult.variables || {}).slice(0, 10).join(', ')}${Object.keys(stepAResult.variables || {}).length > 10 ? '...' : ''}

请检查：
1. 您是否将静态文件路径填到了动态数据目录？
2. 动态数据文件是否确实包含时间维度？
3. 时间维度的名称是否为标准名称（time, ocean_time, t 等）？`

      ctx.emit('pipeline_failed', { step: 'A', error: '未找到动态变量' })
      return result
    }

    // 检查用户指定的研究变量是否存在于动态变量候选中
    const missingVars = dyn_vars.filter((v: string) => !dynCandidates.includes(v))
    if (missingVars.length > 0) {
      // 不是所有指定的变量都在动态候选中
      const allVarNames = Object.keys(stepAResult.variables || {})

      result.overall_status = 'error'
      result.message = `您指定的研究变量不在动态变量候选列表中！

【您指定的研究变量】
${dyn_vars.join(', ')}

【缺失的变量】
${missingVars.join(', ')}

【可用的动态变量候选】（有时间维度）
${dynCandidates.length > 0 ? dynCandidates.join(', ') : '（无）'}

【所有检测到的变量】
${allVarNames.slice(0, 15).join(', ')}${allVarNames.length > 15 ? '...' : ''}

请检查：
1. 变量名是否拼写正确？
2. 这些变量是否确实在数据文件中？
3. 这些变量是否有时间维度？`

      ctx.emit('pipeline_failed', { step: 'A', error: '研究变量不存在' })
      return result
    }

    // P0 修复 v2.3.2：必须等待用户显式确认后才能继续处理
    // 使用 user_confirmed 参数而非依赖 mask_vars/stat_vars 的存在
    // 这样即使数据集没有 mask/static 变量，用户也能明确确认继续

    // 如果用户未显式确认，必须返回 awaiting_confirmation
    if (!user_confirmed) {
      // 格式化变量信息表格
      const formatVarInfo = (vars: Record<string, any>) => {
        const lines: string[] = []
        for (const [name, info] of Object.entries(vars)) {
          const dims = info.dims?.join(',') || '?'
          const dtype = info.dtype || '?'
          const suspected = info.suspected_type || '?'
          lines.push(`  ${name.padEnd(20)} | ${dims.padEnd(25)} | ${dtype.padEnd(10)} | ${suspected}`)
        }
        return lines.join('\n')
      }

      // 简化返回结果
      const simplifiedStepA = {
        status: stepAResult.status,
        nc_folder: stepAResult.nc_folder,
        file_count: stepAResult.file_count,
        file_list: stepAResult.file_list?.slice(0, 5),
        dynamic_vars_candidates: stepAResult.dynamic_vars_candidates,
        suspected_masks: stepAResult.suspected_masks,
        suspected_coordinates: stepAResult.suspected_coordinates,
        static_vars_found: stepAResult.static_vars_found,
        dynamic_files: stepAResult.dynamic_files,
        suspected_static_files: stepAResult.suspected_static_files,
        message: stepAResult.message
      }

      // 检测静态文件混入的警告
      const suspectedStaticFiles = stepAResult.suspected_static_files || []
      const dynamicFiles = stepAResult.dynamic_files || []
      const hasStaticFileMixedIn = suspectedStaticFiles.length > 0

      result.step_a = simplifiedStepA
      result.overall_status = 'awaiting_confirmation'
      result.message = `数据分析完成，**必须等待用户确认**后才能继续处理：

================================================================================
                              NC 文件分析结果
================================================================================

【文件信息】
- 动态数据目录: ${actualNcFolder}
- 找到文件数量: ${stepAResult.file_count} 个
- 静态文件: ${static_file || '无'}
${hasStaticFileMixedIn ? `
⚠️ **警告：检测到目录中混入了疑似静态文件！**
- 动态文件（有时间维度）: ${dynamicFiles.length} 个
  ${dynamicFiles.slice(0, 3).join(', ')}${dynamicFiles.length > 3 ? '...' : ''}
- 疑似静态文件（无时间维度）: ${suspectedStaticFiles.length} 个
  ${suspectedStaticFiles.join(', ')}

请确认这些文件是否应该：
1. 排除出处理列表？
2. 作为 static_file 使用？
` : ''}
【变量详情】
  变量名               | 维度                      | 数据类型   | 疑似类型
  -------------------- | ------------------------- | ---------- | ----------
${formatVarInfo(stepAResult.variables || {})}

================================================================================

【分类汇总】

1. 动态变量候选（有时间维度，可作为研究目标）:
${stepAResult.dynamic_vars_candidates?.map((v: string) => `   - ${v}`).join('\n') || '   无'}

2. 疑似掩码变量（mask/land 等关键字）:
${stepAResult.suspected_masks?.map((v: string) => `   - ${v}`).join('\n') || '   无'}

3. 疑似静态/坐标变量（lat/lon/depth/angle 等关键字）:
${stepAResult.suspected_coordinates?.map((v: string) => `   - ${v}`).join('\n') || '   无'}

================================================================================

【⚠️ 必须由用户确认的信息】

当前指定的研究变量: ${dyn_vars.join(', ')}

请用户逐一确认以下问题（Agent 不得代替用户决定）：

1. **研究变量**：您要研究的动态变量是 ${dyn_vars.join(', ')} 吗？
   - 可选动态变量: ${dynCandidates.join(', ')}
${hasStaticFileMixedIn ? `
2. **文件筛选**：目录中有 ${suspectedStaticFiles.length} 个疑似静态文件，如何处理？
   - 排除这些文件？
   - 将某个文件作为 static_file？
` : ''}
${hasStaticFileMixedIn ? '3' : '2'}. **掩码变量**：使用哪些掩码变量？
   - 检测到的疑似掩码: ${(stepAResult.suspected_masks || []).join(', ') || '无'}
   - 如果数据中没有掩码变量，可以跳过（设置 mask_vars: []）

${hasStaticFileMixedIn ? '4' : '3'}. **静态变量**：需要保存哪些静态变量？
   - 检测到的疑似坐标: ${(stepAResult.suspected_coordinates || []).join(', ') || '无'}
   - 如果数据中没有静态变量，可以跳过（设置 stat_vars: []）

${hasStaticFileMixedIn ? '5' : '4'}. **NaN/Inf 处理**：数据中是否允许 NaN/Inf 值存在？
   - 当前设置: allow_nan = ${allow_nan}

${hasStaticFileMixedIn ? '6' : '5'}. **数据集划分**【必须由用户决定】：
   - 请指定 train_ratio, valid_ratio, test_ratio（三者之和应为 1.0）
   - 示例: train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15
   - 数据将按时间顺序划分（不随机），保存到 train/hr/, valid/hr/, test/hr/
   ⚠️ Agent 不得自动决定划分比例！

${hasStaticFileMixedIn ? '7' : '6'}. **下采样参数**【必须由用户决定】：
   - scale: 下采样倍数（如 4 表示尺寸缩小为 1/4）
   - downsample_method: 插值方法，可选：
     • area（推荐）：区域平均，最接近真实低分辨率
     • cubic：三次插值，较平滑
     • linear：双线性插值
     • nearest：最近邻插值，保留原始值
     • lanczos：Lanczos 插值，高质量
   ⚠️ Agent 不得自动决定下采样参数！

**用户确认后**，请 Agent 使用以下参数重新调用工具：
- user_confirmed: true  ← 【必须】表示用户已确认
- train_ratio: 用户指定的训练集比例  ← 【必须】
- valid_ratio: 用户指定的验证集比例  ← 【必须】
- test_ratio: 用户指定的测试集比例   ← 【必须】
- scale: 用户指定的下采样倍数  ← 【必须，如 4】
- downsample_method: 用户指定的插值方法  ← 【必须，如 area】
- mask_vars: [用户确认的掩码变量列表]（无则传空数组 []）
- stat_vars: [用户确认的静态变量列表]（无则传空数组 []）
- nc_files: [要处理的文件列表]（如果需要排除某些文件）
================================================================================`

      ctx.emit('awaiting_user_confirmation', {
        requires_confirmation: true,
        user_confirmed: false,
        requires_split_ratio: true,
        suspected_masks: stepAResult.suspected_masks,
        suspected_coordinates: stepAResult.suspected_coordinates,
        dynamic_vars_candidates: stepAResult.dynamic_vars_candidates
      })
      return result
    }

    // 检查用户是否提供了数据集划分比例（必须由用户决定，Agent 不能自动设置）
    const userProvidedSplitRatio = train_ratio !== undefined && valid_ratio !== undefined && test_ratio !== undefined
    if (!userProvidedSplitRatio) {
      result.step_a = stepAResult
      result.overall_status = 'error'
      result.message = `数据集划分比例必须由用户指定！

【错误】Agent 不能自动决定划分比例。

请用户明确指定以下参数：
- train_ratio: 训练集比例（如 0.7）
- valid_ratio: 验证集比例（如 0.15）
- test_ratio: 测试集比例（如 0.15）

注意：三者之和应为 1.0

示例：
train_ratio: 0.7
valid_ratio: 0.15
test_ratio: 0.15`

      ctx.emit('error', {
        type: 'missing_split_ratio',
        message: '用户未指定数据集划分比例',
        required: ['train_ratio', 'valid_ratio', 'test_ratio']
      })
      return result
    }

    // 检查用户是否提供了下采样倍数（必须由用户决定）
    if (!scale || scale <= 1) {
      result.step_a = stepAResult
      result.overall_status = 'error'
      result.message = `下采样倍数必须由用户指定！

【错误】Agent 不能自动决定下采样倍数。

请用户明确指定 scale 参数（如 scale=4 表示尺寸缩小为 1/4）。

示例：
scale: 4`

      ctx.emit('error', {
        type: 'missing_scale',
        message: '用户未指定下采样倍数',
        required: ['scale']
      })
      return result
    }

    // 检查用户是否提供了下采样方法（必须由用户决定）
    const validMethods = ['nearest', 'linear', 'cubic', 'area', 'lanczos']
    if (!downsample_method || !validMethods.includes(downsample_method)) {
      result.step_a = stepAResult
      result.overall_status = 'error'
      result.message = `下采样插值方法必须由用户指定！

【错误】Agent 不能自动决定下采样方法。

请用户从以下方法中选择一个：
- area（推荐）：区域平均，最接近真实低分辨率
- cubic：三次插值，较平滑
- linear：双线性插值
- nearest：最近邻插值，保留原始值
- lanczos：Lanczos 插值，高质量

示例：
downsample_method: area`

      ctx.emit('error', {
        type: 'missing_downsample_method',
        message: '用户未指定下采样方法',
        required: ['downsample_method'],
        valid_options: validMethods
      })
      return result
    }

    // 验证划分比例之和
    const totalRatio = train_ratio + valid_ratio + test_ratio
    if (Math.abs(totalRatio - 1.0) > 0.01) {
      result.step_a = stepAResult
      result.overall_status = 'error'
      result.message = `数据集划分比例之和必须为 1.0！

当前设置：
- train_ratio: ${train_ratio}
- valid_ratio: ${valid_ratio}
- test_ratio: ${test_ratio}
- 总和: ${totalRatio}

请调整比例使其总和为 1.0`

      ctx.emit('error', {
        type: 'invalid_split_ratio',
        message: `划分比例之和 ${totalRatio} != 1.0`
      })
      return result
    }

    // P0 修复：移除硬编码默认值，必须使用用户确认的值或从数据检测的值
    // 如果没有检测到任何掩码或坐标变量，且用户未提供，应该报错而非使用默认值

    // 掩码变量：由用户指定或从 Step A 检测到
    // 注意：某些数据集可能没有掩码变量，这是允许的
    const detectedMaskVars = stepAResult.suspected_masks || []
    const finalMaskVars = mask_vars || (detectedMaskVars.length > 0 ? detectedMaskVars : [])

    // 如果没有掩码变量，发出警告但继续（不强制报错）
    if (finalMaskVars.length === 0) {
      ctx.emit('warning', {
        type: 'no_mask_vars',
        message: '未检测到掩码变量，将跳过掩码相关处理',
        suggestion: '如果数据中有掩码变量，请通过 mask_vars 参数指定'
      })
    }

    // 静态变量：由用户指定或从 Step A 检测到
    // 注意：某些数据集可能没有静态变量，这是允许的
    const detectedCoordVars = stepAResult.suspected_coordinates || []
    const finalStaticVars = stat_vars || (detectedCoordVars.length > 0
      ? [...detectedCoordVars, ...detectedMaskVars]
      : [])

    // 如果没有静态变量，发出警告但继续
    if (finalStaticVars.length === 0) {
      ctx.emit('warning', {
        type: 'no_static_vars',
        message: '未检测到静态变量，将跳过静态变量保存',
        suggestion: '如果需要保存坐标等静态变量，请通过 stat_vars 参数指定'
      })
    }

    // 主掩码变量选择（如果有掩码变量的话）
    let primaryMaskVar: string | undefined
    if (finalMaskVars.length === 1) {
      primaryMaskVar = finalMaskVars[0]
    } else if (finalMaskVars.length > 1) {
      // 有多个掩码变量时，优先选择 rho 网格的（ROMS 模型常见）
      const rhoMask = finalMaskVars.find((m: string) => m.includes('rho'))
      primaryMaskVar = rhoMask || finalMaskVars[0]
      ctx.emit('info', {
        type: 'primary_mask_selected',
        message: `自动选择主掩码变量: ${primaryMaskVar}（共有 ${finalMaskVars.length} 个掩码变量）`,
        all_masks: finalMaskVars
      })
    }
    // 如果没有掩码变量，primaryMaskVar 保持 undefined

    // P0 修复：经纬度变量必须从数据中检测到或由用户指定，不使用硬编码默认值
    const detectedLonVar = finalStaticVars.find((v: string) =>
      v.toLowerCase().includes('lon') && !v.toLowerCase().includes('mask')
    )
    const detectedLatVar = finalStaticVars.find((v: string) =>
      v.toLowerCase().includes('lat') && !v.toLowerCase().includes('mask')
    )
    const finalLonVar = lon_var || detectedLonVar
    const finalLatVar = lat_var || detectedLatVar

    // 如果未检测到经纬度变量，发出警告但继续（某些数据集可能不需要）
    if (!finalLonVar || !finalLatVar) {
      ctx.emit('warning', {
        type: 'missing_coordinate_vars',
        message: `未检测到经纬度变量：lon_var=${finalLonVar || '未知'}, lat_var=${finalLatVar || '未知'}`,
        suggestion: '如果需要坐标验证，请通过 lon_var/lat_var 参数指定'
      })
    }

    // Step B
    ctx.emit('step_started', { step: 'B', description: '进行张量约定验证' })

    const tempDir = path.resolve(ctx.sandbox.workDir, 'ocean_preprocess_temp')
    const inspectResultPath = path.join(tempDir, 'inspect_result.json')

    const stepBResult = await oceanValidateTensorTool.exec({
      inspect_result_path: inspectResultPath,
      research_vars: dyn_vars,
      mask_vars: finalMaskVars
    }, ctx)

    result.step_b = stepBResult

    if (stepBResult.status === 'error') {
      result.overall_status = 'error'
      result.message = 'Step B 失败'
      ctx.emit('pipeline_failed', { step: 'B', result })
      return result
    }

    // Step C
    ctx.emit('step_started', { step: 'C', description: '转换为NPY格式存储' })

    const stepCResult = await oceanConvertNpyTool.exec({
      nc_folder: actualNcFolder,
      output_base,
      dyn_vars,
      static_file,
      dyn_file_pattern: actualFilePattern,
      stat_vars: finalStaticVars,
      mask_vars: finalMaskVars,
      lon_var: finalLonVar,
      lat_var: finalLatVar,
      run_validation,
      allow_nan,
      lon_range,
      lat_range,
      // Rule 2/3 验证参数（使用检测到的主掩码变量）
      mask_src_var: primaryMaskVar,
      mask_derive_op: 'identity',
      heuristic_check_var: dyn_vars?.[0],  // 使用第一个动态变量进行启发式验证
      land_threshold_abs: 1e-12,
      heuristic_sample_size: 2000,
      require_sorted: true,
      // 数据集划分参数
      train_ratio,
      valid_ratio,
      test_ratio,
      // 裁剪参数
      h_slice,
      w_slice,
      scale,
      workers
    }, ctx)

    result.step_c = stepCResult

    if (stepCResult.status !== 'pass') {
      result.overall_status = 'error'
      result.message = 'Step C 失败'
      ctx.emit('pipeline_failed', { step: 'C', result })
      return result
    }

    // Step D: 下采样
    if (!skip_downsample) {
      ctx.emit('step_started', { step: 'D', description: 'HR → LR 下采样' })

      const stepDResult = await oceanDownsampleTool.exec({
        dataset_root: output_base,
        scale: scale,
        method: downsample_method,
        splits: ['train', 'valid', 'test'],
        include_static: false
      }, ctx)

      result.step_d = stepDResult

      if (stepDResult.status === 'error') {
        result.overall_status = 'error'
        result.message = 'Step D 下采样失败'
        ctx.emit('pipeline_failed', { step: 'D', result })
        return result
      }

      ctx.emit('step_completed', { step: 'D', result: stepDResult })
    } else {
      result.step_d = { status: 'skipped', reason: 'skip_downsample=true' }
    }

    // Step E: 可视化
    if (!skip_visualize) {
      ctx.emit('step_started', { step: 'E', description: '生成可视化对比图' })

      const stepEResult = await oceanVisualizeTool.exec({
        dataset_root: output_base,
        splits: ['train', 'valid', 'test']
      }, ctx)

      result.step_e = stepEResult

      if (stepEResult.status === 'error') {
        // 可视化失败不阻止整体流程，只是警告
        ctx.emit('warning', {
          type: 'visualize_failed',
          message: '可视化生成失败，但不影响数据处理结果',
          error: stepEResult.errors
        })
      } else {
        ctx.emit('step_completed', { step: 'E', result: stepEResult })
      }
    } else {
      result.step_e = { status: 'skipped', reason: 'skip_visualize=true' }
    }

    // 最终状态
    if (stepCResult.status === 'pass') {
      result.overall_status = 'pass'
      result.message = '预处理完成，所有检查通过'
      result.validation_summary = stepCResult.post_validation
      ctx.emit('pipeline_completed', { result })
    } else {
      result.overall_status = 'error'
      result.message = 'Step C 失败'
      ctx.emit('pipeline_failed', { step: 'C', result })
    }

    return result
  }
})
