/**
 * @file full.ts
 * @description 完整的海洋数据预处理流程工具
 *              串联 Step A -> B -> C 三个步骤
 *
 * @author leizheng
 * @date 2026-02-02
 * @version 2.3.0
 *
 * @changelog
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

export const oceanPreprocessFullTool = defineTool({
  name: 'ocean_preprocess_full',
  description: `运行完整的超分辨率数据预处理流程 (A -> B -> C)

自动执行所有三个步骤：
1. Step A: 查看数据并定义变量
2. Step B: 进行张量约定验证
3. Step C: 转换为NPY格式存储（含后置验证 Rule 1/2/3）

**重要**：如果 Step A 检测到疑似变量但未提供 mask_vars/stat_vars，会返回 awaiting_confirmation 状态，此时需要用户确认后重新调用。

**注意**：研究变量必须由用户明确指定

**输出目录结构**：
- output_base/target_variables/变量.npy - 动态研究变量
- output_base/static_variables/编号_变量.npy - 静态变量（带编号）
- output_base/preprocess_manifest.json - 数据溯源清单

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
      lat_range
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

    // P0 修复：必须等待用户确认后才能继续处理
    // 判断条件：用户必须同时提供 mask_vars 和 stat_vars 才表示已确认
    // 如果只检测到了变量但用户未明确确认，必须返回 awaiting_confirmation

    const userProvidedMaskVars = mask_vars !== undefined && mask_vars.length > 0
    const userProvidedStaticVars = stat_vars !== undefined && stat_vars.length > 0
    const userConfirmedAllVars = userProvidedMaskVars && userProvidedStaticVars

    // 如果用户没有同时提供 mask_vars 和 stat_vars，说明还没确认，必须暂停
    if (!userConfirmedAllVars) {
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
   ${!userProvidedMaskVars ? '   ⚠️ 您尚未指定 mask_vars' : ''}

${hasStaticFileMixedIn ? '4' : '3'}. **静态变量**：需要保存哪些静态变量？
   - 检测到的疑似坐标: ${(stepAResult.suspected_coordinates || []).join(', ') || '无'}
   ${!userProvidedStaticVars ? '   ⚠️ 您尚未指定 stat_vars' : ''}

${hasStaticFileMixedIn ? '5' : '4'}. **NaN/Inf 处理**：数据中是否允许 NaN/Inf 值存在？
   - 当前设置: allow_nan = ${allow_nan}

**用户确认后**，请 Agent 使用以下参数重新调用工具：
- nc_files: [要处理的文件列表]（如果需要排除某些文件）
- mask_vars: [用户确认的掩码变量列表]
- stat_vars: [用户确认的静态变量列表]
================================================================================`

      ctx.emit('awaiting_user_confirmation', {
        requires_confirmation: true,
        missing_mask_vars: !userProvidedMaskVars,
        missing_stat_vars: !userProvidedStaticVars,
        suspected_masks: stepAResult.suspected_masks,
        suspected_coordinates: stepAResult.suspected_coordinates,
        dynamic_vars_candidates: stepAResult.dynamic_vars_candidates
      })
      return result
    }

    // P0 修复：移除硬编码默认值，必须使用用户确认的值或从数据检测的值
    // 如果没有检测到任何掩码或坐标变量，且用户未提供，应该报错而非使用默认值

    // 掩码变量：必须由用户指定或从 Step A 检测到
    const detectedMaskVars = stepAResult.suspected_masks || []
    const finalMaskVars = mask_vars || (detectedMaskVars.length > 0 ? detectedMaskVars : null)

    if (!finalMaskVars || finalMaskVars.length === 0) {
      result.overall_status = 'error'
      result.message = `未检测到掩码变量，且用户未指定 mask_vars 参数！

【问题说明】
掩码变量（mask）用于标识陆地/海洋区域，是海洋数据预处理的必要组件。

【检测结果】
- 未自动检测到任何疑似掩码变量（如 mask_rho, mask_u, mask_v 等）

【解决方案】
请使用 mask_vars 参数明确指定掩码变量，例如：
  mask_vars: ["mask_rho", "mask_u", "mask_v"]

如果数据中确实没有掩码变量，请检查：
1. 数据文件是否完整？
2. 是否需要从其他静态文件中获取掩码？`
      ctx.emit('pipeline_failed', { step: 'A', error: '缺少掩码变量' })
      return result
    }

    // 静态变量：必须由用户指定或从 Step A 检测到
    const detectedCoordVars = stepAResult.suspected_coordinates || []
    const finalStaticVars = stat_vars || (detectedCoordVars.length > 0
      ? [...detectedCoordVars, ...detectedMaskVars]
      : null)

    if (!finalStaticVars || finalStaticVars.length === 0) {
      result.overall_status = 'error'
      result.message = `未检测到静态/坐标变量，且用户未指定 stat_vars 参数！

【问题说明】
静态变量（如经纬度、地形深度等）是海洋数据的重要辅助信息。

【检测结果】
- 未自动检测到任何疑似坐标变量（如 lon_rho, lat_rho, h, angle 等）

【解决方案】
请使用 stat_vars 参数明确指定静态变量，例如：
  stat_vars: ["lon_rho", "lat_rho", "h", "angle", "mask_rho"]

如果数据中确实没有静态变量，请确认是否需要提供静态文件（static_file 参数）。`
      ctx.emit('pipeline_failed', { step: 'A', error: '缺少静态变量' })
      return result
    }

    // P0 修复：主掩码变量选择需要用户确认，不再自动选择
    // 但如果只有一个掩码变量，可以直接使用
    let primaryMaskVar: string
    if (finalMaskVars.length === 1) {
      primaryMaskVar = finalMaskVars[0]
    } else {
      // 有多个掩码变量时，优先选择 rho 网格的（ROMS 模型常见）
      const rhoMask = finalMaskVars.find((m: string) => m.includes('rho'))
      primaryMaskVar = rhoMask || finalMaskVars[0]
      // 在返回结果中告知用户使用了哪个主掩码
      ctx.emit('info', {
        type: 'primary_mask_selected',
        message: `自动选择主掩码变量: ${primaryMaskVar}（共有 ${finalMaskVars.length} 个掩码变量）`,
        all_masks: finalMaskVars
      })
    }

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
      require_sorted: true
    }, ctx)

    result.step_c = stepCResult

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
