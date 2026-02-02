/**
 * @file full.ts
 * @description 完整的海洋数据预处理流程工具
 *              串联 Step A -> B -> C 三个步骤
 *
 * @author leizheng
 * @date 2026-02-02
 * @version 2.1.0
 *
 * @changelog
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
      description: '动态文件的 glob 匹配模式，如 "*.nc" 或 "*avg*.nc"',
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
      description: '经度参考变量名',
      required: false,
      default: 'lon_rho'
    },
    lat_var: {
      type: 'string',
      description: '纬度参考变量名',
      required: false,
      default: 'lat_rho'
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
      output_base,
      dyn_vars,
      static_file,
      dyn_file_pattern = '*.nc',
      mask_vars,
      stat_vars,
      lon_var = 'lon_rho',
      lat_var = 'lat_rho',
      run_validation = true,
      allow_nan = false,
      lon_range,
      lat_range
    } = args

    ctx.emit('pipeline_started', {
      nc_folder,
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
      nc_folder,
      static_file,
      dyn_file_pattern
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
- 搜索目录: ${nc_folder}
- 文件匹配模式: "${dyn_file_pattern}"
请检查：
1. nc_folder 路径是否正确
2. dyn_file_pattern 是否匹配你的文件名`
      ctx.emit('pipeline_failed', { step: 'A', error: '未找到动态数据文件' })
      return result
    }

    // 检查是否需要用户确认
    const hasSuspectedVars = (stepAResult.suspected_masks && stepAResult.suspected_masks.length > 0) ||
                             (stepAResult.suspected_coordinates && stepAResult.suspected_coordinates.length > 0)

    const userProvidedMaskVars = mask_vars !== undefined
    const userProvidedStaticVars = stat_vars !== undefined

    if (hasSuspectedVars && (!userProvidedMaskVars || !userProvidedStaticVars)) {
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
        message: stepAResult.message
      }

      result.step_a = simplifiedStepA
      result.overall_status = 'awaiting_confirmation'
      result.message = `数据分析完成，请用户确认变量分类：

================================================================================
                              NC 文件分析结果
================================================================================

【文件信息】
- 动态数据目录: ${nc_folder}
- 找到文件数量: ${stepAResult.file_count} 个
- 静态文件: ${static_file || '无'}

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

【请用户确认】
当前指定的动态变量: ${dyn_vars.join(', ')}

请确认：
1. 动态变量（dyn_vars）是否正确？
2. 掩码变量（mask_vars）应该包含哪些？
3. 静态变量（stat_vars）应该包含哪些？

确认后，Agent 将使用确认的参数继续执行数据提取。
================================================================================`

      ctx.emit('awaiting_user_confirmation', {
        suspected_masks: stepAResult.suspected_masks,
        suspected_coordinates: stepAResult.suspected_coordinates,
        dynamic_vars_candidates: stepAResult.dynamic_vars_candidates
      })
      return result
    }

    // 如果没有疑似变量或用户已提供配置，按优先级使用：
    // 1. 用户提供的值
    // 2. Step A 检测到的值
    // 3. 硬编码默认值（仅作为最后保底）

    // 掩码变量：优先用户指定 > Step A 检测 > 默认值
    const finalMaskVars = mask_vars
      || (stepAResult.suspected_masks?.length > 0 ? stepAResult.suspected_masks : null)
      || ['mask_rho', 'mask_u', 'mask_v', 'mask_psi']

    // 静态变量：优先用户指定 > Step A 检测 > 默认值
    const finalStaticVars = stat_vars
      || (stepAResult.suspected_coordinates?.length > 0
        ? [...stepAResult.suspected_coordinates, ...(stepAResult.suspected_masks || [])]
        : null)
      || [
        'lon_rho', 'lat_rho', 'lon_u', 'lat_u', 'lon_v', 'lat_v',
        'angle', 'h', 'f', 'pm', 'pn',
        'mask_rho', 'mask_u', 'mask_v', 'mask_psi'
      ]

    // 确定主掩码变量（用于精确对比和启发式验证）
    // 优先选择包含 'rho' 的，其次选择第一个
    const primaryMaskVar = finalMaskVars.find((m: string) => m.includes('rho'))
      || finalMaskVars[0]
      || 'mask_rho'

    // 确定经纬度变量（从静态变量中查找）
    const detectedLonVar = finalStaticVars.find((v: string) =>
      v.toLowerCase().includes('lon') && !v.toLowerCase().includes('mask')
    )
    const detectedLatVar = finalStaticVars.find((v: string) =>
      v.toLowerCase().includes('lat') && !v.toLowerCase().includes('mask')
    )
    const finalLonVar = lon_var || detectedLonVar || 'lon_rho'
    const finalLatVar = lat_var || detectedLatVar || 'lat_rho'

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
      nc_folder,
      output_base,
      dyn_vars,
      static_file,
      dyn_file_pattern,
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
