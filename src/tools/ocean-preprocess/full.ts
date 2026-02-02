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
3. Step C: 转换为NPY格式存储

**重要**：如果 Step A 检测到疑似变量但未提供 mask_vars/static_vars，会返回 awaiting_confirmation 状态，此时需要用户确认后重新调用。

**注意**：研究变量必须由用户明确指定

**输出目录结构**：
- output_base/hr/变量.npy - 高分辨率动态数据
- output_base/static/变量.npy - 静态数据

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
    research_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '研究变量列表（必须由用户指定）'
    },
    static_file: {
      type: 'string',
      description: '静态NC文件路径（可选）',
      required: false
    },
    file_filter: {
      type: 'string',
      description: '文件名过滤关键字',
      required: false,
      default: 'avg'
    },
    mask_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '掩码变量列表（建议从 Step A 的 suspected_masks 中选择。如不提供，首次调用会返回建议列表）',
      required: false
    },
    static_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '静态变量列表（建议从 Step A 的 suspected_coordinates 中选择。如不提供，首次调用会返回建议列表）',
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
      research_vars,
      static_file,
      file_filter = 'avg',
      mask_vars,
      static_vars
    } = args

    ctx.emit('pipeline_started', {
      nc_folder,
      output_base,
      research_vars
    })

    const result = {
      step_a: null as any,
      step_b: null as any,
      step_c: null as any,
      overall_status: 'pending' as string,
      message: ''
    }

    // Step A
    ctx.emit('step_started', { step: 'A', description: '查看数据并定义变量' })

    const stepAResult = await oceanInspectDataTool.exec({
      nc_folder,
      static_file,
      file_filter
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
- 文件过滤器: "${file_filter}"
请检查：
1. nc_folder 路径是否正确
2. file_filter 是否匹配你的文件名（例如文件名包含 "copernicus" 时，使用 file_filter: "copernicus"）`
      ctx.emit('pipeline_failed', { step: 'A', error: '未找到动态数据文件' })
      return result
    }

    // 检查是否需要用户确认
    const hasSuspectedVars = (stepAResult.suspected_masks && stepAResult.suspected_masks.length > 0) ||
                             (stepAResult.suspected_coordinates && stepAResult.suspected_coordinates.length > 0)

    const userProvidedMaskVars = mask_vars !== undefined
    const userProvidedStaticVars = static_vars !== undefined

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
        file_list: stepAResult.file_list?.slice(0, 5),  // 只展示前5个文件
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
当前指定的研究变量: ${research_vars.join(', ')}

请确认：
1. 研究变量（dyn_vars）是否正确？
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

    // 如果没有疑似变量或用户已提供配置，使用用户提供的值或默认值
    const finalMaskVars = mask_vars || ['mask_u', 'mask_rho', 'mask_v']
    const finalStaticVars = static_vars || ['angle', 'h', 'mask_u', 'mask_rho', 'mask_v', 'pn', 'pm', 'f',
                                             'x_rho', 'x_u', 'x_v', 'y_rho', 'y_u', 'y_v', 'lat_psi', 'lon_psi']

    // Step B
    ctx.emit('step_started', { step: 'B', description: '进行张量约定验证' })

    // 使用与 Step A 相同的路径（基于 sandbox.workDir）
    const tempDir = path.resolve(ctx.sandbox.workDir, 'ocean_preprocess_temp')
    const inspectResultPath = path.join(tempDir, 'inspect_result.json')

    const stepBResult = await oceanValidateTensorTool.exec({
      inspect_result_path: inspectResultPath,
      research_vars,
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
      dyn_vars: research_vars,
      static_file,
      dyn_file_pattern: file_filter ? `*${file_filter}*.nc` : '*.nc',
      stat_vars: finalStaticVars,
      mask_vars: finalMaskVars
    }, ctx)

    result.step_c = stepCResult

    if (stepCResult.status === 'pass') {
      result.overall_status = 'pass'
      result.message = '预处理完成，所有检查通过'
      ctx.emit('pipeline_completed', { result })
    } else {
      result.overall_status = 'error'
      result.message = 'Step C 失败'
      ctx.emit('pipeline_failed', { step: 'C', result })
    }

    return result
  }
})
