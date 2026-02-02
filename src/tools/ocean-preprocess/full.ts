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

**注意**：研究变量必须由用户明确指定

**输出目录结构**：
- output_base/hr/变量.npy - 高分辨率动态数据
- output_base/static/变量.npy - 静态数据

**返回**：各步骤结果、整体状态（pass表示全部通过）`,

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
    }
  },

  attributes: {
    readonly: false,
    noEffect: false
  },

  async exec(args, ctx) {
    const { nc_folder, output_base, research_vars, static_file, file_filter = 'avg' } = args

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

    // Step B
    ctx.emit('step_started', { step: 'B', description: '进行张量约定验证' })

    const inspectResultPath = '/tmp/ocean_preprocess/inspect_result.json'

    const stepBResult = await oceanValidateTensorTool.exec({
      inspect_result_path: inspectResultPath,
      research_vars
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
      research_vars,
      static_file,
      file_filter
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
