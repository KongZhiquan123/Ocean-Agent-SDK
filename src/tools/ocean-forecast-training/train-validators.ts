import type { ForecastDatasetInfo } from './workflow'
import path from 'node:path'
import { shellEscapeDouble, extractTaggedJson } from '@/utils/shell'

export async function validateDataset(
  datasetRoot: string,
  pythonPath: string,
  trainingDir: string,
  ctx: { sandbox: { exec: (cmd: string, options?: { timeoutMs?: number }) => Promise<{ code: number; stdout: string; stderr: string }> } },
): Promise<ForecastDatasetInfo> {
  const validateScript = path.join(trainingDir, 'validate_dataset.py')
  const validateResult = await ctx.sandbox.exec(
    `"${shellEscapeDouble(pythonPath)}" "${shellEscapeDouble(validateScript)}" --dataset_root "${shellEscapeDouble(datasetRoot)}"`,
    { timeoutMs: 60000 }
  )
  if (validateResult.code === 0) {
    return JSON.parse(validateResult.stdout)
  }

  return {
    status: 'error',
    dataset_root: datasetRoot,
    dyn_vars: [],
    spatial_shape: null,
    splits: {},
    total_timesteps: 0,
    time_range: null,
    has_static: false,
    static_vars: [],
    warnings: [],
    errors: [`验证脚本执行失败: ${validateResult.stderr}`]
  }
}

/**
 * 调用 recommend_hyperparams.py 获取超参数推荐。
 * 失败时返回 null，不抛出异常（不影响主流程）。
 */
export async function runHyperparamRecommendation(
  args: {
    dataset_root?: string
    model_name?: string
    dyn_vars?: string[]
    in_t?: number
    out_t?: number
    device_ids?: number[]
  },
  pythonPath: string,
  trainingDir: string,
  ctx: { sandbox: { exec: (cmd: string, options?: { timeoutMs?: number }) => Promise<{ code: number; stdout: string; stderr: string }> } },
): Promise<Record<string, unknown> | null> {
  if (!args.dataset_root || !args.model_name || !args.dyn_vars?.length) {
    return null
  }
  try {
    const recommendScript = path.join(trainingDir, 'recommend_hyperparams.py')
    const deviceId = Number(args.device_ids?.[0] ?? 0)
    const inT = args.in_t ?? 7
    const outT = args.out_t ?? 1
    const cmd = [
      `cd "${shellEscapeDouble(trainingDir)}"`,
      `&&`,
      `CUDA_VISIBLE_DEVICES=${deviceId}`,
      `"${shellEscapeDouble(pythonPath)}"`,
      `"${shellEscapeDouble(recommendScript)}"`,
      `--dataset_root "${shellEscapeDouble(args.dataset_root)}"`,
      `--model_name "${shellEscapeDouble(args.model_name)}"`,
      `--dyn_vars "${shellEscapeDouble(args.dyn_vars.join(','))}"`,
      `--in_t ${inT}`,
      `--out_t ${outT}`,
      `--device 0`,
    ].join(' ')
    const result = await ctx.sandbox.exec(cmd, { timeoutMs: 180000 })
    if (result.code !== 0) return null
    return extractTaggedJson(result.stdout, 'recommend')
  } catch {
    return null
  }
}
