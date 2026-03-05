/**
 * @file train.ts
 * @description Ocean forecast training tool - 4-stage confirmation workflow
 * @author Leizheng
 * @date 2026-02-26
 * @version 1.1.0
 *
 * @changelog
 *   - 2026-02-26 Leizheng: v1.1.0 fix PASS phase reads device_ids from mergedParams instead of raw args
 *   - 2026-02-26 Leizheng: v1.0.0 initial version for ocean forecast training
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findPythonWithModule, findFirstPythonPath } from '@/utils/python-manager'
import { trainingProcessManager } from '@/utils/training-process-manager'
import { saveOrAppendNotebook, generateForecastTrainCells } from './notebook'
import type { ForecastTrainNotebookParams } from './notebook'
import path from 'node:path'
import { shellEscapeDouble, shellSafeJson, extractTaggedJson } from '@/utils/shell'
import { isPortFree, findFreePort } from '@/utils/port'
import { loadSessionParams, saveSessionParams, formatRecommendationMessage } from '@/utils/training-utils'
import {
  ForecastTrainingWorkflow,
  ForecastTrainingState,
  type ForecastWorkflowParams,
  type ForecastDatasetInfo,
  type GpuInfo,
  type ModelInfo,
} from './workflow-state'

async function validateDataset(
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
async function runHyperparamRecommendation(
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

export const oceanForecastTrainTool = defineTool({
  name: 'ocean_forecast_train_start',
  description: `执行海洋时序预测模型训练或测试。

**分阶段确认流程**（每阶段必须等待用户确认）：
1. 确认数据目录和输出目录（自动检测变量和时序信息）
2. 选择训练模型
3. 确认训练参数（包括 GPU 选择）
4. 最终确认执行

**首次调用**：只传 dataset_root 和 log_dir，工具会自动检测数据并展示信息
**逐步补充参数**：每次调用补充该阶段需要的参数，直到所有阶段通过
**最终执行**：传入 user_confirmed=true 和 confirmation_token 后启动后台训练

**后台执行模式**：
- 训练启动后立即返回 process_id，不会阻塞等待训练完成
- 使用 ocean_forecast_train_status 工具查询训练状态和实时日志
- 服务器关闭时会自动终止训练进程

**训练模式 (mode=train)**：执行完整训练流程，包含验证和早停
**测试模式 (mode=test)**：加载最佳模型，在测试集上评估
**预测模式 (mode=predict)**：加载模型对测试集执行全量预测，输出 NPY 文件（跳过训练工作流）

**GPU 模式**：
- 单卡：device_ids 长度为 1
- 多卡 DP：distribute=true, distribute_mode="DP"
- 多卡 DDP（推荐）：distribute=true, distribute_mode="DDP"`,

  params: {
    dataset_root: {
      type: 'string',
      description: '预处理数据根目录（ocean-forecast-data-preprocess 输出目录）',
      required: false
    },
    log_dir: {
      type: 'string',
      description: '训练日志输出目录',
      required: false
    },
    model_name: {
      type: 'string',
      description: '模型名称（如 FNO2d, UNet2d, SwinTransformerV2 等）',
      required: false
    },
    dyn_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '动态变量列表（如 ["uo", "vo"]）。如不提供，将从数据目录自动检测并要求确认。',
      required: false
    },
    in_t: {
      type: 'number',
      description: '输入时间步数（用于构建时序窗口）',
      required: false
    },
    out_t: {
      type: 'number',
      description: '输出时间步数（预测未来步数）',
      required: false
    },
    stride: {
      type: 'number',
      description: '时间窗口滑动步长',
      required: false
    },
    mode: {
      type: 'string',
      description: '运行模式: "train", "test" 或 "predict"（predict 跳过训练工作流，直接推理）',
      required: false,
      default: 'train'
    },
    epochs: {
      type: 'number',
      description: '训练轮数',
      required: false,
      default: 500
    },
    lr: {
      type: 'number',
      description: '学习率',
      required: false,
      default: 0.001
    },
    batch_size: {
      type: 'number',
      description: '训练 batch size',
      required: false,
      default: 4
    },
    eval_batch_size: {
      type: 'number',
      description: '评估 batch size',
      required: false,
      default: 4
    },
    device_ids: {
      type: 'array',
      items: { type: 'number' },
      description: '使用的 GPU 列表（如 [0, 1, 2, 3]）。必须由用户确认。若启用多卡训练，至少需要两个 GPU。',
      required: false
    },
    distribute: {
      type: 'boolean',
      description: '是否启用多卡训练',
      required: false,
      default: false
    },
    distribute_mode: {
      type: 'string',
      description: '多卡模式: "DP" 或 "DDP"',
      required: false,
      default: 'DDP'
    },
    master_port: {
      type: 'number',
      description: 'DDP 主端口（可选，端口冲突时可指定）',
      required: false
    },
    use_amp: {
      type: 'boolean',
      description: '是否启用 AMP 混合精度训练（减少约 40-50% 显存）',
      required: false
    },
    gradient_checkpointing: {
      type: 'boolean',
      description: '是否启用 Gradient Checkpointing（减少约 60% 激活显存，增加约 30% 计算时间，默认开启）',
      required: false,
      default: true
    },
    normalize: {
      type: 'boolean',
      description: '是否归一化',
      required: false,
      default: true
    },
    normalizer_type: {
      type: 'string',
      description: '归一化类型: "PGN" 或 "GN"',
      required: false,
      default: 'PGN'
    },
    optimizer: {
      type: 'string',
      description: '优化器: "AdamW", "Adam", "SGD"',
      required: false,
      default: 'AdamW'
    },
    weight_decay: {
      type: 'number',
      description: '权重衰减',
      required: false,
      default: 0.001
    },
    scheduler: {
      type: 'string',
      description: '学习率调度器: "StepLR", "MultiStepLR", "OneCycleLR"',
      required: false,
      default: 'StepLR'
    },
    patience: {
      type: 'number',
      description: '早停耐心值',
      required: false,
      default: 10
    },
    eval_freq: {
      type: 'number',
      description: '评估频率（每 N 个 epoch）',
      required: false,
      default: 5
    },
    seed: {
      type: 'number',
      description: '随机种子',
      required: false,
      default: 42
    },
    ckpt_path: {
      type: 'string',
      description: '恢复训练的检查点路径',
      required: false
    },
    user_confirmed: {
      type: 'boolean',
      description: '【必须】用户确认标志。必须在展示参数汇总并获得用户明确确认后，才能设置为 true。禁止自动设置！',
      required: false,
      default: false
    },
    confirmation_token: {
      type: 'string',
      description: '执行确认 Token。必须从 awaiting_execution 阶段的返回值中获取。',
      required: false
    }
  },

  async exec(args, ctx) {
    // 训练工具需要 torch，优先查找安装了 torch 的 Python
    const pythonPath = (await findPythonWithModule('torch')) || (await findFirstPythonPath())
    if (!pythonPath) {
      throw new Error('未找到可用的 Python 解释器（需要安装 torch）')
    }

    const trainingDir = path.resolve(process.cwd(), 'scripts/ocean-forecast-training')

    // ===== 1. 构建工作流参数（合并 session 缓存，防止可选参数跨调用丢失） =====
    const SESSION_FILENAME = '.ocean_forecast_train_session.json' as const
    const workflowArgs = { ...args }
    const sessionParams = args.log_dir ? await loadSessionParams<ForecastWorkflowParams>(args.log_dir, SESSION_FILENAME, ctx) : null
    const workflow = new ForecastTrainingWorkflow(workflowArgs, sessionParams ?? undefined)
    const stateCheck = workflow.determineCurrentState()

    // ===== 2. 如果未到 PASS 阶段，收集上下文信息并返回提示 =====
    if (stateCheck.currentState !== ForecastTrainingState.PASS) {
      const context: {
        datasetInfo?: ForecastDatasetInfo
        gpuInfo?: GpuInfo
        modelList?: ModelInfo[]
      } = {}

      // 如果有 dataset_root，验证数据目录
      if (args.dataset_root) {
        context.datasetInfo = await validateDataset(args.dataset_root, pythonPath, trainingDir, ctx)
      }

      // 阶段2+需要模型列表
      if (stateCheck.currentState === ForecastTrainingState.AWAITING_MODEL_SELECTION) {
        const listScript = path.join(trainingDir, 'list_models.py')
        const listResult = await ctx.sandbox.exec(
          `"${shellEscapeDouble(pythonPath)}" "${shellEscapeDouble(listScript)}"`,
          { timeoutMs: 30000 }
        )
        if (listResult.code === 0) {
          const parsed = JSON.parse(listResult.stdout)
          context.modelList = parsed.models
        }
      }

      // 阶段3+需要 GPU 信息
      if (
        stateCheck.currentState === ForecastTrainingState.AWAITING_PARAMETERS ||
        stateCheck.currentState === ForecastTrainingState.AWAITING_EXECUTION ||
        stateCheck.currentState === ForecastTrainingState.TOKEN_INVALID
      ) {
        const gpuScript = path.join(trainingDir, 'check_gpu.py')
        const gpuResult = await ctx.sandbox.exec(
          `"${shellEscapeDouble(pythonPath)}" "${shellEscapeDouble(gpuScript)}"`,
          { timeoutMs: 30000 }
        )
        if (gpuResult.code === 0) {
          context.gpuInfo = JSON.parse(gpuResult.stdout)
        }
      }

      const prompt = workflow.getStagePrompt(context)

      // AWAITING_EXECUTION 时持久化全量参数，供后续执行调用恢复可选参数（如 normalizer_type）
      if (stateCheck.currentState === ForecastTrainingState.AWAITING_EXECUTION && args.log_dir) {
        await saveSessionParams(args.log_dir, SESSION_FILENAME, workflow.getParams(), ctx)
      }
      // AWAITING_EXECUTION 时运行超参数推荐（实测显存 + 数据集分析）
      if (stateCheck.currentState === ForecastTrainingState.AWAITING_EXECUTION) {
        const recResult = await runHyperparamRecommendation(args, pythonPath, trainingDir, ctx)
        if (recResult?.status === 'success') {
          const recMsg = formatRecommendationMessage(recResult, { datasetShapeKey: 'spatial_shape', datasetShapeLabel: '空间分辨率' })
          if (recMsg) {
            prompt.message = `${prompt.message}\n\n${recMsg}`
          }
          prompt.data = {
            ...(prompt.data ?? {}),
            hyperparameter_recommendations: recResult,
          }
        }
      }
      return {
        status: prompt.status,
        message: prompt.message,
        canExecute: prompt.canExecute,
        ...prompt.data
      }
    }

    // ===== 3. PASS 阶段：执行训练 =====
    // Read from merged workflow params (session-confirmed values) to ensure
    // user-confirmed device_ids etc. are not overwritten by raw args defaults.
    const mergedForExec = workflow.getParams()
    const dataset_root = mergedForExec.dataset_root ?? args.dataset_root
    const log_dir = mergedForExec.log_dir ?? args.log_dir
    const model_name = mergedForExec.model_name ?? args.model_name
    const dyn_vars = mergedForExec.dyn_vars ?? args.dyn_vars
    const mode = mergedForExec.mode ?? args.mode ?? 'train'
    const device_ids = mergedForExec.device_ids ?? args.device_ids ?? [0]
    const distribute = mergedForExec.distribute ?? args.distribute ?? false
    const distribute_mode = mergedForExec.distribute_mode ?? args.distribute_mode ?? 'DDP'
    const ckpt_path = mergedForExec.ckpt_path ?? args.ckpt_path

    if (!log_dir) {
      return {
        status: 'error',
        error: '未指定训练日志输出目录 (log_dir)',
        suggestion: '请在参数中提供 log_dir'
      }
    }

    // ===== predict 快速通道：跳过训练专属步骤（OOM 等），直接准备 + 启动 =====
    if (mode === 'predict') {
      if (!dataset_root) {
        return { status: 'error', error: '需要 dataset_root', suggestion: '请提供预处理数据根目录' }
      }
      if (!model_name) {
        return { status: 'error', error: '需要 model_name', suggestion: '请提供模型名称' }
      }

      const normalizedDeviceIds = Array.isArray(device_ids) && device_ids.length > 0 ? device_ids : [0]

      // 准备工作空间
      const workspaceDir = path.resolve(log_dir, '_ocean_forecast_code')
      const prepareScript = path.join(trainingDir, 'prepare_workspace.py')
      const prepareResult = await ctx.sandbox.exec(
        `"${shellEscapeDouble(pythonPath)}" "${shellEscapeDouble(prepareScript)}" --source_dir "${shellEscapeDouble(trainingDir)}" --target_dir "${shellEscapeDouble(workspaceDir)}" --model_name "${shellEscapeDouble(model_name)}" --data_name ocean_forecast_npy`,
        { timeoutMs: 60000 }
      )
      if (prepareResult.code !== 0) {
        return {
          status: 'error',
          error: `工作空间准备失败: ${prepareResult.stderr}`,
          suggestion: `请检查输出目录 ${log_dir} 是否存在且有写入权限`
        }
      }
      const prepareInfo = JSON.parse(prepareResult.stdout)

      // 生成配置（predict 最小参数集）
      const predictMergedParams = workflow.getParams()
      const generateScript = path.join(workspaceDir, 'generate_config.py')
      const configParams: Record<string, unknown> = {
        model_name, dataset_root, dyn_vars, log_dir,
        in_t: predictMergedParams.in_t ?? 7,
        out_t: predictMergedParams.out_t ?? 1,
        stride: predictMergedParams.stride ?? 1,
        device: normalizedDeviceIds[0], device_ids: normalizedDeviceIds,
        distribute: false, distribute_mode: 'single',
        ckpt_path: ckpt_path || path.join(log_dir, 'best_model.pth'),
        epochs: 1, batch_size: 1, eval_batch_size: 1,
        use_amp: predictMergedParams.use_amp ?? true,
        gradient_checkpointing: false,
        normalize: predictMergedParams.normalize, normalizer_type: predictMergedParams.normalizer_type,
      }
      const configPath = path.join(workspaceDir, `${model_name}_config.yaml`)
      const paramsJson = JSON.stringify(configParams)
      const genResult = await ctx.sandbox.exec(
        `"${shellEscapeDouble(pythonPath)}" "${shellEscapeDouble(generateScript)}" --params '${shellSafeJson(paramsJson)}' --output "${shellEscapeDouble(configPath)}"`,
        { timeoutMs: 60000 }
      )
      if (genResult.code !== 0) {
        return {
          status: 'error',
          error: `配置生成失败: ${genResult.stderr}`,
          suggestion: '请检查 dataset_root 路径是否正确，以及 model_name 是否在支持列表中'
        }
      }
      const genInfo = JSON.parse(genResult.stdout)

      // 构建命令（predict 始终单卡）
      const cudaDevice = String(normalizedDeviceIds[0])
      const mainPy = path.join(workspaceDir, 'main.py')
      const cmdPath = pythonPath
      const cmdArgs = [mainPy, '--mode', 'predict', '--config', configPath]
      const cmdEnv = { CUDA_VISIBLE_DEVICES: cudaDevice }

      // 启动后台进程
      const processInfo = await trainingProcessManager.startProcess({
        cmd: cmdPath,
        args: cmdArgs,
        cwd: workspaceDir,
        logDir: log_dir,
        env: cmdEnv,
        metadata: {
          modelName: model_name,
          datasetRoot: dataset_root,
          logDir: log_dir,
          configPath: genInfo.config_path,
          workspaceDir: workspaceDir,
          deviceIds: normalizedDeviceIds,
          mode: 'predict',
        },
      })

      // 等待 predict_start 事件
      const STARTUP_TIMEOUT_MS = 300000
      const startupResult = await trainingProcessManager.waitForEvent(
        processInfo.id, 'predict_start', STARTUP_TIMEOUT_MS
      )

      if (startupResult.processStatus === 'failed' || startupResult.processStatus === 'killed') {
        const failedInfo = trainingProcessManager.getProcess(processInfo.id)
        return {
          status: 'error',
          error: '预测推理在启动阶段崩溃（数据加载/模型加载失败）',
          process_id: processInfo.id,
          error_summary: failedInfo?.errorSummary ?? null,
          error_log_tail: (await trainingProcessManager.readLogs(processInfo.id, { tail: 50 }))?.content,
          suggestions: failedInfo?.errorSummary?.suggestions ?? [],
        }
      }

      const predictionsDir = path.join(log_dir, 'predictions')
      if (startupResult.found) {
        return {
          status: 'started',
          message: '预测推理已启动。使用 ocean_forecast_train_status 监控进度。',
          process_id: processInfo.id,
          pid: processInfo.pid,
          mode: 'predict',
          model: model_name,
          config_path: genInfo.config_path,
          log_dir,
          log_file: processInfo.logFile,
          predictions_dir: predictionsDir,
          workspace_dir: workspaceDir,
          workspace_info: prepareInfo,
          next_steps: [
            `调用 ocean_forecast_train_status({ action: "wait", process_id: "${processInfo.id}", timeout: 300 }) 等待推理完成`,
            `调用 ocean_forecast_train_status({ process_id: "${processInfo.id}" }) 查看推理状态`,
            `调用 ocean_forecast_train_status({ action: "logs", process_id: "${processInfo.id}", tail: 50 }) 查看最新日志`,
            `推理完成后调用 ocean_forecast_train_visualize({ log_dir: "${log_dir}", mode: "predict" }) 生成可视化`,
          ],
        }
      }

      return {
        status: 'started',
        message: '预测进程已启动，仍在初始化中（可能数据量较大）。使用 ocean_forecast_train_status 监控。',
        process_id: processInfo.id,
        pid: processInfo.pid,
        mode: 'predict',
        model: model_name,
        config_path: genInfo.config_path,
        log_dir,
        log_file: processInfo.logFile,
        predictions_dir: predictionsDir,
        workspace_dir: workspaceDir,
        workspace_info: prepareInfo,
        next_steps: [
          `调用 ocean_forecast_train_status({ action: "wait", process_id: "${processInfo.id}", timeout: 300 }) 等待推理完成`,
          `调用 ocean_forecast_train_status({ process_id: "${processInfo.id}" }) 查看推理状态`,
        ],
      }
    }

    // ===== 3.0 模型支持性检查（若模型未接入，提前阻断） =====
    let modelSupportInfo: ModelInfo | undefined
    const listScript = path.join(trainingDir, 'list_models.py')
    const listResult = await ctx.sandbox.exec(
      `"${shellEscapeDouble(pythonPath)}" "${shellEscapeDouble(listScript)}"`,
      { timeoutMs: 30000 }
    )
    if (listResult.code === 0) {
      try {
        const parsed = JSON.parse(listResult.stdout)
        if (Array.isArray(parsed.models)) {
          modelSupportInfo = parsed.models.find((m: ModelInfo) => m.name === model_name)
        }
      } catch {
        modelSupportInfo = undefined
      }
    }
    if (modelSupportInfo && modelSupportInfo.supported === false) {
      return {
        status: 'error',
        error: '模型 ' + model_name + ' 未接入训练流程',
        reason: modelSupportInfo.notes ?? modelSupportInfo.description,
        suggestion: '请改用已接入的模型，或补齐模型注册 / trainer / 配置模板后再试'
      }
    }
    if (listResult.code === 0 && !modelSupportInfo) {
      return {
        status: 'error',
        error: '未知模型: ' + model_name,
        suggestion: '请从模型列表中选择，或确认模型名称是否拼写正确'
      }
    }

    const normalizedDeviceIds = Array.isArray(device_ids) && device_ids.length > 0 ? device_ids : [0]
    const effectiveDistribute = distribute && normalizedDeviceIds.length > 1
    const effectiveDistributeMode = effectiveDistribute ? distribute_mode : 'single'

    const execWarnings: string[] = []

    let masterPort: number | null = null
    if (effectiveDistribute && distribute_mode === 'DDP') {
      const requestedPort = typeof args.master_port === 'number' ? Math.trunc(args.master_port) : null
      if (requestedPort && requestedPort > 0 && requestedPort <= 65535) {
        if (await isPortFree(requestedPort)) {
          masterPort = requestedPort
        } else {
          const fallbackPort = await findFreePort(29500, 29600)
          masterPort = fallbackPort ?? requestedPort
          execWarnings.push(`DDP master_port ${requestedPort} 已被占用，已切换为 ${masterPort}。`)
        }
      } else {
        const fallbackPort = await findFreePort(29500, 29600)
        masterPort = fallbackPort ?? 29500
        if (masterPort !== 29500) {
          execWarnings.push(`DDP master_port 自动选择为 ${masterPort}。`)
        }
      }
    }

    if (distribute && normalizedDeviceIds.length <= 1) {
      execWarnings.push(
        '已请求多卡/DP/DDP 但 device_ids 只有 1 张 GPU，已自动降级为单卡训练以避免 DDP 初始化失败。'
      )
    }

    // ===== 3a. 准备训练工作空间（只复制所选模型相关代码） =====
    const workspaceDir = path.resolve(log_dir, '_ocean_forecast_code')
    const prepareScript = path.join(trainingDir, 'prepare_workspace.py')
    const prepareResult = await ctx.sandbox.exec(
      `"${shellEscapeDouble(pythonPath)}" "${shellEscapeDouble(prepareScript)}" --source_dir "${shellEscapeDouble(trainingDir)}" --target_dir "${shellEscapeDouble(workspaceDir)}" --model_name "${shellEscapeDouble(model_name)}" --data_name ocean_forecast_npy`,
      { timeoutMs: 60000 }
    )
    if (prepareResult.code !== 0) {
      return {
        status: 'error',
        error: `工作空间准备失败: ${prepareResult.stderr}`,
        reason: '无法将训练代码复制到输出目录',
        suggestion: `请检查输出目录 ${log_dir} 是否存在且有写入权限`
      }
    }
    const prepareInfo = JSON.parse(prepareResult.stdout)

    const generateScript = path.join(workspaceDir, 'generate_config.py')

    // ===== 3b. 生成配置文件 =====
    const mergedParams = workflow.getParams()
    const effectiveUseAmp = mergedParams.use_amp ?? true
    const configParams: Record<string, unknown> = {
      model_name,
      dataset_root,
      dyn_vars,
      in_t: mergedParams.in_t ?? 7,
      out_t: mergedParams.out_t ?? 1,
      stride: mergedParams.stride ?? 1,
      log_dir,
      device: normalizedDeviceIds[0],
      device_ids: normalizedDeviceIds,
      distribute: effectiveDistribute,
      distribute_mode: effectiveDistributeMode,
      master_port: masterPort ?? undefined,
      ckpt_path,
      epochs: mergedParams.epochs,
      lr: mergedParams.lr,
      batch_size: mergedParams.batch_size,
      eval_batch_size: mergedParams.eval_batch_size,
      patience: mergedParams.patience,
      eval_freq: mergedParams.eval_freq,
      normalize: mergedParams.normalize,
      normalizer_type: mergedParams.normalizer_type,
      optimizer: mergedParams.optimizer,
      weight_decay: mergedParams.weight_decay,
      scheduler: mergedParams.scheduler,
      scheduler_step_size: mergedParams.scheduler_step_size,
      scheduler_gamma: mergedParams.scheduler_gamma,
      seed: mergedParams.seed,
      wandb: mergedParams.wandb,
      use_amp: effectiveUseAmp,
      gradient_checkpointing: mergedParams.gradient_checkpointing,
    }

    const configPath = path.join(workspaceDir, `${model_name}_config.yaml`)

    const paramsJson = JSON.stringify(configParams)
    const genResult = await ctx.sandbox.exec(
      `"${shellEscapeDouble(pythonPath)}" "${shellEscapeDouble(generateScript)}" --params '${shellSafeJson(paramsJson)}' --output "${shellEscapeDouble(configPath)}"`,
      { timeoutMs: 60000 }
    )

    if (genResult.code !== 0) {
      return {
        status: 'error',
        error: `配置生成失败: ${genResult.stderr}`,
        reason: '参数可能不兼容所选模型，或数据目录不可访问',
        suggestion: '请检查 dataset_root 路径是否正确，以及 model_name 是否在支持列表中'
      }
    }

    const genInfo = JSON.parse(genResult.stdout)

    // ===== 3c. OOM 自动防护：自动降低 batch_size（不可跳过） =====
    if (mode === 'train') {
      const estimateScript = path.join(workspaceDir, 'estimate_memory.py')
      const cudaDevice = normalizedDeviceIds[0]
      let currentBatchSize = (configParams.batch_size as number) ?? 4
      let currentAmp = (configParams.use_amp as boolean) ?? true
      const MAX_ATTEMPTS = 5

      for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
        // 每次调参后重新生成配置
        if (attempt > 0) {
          configParams.batch_size = currentBatchSize
          configParams.use_amp = currentAmp
          const regenJson = JSON.stringify(configParams)
          // eslint-disable-next-line no-await-in-loop
          const regenResult = await ctx.sandbox.exec(
            `"${shellEscapeDouble(pythonPath)}" "${shellEscapeDouble(generateScript)}" --params '${shellSafeJson(regenJson)}' --output "${shellEscapeDouble(configPath)}"`,
            { timeoutMs: 60000 }
          )
          if (regenResult.code !== 0) {
            execWarnings.push(`显存预估前重建配置失败，已跳过自动调参：${regenResult.stderr || regenResult.stdout}`)
            break
          }
        }

        // eslint-disable-next-line no-await-in-loop
        const estimateResult = await ctx.sandbox.exec(
          `cd "${shellEscapeDouble(workspaceDir)}" && CUDA_VISIBLE_DEVICES=${Number(cudaDevice) || 0} "${shellEscapeDouble(pythonPath)}" "${shellEscapeDouble(estimateScript)}" --config "${shellEscapeDouble(configPath)}" --device 0`,
          { timeoutMs: 120000 }
        )
        if (estimateResult.code !== 0) {
          execWarnings.push(`显存预估失败，已跳过自动调参并继续训练：${estimateResult.stderr || estimateResult.stdout}`)
          break
        }

        try {
          const mem = JSON.parse(estimateResult.stdout)
          if (mem.status === 'success' && mem.utilization_pct <= 85) {
            // 通过 → 跳出循环
            break
          }

          // OOM 或 >85%：仅降低 batch_size（forecast 没有 patch_size 可缩减）
          if (!currentAmp) {
            // 先尝试开启 AMP
            currentAmp = true
          } else if (currentBatchSize > 1) {
            currentBatchSize = Math.max(1, Math.floor(currentBatchSize / 2))
          } else {
            // 所有手段耗尽
            return {
              status: 'error',
              error: 'GPU 显存不足，已尝试所有自动优化手段仍无法适配',
              memory_estimate: mem,
              applied_optimizations: { use_amp: currentAmp, batch_size: currentBatchSize },
              recommendations: mem.recommendations,
              suggestion: '请使用更大显存的 GPU，或减少 in_t / out_t / dyn_vars 数量'
            }
          }
        } catch {
          // 解析失败不阻止训练
          execWarnings.push('显存预估输出解析失败，已跳过自动调参并继续训练')
          break
        }
      }
    }

    // ===== 3d. 构建运行命令 =====
    let cmdPath: string
    let cmdArgs: string[]
    let cmdEnv: Record<string, string> = {}

    if (effectiveDistribute && distribute_mode === 'DDP') {
      const nproc = normalizedDeviceIds.length
      const cudaDevices = normalizedDeviceIds.join(',')
      const mainDdp = path.join(workspaceDir, 'main_ddp.py')
      cmdPath = pythonPath
      cmdArgs = ['-m', 'torch.distributed.run', `--nproc_per_node=${nproc}`, `--master_port=${masterPort ?? 29500}`, mainDdp, '--mode', mode, '--config', configPath]
      cmdEnv = { CUDA_VISIBLE_DEVICES: cudaDevices, MASTER_PORT: String(masterPort ?? 29500) }
    } else if (effectiveDistribute && distribute_mode === 'DP') {
      const mainPy = path.join(workspaceDir, 'main.py')
      cmdPath = pythonPath
      cmdArgs = [mainPy, '--mode', mode, '--config', configPath]
      // DP 直接使用用户选择的物理 GPU 编号，避免被单卡 CUDA_VISIBLE_DEVICES 限制
      cmdEnv = {}
    } else {
      const cudaDevice = String(normalizedDeviceIds[0])
      const mainPy = path.join(workspaceDir, 'main.py')
      cmdPath = pythonPath
      cmdArgs = [mainPy, '--mode', mode, '--config', configPath]
      cmdEnv = { CUDA_VISIBLE_DEVICES: cudaDevice }
    }

    // ===== 3e. 启动后台训练进程 =====
    const processInfo = await trainingProcessManager.startProcess({
      cmd: cmdPath,
      args: cmdArgs,
      cwd: workspaceDir,
      logDir: log_dir,
      env: cmdEnv,
      metadata: {
        modelName: model_name,
        datasetRoot: dataset_root,
        logDir: log_dir,
        configPath: genInfo.config_path,
        workspaceDir: workspaceDir,
        deviceIds: normalizedDeviceIds,
      },
    })

    // ===== 3f. 等待训练启动成功（事件驱动） =====
    const STARTUP_TIMEOUT_MS = 300000  // 5 分钟（数据加载可能很久）
    const startupResult = await trainingProcessManager.waitForEvent(
      processInfo.id, 'training_start', STARTUP_TIMEOUT_MS
    )

    if (startupResult.processStatus === 'failed' || startupResult.processStatus === 'killed') {
      // 启动阶段崩溃 → 直接返回错误
      const failedInfo = trainingProcessManager.getProcess(processInfo.id)
      return {
        status: 'error',
        error: '训练在启动阶段崩溃（数据加载/模型构建失败）',
        process_id: processInfo.id,
        error_summary: failedInfo?.errorSummary ?? null,
        error_log_tail: (await trainingProcessManager.readLogs(processInfo.id, { tail: 50 }))?.content,
        suggestions: failedInfo?.errorSummary?.suggestions ?? [],
      }
    }

    // ===== 3g. 生成 Jupyter Notebook（训练成功启动后） =====
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const metadataNotebookPath = (ctx.agent as any)?.config?.metadata?.notebookPath as string | undefined
    const notebookPath = metadataNotebookPath
      ? path.resolve(metadataNotebookPath)
      : path.resolve(ctx.sandbox.workDir, `${path.basename(ctx.sandbox.workDir)}.ipynb`)
    try {
      const nbParams: ForecastTrainNotebookParams = {
        logDir: log_dir,
        datasetRoot: dataset_root ?? '',
        modelName: model_name ?? '',
        configPath: genInfo.config_path ?? configPath,
        workspaceDir,
        pythonPath,
        deviceIds: normalizedDeviceIds,
        distribute: effectiveDistribute,
        distributeMode: effectiveDistributeMode,
        masterPort: masterPort ?? undefined,
        mode,
        inT: mergedParams.in_t,
        outT: mergedParams.out_t,
        stride: mergedParams.stride,
        dynVars: mergedParams.dyn_vars,
        epochs: mergedParams.epochs,
        lr: mergedParams.lr,
        batchSize: mergedParams.batch_size,
        evalBatchSize: mergedParams.eval_batch_size,
        patience: mergedParams.patience,
        evalFreq: mergedParams.eval_freq,
        normalize: mergedParams.normalize,
        normalizerType: mergedParams.normalizer_type,
        optimizer: mergedParams.optimizer,
        weightDecay: mergedParams.weight_decay,
        scheduler: mergedParams.scheduler,
        schedulerStepSize: mergedParams.scheduler_step_size,
        schedulerGamma: mergedParams.scheduler_gamma,
        seed: mergedParams.seed,
        useAmp: effectiveUseAmp,
        gradientCheckpointing: mergedParams.gradient_checkpointing,
        ckptPath: ckpt_path,
        wandb: mergedParams.wandb,
      }
      const cells = generateForecastTrainCells(nbParams)
      await saveOrAppendNotebook(ctx, notebookPath, cells)
    } catch (e) {
      console.warn('Notebook 生成失败:', e)
    }

    // 公共基础响应
    const baseResponse = {
      status: 'started',
      process_id: processInfo.id,
      pid: processInfo.pid,
      mode,
      model: model_name,
      config_path: genInfo.config_path,
      log_dir,
      log_file: processInfo.logFile,
      notebook_path: notebookPath,
      distribute: effectiveDistribute,
      distribute_mode: effectiveDistributeMode,
      device_ids: normalizedDeviceIds,
      master_port: masterPort ?? undefined,
      workspace_dir: workspaceDir,
      workspace_info: prepareInfo,
      warnings: execWarnings.length > 0 ? execWarnings : undefined,
    }

    if (startupResult.found) {
      return {
        ...baseResponse,
        message: '训练已启动并正常运行中。使用 ocean_forecast_train_status 工具监控进度。',
        next_steps: [
          `调用 ocean_forecast_train_status({ action: "wait", process_id: "${processInfo.id}", timeout: 120 }) 等待训练状态变化`,
          `调用 ocean_forecast_train_status({ action: "watch", process_id: "${processInfo.id}", timeout: 300 }) 等待关键推送事件`,
          `调用 ocean_forecast_train_status({ process_id: "${processInfo.id}" }) 查看训练状态`,
          `调用 ocean_forecast_train_status({ action: "logs", process_id: "${processInfo.id}", tail: 50 }) 查看最新日志`,
          `调用 ocean_forecast_train_status({ action: "kill", process_id: "${processInfo.id}" }) 终止训练`,
        ],
      }
    }

    return {
      ...baseResponse,
      message: '训练进程已启动，仍在初始化中（可能数据量较大）。使用 ocean_forecast_train_status 监控。',
      next_steps: [
        `调用 ocean_forecast_train_status({ action: "wait", process_id: "${processInfo.id}", timeout: 120 }) 等待训练状态变化`,
        `调用 ocean_forecast_train_status({ action: "watch", process_id: "${processInfo.id}", timeout: 300 }) 等待关键推送事件`,
        `调用 ocean_forecast_train_status({ process_id: "${processInfo.id}" }) 查看训练状态`,
        `调用 ocean_forecast_train_status({ action: "logs", process_id: "${processInfo.id}", tail: 50 }) 查看最新日志`,
      ],
    }
  }
})
