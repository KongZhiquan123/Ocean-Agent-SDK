/**
 * @file train.ts
 *
 * @description 海洋超分辨率模型训练工具
 *              集成状态机实现分阶段确认流程
 *              支持后台执行和实时日志流
 * @author Leizheng
 * @contributors kongzhiquan, Leizheng
 * @date 2026-02-09
 * @version 5.1.0
 *
 * @changelog
 *   - 2026-03-13 kongzhiquan: v5.1.0 提取模型常量/形状函数到 train-model-config.ts，
 *     提取验证/警告函数到 train-validators.ts（纯重构，行为不变）
 *   - 2026-03-12 kongzhiquan: v5.0.0 适配 workflow.ts 函数式重构
 *     - 将 new TrainingWorkflow() 替换为 mergeParams() + resolveStage()
 *     - TrainingState 常量替换为字符串字面量
 *   - 2026-02-26 kongzhiquan: v4.8.1 Notebook 路径改用后端传入的 notebookPath（从 agent metadata 读取）
 *   - 2026-02-25 kongzhiquan: v4.8.0 训练成功启动后生成可复现 Jupyter Notebook
 *     - 在 PASS 阶段 status=started 时调用 generateTrainCells + saveOrAppendNotebook
 *     - Notebook 保存至 {log_dir}/{basename}.ipynb
 *     - 包含评估、推理、完整训练命令等 cells
 *   - 2026-02-25 Leizheng: v4.7.0 AWAITING_EXECUTION 阶段集成超参数推荐
 *     - 调用 recommend_hyperparams.py 实测显存 + 数据集扫描
 *     - 推荐 batch_size / epochs / lr，并附数据频谱分析说明
 *     - 失败时静默跳过，不影响现有训练流程
 *   - 2026-02-25 Leizheng: v4.6.0 session 文件持久化用户确认参数
 *     - AWAITING_EXECUTION 时将全量参数保存到 {log_dir}/.ocean_sr_session.json
 *     - PASS 执行时读取 session 文件作为 sessionOverrides 传入 TrainingWorkflow
 *     - 彻底解决可选参数（normalizer_type 等）在无状态多轮调用中丢失的问题
 *     - use_amp 自动设置不纳入 definedArgs，保留用户通过 session 指定的值
 *   - 2026-02-24 Leizheng: v4.5.0 configParams 改用 workflow.getParams()
 *     - 解决无状态工作流中用户确认参数（如 normalizer_type）在执行调用时丢失的问题
 *     - 执行阶段从 workflow 合并参数取值，不再依赖原始 args（可能缺失字段）
 *   - 2026-02-11 Leizheng: v4.4.0 predict 快速通道
 *     - mode 参数支持 "predict"，跳过训练工作流（OOM/shape/FFT 检查）
 *     - predict 分支直接准备工作空间 → 生成配置 → 启动 → 等待 predict_start
 *     - 启动事件根据 mode 选择 predict_start 或 training_start
 *   - 2026-02-11 Leizheng: v4.3.0 ResShift divisor 修正 + TOKEN_INVALID GPU 信息
 *     - ResShift divisor 8→64（Swin window_size=8）
 *     - TOKEN_INVALID 状态下也获取 GPU 信息供重新确认
 *   - 2026-02-09 Leizheng: v4.2.7 FFT 模型 AMP 默认策略修复 + 模型支持性预检
 *   - 2026-02-09 Leizheng: v4.2.6 默认 batch_size 下调为 4 + 默认开启 gradient_checkpointing
 *   - 2026-02-09 Leizheng: v4.2.5 训练前输出尺寸预检 + 默认 batch_size 下调
 *     - 训练启动前检查模型输出尺寸是否与 HR 匹配
 *     - batch_size/eval_batch_size 默认降为 16
 *   - 2026-02-09 Leizheng: v4.2.4 FFT + AMP 推荐 patch_size 提示
 *   - 2026-02-09 Leizheng: v4.2.3 FFT 模型 AMP 默认策略
 *     - FFT 模型默认关闭 AMP（用户显式开启则尊重并提示风险）
 *   - 2026-02-09 Leizheng: v4.2.2 FFT 提示策略调整 + 显存预警
 *     - FFT 预检扩展到所有模型，改为提示不拦截
 *     - 新增 GPU 低空闲显存提示（不拦截）
 *   - 2026-02-09 Leizheng: v4.2.1 FFT + AMP 兼容性预检扩展
 *     - 适配 HiNOTE 等 FFT 模型，统一前置拦截
 *     - 提示与错误信息统一为 FFT 维度要求
 *   - 2026-02-09 Leizheng: v4.2.0 FNO + AMP 兼容性预检
 *     - 训练前检测 FFT 输入尺寸是否为 2 的幂
 *     - 在阶段提示中追加明确告警，避免运行时 cuFFT 崩溃
 *   - 2026-02-08 Leizheng: v4.1.0 修复参数传递 + 显式白名单
 *     - configParams 改用显式白名单，避免 restParams 泄漏状态机内部字段
 *     - 显式传递 ckpt_path（之前被解构后从 restParams 中丢失）
 *   - 2026-02-07 kongzhiquan: v4.0.0 OOM 自动防护 + 事件驱动启动监控
 *     - 显存预估改为自动循环调参（AMP→减batch_size→报错），不可跳过
 *     - 移除 skip_memory_check 参数，use_amp 默认改为 true
 *     - 启动后等待 training_start 事件（最长 5 分钟），捕获早期崩溃
 *     - 启动阶段崩溃时直接返回 error_summary + suggestions
 *   - 2026-02-07 kongzhiquan: v3.0.0 后台执行模式
 *     - 使用 TrainingProcessManager 启动后台训练进程
 *     - 训练启动后立即返回 process_id，不再阻塞等待
 *     - 支持实时日志流（通过 ocean_sr_train_status 工具查询）
 *     - 服务器关闭时自动清理训练进程
 *   - 2026-02-07 Leizheng: v3.0.0 OOM 防护三件套
 *     - 新增 use_amp / gradient_checkpointing / patch_size 参数
 *     - 训练前自动运行显存预估，OOM 提前拦截并给出建议
 *     - 新参数通过 generate_config.py 写入 YAML 配置
 *   - 2026-02-07 Leizheng: v2.3.0 按模型选择性复制代码到用户输出目录执行，保持 SDK 源码不被修改
 *   - 2026-02-07 Leizheng: v2.2.0 使用 findPythonWithModule('torch') 自动查找带 PyTorch 的 Python
 *   - 2026-02-06 Leizheng: v2.1.0 指向 masked 版本训练框架
 *     - trainingDir 改为 scripts/ocean-SR-training-masked
 *   - 2026-02-06 Leizheng: v2.0.0 集成训练工作流状态机
 *     - 4 阶段确认: 数据 → 模型 → 参数(GPU) → 执行
 *     - 自动检测 dyn_vars / scale / shape
 *     - GPU 信息集成到参数确认阶段
 *     - Token 防跳步机制
 *   - 2026-02-06 Leizheng: v1.0.0 初始版本
 *     - 支持单卡/多卡(DP/DDP)训练
 *     - 自动生成 YAML 配置文件
 *     - 支持 train/test 两种模式
 */

import { defineTool } from '@shareai-lab/kode-sdk'
import { findPythonWithModule, findFirstPythonPath } from '@/utils/python-manager'
import { trainingProcessManager } from '@/utils/training-process-manager'
import path from 'node:path'
import {
  mergeParams,
  resolveStage,
  type SrTrainingWorkflowParams,
  type SrDatasetInfo,
  type SrGpuInfo,
  type SrModelInfo,
} from './workflow'
import { generateTrainCells, saveOrAppendNotebook } from './notebook'
import { shellEscapeDouble, shellSafeJson, extractTaggedJson } from '@/utils/shell'
import { isPortFree, findFreePort } from '@/utils/port'
import { loadSessionParams, saveSessionParams, formatRecommendationMessage } from '@/utils/training-utils'
import {
  isFftSensitiveModel,
  isAmpDefaultOffModel,
  FFT_AMP_SENSITIVE_MODELS,
} from './train-model-config'
import {
  buildFftAmpIncompatibility,
  buildOomPreWarning,
  validateDataset,
  runHyperparamRecommendation,
} from './train-validators'

export const oceanSrTrainStartTool = defineTool({
  name: 'ocean_sr_train_start',
  description: `执行海洋超分辨率模型训练或测试。

**分阶段确认流程**（每阶段必须等待用户确认）：
1. 确认数据目录和输出目录（自动检测变量和 scale）
2. 选择训练模型
3. 确认训练参数（包括 GPU 选择）
4. 最终确认执行

**首次调用**：只传 dataset_root 和 log_dir，工具会自动检测数据并展示信息
**逐步补充参数**：每次调用补充该阶段需要的参数，直到所有阶段通过
**最终执行**：传入 user_confirmed=true 和 confirmation_token 后启动后台训练

**后台执行模式**：
- 训练启动后立即返回 process_id，不会阻塞等待训练完成
- 使用 ocean_sr_train_status 工具查询训练状态和实时日志
- 服务器关闭时会自动终止训练进程

**训练模式 (mode=train)**：执行完整训练流程，包含验证和早停
**测试模式 (mode=test)**：加载最佳模型，在测试集上评估
**预测模式 (mode=predict)**：加载模型对测试集执行全图 SR 推理，输出 NPY 文件（跳过训练工作流）

**GPU 模式**：
- 单卡：device_ids 长度为 1
- 多卡 DP：distribute=true, distribute_mode="DP"
- 多卡 DDP（推荐）：distribute=true, distribute_mode="DDP"`,

  params: {
    dataset_root: {
      type: 'string',
      description: '预处理数据根目录（ocean-SR-data-preprocess 输出目录）'
    },
    log_dir: {
      type: 'string',
      description: '训练日志输出目录'
    },
    model_name: {
      type: 'string',
      description: '模型名称（如 SwinIR, FNO2d, DDPM 等）',
      required: false
    },
    dyn_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '动态变量列表（如 ["temp", "salt"]）。如不提供，将从数据目录自动检测并要求确认。',
      required: false
    },
    scale: {
      type: 'number',
      description: '超分辨率倍数。如不提供，将从数据目录自动推算并要求确认。',
      required: false
    },
    mode: {
      type: 'string',
      description: '运行模式: "train", "test" 或 "predict"（predict 跳过训练工作流，直接推理）',
      enum: ['train', 'test', 'predict'],
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
      enum: ['DP', 'DDP'],
      required: false,
      default: 'DDP'
    },
    master_port: {
      type: 'number',
      description: 'DDP 主端口（可选，端口冲突时可指定）',
      required: false
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
    normalize: {
      type: 'boolean',
      description: '是否归一化',
      required: false,
      default: true
    },
    normalizer_type: {
      type: 'string',
      description: '归一化类型: "PGN" 或 "GN"',
      enum: ['PGN', 'GN'],
      required: false,
      default: 'PGN'
    },
    optimizer: {
      type: 'string',
      description: '优化器: "AdamW", "Adam", "SGD"',
      enum: ['AdamW', 'Adam', 'SGD'],
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
      enum: ['StepLR', 'MultiStepLR', 'OneCycleLR'],
      required: false,
      default: 'StepLR'
    },
    scheduler_step_size: {
      type: 'number',
      description: '调度器步长',
      required: false,
      default: 300
    },
    scheduler_gamma: {
      type: 'number',
      description: '调度器衰减率',
      required: false,
      default: 0.5
    },
    seed: {
      type: 'number',
      description: '随机种子',
      required: false,
      default: 42
    },
    wandb: {
      type: 'boolean',
      description: '是否启用 WandB 日志',
      required: false,
      default: false
    },
    use_amp: {
      type: 'boolean',
      description: '是否启用 AMP 混合精度训练（减少约 40-50% 显存；默认：非 FFT 开启 / FFT 频域模型关闭）',
      required: false
    },
    gradient_checkpointing: {
      type: 'boolean',
      description: '是否启用 Gradient Checkpointing（减少约 60% 激活显存，增加约 30% 计算时间，默认开启）',
      required: false,
      default: true
    },
    patch_size: {
      type: 'number',
      description: 'HR Patch 裁剪尺寸（如 64, 128），设置后训练时随机裁剪小区域而非全图训练。必须能被 scale 整除。',
      required: false
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

    const trainingDir = path.resolve(process.cwd(), 'scripts/ocean-SR-training-masked')

    const userSpecifiedUseAmp = Object.prototype.hasOwnProperty.call(args, 'use_amp')
    const ampDefaultOff = isAmpDefaultOffModel(args.model_name as string | undefined)
    let autoDisabledAmp = false

    if (!userSpecifiedUseAmp && args.model_name) {
      if (ampDefaultOff) {
        args.use_amp = false
        autoDisabledAmp = true
      } else {
        args.use_amp = true
      }
    }

    // ===== 1. 构建工作流参数（合并 session 缓存，防止可选参数跨调用丢失） =====
    const SESSION_FILENAME = '.ocean_sr_train_session.json' as const
    // use_amp 若非用户显式传入，则从 args 中移除，避免自动设置值覆盖 session 中用户的选择
    const workflowArgs = { ...args }
    if (!userSpecifiedUseAmp) {
      delete workflowArgs.use_amp
    }
    const sessionParams = await loadSessionParams<SrTrainingWorkflowParams>(args.log_dir, SESSION_FILENAME, ctx)
    const mergedForWorkflow = mergeParams(workflowArgs, sessionParams ?? undefined)
    const stageResult = resolveStage(mergedForWorkflow)

    // ===== 2. 如果未到 PASS 阶段，收集上下文信息并返回提示 =====
    if (stageResult !== null) {
      const context: {
        datasetInfo?: SrDatasetInfo
        gpuInfo?: SrGpuInfo
        modelList?: SrModelInfo[]
      } = {}

      context.datasetInfo = await validateDataset(args.dataset_root, pythonPath, trainingDir, ctx)

      // 根据当前阶段收集额外上下文
      if (stageResult.status === 'awaiting_model_selection') {
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

      if (
        stageResult.status === 'awaiting_parameters' ||
        stageResult.status === 'awaiting_execution' ||
        stageResult.status === 'token_invalid'
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

      // 有上下文时重新调用 resolveStage 以获取含上下文的提示
      const prompt = Object.keys(context).length > 0
        ? resolveStage(mergedForWorkflow, context) ?? stageResult
        : stageResult
      const fnoAmpWarning = buildFftAmpIncompatibility({
        model_name: args.model_name,
        use_amp: args.use_amp ?? true,
        datasetInfo: context.datasetInfo,
        scale: args.scale,
        patch_size: args.patch_size ?? null,
      })
      if (fnoAmpWarning) {
        prompt.message = `${prompt.message}\n\n${fnoAmpWarning.message}`
        prompt.data = {
          ...(prompt.data ?? {}),
          fft_amp_warning: fnoAmpWarning.details,
          fno_amp_warning: fnoAmpWarning.details,
        }
      }

      if (args.use_amp === true && isAmpDefaultOffModel(args.model_name) && !isFftSensitiveModel(args.model_name)) {
        prompt.message = `${prompt.message}\n\n⚠️ 检测到模型 ${args.model_name} 默认关闭 AMP，但当前 use_amp=true 可能导致数值不稳定（如 NaN）。建议 use_amp=false；如需强行开启请自行承担风险。`
      }

      if (autoDisabledAmp) {
        prompt.message = `${prompt.message}\n\n检测到模型 ${args.model_name} 属于 FFT/数值敏感模型，已默认关闭 AMP（use_amp=false）。如需强制开启，请明确设置 use_amp=true。`
        prompt.data = {
          ...(prompt.data ?? {}),
          amp_auto_disabled: { model: args.model_name },
        }
      }

      if (
        args.model_name &&
        FFT_AMP_SENSITIVE_MODELS.has(args.model_name as string) &&
        userSpecifiedUseAmp &&
        args.use_amp === true
      ) {
        prompt.message = `${prompt.message}\n\n你已手动启用 AMP（强烈不建议）。FFT 类模型可能触发 cuFFT 尺寸限制错误，如仍要继续请确认。`
        prompt.data = {
          ...(prompt.data ?? {}),
          amp_user_override: { model: args.model_name },
        }
      }

      const oomWarning = buildOomPreWarning({
        gpuInfo: context.gpuInfo,
        device_ids: args.device_ids,
      })
      if (oomWarning) {
        prompt.message = `${prompt.message}\n\n${oomWarning.message}`
        prompt.data = {
          ...(prompt.data ?? {}),
          oom_warning: oomWarning.details,
        }
      }
      
      if (prompt.status.startsWith('awaiting')) {
        await saveSessionParams(args.log_dir, SESSION_FILENAME, mergedForWorkflow, ctx)
      }

      // AWAITING_EXECUTION 时运行超参数推荐（实测显存 + 数据集分析）
      if (prompt.status === 'awaiting_execution') {
        const recResult = await runHyperparamRecommendation(args, pythonPath, trainingDir, ctx)
        if (recResult?.status === 'success') {
          const recMsg = formatRecommendationMessage(recResult, { datasetShapeKey: 'hr_shape', datasetShapeLabel: 'HR' })
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
    const {
      dataset_root,
      log_dir,
      model_name,
      dyn_vars,
      scale,
      mode = 'train',
      device_ids = [0],
      distribute = false,
      distribute_mode = 'DDP',
      ckpt_path,
    } = args

    // ===== predict 快速通道：跳过训练专属步骤（OOM/shape/FFT），直接准备 + 启动 =====
    if (mode === 'predict') {
      const normalizedDeviceIds = Array.isArray(device_ids) && device_ids.length > 0 ? device_ids : [0]

      // 准备工作空间
      const workspaceDir = path.resolve(log_dir, '_ocean_sr_code')
      const prepareScript = path.join(trainingDir, 'prepare_workspace.py')
      const prepareResult = await ctx.sandbox.exec(
        `"${shellEscapeDouble(pythonPath)}" "${shellEscapeDouble(prepareScript)}" --source_dir "${shellEscapeDouble(trainingDir)}" --target_dir "${shellEscapeDouble(workspaceDir)}" --model_name "${shellEscapeDouble(model_name)}"`,
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

      // 生成配置（predict 最小参数集，normalizer_type 从 mergedParams 获取以保留用户选择）
      const predictMergedParams = mergedForWorkflow
      const generateScript = path.join(workspaceDir, 'generate_config.py')
      const configParams: Record<string, unknown> = {
        model_name, dataset_root, dyn_vars, scale, log_dir,
        device: normalizedDeviceIds[0], device_ids: normalizedDeviceIds,
        distribute: false, distribute_mode: 'single',
        ckpt_path: ckpt_path || path.join(log_dir, 'best_model.pth'),
        epochs: 1, batch_size: 1, eval_batch_size: 1,
        use_amp: predictMergedParams.use_amp ?? (ampDefaultOff ? false : true),
        gradient_checkpointing: false,
        patch_size: predictMergedParams.patch_size,
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
          message: '预测推理已启动。使用 ocean_sr_train_status 监控进度。',
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
            `调用 ocean_sr_train_status({ action: "wait", process_id: "${processInfo.id}", timeout: 300 }) 等待推理完成`,
            `调用 ocean_sr_train_status({ process_id: "${processInfo.id}" }) 查看推理状态`,
            `调用 ocean_sr_train_status({ action: "logs", process_id: "${processInfo.id}", tail: 50 }) 查看最新日志`,
            `推理完成后调用 ocean_sr_train_visualize({ log_dir: "${log_dir}", mode: "predict" }) 生成可视化`,
          ],
        }
      }

      return {
        status: 'started',
        message: '预测进程已启动，仍在初始化中（可能数据量较大）。使用 ocean_sr_train_status 监控。',
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
          `调用 ocean_sr_train_status({ action: "wait", process_id: "${processInfo.id}", timeout: 300 }) 等待推理完成`,
          `调用 ocean_sr_train_status({ process_id: "${processInfo.id}" }) 查看推理状态`,
        ],
      }
    }

    // ===== 3.0 模型支持性检查（若模型未接入，提前阻断） =====
    let modelSupportInfo: SrModelInfo | undefined
    const listScript = path.join(trainingDir, 'list_models.py')
    const listResult = await ctx.sandbox.exec(
      `"${shellEscapeDouble(pythonPath)}" "${shellEscapeDouble(listScript)}"`,
      { timeoutMs: 30000 }
    )
    if (listResult.code === 0) {
      try {
        const parsed = JSON.parse(listResult.stdout)
        if (Array.isArray(parsed.models)) {
          modelSupportInfo = parsed.models.find((m: SrModelInfo) => m.name === model_name)
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

    // ===== 3.1 FFT + AMP 兼容性预检（提示，不拦截） =====
    let fftAmpWarningAtStart: ReturnType<typeof buildFftAmpIncompatibility> | null = null
    if (isFftSensitiveModel(model_name) && (args.use_amp ?? true)) {
      const datasetInfo = await validateDataset(dataset_root, pythonPath, trainingDir, ctx)
      fftAmpWarningAtStart = buildFftAmpIncompatibility({
        model_name,
        use_amp: args.use_amp ?? true,
        datasetInfo,
        scale,
        patch_size: args.patch_size ?? null,
      })
    }
    const execWarnings: string[] = []
    if (fftAmpWarningAtStart) {
      execWarnings.push(
        `FFT + AMP 可能不兼容：LR ${fftAmpWarningAtStart.details.lr_height}×${fftAmpWarningAtStart.details.lr_width} 不是 2 的幂，建议 use_amp=false 或调整尺寸。`
      )
    }

    const normalizedDeviceIds = Array.isArray(device_ids) && device_ids.length > 0 ? device_ids : [0]
    const effectiveDistribute = distribute && normalizedDeviceIds.length > 1
    const effectiveDistributeMode = effectiveDistribute ? distribute_mode : 'single'

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
    // 训练在副本上执行，保持 Agent SDK 源码不被修改；
    // Agent 运行时如需调整代码，可直接修改副本而不影响 SDK；
    // 切换模型时会自动清理旧模型代码并替换为新模型代码
    const workspaceDir = path.resolve(log_dir, '_ocean_sr_code')
    const prepareScript = path.join(trainingDir, 'prepare_workspace.py')
    const prepareResult = await ctx.sandbox.exec(
      `"${shellEscapeDouble(pythonPath)}" "${shellEscapeDouble(prepareScript)}" --source_dir "${shellEscapeDouble(trainingDir)}" --target_dir "${shellEscapeDouble(workspaceDir)}" --model_name "${shellEscapeDouble(model_name)}"`,
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
    // 使用合并后参数（用户传入值 > 默认值），
    // 避免因 Agent 后续调用未携带某字段而丢失用户确认过的参数
    const mergedParams = mergedForWorkflow
    // use_amp 回落：若 session 和当前调用均未明确指定，则使用模型自动计算值
    const effectiveUseAmp = mergedParams.use_amp ?? (ampDefaultOff ? false : true)
    const configParams: Record<string, unknown> = {
      model_name,
      dataset_root,
      dyn_vars,
      scale,
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
      patch_size: mergedParams.patch_size,
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
    if (genInfo?.eval_batchsize_clamped) {
      const requested = genInfo.eval_batchsize_requested ?? args.eval_batch_size
      const applied = genInfo.eval_batchsize ?? 4
      execWarnings.push(
        `扩散模型评估显存开销大，eval_batch_size 已从 ${requested} 限制为 ${applied}（上限 4）`
      )
    }

    // ===== 3b.1 训练前模型输出尺寸预检 =====
    if (mode === 'train') {
      const shapeCheckScript = path.join(workspaceDir, 'check_output_shape.py')
      const deviceId = normalizedDeviceIds[0] ?? 0
      const shapeResult = await ctx.sandbox.exec(
        `"${shellEscapeDouble(pythonPath)}" "${shellEscapeDouble(shapeCheckScript)}" --config "${shellEscapeDouble(configPath)}" --device ${Number(deviceId) || 0}`,
        { timeoutMs: 120000 }
      )

      if (shapeResult.code !== 0) {
        // 优先从 stdout 提取 tagged JSON（stderr 可能只含 FutureWarning 等无关警告）
        const shapeInfoOnError = extractTaggedJson(shapeResult.stdout, 'shape_check')
        if (shapeInfoOnError) {
          return {
            status: 'error',
            error: shapeInfoOnError.error ?? `输出尺寸预检失败 (exit code ${shapeResult.code})`,
            reason: shapeInfoOnError.reason ?? '无法完成模型输出尺寸检查',
            details: shapeInfoOnError.details,
            suggestion: '请检查模型配置或数据目录是否可用'
          }
        }
        // stdout 无结构化输出时才回退到 stderr
        return {
          status: 'error',
          error: `输出尺寸预检失败: ${shapeResult.stderr || shapeResult.stdout}`,
          reason: '无法完成模型输出尺寸检查',
          suggestion: '请检查模型配置或数据目录是否可用'
        }
      }

      const shapeInfo = extractTaggedJson(shapeResult.stdout, 'shape_check')
      if (shapeInfo && shapeInfo.status === 'error') {
        return {
          status: 'error',
          error: shapeInfo.error ?? '模型输出尺寸与 HR 不匹配',
          reason: shapeInfo.reason ?? '模型输出尺寸与目标尺寸不一致',
          details: shapeInfo.details,
          suggestion:
            '请检查 scale/upsample_factor 配置，或调整 patch_size/模型参数使输出与 HR 对齐'
        }
      }
    }

    // ===== 3c. 自动显存预估 + 自动调参（不可跳过） =====
    if (mode === 'train') {
      const estimateScript = path.join(workspaceDir, 'estimate_memory.py')
      const cudaDevice = normalizedDeviceIds[0]
      let currentBatchSize = (configParams.batch_size as number) ?? 4
      let currentAmp = (configParams.use_amp as boolean) ?? true
      const allowAutoEnableAmp = !ampDefaultOff || args.use_amp === true
      const MAX_ATTEMPTS = 5

      for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
        // 每次调参后重新生成配置
        if (attempt > 0) {
          configParams.batch_size = currentBatchSize
          configParams.use_amp = currentAmp
          const regenJson = JSON.stringify(configParams)
          const regenResult = await ctx.sandbox.exec(
            `"${shellEscapeDouble(pythonPath)}" "${shellEscapeDouble(generateScript)}" --params '${shellSafeJson(regenJson)}' --output "${shellEscapeDouble(configPath)}"`,
            { timeoutMs: 60000 }
          )
          if (regenResult.code !== 0) {
            execWarnings.push(`显存预估前重建配置失败，已跳过自动调参：${regenResult.stderr || regenResult.stdout}`)
            break
          }
        }

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

          // OOM 或 >85%：依次降级
          if (!currentAmp) {
            if (allowAutoEnableAmp) {
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
                suggestion: '请使用更大显存的 GPU，或手动设置更小的 patch_size'
              }
            }
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
              suggestion: '请使用更大显存的 GPU，或手动设置更小的 patch_size'
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
    // 注：代码快照由 Python 的 main.py / main_ddp.py 在训练开始前自动保存到 saving_path/code/
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
    const metadataNotebookPath = (ctx.agent as any)?.config?.metadata?.notebookPath as string | undefined
    const notebookPath = metadataNotebookPath
      ? path.resolve(metadataNotebookPath)
      : path.resolve(ctx.sandbox.workDir, `${path.basename(ctx.sandbox.workDir)}.ipynb`)
    try {
      const cells = generateTrainCells({
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
        scale: mergedParams.scale,
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
        patchSize: mergedParams.patch_size,
        ckptPath: ckpt_path,
        wandb: mergedParams.wandb,
      })
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
      amp_auto_disabled: autoDisabledAmp ? { model: args.model_name } : undefined,
      warnings: execWarnings.length > 0 ? execWarnings : undefined,
      fft_amp_warning: fftAmpWarningAtStart?.details,
    };

    if (startupResult.found) {
      return {
        ...baseResponse,
        message: '训练已启动并正常运行中。使用 ocean_sr_train_status 工具监控进度。',
        next_steps: [
          `调用 ocean_sr_train_status({ action: "wait", process_id: "${processInfo.id}", timeout: 120 }) 等待训练状态变化`,
          `调用 ocean_sr_train_status({ action: "watch", process_id: "${processInfo.id}", timeout: 300 }) 等待关键推送事件`,
          `调用 ocean_sr_train_status({ process_id: "${processInfo.id}" }) 查看训练状态`,
          `调用 ocean_sr_train_status({ action: "logs", process_id: "${processInfo.id}", tail: 50 }) 查看最新日志`,
          `调用 ocean_sr_train_status({ action: "kill", process_id: "${processInfo.id}" }) 终止训练`,
        ],
      };
    }

    return {
      ...baseResponse,
      message: '训练进程已启动，仍在初始化中（可能数据量较大）。使用 ocean_sr_train_status 监控。',
      next_steps: [
        `调用 ocean_sr_train_status({ action: "wait", process_id: "${processInfo.id}", timeout: 120 }) 等待训练状态变化`,
        `调用 ocean_sr_train_status({ action: "watch", process_id: "${processInfo.id}", timeout: 300 }) 等待关键推送事件`,
        `调用 ocean_sr_train_status({ process_id: "${processInfo.id}" }) 查看训练状态`,
        `调用 ocean_sr_train_status({ action: "logs", process_id: "${processInfo.id}", tail: 50 }) 查看最新日志`,
      ],
    };
  }
})
