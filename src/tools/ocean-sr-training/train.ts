/**
 * @file train.ts
 *
 * @description 海洋超分辨率模型训练工具
 *              集成状态机实现分阶段确认流程
 *              支持后台执行和实时日志流
 * @author Leizheng
 * @contributors kongzhiquan
 * @date 2026-02-06
 * @version 3.0.0
 *
 * @changelog
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
 *     - trainingDir 改为 scripts/ocean_SR_training_masked
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
  TrainingWorkflow,
  TrainingState,
  type DatasetValidationInfo,
  type GpuInfo,
  type ModelInfo,
} from './workflow-state'

export const oceanSrTrainTool = defineTool({
  name: 'ocean_sr_train',
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

**GPU 模式**：
- 单卡：device_ids 长度为 1
- 多卡 DP：distribute=true, distribute_mode="DP"
- 多卡 DDP（推荐）：distribute=true, distribute_mode="DDP"`,

  params: {
    dataset_root: {
      type: 'string',
      description: '预处理数据根目录（ocean-preprocess 输出目录）',
      required: false
    },
    log_dir: {
      type: 'string',
      description: '训练日志输出目录',
      required: false
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
      description: '运行模式: "train" 或 "test"',
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
      default: 32
    },
    eval_batch_size: {
      type: 'number',
      description: '评估 batch size',
      required: false,
      default: 32
    },
    device_ids: {
      type: 'array',
      items: { type: 'number' },
      description: '使用的 GPU 列表（如 [0, 1, 2, 3]）。必须由用户确认。',
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
      description: '是否启用 AMP 混合精度训练（减少约 40-50% 显存）',
      required: false,
      default: false
    },
    gradient_checkpointing: {
      type: 'boolean',
      description: '是否启用 Gradient Checkpointing（减少约 60% 激活显存，增加约 30% 计算时间）',
      required: false,
      default: false
    },
    patch_size: {
      type: 'number',
      description: 'HR Patch 裁剪尺寸（如 64, 128），设置后训练时随机裁剪小区域而非全图训练。必须能被 scale 整除。',
      required: false
    },
    skip_memory_check: {
      type: 'boolean',
      description: '跳过训练前显存预估',
      required: false,
      default: false
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
    const pythonPath = findPythonWithModule('torch') || findFirstPythonPath()
    if (!pythonPath) {
      throw new Error('未找到可用的 Python 解释器（需要安装 torch）')
    }

    const trainingDir = path.resolve(process.cwd(), 'scripts/ocean_SR_training_masked')
    args.log_dir = args.log_dir ? path.resolve(ctx.sandbox.workDir, args.log_dir) : undefined
    args.dataset_root = args.dataset_root ? path.resolve(ctx.sandbox.workDir, args.dataset_root) : undefined
    // ===== 1. 构建工作流参数 =====
    const workflowParams = { ...args }
    const workflow = new TrainingWorkflow(workflowParams)
    const stateCheck = workflow.determineCurrentState()

    // ===== 2. 如果未到 PASS 阶段，收集上下文信息并返回提示 =====
    if (stateCheck.currentState !== TrainingState.PASS) {
      const context: {
        datasetInfo?: DatasetValidationInfo
        gpuInfo?: GpuInfo
        modelList?: ModelInfo[]
      } = {}

      // 如果有 dataset_root，验证数据目录
      if (args.dataset_root) {
        const validateScript = path.join(trainingDir, 'validate_dataset.py')
        const validateResult = await ctx.sandbox.exec(
          `"${pythonPath}" "${validateScript}" --dataset_root "${args.dataset_root}"`,
          { timeoutMs: 60000 }
        )
        if (validateResult.code === 0) {
          context.datasetInfo = JSON.parse(validateResult.stdout)
        } else {
          context.datasetInfo = {
            status: 'error',
            dataset_root: args.dataset_root,
            dyn_vars: [],
            scale: null,
            hr_shape: null,
            lr_shape: null,
            splits: {},
            has_static: false,
            static_vars: [],
            total_samples: { hr: 0, lr: 0 },
            warnings: [],
            errors: [`验证脚本执行失败: ${validateResult.stderr}`]
          }
        }
      }

      // 阶段2+需要模型列表
      if (stateCheck.currentState === TrainingState.AWAITING_MODEL_SELECTION) {
        const listScript = path.join(trainingDir, 'list_models.py')
        const listResult = await ctx.sandbox.exec(
          `"${pythonPath}" "${listScript}"`,
          { timeoutMs: 30000 }
        )
        if (listResult.code === 0) {
          const parsed = JSON.parse(listResult.stdout)
          context.modelList = parsed.models
        }
      }

      // 阶段3+需要 GPU 信息
      if (
        stateCheck.currentState === TrainingState.AWAITING_PARAMETERS ||
        stateCheck.currentState === TrainingState.AWAITING_EXECUTION
      ) {
        const gpuScript = path.join(trainingDir, 'check_gpu.py')
        const gpuResult = await ctx.sandbox.exec(
          `"${pythonPath}" "${gpuScript}"`,
          { timeoutMs: 30000 }
        )
        if (gpuResult.code === 0) {
          context.gpuInfo = JSON.parse(gpuResult.stdout)
        }
      }

      const prompt = workflow.getStagePrompt(context)
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
      // 排除状态机参数
      user_confirmed: _uc,
      confirmation_token: _ct,
      ...restParams
    } = args

    if (!log_dir) {
      return {
        status: 'error',
        error: '未指定训练日志输出目录 (log_dir)',
        suggestion: '请在参数中提供 log_dir'
      }
    }

    // ===== 3a. 准备训练工作空间（只复制所选模型相关代码） =====
    // 训练在副本上执行，保持 Agent SDK 源码不被修改；
    // Agent 运行时如需调整代码，可直接修改副本而不影响 SDK；
    // 切换模型时会自动清理旧模型代码并替换为新模型代码
    const workspaceDir = path.resolve(log_dir, '_ocean_sr_code')
    const prepareScript = path.join(trainingDir, 'prepare_workspace.py')
    const prepareResult = await ctx.sandbox.exec(
      `"${pythonPath}" "${prepareScript}" --source_dir "${trainingDir}" --target_dir "${workspaceDir}" --model_name "${model_name}"`,
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
    const configParams = {
      model_name,
      dataset_root,
      dyn_vars,
      scale,
      log_dir,
      device: device_ids[0],
      device_ids,
      distribute,
      distribute_mode,
      ...restParams
    }

    const configPath = path.join(workspaceDir, `${model_name}_config.yaml`)

    const paramsJson = JSON.stringify(configParams)
    const genResult = await ctx.sandbox.exec(
      `"${pythonPath}" "${generateScript}" --params '${paramsJson}' --output "${configPath}"`,
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

    // ===== 3c. 训练前显存预估 =====
    if (!args.skip_memory_check && mode === 'train') {
      const estimateScript = path.join(workspaceDir, 'estimate_memory.py')
      const cudaDevice = device_ids[0]
      const estimateResult = await ctx.sandbox.exec(
        `cd "${workspaceDir}" && CUDA_VISIBLE_DEVICES=${cudaDevice} "${pythonPath}" "${estimateScript}" --config "${configPath}" --device 0`,
        { timeoutMs: 120000 }
      )

      if (estimateResult.code === 0) {
        try {
          const memoryEstimate = JSON.parse(estimateResult.stdout)
          if (memoryEstimate.status === 'oom') {
            return {
              status: 'error',
              error: 'GPU 显存不足（预估阶段已检测到 OOM）',
              memory_estimate: memoryEstimate,
              recommendations: memoryEstimate.recommendations,
              config_path: configPath,
              workspace_dir: workspaceDir,
              suggestion: memoryEstimate.recommendations.join('\n')
            }
          }
          // 记录预估信息（非 OOM 时不阻止训练）
          if (memoryEstimate.status === 'success') {
            const pct = memoryEstimate.utilization_pct
            if (pct > 90) {
              // 高风险但不阻止，只记录警告
              console.warn(`[Memory] 显存使用率 ${pct}%，训练中可能因波动 OOM`)
            }
          }
        } catch {
          // 解析失败不阻止训练
        }
      }
    }

    // ===== 3d. 构建运行命令 =====
    // 注：代码快照由 Python 的 main.py / main_ddp.py 在训练开始前自动保存到 saving_path/code/
    let cmdPath: string
    let cmdArgs: string[]
    let cmdEnv: Record<string, string> = {}

    if (distribute && distribute_mode === 'DDP' && device_ids.length > 1) {
      const nproc = device_ids.length
      const cudaDevices = device_ids.join(',')
      const mainDdp = path.join(workspaceDir, 'main_ddp.py')
      cmdPath = pythonPath
      cmdArgs = ['-m', 'torch.distributed.run', `--nproc_per_node=${nproc}`, mainDdp, '--mode', mode, '--config', configPath]
      cmdEnv = { CUDA_VISIBLE_DEVICES: cudaDevices }
    } else {
      const cudaDevice = String(device_ids[0])
      const mainPy = path.join(workspaceDir, 'main.py')
      cmdPath = pythonPath
      cmdArgs = [mainPy, '--mode', mode, '--config', configPath]
      cmdEnv = { CUDA_VISIBLE_DEVICES: cudaDevice }
    }

    // ===== 3e. 启动后台训练进程 =====
    const processInfo = trainingProcessManager.startProcess({
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
      },
    })

    // ===== 3f. 返回启动信息（不等待训练完成） =====
    return {
      status: 'started',
      message: `训练已在后台启动。使用 ocean_sr_train_status 工具查询进度和日志。`,
      process_id: processInfo.id,
      pid: processInfo.pid,
      mode,
      model: model_name,
      config_path: genInfo.config_path,
      log_dir,
      log_file: processInfo.logFile,
      distribute: distribute && device_ids.length > 1,
      distribute_mode: distribute ? distribute_mode : 'single',
      device_ids,
      workspace_dir: workspaceDir,
      workspace_info: prepareInfo,
      next_steps: [
        `调用 ocean_sr_train_status({ process_id: "${processInfo.id}" }) 查看训练状态`,
        `调用 ocean_sr_train_status({ process_id: "${processInfo.id}", tail: 50 }) 查看最新日志`,
        `调用 ocean_sr_train_status({ action: "kill", process_id: "${processInfo.id}" }) 终止训练`,
      ],
    }
  }
})
