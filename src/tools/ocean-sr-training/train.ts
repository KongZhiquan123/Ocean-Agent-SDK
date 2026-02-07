/**
 * @file train.ts
 *
 * @description 海洋超分辨率模型训练工具
 *              集成状态机实现分阶段确认流程
 * @author Leizheng
 * @date 2026-02-06
 * @version 2.3.0
 *
 * @changelog
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
**最终执行**：传入 user_confirmed=true 和 confirmation_token 后执行训练

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

    // ===== 3c. 构建运行命令 =====
    // 注：代码快照由 Python 的 main.py / main_ddp.py 在训练开始前自动保存到 saving_path/code/
    let runCmd: string

    if (distribute && distribute_mode === 'DDP' && device_ids.length > 1) {
      const nproc = device_ids.length
      const cudaDevices = device_ids.join(',')
      const mainDdp = path.join(workspaceDir, 'main_ddp.py')
      runCmd = `cd "${workspaceDir}" && CUDA_VISIBLE_DEVICES=${cudaDevices} "${pythonPath}" -m torch.distributed.run --nproc_per_node=${nproc} "${mainDdp}" --mode ${mode} --config "${configPath}"`
    } else {
      const cudaDevice = device_ids[0]
      const mainPy = path.join(workspaceDir, 'main.py')
      runCmd = `cd "${workspaceDir}" && CUDA_VISIBLE_DEVICES=${cudaDevice} "${pythonPath}" "${mainPy}" --mode ${mode} --config "${configPath}"`
    }

    // ===== 3d. 执行训练 =====
    const trainResult = await ctx.sandbox.exec(runCmd, {
      timeoutMs: 72 * 3600 * 1000  // 最长 72 小时
    })

    if (trainResult.code !== 0) {
      const stderr = trainResult.stderr || ''
      let reason = '训练过程中出现错误'
      let suggestion = '请查看完整错误日志'

      if (stderr.includes('CUDA out of memory')) {
        reason = 'GPU 显存不足'
        suggestion = '建议：1) 减小 batch_size  2) 使用多卡训练  3) 选择更轻量的模型'
      } else if (stderr.includes('No such file or directory')) {
        reason = '找不到数据文件或脚本'
        suggestion = '请检查 dataset_root 路径是否正确，以及数据是否完整'
      } else if (stderr.includes('NCCL')) {
        reason = 'GPU 多卡通信失败 (NCCL)'
        suggestion = '建议：1) 检查 GPU 是否都可用  2) 尝试 DP 模式替代 DDP  3) 减少 GPU 数量'
      } else if (stderr.includes('NaN')) {
        reason = '训练过程中出现 NaN（梯度爆炸）'
        suggestion = '建议：1) 降低学习率  2) 检查数据是否包含异常值  3) 尝试不同的归一化方式'
      } else if (stderr.includes('not implemented') || stderr.includes('NotImplementedError')) {
        reason = '模型或功能未实现'
        suggestion = '请调用 ocean_sr_list_models 确认模型名称是否正确'
      }

      return {
        status: 'error',
        error: stderr.slice(-2000),
        reason,
        suggestion,
        config_path: genInfo.config_path,
        workspace_dir: workspaceDir,
        stdout_tail: (trainResult.stdout || '').slice(-1000)
      }
    }

    // ===== 3e. 返回结果 =====
    return {
      status: 'success',
      mode,
      model: model_name,
      config_path: genInfo.config_path,
      log_dir: genInfo.config_path ? path.dirname(genInfo.config_path) : log_dir,
      distribute: distribute && device_ids.length > 1,
      distribute_mode: distribute ? distribute_mode : 'single',
      device_ids,
      workspace_dir: workspaceDir,
      workspace_info: prepareInfo,
      stdout_tail: (trainResult.stdout || '').slice(-2000),
      message: mode === 'train'
        ? `训练完成。最佳模型和代码快照保存在日志目录下`
        : `测试完成。请查看输出指标。`
    }
  }
})
