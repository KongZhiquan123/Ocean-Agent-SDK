/**
 * @file notebook.ts
 *
 * @description Ocean forecast training Jupyter Notebook generation module.
 *              Generates reproducible notebooks after training starts.
 *              Main path: load checkpoint for inference; appendix: full re-training.
 *
 * @author Leizheng
 * @date 2026-02-26
 * @version 1.0.0
 *
 * @changelog
 *   - 2026-02-26 Leizheng: v1.0.0 initial version for ocean forecast training
 */

import { toPyRepr, mdCell, codeCell, saveOrAppendNotebook } from '@/utils/notebook'
import type { NotebookCell } from '@/utils/notebook'

export { saveOrAppendNotebook }

// ========================================
// Parameter interface
// ========================================

export interface ForecastTrainNotebookParams {
  // Paths
  logDir: string
  datasetRoot: string
  modelName: string
  configPath: string
  workspaceDir: string
  pythonPath: string

  // GPU config
  deviceIds: number[]
  distribute: boolean
  distributeMode: string
  masterPort?: number

  // Training mode
  mode: string

  // Forecast-specific params (replaces scale/patch_size)
  inT?: number
  outT?: number
  stride?: number
  dynVars?: string[]

  // Training hyperparams
  epochs?: number
  lr?: number
  batchSize?: number
  evalBatchSize?: number
  patience?: number
  evalFreq?: number
  normalize?: boolean
  normalizerType?: string
  optimizer?: string
  weightDecay?: number
  scheduler?: string
  schedulerStepSize?: number
  schedulerGamma?: number
  seed?: number
  useAmp?: boolean
  gradientCheckpointing?: boolean
  ckptPath?: string
  wandb?: boolean
}

// ========================================
// Cell generation functions
// ========================================

function generateTitleCell(params: ForecastTrainNotebookParams): NotebookCell {
  const timestamp = new Date().toISOString().replace('T', ' ').slice(0, 19)
  const gpuMode = params.deviceIds.length > 1
    ? (params.distributeMode === 'DDP'
        ? `多卡 DDP (GPU ${params.deviceIds.join(',')})`
        : `多卡 DP (GPU ${params.deviceIds.join(',')})`)
    : `单卡 (GPU ${params.deviceIds[0] ?? 0})`

  return mdCell(
    `# 海洋时序预测训练 Notebook\n` +
    `\n` +
    `本 Notebook 记录了本次训练的完整配置，可用于推理、评估或重新训练。\n` +
    `运行 **Cell 1-3** 可重新生成配置文件，之后各 cell 均可独立运行。\n` +
    `\n` +
    `| 项目 | 值 |\n` +
    `|------|----|\n` +
    `| 模型 | \`${params.modelName}\` |\n` +
    `| 模式 | \`${params.mode}\` |\n` +
    `| GPU | ${gpuMode} |\n` +
    `| in_t / out_t / stride | ${params.inT ?? 7} / ${params.outT ?? 1} / ${params.stride ?? 1} |\n` +
    `| 变量 | ${params.dynVars?.join(', ') ?? '—'} |\n` +
    `| 生成时间 | ${timestamp} |\n` +
    `| 日志目录 | \`${params.logDir}\` |\n` +
    `| 数据目录 | \`${params.datasetRoot}\` |`
  )
}

/** Cell 1: Paths & GPU */
function generatePathCell(params: ForecastTrainNotebookParams): NotebookCell {
  const lines: string[] = [
    'import os',
    'import subprocess',
    'import sys',
    'import json',
    '',
    '# ====== 路径配置 ======',
    `LOG_DIR       = ${toPyRepr(params.logDir)}`,
    `DATASET_ROOT  = ${toPyRepr(params.datasetRoot)}`,
    `WORKSPACE_DIR = ${toPyRepr(params.workspaceDir)}`,
    `PYTHON_PATH   = ${toPyRepr(params.pythonPath)}`,
    '',
    '# CONFIG_PATH 由下方"生成配置文件" cell 写入',
    `CONFIG_PATH   = ${toPyRepr(params.configPath)}`,
    '',
    '# ====== GPU 配置 ======',
    `DEVICE_IDS      = ${toPyRepr(params.deviceIds)}`,
    `DISTRIBUTE      = ${toPyRepr(params.distribute)}`,
    `DISTRIBUTE_MODE = ${toPyRepr(params.distributeMode)}`,
    `MASTER_PORT     = ${toPyRepr(params.masterPort ?? 29500)}`,
    '',
    '# ====== 检查点路径 ======',
    `CHECKPOINT_PATH = ${toPyRepr(params.ckptPath ?? `${params.logDir}/best_model.pth`)}`,
  ]

  return codeCell(lines.join('\n'))
}

/** Cell 2: Forecast-specific hyperparams */
function generateHyperparamCell(params: ForecastTrainNotebookParams): NotebookCell {
  const lines: string[] = [
    '# ====== 模型与数据 ======',
    `MODEL_NAME = ${toPyRepr(params.modelName)}`,
    `DYN_VARS   = ${toPyRepr(params.dynVars)}`,
    '',
    '# ====== 时序参数（预报特有）======',
    `IN_T   = ${toPyRepr(params.inT ?? 7)}`,
    `OUT_T  = ${toPyRepr(params.outT ?? 1)}`,
    `STRIDE = ${toPyRepr(params.stride ?? 1)}`,
    '',
    '# ====== 训练核心参数 ======',
    `EPOCHS           = ${toPyRepr(params.epochs ?? 500)}`,
    `LR               = ${toPyRepr(params.lr ?? 0.001)}`,
    `BATCH_SIZE       = ${toPyRepr(params.batchSize ?? 4)}`,
    `EVAL_BATCH_SIZE  = ${toPyRepr(params.evalBatchSize ?? 4)}`,
    `PATIENCE         = ${toPyRepr(params.patience ?? 10)}`,
    `EVAL_FREQ        = ${toPyRepr(params.evalFreq ?? 5)}`,
    `SEED             = ${toPyRepr(params.seed ?? 42)}`,
    '',
    '# ====== 优化器与调度器 ======',
    `OPTIMIZER          = ${toPyRepr(params.optimizer ?? 'AdamW')}`,
    `WEIGHT_DECAY       = ${toPyRepr(params.weightDecay ?? 0.001)}`,
    `SCHEDULER          = ${toPyRepr(params.scheduler ?? 'StepLR')}`,
    `SCHEDULER_STEP     = ${toPyRepr(params.schedulerStepSize ?? 300)}`,
    `SCHEDULER_GAMMA    = ${toPyRepr(params.schedulerGamma ?? 0.5)}`,
    '',
    '# ====== 归一化 ======',
    `NORMALIZE       = ${toPyRepr(params.normalize ?? true)}`,
    `NORMALIZER_TYPE = ${toPyRepr(params.normalizerType ?? 'PGN')}`,
    '',
    '# ====== OOM 防护 ======',
    `USE_AMP    = ${toPyRepr(params.useAmp ?? true)}`,
    `GRAD_CKPT  = ${toPyRepr(params.gradientCheckpointing ?? true)}`,
    '',
    '# ====== 其他 ======',
    `WANDB = ${toPyRepr(params.wandb ?? false)}`,
  ]

  return codeCell(lines.join('\n'))
}

/** Cell 3: Generate config YAML using forecast generate_config.py */
function generateConfigCell(params: ForecastTrainNotebookParams): NotebookCell {
  const effectiveDistribute = params.distribute && params.deviceIds.length > 1
  const effectiveDistributeMode = effectiveDistribute ? params.distributeMode : 'single'

  return codeCell(
    `# 根据以上超参重新生成训练配置文件\n` +
    `config_params = {\n` +
    `    "model_name": MODEL_NAME,\n` +
    `    "dataset_root": DATASET_ROOT,\n` +
    `    "dyn_vars": DYN_VARS,\n` +
    `    "in_t": IN_T,\n` +
    `    "out_t": OUT_T,\n` +
    `    "stride": STRIDE,\n` +
    `    "log_dir": LOG_DIR,\n` +
    `    "device": DEVICE_IDS[0],\n` +
    `    "device_ids": DEVICE_IDS,\n` +
    `    "distribute": ${toPyRepr(effectiveDistribute)},\n` +
    `    "distribute_mode": ${toPyRepr(effectiveDistributeMode)},\n` +
    `    "master_port": MASTER_PORT,\n` +
    `    "epochs": EPOCHS,\n` +
    `    "lr": LR,\n` +
    `    "batch_size": BATCH_SIZE,\n` +
    `    "eval_batch_size": EVAL_BATCH_SIZE,\n` +
    `    "patience": PATIENCE,\n` +
    `    "eval_freq": EVAL_FREQ,\n` +
    `    "normalize": NORMALIZE,\n` +
    `    "normalizer_type": NORMALIZER_TYPE,\n` +
    `    "optimizer": OPTIMIZER,\n` +
    `    "weight_decay": WEIGHT_DECAY,\n` +
    `    "scheduler": SCHEDULER,\n` +
    `    "scheduler_step_size": SCHEDULER_STEP,\n` +
    `    "scheduler_gamma": SCHEDULER_GAMMA,\n` +
    `    "seed": SEED,\n` +
    `    "wandb": WANDB,\n` +
    `    "use_amp": USE_AMP,\n` +
    `    "gradient_checkpointing": GRAD_CKPT,\n` +
    `}\n` +
    `\n` +
    `generate_script = os.path.join(WORKSPACE_DIR, "generate_config.py")\n` +
    `result = subprocess.run(\n` +
    `    [PYTHON_PATH, generate_script,\n` +
    `     "--params", json.dumps(config_params),\n` +
    `     "--output", CONFIG_PATH],\n` +
    `    capture_output=True, text=True\n` +
    `)\n` +
    `if result.returncode == 0:\n` +
    `    info = json.loads(result.stdout)\n` +
    `    print(f"配置已写入: {info.get('config_path', CONFIG_PATH)}")\n` +
    `else:\n` +
    `    print("配置生成失败:", result.stderr)\n` +
    `    raise RuntimeError("无法生成训练配置，请检查参数")`
  )
}

function generateGpuCheckCell(): NotebookCell {
  return codeCell(
    `result = subprocess.run(\n` +
    `    [PYTHON_PATH, os.path.join(WORKSPACE_DIR, "..", "check_gpu.py")],\n` +
    `    capture_output=True, text=True\n` +
    `)\n` +
    `if result.returncode == 0:\n` +
    `    gpu_info = json.loads(result.stdout)\n` +
    `    if gpu_info.get("cuda_available"):\n` +
    `        for g in gpu_info.get("gpus", []):\n` +
    `            print(f"GPU {g['id']}: {g['name']} | 总计 {g['total_memory_gb']} GB | 可用 {g['free_memory_gb']} GB")\n` +
    `    else:\n` +
    `        print("未检测到可用 CUDA GPU")\n` +
    `else:\n` +
    `    print("GPU 信息获取失败:", result.stderr)`
  )
}

function buildSingleCmd(mode: string): string {
  return (
    `env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(DEVICE_IDS[0])}\n` +
    `result = subprocess.run(\n` +
    `    [PYTHON_PATH, os.path.join(WORKSPACE_DIR, "main.py"),\n` +
    `     "--mode", "${mode}", "--config", CONFIG_PATH],\n` +
    `    env=env, cwd=WORKSPACE_DIR\n` +
    `)\n` +
    `print("退出码:", result.returncode)`
  )
}

function buildDdpCmd(mode: string): string {
  return (
    `nproc = len(DEVICE_IDS)\n` +
    `cuda_devices = ",".join(map(str, DEVICE_IDS))\n` +
    `env = {**os.environ, "CUDA_VISIBLE_DEVICES": cuda_devices, "MASTER_PORT": str(MASTER_PORT)}\n` +
    `result = subprocess.run(\n` +
    `    [PYTHON_PATH, "-m", "torch.distributed.run",\n` +
    `     f"--nproc_per_node={nproc}", f"--master_port={MASTER_PORT}",\n` +
    `     os.path.join(WORKSPACE_DIR, "main.py"),\n` +
    `     "--mode", "${mode}", "--config", CONFIG_PATH],\n` +
    `    env=env, cwd=WORKSPACE_DIR\n` +
    `)\n` +
    `print("退出码:", result.returncode)`
  )
}

function generateEvalCells(params: ForecastTrainNotebookParams): NotebookCell[] {
  const isDdp = params.distribute && params.distributeMode === 'DDP' && params.deviceIds.length > 1
  const runCode = isDdp ? buildDdpCmd('test') : buildSingleCmd('test')

  return [
    mdCell(
      `## 评估（Test 模式）\n` +
      `\n` +
      `加载 \`best_model.pth\` 检查点，在测试集上计算 RMSE / MAE 等指标。\n` +
      `需要训练完成后才能运行。`
    ),
    codeCell(
      `assert os.path.exists(CHECKPOINT_PATH), f"检查点不存在: {CHECKPOINT_PATH}\\n请先完成训练。"\n` +
      `print(f"使用检查点: {CHECKPOINT_PATH}")\n` +
      `\n` +
      runCode
    ),
  ]
}

function generatePredictCells(params: ForecastTrainNotebookParams): NotebookCell[] {
  return [
    mdCell(
      `## 推理（Predict 模式）\n` +
      `\n` +
      `对测试集执行全量推理，输出预测 NPY 文件到 \`${params.logDir}/predictions/\`。`
    ),
    codeCell(
      `assert os.path.exists(CHECKPOINT_PATH), f"检查点不存在: {CHECKPOINT_PATH}\\n请先完成训练。"\n` +
      `print(f"使用检查点: {CHECKPOINT_PATH}")\n` +
      `\n` +
      `# predict 始终单卡\n` +
      `env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(DEVICE_IDS[0])}\n` +
      `result = subprocess.run(\n` +
      `    [PYTHON_PATH, os.path.join(WORKSPACE_DIR, "main.py"),\n` +
      `     "--mode", "predict", "--config", CONFIG_PATH],\n` +
      `    env=env, cwd=WORKSPACE_DIR\n` +
      `)\n` +
      `print("退出码:", result.returncode)\n` +
      `print(f"预测结果保存在: {os.path.join(LOG_DIR, 'predictions')}")`
    ),
  ]
}

function generateTrainCmdCell(params: ForecastTrainNotebookParams): NotebookCell[] {
  const isDdp = params.distribute && params.distributeMode === 'DDP' && params.deviceIds.length > 1
  const runCode = isDdp ? buildDdpCmd('train') : buildSingleCmd('train')

  return [
    mdCell(
      `## 附录：完整重新训练\n` +
      `\n` +
      `使用与工具调用完全相同的参数从头开始训练。\n` +
      `注意：这会覆盖已有的检查点，请确认后再运行。`
    ),
    codeCell(runCode),
  ]
}

function generateTrainVisualizeCells(): NotebookCell[] {
  return [
    mdCell(
      `## 训练可视化\n` +
      `\n` +
      `生成 loss / RMSE / MAE / 学习率 等图表，保存在 \`LOG_DIR/plots/\` 目录下。`
    ),
    codeCell(
      `plot_script = os.path.join(WORKSPACE_DIR, "..", "generate_training_plots.py")\n` +
      `result = subprocess.run(\n` +
      `    [PYTHON_PATH, plot_script, "--log_dir", LOG_DIR],\n` +
      `    capture_output=True, text=True\n` +
      `)\n` +
      `if result.returncode == 0:\n` +
      `    print("训练可视化图表已生成")\n` +
      `    plots_dir = os.path.join(LOG_DIR, "plots")\n` +
      `    if os.path.isdir(plots_dir):\n` +
      `        for f in sorted(os.listdir(plots_dir)):\n` +
      `            if f.endswith(".png"):\n` +
      `                print(f"  - {f}")\n` +
      `else:\n` +
      `    print("训练可视化失败:", result.stderr)`
    ),
  ]
}

function generatePredictVisualizeCells(params: ForecastTrainNotebookParams): NotebookCell[] {
  return [
    mdCell(
      `## 预测可视化\n` +
      `\n` +
      `从 \`predictions/\` 目录读取预测 NPY 文件，\n` +
      `对比真值数据生成 **Prediction / Ground Truth / Error** 对比图。`
    ),
    codeCell(
      `predict_plot_script = os.path.join(WORKSPACE_DIR, "..", "generate_predict_plots.py")\n` +
      `n_samples = 5\n` +
      `\n` +
      `cmd = [PYTHON_PATH, predict_plot_script, "--log_dir", LOG_DIR, "--n_samples", str(n_samples)]\n` +
      `if DATASET_ROOT:\n` +
      `    cmd.extend(["--dataset_root", DATASET_ROOT])\n` +
      `\n` +
      `result = subprocess.run(cmd, capture_output=True, text=True)\n` +
      `if result.returncode == 0:\n` +
      `    print("预测可视化图表已生成")\n` +
      `    plots_dir = os.path.join(LOG_DIR, "plots")\n` +
      `    if os.path.isdir(plots_dir):\n` +
      `        for f in sorted(os.listdir(plots_dir)):\n` +
      `            if f.startswith("predict_") and f.endswith(".png"):\n` +
      `                print(f"  - {f}")\n` +
      `else:\n` +
      `    print("预测可视化失败:", result.stderr)`
    ),
  ]
}

function generateCompletionCell(params: ForecastTrainNotebookParams): NotebookCell {
  return mdCell(
    `## 输出目录结构\n` +
    `\n` +
    `\`\`\`\n` +
    `${params.logDir}/\n` +
    `├── best_model.pth          # 最优检查点\n` +
    `├── last_model.pth          # 最后一个 epoch 的检查点\n` +
    `├── train.log               # 完整训练日志\n` +
    `├── config.yaml             # 训练配置备份\n` +
    `├── predictions/            # predict 模式输出的 NPY 文件\n` +
    `├── plots/                  # 可视化图表\n` +
    `└── _ocean_forecast_code/   # 训练代码快照\n` +
    `    ├── main.py\n` +
    `    └── ${params.modelName}_config.yaml\n` +
    `\`\`\``
  )
}

// ========================================
// Main export function
// ========================================

export function generateForecastTrainCells(params: ForecastTrainNotebookParams): NotebookCell[] {
  const cells: NotebookCell[] = []

  cells.push(generateTitleCell(params))

  // Cell 1: Paths & GPU
  cells.push(mdCell('## Cell 1：路径与 GPU 配置'))
  cells.push(generatePathCell(params))

  // Cell 2: Hyperparams
  cells.push(mdCell('## Cell 2：训练超参配置\n\n修改此 cell 后重新运行 Cell 3 即可更新配置。'))
  cells.push(generateHyperparamCell(params))

  // Cell 3: Generate config
  cells.push(mdCell(
    '## Cell 3：生成训练配置文件\n\n' +
    '将 Cell 2 中的超参写入 `CONFIG_PATH` 对应的 YAML 文件。\n' +
    '后续所有 cell 均依赖此配置文件，修改超参后须重新运行本 cell。'
  ))
  cells.push(generateConfigCell(params))

  // GPU check
  cells.push(mdCell('## GPU 环境检查'))
  cells.push(generateGpuCheckCell())

  // Evaluate
  cells.push(...generateEvalCells(params))

  // Training visualization
  cells.push(...generateTrainVisualizeCells())

  // Predict
  cells.push(...generatePredictCells(params))

  // Prediction visualization
  cells.push(...generatePredictVisualizeCells(params))

  // Appendix: re-train
  cells.push(...generateTrainCmdCell(params))

  // Output directory explanation
  cells.push(generateCompletionCell(params))

  return cells
}
