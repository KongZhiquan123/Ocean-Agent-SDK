/**
 * @file notebook.ts
 *
 * @description 海洋超分辨率训练 Jupyter Notebook 生成模块
 *              在训练成功启动后，生成可复现的 Notebook
 *              Notebook 使用 subprocess 调用 main.py（保持与训练框架一致）
 *              主路径：加载 checkpoint 做推理/评估；附录：完整重新训练命令
 *
 * @author kongzhiquan
 * @date 2026-02-25
 * @version 1.1.0
 *
 * @changelog
 *   - 2026-02-25 kongzhiquan: v1.2.0 新增可视化 cells
 *     - 训练可视化：调用 generate_training_plots.py 生成 loss/metrics/lr 等图表
 *     - 预测可视化：调用 generate_predict_plots.py 生成 SR 对比图
 *   - 2026-02-25 kongzhiquan: v1.1.0 超参变量实际参与配置生成，notebook 完全自包含
 *     - 拆分 setup cell：路径/GPU cell + 超参 cell
 *     - 新增"生成配置文件" cell：用超参调用 generate_config.py 产出 CONFIG_PATH
 *     - 所有 run cell 使用该 CONFIG_PATH，超参变量不再悬空
 *     - TrainNotebookParams 补全 patience/evalFreq/weightDecay/schedulerStepSize/schedulerGamma
 *   - 2026-02-25 kongzhiquan: v1.0.0 初始版本
 */

import { toPyRepr, mdCell, codeCell, saveOrAppendNotebook } from '@/utils/notebook'
import type { NotebookCell } from '@/utils/notebook'

export { saveOrAppendNotebook }

// ========================================
// 参数接口
// ========================================

export interface TrainNotebookParams {
  // 路径
  logDir: string
  datasetRoot: string
  modelName: string
  configPath: string
  workspaceDir: string
  pythonPath: string

  // GPU 配置
  deviceIds: number[]
  distribute: boolean
  distributeMode: string
  masterPort?: number

  // 训练模式
  mode: string

  // 训练超参（全部用于 generate_config.py，notebook 完全自包含）
  scale?: number
  dynVars?: string[]
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
  patchSize?: number | null
  ckptPath?: string
  wandb?: boolean
}

// ========================================
// Cell 生成函数
// ========================================

function generateTitleCell(params: TrainNotebookParams): NotebookCell {
  const timestamp = new Date().toISOString().replace('T', ' ').slice(0, 19)
  const gpuMode = params.deviceIds.length > 1
    ? (params.distributeMode === 'DDP'
        ? `多卡 DDP (GPU ${params.deviceIds.join(',')})`
        : `多卡 DP (GPU ${params.deviceIds.join(',')})`)
    : `单卡 (GPU ${params.deviceIds[0] ?? 0})`

  return mdCell(
    `# 海洋超分辨率训练 Notebook\n` +
    `\n` +
    `本 Notebook 记录了本次训练的完整配置，可用于推理、评估或重新训练。\n` +
    `运行 **Cell 1-3** 可重新生成配置文件，之后各 cell 均可独立运行。\n` +
    `\n` +
    `| 项目 | 值 |\n` +
    `|------|----|\n` +
    `| 模型 | \`${params.modelName}\` |\n` +
    `| 模式 | \`${params.mode}\` |\n` +
    `| GPU | ${gpuMode} |\n` +
    `| Scale | ${params.scale ?? '—'} |\n` +
    `| 生成时间 | ${timestamp} |\n` +
    `| 日志目录 | \`${params.logDir}\` |\n` +
    `| 数据目录 | \`${params.datasetRoot}\` |`
  )
}

/** Cell 1：路径与 GPU 配置（所有 run cell 都会用到） */
function generatePathCell(params: TrainNotebookParams): NotebookCell {
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
    '# CONFIG_PATH 由下方"生成配置文件" cell 写入，也可直接指向已有 YAML',
    `CONFIG_PATH   = ${toPyRepr(params.configPath)}`,
    '',
    '# ====== GPU 配置 ======',
    `DEVICE_IDS      = ${toPyRepr(params.deviceIds)}`,
    `DISTRIBUTE      = ${toPyRepr(params.distribute)}`,
    `DISTRIBUTE_MODE = ${toPyRepr(params.distributeMode)}`,
    `MASTER_PORT     = ${toPyRepr(params.masterPort ?? 29500)}`,
    '',
    '# ====== 检查点路径（训练完成后生成）======',
    `CHECKPOINT_PATH = ${toPyRepr(params.ckptPath ?? `${params.logDir}/best_model.pth`)}`,
  ]

  return codeCell(lines.join('\n'))
}

/** Cell 2：训练超参配置（供下方 generate_config.py cell 使用） */
function generateHyperparamCell(params: TrainNotebookParams): NotebookCell {
  const lines: string[] = [
    '# ====== 模型与数据 ======',
    `MODEL_NAME = ${toPyRepr(params.modelName)}`,
    `SCALE      = ${toPyRepr(params.scale)}`,
    `DYN_VARS   = ${toPyRepr(params.dynVars)}`,
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
    `NORMALIZE      = ${toPyRepr(params.normalize ?? true)}`,
    `NORMALIZER_TYPE = ${toPyRepr(params.normalizerType ?? 'PGN')}`,
    '',
    '# ====== OOM 防护 ======',
    `USE_AMP    = ${toPyRepr(params.useAmp ?? true)}`,
    `GRAD_CKPT  = ${toPyRepr(params.gradientCheckpointing ?? true)}`,
    `PATCH_SIZE = ${toPyRepr(params.patchSize ?? null)}`,
    '',
    '# ====== 其他 ======',
    `WANDB = ${toPyRepr(params.wandb ?? false)}`,
  ]

  return codeCell(lines.join('\n'))
}

/** Cell 3：调用 generate_config.py 生成 YAML，CONFIG_PATH 从此真正可用 */
function generateConfigCell(params: TrainNotebookParams): NotebookCell {
  const isDdp = params.distribute && params.distributeMode === 'DDP' && params.deviceIds.length > 1
  const effectiveDistribute = params.distribute && params.deviceIds.length > 1
  const effectiveDistributeMode = effectiveDistribute ? params.distributeMode : 'single'

  return codeCell(
    `# 根据以上超参重新生成训练配置文件\n` +
    `# 修改 Cell 2 中的变量后重新运行此 cell，即可使新参数生效\n` +
    `config_params = {\n` +
    `    "model_name": MODEL_NAME,\n` +
    `    "dataset_root": DATASET_ROOT,\n` +
    `    "dyn_vars": DYN_VARS,\n` +
    `    "scale": SCALE,\n` +
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
    `    "patch_size": PATCH_SIZE,\n` +
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
    `    raise RuntimeError("无法生成训练配置，请检查 WORKSPACE_DIR 和参数是否正确")`
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

/** 生成单卡/DP 的 subprocess 调用代码 */
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

/** 生成 DDP 的 subprocess 调用代码 */
function buildDdpCmd(mode: string): string {
  return (
    `nproc = len(DEVICE_IDS)\n` +
    `cuda_devices = ",".join(map(str, DEVICE_IDS))\n` +
    `env = {**os.environ, "CUDA_VISIBLE_DEVICES": cuda_devices, "MASTER_PORT": str(MASTER_PORT)}\n` +
    `result = subprocess.run(\n` +
    `    [PYTHON_PATH, "-m", "torch.distributed.run",\n` +
    `     f"--nproc_per_node={nproc}", f"--master_port={MASTER_PORT}",\n` +
    `     os.path.join(WORKSPACE_DIR, "main_ddp.py"),\n` +
    `     "--mode", "${mode}", "--config", CONFIG_PATH],\n` +
    `    env=env, cwd=WORKSPACE_DIR\n` +
    `)\n` +
    `print("退出码:", result.returncode)`
  )
}

function generateEvalCells(params: TrainNotebookParams): NotebookCell[] {
  const isDdp = params.distribute && params.distributeMode === 'DDP' && params.deviceIds.length > 1
  const runCode = isDdp ? buildDdpCmd('test') : buildSingleCmd('test')

  return [
    mdCell(
      `## 评估（Test 模式）\n` +
      `\n` +
      `加载 \`best_model.pth\` 检查点，在测试集上计算 RMSE / SSIM 等指标。\n` +
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

function generatePredictCells(params: TrainNotebookParams): NotebookCell[] {
  return [
    mdCell(
      `## 推理（Predict 模式）\n` +
      `\n` +
      `对测试集执行全图 SR 推理，输出 NPY 文件到 \`${params.logDir}/predictions/\`。`
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
      `print(f"推理结果保存在: {os.path.join(LOG_DIR, 'predictions')}")`
    ),
  ]
}

function generateTrainCmdCell(params: TrainNotebookParams): NotebookCell[] {
  const isDdp = params.distribute && params.distributeMode === 'DDP' && params.deviceIds.length > 1
  const runCode = isDdp ? buildDdpCmd('train') : buildSingleCmd('train')

  return [
    mdCell(
      `## 附录：完整重新训练\n` +
      `\n` +
      `使用与工具调用完全相同的参数从头开始训练。\n` +
      `⚠️ 这会覆盖已有的检查点，请确认后再运行。`
    ),
    codeCell(runCode),
  ]
}

function generateCompletionCell(params: TrainNotebookParams): NotebookCell {
  return mdCell(
    `## 输出目录结构\n` +
    `\n` +
    `\`\`\`\n` +
    `${params.logDir}/\n` +
    `├── best_model.pth          # 最优检查点（验证指标最低时保存）\n` +
    `├── last_model.pth          # 最后一个 epoch 的检查点\n` +
    `├── train.log               # 完整训练日志\n` +
    `├── predictions/            # predict 模式输出的 NPY 文件\n` +
    `└── _ocean_sr_code/         # 训练代码快照\n` +
    `    ├── main.py\n` +
    `    ├── main_ddp.py\n` +
    `    └── ${params.modelName}_config.yaml\n` +
    `\`\`\``
  )
}

// ========================================
// 可视化 Cells
// ========================================

/** 训练可视化：调用 generate_training_plots.py 生成 loss/metrics/lr 等图表 */
function generateTrainVisualizeCells(params: TrainNotebookParams): NotebookCell[] {
  const scriptRel = 'generate_training_plots.py'
  return [
    mdCell(
      `## 训练可视化\n` +
      `\n` +
      `调用 \`${scriptRel}\` 从训练日志生成以下图表：\n` +
      `- **loss_curve.png** — 训练/验证损失曲线\n` +
      `- **metrics_curve.png** — MSE/RMSE/PSNR/SSIM 变化曲线\n` +
      `- **lr_curve.png** — 学习率变化曲线\n` +
      `- **metrics_comparison.png** — 验证集与测试集指标对比\n` +
      `- **training_summary.png** — 训练总结表格\n` +
      `- **sample_comparison.png** — 测试样本 LR/SR/HR 对比\n` +
      `\n` +
      `图表保存在 \`LOG_DIR/plots/\` 目录下。`
    ),
    codeCell(
      `plot_script = os.path.join(WORKSPACE_DIR, "..", "${scriptRel}")\n` +
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

/** 预测可视化：调用 generate_predict_plots.py 生成 SR 对比图 */
function generatePredictVisualizeCells(params: TrainNotebookParams): NotebookCell[] {
  const scriptRel = 'generate_predict_plots.py'
  const dynVarsDefault = params.dynVars && params.dynVars.length > 0
    ? params.dynVars.join(',')
    : ''

  return [
    mdCell(
      `## 预测可视化\n` +
      `\n` +
      `调用 \`${scriptRel}\` 从 \`predictions/\` 目录读取 SR NPY 文件，\n` +
      `对比原始 HR/LR 数据生成每个样本的 **LR / SR / HR / Error** 四面板对比图。\n` +
      `\n` +
      `图表保存在 \`LOG_DIR/plots/\` 目录下。`
    ),
    codeCell(
      `predict_plot_script = os.path.join(WORKSPACE_DIR, "..", "${scriptRel}")\n` +
      `dyn_vars_str = ",".join(DYN_VARS) if DYN_VARS else ${toPyRepr(dynVarsDefault)}\n` +
      `max_samples = 4\n` +
      `\n` +
      `result = subprocess.run(\n` +
      `    [PYTHON_PATH, predict_plot_script,\n` +
      `     "--log_dir", LOG_DIR,\n` +
      `     "--dataset_root", DATASET_ROOT,\n` +
      `     "--dyn_vars", dyn_vars_str,\n` +
      `     "--max_samples", str(max_samples)],\n` +
      `    capture_output=True, text=True\n` +
      `)\n` +
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

// ========================================
// 主导出函数
// ========================================

/**
 * 根据训练参数生成所有 Notebook cells
 *
 * 执行顺序：
 *   Cell 1（路径/GPU） → Cell 2（超参） → Cell 3（生成 YAML 配置）
 *   → GPU 检查 → 评估 → 训练可视化 → 推理 → 预测可视化 → [附录] 重新训练
 */
export function generateTrainCells(params: TrainNotebookParams): NotebookCell[] {
  const cells: NotebookCell[] = []

  cells.push(generateTitleCell(params))

  // Cell 1: 路径 & GPU
  cells.push(mdCell('## Cell 1：路径与 GPU 配置'))
  cells.push(generatePathCell(params))

  // Cell 2: 超参
  cells.push(mdCell('## Cell 2：训练超参配置\n\n修改此 cell 后重新运行 Cell 3 即可更新配置。'))
  cells.push(generateHyperparamCell(params))

  // Cell 3: 生成配置文件
  cells.push(mdCell(
    '## Cell 3：生成训练配置文件\n\n' +
    '将 Cell 2 中的超参写入 `CONFIG_PATH` 对应的 YAML 文件。\n' +
    '后续所有 cell 均依赖此配置文件，修改超参后须重新运行本 cell。'
  ))
  cells.push(generateConfigCell(params))

  // GPU 检查
  cells.push(mdCell('## GPU 环境检查'))
  cells.push(generateGpuCheckCell())

  // 评估
  cells.push(...generateEvalCells(params))

  // 训练可视化
  cells.push(...generateTrainVisualizeCells(params))

  // 推理
  cells.push(...generatePredictCells(params))

  // 预测可视化
  cells.push(...generatePredictVisualizeCells(params))

  // 附录：重新训练
  cells.push(...generateTrainCmdCell(params))

  // 输出目录说明
  cells.push(generateCompletionCell(params))

  return cells
}
