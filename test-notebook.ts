/**
 * @file test-notebook.ts
 *
 * @description 快速测试：生成训练/预处理/预报预处理 Notebook 并写入本地文件
 * @author kongzhiquan
 * @date 2026-02-26
 * @version 1.1.0
 *
 * @changelog
 *   - 2026-02-26 kongzhiquan: v1.1.0 新增预报数据预处理 Notebook 测试
 *   - 2026-02-25 kongzhiquan: v1.0.0 初始版本
 *
 * 运行：npx tsx test-notebook.ts
 * 产物：test_outputs/test-train.ipynb, test_outputs/test-preprocess.ipynb, test_outputs/test-forecast-preprocess.ipynb
 */

import { generateTrainCells } from './src/tools/ocean-sr-training/notebook'
import { generatePreprocessCells } from './src/tools/ocean-SR-data-preprocess/notebook'
import { generateForecastPreprocessCells } from './src/tools/ocean-forecast-data-preprocess/notebook'
import { createEmptyNotebook } from './src/utils/notebook'
import fs from 'node:fs'
import path from 'node:path'

// ========== 训练 Notebook ==========
const notebook = createEmptyNotebook()
notebook.cells = generateTrainCells({
  logDir: '/data/experiments/unet_logs',
  datasetRoot: '/data/ocean_dataset',
  modelName: 'SwinIR',
  configPath: '/data/experiments/unet_logs/_ocean_sr_code/SwinIR_config.yaml',
  workspaceDir: '/data/experiments/unet_logs/_ocean_sr_code',
  pythonPath: '/home/lz/miniconda3/envs/pytorch/bin/python',

  deviceIds: [0, 1],
  distribute: true,
  distributeMode: 'DDP',
  masterPort: 29500,

  mode: 'train',
  epochs: 500,
  lr: 0.001,
  batchSize: 4,
  evalBatchSize: 4,
  scale: 4,
  dynVars: ['temp', 'salt'],
  normalize: true,
  normalizerType: 'PGN',
  optimizer: 'AdamW',
  scheduler: 'StepLR',
  seed: 42,
  useAmp: true,
  gradientCheckpointing: true,
  patchSize: 128,
  wandb: false,
})

const outDir = path.resolve('test_outputs')
fs.mkdirSync(outDir, { recursive: true })
const outPath = path.join(outDir, 'test-train.ipynb')
fs.writeFileSync(outPath, JSON.stringify(notebook, null, 2), 'utf-8')

console.log(`训练 Notebook 已写入: ${outPath}`)
console.log(`共 ${notebook.cells.length} 个 cells`)
notebook.cells.forEach((c, i) => {
  const preview = c.source.slice(0, 2).join('').replace(/\n/g, ' ').slice(0, 60)
  console.log(`  [${i}] ${c.cell_type.padEnd(8)} ${preview}...`)
})

// ========== 预处理 Notebook ==========
const preprocessNotebook = createEmptyNotebook()
preprocessNotebook.cells = generatePreprocessCells({
  outputBase: '/data/ocean_dataset_processed',
  ncFolder: '/data/raw_nc_files',
  staticFile: '/data/raw_nc_files/static.nc',
  dynVars: ['temp', 'salt'],
  statVars: ['lon_rho', 'lat_rho'],
  maskVars: ['mask_rho'],
  lonVar: 'lon_rho',
  latVar: 'lat_rho',
  primaryMaskVar: 'mask_rho',
  trainRatio: 0.7,
  validRatio: 0.15,
  testRatio: 0.15,
  scale: 4,
  downsampleMethod: 'area',
  hSlice: '0:680',
  wSlice: '0:1440',
  workers: 8,
  allowNan: true,
  dynFilePattern: '*.nc',
  enableRegionCrop: false,
  cropMode: 'two_step',
  useDateFilename: true,
  dateFormat: 'auto',
  isNumericalModelMode: false,
  skipDownsample: false,
  skipVisualize: false,
  pythonPath: '/home/lz/miniconda3/envs/pytorch/bin/python',
})

const preprocessOutPath = path.join(outDir, 'test-preprocess.ipynb')
fs.writeFileSync(preprocessOutPath, JSON.stringify(preprocessNotebook, null, 2), 'utf-8')

console.log(`\n预处理 Notebook 已写入: ${preprocessOutPath}`)
console.log(`共 ${preprocessNotebook.cells.length} 个 cells`)
preprocessNotebook.cells.forEach((c, i) => {
  const preview = c.source.slice(0, 2).join('').replace(/\n/g, ' ').slice(0, 60)
  console.log(`  [${i}] ${c.cell_type.padEnd(8)} ${preview}...`)
})

// ========== 预报数据预处理 Notebook ==========
const forecastNotebook = createEmptyNotebook()
forecastNotebook.cells = generateForecastPreprocessCells({
  outputBase: '/data/ocean_forecast_processed',
  ncFolder: '/data/raw_forecast_nc',
  staticFile: '/data/raw_forecast_nc/grid.nc',
  dynVars: ['temp', 'salt', 'zeta'],
  statVars: ['lon_rho', 'lat_rho', 'h'],
  maskVars: ['mask_rho'],
  lonVar: 'lon_rho',
  latVar: 'lat_rho',
  trainRatio: 0.7,
  validRatio: 0.15,
  testRatio: 0.15,
  hSlice: '0:512',
  wSlice: '0:1024',
  allowNan: true,
  dynFilePattern: 'ocean_avg_*.nc',
  chunkSize: 200,
  useDateFilename: true,
  dateFormat: 'auto',
  timeVar: 'ocean_time',
  maxFiles: undefined,
  skipVisualize: false,
  pythonPath: '/home/lz/miniconda3/envs/pytorch/bin/python',
})

const forecastOutPath = path.join(outDir, 'test-forecast-preprocess.ipynb')
fs.writeFileSync(forecastOutPath, JSON.stringify(forecastNotebook, null, 2), 'utf-8')

console.log(`\n预报预处理 Notebook 已写入: ${forecastOutPath}`)
console.log(`共 ${forecastNotebook.cells.length} 个 cells`)
forecastNotebook.cells.forEach((c, i) => {
  const preview = c.source.slice(0, 2).join('').replace(/\n/g, ' ').slice(0, 60)
  console.log(`  [${i}] ${c.cell_type.padEnd(8)} ${preview}...`)
})
