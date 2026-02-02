import { defineTool } from '@shareai-lab/kode-sdk'
import { findFirstPythonPath } from '@/utils/python-manager'
import path from 'node:path'

export interface ConvertResult {
  status: 'pass' | 'error' | 'pending'
  output_dir: string
  saved_files: Record<string, any>
  post_validation: Record<string, any>
  warnings: string[]
  errors: string[]
  message: string
}

function generateConvertScript(
  ncFolder: string,
  outputBase: string,
  researchVars: string[],
  staticFile: string | null,
  staticVars: string[],
  maskVars: string[],
  filePattern: string,
  outputJson: string
): string {
  return `
import os
import glob
import json
import numpy as np
import warnings

try:
    import xarray as xr
except ImportError:
    print(json.dumps({"status": "error", "errors": ["需要安装 xarray: pip install xarray netCDF4 dask"]}))
    exit(1)

# 配置
NC_FOLDER = ${JSON.stringify(ncFolder)}
OUTPUT_BASE = ${JSON.stringify(outputBase)}
DYN_VARS = ${JSON.stringify(researchVars)}
STATIC_FILE = ${staticFile ? JSON.stringify(staticFile) : 'None'}
STAT_VARS = ${JSON.stringify(staticVars)}
MASK_VARS = ${JSON.stringify(maskVars)}
DYN_FILE_PATTERN = ${JSON.stringify(filePattern)}
OUTPUT_JSON = ${JSON.stringify(outputJson)}

result = {
    "status": "pending",
    "output_dir": OUTPUT_BASE,
    "saved_files": {},
    "post_validation": {},
    "warnings": [],
    "errors": [],
    "message": ""
}

try:
    # --- 1. 路径与环境设置 ---
    dyn_out_dir = os.path.join(OUTPUT_BASE, 'target_variables')
    stat_out_dir = os.path.join(OUTPUT_BASE, 'static_variables')

    os.makedirs(dyn_out_dir, exist_ok=True)
    os.makedirs(stat_out_dir, exist_ok=True)

    print(f"Start processing...")
    print(f"Dynamic output: {dyn_out_dir}")
    print(f"Static output:  {stat_out_dir}")

    # --- 2. 处理动态变量 (Dynamic Variables) ---
    print(f"\\n--- Processing Dynamic Files in: {NC_FOLDER} ---")

    # 使用 glob 获取并排序文件（与原脚本一致）
    search_path = os.path.join(NC_FOLDER, DYN_FILE_PATTERN)
    nc_files = sorted(glob.glob(search_path))

    if not nc_files:
        result["errors"].append(f"未找到匹配 '{DYN_FILE_PATTERN}' 的NC文件，搜索路径: {search_path}")
        result["status"] = "error"
    else:
        print(f"Found {len(nc_files)} dynamic files. Using xarray multi-file dataset...")

        try:
            # 使用 open_mfdataset 进行惰性加载和自动拼接
            # combine='by_coords' 智能按坐标合并
            # parallel=True 利用多核加速元数据读取
            # chunks='auto' 开启 dask 惰性加载，防止内存溢出
            with xr.open_mfdataset(
                nc_files,
                combine='by_coords',
                chunks='auto',
                parallel=True,
                decode_times=False  # 防止非常规时间格式报错
            ) as ds:

                for var in DYN_VARS:
                    out_fp = os.path.join(dyn_out_dir, f"{var}.npy")

                    if var in ds.data_vars or var in ds.coords:
                        try:
                            print(f"Extracting dynamic variable: {var} ...", end="", flush=True)

                            # .values 会触发实际的计算和读取 (compute)
                            data_arr = ds[var].values

                            np.save(out_fp, data_arr)
                            print(f" Done. Shape: {data_arr.shape}")

                            # 维度解释
                            ndim = data_arr.ndim
                            if ndim == 3:
                                interp = f"[T={data_arr.shape[0]}, H={data_arr.shape[1]}, W={data_arr.shape[2]}]"
                            elif ndim == 4:
                                interp = f"[T={data_arr.shape[0]}, D={data_arr.shape[1]}, H={data_arr.shape[2]}, W={data_arr.shape[3]}]"
                            else:
                                interp = f"shape={data_arr.shape}"

                            result["saved_files"][var] = {
                                "path": out_fp,
                                "shape": list(data_arr.shape),
                                "dtype": str(data_arr.dtype),
                                "interpretation": interp
                            }

                            # 显式删除引用，辅助垃圾回收
                            del data_arr

                        except Exception as e:
                            result["warnings"].append(f"处理变量 '{var}' 失败: {str(e)}")
                    else:
                        result["warnings"].append(f"变量 '{var}' 在数据集中不存在")

        except Exception as e:
            result["errors"].append(f"打开动态数据集失败: {str(e)}")
            result["warnings"].append("提示: 检查所有NetCDF文件是否具有一致的维度和坐标")

    # --- 3. 处理静态变量 (Static Variables) ---
    print(f"\\n--- Processing Static File: {STATIC_FILE} ---")

    if STATIC_FILE and os.path.exists(STATIC_FILE):
        try:
            with xr.open_dataset(STATIC_FILE) as ds:
                for var in STAT_VARS:
                    out_fp = os.path.join(stat_out_dir, f"{var}.npy")

                    if var in ds.variables:  # variables 包含 data_vars 和 coords
                        try:
                            data_arr = ds[var].values
                            np.save(out_fp, data_arr)
                            print(f"Saved static {var}.npy, shape={data_arr.shape}")

                            is_mask = var in MASK_VARS
                            result["saved_files"][var] = {
                                "path": out_fp,
                                "shape": list(data_arr.shape),
                                "dtype": str(data_arr.dtype),
                                "is_mask": is_mask
                            }
                        except Exception as e:
                            result["warnings"].append(f"保存静态变量 '{var}' 失败: {str(e)}")
                    else:
                        result["warnings"].append(f"静态文件中不存在变量 '{var}'")
        except Exception as e:
            result["errors"].append(f"读取静态文件失败: {str(e)}")
    elif STATIC_FILE:
        result["warnings"].append(f"静态文件不存在: {STATIC_FILE}")

    # --- 4. 事后验证 ---
    print("\\n--- Post Validation ---")
    validation_passed = True

    if not os.path.exists(dyn_out_dir):
        result["errors"].append("事后检查失败：target_variables 目录不存在")
        validation_passed = False

    for var in DYN_VARS:
        expected_file = os.path.join(dyn_out_dir, f"{var}.npy")
        if not os.path.exists(expected_file):
            result["errors"].append(f"事后检查失败：{var}.npy 不存在")
            validation_passed = False
        else:
            loaded = np.load(expected_file)
            if loaded.ndim not in [3, 4]:
                result["errors"].append(f"数据维度检查有问题，请检查 '{var}'：期望3D或4D，实际{loaded.ndim}D, shape={loaded.shape}")
                validation_passed = False
            else:
                result["post_validation"][var] = {
                    "shape": list(loaded.shape),
                    "valid": True
                }

    if result["errors"]:
        result["status"] = "error"
        result["message"] = f"处理失败，存在 {len(result['errors'])} 个错误"
    else:
        result["status"] = "pass"
        result["message"] = f"预处理完成，已保存 {len(result['saved_files'])} 个文件到 {OUTPUT_BASE}"

    print("\\nAll processing complete.")

except Exception as e:
    result["status"] = "error"
    result["errors"].append(str(e))

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(json.dumps(result, ensure_ascii=False))
`
}

export const oceanConvertNpyTool = defineTool({
  name: 'ocean_convert_npy',
  description: `Step C: 转换为NPY格式并按目录结构存储

将NC文件中的变量转换为NPY格式，按 OceanSRDataset 要求的目录结构保存。

**核心特性**：
- 使用 xr.open_mfdataset 进行多文件惰性加载和自动拼接
- 支持 Dask 惰性加载（chunks='auto'），防止内存溢出
- 支持多核并行加速（parallel=True）

**输出目录结构**：
- output_base/target_variables/变量.npy - 动态研究变量
- output_base/static_variables/变量.npy - 静态变量

**防错规则**：
- C1: 自动创建输出目录结构
- C2: 保存前验证维度：动态 [T,H,W] 或 [T,D,H,W]
- C3: 掩码变量（mask_*）原样保存，不做任何修改
- C4: NC文件按文件名排序后处理，确保时间顺序

**事后验证**：
- 检查目录结构是否存在
- 检查每个文件的维度是否正确

**返回**：保存的文件列表、验证结果、状态（pass/error）`,

  params: {
    nc_folder: {
      type: 'string',
      description: '动态NC文件所在目录 (dyn_dir)'
    },
    output_base: {
      type: 'string',
      description: '输出根目录 (output_base_dir)'
    },
    dyn_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '动态变量列表 (dyn_vars)'
    },
    static_file: {
      type: 'string',
      description: '静态NC文件路径 (stat_file)',
      required: false
    },
    stat_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '静态变量列表 (stat_vars)',
      required: false,
      default: ['angle', 'h', 'mask_u', 'mask_rho', 'mask_v', 'pn', 'pm', 'f',
                'x_rho', 'x_u', 'x_v', 'y_rho', 'y_u', 'y_v', 'lat_psi', 'lon_psi']
    },
    mask_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '掩码变量列表',
      required: false,
      default: ['mask_u', 'mask_rho', 'mask_v']
    },
    dyn_file_pattern: {
      type: 'string',
      description: '动态文件的 glob 匹配模式，如 "*.nc" 或 "*avg*.nc"',
      required: false,
      default: '*.nc'
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
      dyn_vars,
      static_file,
      stat_vars = ['angle', 'h', 'mask_u', 'mask_rho', 'mask_v', 'pn', 'pm', 'f',
                   'x_rho', 'x_u', 'x_v', 'y_rho', 'y_u', 'y_v', 'lat_psi', 'lon_psi'],
      mask_vars = ['mask_u', 'mask_rho', 'mask_v'],
      dyn_file_pattern = '*.nc'
    } = args

    ctx.emit('step_started', { step: 'C', description: '转换为NPY格式存储' })

    const pythonPath = findFirstPythonPath()
    if (!pythonPath) {
      const errorMsg = '未找到可用的Python解释器，请安装Python或配置PYTHON/PYENV'
      ctx.emit('step_failed', { step: 'C', error: errorMsg })
      return {
        status: 'error',
        errors: [errorMsg],
        message: '转换失败'
      }
    }
    const pythonCmd = `"${pythonPath}"`
    const tempDir = path.join(ctx.sandbox.workDir, 'ocean_preprocess_temp')
    const outputJson = `${tempDir}/convert_result.json`

    const script = generateConvertScript(
      nc_folder,
      output_base,
      dyn_vars,
      static_file || null,
      stat_vars,
      mask_vars,
      dyn_file_pattern,
      outputJson
    )
    const scriptPath = `${tempDir}/convert_npy.py`

    try {
      await ctx.sandbox.exec(`mkdir -p ${tempDir}`)
      await ctx.sandbox.fs.write(scriptPath, script)

      const result = await ctx.sandbox.exec(`${pythonCmd} ${scriptPath}`, {
        timeoutMs: 600000,
      })

      if (result.code !== 0) {
        ctx.emit('step_failed', { step: 'C', error: result.stderr })
        return {
          status: 'error',
          errors: [`Python执行失败: ${result.stderr}`],
          message: '转换失败'
        }
      }

      const jsonContent = await ctx.sandbox.fs.read(outputJson)
      const convertResult: ConvertResult = JSON.parse(jsonContent)

      ctx.emit('step_completed', {
        step: 'C',
        status: convertResult.status,
        saved_files: Object.keys(convertResult.saved_files)
      })

      return convertResult
    } catch (error: any) {
      ctx.emit('step_failed', { step: 'C', error: error.message })
      return {
        status: 'error',
        errors: [error.message],
        message: '转换执行异常'
      }
    }
  }
})
