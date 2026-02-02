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
  fileFilter: string,
  outputJson: string
): string {
  return `
import os
import json
import numpy as np

try:
    import xarray as xr
    from tqdm import tqdm
except ImportError:
    print(json.dumps({"status": "error", "errors": ["需要安装 xarray tqdm: pip install xarray netCDF4 tqdm"]}))
    exit(1)

# 配置
NC_FOLDER = ${JSON.stringify(ncFolder)}
OUTPUT_BASE = ${JSON.stringify(outputBase)}
RESEARCH_VARS = ${JSON.stringify(researchVars)}
STATIC_FILE = ${staticFile ? JSON.stringify(staticFile) : 'None'}
STATIC_VARS = ${JSON.stringify(staticVars)}
MASK_VARS = ${JSON.stringify(maskVars)}
FILE_FILTER = ${JSON.stringify(fileFilter)}
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
    # 防错规则 C1: 创建目录结构
    hr_dir = os.path.join(OUTPUT_BASE, "hr")
    stat_dir = os.path.join(OUTPUT_BASE, "static")
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(stat_dir, exist_ok=True)

    # 防错规则 C4: NC文件排序
    nc_files = sorted([f for f in os.listdir(NC_FOLDER)
                      if f.endswith('.nc') and FILE_FILTER in f])

    if not nc_files:
        result["errors"].append("未找到NC文件")
        result["status"] = "error"
    else:
        # 按变量收集数据
        data_accum = {var: [] for var in RESEARCH_VARS}

        for fname in tqdm(nc_files, desc="读取NC文件"):
            fp = os.path.join(NC_FOLDER, fname)
            try:
                with xr.open_dataset(fp) as ds:
                    for var in RESEARCH_VARS:
                        if var in ds.variables:
                            arr = ds[var].values
                            data_accum[var].append(arr)
            except Exception as e:
                result["warnings"].append(f"读取 {fname} 失败: {str(e)}")

        # 拼接并保存动态变量
        for var, arr_list in data_accum.items():
            if not arr_list:
                result["errors"].append(f"变量 '{var}' 无数据")
                continue

            # 检查空间维度一致性
            shape0 = arr_list[0].shape[1:]
            valid = True
            for i, arr in enumerate(arr_list):
                if arr.shape[1:] != shape0:
                    result["errors"].append(f"变量 '{var}' 空间维度不一致: 文件#{i}")
                    valid = False
                    break

            if not valid:
                continue

            # 按时间轴拼接
            data_concat = np.concatenate(arr_list, axis=0)

            # 防错规则 C2: 维度检查
            ndim = data_concat.ndim
            if ndim == 3:
                interp = f"[T={data_concat.shape[0]}, H={data_concat.shape[1]}, W={data_concat.shape[2]}]"
            elif ndim == 4:
                interp = f"[T={data_concat.shape[0]}, D={data_concat.shape[1]}, H={data_concat.shape[2]}, W={data_concat.shape[3]}]"
            else:
                result["errors"].append(f"数据维度检查有问题，请检查变量 '{var}'：期望3D或4D，实际{ndim}D")
                continue

            out_fp = os.path.join(hr_dir, f"{var}.npy")
            np.save(out_fp, data_concat)

            result["saved_files"][var] = {
                "path": out_fp,
                "shape": list(data_concat.shape),
                "dtype": str(data_concat.dtype),
                "interpretation": interp
            }

        # 防错规则 C3: 处理静态变量（掩码原样保存）
        if STATIC_FILE and os.path.exists(STATIC_FILE):
            with xr.open_dataset(STATIC_FILE) as ds:
                for var in STATIC_VARS:
                    if var in ds.variables:
                        arr = ds[var].values
                        is_mask = var in MASK_VARS
                        out_fp = os.path.join(stat_dir, f"{var}.npy")
                        np.save(out_fp, arr)

                        result["saved_files"][var] = {
                            "path": out_fp,
                            "shape": list(arr.shape),
                            "dtype": str(arr.dtype),
                            "is_mask": is_mask
                        }

        # 事后防错规则 1 & 2: 验证
        validation_passed = True

        if not os.path.exists(hr_dir):
            result["errors"].append("事后检查失败：hr目录不存在")
            validation_passed = False

        for var in RESEARCH_VARS:
            expected_file = os.path.join(hr_dir, f"{var}.npy")
            if not os.path.exists(expected_file):
                result["errors"].append(f"事后检查失败：{var}.npy 不存在")
                validation_passed = False
            else:
                loaded = np.load(expected_file)
                if loaded.ndim not in [3, 4]:
                    result["errors"].append(f"数据维度检查有问题，请检查 '{var}' 部分：期望[T,H,W]，实际shape={loaded.shape}")
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
            result["message"] = f"预处理完成，已保存 {len(result['saved_files'])} 个文件"

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

**防错规则**：
- C1: 输出目录结构为 output_base/hr/变量.npy 和 output_base/static/变量.npy
- C2: 保存前验证维度：动态 [T,H,W] 或 [T,D,H,W]，静态 [H,W]
- C3: 掩码变量（mask_*）原样保存，不做任何修改
- C4: NC文件按文件名排序后处理，确保时间顺序

**事后验证**：
- 检查目录结构是否存在
- 检查每个文件的维度是否正确
- 如果不对，报错"数据维度检查有问题，请检查xxx部分"

**返回**：保存的文件列表、验证结果、状态（pass/error）`,

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
      description: '研究变量列表'
    },
    static_file: {
      type: 'string',
      description: '静态NC文件路径（可选）',
      required: false
    },
    static_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '静态变量列表',
      required: false,
      default: ['angle', 'h', 'mask_u', 'mask_rho', 'mask_v', 'pn', 'pm', 'f',
                'x_rho', 'x_u', 'x_v', 'y_rho', 'y_u', 'y_v', 'lat_psi', 'lon_psi']
    },
    mask_vars: {
      type: 'array',
      items: { type: 'string' },
      description: '掩码变量列表（不可修改）',
      required: false,
      default: ['mask_u', 'mask_rho', 'mask_v']
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
    const {
      nc_folder,
      output_base,
      research_vars,
      static_file,
      static_vars = ['angle', 'h', 'mask_u', 'mask_rho', 'mask_v', 'pn', 'pm', 'f',
                     'x_rho', 'x_u', 'x_v', 'y_rho', 'y_u', 'y_v', 'lat_psi', 'lon_psi'],
      mask_vars = ['mask_u', 'mask_rho', 'mask_v'],
      file_filter = 'avg'
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
      research_vars,
      static_file || null,
      static_vars,
      mask_vars,
      file_filter,
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
