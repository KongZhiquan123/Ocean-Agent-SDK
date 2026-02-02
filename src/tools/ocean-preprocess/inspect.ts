import { defineTool } from '@shareai-lab/kode-sdk'

interface VariableInfo {
  name: string
  category: 'dynamic' | 'static' | 'ignored'
  dims: string[]
  shape: number[]
  dtype: string
  units: string
  long_name: string
  is_mask: boolean
}

export interface InspectResult {
  status: 'success' | 'error' | 'awaiting_confirmation'
  nc_folder: string
  file_count: number
  file_list: string[]
  variables: Record<string, VariableInfo>
  dynamic_vars_candidates: string[]
  statistics: Record<string, any>
  warnings: string[]
  errors: string[]
  message: string
}

const TEMP_DIR = '/tmp/ocean_preprocess'

function generateInspectScript(
  ncFolder: string,
  staticFile: string | null,
  fileFilter: string,
  outputJson: string
): string {
  return `
import os
import json
import numpy as np

try:
    import xarray as xr
except ImportError:
    print(json.dumps({"status": "error", "errors": ["需要安装 xarray: pip install xarray netCDF4"]}))
    exit(1)

# 配置
NC_FOLDER = ${JSON.stringify(ncFolder)}
STATIC_FILE = ${staticFile ? JSON.stringify(staticFile) : 'None'}
FILE_FILTER = ${JSON.stringify(fileFilter)}
OUTPUT_JSON = ${JSON.stringify(outputJson)}

# 掩码变量列表（不可修改）
MASK_VARS = ['mask_u', 'mask_rho', 'mask_v']
STATIC_VARS = ['angle', 'h', 'mask_u', 'mask_rho', 'mask_v', 'pn', 'pm', 'f',
               'x_rho', 'x_u', 'x_v', 'y_rho', 'y_u', 'y_v', 'lat_psi', 'lon_psi']

result = {
    "status": "success",
    "nc_folder": NC_FOLDER,
    "file_count": 0,
    "file_list": [],
    "variables": {},
    "dynamic_vars_candidates": [],
    "statistics": {},
    "warnings": [],
    "errors": [],
    "message": ""
}

try:
    # 防错规则 A3: NC文件必须排序
    if not os.path.exists(NC_FOLDER):
        result["errors"].append(f"目录不存在: {NC_FOLDER}")
        result["status"] = "error"
    else:
        nc_files = sorted([f for f in os.listdir(NC_FOLDER)
                          if f.endswith('.nc') and FILE_FILTER in f])
        result["file_list"] = nc_files
        result["file_count"] = len(nc_files)

        if nc_files:
            first_file = os.path.join(NC_FOLDER, nc_files[0])
            with xr.open_dataset(first_file) as ds:
                for var_name in ds.data_vars:
                    var = ds[var_name]
                    dims = list(var.dims)
                    has_time = any(d in dims for d in ['time', 'ocean_time', 't'])

                    # 防错规则 A1: 变量分类
                    if var_name in MASK_VARS:
                        category = "static"
                        is_mask = True
                    elif var_name in STATIC_VARS:
                        category = "static"
                        is_mask = False
                    elif has_time:
                        category = "dynamic"
                        is_mask = False
                    else:
                        category = "ignored"
                        is_mask = False

                    var_info = {
                        "name": var_name,
                        "category": category,
                        "dims": dims,
                        "shape": list(var.shape),
                        "dtype": str(var.dtype),
                        "units": var.attrs.get("units", "unknown"),
                        "long_name": var.attrs.get("long_name", var_name),
                        "is_mask": is_mask
                    }
                    result["variables"][var_name] = var_info

                    if category == "dynamic":
                        result["dynamic_vars_candidates"].append(var_name)

                    # 计算统计信息
                    try:
                        values = var.values
                        result["statistics"][var_name] = {
                            "min": float(np.nanmin(values)),
                            "max": float(np.nanmax(values)),
                            "mean": float(np.nanmean(values)),
                            "nan_count": int(np.isnan(values).sum()),
                            "zero_count": int((values == 0).sum())
                        }
                    except:
                        pass

        # 读取静态文件
        if STATIC_FILE and os.path.exists(STATIC_FILE):
            with xr.open_dataset(STATIC_FILE) as ds:
                for var_name in ds.data_vars:
                    if var_name not in result["variables"]:
                        var = ds[var_name]
                        is_mask = var_name in MASK_VARS
                        result["variables"][var_name] = {
                            "name": var_name,
                            "category": "static",
                            "dims": list(var.dims),
                            "shape": list(var.shape),
                            "dtype": str(var.dtype),
                            "units": var.attrs.get("units", "unknown"),
                            "long_name": var.attrs.get("long_name", var_name),
                            "is_mask": is_mask
                        }

        result["status"] = "awaiting_confirmation"
        result["message"] = f"找到 {len(nc_files)} 个NC文件，{len(result['dynamic_vars_candidates'])} 个动态变量候选"

except Exception as e:
    result["status"] = "error"
    result["errors"].append(str(e))

with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(json.dumps(result, ensure_ascii=False))
`
}

export const oceanInspectDataTool = defineTool({
  name: 'ocean_inspect_data',
  description: `Step A: 查看NC数据并定义变量

用于超分辨率场景的数据预处理第一步。从NC文件中提取变量信息，自动分类动态/静态/掩码变量。

**防错规则**：
- A1: 自动区分动态变量（有时间维）、静态变量（无时间维）、掩码变量（mask_*）
- A2: 陆地掩码变量（mask_u, mask_rho, mask_v）会被标记为不可修改
- A3: NC文件会自动排序以确保时间顺序正确

**返回**：变量列表、形状信息、统计信息、动态变量候选列表

**重要**：执行后需要用户确认研究变量是什么`,

  params: {
    nc_folder: {
      type: 'string',
      description: 'NC文件所在目录的绝对路径'
    },
    static_file: {
      type: 'string',
      description: '静态NC文件的绝对路径（可选）',
      required: false
    },
    file_filter: {
      type: 'string',
      description: '文件名过滤关键字，默认为"avg"',
      required: false,
      default: 'avg'
    }
  },

  attributes: {
    readonly: true,
    noEffect: true
  },

  async exec(args, ctx) {
    const { nc_folder, static_file, file_filter = 'avg' } = args

    ctx.emit('step_started', { step: 'A', description: '查看数据并定义变量' })

    const outputJson = `${TEMP_DIR}/inspect_result.json`

    const script = generateInspectScript(nc_folder, static_file || null, file_filter, outputJson)
    const scriptPath = `${TEMP_DIR}/inspect_data.py`

    try {
      await ctx.sandbox.exec(`mkdir -p ${TEMP_DIR}`)
      await ctx.sandbox.fs.write(scriptPath, script)

      const result = await ctx.sandbox.exec(`python3 ${scriptPath}`, {
        timeoutMs: 300000,
      })

      if (result.code !== 0) {
        ctx.emit('step_failed', { step: 'A', error: result.stderr })
        return {
          status: 'error',
          errors: [`Python执行失败: ${result.stderr}`],
          message: '数据检查失败'
        }
      }

      const jsonContent = await ctx.sandbox.fs.read(outputJson)
      const inspectResult: InspectResult = JSON.parse(jsonContent)

      ctx.emit('step_completed', {
        step: 'A',
        file_count: inspectResult.file_count,
        dynamic_vars: inspectResult.dynamic_vars_candidates
      })

      return inspectResult
    } catch (error: any) {
      ctx.emit('step_failed', { step: 'A', error: error.message })
      return {
        status: 'error',
        errors: [error.message],
        message: '数据检查执行异常'
      }
    }
  }
})
