#!/usr/bin/env python3
"""
inspect_data.py - Step A: 数据检查与变量分类

@author leizheng
@date 2026-02-02
@version 1.0.0

功能:
- 扫描 NC 文件目录，获取文件列表
- 分析变量信息（维度、形状、类型）
- 自动分类：动态变量、静态变量、掩码变量
- 计算统计信息（min, max, mean, nan_count）

用法:
    python inspect_data.py --config config.json --output result.json

配置文件格式:
{
    "nc_folder": "/path/to/nc/files",
    "static_file": "/path/to/static.nc",  // 可选
    "file_filter": "",                     // 文件名过滤关键字，可选
    "dyn_file_pattern": "*.nc"             // glob 模式
}
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import xarray as xr
except ImportError:
    print(json.dumps({
        "status": "error",
        "errors": ["需要安装 xarray: pip install xarray netCDF4"]
    }))
    sys.exit(1)


# ========================================
# 常量定义
# ========================================

# 时间维度检测模式
TIME_DIM_PATTERNS = [
    'time', 'ocean_time', 't', 'Time', 'TIME',
    'nt', 'ntime', 'MT', 'time_counter'
]

# 掩码变量关键字
MASK_KEYWORDS = ['mask', 'land', 'lsm', 'landmask']

# 坐标变量关键字
COORD_KEYWORDS = ['lat', 'lon', 'x_', 'y_', 'angle', 'depth', 'h', 'bathymetry', 'f', 'pn', 'pm']

# 默认静态变量列表（ROMS 模型）
DEFAULT_STATIC_VARS = [
    'angle', 'h', 'hraw', 'mask_u', 'mask_rho', 'mask_v', 'mask_psi',
    'pn', 'pm', 'f', 'dmde', 'dndx',
    'x_rho', 'x_u', 'x_v', 'y_rho', 'y_u', 'y_v',
    'lat_psi', 'lon_psi', 'lat_rho', 'lon_rho',
    'lat_u', 'lon_u', 'lat_v', 'lon_v'
]

# 默认掩码变量列表
DEFAULT_MASK_VARS = ['mask_u', 'mask_rho', 'mask_v', 'mask_psi']


# ========================================
# 辅助函数
# ========================================

def has_time_dimension(dims: List[str]) -> bool:
    """检查维度列表中是否包含时间维度"""
    for dim in dims:
        for pattern in TIME_DIM_PATTERNS:
            if pattern.lower() in dim.lower():
                return True
    return False


def guess_variable_type(var_name: str, dims: List[str], has_time: bool) -> str:
    """
    猜测变量类型
    返回: "suspected_mask" / "suspected_coordinate" / "dynamic" / "static" / "unknown"
    """
    name_lower = var_name.lower()

    # 1. 检测掩码
    if any(kw in name_lower for kw in MASK_KEYWORDS):
        return "suspected_mask"

    # 2. 检测坐标/地形
    if any(kw in name_lower for kw in COORD_KEYWORDS):
        return "suspected_coordinate"

    # 3. 根据时间维度判断
    if has_time:
        return "dynamic"

    return "static"


def compute_statistics(var_data: np.ndarray) -> Dict[str, Any]:
    """计算变量统计信息"""
    try:
        values = np.asarray(var_data)
        return {
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
            "mean": float(np.nanmean(values)),
            "nan_count": int(np.isnan(values).sum()),
            "zero_count": int((values == 0).sum())
        }
    except Exception:
        return {}


def analyze_variable(ds: xr.Dataset, var_name: str,
                     mask_vars: List[str],
                     static_vars: List[str]) -> Dict[str, Any]:
    """分析单个变量"""
    var = ds[var_name]
    dims = list(var.dims)
    shape = list(var.shape)
    has_time = has_time_dimension(dims)

    # 确定分类
    if var_name in mask_vars:
        category = "mask"
        is_mask = True
    elif var_name in static_vars:
        category = "static"
        is_mask = False
    elif has_time:
        category = "dynamic"
        is_mask = False
    else:
        category = "static"
        is_mask = False

    return {
        "name": var_name,
        "category": category,
        "dims": dims,
        "shape": shape,
        "dtype": str(var.dtype),
        "units": var.attrs.get("units", "unknown"),
        "long_name": var.attrs.get("long_name", var_name),
        "is_mask": is_mask,
        "has_time": has_time,
        "suspected_type": guess_variable_type(var_name, dims, has_time)
    }


# ========================================
# 主函数
# ========================================

def inspect_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行数据检查

    Args:
        config: 配置字典

    Returns:
        检查结果字典
    """
    nc_folder = config.get("nc_folder", "")
    static_file = config.get("static_file")
    file_filter = config.get("file_filter", "")
    dyn_file_pattern = config.get("dyn_file_pattern", "*.nc")
    mask_vars = config.get("mask_vars", DEFAULT_MASK_VARS)
    static_vars = config.get("static_vars", DEFAULT_STATIC_VARS)

    result = {
        "status": "success",
        "nc_folder": nc_folder,
        "file_count": 0,
        "file_list": [],
        "variables": {},
        "dynamic_vars_candidates": [],
        "static_vars_found": [],
        "mask_vars_found": [],
        "statistics": {},
        "warnings": [],
        "errors": [],
        "message": "",
        "suspected_masks": [],
        "suspected_coordinates": []
    }

    try:
        # 1. 检查目录存在
        if not os.path.exists(nc_folder):
            result["errors"].append(f"目录不存在: {nc_folder}")
            result["status"] = "error"
            return result

        # 2. 获取 NC 文件列表（排序）
        import glob
        search_path = os.path.join(nc_folder, dyn_file_pattern)
        nc_files = sorted(glob.glob(search_path))

        # 应用文件过滤
        if file_filter:
            nc_files = [f for f in nc_files if file_filter in os.path.basename(f)]

        result["file_list"] = [os.path.basename(f) for f in nc_files]
        result["file_count"] = len(nc_files)

        if not nc_files:
            result["warnings"].append(f"未找到匹配的 NC 文件，模式: {dyn_file_pattern}")

        # 3. 分析第一个动态文件
        if nc_files:
            first_file = nc_files[0]
            print(f"分析动态文件: {first_file}", file=sys.stderr)

            with xr.open_dataset(first_file, decode_times=False) as ds:
                for var_name in ds.data_vars:
                    var_info = analyze_variable(ds, var_name, mask_vars, static_vars)
                    result["variables"][var_name] = var_info

                    if var_info["category"] == "dynamic":
                        result["dynamic_vars_candidates"].append(var_name)
                    elif var_info["category"] == "mask":
                        result["mask_vars_found"].append(var_name)
                    elif var_info["category"] == "static":
                        result["static_vars_found"].append(var_name)

                    # 计算统计信息
                    result["statistics"][var_name] = compute_statistics(ds[var_name].values)

        # 4. 分析静态文件
        if static_file and os.path.exists(static_file):
            print(f"分析静态文件: {static_file}", file=sys.stderr)

            with xr.open_dataset(static_file, decode_times=False) as ds:
                for var_name in ds.data_vars:
                    if var_name not in result["variables"]:
                        var_info = analyze_variable(ds, var_name, mask_vars, static_vars)
                        # 静态文件中的变量默认为静态
                        if var_info["category"] == "dynamic":
                            var_info["category"] = "static"
                        result["variables"][var_name] = var_info

                        if var_info["category"] == "mask":
                            result["mask_vars_found"].append(var_name)
                        else:
                            result["static_vars_found"].append(var_name)

                        result["statistics"][var_name] = compute_statistics(ds[var_name].values)
        elif static_file:
            result["warnings"].append(f"静态文件不存在: {static_file}")

        # 5. 收集疑似变量
        for var_name, var_info in result["variables"].items():
            suspected = var_info.get("suspected_type", "")
            if suspected == "suspected_mask":
                result["suspected_masks"].append(var_name)
            elif suspected == "suspected_coordinate":
                result["suspected_coordinates"].append(var_name)

        # 6. 生成摘要消息
        result["status"] = "awaiting_confirmation"
        result["message"] = (
            f"找到 {len(nc_files)} 个 NC 文件，"
            f"{len(result['dynamic_vars_candidates'])} 个动态变量候选，"
            f"{len(result['static_vars_found'])} 个静态变量，"
            f"{len(result['mask_vars_found'])} 个掩码变量"
        )

    except Exception as e:
        result["status"] = "error"
        result["errors"].append(str(e))
        import traceback
        result["traceback"] = traceback.format_exc()

    return result


def main():
    parser = argparse.ArgumentParser(description="Step A: 数据检查与变量分类")
    parser.add_argument("--config", required=True, help="JSON 配置文件路径")
    parser.add_argument("--output", required=True, help="结果输出 JSON 路径")
    args = parser.parse_args()

    # 读取配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 执行检查
    result = inspect_data(config)

    # 写入结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 同时输出到 stdout
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
