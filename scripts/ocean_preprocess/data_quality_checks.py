#!/usr/bin/env python3
"""
data_quality_checks.py

Author: leizheng
Time: 2026-02-02
Description: 数据质量检测工具（可选）
             这些检测不影响数据切割，但对后续数据处理/训练可能有用：
             - 时间单调性、重复、间隙检测
             - 数据质量检测（全零、常数、极端值）
             - 坐标单调性/跳跃检测
             - 启发式掩码验证
             - 坐标范围验证
Version: 1.0.0

Changelog:
  - 2026-02-02 leizheng: v1.0.0 从 convert_npy.py 分离可选检测逻辑

用法:
    from data_quality_checks import (
        check_time_validity,
        check_data_quality,
        check_coordinate_monotonicity,
        heuristic_mask_check,
        verify_coordinate_range_full
    )
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple


# ========================================
# 常量
# ========================================

LAND_THRESHOLD_ABS = 1e-12
HEURISTIC_SAMPLE_SIZE = 2000


# ========================================
# 时间检测
# ========================================

def check_time_validity(time_arr: np.ndarray) -> Dict[str, Any]:
    """
    检测时间坐标的有效性

    检测项:
    - 时间单调性（必须单调递增）
    - 时间重复（不允许重复值）
    - 负数时间（警告）
    - 时间间隙（检测异常大的间隔）

    Args:
        time_arr: 时间坐标数组

    Returns:
        检测结果字典
    """
    result = {
        "valid": True,
        "is_monotonic": True,
        "has_duplicates": False,
        "has_negative": False,
        "has_large_gap": False,
        "length": len(time_arr) if time_arr is not None else 0,
        "errors": [],
        "warnings": []
    }

    if time_arr is None or len(time_arr) == 0:
        result["valid"] = False
        result["errors"].append("时间坐标为空")
        return result

    # 检测单调性
    if len(time_arr) > 1:
        diffs = np.diff(time_arr)
        if not np.all(diffs > 0):
            result["is_monotonic"] = False
            result["valid"] = False
            # 检测是否有重复
            if np.any(diffs == 0):
                result["has_duplicates"] = True
                dup_count = int(np.sum(diffs == 0))
                result["errors"].append(f"时间坐标包含 {dup_count} 个重复值，时间必须严格单调递增")
            # 检测是否有逆序
            if np.any(diffs < 0):
                reverse_count = int(np.sum(diffs < 0))
                result["errors"].append(f"时间坐标有 {reverse_count} 处逆序，时间必须单调递增")

        # 检测异常大的时间间隙
        if len(diffs) > 1:
            median_diff = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 0
            if median_diff > 0:
                large_gaps = diffs > median_diff * 10  # 超过中位数 10 倍认为是大间隙
                if np.any(large_gaps):
                    result["has_large_gap"] = True
                    gap_count = int(np.sum(large_gaps))
                    result["warnings"].append(f"时间序列有 {gap_count} 个异常大的间隙（可能缺失数据）")

    # 检测负数时间
    if np.any(time_arr < 0):
        result["has_negative"] = True
        neg_count = int(np.sum(time_arr < 0))
        result["warnings"].append(f"时间坐标包含 {neg_count} 个负数值")

    return result


# ========================================
# 数据质量检测
# ========================================

def check_data_quality(arr: np.ndarray, var_name: str) -> Dict[str, Any]:
    """
    检测数据质量

    检测项:
    - 全零数据
    - 常数值数据
    - 极端值（可能是 scale 溢出）

    Args:
        arr: 输入数组
        var_name: 变量名

    Returns:
        检测结果字典
    """
    result = {
        "var_name": var_name,
        "all_zeros": False,
        "all_constant": False,
        "has_extreme": False,
        "extreme_threshold": 1e30,
        "warnings": []
    }

    if arr.size == 0:
        return result

    # 忽略 NaN 进行统计
    valid_data = arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr.flatten()

    if len(valid_data) == 0:
        result["warnings"].append(f"变量 '{var_name}' 所有数据都是 NaN")
        return result

    # 检测全零
    if np.all(valid_data == 0):
        result["all_zeros"] = True
        result["warnings"].append(f"变量 '{var_name}' 所有数据为零")

    # 检测常数值（采样检查）
    unique_sample = np.unique(valid_data[:min(10000, len(valid_data))])
    if len(unique_sample) == 1:
        result["all_constant"] = True
        result["warnings"].append(f"变量 '{var_name}' 所有数据为常数值: {unique_sample[0]}")

    # 检测极端值（可能是 scale 溢出）
    max_abs = np.max(np.abs(valid_data))
    if max_abs > result["extreme_threshold"]:
        result["has_extreme"] = True
        result["warnings"].append(
            f"变量 '{var_name}' 包含极端值 (max |value| = {max_abs:.2e})，可能是 scale_factor 溢出"
        )

    return result


# ========================================
# 坐标单调性检测
# ========================================

def check_coordinate_monotonicity(coord_arr: np.ndarray, coord_name: str) -> Dict[str, Any]:
    """
    检测坐标变量的单调性和连续性

    检测项:
    - 单调性
    - 跳跃（不连续）

    Args:
        coord_arr: 坐标数组
        coord_name: 坐标名称

    Returns:
        检测结果字典
    """
    result = {
        "coord_name": coord_name,
        "is_monotonic": True,
        "has_jump": False,
        "jump_positions": [],
        "warnings": []
    }

    if coord_arr is None or len(coord_arr) < 2:
        return result

    # 展平多维坐标数组
    if coord_arr.ndim > 1:
        coord_arr = coord_arr.flatten()

    diffs = np.diff(coord_arr)

    # 检测单调性
    if not (np.all(diffs >= 0) or np.all(diffs <= 0)):
        result["is_monotonic"] = False
        result["warnings"].append(f"坐标 '{coord_name}' 不是单调的")

    # 检测跳跃（不连续）
    if len(diffs) > 1:
        median_diff = np.median(np.abs(diffs))
        if median_diff > 0:
            jumps = np.abs(diffs) > median_diff * 5
            if np.any(jumps):
                result["has_jump"] = True
                jump_positions = np.where(jumps)[0].tolist()
                result["jump_positions"] = jump_positions[:10]
                result["warnings"].append(
                    f"坐标 '{coord_name}' 有 {int(np.sum(jumps))} 处跳跃（不连续），位置: {jump_positions[:5]}..."
                )

    return result


# ========================================
# 坐标范围验证（完整版）
# ========================================

def verify_coordinate_range_full(
    arr: np.ndarray,
    var_name: str,
    expected_range: Optional[Tuple[float, float]] = None
) -> Dict[str, Any]:
    """
    验证坐标变量的值范围（包括 NaN/Inf 检测和范围检测）

    Args:
        arr: 坐标数组
        var_name: 变量名
        expected_range: 预期范围 (min, max)

    Returns:
        验证结果字典
    """
    nan_count = int(np.sum(np.isnan(arr)))
    inf_count = int(np.sum(np.isinf(arr)))

    result = {
        "var_name": var_name,
        "shape": list(arr.shape),
        "actual_min": float(np.nanmin(arr)) if arr.size > nan_count else None,
        "actual_max": float(np.nanmax(arr)) if arr.size > nan_count else None,
        "expected_range": list(expected_range) if expected_range else None,
        "in_range": True,
        "has_nan": nan_count > 0,
        "has_inf": inf_count > 0,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "message": ""
    }

    if nan_count > 0:
        result["in_range"] = False
        result["message"] = f"坐标 {var_name} 包含 {nan_count} 个 NaN 值"
        return result

    if inf_count > 0:
        result["in_range"] = False
        result["message"] = f"坐标 {var_name} 包含 {inf_count} 个 Inf 值"
        return result

    if expected_range:
        min_val, max_val = expected_range
        if result["actual_min"] < min_val or result["actual_max"] > max_val:
            result["in_range"] = False
            result["message"] = (
                f"坐标 {var_name} 超出预期范围: "
                f"实际 [{result['actual_min']:.4f}, {result['actual_max']:.4f}], "
                f"预期 [{min_val}, {max_val}]"
            )
    else:
        result["message"] = f"坐标 {var_name} 范围: [{result['actual_min']:.4f}, {result['actual_max']:.4f}]"

    return result


# ========================================
# 启发式掩码验证
# ========================================

def heuristic_mask_check(
    data_arr: np.ndarray,
    mask_arr: np.ndarray,
    var_name: str,
    land_threshold: float = LAND_THRESHOLD_ABS,
    sample_size: int = HEURISTIC_SAMPLE_SIZE
) -> Dict[str, Any]:
    """
    启发式掩码验证

    逻辑:
    - 采样陆地点（mask=0），检查数据是否接近零
    - 采样海洋点（mask=1），检查数据是否有非零值
    - 如果陆地有大量非零数据，或海洋全是零，可能掩码反了

    Args:
        data_arr: 数据数组 (T, [D,] H, W)
        mask_arr: 掩码数组 (H, W)
        var_name: 变量名
        land_threshold: 陆地零值判定阈值
        sample_size: 采样数量

    Returns:
        检测结果字典
    """
    result = {
        "var_name": var_name,
        "passed": True,
        "land_nonzero_ratio": 0.0,
        "ocean_zero_ratio": 0.0,
        "warnings": [],
        "details": {}
    }

    # 获取数据的空间维度
    if data_arr.ndim == 3:  # (T, H, W)
        spatial_data = data_arr[0]  # 取第一个时间步
    elif data_arr.ndim == 4:  # (T, D, H, W)
        spatial_data = data_arr[0, 0]  # 取第一个时间步和深度
    elif data_arr.ndim == 2:  # (H, W)
        spatial_data = data_arr
    else:
        result["warnings"].append(f"无法处理 {data_arr.ndim}D 数据进行启发式检查")
        return result

    # 确保形状匹配
    if spatial_data.shape != mask_arr.shape:
        result["warnings"].append(
            f"数据空间形状 {spatial_data.shape} 与掩码形状 {mask_arr.shape} 不匹配"
        )
        return result

    # 找到陆地点和海洋点
    land_mask = (mask_arr == 0)
    ocean_mask = (mask_arr == 1)

    land_indices = np.where(land_mask.flatten())[0]
    ocean_indices = np.where(ocean_mask.flatten())[0]

    result["details"]["land_count"] = len(land_indices)
    result["details"]["ocean_count"] = len(ocean_indices)

    # 采样陆地点
    if len(land_indices) > 0:
        sample_land = min(sample_size, len(land_indices))
        rng = np.random.default_rng(42)
        land_sample_idx = rng.choice(land_indices, size=sample_land, replace=False)
        land_values = spatial_data.flatten()[land_sample_idx]

        # 检查陆地点的非零比例
        land_nonzero = np.sum(np.abs(land_values) > land_threshold)
        result["land_nonzero_ratio"] = land_nonzero / sample_land

        if result["land_nonzero_ratio"] > 0.5:
            result["warnings"].append(
                f"陆地点有 {result['land_nonzero_ratio']*100:.1f}% 非零数据，掩码可能有误"
            )

    # 采样海洋点
    if len(ocean_indices) > 0:
        sample_ocean = min(sample_size, len(ocean_indices))
        rng = np.random.default_rng(43)
        ocean_sample_idx = rng.choice(ocean_indices, size=sample_ocean, replace=False)
        ocean_values = spatial_data.flatten()[ocean_sample_idx]

        # 检查海洋点的零值比例
        ocean_zero = np.sum(np.abs(ocean_values) <= land_threshold)
        result["ocean_zero_ratio"] = ocean_zero / sample_ocean

        if result["ocean_zero_ratio"] > 0.9:
            result["warnings"].append(
                f"海洋点有 {result['ocean_zero_ratio']*100:.1f}% 为零，掩码可能反转"
            )

    # 判断是否通过
    if result["land_nonzero_ratio"] > 0.5 and result["ocean_zero_ratio"] > 0.9:
        result["passed"] = False
        result["warnings"].append("强烈怀疑掩码定义反转（陆地=1, 海洋=0）")

    return result


# ========================================
# 掩码二值性检测
# ========================================

def check_mask_binary(mask_arr: np.ndarray, mask_name: str) -> Dict[str, Any]:
    """
    检测掩码是否为二值

    Args:
        mask_arr: 掩码数组
        mask_name: 掩码名称

    Returns:
        检测结果字典
    """
    result = {
        "mask_name": mask_name,
        "is_binary": True,
        "unique_values": [],
        "warnings": []
    }

    unique_vals = np.unique(mask_arr)
    result["unique_values"] = [int(v) for v in unique_vals[:10]]  # 只记录前 10 个

    is_binary = len(unique_vals) <= 2 and all(v in [0, 1] for v in unique_vals)
    result["is_binary"] = is_binary

    if not is_binary:
        result["warnings"].append(
            f"掩码 '{mask_name}' 不是二值: unique={result['unique_values']}"
        )

    return result


# ========================================
# 批量数据质量检测
# ========================================

def run_full_quality_check(
    nc_files: List[str],
    dyn_vars: List[str],
    static_file: Optional[str] = None,
    mask_var: Optional[str] = None,
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None
) -> Dict[str, Any]:
    """
    运行完整的数据质量检测

    Args:
        nc_files: NC 文件列表
        dyn_vars: 动态变量列表
        static_file: 静态文件路径
        mask_var: 掩码变量名
        lon_range: 经度范围
        lat_range: 纬度范围

    Returns:
        检测结果字典
    """
    import xarray as xr

    results = {
        "time_check": None,
        "coord_checks": {},
        "data_quality": {},
        "mask_checks": {},
        "warnings": [],
        "errors": []
    }

    try:
        with xr.open_mfdataset(nc_files, combine='by_coords', decode_times=False) as ds:
            # 时间检测
            for time_name in ['time', 'ocean_time']:
                if time_name in ds.coords:
                    time_arr = ds[time_name].values
                    results["time_check"] = check_time_validity(time_arr)
                    break

            # 坐标检测
            for coord_name in ['latitude', 'longitude', 'lat', 'lon']:
                if coord_name in ds.coords:
                    coord_arr = ds[coord_name].values
                    results["coord_checks"][coord_name] = check_coordinate_monotonicity(coord_arr, coord_name)

            # 数据质量检测
            for var in dyn_vars:
                if var in ds.data_vars:
                    data_arr = ds[var].values
                    results["data_quality"][var] = check_data_quality(data_arr, var)

    except Exception as e:
        results["errors"].append(f"打开数据集失败: {str(e)}")

    # 静态文件检测
    if static_file:
        try:
            with xr.open_dataset(static_file) as ds_static:
                # 掩码检测
                if mask_var and mask_var in ds_static.data_vars:
                    mask_arr = ds_static[mask_var].values
                    results["mask_checks"][mask_var] = check_mask_binary(mask_arr, mask_var)

                # 坐标范围检测
                for coord_name in ['lon_rho', 'lat_rho', 'longitude', 'latitude']:
                    if coord_name in ds_static.data_vars or coord_name in ds_static.coords:
                        coord_arr = ds_static[coord_name].values
                        expected_range = lon_range if 'lon' in coord_name.lower() else lat_range
                        results["coord_checks"][coord_name] = verify_coordinate_range_full(
                            coord_arr, coord_name, expected_range
                        )

        except Exception as e:
            results["errors"].append(f"打开静态文件失败: {str(e)}")

    # 汇总警告
    if results["time_check"]:
        results["warnings"].extend(results["time_check"].get("warnings", []))
        results["errors"].extend(results["time_check"].get("errors", []))

    for check in results["coord_checks"].values():
        results["warnings"].extend(check.get("warnings", []))

    for check in results["data_quality"].values():
        results["warnings"].extend(check.get("warnings", []))

    for check in results["mask_checks"].values():
        results["warnings"].extend(check.get("warnings", []))

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据质量检测工具")
    parser.add_argument("--nc-folder", required=True, help="NC 文件目录")
    parser.add_argument("--pattern", default="*.nc", help="文件匹配模式")
    parser.add_argument("--vars", required=True, help="变量列表，逗号分隔")
    parser.add_argument("--static-file", help="静态文件路径")
    parser.add_argument("--mask-var", help="掩码变量名")
    parser.add_argument("--output", help="输出 JSON 文件")

    args = parser.parse_args()

    import glob
    import json

    nc_files = sorted(glob.glob(f"{args.nc_folder}/{args.pattern}"))
    dyn_vars = [v.strip() for v in args.vars.split(",")]

    print(f"检测文件: {len(nc_files)} 个")
    print(f"检测变量: {dyn_vars}")

    results = run_full_quality_check(
        nc_files=nc_files,
        dyn_vars=dyn_vars,
        static_file=args.static_file,
        mask_var=args.mask_var
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存: {args.output}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))
