#!/usr/bin/env python3
"""
convert_npy.py - Step C: NC 转 NPY 转换（含后置验证）

@author leizheng
@date 2026-02-02
@version 2.1.0

功能:
- 将 NC 文件中的变量转换为 NPY 格式
- 静态变量添加编号前缀（00_lon_rho, 01_lat_rho, 99_mask_rho）
- 自动生成 preprocess_manifest.json
- 执行后置验证 (Rule 1/2/3)
- NaN/Inf 检测（采样校验）
- 多网格支持（C-grid staggered mesh）
- object dtype 检查
- 启发式掩码验证（陆地/海洋采样）

用法:
    python convert_npy.py --config config.json --output result.json

配置文件格式:
{
    "nc_folder": "/path/to/nc/files",
    "output_base": "/path/to/output",
    "dyn_vars": ["uo", "vo"],
    "static_file": "/path/to/static.nc",
    "stat_vars": ["lon_rho", "lat_rho", "h", "mask_rho"],
    "mask_vars": ["mask_rho", "mask_u", "mask_v"],
    "lon_var": "lon_rho",
    "lat_var": "lat_rho",
    "dyn_file_pattern": "*.nc",
    "run_validation": true,
    "allow_nan": false,
    "lon_range": [-180, 180],
    "lat_range": [-90, 90],
    "heuristic_check_var": "uo",
    "land_threshold_abs": 1e-12,
    "heuristic_sample_size": 2000
}
"""

import argparse
import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import xarray as xr
except ImportError:
    print(json.dumps({
        "status": "error",
        "errors": ["需要安装 xarray: pip install xarray netCDF4 dask"]
    }))
    sys.exit(1)


# ========================================
# 常量定义
# ========================================

# 静态变量编号前缀
# 00-09: 经度坐标
# 10-19: 纬度坐标
# 20-89: 其他静态变量
# 90-99: 掩码变量（排最后）

LON_VARS = ['lon_rho', 'lon_u', 'lon_v', 'lon_psi']
LAT_VARS = ['lat_rho', 'lat_u', 'lat_v', 'lat_psi']
MASK_VARS_DEFAULT = ['mask_rho', 'mask_u', 'mask_v', 'mask_psi']

# 时间坐标候选
TIME_COORD_CANDIDATES = ['ocean_time', 'time', 't', 'Time', 'TIME']

# NaN/Inf 检测采样参数
NAN_CHECK_SAMPLE_SIZE = 10000  # 采样点数
NAN_CHECK_RANDOM_SEED = 42     # 固定随机种子保证可复现

# 启发式掩码验证参数
HEURISTIC_SAMPLE_SIZE = 2000   # 陆地/海洋采样点数
LAND_THRESHOLD_ABS = 1e-12     # 陆地零值判定阈值
LAND_ZERO_RATIO_MIN = 0.90     # 陆地点零值比例下限
OCEAN_ZERO_RATIO_MAX = 0.90    # 海洋点零值比例上限


# ========================================
# 辅助函数
# ========================================

def is_object_dtype(arr: np.ndarray) -> bool:
    """
    检查数组是否是 object dtype（禁止使用）

    Args:
        arr: 输入数组

    Returns:
        True 如果是 object dtype
    """
    return arr.dtype == np.object_ or arr.dtype.kind == 'O'

def check_nan_inf_sampling(arr: np.ndarray, var_name: str, sample_size: int = NAN_CHECK_SAMPLE_SIZE) -> Dict[str, Any]:
    """
    采样检测 NaN/Inf 值

    Args:
        arr: 输入数组
        var_name: 变量名（用于报告）
        sample_size: 采样数量

    Returns:
        检测结果字典
    """
    result = {
        "var_name": var_name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "has_nan": False,
        "has_inf": False,
        "nan_count": 0,
        "inf_count": 0,
        "checked_samples": 0,
        "total_elements": int(np.prod(arr.shape)),
        "pass": True
    }

    total = result["total_elements"]

    if total == 0:
        return result

    # 如果数组较小，直接全量检测
    if total <= sample_size:
        flat = arr.flatten()
        result["checked_samples"] = total
    else:
        # 采样检测
        rng = np.random.default_rng(NAN_CHECK_RANDOM_SEED)
        indices = rng.choice(total, size=sample_size, replace=False)
        flat = arr.flatten()[indices]
        result["checked_samples"] = sample_size

    # 检测 NaN
    nan_mask = np.isnan(flat)
    result["nan_count"] = int(np.sum(nan_mask))
    result["has_nan"] = result["nan_count"] > 0

    # 检测 Inf
    inf_mask = np.isinf(flat)
    result["inf_count"] = int(np.sum(inf_mask))
    result["has_inf"] = result["inf_count"] > 0

    result["pass"] = not (result["has_nan"] or result["has_inf"])

    return result


def get_spatial_shape(arr: np.ndarray) -> Tuple[int, int]:
    """
    获取数组的空间维度 (H, W)

    对于不同维度的数组：
    - 2D (H, W): 直接返回
    - 3D (T, H, W): 返回最后两维
    - 4D (T, D, H, W): 返回最后两维

    Args:
        arr: 输入数组

    Returns:
        (H, W) 元组
    """
    if arr.ndim == 2:
        return (arr.shape[0], arr.shape[1])
    elif arr.ndim >= 3:
        return (arr.shape[-2], arr.shape[-1])
    else:
        return (0, 0)


def verify_coordinate_range(
    arr: np.ndarray,
    var_name: str,
    expected_range: Optional[Tuple[float, float]] = None
) -> Dict[str, Any]:
    """
    验证坐标变量的值范围

    Args:
        arr: 坐标数组
        var_name: 变量名
        expected_range: 预期范围 (min, max)

    Returns:
        验证结果字典
    """
    # 首先检查 NaN/Inf
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

    # 坐标变量不应该有 NaN 或 Inf
    if nan_count > 0:
        result["in_range"] = False
        result["message"] = f"坐标 {var_name} 包含 {nan_count} 个 NaN 值！坐标变量不允许有 NaN"
        return result

    if inf_count > 0:
        result["in_range"] = False
        result["message"] = f"坐标 {var_name} 包含 {inf_count} 个 Inf 值！坐标变量不允许有 Inf"
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

def get_static_var_prefix(var_name: str, mask_vars: List[str], idx: int) -> str:
    """
    获取静态变量的编号前缀

    规则:
    - 经度变量: 00-09
    - 纬度变量: 10-19
    - 掩码变量: 90-99
    - 其他: 20 开始递增
    """
    if var_name in LON_VARS:
        return f"{LON_VARS.index(var_name):02d}"
    elif var_name in LAT_VARS:
        return f"{10 + LAT_VARS.index(var_name):02d}"
    elif var_name in mask_vars:
        return f"{90 + mask_vars.index(var_name):02d}"
    else:
        return f"{20 + idx:02d}"


def sha256_file(path: str) -> str:
    """计算文件的 SHA256 哈希"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_time_coord(ds: xr.Dataset) -> Optional[str]:
    """查找时间坐标"""
    for c in TIME_COORD_CANDIDATES:
        if c in ds.coords or c in ds.variables:
            return c
    # 模糊匹配
    for d in ds.dims:
        if "time" in d.lower():
            return d
    return None


def derive_mask(src_mask: np.ndarray, op: str) -> np.ndarray:
    """
    根据操作类型推导掩码

    Args:
        src_mask: 源掩码数组
        op: 操作类型
            - land_is_zero: 陆地=0, 海洋=非0 → 输出 (src == 0)
            - ocean_is_one: 海洋=1, 陆地=非1 → 输出 (src != 1)
            - identity: 原样
            - invert01: 反转 0/1

    Returns:
        推导后的掩码数组
    """
    if op == "land_is_zero":
        return (src_mask == 0).astype(np.uint8)
    elif op == "ocean_is_one":
        return (src_mask != 1).astype(np.uint8)
    elif op == "identity":
        return np.asarray(src_mask)
    elif op == "invert01":
        return (1 - src_mask).astype(np.uint8)
    else:
        return np.asarray(src_mask)


def derive_staggered_mask(mask_rho: np.ndarray, grid_type: str) -> np.ndarray:
    """
    从 rho 网格掩码派生 u/v 网格掩码（C-grid 交错网格）

    原理：相邻两个 rho 点都为海洋，u/v 点才算海洋

    Args:
        mask_rho: rho 网格掩码，shape (Y, X)，海洋=1，陆地=0
        grid_type: 目标网格类型 "u" 或 "v"

    Returns:
        派生的掩码数组
        - u 网格: shape (Y, X-1)
        - v 网格: shape (Y-1, X)
    """
    if grid_type == "u":
        # u 在 xi 方向中点，相邻两个 rho 点都为海洋才算海洋
        # mask_u[j, i] = mask_rho[j, i] & mask_rho[j, i+1]
        return (mask_rho[:, :-1] * mask_rho[:, 1:]).astype(np.uint8)
    elif grid_type == "v":
        # v 在 eta 方向中点，相邻两个 rho 点都为海洋才算海洋
        # mask_v[j, i] = mask_rho[j, i] & mask_rho[j+1, i]
        return (mask_rho[:-1, :] * mask_rho[1:, :]).astype(np.uint8)
    else:
        return mask_rho


def heuristic_mask_check(
    mask_arr: np.ndarray,
    dyn_arr: np.ndarray,
    var_name: str,
    sample_size: int = HEURISTIC_SAMPLE_SIZE,
    land_threshold: float = LAND_THRESHOLD_ABS,
    land_zero_min: float = LAND_ZERO_RATIO_MIN,
    ocean_zero_max: float = OCEAN_ZERO_RATIO_MAX
) -> Dict[str, Any]:
    """
    启发式掩码验证：采样陆地/海洋点，检查动态变量零值比例

    原理：
    - 在陆地点（mask==0），动态变量应该大部分为零或 NaN
    - 在海洋点（mask!=0），动态变量应该大部分非零

    Args:
        mask_arr: 掩码数组 (2D, land=0, ocean=1)
        dyn_arr: 动态变量数组 (3D or 4D，取第一个时间步)
        var_name: 变量名（用于报告）
        sample_size: 采样点数
        land_threshold: 零值判定阈值（|value| <= threshold 视为零）
        land_zero_min: 陆地点零值比例下限（默认 0.90）
        ocean_zero_max: 海洋点零值比例上限（默认 0.90）

    Returns:
        检查结果字典
    """
    result = {
        "var_name": var_name,
        "passed": True,
        "land_zero_ratio": None,
        "ocean_zero_ratio": None,
        "land_samples": 0,
        "ocean_samples": 0,
        "warnings": [],
        "details": {}
    }

    # 获取动态变量的第一个时间步，并取最后两个空间维度
    if dyn_arr.ndim == 3:
        # (T, H, W) -> 取 t=0
        dyn_slice = dyn_arr[0, :, :]
    elif dyn_arr.ndim == 4:
        # (T, D, H, W) -> 取 t=0, 深度取平均或第一层
        dyn_slice = dyn_arr[0, 0, :, :]  # 取第一层
    elif dyn_arr.ndim == 2:
        dyn_slice = dyn_arr
    else:
        result["warnings"].append(f"不支持的维度: {dyn_arr.ndim}D")
        return result

    # 确保形状匹配
    if dyn_slice.shape != mask_arr.shape:
        result["warnings"].append(
            f"形状不匹配: dyn_slice {dyn_slice.shape} vs mask {mask_arr.shape}"
        )
        return result

    # 展平数组
    mask_flat = mask_arr.flatten()
    dyn_flat = dyn_slice.flatten()

    # 找到陆地点和海洋点的索引
    land_indices = np.where(mask_flat == 0)[0]
    ocean_indices = np.where(mask_flat != 0)[0]

    result["details"]["total_land_points"] = len(land_indices)
    result["details"]["total_ocean_points"] = len(ocean_indices)

    rng = np.random.default_rng(NAN_CHECK_RANDOM_SEED)

    # 采样陆地点
    if len(land_indices) > 0:
        n_land = min(sample_size, len(land_indices))
        land_sample_idx = rng.choice(land_indices, size=n_land, replace=False)
        land_values = dyn_flat[land_sample_idx]

        # 计算零值比例（|value| <= threshold 或 NaN）
        land_zero_mask = (np.abs(land_values) <= land_threshold) | np.isnan(land_values)
        land_zero_ratio = float(np.sum(land_zero_mask)) / n_land

        result["land_samples"] = n_land
        result["land_zero_ratio"] = land_zero_ratio
        result["details"]["land_threshold"] = land_threshold

        if land_zero_ratio < land_zero_min:
            result["warnings"].append(
                f"陆地点零值比例过低: {land_zero_ratio:.2%} < {land_zero_min:.0%}，"
                f"掩码可能不正确或 land_threshold 设置过小"
            )
            result["passed"] = False

    # 采样海洋点
    if len(ocean_indices) > 0:
        n_ocean = min(sample_size, len(ocean_indices))
        ocean_sample_idx = rng.choice(ocean_indices, size=n_ocean, replace=False)
        ocean_values = dyn_flat[ocean_sample_idx]

        # 计算零值比例
        ocean_zero_mask = (np.abs(ocean_values) <= land_threshold) | np.isnan(ocean_values)
        ocean_zero_ratio = float(np.sum(ocean_zero_mask)) / n_ocean

        result["ocean_samples"] = n_ocean
        result["ocean_zero_ratio"] = ocean_zero_ratio

        if ocean_zero_ratio > ocean_zero_max:
            result["warnings"].append(
                f"海洋点零值比例过高: {ocean_zero_ratio:.2%} > {ocean_zero_max:.0%}，"
                f"掩码可能反转了"
            )
            result["passed"] = False

    return result


# ========================================
# 转换函数
# ========================================

def convert_dynamic_vars(
    nc_files: List[str],
    dyn_vars: List[str],
    output_dir: str,
    result: Dict[str, Any],
    allow_nan: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    转换动态变量

    Args:
        nc_files: NC 文件列表
        dyn_vars: 动态变量列表
        output_dir: 输出目录
        result: 结果字典（用于添加警告/错误）
        allow_nan: 是否允许 NaN/Inf 值

    Returns:
        保存的文件信息字典
    """
    dyn_out_dir = os.path.join(output_dir, 'target_variables')
    os.makedirs(dyn_out_dir, exist_ok=True)

    saved_files = {}
    time_lengths = {}
    nan_check_results = {}

    print(f"处理动态变量，文件数: {len(nc_files)}", file=sys.stderr)

    try:
        # 使用 open_mfdataset 进行多文件惰性加载
        with xr.open_mfdataset(
            nc_files,
            combine='by_coords',
            chunks='auto',
            parallel=True,
            decode_times=False
        ) as ds:
            # ========== 坐标变量 NaN/Inf 检测（核心检测）==========
            coord_var_names = ['latitude', 'longitude', 'lat', 'lon', 'time', 'depth',
                              'lat_rho', 'lon_rho', 'lat_u', 'lon_u', 'lat_v', 'lon_v',
                              'ocean_time', 's_rho', 's_w']
            for coord_name in coord_var_names:
                if coord_name in ds.coords or coord_name in ds.dims:
                    try:
                        coord_arr = ds[coord_name].values if coord_name in ds.coords else None
                        if coord_arr is not None:
                            nan_count = int(np.sum(np.isnan(coord_arr)))
                            inf_count = int(np.sum(np.isinf(coord_arr)))
                            if nan_count > 0 or inf_count > 0:
                                msg = f"坐标变量 '{coord_name}' 包含非法值: NaN={nan_count}, Inf={inf_count}。坐标变量不允许有 NaN/Inf"
                                result["errors"].append(msg)
                                print(f"  错误: {msg}", file=sys.stderr)
                    except Exception as e:
                        print(f"  警告: 检查坐标 {coord_name} 时出错: {e}", file=sys.stderr)

            # 如果坐标有严重错误，直接返回
            if result["errors"]:
                return saved_files

            for var in dyn_vars:
                if var not in ds.data_vars and var not in ds.coords:
                    result["warnings"].append(f"动态变量 '{var}' 不存在于数据集中")
                    continue

                out_fp = os.path.join(dyn_out_dir, f"{var}.npy")

                try:
                    print(f"  提取动态变量: {var} ...", file=sys.stderr)
                    data_arr = ds[var].values

                    # object dtype 检查（禁止使用）
                    if is_object_dtype(data_arr):
                        msg = f"动态变量 '{var}' 是 object dtype，禁止使用"
                        result["errors"].append(msg)
                        print(f"    错误: {msg}", file=sys.stderr)
                        continue

                    # ========== 核心维度检测 ==========
                    # 检测零长度维度
                    if any(d == 0 for d in data_arr.shape):
                        zero_dims = [i for i, d in enumerate(data_arr.shape) if d == 0]
                        msg = f"动态变量 '{var}' 有零长度维度，位置: {zero_dims}"
                        result["errors"].append(msg)
                        print(f"    错误: {msg}", file=sys.stderr)
                        continue

                    # 检测维度数量（动态变量应该是 3D 或 4D）
                    if data_arr.ndim not in [3, 4]:
                        msg = f"动态变量 '{var}' 维度数量错误: 实际 {data_arr.ndim}D，预期 3D 或 4D"
                        result["errors"].append(msg)
                        print(f"    错误: {msg}", file=sys.stderr)
                        continue

                    # NaN/Inf 采样检测
                    nan_result = check_nan_inf_sampling(data_arr, var)
                    nan_check_results[var] = nan_result

                    if not nan_result["pass"]:
                        msg = f"动态变量 '{var}' 含有非法值: NaN={nan_result['nan_count']}, Inf={nan_result['inf_count']}"
                        if allow_nan:
                            result["warnings"].append(msg + " (allow_nan=True, 允许)")
                        else:
                            result["errors"].append(msg)
                            print(f"    错误: {msg}", file=sys.stderr)
                            continue

                    np.save(out_fp, data_arr)

                    # 记录时间长度
                    if data_arr.ndim >= 1:
                        time_lengths[var] = data_arr.shape[0]

                    # 获取空间维度
                    spatial_shape = get_spatial_shape(data_arr)

                    # 维度解释
                    ndim = data_arr.ndim
                    if ndim == 3:
                        interp = f"[T={data_arr.shape[0]}, H={data_arr.shape[1]}, W={data_arr.shape[2]}]"
                    elif ndim == 4:
                        interp = f"[T={data_arr.shape[0]}, D={data_arr.shape[1]}, H={data_arr.shape[2]}, W={data_arr.shape[3]}]"
                    else:
                        interp = f"shape={data_arr.shape}"

                    saved_files[var] = {
                        "path": out_fp,
                        "filename": f"{var}.npy",
                        "shape": list(data_arr.shape),
                        "spatial_shape": list(spatial_shape),  # 新增：空间维度
                        "dtype": str(data_arr.dtype),
                        "interpretation": interp,
                        "is_dynamic": True,
                        "nan_check": nan_result
                    }

                    print(f"    完成，shape={data_arr.shape}, spatial={spatial_shape}", file=sys.stderr)
                    del data_arr

                except Exception as e:
                    result["warnings"].append(f"处理动态变量 '{var}' 失败: {str(e)}")

    except Exception as e:
        result["errors"].append(f"打开动态数据集失败: {str(e)}")

    result["time_lengths"] = time_lengths
    result["nan_check_results"] = nan_check_results
    return saved_files


def convert_static_vars(
    static_file: str,
    stat_vars: List[str],
    mask_vars: List[str],
    output_dir: str,
    result: Dict[str, Any],
    allow_nan: bool = False,
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    转换静态变量（带编号前缀）

    Args:
        static_file: 静态文件路径
        stat_vars: 静态变量列表
        mask_vars: 掩码变量列表
        output_dir: 输出目录
        result: 结果字典（用于添加警告/错误）
        allow_nan: 是否允许 NaN/Inf 值
        lon_range: 经度有效范围 (min, max)
        lat_range: 纬度有效范围 (min, max)

    Returns:
        保存的文件信息字典
    """
    sta_out_dir = os.path.join(output_dir, 'static_variables')
    os.makedirs(sta_out_dir, exist_ok=True)

    saved_files = {}
    other_idx = 0  # 用于其他静态变量的编号
    coord_checks = {}  # 坐标范围检查结果

    if not static_file or not os.path.exists(static_file):
        result["warnings"].append(f"静态文件不存在: {static_file}")
        return saved_files

    print(f"处理静态变量: {static_file}", file=sys.stderr)

    try:
        with xr.open_dataset(static_file, decode_times=False) as ds:
            for var in stat_vars:
                if var not in ds.variables:
                    result["warnings"].append(f"静态变量 '{var}' 不存在于静态文件中")
                    continue

                # 获取编号前缀
                prefix = get_static_var_prefix(var, mask_vars, other_idx)
                if var not in LON_VARS and var not in LAT_VARS and var not in mask_vars:
                    other_idx += 1

                filename = f"{prefix}_{var}.npy"
                out_fp = os.path.join(sta_out_dir, filename)

                try:
                    print(f"  提取静态变量: {var} -> {filename} ...", file=sys.stderr)
                    data_arr = ds[var].values

                    is_mask = var in mask_vars
                    is_lon = var in LON_VARS
                    is_lat = var in LAT_VARS

                    # object dtype 检查（禁止使用）
                    if is_object_dtype(data_arr):
                        msg = f"静态变量 '{var}' 是 object dtype，禁止使用"
                        result["errors"].append(msg)
                        print(f"    错误: {msg}", file=sys.stderr)
                        continue

                    # NaN/Inf 采样检测（掩码变量跳过）
                    nan_result = None
                    if not is_mask:
                        nan_result = check_nan_inf_sampling(data_arr, var)
                        if not nan_result["pass"]:
                            msg = f"静态变量 '{var}' 含有非法值: NaN={nan_result['nan_count']}, Inf={nan_result['inf_count']}"
                            if allow_nan:
                                result["warnings"].append(msg + " (allow_nan=True, 允许)")
                            else:
                                result["errors"].append(msg)
                                print(f"    错误: {msg}", file=sys.stderr)
                                continue

                    # 坐标范围检查（包括 NaN/Inf 检测）
                    if is_lon and lon_range:
                        coord_check = verify_coordinate_range(data_arr, var, lon_range)
                        coord_checks[var] = coord_check
                        if coord_check.get("has_nan") or coord_check.get("has_inf"):
                            # 坐标变量有 NaN/Inf 是严重错误
                            result["errors"].append(coord_check["message"])
                            print(f"    错误: {coord_check['message']}", file=sys.stderr)
                            continue
                        elif not coord_check["in_range"]:
                            result["warnings"].append(coord_check["message"])
                        if coord_check["actual_min"] is not None:
                            print(f"    经度范围: [{coord_check['actual_min']:.4f}, {coord_check['actual_max']:.4f}]", file=sys.stderr)

                    if is_lat and lat_range:
                        coord_check = verify_coordinate_range(data_arr, var, lat_range)
                        coord_checks[var] = coord_check
                        if coord_check.get("has_nan") or coord_check.get("has_inf"):
                            # 坐标变量有 NaN/Inf 是严重错误
                            result["errors"].append(coord_check["message"])
                            print(f"    错误: {coord_check['message']}", file=sys.stderr)
                            continue
                        elif not coord_check["in_range"]:
                            result["warnings"].append(coord_check["message"])
                        if coord_check["actual_min"] is not None:
                            print(f"    纬度范围: [{coord_check['actual_min']:.4f}, {coord_check['actual_max']:.4f}]", file=sys.stderr)

                    # 掩码二值性检查
                    if is_mask:
                        unique_vals = np.unique(data_arr)
                        is_binary = len(unique_vals) <= 2 and all(v in [0, 1] for v in unique_vals)
                        if not is_binary:
                            result["warnings"].append(
                                f"掩码变量 '{var}' 不是二值 (0/1): 唯一值 = {unique_vals[:10].tolist()}"
                                + ("..." if len(unique_vals) > 10 else "")
                            )

                    np.save(out_fp, data_arr)

                    # 获取空间维度
                    spatial_shape = get_spatial_shape(data_arr)

                    saved_files[var] = {
                        "path": out_fp,
                        "filename": filename,
                        "prefix": prefix,
                        "shape": list(data_arr.shape),
                        "spatial_shape": list(spatial_shape),
                        "dtype": str(data_arr.dtype),
                        "is_mask": is_mask,
                        "is_lon": is_lon,
                        "is_lat": is_lat,
                        "is_dynamic": False,
                        "nan_check": nan_result,
                        "coord_check": coord_checks.get(var)
                    }

                    print(f"    完成，shape={data_arr.shape}", file=sys.stderr)

                except Exception as e:
                    result["warnings"].append(f"保存静态变量 '{var}' 失败: {str(e)}")

    except Exception as e:
        result["errors"].append(f"读取静态文件失败: {str(e)}")

    result["coord_checks"] = coord_checks
    return saved_files


# ========================================
# 后置验证 (Rule 1/2/3)
# ========================================

def validate_rule1(
    output_dir: str,
    dyn_vars: List[str],
    stat_vars: List[str],
    mask_vars: List[str],
    lon_var: str,
    lat_var: str,
    saved_files: Dict[str, Dict[str, Any]],
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Rule 1: 输出完整性与形状约定

    检查项:
    - 1.1 目录结构存在
    - 1.2 预期文件完整
    - 1.3 网格形状一致
    - 1.4 掩码文件排最后（字典序）
    - 1.5 动态变量时间长度一致
    """
    validation = {
        "rule": "Rule1",
        "description": "输出完整性与形状约定",
        "passed": True,
        "errors": [],
        "warnings": [],
        "details": {}
    }

    dyn_out = os.path.join(output_dir, 'target_variables')
    sta_out = os.path.join(output_dir, 'static_variables')

    # 1.1 检查目录存在
    if not os.path.isdir(dyn_out):
        validation["errors"].append(f"目录不存在: {dyn_out}")
        validation["passed"] = False
    if not os.path.isdir(sta_out):
        validation["errors"].append(f"目录不存在: {sta_out}")
        validation["passed"] = False

    if not validation["passed"]:
        return validation

    # 1.2 检查预期文件完整
    # 动态变量
    for var in dyn_vars:
        expected = os.path.join(dyn_out, f"{var}.npy")
        if not os.path.exists(expected):
            validation["errors"].append(f"缺少动态变量文件: {var}.npy")
            validation["passed"] = False

    # 静态变量（需要匹配带前缀的文件名）
    sta_files = sorted(os.listdir(sta_out)) if os.path.isdir(sta_out) else []
    validation["details"]["static_files"] = sta_files

    for var in stat_vars:
        # 查找匹配的文件
        found = any(f.endswith(f"_{var}.npy") for f in sta_files)
        if not found:
            validation["warnings"].append(f"缺少静态变量文件: *_{var}.npy")

    # 1.3 检查网格形状
    grid_shape = None
    for var in [lon_var, lat_var]:
        if var in saved_files:
            shape = tuple(saved_files[var]["shape"])
            if grid_shape is None:
                grid_shape = shape
            elif shape != grid_shape:
                validation["errors"].append(
                    f"网格形状不一致: {var} shape={shape} != 预期 {grid_shape}"
                )
                validation["passed"] = False

    validation["details"]["grid_shape"] = list(grid_shape) if grid_shape else None

    # 1.4 检查掩码文件排最后
    if sta_files:
        mask_files = [f for f in sta_files if any(f.endswith(f"_{m}.npy") for m in mask_vars)]
        other_files = [f for f in sta_files if f not in mask_files]

        if mask_files and other_files:
            last_other = sorted(other_files)[-1] if other_files else ""
            first_mask = sorted(mask_files)[0] if mask_files else ""

            if first_mask < last_other:
                validation["warnings"].append(
                    f"掩码文件应排在最后: {first_mask} < {last_other}"
                )

    # 1.5 检查时间长度一致
    time_lengths = result.get("time_lengths", {})
    if time_lengths:
        unique_lens = set(time_lengths.values())
        if len(unique_lens) > 1:
            validation["errors"].append(
                f"动态变量时间长度不一致: {time_lengths}"
            )
            validation["passed"] = False
        else:
            validation["details"]["time_length"] = list(unique_lens)[0] if unique_lens else None

    # 1.6 检查动态变量空间形状一致性
    dyn_spatial_shapes = {}
    for var in dyn_vars:
        if var in saved_files and "spatial_shape" in saved_files[var]:
            dyn_spatial_shapes[var] = tuple(saved_files[var]["spatial_shape"])

    if dyn_spatial_shapes:
        unique_spatial = set(dyn_spatial_shapes.values())
        if len(unique_spatial) > 1:
            validation["errors"].append(
                f"动态变量空间形状不一致: {dyn_spatial_shapes}"
            )
            validation["passed"] = False
        else:
            validation["details"]["dyn_spatial_shape"] = list(list(unique_spatial)[0]) if unique_spatial else None

    # 1.7 检查动态变量与网格坐标的空间形状匹配
    if grid_shape and dyn_spatial_shapes:
        for var, spatial in dyn_spatial_shapes.items():
            if spatial != grid_shape:
                validation["warnings"].append(
                    f"动态变量 '{var}' 空间形状 {spatial} 与网格形状 {grid_shape} 不匹配"
                )
            validation["details"]["time_length"] = list(unique_lens)[0]

    return validation


def validate_rule2(
    output_dir: str,
    static_file: str,
    mask_vars: List[str],
    mask_src_var: str,
    mask_derive_op: str,
    saved_files: Dict[str, Dict[str, Any]],
    result: Dict[str, Any],
    heuristic_check_var: Optional[str] = None,
    land_threshold: float = LAND_THRESHOLD_ABS,
    heuristic_sample_size: int = HEURISTIC_SAMPLE_SIZE
) -> Dict[str, Any]:
    """
    Rule 2: 掩码不可变性检查

    检查项:
    - 2.1 掩码形状与网格一致（使用空间维度比较）
    - 2.2 掩码与源文件精确对比
    - 2.3 掩码二值性检查
    - 2.4 动态变量与对应掩码的空间维度匹配
    - 2.5 启发式掩码验证（陆地/海洋采样检查）
    """
    validation = {
        "rule": "Rule2",
        "description": "掩码不可变性检查",
        "passed": True,
        "errors": [],
        "warnings": [],
        "details": {}
    }

    sta_out = os.path.join(output_dir, 'static_variables')

    # 构建掩码空间形状映射（用于多网格支持）
    # 例如: mask_rho -> (400, 800), mask_u -> (400, 799), mask_v -> (399, 800)
    mask_spatial_shapes = {}
    mask_arrays = {}  # 缓存掩码数组，用于派生和启发式检查

    for mask_var in mask_vars:
        if mask_var in saved_files:
            spatial = tuple(saved_files[mask_var].get("spatial_shape", saved_files[mask_var]["shape"]))
            mask_spatial_shapes[mask_var] = spatial
            validation["details"][f"{mask_var}_spatial_shape"] = list(spatial)
            # 加载掩码数组（添加异常处理）
            try:
                mask_path = saved_files[mask_var]["path"]
                if os.path.exists(mask_path):
                    mask_arrays[mask_var] = np.load(mask_path)
                else:
                    validation["warnings"].append(f"掩码文件不存在: {mask_path}")
            except Exception as e:
                validation["warnings"].append(f"加载掩码 '{mask_var}' 失败: {str(e)}")

    # 如果有 mask_rho 但没有 mask_u/mask_v，自动派生
    derived_masks = {}
    if 'mask_rho' in mask_arrays:
        mask_rho = mask_arrays['mask_rho']
        rho_shape = mask_rho.shape

        # 派生 mask_u（如果没有）
        if 'mask_u' not in mask_spatial_shapes:
            derived_u = derive_staggered_mask(mask_rho, 'u')
            derived_masks['mask_u'] = derived_u
            mask_spatial_shapes['mask_u'] = derived_u.shape
            mask_arrays['mask_u'] = derived_u
            validation["details"]["mask_u_derived"] = True
            validation["details"]["mask_u_spatial_shape"] = list(derived_u.shape)
            print(f"  从 mask_rho {rho_shape} 派生 mask_u {derived_u.shape}", file=sys.stderr)

        # 派生 mask_v（如果没有）
        if 'mask_v' not in mask_spatial_shapes:
            derived_v = derive_staggered_mask(mask_rho, 'v')
            derived_masks['mask_v'] = derived_v
            mask_spatial_shapes['mask_v'] = derived_v.shape
            mask_arrays['mask_v'] = derived_v
            validation["details"]["mask_v_derived"] = True
            validation["details"]["mask_v_spatial_shape"] = list(derived_v.shape)
            print(f"  从 mask_rho {rho_shape} 派生 mask_v {derived_v.shape}", file=sys.stderr)

    # 根据变量名后缀确定使用哪个掩码
    # 例如: uo -> mask_u, vo -> mask_v, temp -> mask_rho
    def get_mask_for_var(var_name: str) -> Optional[str]:
        """根据变量名推断应使用的掩码"""
        var_lower = var_name.lower()
        if var_lower.endswith('_u') or var_lower in ['uo', 'u']:
            return 'mask_u'
        elif var_lower.endswith('_v') or var_lower in ['vo', 'v']:
            return 'mask_v'
        elif var_lower.endswith('_psi'):
            return 'mask_psi'
        else:
            return 'mask_rho'

    # 2.4 检查动态变量与对应掩码的空间维度匹配
    dyn_spatial_checks = {}
    for var, info in saved_files.items():
        if not info.get("is_dynamic"):
            continue

        # 获取动态变量的空间维度（最后两维）
        var_spatial = tuple(info.get("spatial_shape", info["shape"][-2:] if len(info["shape"]) >= 2 else []))

        # 推断应使用的掩码
        expected_mask = get_mask_for_var(var)
        if expected_mask and expected_mask in mask_spatial_shapes:
            mask_spatial = mask_spatial_shapes[expected_mask]
            match = (var_spatial == mask_spatial)

            # 检查是否是派生的掩码
            is_derived = expected_mask in derived_masks

            dyn_spatial_checks[var] = {
                "var_spatial": list(var_spatial),
                "expected_mask": expected_mask,
                "mask_spatial": list(mask_spatial),
                "mask_derived": is_derived,
                "match": match
            }

            if not match:
                # 提供更详细的错误信息
                if expected_mask in ['mask_u', 'mask_v'] and 'mask_rho' in mask_spatial_shapes:
                    rho_shape = mask_spatial_shapes['mask_rho']
                    hint = f"（mask_rho 形状为 {rho_shape}，派生的 {expected_mask} 形状为 {mask_spatial}）"
                else:
                    hint = ""

                validation["errors"].append(
                    f"动态变量 '{var}' 空间维度 {var_spatial} 与掩码 '{expected_mask}' "
                    f"空间维度 {mask_spatial} 不匹配{hint}"
                )
                validation["passed"] = False
        else:
            dyn_spatial_checks[var] = {
                "var_spatial": list(var_spatial),
                "expected_mask": expected_mask,
                "mask_spatial": None,
                "match": None,
                "note": f"未找到掩码 {expected_mask}，且无法从 mask_rho 派生"
            }
            validation["warnings"].append(
                f"动态变量 '{var}' 未找到对应掩码 '{expected_mask}'，"
                f"且没有 mask_rho 可用于派生"
            )

    validation["details"]["dyn_spatial_checks"] = dyn_spatial_checks

    # 2.2 精确对比（如果有源文件）
    if static_file and os.path.exists(static_file) and mask_src_var:
        try:
            with xr.open_dataset(static_file, decode_times=False) as ds:
                if mask_src_var in ds.variables:
                    src_mask = ds[mask_src_var].values
                    expected_mask = derive_mask(src_mask, mask_derive_op)

                    # 查找输出掩码文件
                    if mask_src_var in saved_files:
                        out_mask_path = saved_files[mask_src_var]["path"]
                        out_mask = np.load(out_mask_path)

                        if expected_mask.shape != out_mask.shape:
                            validation["errors"].append(
                                f"掩码形状不匹配: 预期 {expected_mask.shape}, 实际 {out_mask.shape}"
                            )
                            validation["passed"] = False
                        else:
                            mismatch = int(np.count_nonzero(expected_mask != out_mask))
                            validation["details"]["mismatch_cells"] = mismatch

                            if mismatch > 0:
                                validation["errors"].append(
                                    f"掩码被修改，不匹配像素数: {mismatch}"
                                )
                                validation["passed"] = False
                else:
                    validation["warnings"].append(
                        f"源掩码变量 '{mask_src_var}' 不存在于静态文件中"
                    )
        except Exception as e:
            validation["warnings"].append(f"掩码对比失败: {str(e)}")
    else:
        validation["warnings"].append("无法进行掩码精确对比（缺少源文件或配置）")

    # 2.5 启发式掩码验证（陆地/海洋采样检查）
    if heuristic_check_var:
        print(f"  执行启发式掩码验证: {heuristic_check_var}", file=sys.stderr)

        # 根据变量名确定应使用的掩码
        check_mask_name = get_mask_for_var(heuristic_check_var)

        # 优先使用缓存的掩码数组（包括派生的）
        if check_mask_name in mask_arrays:
            mask_arr = mask_arrays[check_mask_name]
            mask_source = "派生" if check_mask_name in derived_masks else "原始"
            print(f"    使用掩码: {check_mask_name} ({mask_source}), shape={mask_arr.shape}", file=sys.stderr)

            # 查找检查变量
            if heuristic_check_var in saved_files:
                dyn_path = saved_files[heuristic_check_var]["path"]
                dyn_arr = np.load(dyn_path)

                # 执行启发式检查
                heuristic_result = heuristic_mask_check(
                    mask_arr=mask_arr,
                    dyn_arr=dyn_arr,
                    var_name=heuristic_check_var,
                    sample_size=heuristic_sample_size,
                    land_threshold=land_threshold
                )

                heuristic_result["mask_used"] = check_mask_name
                heuristic_result["mask_derived"] = check_mask_name in derived_masks
                validation["details"]["heuristic_check"] = heuristic_result

                if not heuristic_result["passed"]:
                    for warn in heuristic_result["warnings"]:
                        validation["warnings"].append(f"启发式检查: {warn}")

                print(f"    陆地零值比例: {heuristic_result.get('land_zero_ratio', 'N/A')}", file=sys.stderr)
                print(f"    海洋零值比例: {heuristic_result.get('ocean_zero_ratio', 'N/A')}", file=sys.stderr)
            else:
                validation["warnings"].append(f"启发式检查变量 '{heuristic_check_var}' 未找到")
        else:
            validation["warnings"].append(
                f"启发式检查无法执行：未找到掩码 '{check_mask_name}'"
            )

    return validation


def validate_rule3(
    output_dir: str,
    nc_folder: str,
    nc_files: List[str],
    dyn_vars: List[str],
    stat_vars: List[str],
    dyn_file_pattern: str,
    static_file: str,
    result: Dict[str, Any],
    require_sorted: bool = True
) -> Dict[str, Any]:
    """
    Rule 3: 排序确定性

    检查项:
    - 3.1 生成 preprocess_manifest.json
    - 3.2 验证 NC 文件是否字典序排序
    - 3.3 NC 文件时间单调
    - 3.4 时间间隔一致性检查（检测 gap/重复）
    """
    validation = {
        "rule": "Rule3",
        "description": "排序确定性检查",
        "passed": True,
        "errors": [],
        "warnings": [],
        "details": {}
    }

    # 3.1 检查文件是否字典序排序
    is_sorted = nc_files == sorted(nc_files)
    validation["details"]["nc_files_sorted"] = is_sorted

    if not is_sorted and require_sorted:
        validation["errors"].append(
            "NC 文件列表未按字典序排序，这可能导致时间顺序错误"
        )
        validation["passed"] = False
    elif not is_sorted:
        validation["warnings"].append(
            "NC 文件列表未按字典序排序（require_sorted=False，仅警告）"
        )

    # 3.2 生成 manifest
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dyn_dir": nc_folder,
        "dyn_file_pattern": dyn_file_pattern,
        "stat_file": static_file or "",
        "output_dir": output_dir,
        "dyn_vars": dyn_vars,
        "stat_vars": stat_vars,
        "nc_files": [os.path.basename(f) for f in nc_files],
        "nc_files_full": nc_files,
        "sorted_lexicographic": nc_files == sorted(nc_files),
        "file_count": len(nc_files)
    }

    manifest_path = os.path.join(output_dir, "preprocess_manifest.json")
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        validation["details"]["manifest_path"] = manifest_path
        print(f"已生成 manifest: {manifest_path}", file=sys.stderr)
    except Exception as e:
        validation["warnings"].append(f"生成 manifest 失败: {str(e)}")

    # 3.2 & 3.3 检查时间单调性和间隔一致性
    if nc_files:
        try:
            time_values = []  # 所有时间值（展平）
            time_ranges = []
            time_coord = None

            # 检查前 50 个文件
            for fp in nc_files[:50]:
                with xr.open_dataset(fp, decode_times=False) as ds:
                    if time_coord is None:
                        time_coord = find_time_coord(ds)

                    if time_coord and time_coord in ds.variables:
                        coord = ds[time_coord]
                        if coord.ndim == 0:
                            t_val = float(np.asarray(coord.values))
                            time_values.append(t_val)
                            time_ranges.append((t_val, t_val, os.path.basename(fp)))
                        else:
                            dim = coord.dims[0]
                            t_arr = np.asarray(coord.values).flatten()
                            time_values.extend(t_arr.tolist())
                            t0 = float(t_arr[0])
                            t1 = float(t_arr[-1])
                            time_ranges.append((t0, t1, os.path.basename(fp)))

            validation["details"]["time_coord"] = time_coord
            validation["details"]["time_ranges_checked"] = len(time_ranges)
            validation["details"]["total_time_steps"] = len(time_values)

            # 检查单调性
            if len(time_ranges) > 1:
                monotonic = True
                for i in range(1, len(time_ranges)):
                    if time_ranges[i][0] < time_ranges[i-1][1]:
                        monotonic = False
                        break

                if not monotonic:
                    validation["errors"].append(
                        "NC 文件时间范围不单调，可能排序错误或文件名不一致"
                    )
                    validation["passed"] = False
                else:
                    validation["details"]["time_monotonic"] = True

            # 3.3 检查时间间隔一致性（检测 gap/重复）
            if len(time_values) > 2:
                time_arr = np.array(time_values)
                diffs = np.diff(time_arr)

                # 计算间隔统计
                median_diff = float(np.median(diffs))
                min_diff = float(np.min(diffs))
                max_diff = float(np.max(diffs))

                validation["details"]["time_interval"] = {
                    "median": median_diff,
                    "min": min_diff,
                    "max": max_diff
                }

                # 检测重复（间隔 <= 0）
                dup_count = int(np.sum(diffs <= 0))
                if dup_count > 0:
                    validation["warnings"].append(
                        f"检测到 {dup_count} 处时间重复或倒退（间隔 <= 0）"
                    )
                    validation["details"]["duplicate_count"] = dup_count

                # 检测异常间隔（超过中位数 2 倍）
                if median_diff > 0:
                    gap_threshold = median_diff * 2
                    large_gaps = np.where(diffs > gap_threshold)[0]
                    if len(large_gaps) > 0:
                        validation["warnings"].append(
                            f"检测到 {len(large_gaps)} 处异常大间隔（> {gap_threshold:.2f}）"
                        )
                        validation["details"]["large_gap_indices"] = large_gaps[:10].tolist()

        except Exception as e:
            validation["warnings"].append(f"时间单调性检查失败: {str(e)}")

    return validation


# ========================================
# 主函数
# ========================================

def convert_npy(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行 NC 转 NPY 转换

    Args:
        config: 配置字典

    Returns:
        转换结果字典
    """
    # 读取配置
    nc_folder = config.get("nc_folder", "")
    output_base = config.get("output_base", "")
    dyn_vars = config.get("dyn_vars", [])
    static_file = config.get("static_file")
    stat_vars = config.get("stat_vars", [])
    mask_vars = config.get("mask_vars", MASK_VARS_DEFAULT)
    lon_var = config.get("lon_var", "lon_rho")
    lat_var = config.get("lat_var", "lat_rho")
    dyn_file_pattern = config.get("dyn_file_pattern", "*.nc")
    run_validation = config.get("run_validation", True)
    mask_src_var = config.get("mask_src_var", "mask_rho")
    mask_derive_op = config.get("mask_derive_op", "identity")

    # 新增 P0 参数
    allow_nan = config.get("allow_nan", False)
    lon_range = config.get("lon_range")  # 可选，如 [-180, 180]
    lat_range = config.get("lat_range")  # 可选，如 [-90, 90]

    # 启发式验证参数
    heuristic_check_var = config.get("heuristic_check_var")  # 用于启发式验证的动态变量
    land_threshold = config.get("land_threshold_abs", LAND_THRESHOLD_ABS)
    heuristic_sample_size = config.get("heuristic_sample_size", HEURISTIC_SAMPLE_SIZE)

    # Rule 3 参数
    require_sorted = config.get("require_sorted", True)

    # 转换为元组
    if lon_range and isinstance(lon_range, list):
        lon_range = tuple(lon_range)
    if lat_range and isinstance(lat_range, list):
        lat_range = tuple(lat_range)

    result = {
        "status": "pending",
        "output_dir": output_base,
        "saved_files": {},
        "post_validation": {},
        "warnings": [],
        "errors": [],
        "message": "",
        "config": {
            "allow_nan": allow_nan,
            "lon_range": list(lon_range) if lon_range else None,
            "lat_range": list(lat_range) if lat_range else None
        }
    }

    try:
        # 1. 获取 NC 文件列表
        search_path = os.path.join(nc_folder, dyn_file_pattern)
        nc_files = sorted(glob.glob(search_path))

        if not nc_files:
            result["errors"].append(f"未找到匹配的 NC 文件: {search_path}")
            result["status"] = "error"
            return result

        print(f"找到 {len(nc_files)} 个 NC 文件", file=sys.stderr)
        result["nc_file_count"] = len(nc_files)

        # 2. 创建输出目录
        os.makedirs(output_base, exist_ok=True)

        # 3. 转换动态变量
        dyn_saved = convert_dynamic_vars(
            nc_files, dyn_vars, output_base, result,
            allow_nan=allow_nan
        )
        result["saved_files"].update(dyn_saved)

        # 4. 转换静态变量
        if stat_vars:
            sta_saved = convert_static_vars(
                static_file, stat_vars, mask_vars, output_base, result,
                allow_nan=allow_nan,
                lon_range=lon_range,
                lat_range=lat_range
            )
            result["saved_files"].update(sta_saved)

        # 5. 后置验证
        if run_validation:
            print("\n--- 执行后置验证 (Rule 1/2/3) ---", file=sys.stderr)

            # Rule 1
            rule1 = validate_rule1(
                output_base, dyn_vars, stat_vars, mask_vars,
                lon_var, lat_var, result["saved_files"], result
            )
            result["validation_rule1"] = rule1

            # Rule 2
            rule2 = validate_rule2(
                output_base, static_file, mask_vars,
                mask_src_var, mask_derive_op, result["saved_files"], result,
                heuristic_check_var=heuristic_check_var,
                land_threshold=land_threshold,
                heuristic_sample_size=heuristic_sample_size
            )
            result["validation_rule2"] = rule2

            # Rule 3
            rule3 = validate_rule3(
                output_base, nc_folder, nc_files,
                dyn_vars, stat_vars, dyn_file_pattern, static_file, result,
                require_sorted=require_sorted
            )
            result["validation_rule3"] = rule3

            # 汇总验证结果
            all_passed = rule1["passed"] and rule2["passed"] and rule3["passed"]
            result["post_validation"] = {
                "all_passed": all_passed,
                "rule1_passed": rule1["passed"],
                "rule2_passed": rule2["passed"],
                "rule3_passed": rule3["passed"],
                "total_errors": len(rule1["errors"]) + len(rule2["errors"]) + len(rule3["errors"]),
                "total_warnings": len(rule1["warnings"]) + len(rule2["warnings"]) + len(rule3["warnings"])
            }

        # 6. 设置最终状态
        if result["errors"]:
            result["status"] = "error"
            result["message"] = f"转换失败，存在 {len(result['errors'])} 个错误"
        else:
            result["status"] = "pass"
            result["message"] = f"转换完成，已保存 {len(result['saved_files'])} 个文件到 {output_base}"

        print(f"\n{result['message']}", file=sys.stderr)

    except Exception as e:
        result["status"] = "error"
        result["errors"].append(str(e))
        import traceback
        result["traceback"] = traceback.format_exc()

    return result


def main():
    parser = argparse.ArgumentParser(description="Step C: NC 转 NPY 转换")
    parser.add_argument("--config", required=True, help="JSON 配置文件路径")
    parser.add_argument("--output", required=True, help="结果输出 JSON 路径")
    args = parser.parse_args()

    # 读取配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 执行转换
    result = convert_npy(config)

    # 写入结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 同时输出到 stdout
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
