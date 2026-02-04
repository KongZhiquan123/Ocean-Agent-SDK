#!/usr/bin/env python3
"""
convert_npy.py - Step C: NC 转 NPY 转换（含后置验证）

@author leizheng
@date 2026-02-02
@version 2.9.0

功能:
- 将 NC 文件中的变量转换为 NPY 格式
- 按时间顺序划分数据集（train/valid/test）
- v2.9.0: 每个时间步单独保存为一个文件（000000.npy, 000001.npy, ...）
- 输出目录结构: train/hr/uo/, train/hr/vo/, valid/hr/uo/, ..., static_variables/
- 支持 output_subdir 参数，可输出到 lr/ 子目录（用于粗网格数据）
- 支持从动态文件中提取静态变量（当没有单独的静态文件时）
- 静态变量添加编号前缀（00_lon_rho, 01_lat_rho, 99_mask_rho）
- 自动生成 preprocess_manifest.json
- 执行后置验证 (Rule 1/2/3)
- NaN/Inf 检测（采样校验）
- 多网格支持（C-grid staggered mesh）

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
    "train_ratio": 0.7,
    "valid_ratio": 0.15,
    "test_ratio": 0.15,
    "h_slice": "0:680",
    "w_slice": "0:1440",
    "scale": 4,
    "workers": 32,
    "output_subdir": "hr",  // 或 "lr" 用于粗网格数据
    ...
}

Changelog:
    - 2026-02-04 leizheng v2.9.0: 逐时间步保存文件
        - 每个时间步单独保存为一个文件（000000.npy, 000001.npy, ...）
        - 目录结构改为: train/hr/uo/, train/hr/vo/ 等
        - 每个样本形状为 [H, W] 或 [D, H, W]，而非合并的 [T, H, W]
    - 2026-02-04 leizheng v2.7.0: 修复 1D 坐标裁剪问题
        - latitude 正确使用 h_slice 裁剪
        - longitude 正确使用 w_slice 裁剪
        - 修复可视化中坐标轴显示错误的问题
    - 2026-02-04 leizheng v2.6.0: 文件级并行处理
        - 使用 multiprocessing.Pool 替代 xr.open_mfdataset
        - 每个进程独立打开文件，避免 HDF5/netCDF4 并发问题
        - 彻底解决段错误问题
        - 支持 1D 坐标数组（latitude/longitude）
    - 2026-02-04 leizheng v2.5.1: 修复 workers 参数未生效问题
        - workers 参数现在正确配置 dask 线程数
        - 使用 dask 线程调度器（避免多进程段错误）
        - 打印并行线程数便于调试
    - 2026-02-04 leizheng v2.5.0: 支持从动态文件提取静态变量
        - 新增 fallback_nc_files 参数
        - 当 static_file 不存在时，自动从动态文件中提取静态变量
        - 适用于静态变量与动态变量在同一文件中的情况
    - 2026-02-04 leizheng v2.4.0: 支持粗网格模式
        - 新增 output_subdir 参数（默认 'hr'，可设为 'lr'）
        - 用于支持粗网格数据直接输出到 lr/ 目录
    - 2026-02-03 leizheng v2.3.0: 裁剪与多线程
        - 新增 h_slice/w_slice 参数，在转换时直接裁剪
        - 新增 scale 参数，验证裁剪后尺寸能否被整除
        - 新增 workers 参数，多线程并行处理（默认 32）
    - 2026-02-03 leizheng v2.2.0: 数据集划分功能
        - 新增 train_ratio/valid_ratio/test_ratio 参数
        - 按时间顺序划分数据到 train/valid/test 目录
        - 输出目录结构改为 train/hr/, valid/hr/, test/hr/
    - 2026-02-03 leizheng v2.1.0: 增加 P0 特性
    - 2026-02-02 leizheng v2.0.0: 初始版本
"""

import argparse
import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

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

# 多线程默认参数
DEFAULT_WORKERS = 32


# ========================================
# 文件级并行处理函数（multiprocessing worker）
# ========================================

def _extract_var_from_file(nc_file: str, var_name: str, h_slice: Optional[slice] = None, w_slice: Optional[slice] = None) -> Dict[str, Any]:
    """
    从单个 NC 文件中提取变量数据（worker 函数）

    每个进程独立打开文件，避免 HDF5/netCDF4 并发问题

    Args:
        nc_file: NC 文件路径
        var_name: 变量名
        h_slice: H 方向裁剪切片
        w_slice: W 方向裁剪切片

    Returns:
        包含数据和元信息的字典
    """
    try:
        with xr.open_dataset(nc_file, decode_times=False) as ds:
            if var_name not in ds.data_vars and var_name not in ds.coords:
                return {"status": "error", "error": f"变量 '{var_name}' 不存在", "file": nc_file}

            data = ds[var_name].values

            # 裁剪空间维度
            if h_slice is not None or w_slice is not None:
                h_sl = h_slice if h_slice else slice(None)
                w_sl = w_slice if w_slice else slice(None)
                ndim = data.ndim
                if ndim == 2:
                    data = data[h_sl, w_sl]
                elif ndim == 3:
                    data = data[:, h_sl, w_sl]
                elif ndim == 4:
                    data = data[:, :, h_sl, w_sl]

            return {
                "status": "success",
                "file": nc_file,
                "data": data,
                "shape": data.shape,
                "dtype": str(data.dtype)
            }
    except Exception as e:
        return {"status": "error", "error": str(e), "file": nc_file}


def _parallel_extract_var(nc_files: List[str], var_name: str, workers: int,
                          h_slice: Optional[slice] = None, w_slice: Optional[slice] = None) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    并行从多个 NC 文件中提取变量并合并

    Args:
        nc_files: NC 文件列表（已排序）
        var_name: 变量名
        workers: 并行进程数
        h_slice: H 方向裁剪切片
        w_slice: W 方向裁剪切片

    Returns:
        (合并后的数组, 错误列表)
    """
    errors = []

    # 限制 workers 数量
    actual_workers = min(workers, len(nc_files), cpu_count())

    # 创建 partial 函数固定参数
    extract_func = partial(_extract_var_from_file, var_name=var_name, h_slice=h_slice, w_slice=w_slice)

    print(f"    并行提取 '{var_name}'，进程数: {actual_workers}，文件数: {len(nc_files)}", file=sys.stderr)

    if actual_workers > 1:
        # 多进程并行
        with Pool(processes=actual_workers) as pool:
            results = pool.map(extract_func, nc_files)
    else:
        # 单进程顺序执行
        results = [extract_func(f) for f in nc_files]

    # 收集结果
    data_list = []
    for r in results:
        if r["status"] == "success":
            data_list.append(r["data"])
        else:
            errors.append(f"{r['file']}: {r.get('error', 'unknown error')}")

    if not data_list:
        return None, errors

    # 沿时间轴合并（第 0 维）
    try:
        combined = np.concatenate(data_list, axis=0)
        return combined, errors
    except Exception as e:
        errors.append(f"合并数据失败: {str(e)}")
        return None, errors


# ========================================
# 裁剪相关函数
# ========================================

def parse_slice_str(slice_str: Optional[str]) -> Optional[slice]:
    """
    解析切片字符串为 slice 对象

    支持格式:
    - "0:680"   -> slice(0, 680)
    - ":680"    -> slice(None, 680)
    - "1:"      -> slice(1, None)
    - "1:-1"    -> slice(1, -1)
    - None      -> None (不裁剪)

    Args:
        slice_str: 切片字符串

    Returns:
        slice 对象或 None
    """
    if not slice_str:
        return None

    parts = slice_str.split(':')
    if len(parts) != 2:
        raise ValueError(f"无效的切片格式: '{slice_str}'，应为 'start:end'")

    start_str, end_str = parts

    start = int(start_str) if start_str.strip() else None
    end = int(end_str) if end_str.strip() else None

    return slice(start, end)


def crop_spatial(arr: np.ndarray, h_slice: Optional[slice], w_slice: Optional[slice]) -> np.ndarray:
    """
    裁剪数组的空间维度（最后两维）

    Args:
        arr: 输入数组，支持 1D/2D/3D/4D
        h_slice: H 方向切片
        w_slice: W 方向切片

    Returns:
        裁剪后的数组
    """
    if h_slice is None and w_slice is None:
        return arr

    h_sl = h_slice if h_slice else slice(None)
    w_sl = w_slice if w_slice else slice(None)

    ndim = arr.ndim

    if ndim == 1:
        # 1D 数组（如 latitude 或 longitude）
        # latitude 用 h_slice，longitude 用 w_slice
        # 如果两个都提供，优先使用非空的那个
        if h_slice is not None:
            return arr[h_sl]
        elif w_slice is not None:
            return arr[w_sl]
        else:
            return arr
    elif ndim == 2:
        return arr[h_sl, w_sl]
    elif ndim == 3:
        return arr[:, h_sl, w_sl]
    elif ndim == 4:
        return arr[:, :, h_sl, w_sl]
    else:
        raise ValueError(f"不支持的维度: {ndim}D")


def validate_crop_divisible(h: int, w: int, scale: int) -> Tuple[bool, str]:
    """
    验证裁剪后的尺寸能否被 scale 整除

    Args:
        h: 高度
        w: 宽度
        scale: 下采样倍数

    Returns:
        (is_valid, message)
    """
    h_remainder = h % scale
    w_remainder = w % scale

    if h_remainder == 0 and w_remainder == 0:
        return True, f"尺寸验证通过: H={h} % {scale} = 0, W={w} % {scale} = 0"

    # 计算建议值
    suggested_h = (h // scale) * scale
    suggested_w = (w // scale) * scale

    msg = f"尺寸无法被 scale={scale} 整除:\n"
    msg += f"  当前: H={h} (余 {h_remainder}), W={w} (余 {w_remainder})\n"
    msg += f"  建议: H={suggested_h}, W={suggested_w}"

    return False, msg


def get_cropped_shape(original_shape: Tuple, h_slice: Optional[slice], w_slice: Optional[slice]) -> Tuple[int, int]:
    """
    计算裁剪后的空间尺寸

    Args:
        original_shape: 原始 shape
        h_slice: H 方向切片
        w_slice: W 方向切片

    Returns:
        (cropped_h, cropped_w)
    """
    # 获取原始空间尺寸（最后两维）
    if len(original_shape) >= 2:
        orig_h, orig_w = original_shape[-2], original_shape[-1]
    else:
        raise ValueError(f"数组维度不足: {original_shape}")

    # 计算裁剪后的尺寸
    def calc_slice_len(orig_len: int, sl: Optional[slice]) -> int:
        if sl is None:
            return orig_len
        start = sl.start if sl.start is not None else 0
        stop = sl.stop if sl.stop is not None else orig_len
        # 处理负数索引
        if start < 0:
            start = orig_len + start
        if stop < 0:
            stop = orig_len + stop
        return max(0, stop - start)

    cropped_h = calc_slice_len(orig_h, h_slice)
    cropped_w = calc_slice_len(orig_w, w_slice)

    return cropped_h, cropped_w


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


def get_spatial_shape(arr: np.ndarray) -> Tuple[int, ...]:
    """
    获取数组的空间维度

    对于不同维度的数组：
    - 1D (N,): 返回 (N,)
    - 2D (H, W): 返回 (H, W)
    - 3D (T, H, W): 返回 (H, W)
    - 4D (T, D, H, W): 返回 (H, W)

    Args:
        arr: 输入数组

    Returns:
        空间维度元组
    """
    if arr.ndim == 1:
        return (arr.shape[0],)
    elif arr.ndim == 2:
        return (arr.shape[0], arr.shape[1])
    elif arr.ndim >= 3:
        return (arr.shape[-2], arr.shape[-1])
    else:
        return ()


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
    allow_nan: bool = False,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    h_slice: Optional[slice] = None,
    w_slice: Optional[slice] = None,
    scale: Optional[int] = None,
    workers: int = DEFAULT_WORKERS,
    output_subdir: str = 'hr'
) -> Dict[str, Dict[str, Any]]:
    """
    转换动态变量，并按时间顺序划分为 train/valid/test

    Args:
        nc_files: NC 文件列表
        dyn_vars: 动态变量列表
        output_dir: 输出目录
        result: 结果字典（用于添加警告/错误）
        allow_nan: 是否允许 NaN/Inf 值
        train_ratio: 训练集比例（默认 0.7）
        valid_ratio: 验证集比例（默认 0.15）
        test_ratio: 测试集比例（默认 0.15）
        h_slice: H 方向裁剪切片
        w_slice: W 方向裁剪切片
        scale: 下采样倍数（用于验证裁剪后尺寸）
        workers: 并行线程数（默认 32）
        output_subdir: 输出子目录名（默认 'hr'，可设为 'lr' 用于粗网格数据）

    Returns:
        保存的文件信息字典
    """
    # 创建输出目录结构: train/{output_subdir}, train/lr (预留), valid/{output_subdir}, etc.
    splits = ['train', 'valid', 'test']
    for split in splits:
        subdir = os.path.join(output_dir, split, output_subdir)
        lr_dir = os.path.join(output_dir, split, 'lr')
        os.makedirs(subdir, exist_ok=True)
        os.makedirs(lr_dir, exist_ok=True)  # 预留 lr 目录

    saved_files = {}
    time_lengths = {}
    nan_check_results = {}
    split_info = {}  # 记录划分信息

    print(f"处理动态变量，文件数: {len(nc_files)}", file=sys.stderr)
    print(f"划分比例: train={train_ratio}, valid={valid_ratio}, test={test_ratio}", file=sys.stderr)
    print(f"并行进程数: {workers}", file=sys.stderr)

    # ========== 使用文件级并行处理（避免 HDF5/netCDF4 并发问题）==========
    for var in dyn_vars:
        try:
            # 并行提取变量数据
            data_arr, extract_errors = _parallel_extract_var(
                nc_files, var, workers, h_slice, w_slice
            )

            if extract_errors:
                for err in extract_errors:
                    result["warnings"].append(f"提取 '{var}' 警告: {err}")

            if data_arr is None:
                result["warnings"].append(f"动态变量 '{var}' 提取失败")
                continue

            print(f"  变量 '{var}' 提取完成，形状: {data_arr.shape}", file=sys.stderr)

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

            # ========== 验证裁剪后尺寸能否被 scale 整除 ==========
            if scale is not None:
                cropped_h, cropped_w = data_arr.shape[-2], data_arr.shape[-1]
                is_valid, msg = validate_crop_divisible(cropped_h, cropped_w, scale)
                if not is_valid:
                    result["errors"].append(f"变量 '{var}' {msg}")
                    print(f"    错误: {msg}", file=sys.stderr)
                    continue
                else:
                    print(f"    {msg}", file=sys.stderr)

            # ========== 按时间顺序划分数据集 ==========
            total_time = data_arr.shape[0]

            # 边界情况：时间步太少，无法划分
            if total_time < 3:
                # 所有数据放入 train
                print(f"    时间步数({total_time})太少，全部放入 train", file=sys.stderr)
                train_data = data_arr
                valid_data = np.array([]).reshape(0, *data_arr.shape[1:])
                test_data = np.array([]).reshape(0, *data_arr.shape[1:])
                train_end = total_time
                valid_end = total_time

                split_info[var] = {
                    "total_time": total_time,
                    "train": {"start": 0, "end": total_time, "count": total_time},
                    "valid": {"start": total_time, "end": total_time, "count": 0},
                    "test": {"start": total_time, "end": total_time, "count": 0},
                    "note": "时间步太少，全部放入 train"
                }
            else:
                train_end = int(total_time * train_ratio)
                valid_end = int(total_time * (train_ratio + valid_ratio))

                # 确保至少有 1 个时间步在 train
                train_end = max(1, train_end)
                # 确保 valid 和 test 不重叠
                valid_end = max(train_end, min(valid_end, total_time))

                # 如果 valid_ratio > 0 但没有分到数据，调整
                if valid_ratio > 0 and valid_end == train_end and train_end < total_time:
                    valid_end = min(train_end + 1, total_time)

                # 划分数据
                train_data = data_arr[0:train_end]
                valid_data = data_arr[train_end:valid_end] if valid_end > train_end else np.array([]).reshape(0, *data_arr.shape[1:])
                test_data = data_arr[valid_end:total_time] if total_time > valid_end else np.array([]).reshape(0, *data_arr.shape[1:])

                # 记录划分信息
                split_info[var] = {
                    "total_time": total_time,
                    "train": {"start": 0, "end": train_end, "count": train_end},
                    "valid": {"start": train_end, "end": valid_end, "count": valid_end - train_end},
                    "test": {"start": valid_end, "end": total_time, "count": total_time - valid_end}
                }

            print(f"    划分: train={split_info[var]['train']['count']}, valid={split_info[var]['valid']['count']}, test={split_info[var]['test']['count']}", file=sys.stderr)

            # 保存到对应目录 - v2.9.0: 每个时间步单独保存为一个文件
            spatial_shape = get_spatial_shape(data_arr)
            var_saved_files = {}

            for split_name, split_data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
                if split_data.size == 0:
                    continue

                # 创建变量子目录: train/hr/uo/, train/hr/vo/ 等
                var_dir = os.path.join(output_dir, split_name, output_subdir, var)
                os.makedirs(var_dir, exist_ok=True)

                # 逐时间步保存
                time_steps = split_data.shape[0]
                saved_count = 0
                for t in range(time_steps):
                    # 使用 6 位数字编号: 000000.npy, 000001.npy, ...
                    filename = f"{t:06d}.npy"
                    out_fp = os.path.join(var_dir, filename)
                    np.save(out_fp, split_data[t])  # 保存单个时间步 [H, W] 或 [D, H, W]
                    saved_count += 1

                var_saved_files[split_name] = {
                    "dir": var_dir,
                    "file_count": saved_count,
                    "sample_shape": list(split_data[0].shape),  # 单个样本的形状
                    "total_shape": list(split_data.shape),
                    "spatial_shape": list(spatial_shape),
                    "dtype": str(split_data.dtype),
                    "time_steps": time_steps
                }

                print(f"      {split_name}/{output_subdir}/{var}/: {saved_count} 个文件, 每个 shape={split_data[0].shape}", file=sys.stderr)

            # 记录时间长度
            time_lengths[var] = total_time

            # 维度解释
            ndim = data_arr.ndim
            if ndim == 3:
                interp = f"[T={data_arr.shape[0]}, H={data_arr.shape[1]}, W={data_arr.shape[2]}]"
            elif ndim == 4:
                interp = f"[T={data_arr.shape[0]}, D={data_arr.shape[1]}, H={data_arr.shape[2]}, W={data_arr.shape[3]}]"
            else:
                interp = f"shape={data_arr.shape}"

            saved_files[var] = {
                "splits": var_saved_files,
                "total_shape": list(data_arr.shape),
                "spatial_shape": list(spatial_shape),
                "dtype": str(data_arr.dtype),
                "interpretation": interp,
                "is_dynamic": True,
                "nan_check": nan_result,
                "split_info": split_info[var]
            }

            print(f"    完成，total_shape={data_arr.shape}, spatial={spatial_shape}", file=sys.stderr)
            del data_arr, train_data, valid_data, test_data

        except Exception as e:
            import traceback
            result["warnings"].append(f"处理动态变量 '{var}' 失败: {str(e)}")
            print(f"    异常: {traceback.format_exc()}", file=sys.stderr)

    result["time_lengths"] = time_lengths
    result["nan_check_results"] = nan_check_results
    result["split_info"] = split_info
    return saved_files


def convert_static_vars(
    static_file: str,
    stat_vars: List[str],
    mask_vars: List[str],
    output_dir: str,
    result: Dict[str, Any],
    allow_nan: bool = False,
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None,
    h_slice: Optional[slice] = None,
    w_slice: Optional[slice] = None,
    fallback_nc_files: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    转换静态变量（带编号前缀）

    Args:
        static_file: 静态文件路径（如果为空，将尝试从 fallback_nc_files 中提取）
        stat_vars: 静态变量列表
        mask_vars: 掩码变量列表
        output_dir: 输出目录
        result: 结果字典（用于添加警告/错误）
        allow_nan: 是否允许 NaN/Inf 值
        lon_range: 经度有效范围 (min, max)
        lat_range: 纬度有效范围 (min, max)
        h_slice: H 方向裁剪切片
        w_slice: W 方向裁剪切片
        fallback_nc_files: 备用 NC 文件列表（当 static_file 为空时，尝试从这些文件中提取静态变量）

    Returns:
        保存的文件信息字典
    """
    sta_out_dir = os.path.join(output_dir, 'static_variables')
    os.makedirs(sta_out_dir, exist_ok=True)

    saved_files = {}
    other_idx = 0  # 用于其他静态变量的编号
    coord_checks = {}  # 坐标范围检查结果

    # 确定要读取的文件
    source_file = None
    source_type = None

    if static_file and os.path.exists(static_file):
        source_file = static_file
        source_type = "static_file"
    elif fallback_nc_files and len(fallback_nc_files) > 0:
        # 没有静态文件，尝试从动态文件中提取静态变量
        source_file = fallback_nc_files[0]
        source_type = "dynamic_file"
        print(f"静态文件未提供，尝试从动态文件中提取静态变量: {source_file}", file=sys.stderr)
    else:
        if static_file:
            result["warnings"].append(f"静态文件不存在: {static_file}")
        else:
            result["warnings"].append("未提供静态文件，且无动态文件可用于提取静态变量")
        return saved_files

    print(f"处理静态变量 (来源: {source_type}): {source_file}", file=sys.stderr)

    try:
        with xr.open_dataset(source_file, decode_times=False) as ds:
            for var in stat_vars:
                if var not in ds.variables:
                    result["warnings"].append(f"静态变量 '{var}' 不存在于{'静态文件' if source_type == 'static_file' else '动态文件'}中")
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

                    # ========== 空间裁剪 ==========
                    original_shape = data_arr.shape
                    if h_slice is not None or w_slice is not None:
                        # 对于 1D 坐标数组，需要根据变量类型选择正确的切片
                        # latitude 对应 H 维度，longitude 对应 W 维度
                        if data_arr.ndim == 1:
                            if is_lat and h_slice is not None:
                                # latitude 用 h_slice
                                data_arr = data_arr[h_slice]
                                print(f"    裁剪 latitude: {original_shape} -> {data_arr.shape} (使用 h_slice)", file=sys.stderr)
                            elif is_lon and w_slice is not None:
                                # longitude 用 w_slice
                                data_arr = data_arr[w_slice]
                                print(f"    裁剪 longitude: {original_shape} -> {data_arr.shape} (使用 w_slice)", file=sys.stderr)
                            else:
                                # 其他 1D 变量，不裁剪
                                print(f"    1D 变量 '{var}'，不裁剪", file=sys.stderr)
                        else:
                            # 2D/3D/4D 变量，正常裁剪
                            data_arr = crop_spatial(data_arr, h_slice, w_slice)
                            print(f"    裁剪: {original_shape} -> {data_arr.shape}", file=sys.stderr)

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
        # 防御性检查：确保 info 中有 shape 或 spatial_shape
        if "spatial_shape" in info:
            var_spatial = tuple(info["spatial_shape"])
        elif "shape" in info and len(info["shape"]) >= 2:
            var_spatial = tuple(info["shape"][-2:])
        else:
            # 跳过没有形状信息的变量
            continue

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

    # 数据集划分参数
    train_ratio = config.get("train_ratio", 0.7)
    valid_ratio = config.get("valid_ratio", 0.15)
    test_ratio = config.get("test_ratio", 0.15)

    # 裁剪参数
    h_slice_str = config.get("h_slice")  # 如 "0:680"
    w_slice_str = config.get("w_slice")  # 如 "0:1440"
    scale = config.get("scale")  # 下采样倍数，用于验证
    workers = config.get("workers", DEFAULT_WORKERS)

    # 输出子目录（默认 'hr'，用于粗网格数据时设为 'lr'）
    output_subdir = config.get("output_subdir", "hr")

    # 解析切片字符串
    try:
        h_slice = parse_slice_str(h_slice_str)
        w_slice = parse_slice_str(w_slice_str)
    except ValueError as e:
        return {
            "status": "error",
            "errors": [str(e)],
            "message": f"切片参数解析失败: {e}"
        }

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
            "lat_range": list(lat_range) if lat_range else None,
            "train_ratio": train_ratio,
            "valid_ratio": valid_ratio,
            "test_ratio": test_ratio,
            "h_slice": h_slice_str,
            "w_slice": w_slice_str,
            "scale": scale,
            "workers": workers,
            "output_subdir": output_subdir
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

        # 3. 转换动态变量（按时间顺序划分）
        dyn_saved = convert_dynamic_vars(
            nc_files, dyn_vars, output_base, result,
            allow_nan=allow_nan,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            h_slice=h_slice,
            w_slice=w_slice,
            scale=scale,
            workers=workers,
            output_subdir=output_subdir
        )
        result["saved_files"].update(dyn_saved)

        # 4. 转换静态变量
        if stat_vars:
            sta_saved = convert_static_vars(
                static_file, stat_vars, mask_vars, output_base, result,
                allow_nan=allow_nan,
                lon_range=lon_range,
                lat_range=lat_range,
                h_slice=h_slice,
                w_slice=w_slice,
                fallback_nc_files=nc_files  # 如果没有静态文件，从动态文件中提取
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
