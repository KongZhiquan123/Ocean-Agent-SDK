"""
check.py - 数据检测函数

从 convert_npy.py 拆分
"""

import numpy as np
import xarray as xr
from typing import Any, Dict, List, Optional, Tuple

from .constants import (
    NAN_CHECK_SAMPLE_SIZE,
    NAN_CHECK_RANDOM_SEED,
    LON_VARS,
    LAT_VARS,
    TIME_COORD_CANDIDATES
)


def is_object_dtype(arr: np.ndarray) -> bool:
    """
    检查数组是否是 object dtype（禁止使用）
    """
    return arr.dtype == np.object_ or arr.dtype.kind == 'O'


def check_nan_inf_sampling(arr: np.ndarray, var_name: str, sample_size: int = NAN_CHECK_SAMPLE_SIZE) -> Dict[str, Any]:
    """
    采样检测 NaN/Inf 值
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
