"""
mask.py - 掩码相关函数

从 convert_npy.py 拆分
"""

import numpy as np
from typing import Any, Dict

from .constants import (
    NAN_CHECK_RANDOM_SEED,
    HEURISTIC_SAMPLE_SIZE,
    LAND_THRESHOLD_ABS,
    LAND_ZERO_RATIO_MIN,
    OCEAN_ZERO_RATIO_MAX
)


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
        return (mask_rho[:, :-1] * mask_rho[:, 1:]).astype(np.uint8)
    elif grid_type == "v":
        # v 在 eta 方向中点，相邻两个 rho 点都为海洋才算海洋
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
        dyn_slice = dyn_arr[0, :, :]
    elif dyn_arr.ndim == 4:
        dyn_slice = dyn_arr[0, 0, :, :]
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
