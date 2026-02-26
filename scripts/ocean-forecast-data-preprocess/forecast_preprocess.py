#!/usr/bin/env python3
"""
@file forecast_preprocess.py

@description 海洋预报数据预处理（Step B）
             将 NC 文件按时间严格排序，转换为 NPY 格式
             输出目录结构: {split}/{var_name}/{date_str}.npy
             无下采样，无 hr/lr 层级

@author Leizheng
@contributers kongzhiquan
@date 2026-02-25
@version 1.0.0

@changelog
  - 2026-02-26 Leizheng: v1.1.0 新增网格文件支持
    - 静态变量提取新增三级搜索：数据文件 → 用户指定网格文件 → 自动检测网格文件
    - 新增 grid_file 配置参数
    - 自动检测 nc_folder 父目录中的独立 NC 文件作为网格文件
    - 同时检查 ds.variables（含 data_vars + coords）确保不遗漏坐标变量
  - 2026-02-25 Leizheng: v1.0.0 初始版本
    - NC 文件按内部时间变量严格排序（支持 decode_times=True/False 两种模式）
    - 支持多时间步/文件 和 单时间步/文件 两种 NC 文件组织形式
    - 按比例切分 train/valid/test（时间顺序，不打乱）
    - 每个时间步保存独立 NPY 文件（float32，C 连续）
    - 生成 time_index.json（完整时间戳溯源）
    - 生成 var_names.json（变量配置，供 DataLoader 使用）
    - 生成 preprocess_manifest.json（溯源清单）
    - 后置验证 Rule 1（完整性）/ Rule 2（时间单调性）/ Rule 3（掩码 NaN 一致性）

用法:
    python forecast_preprocess.py --config config.json --output result.json

配置文件格式:
{
    "nc_folder": "/path/to/nc",
    "output_base": "/path/to/output",
    "dyn_vars": ["uo", "vo"],
    "stat_vars": ["lon_rho", "lat_rho"],
    "mask_vars": ["mask_rho"],
    "lon_var": "lon_rho",
    "lat_var": "lat_rho",
    "train_ratio": 0.7,
    "valid_ratio": 0.15,
    "test_ratio": 0.15,
    "h_slice": null,
    "w_slice": null,
    "dyn_file_pattern": "*.nc",
    "chunk_size": 200,
    "use_date_filename": true,
    "date_format": "auto",
    "time_var": null,
    "max_files": null,
    "run_validation": true,
    "allow_nan": false
}
"""

import argparse
import glob
import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import xarray as xr
except ImportError:
    print(json.dumps({"status": "error", "errors": ["需要安装 xarray: pip install xarray netCDF4"]}))
    sys.exit(1)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ========================================
# 常量定义
# ========================================

TIME_VAR_NAMES = [
    'time', 'ocean_time', 'Time', 'TIME', 'time_counter',
    'MT', 't', 'nt', 'ntime', 'times', 'timec'
]

DATE_FORMAT_AUTO    = 'auto'
DATE_FORMAT_DAY     = 'YYYYMMDD'
DATE_FORMAT_HOUR    = 'YYYYMMDDHH'
DATE_FORMAT_MINUTE  = 'YYYYMMDDHHmm'

# 时间间隔容忍度（允许 ±10% 的抖动）
GAP_TOLERANCE = 0.10


# ========================================
# 辅助函数
# ========================================

def _safe_float(v) -> Optional[float]:
    """将 nan/inf 转为 None，安全用于 JSON"""
    if v is None:
        return None
    try:
        if math.isnan(v) or math.isinf(v):
            return None
    except (TypeError, ValueError):
        return None
    return float(v)


def _parse_slice(slice_str: Optional[str]) -> Optional[slice]:
    """解析切片字符串 'start:end' → slice(start, end)"""
    if not slice_str:
        return None
    parts = slice_str.strip().split(':')
    if len(parts) == 2:
        try:
            return slice(int(parts[0]), int(parts[1]))
        except ValueError:
            pass
    raise ValueError(f"无效的切片格式: '{slice_str}'，期望格式为 'start:end'，如 '0:680'")


def _find_time_var(ds: 'xr.Dataset', hint: Optional[str] = None) -> Optional[str]:
    """在数据集的 data_vars 和 coords 中查找时间变量"""
    all_vars = list(ds.variables.keys())
    # 优先使用用户提示
    if hint and hint in all_vars:
        return hint
    # 精确匹配
    for name in TIME_VAR_NAMES:
        if name in all_vars:
            return name
    # 模糊匹配
    for name in all_vars:
        if 'time' in name.lower():
            return name
    return None


def _format_date_filename(ts_str: str, date_format: str, step_seconds: Optional[float]) -> str:
    """
    将时间戳字符串（YYYYMMDDHHmmSS 格式）格式化为 NPY 文件名
    ts_str: 14位字符串，如 '20200101120000'
    """
    if date_format == DATE_FORMAT_AUTO or date_format is None:
        if step_seconds is None or step_seconds >= 86400 * 0.9:
            return ts_str[:8]    # YYYYMMDD（日级）
        elif step_seconds >= 3600 * 0.9:
            return ts_str[:10]   # YYYYMMDDHH（小时级）
        else:
            return ts_str[:12]   # YYYYMMDDHHmm（分钟级）
    elif date_format == DATE_FORMAT_DAY:
        return ts_str[:8]
    elif date_format == DATE_FORMAT_HOUR:
        return ts_str[:10]
    elif date_format == DATE_FORMAT_MINUTE:
        return ts_str[:12]
    else:
        return ts_str[:8]


class TimeStepInfo:
    """代表单个时间步的元信息（不加载数据）"""
    __slots__ = ['ts_str', 'ts_float', 'file_path', 'local_idx']

    def __init__(self, ts_str: str, ts_float: float, file_path: str, local_idx: int):
        self.ts_str = ts_str
        self.ts_float = ts_float
        self.file_path = file_path
        self.local_idx = local_idx


# ========================================
# 时间戳提取
# ========================================

def _extract_timestamps_from_file(
    file_path: str,
    time_var_hint: Optional[str] = None
) -> List[Tuple[str, float, int]]:
    """
    从单个 NC 文件提取所有时间步的时间戳

    Returns:
        List of (ts_str_14, ts_float, local_idx)
        ts_str_14: 14位字符串 'YYYYMMDDHHmmSS'
        ts_float: Unix timestamp (float)
        local_idx: 该时间步在文件中的索引
    """
    # 策略1：decode_times=True（xarray 自动解析，最通用）
    try:
        with xr.open_dataset(file_path, decode_times=True) as ds:
            tv = _find_time_var(ds, time_var_hint)
            if tv is None:
                return []

            time_data = ds[tv].values  # numpy.datetime64 或 cftime

            results = []
            for i, t in enumerate(np.atleast_1d(time_data)):
                try:
                    if HAS_PANDAS:
                        ts = pd.Timestamp(t)
                        ts_float = ts.timestamp()
                        ts_str = ts.strftime('%Y%m%d%H%M%S')
                    else:
                        ns = int(np.datetime64(t, 'ns').astype('int64'))
                        ts_float = ns / 1e9
                        dt = datetime.fromtimestamp(ts_float, tz=timezone.utc)
                        ts_str = dt.strftime('%Y%m%d%H%M%S')
                    results.append((ts_str, ts_float, i))
                except Exception:
                    # 单个时间步转换失败，用序号兜底
                    ts_float = os.path.getmtime(file_path) + i
                    ts_str = datetime.utcfromtimestamp(ts_float).strftime('%Y%m%d%H%M%S')
                    results.append((ts_str, ts_float, i))

            return results
    except Exception:
        pass  # 进入策略2

    # 策略2：decode_times=False + 手动转换（用于非标准时间单位）
    try:
        with xr.open_dataset(file_path, decode_times=False) as ds:
            tv = _find_time_var(ds, time_var_hint)
            if tv is None:
                return []

            time_var = ds[tv]
            time_values = np.atleast_1d(time_var.values)
            units = time_var.attrs.get('units', '')
            calendar = time_var.attrs.get('calendar', 'standard')

            try:
                import cftime
                dates = cftime.num2date(time_values, units=units, calendar=calendar)
                results = []
                for i, d in enumerate(dates):
                    try:
                        if HAS_PANDAS:
                            ts = pd.Timestamp(d.isoformat())
                            ts_float = ts.timestamp()
                            ts_str = ts.strftime('%Y%m%d%H%M%S')
                        else:
                            ts_str = d.strftime('%Y%m%d%H%M%S')
                            dt = datetime.strptime(ts_str, '%Y%m%d%H%M%S')
                            ts_float = dt.replace(tzinfo=timezone.utc).timestamp()
                        results.append((ts_str, ts_float, i))
                    except Exception:
                        ts_float = float(time_values[i]) * 86400.0  # 假设单位为天
                        ts_str = datetime.utcfromtimestamp(ts_float).strftime('%Y%m%d%H%M%S')
                        results.append((ts_str, ts_float, i))
                return results
            except ImportError:
                pass

            # 策略3：直接用 float 数值排序（无法解析单位时兜底）
            results = []
            for i, v in enumerate(time_values.flatten()):
                ts_float = float(v)
                ts_str = f"{int(abs(ts_float)):014d}"
                results.append((ts_str, ts_float, i))
            return results

    except Exception as e:
        print(f"  ⚠️ 无法提取 {os.path.basename(file_path)} 的时间戳: {e}", file=sys.stderr)
        return []


# ========================================
# 时间连续性验证
# ========================================

def _check_time_continuity(
    time_steps: List[TimeStepInfo],
    tolerance: float = GAP_TOLERANCE
) -> Tuple[List[Dict], Optional[float]]:
    """
    检查时间步是否连续等间隔
    Returns: (gap_list, estimated_step_seconds)
    gap_list: [{"index": i, "prev_ts": ..., "next_ts": ..., "gap_seconds": ..., "expected_seconds": ...}]
    """
    if len(time_steps) < 2:
        return [], None

    # 计算相邻时间间隔
    intervals = []
    for i in range(1, len(time_steps)):
        dt = time_steps[i].ts_float - time_steps[i - 1].ts_float
        intervals.append(dt)

    # 估计标准时间步（中位数，更鲁棒）
    sorted_intervals = sorted(intervals)
    median_step = sorted_intervals[len(sorted_intervals) // 2]

    if median_step <= 0:
        return [{"error": "时间步为负值或零，数据时间序列存在问题"}], None

    # 检测异常间隔
    gaps = []
    for i, dt in enumerate(intervals):
        ratio = dt / median_step
        if abs(ratio - round(ratio)) > tolerance or round(ratio) > 1:
            gaps.append({
                "index": i + 1,
                "prev_ts": time_steps[i].ts_str,
                "next_ts": time_steps[i + 1].ts_str,
                "gap_seconds": _safe_float(dt),
                "expected_seconds": _safe_float(median_step),
                "ratio": _safe_float(ratio),
                "type": "gap" if ratio > 1 + tolerance else "overlap"
            })

    return gaps, median_step


# ========================================
# 数据写入
# ========================================

def _apply_spatial_slice(
    data: np.ndarray,
    h_slice: Optional[slice],
    w_slice: Optional[slice]
) -> np.ndarray:
    """
    对空间维度应用裁剪切片
    支持 1D、2D (H, W) 和 3D (D, H, W) 数组

    1D 数组不做裁剪（需要调用方根据维度语义自行处理）
    """
    if data.ndim <= 1:
        # 1D 坐标变量或标量，无法确定 H/W 维度，原样返回
        return data
    elif data.ndim == 2:
        h = h_slice if h_slice else slice(None)
        w = w_slice if w_slice else slice(None)
        return data[h, w]
    elif data.ndim == 3:
        h = h_slice if h_slice else slice(None)
        w = w_slice if w_slice else slice(None)
        return data[:, h, w]
    else:
        # 其他维度：只尝试最后两维
        if h_slice or w_slice:
            idx = [slice(None)] * data.ndim
            if h_slice:
                idx[-2] = h_slice
            if w_slice:
                idx[-1] = w_slice
            return data[tuple(idx)]
        return data


def _write_timestep(
    output_dir: str,
    filename: str,
    data: np.ndarray
) -> str:
    """保存单个时间步的 NPY 文件，返回完整路径"""
    os.makedirs(output_dir, exist_ok=True)
    fpath = os.path.join(output_dir, f"{filename}.npy")
    np.save(fpath, data.astype(np.float32, copy=False))
    return fpath


# ========================================
# 静态变量提取
# ========================================

def _auto_detect_grid_file(nc_folder: str, dyn_file_pattern: str) -> Optional[str]:
    """
    自动检测网格文件。

    搜索策略：
    1. nc_folder 的父目录中寻找独立的 .nc 文件（不在 dyn/ 子目录中）
    2. nc_folder 本身中寻找不匹配 dyn_file_pattern 的 .nc 文件

    ROMS 等模式通常将网格变量（lon_rho, mask_rho, h 等）存储在
    独立的网格文件中，而非历史输出文件中。
    """
    import fnmatch

    candidates = []

    # 策略1：父目录中的 NC 文件
    parent_dir = os.path.dirname(nc_folder.rstrip('/'))
    if parent_dir and os.path.isdir(parent_dir):
        for f in os.listdir(parent_dir):
            if f.lower().endswith('.nc') and os.path.isfile(os.path.join(parent_dir, f)):
                candidates.append(os.path.join(parent_dir, f))

    # 策略2：nc_folder 内不匹配 dyn_file_pattern 的 NC 文件
    if os.path.isdir(nc_folder):
        for f in os.listdir(nc_folder):
            if f.lower().endswith('.nc') and not fnmatch.fnmatch(f, dyn_file_pattern):
                full_path = os.path.join(nc_folder, f)
                if os.path.isfile(full_path) and full_path not in candidates:
                    candidates.append(full_path)

    if not candidates:
        return None

    # 优先选择名称中包含 grid/grd/mesh/coord 关键词的文件
    for keyword in ['grid', 'grd', 'mesh', 'coord']:
        for c in candidates:
            if keyword in os.path.basename(c).lower():
                return c

    # 否则返回第一个候选
    return candidates[0] if candidates else None


def _extract_var_from_dataset(
    ds: 'xr.Dataset',
    var_name: str,
    h_slice: Optional[slice],
    w_slice: Optional[slice],
    h_dim_name: Optional[str],
    w_dim_name: Optional[str]
) -> Optional[np.ndarray]:
    """
    从 xarray Dataset 中提取单个变量（同时检查 data_vars 和 coords）。

    Returns: numpy array 或 None（未找到）
    """
    if var_name not in ds.variables:
        return None

    da = ds[var_name]
    var_data = da.values
    var_dims = da.dims

    # 如果变量有时间维度，取第一个时间步
    if var_data.ndim > 2:
        var_data = var_data[0] if var_data.shape[0] < 10 else var_data.squeeze()
        if var_data.ndim > 2:
            var_data = var_data[0]

    # 对 1D 变量，根据维度名判断裁剪轴
    if var_data.ndim == 1 and len(var_dims) == 1:
        dim_name = var_dims[0]
        if dim_name == h_dim_name and h_slice is not None:
            var_data = var_data[h_slice]
        elif dim_name == w_dim_name and w_slice is not None:
            var_data = var_data[w_slice]
    else:
        var_data = _apply_spatial_slice(var_data, h_slice, w_slice)

    return var_data


def _extract_and_save_static_vars(
    nc_files: List[str],
    stat_vars: List[str],
    mask_vars: List[str],
    output_base: str,
    h_slice: Optional[slice],
    w_slice: Optional[slice],
    dyn_vars: List[str],
    warnings_list: List[str],
    grid_file: Optional[str] = None,
    nc_folder: Optional[str] = None,
    dyn_file_pattern: str = '*.nc'
) -> Dict[str, str]:
    """
    从 NC 文件中提取静态变量并保存。

    搜索优先级：
    1. 数据 NC 文件（nc_files）—— 很多模式的静态变量就在动态文件里
    2. 网格文件（grid_file）—— ROMS 等模式的网格变量在独立文件中
    3. 自动检测网格文件 —— 如果前两步都没找全，尝试从父目录发现网格文件

    对 2D 变量直接使用 _apply_spatial_slice(h_slice, w_slice)
    对 1D 坐标变量，根据其维度名与动态变量的 H/W 维度匹配决定裁剪轴

    Returns: {var_name: output_path}
    """
    static_dir = os.path.join(output_base, 'static_variables')
    os.makedirs(static_dir, exist_ok=True)

    all_static = list(stat_vars) + list(mask_vars)
    saved = {}

    if not all_static:
        return saved

    # 从动态变量推断 H/W 维度名（从第一个数据文件）
    h_dim_name, w_dim_name = None, None
    for nc_file in nc_files[:3]:
        try:
            with xr.open_dataset(nc_file, decode_times=False) as ds:
                for dv in dyn_vars:
                    if dv in ds:
                        dv_dims = ds[dv].dims
                        if len(dv_dims) >= 2:
                            h_dim_name = dv_dims[-2]
                            w_dim_name = dv_dims[-1]
                            break
            if h_dim_name:
                break
        except Exception:
            continue

    def _save_var(idx: int, var_name: str, var_data: np.ndarray) -> str:
        prefix = f"{idx:02d}"
        filename = f"{prefix}_{var_name}.npy"
        out_path = os.path.join(static_dir, filename)
        np.save(out_path, var_data.astype(np.float32, copy=False))
        return out_path

    # ---- 第一阶段：从数据 NC 文件中搜索（静态变量可能就在动态文件里） ----
    for nc_file in nc_files:
        try:
            with xr.open_dataset(nc_file, decode_times=False) as ds:
                for idx, var_name in enumerate(all_static):
                    if var_name in saved:
                        continue
                    var_data = _extract_var_from_dataset(
                        ds, var_name, h_slice, w_slice, h_dim_name, w_dim_name
                    )
                    if var_data is not None:
                        saved[var_name] = _save_var(idx, var_name, var_data)

                if len(saved) == len(all_static):
                    break
        except Exception as e:
            warnings_list.append(f"从 {os.path.basename(nc_file)} 提取静态变量失败: {e}")
            continue

    if len(saved) == len(all_static):
        return saved

    # ---- 第二阶段：从网格文件中搜索（用户指定或自动检测） ----
    missing_vars = [v for v in all_static if v not in saved]

    # 如果用户没指定 grid_file，尝试自动检测
    effective_grid_file = grid_file
    if not effective_grid_file and nc_folder:
        effective_grid_file = _auto_detect_grid_file(nc_folder, dyn_file_pattern)
        if effective_grid_file:
            print(f"  自动检测到网格文件: {os.path.basename(effective_grid_file)}", file=sys.stderr)

    if missing_vars and effective_grid_file and os.path.isfile(effective_grid_file):
        print(
            f"  从网格文件搜索 {len(missing_vars)} 个未找到的静态变量: "
            f"{os.path.basename(effective_grid_file)}",
            file=sys.stderr
        )
        try:
            with xr.open_dataset(effective_grid_file, decode_times=False) as ds:
                for idx, var_name in enumerate(all_static):
                    if var_name in saved:
                        continue
                    var_data = _extract_var_from_dataset(
                        ds, var_name, h_slice, w_slice, h_dim_name, w_dim_name
                    )
                    if var_data is not None:
                        saved[var_name] = _save_var(idx, var_name, var_data)
                        print(f"    ✅ {var_name}: shape={var_data.shape}", file=sys.stderr)
        except Exception as e:
            warnings_list.append(
                f"从网格文件 {os.path.basename(effective_grid_file)} 提取静态变量失败: {e}"
            )

    # 记录未找到的变量
    for var_name in all_static:
        if var_name not in saved:
            warnings_list.append(f"静态变量 '{var_name}' 在所有 NC 文件和网格文件中均未找到")

    return saved


# ========================================
# 后置验证
# ========================================

def _validate_output(
    output_base: str,
    split_steps: Dict[str, List[TimeStepInfo]],
    dyn_vars: List[str],
    stat_vars: List[str],
    mask_vars: List[str],
    filename_map: Dict[str, str],  # ts_str → filename（不含扩展名）
    allow_nan: bool
) -> Dict[str, Any]:
    """
    后置验证三条规则:
    Rule 1: 输出完整性 - 所有 NPY 文件存在且形状正确
    Rule 2: 时间单调性 - 时间戳在各 split 内严格递增
    Rule 3: NaN 一致性 - 非掩码变量中 NaN 只出现在掩码位置
    """
    validation = {
        "rule1_integrity": {"passed": True, "errors": [], "warnings": []},
        "rule2_time_order": {"passed": True, "errors": []},
        "rule3_nan_consistency": {"passed": True, "errors": [], "skipped": False}
    }

    # ---- Rule 1: 完整性 ----
    shape_by_var: Dict[str, Optional[tuple]] = {}
    for split_name, steps in split_steps.items():
        for step in steps:
            fname = filename_map.get(step.ts_str, step.ts_str[:8])
            for var_name in dyn_vars:
                npy_path = os.path.join(output_base, split_name, var_name, f"{fname}.npy")
                if not os.path.exists(npy_path):
                    validation["rule1_integrity"]["errors"].append(
                        f"缺失文件: {split_name}/{var_name}/{fname}.npy"
                    )
                    validation["rule1_integrity"]["passed"] = False
                    continue

                data = np.load(npy_path)
                expected_shape = shape_by_var.get(var_name)
                if expected_shape is None:
                    shape_by_var[var_name] = data.shape
                elif data.shape != expected_shape:
                    validation["rule1_integrity"]["errors"].append(
                        f"形状不一致: {split_name}/{var_name}/{fname}.npy "
                        f"形状={data.shape}，期望={expected_shape}"
                    )
                    validation["rule1_integrity"]["passed"] = False

    # ---- Rule 2: 时间单调性 ----
    for split_name, steps in split_steps.items():
        for i in range(1, len(steps)):
            if steps[i].ts_float <= steps[i - 1].ts_float:
                validation["rule2_time_order"]["errors"].append(
                    f"split={split_name}, index={i}: "
                    f"时间非单调递增 {steps[i - 1].ts_str} ≥ {steps[i].ts_str}"
                )
                validation["rule2_time_order"]["passed"] = False

    # ---- Rule 3: NaN 一致性 ----
    if not allow_nan and mask_vars:
        # 取 train split 的第一个时间步做检查
        train_steps = split_steps.get('train', [])
        if train_steps:
            step = train_steps[0]
            fname = filename_map.get(step.ts_str, step.ts_str[:8])

            # 加载掩码
            mask_data = None
            static_dir = os.path.join(output_base, 'static_variables')
            for f in os.listdir(static_dir):
                base = f.replace('.npy', '')
                # 去掉编号前缀
                var_part = base.split('_', 1)[-1] if '_' in base else base
                if var_part in mask_vars:
                    try:
                        mask_data = np.load(os.path.join(static_dir, f))
                        break
                    except Exception:
                        pass

            if mask_data is not None:
                land_mask = (mask_data == 0)  # 0=陆地，1=海洋
                for var_name in dyn_vars:
                    npy_path = os.path.join(output_base, 'train', var_name, f"{fname}.npy")
                    if not os.path.exists(npy_path):
                        continue
                    try:
                        data = np.load(npy_path)
                        if data.ndim > 2:
                            data_2d = data[0]
                        else:
                            data_2d = data

                        nan_mask = np.isnan(data_2d)
                        # 在非陆地区域出现 NaN → 异常
                        ocean_nan = nan_mask & ~land_mask
                        ocean_nan_count = int(ocean_nan.sum())
                        if ocean_nan_count > 0:
                            ratio = ocean_nan_count / np.prod(data_2d.shape)
                            if ratio > 0.01:  # 超过 1% 才报错
                                validation["rule3_nan_consistency"]["errors"].append(
                                    f"{var_name}: 海洋区域出现 {ocean_nan_count} 个 NaN "
                                    f"({ratio:.1%})，疑似数据质量问题"
                                )
                                validation["rule3_nan_consistency"]["passed"] = False
                    except Exception as e:
                        validation["rule3_nan_consistency"]["errors"].append(
                            f"Rule3 检查 {var_name} 时出错: {e}"
                        )
        else:
            validation["rule3_nan_consistency"]["skipped"] = True
    else:
        validation["rule3_nan_consistency"]["skipped"] = True
        if allow_nan:
            validation["rule3_nan_consistency"]["warnings"] = ["allow_nan=true，跳过 NaN 一致性检查"]

    return validation


# ========================================
# 主函数
# ========================================

def forecast_preprocess(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    海洋预报数据预处理主函数

    Args:
        config: 配置字典

    Returns:
        处理结果字典
    """
    nc_folder        = config.get("nc_folder", "")
    output_base      = config.get("output_base", "")
    dyn_vars         = config.get("dyn_vars", [])
    stat_vars        = config.get("stat_vars", [])
    mask_vars        = config.get("mask_vars", [])
    lon_var          = config.get("lon_var")
    lat_var          = config.get("lat_var")
    train_ratio      = config.get("train_ratio", 0.7)
    valid_ratio      = config.get("valid_ratio", 0.15)
    test_ratio       = config.get("test_ratio", 0.15)
    h_slice_str      = config.get("h_slice")
    w_slice_str      = config.get("w_slice")
    dyn_file_pattern = config.get("dyn_file_pattern", "*.nc")
    chunk_size       = config.get("chunk_size", 200)
    use_date_filename= config.get("use_date_filename", True)
    date_format      = config.get("date_format", DATE_FORMAT_AUTO)
    time_var_hint    = config.get("time_var")
    max_files        = config.get("max_files")
    run_validation   = config.get("run_validation", True)
    allow_nan        = config.get("allow_nan", False)
    grid_file        = config.get("grid_file")

    result: Dict[str, Any] = {
        "status": "pending",
        "message": "",
        "warnings": [],
        "errors": [],
        "splits": {},
        "time_info": {},
        "static_vars_saved": [],
        "output_base": output_base,
        "post_validation": None
    }

    # ---- 1. 参数校验 ----
    if not nc_folder or not os.path.isdir(nc_folder):
        result["status"] = "error"
        result["errors"].append(f"NC 目录不存在或无效: {nc_folder}")
        return result

    if not output_base:
        result["status"] = "error"
        result["errors"].append("output_base 不能为空")
        return result

    if not dyn_vars:
        result["status"] = "error"
        result["errors"].append("dyn_vars 不能为空，必须指定动态变量列表")
        return result

    total_ratio = train_ratio + valid_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        result["status"] = "error"
        result["errors"].append(f"划分比例之和必须为 1.0，当前为 {total_ratio:.4f}")
        return result

    try:
        h_slice = _parse_slice(h_slice_str)
        w_slice = _parse_slice(w_slice_str)
    except ValueError as e:
        result["status"] = "error"
        result["errors"].append(str(e))
        return result

    # ---- 2. 扫描 NC 文件 ----
    search_pattern = os.path.join(nc_folder, dyn_file_pattern)
    nc_files = sorted(glob.glob(search_pattern))

    if not nc_files:
        result["status"] = "error"
        result["errors"].append(
            f"未找到 NC 文件: 目录={nc_folder}, 模式={dyn_file_pattern}"
        )
        return result

    if max_files and max_files > 0:
        nc_files = nc_files[:max_files]

    print(f"找到 {len(nc_files)} 个 NC 文件", file=sys.stderr)

    # ---- 3. 提取所有时间步时间戳 ----
    print("提取时间戳...", file=sys.stderr)

    all_steps: List[TimeStepInfo] = []
    failed_files: List[str] = []

    for nc_file in nc_files:
        ts_list = _extract_timestamps_from_file(nc_file, time_var_hint)
        if not ts_list:
            failed_files.append(os.path.basename(nc_file))
            print(f"  ⚠️ {os.path.basename(nc_file)}: 无法提取时间戳", file=sys.stderr)
            continue

        for ts_str, ts_float, local_idx in ts_list:
            all_steps.append(TimeStepInfo(ts_str, ts_float, nc_file, local_idx))

    if not all_steps:
        result["status"] = "error"
        result["errors"].append(
            "所有 NC 文件均无法提取时间戳！\n"
            "请检查：1) 时间变量名（可指定 time_var 参数）\n"
            "         2) 文件是否包含时间维度"
        )
        return result

    if failed_files:
        result["warnings"].append(
            f"以下 {len(failed_files)} 个文件无法提取时间戳（已跳过）: "
            + ", ".join(failed_files[:5])
            + ("..." if len(failed_files) > 5 else "")
        )

    # ---- 4. 按时间排序 ----
    print(f"按时间排序 {len(all_steps)} 个时间步...", file=sys.stderr)
    all_steps.sort(key=lambda s: s.ts_float)

    # 检查重复时间步
    seen_floats = {}
    dedup_steps: List[TimeStepInfo] = []
    for step in all_steps:
        if step.ts_float in seen_floats:
            result["warnings"].append(
                f"重复时间步: {step.ts_str}（来自 {os.path.basename(step.file_path)} 和 "
                f"{os.path.basename(seen_floats[step.ts_float].file_path)}），已保留第一个"
            )
        else:
            seen_floats[step.ts_float] = step
            dedup_steps.append(step)

    all_steps = dedup_steps
    total_steps = len(all_steps)
    print(f"去重后 {total_steps} 个时间步", file=sys.stderr)

    # ---- 5. 检查时间连续性 ----
    time_gaps, estimated_step_seconds = _check_time_continuity(all_steps)
    if time_gaps:
        gaps_summary = f"检测到 {len(time_gaps)} 处时间间隔异常"
        if len(time_gaps) <= 5:
            for g in time_gaps:
                gaps_summary += f"\n  {g.get('prev_ts', '?')} → {g.get('next_ts', '?')}: {g}"
        result["warnings"].append(gaps_summary)
        print(f"  ⚠️ {gaps_summary}", file=sys.stderr)
    else:
        print(
            f"  ✅ 时间连续性检查通过"
            + (f"，估计时间步 ≈ {estimated_step_seconds:.0f}s" if estimated_step_seconds else ""),
            file=sys.stderr
        )

    # ---- 6. 确定文件名格式 ----
    # 决定日期格式（需要在主循环前确定，用于命名一致性）
    filename_map: Dict[str, str] = {}  # ts_str → 文件名（无扩展名）

    if use_date_filename:
        # 预生成所有时间步的文件名，处理重名（同一天多步时加后缀）
        date_counter: Dict[str, int] = {}
        for step in all_steps:
            base_name = _format_date_filename(step.ts_str, date_format, estimated_step_seconds)
            count = date_counter.get(base_name, 0)
            if count == 0:
                final_name = base_name
            else:
                # 加上 HHmm 后缀避免冲突
                final_name = base_name + step.ts_str[8:12]
            date_counter[base_name] = count + 1
            filename_map[step.ts_str] = final_name
    else:
        for i, step in enumerate(all_steps):
            filename_map[step.ts_str] = f"{i:06d}"

    # ---- 7. 划分 train/valid/test ----
    t_train = int(total_steps * train_ratio)
    t_valid = int(total_steps * (train_ratio + valid_ratio))

    # 确保各 split 至少有 1 个时间步
    if t_train == 0:
        t_train = 1
    if t_valid <= t_train:
        t_valid = t_train + 1
    if t_valid >= total_steps:
        t_valid = total_steps - 1

    split_steps: Dict[str, List[TimeStepInfo]] = {
        'train': all_steps[:t_train],
        'valid': all_steps[t_train:t_valid],
        'test':  all_steps[t_valid:]
    }

    for split_name, steps in split_steps.items():
        print(f"  {split_name}: {len(steps)} 个时间步", file=sys.stderr)
        result["splits"][split_name] = {"timestep_count": len(steps)}

    # ---- 8. 创建输出目录 ----
    for split_name in ['train', 'valid', 'test']:
        for var_name in dyn_vars:
            os.makedirs(os.path.join(output_base, split_name, var_name), exist_ok=True)

    # ---- 9. 核心转换循环（按 chunk 分批处理） ----
    print(f"开始转换（chunk_size={chunk_size}）...", file=sys.stderr)

    total_saved = 0
    conversion_errors: List[str] = []

    # 按文件分组处理（避免重复打开同一文件）
    # 构建 {file_path → [list of TimeStepInfo]} 映射，并按原始排序顺序保留
    from collections import OrderedDict
    file_to_steps: Dict[str, List[TimeStepInfo]] = OrderedDict()
    for split_name, steps in split_steps.items():
        for step in steps:
            if step.file_path not in file_to_steps:
                file_to_steps[step.file_path] = []
            file_to_steps[step.file_path].append(step)

    # 按 chunk_size 批量打开文件
    file_paths = list(file_to_steps.keys())
    total_files = len(file_paths)
    processed_files = 0

    for chunk_start in range(0, total_files, chunk_size):
        chunk_files = file_paths[chunk_start: chunk_start + chunk_size]
        chunk_steps_all = sum((file_to_steps[f] for f in chunk_files), [])
        print(
            f"  处理文件 {chunk_start + 1}-{min(chunk_start + chunk_size, total_files)}"
            f"/{total_files}（{len(chunk_steps_all)} 步）...",
            file=sys.stderr
        )

        for nc_file in chunk_files:
            steps_in_file = file_to_steps[nc_file]
            try:
                with xr.open_dataset(nc_file, decode_times=False) as ds:
                    for step in steps_in_file:
                        fname = filename_map[step.ts_str]
                        # 判断当前 step 属于哪个 split
                        step_split = None
                        for split_name, split_list in split_steps.items():
                            if any(s.ts_str == step.ts_str and s.file_path == step.file_path for s in split_list):
                                step_split = split_name
                                break

                        if step_split is None:
                            continue

                        for var_name in dyn_vars:
                            if var_name not in ds.variables:
                                conversion_errors.append(
                                    f"变量 '{var_name}' 不在文件 {os.path.basename(nc_file)} 中"
                                )
                                continue

                            try:
                                var_data = ds[var_name].values
                                # 提取指定时间步
                                if var_data.ndim > 2:
                                    ts_data = var_data[step.local_idx]
                                else:
                                    ts_data = var_data  # 静态变量作为动态处理（罕见）

                                # 应用空间裁剪
                                ts_data = _apply_spatial_slice(ts_data, h_slice, w_slice)

                                # 保存
                                out_dir = os.path.join(output_base, step_split, var_name)
                                _write_timestep(out_dir, fname, ts_data)
                                total_saved += 1

                            except Exception as e:
                                conversion_errors.append(
                                    f"{step_split}/{var_name}/{fname}: {e}"
                                )

            except Exception as e:
                conversion_errors.append(f"打开文件 {os.path.basename(nc_file)} 失败: {e}")

        processed_files += len(chunk_files)

    if conversion_errors:
        result["warnings"].extend(conversion_errors[:20])
        if len(conversion_errors) > 20:
            result["warnings"].append(f"... 还有 {len(conversion_errors) - 20} 个转换错误")

    print(f"  ✅ 共保存 {total_saved} 个 NPY 文件", file=sys.stderr)

    # ---- 10. 保存静态变量 ----
    print("保存静态变量...", file=sys.stderr)
    static_saved = _extract_and_save_static_vars(
        nc_files=nc_files,
        stat_vars=stat_vars,
        mask_vars=mask_vars,
        output_base=output_base,
        h_slice=h_slice,
        w_slice=w_slice,
        dyn_vars=dyn_vars,
        warnings_list=result["warnings"],
        grid_file=grid_file,
        nc_folder=nc_folder,
        dyn_file_pattern=dyn_file_pattern
    )
    result["static_vars_saved"] = list(static_saved.keys())

    # ---- 11. 确定空间形状 ----
    spatial_shape = None
    # 从已保存的文件中读取形状
    for split_name, steps in split_steps.items():
        if steps and dyn_vars:
            fname = filename_map[steps[0].ts_str]
            npy_path = os.path.join(output_base, split_name, dyn_vars[0], f"{fname}.npy")
            if os.path.exists(npy_path):
                arr = np.load(npy_path)
                spatial_shape = list(arr.shape)
                break

    # ---- 11.5 自动补全 lon/lat 坐标数组 ----
    # 如果用户指定的 lon_var/lat_var 不是 2D 坐标网格，自动寻找匹配数据空间维度的坐标
    # 典型场景：ROMS 数据中 lon_rho/lat_rho 是 (H, W) 网格，匹配 rho 网格点
    effective_lon_var = lon_var
    effective_lat_var = lat_var

    if spatial_shape is not None:
        hw_shape = tuple(spatial_shape[-2:])  # (H, W)
        static_dir = os.path.join(output_base, 'static_variables')

        # 检查当前 lon_var 是否为有效的 2D 坐标网格
        lon_is_2d = False
        lat_is_2d = False
        for f in os.listdir(static_dir):
            if not f.endswith('.npy'):
                continue
            var_part = f.split('_', 1)[-1].replace('.npy', '')
            if var_part == lon_var:
                arr = np.load(os.path.join(static_dir, f))
                if arr.ndim == 2 and arr.shape == hw_shape:
                    lon_is_2d = True
            if var_part == lat_var:
                arr = np.load(os.path.join(static_dir, f))
                if arr.ndim == 2 and arr.shape == hw_shape:
                    lat_is_2d = True

        # 如果 lon_var/lat_var 不是有效的 2D 网格，尝试从网格文件补充
        if not lon_is_2d or not lat_is_2d:
            # 常见的 rho 网格坐标名（优先级从高到低）
            lon_candidates = ['lon_rho', 'longitude', 'nav_lon', 'TLONG']
            lat_candidates = ['lat_rho', 'latitude', 'nav_lat', 'TLAT']

            effective_grid = grid_file
            if not effective_grid and nc_folder:
                effective_grid = _auto_detect_grid_file(nc_folder, dyn_file_pattern)

            # 搜索来源：数据文件 + 网格文件
            search_files = nc_files[:1]
            if effective_grid and os.path.isfile(effective_grid):
                search_files.append(effective_grid)

            for src_file in search_files:
                if lon_is_2d and lat_is_2d:
                    break
                try:
                    with xr.open_dataset(src_file, decode_times=False) as ds:
                        if not lon_is_2d:
                            for cand in lon_candidates:
                                if cand in ds.variables and ds[cand].ndim == 2:
                                    cand_shape = ds[cand].shape
                                    if cand_shape == hw_shape:
                                        data = ds[cand].values.astype(np.float32)
                                        idx = len(stat_vars) + len(mask_vars)
                                        fname = f"{idx:02d}_{cand}.npy"
                                        out_path = os.path.join(static_dir, fname)
                                        np.save(out_path, data)
                                        static_saved[cand] = out_path
                                        effective_lon_var = cand
                                        lon_is_2d = True
                                        print(f"  ✅ 自动补充坐标: {cand} shape={cand_shape}", file=sys.stderr)
                                        break
                        if not lat_is_2d:
                            for cand in lat_candidates:
                                if cand in ds.variables and ds[cand].ndim == 2:
                                    cand_shape = ds[cand].shape
                                    if cand_shape == hw_shape:
                                        data = ds[cand].values.astype(np.float32)
                                        idx = len(stat_vars) + len(mask_vars) + 1
                                        fname = f"{idx:02d}_{cand}.npy"
                                        out_path = os.path.join(static_dir, fname)
                                        np.save(out_path, data)
                                        static_saved[cand] = out_path
                                        effective_lat_var = cand
                                        lat_is_2d = True
                                        print(f"  ✅ 自动补充坐标: {cand} shape={cand_shape}", file=sys.stderr)
                                        break
                except Exception:
                    continue

        result["static_vars_saved"] = list(static_saved.keys())

    # ---- 12. 生成 var_names.json ----
    var_names_data = {
        "dynamic": dyn_vars,
        "static": stat_vars,
        "mask": mask_vars,
        "lon_var": effective_lon_var,
        "lat_var": effective_lat_var,
        "spatial_shape": spatial_shape,
        "use_date_filename": use_date_filename,
        "date_format": date_format,
        "splits": {k: len(v) for k, v in split_steps.items()},
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    }
    var_names_path = os.path.join(output_base, 'var_names.json')
    with open(var_names_path, 'w', encoding='utf-8') as f:
        json.dump(var_names_data, f, ensure_ascii=False, indent=2)

    # ---- 13. 生成 time_index.json ----
    time_index_data = {
        "global": {
            "total_steps": total_steps,
            "start_ts": all_steps[0].ts_str if all_steps else None,
            "end_ts": all_steps[-1].ts_str if all_steps else None,
            "estimated_step_seconds": _safe_float(estimated_step_seconds),
            "source_files_count": len(nc_files),
            "time_gaps_count": len(time_gaps)
        },
        "splits": {},
        "time_gaps": time_gaps[:50],  # 最多保存50个间隔记录
        "filename_format": date_format
    }
    for split_name, steps in split_steps.items():
        time_index_data["splits"][split_name] = {
            "count": len(steps),
            "start_ts": steps[0].ts_str if steps else None,
            "end_ts": steps[-1].ts_str if steps else None,
            "timestamps": [s.ts_str for s in steps],
            "filenames": [filename_map[s.ts_str] for s in steps]
        }

    time_index_path = os.path.join(output_base, 'time_index.json')
    with open(time_index_path, 'w', encoding='utf-8') as f:
        json.dump(time_index_data, f, ensure_ascii=False, indent=2)

    # ---- 14. 生成 preprocess_manifest.json ----
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "nc_folder": nc_folder,
        "output_base": output_base,
        "source_files": [os.path.basename(f) for f in nc_files],
        "dyn_vars": dyn_vars,
        "stat_vars": stat_vars,
        "mask_vars": mask_vars,
        "spatial_shape": spatial_shape,
        "h_slice": h_slice_str,
        "w_slice": w_slice_str,
        "split_ratios": {
            "train": train_ratio,
            "valid": valid_ratio,
            "test": test_ratio
        },
        "split_counts": {k: len(v) for k, v in split_steps.items()},
        "total_npy_files": total_saved,
        "warnings": result["warnings"]
    }
    manifest_path = os.path.join(output_base, 'preprocess_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ---- 15. 后置验证 ----
    if run_validation:
        print("执行后置验证...", file=sys.stderr)
        validation_result = _validate_output(
            output_base=output_base,
            split_steps=split_steps,
            dyn_vars=dyn_vars,
            stat_vars=stat_vars,
            mask_vars=mask_vars,
            filename_map=filename_map,
            allow_nan=allow_nan
        )
        result["post_validation"] = validation_result

        # 汇总验证状态
        all_passed = all(
            v["passed"] for v in validation_result.values()
            if isinstance(v, dict) and "passed" in v
        )
        if not all_passed:
            for rule_name, rule_result in validation_result.items():
                if isinstance(rule_result, dict) and not rule_result.get("passed", True):
                    for err in rule_result.get("errors", []):
                        result["warnings"].append(f"[{rule_name}] {err}")
    else:
        result["post_validation"] = {"skipped": True}

    # ---- 15. 汇总时间信息 ----
    result["time_info"] = {
        "total_steps": total_steps,
        "start": all_steps[0].ts_str if all_steps else None,
        "end": all_steps[-1].ts_str if all_steps else None,
        "estimated_step_seconds": _safe_float(estimated_step_seconds),
        "time_gaps_count": len(time_gaps)
    }

    # ---- 16. 生成 preprocess_manifest.json ----
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "nc_folder": nc_folder,
        "output_base": output_base,
        "source_files": [os.path.basename(f) for f in nc_files],
        "dyn_vars": dyn_vars,
        "stat_vars": stat_vars,
        "mask_vars": mask_vars,
        "spatial_shape": spatial_shape,
        "h_slice": h_slice_str,
        "w_slice": w_slice_str,
        "split_ratios": {
            "train": train_ratio,
            "valid": valid_ratio,
            "test": test_ratio
        },
        "split_counts": {k: len(v) for k, v in split_steps.items()},
        "total_npy_files": total_saved,
        "warnings": result["warnings"],
        "post_validation": result["post_validation"],
        "time_info": result["time_info"]
    }
    manifest_path = os.path.join(output_base, 'preprocess_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ---- 17. 最终状态 ----
    result["status"] = "pass"
    result["message"] = (
        f"预处理完成！共处理 {total_steps} 个时间步，"
        f"保存 {total_saved} 个 NPY 文件。"
        f"train={len(split_steps['train'])}, "
        f"valid={len(split_steps['valid'])}, "
        f"test={len(split_steps['test'])}。"
        f"输出目录: {output_base}"
    )

    if result["warnings"]:
        result["message"] += f"\n⚠️ 有 {len(result['warnings'])} 个警告，请检查。"

    print(f"✅ {result['message']}", file=sys.stderr)
    return result


# ========================================
# CLI 入口
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description="海洋预报数据预处理：NC → NPY（严格时间排序）"
    )
    parser.add_argument("--config", required=True, help="JSON 配置文件路径")
    parser.add_argument("--output", required=True, help="结果输出 JSON 路径")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    result = forecast_preprocess(config)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj) if not (math.isnan(obj) or math.isinf(obj)) else None
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print(json.dumps(result, ensure_ascii=False, cls=NumpyEncoder))


if __name__ == "__main__":
    main()
