"""
converters.py - 核心转换函数

从 convert_npy.py 拆分
"""

import os
import sys
import numpy as np
import xarray as xr
from typing import Any, Dict, List, Optional, Tuple

from .constants import DEFAULT_WORKERS, LON_VARS, LAT_VARS
from .crop import _parallel_extract_var, crop_spatial, validate_crop_divisible
from .check import (
    is_object_dtype,
    check_nan_inf_sampling,
    get_spatial_shape,
    verify_coordinate_range,
    get_static_var_prefix
)


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
    """
    # 创建输出目录结构
    splits = ['train', 'valid', 'test']
    for split in splits:
        subdir = os.path.join(output_dir, split, output_subdir)
        lr_dir = os.path.join(output_dir, split, 'lr')
        os.makedirs(subdir, exist_ok=True)
        os.makedirs(lr_dir, exist_ok=True)

    saved_files = {}
    time_lengths = {}
    nan_check_results = {}
    split_info = {}

    print(f"处理动态变量，文件数: {len(nc_files)}", file=sys.stderr)
    print(f"划分比例: train={train_ratio}, valid={valid_ratio}, test={test_ratio}", file=sys.stderr)
    print(f"并行进程数: {workers}", file=sys.stderr)

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

            # object dtype 检查
            if is_object_dtype(data_arr):
                msg = f"动态变量 '{var}' 是 object dtype，禁止使用"
                result["errors"].append(msg)
                print(f"    错误: {msg}", file=sys.stderr)
                continue

            # 检测零长度维度
            if any(d == 0 for d in data_arr.shape):
                zero_dims = [i for i, d in enumerate(data_arr.shape) if d == 0]
                msg = f"动态变量 '{var}' 有零长度维度，位置: {zero_dims}"
                result["errors"].append(msg)
                print(f"    错误: {msg}", file=sys.stderr)
                continue

            # 检测维度数量
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

            # 验证裁剪后尺寸能否被 scale 整除
            if scale is not None:
                cropped_h, cropped_w = data_arr.shape[-2], data_arr.shape[-1]
                is_valid, msg = validate_crop_divisible(cropped_h, cropped_w, scale)
                if not is_valid:
                    result["errors"].append(f"变量 '{var}' {msg}")
                    print(f"    错误: {msg}", file=sys.stderr)
                    continue
                else:
                    print(f"    {msg}", file=sys.stderr)

            # 按时间顺序划分数据集
            total_time = data_arr.shape[0]

            if total_time < 3:
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

                train_end = max(1, train_end)
                valid_end = max(train_end, min(valid_end, total_time))

                if valid_ratio > 0 and valid_end == train_end and train_end < total_time:
                    valid_end = min(train_end + 1, total_time)

                train_data = data_arr[0:train_end]
                valid_data = data_arr[train_end:valid_end] if valid_end > train_end else np.array([]).reshape(0, *data_arr.shape[1:])
                test_data = data_arr[valid_end:total_time] if total_time > valid_end else np.array([]).reshape(0, *data_arr.shape[1:])

                split_info[var] = {
                    "total_time": total_time,
                    "train": {"start": 0, "end": train_end, "count": train_end},
                    "valid": {"start": train_end, "end": valid_end, "count": valid_end - train_end},
                    "test": {"start": valid_end, "end": total_time, "count": total_time - valid_end}
                }

            print(f"    划分: train={split_info[var]['train']['count']}, valid={split_info[var]['valid']['count']}, test={split_info[var]['test']['count']}", file=sys.stderr)

            # 保存到对应目录
            spatial_shape = get_spatial_shape(data_arr)
            var_saved_files = {}

            for split_name, split_data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
                if split_data.size == 0:
                    continue

                var_dir = os.path.join(output_dir, split_name, output_subdir, var)
                os.makedirs(var_dir, exist_ok=True)

                time_steps = split_data.shape[0]
                saved_count = 0
                for t in range(time_steps):
                    filename = f"{t:06d}.npy"
                    out_fp = os.path.join(var_dir, filename)
                    np.save(out_fp, split_data[t])
                    saved_count += 1

                var_saved_files[split_name] = {
                    "dir": var_dir,
                    "file_count": saved_count,
                    "sample_shape": list(split_data[0].shape),
                    "total_shape": list(split_data.shape),
                    "spatial_shape": list(spatial_shape),
                    "dtype": str(split_data.dtype),
                    "time_steps": time_steps
                }

                print(f"      {split_name}/{output_subdir}/{var}/: {saved_count} 个文件, 每个 shape={split_data[0].shape}", file=sys.stderr)

            time_lengths[var] = total_time

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
    fallback_nc_files: Optional[List[str]] = None,
    output_subdir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    转换静态变量（带编号前缀）
    """
    # 确定输出目录
    if output_subdir:
        sta_out_dir = os.path.join(output_dir, 'static_variables', output_subdir)
    else:
        sta_out_dir = os.path.join(output_dir, 'static_variables')
    os.makedirs(sta_out_dir, exist_ok=True)

    saved_files = {}
    other_idx = 0
    coord_checks = {}

    # 确定要读取的文件
    source_file = None
    source_type = None

    if static_file and os.path.exists(static_file):
        source_file = static_file
        source_type = "static_file"
    elif fallback_nc_files and len(fallback_nc_files) > 0:
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

                    if is_object_dtype(data_arr):
                        msg = f"静态变量 '{var}' 是 object dtype，禁止使用"
                        result["errors"].append(msg)
                        print(f"    错误: {msg}", file=sys.stderr)
                        continue

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

                    if is_lon and lon_range:
                        coord_check = verify_coordinate_range(data_arr, var, lon_range)
                        coord_checks[var] = coord_check
                        if coord_check.get("has_nan") or coord_check.get("has_inf"):
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
                            result["errors"].append(coord_check["message"])
                            print(f"    错误: {coord_check['message']}", file=sys.stderr)
                            continue
                        elif not coord_check["in_range"]:
                            result["warnings"].append(coord_check["message"])
                        if coord_check["actual_min"] is not None:
                            print(f"    纬度范围: [{coord_check['actual_min']:.4f}, {coord_check['actual_max']:.4f}]", file=sys.stderr)

                    if is_mask:
                        unique_vals = np.unique(data_arr)
                        is_binary = len(unique_vals) <= 2 and all(v in [0, 1] for v in unique_vals)
                        if not is_binary:
                            result["warnings"].append(
                                f"掩码变量 '{var}' 不是二值 (0/1): 唯一值 = {unique_vals[:10].tolist()}"
                                + ("..." if len(unique_vals) > 10 else "")
                            )

                    # 空间裁剪
                    original_shape = data_arr.shape
                    if h_slice is not None or w_slice is not None:
                        if data_arr.ndim == 1:
                            if is_lat and h_slice is not None:
                                data_arr = data_arr[h_slice]
                                print(f"    裁剪 latitude: {original_shape} -> {data_arr.shape} (使用 h_slice)", file=sys.stderr)
                            elif is_lon and w_slice is not None:
                                data_arr = data_arr[w_slice]
                                print(f"    裁剪 longitude: {original_shape} -> {data_arr.shape} (使用 w_slice)", file=sys.stderr)
                            else:
                                print(f"    1D 变量 '{var}'，不裁剪", file=sys.stderr)
                        else:
                            data_arr = crop_spatial(data_arr, h_slice, w_slice)
                            print(f"    裁剪: {original_shape} -> {data_arr.shape}", file=sys.stderr)

                    np.save(out_fp, data_arr)

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
