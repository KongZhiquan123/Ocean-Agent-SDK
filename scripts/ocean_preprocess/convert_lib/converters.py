"""
converters.py - 核心转换函数

从 convert_npy.py 拆分

@changelog
  - 2026-02-07 Leizheng: 修复 BrokenPipeError — stderr 写入加保护
  - 2026-02-07 Leizheng: 流式重构 convert_dynamic_vars，内存从 O(N) 降到 O(1)
  - 2026-02-06 Leizheng: 添加写入进度日志与各阶段耗时统计
  - 2026-02-05 kongzhiquan: 新增日期文件名功能
    - 新增 use_date_filename, date_format, time_var 参数
    - 支持从 NC 文件提取时间戳作为文件名
    - 时间提取失败时自动回退到纯序号命名
"""

import os
import sys
import time
import numpy as np
import xarray as xr
from typing import Any, Dict, List, Optional, Tuple

from .constants import DEFAULT_WORKERS, LON_VARS, LAT_VARS, DEFAULT_DATE_FORMAT
from .crop import (
    _extract_var_from_file,
    _parallel_extract_var,
    crop_spatial,
    validate_crop_divisible,
    parallel_stream_extract_save,
    _safe_print,
)
from .check import (
    is_object_dtype,
    check_nan_inf_sampling,
    get_spatial_shape,
    verify_coordinate_range,
    get_static_var_prefix
)
from .time_utils import (
    extract_timestamps_from_files,
    detect_date_format,
    generate_date_filenames,
    validate_time_monotonic,
    create_time_mapping
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
    output_subdir: str = 'hr',
    use_date_filename: bool = False,
    date_format: str = DEFAULT_DATE_FORMAT,
    time_var: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    转换动态变量，并按时间顺序划分为 train/valid/test

    Args:
        nc_files: NC 文件列表
        dyn_vars: 动态变量名列表
        output_dir: 输出目录
        result: 结果字典（用于记录警告和错误）
        allow_nan: 是否允许 NaN 值
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        test_ratio: 测试集比例
        h_slice: 高度方向裁剪
        w_slice: 宽度方向裁剪
        scale: 下采样倍数（用于验证尺寸）
        workers: 并行进程数
        output_subdir: 输出子目录名（hr 或 lr）
        use_date_filename: 是否使用日期作为文件名
        date_format: 日期格式（auto/YYYYMMDD/YYYYMMDDHH/YYYYMMDDHHmm）
        time_var: 指定的时间变量名（None 则自动检测）

    Returns:
        保存的文件信息字典
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

    # ========== 日期文件名处理 ==========
    timestamps = None
    date_filenames = None
    actual_date_format = None
    detected_time_var = None

    if use_date_filename:
        _safe_print(f"正在提取时间信息用于文件命名...")

        timestamps, detected_time_var, time_warnings = extract_timestamps_from_files(
            nc_files, time_var
        )

        if time_warnings:
            for w in time_warnings:
                result["warnings"].append(f"时间提取: {w}")

        if timestamps is None:
            # 时间提取失败，回退到纯序号
            _safe_print(f"  [Warning] 时间提取失败，回退到纯序号命名")
            result["warnings"].append("时间提取失败，使用纯序号命名")
            use_date_filename = False
        else:
            # 验证时间单调性
            is_monotonic, mono_msg = validate_time_monotonic(timestamps)
            if not is_monotonic:
                _safe_print(f"  [Warning] {mono_msg}，回退到纯序号命名")
                result["warnings"].append(f"时间非单调递增: {mono_msg}，使用纯序号命名")
                use_date_filename = False
            else:
                # 确定日期格式
                if date_format == "auto":
                    actual_date_format = detect_date_format(timestamps)
                    _safe_print(f"  自动检测日期格式: {actual_date_format}")
                else:
                    actual_date_format = date_format

                # 生成文件名
                date_filenames = generate_date_filenames(timestamps, actual_date_format)

                _safe_print(f"  时间变量: {detected_time_var}")
                _safe_print(f"  时间范围: {timestamps[0]} 到 {timestamps[-1]}")
                _safe_print(f"  时间步数: {len(timestamps)}")
                _safe_print(f"  文件命名示例: {date_filenames[0]}.npy")

                # 记录时间信息到结果
                result["time_info"] = {
                    "use_date_filename": True,
                    "detected_time_var": detected_time_var,
                    "date_format": actual_date_format,
                    "total_timestamps": len(timestamps),
                    "time_range": {
                        "start": timestamps[0].isoformat(),
                        "end": timestamps[-1].isoformat()
                    },
                    "filename_examples": {
                        "first": date_filenames[0],
                        "last": date_filenames[-1]
                    }
                }

    _safe_print(f"处理动态变量，文件数: {len(nc_files)}")
    _safe_print(f"划分比例: train={train_ratio}, valid={valid_ratio}, test={test_ratio}")
    _safe_print(f"并行进程数: {workers}")
    if use_date_filename:
        _safe_print(f"文件命名: 日期格式 ({actual_date_format})")
    else:
        _safe_print(f"文件命名: 纯序号 (000000, 000001, ...)")

    for var in dyn_vars:
        try:
            var_t0 = time.time()

            # ========== Phase 0: 预扫描（只读第一个文件，获取 shape/dtype） ==========
            sample = _extract_var_from_file(nc_files[0], var, h_slice, w_slice)
            if sample["status"] != "success":
                result["warnings"].append(f"动态变量 '{var}' 预扫描失败: {sample.get('error', 'unknown')}")
                continue

            sample_data = sample["data"]

            # object dtype 检查
            if is_object_dtype(sample_data):
                msg = f"动态变量 '{var}' 是 object dtype，禁止使用"
                result["errors"].append(msg)
                _safe_print(f"    错误: {msg}")
                continue

            # 维度检查
            if sample_data.ndim not in [3, 4]:
                msg = f"动态变量 '{var}' 维度数量错误: 实际 {sample_data.ndim}D，预期 3D 或 4D"
                result["errors"].append(msg)
                _safe_print(f"    错误: {msg}")
                continue

            # 零长度维度检查
            if any(d == 0 for d in sample_data.shape):
                zero_dims = [i for i, d in enumerate(sample_data.shape) if d == 0]
                msg = f"动态变量 '{var}' 有零长度维度，位置: {zero_dims}"
                result["errors"].append(msg)
                _safe_print(f"    错误: {msg}")
                continue

            per_file_T = sample_data.shape[0]
            spatial_shape = get_spatial_shape(sample_data)
            sample_dtype = str(sample_data.dtype)
            sample_ndim = sample_data.ndim
            # 每个时间步的形状（去掉第0维的时间轴）
            step_shape = list(sample_data.shape[1:])

            total_T = per_file_T * len(nc_files)

            _safe_print(f"  变量 '{var}' 预扫描: per_file_T={per_file_T}, total_T={total_T}, "
                  f"step_shape={step_shape}, dtype={sample_dtype}")

            # 验证裁剪后尺寸能否被 scale 整除
            if scale is not None:
                cropped_h, cropped_w = sample_data.shape[-2], sample_data.shape[-1]
                is_valid, msg = validate_crop_divisible(cropped_h, cropped_w, scale)
                if not is_valid:
                    result["errors"].append(f"变量 '{var}' {msg}")
                    _safe_print(f"    错误: {msg}")
                    continue
                else:
                    _safe_print(f"    {msg}")

            # 释放采样数据
            del sample_data, sample

            # ========== Phase 1: 计算分界 + 构建任务列表 ==========
            if total_T < 3:
                _safe_print(f"    时间步数({total_T})太少，全部放入 train")
                train_end = total_T
                valid_end = total_T

                split_info[var] = {
                    "total_time": total_T,
                    "train": {"start": 0, "end": total_T, "count": total_T},
                    "valid": {"start": total_T, "end": total_T, "count": 0},
                    "test": {"start": total_T, "end": total_T, "count": 0},
                    "note": "时间步太少，全部放入 train"
                }
            else:
                train_end = int(total_T * train_ratio)
                valid_end = int(total_T * (train_ratio + valid_ratio))

                train_end = max(1, train_end)
                valid_end = max(train_end, min(valid_end, total_T))

                if valid_ratio > 0 and valid_end == train_end and train_end < total_T:
                    valid_end = min(train_end + 1, total_T)

                split_info[var] = {
                    "total_time": total_T,
                    "train": {"start": 0, "end": train_end, "count": train_end},
                    "valid": {"start": train_end, "end": valid_end, "count": valid_end - train_end},
                    "test": {"start": valid_end, "end": total_T, "count": total_T - valid_end}
                }

            _safe_print(f"    划分: train={split_info[var]['train']['count']}, "
                  f"valid={split_info[var]['valid']['count']}, "
                  f"test={split_info[var]['test']['count']}")

            # 创建输出目录
            for split in splits:
                var_dir = os.path.join(output_dir, split, output_subdir, var)
                os.makedirs(var_dir, exist_ok=True)

            # 将 slice 转为可 pickle 的 tuple
            h_slice_tuple = (h_slice.start, h_slice.stop) if h_slice is not None else None
            w_slice_tuple = (w_slice.start, w_slice.stop) if w_slice is not None else None

            # 构建任务列表：每个 NC 文件一个任务
            tasks = []
            for file_idx, nc_file in enumerate(nc_files):
                output_map = []  # [(local_t, output_path), ...]
                for local_t in range(per_file_T):
                    global_t = file_idx * per_file_T + local_t

                    # 确定属于哪个 split
                    if global_t < train_end:
                        split_name = "train"
                    elif global_t < valid_end:
                        split_name = "valid"
                    else:
                        split_name = "test"

                    # 文件名
                    if use_date_filename and date_filenames and global_t < len(date_filenames):
                        filename = f"{date_filenames[global_t]}.npy"
                    else:
                        # 使用 split 内部的局部序号
                        split_start = split_info[var][split_name]["start"]
                        local_idx = global_t - split_start
                        filename = f"{local_idx:06d}.npy"

                    out_path = os.path.join(output_dir, split_name, output_subdir, var, filename)
                    output_map.append((local_t, out_path))

                tasks.append({
                    "nc_file": nc_file,
                    "var_name": var,
                    "output_map": output_map,
                    "h_slice": h_slice_tuple,
                    "w_slice": w_slice_tuple,
                })

            # ========== Phase 2: 并行流式处理 ==========
            stream_results, stream_errors = parallel_stream_extract_save(tasks, workers)

            if stream_errors:
                for err in stream_errors:
                    result["warnings"].append(f"提取 '{var}' 警告: {err}")

            if not stream_results:
                result["warnings"].append(f"动态变量 '{var}' 流式处理完全失败")
                continue

            # ========== Phase 3: 汇总结果 ==========
            total_nan = sum(r.get("nan_count", 0) for r in stream_results)
            total_inf = sum(r.get("inf_count", 0) for r in stream_results)
            total_elements = sum(r.get("total_elements", 0) for r in stream_results)
            total_saved = sum(r.get("saved_count", 0) for r in stream_results)

            # 构造与 check_nan_inf_sampling 兼容的结果
            nan_result = {
                "var_name": var,
                "shape": [total_T] + step_shape,
                "dtype": sample_dtype,
                "has_nan": total_nan > 0,
                "has_inf": total_inf > 0,
                "nan_count": total_nan,
                "inf_count": total_inf,
                "checked_samples": total_elements,
                "total_elements": total_elements,
                "pass": total_nan == 0 and total_inf == 0,
            }
            nan_check_results[var] = nan_result

            if not nan_result["pass"]:
                msg = (f"动态变量 '{var}' 含有非法值: "
                       f"NaN={total_nan}, Inf={total_inf}")
                if allow_nan:
                    result["warnings"].append(msg + " (allow_nan=True, 允许)")
                else:
                    # 文件已经保存了，但仍然记录错误
                    result["errors"].append(msg)
                    _safe_print(f"    错误: {msg}")
                    # 注意：流式模式下文件已写入，此处不 continue 以保留 saved_files 信息

            time_lengths[var] = total_T

            # 构造 per-split 的 saved_files 信息
            var_saved_files = {}
            for split_name in splits:
                si = split_info[var][split_name]
                count = si["count"]
                if count == 0:
                    continue

                var_dir = os.path.join(output_dir, split_name, output_subdir, var)

                # 计算首末文件名
                split_start_global = si["start"]
                split_end_global = si["end"]

                if use_date_filename and date_filenames and split_start_global < len(date_filenames):
                    first_file = f"{date_filenames[split_start_global]}.npy"
                    last_idx = min(split_end_global - 1, len(date_filenames) - 1)
                    last_file = f"{date_filenames[last_idx]}.npy"
                else:
                    first_file = f"{0:06d}.npy"
                    last_file = f"{count - 1:06d}.npy"

                var_saved_files[split_name] = {
                    "dir": var_dir,
                    "file_count": count,
                    "sample_shape": step_shape,
                    "total_shape": [count] + step_shape,
                    "spatial_shape": list(spatial_shape),
                    "dtype": sample_dtype,
                    "time_steps": count,
                    "filename_pattern": "date" if use_date_filename else "sequential",
                    "first_file": first_file,
                    "last_file": last_file,
                }

                _safe_print(f"      {split_name}/{output_subdir}/{var}/: {count} 个文件, "
                      f"每个 shape={tuple(step_shape)}")

            # 构造整体形状描述
            total_shape = [total_T] + step_shape
            if sample_ndim == 3:
                interp = f"[T={total_T}, H={step_shape[0]}, W={step_shape[1]}]"
            elif sample_ndim == 4:
                interp = f"[T={total_T}, D={step_shape[0]}, H={step_shape[1]}, W={step_shape[2]}]"
            else:
                interp = f"shape={total_shape}"

            saved_files[var] = {
                "splits": var_saved_files,
                "total_shape": total_shape,
                "spatial_shape": list(spatial_shape),
                "dtype": sample_dtype,
                "interpretation": interp,
                "is_dynamic": True,
                "nan_check": nan_result,
                "split_info": split_info[var]
            }

            _safe_print(f"    完成，total_shape={tuple(total_shape)}, spatial={spatial_shape}, "
                  f"保存 {total_saved} 个文件")
            var_elapsed = time.time() - var_t0
            _safe_print(f"    变量 '{var}' 总耗时: {var_elapsed:.1f}s")

        except Exception as e:
            import traceback
            result["warnings"].append(f"处理动态变量 '{var}' 失败: {str(e)}")
            _safe_print(f"    异常: {traceback.format_exc()}")

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
        _safe_print(f"静态文件未提供，尝试从动态文件中提取静态变量: {source_file}")
    else:
        if static_file:
            result["warnings"].append(f"静态文件不存在: {static_file}")
        else:
            result["warnings"].append("未提供静态文件，且无动态文件可用于提取静态变量")
        return saved_files

    _safe_print(f"处理静态变量 (来源: {source_type}): {source_file}")

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
                    _safe_print(f"  提取静态变量: {var} -> {filename} ...")
                    data_arr = ds[var].values

                    is_mask = var in mask_vars
                    is_lon = var in LON_VARS
                    is_lat = var in LAT_VARS

                    if is_object_dtype(data_arr):
                        msg = f"静态变量 '{var}' 是 object dtype，禁止使用"
                        result["errors"].append(msg)
                        _safe_print(f"    错误: {msg}")
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
                                _safe_print(f"    错误: {msg}")
                                continue

                    if is_lon and lon_range:
                        coord_check = verify_coordinate_range(data_arr, var, lon_range)
                        coord_checks[var] = coord_check
                        if coord_check.get("has_nan") or coord_check.get("has_inf"):
                            result["errors"].append(coord_check["message"])
                            _safe_print(f"    错误: {coord_check['message']}")
                            continue
                        elif not coord_check["in_range"]:
                            result["warnings"].append(coord_check["message"])
                        if coord_check["actual_min"] is not None:
                            _safe_print(f"    经度范围: [{coord_check['actual_min']:.4f}, {coord_check['actual_max']:.4f}]")

                    if is_lat and lat_range:
                        coord_check = verify_coordinate_range(data_arr, var, lat_range)
                        coord_checks[var] = coord_check
                        if coord_check.get("has_nan") or coord_check.get("has_inf"):
                            result["errors"].append(coord_check["message"])
                            _safe_print(f"    错误: {coord_check['message']}")
                            continue
                        elif not coord_check["in_range"]:
                            result["warnings"].append(coord_check["message"])
                        if coord_check["actual_min"] is not None:
                            _safe_print(f"    纬度范围: [{coord_check['actual_min']:.4f}, {coord_check['actual_max']:.4f}]")

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
                                _safe_print(f"    裁剪 latitude: {original_shape} -> {data_arr.shape} (使用 h_slice)")
                            elif is_lon and w_slice is not None:
                                data_arr = data_arr[w_slice]
                                _safe_print(f"    裁剪 longitude: {original_shape} -> {data_arr.shape} (使用 w_slice)")
                            else:
                                _safe_print(f"    1D 变量 '{var}'，不裁剪")
                        else:
                            data_arr = crop_spatial(data_arr, h_slice, w_slice)
                            _safe_print(f"    裁剪: {original_shape} -> {data_arr.shape}")

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

                    _safe_print(f"    完成，shape={data_arr.shape}")

                except Exception as e:
                    result["warnings"].append(f"保存静态变量 '{var}' 失败: {str(e)}")

    except Exception as e:
        result["errors"].append(f"读取静态文件失败: {str(e)}")

    result["coord_checks"] = coord_checks
    return saved_files
