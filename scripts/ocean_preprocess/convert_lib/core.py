"""
core.py - 核心转换函数

从 convert_npy.py 拆分

@author leizheng
@contributors kongzhiquan
@date 2026-02-05
@version 3.1.1

@changelog
  - 2026-02-05 kongzhiquan: v3.1.1 从 convert_npy.py 拆分为独立模块
"""

import os
import sys
import glob
import traceback

import numpy as np
import xarray as xr
from typing import Any, Dict, List, Optional, Tuple


def _safe_print(msg: str) -> None:
    """向 stderr 输出日志，忽略 BrokenPipeError（父进程 pipe 关闭时）"""
    try:
        print(msg, file=sys.stderr, flush=True)
    except BrokenPipeError:
        pass
    except OSError:
        pass


from .constants import (
    MASK_VARS_DEFAULT,
    LAND_THRESHOLD_ABS,
    HEURISTIC_SAMPLE_SIZE,
    DEFAULT_WORKERS
)
from .crop import (
    parse_slice_str,
    load_coordinate_arrays,
    compute_region_crop_indices,
    adjust_crop_for_scale
)
from .converters import convert_dynamic_vars, convert_static_vars
from .validation import validate_rule1, validate_rule2, validate_rule3


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

    # ========== 日期文件名参数 ==========
    use_date_filename = config.get("use_date_filename", False)
    date_format = config.get("date_format", "auto")
    time_var = config.get("time_var")  # 指定的时间变量名

    # ========== 区域裁剪参数 ==========
    enable_region_crop = config.get("enable_region_crop", False)
    crop_lon_range = config.get("crop_lon_range")  # [min, max]
    crop_lat_range = config.get("crop_lat_range")  # [min, max]
    crop_mode = config.get("crop_mode", "two_step")  # "one_step" | "two_step"

    # 转换为元组
    if crop_lon_range and isinstance(crop_lon_range, list):
        crop_lon_range = tuple(crop_lon_range)
    if crop_lat_range and isinstance(crop_lat_range, list):
        crop_lat_range = tuple(crop_lat_range)

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
            "output_subdir": output_subdir,
            "enable_region_crop": enable_region_crop,
            "crop_lon_range": list(crop_lon_range) if crop_lon_range else None,
            "crop_lat_range": list(crop_lat_range) if crop_lat_range else None,
            "crop_mode": crop_mode
        }
    }

    try:
        # 1. 获取 NC 文件列表
        # 优先使用 config 中显式指定的 nc_files 列表（由 full.ts 传递），
        # 否则回退到 glob 搜索目录
        explicit_nc_files = config.get("nc_files")
        if explicit_nc_files and isinstance(explicit_nc_files, list) and len(explicit_nc_files) > 0:
            # 显式文件列表：如果是相对路径，加上 nc_folder 前缀
            nc_files = []
            for f in explicit_nc_files:
                if os.path.isabs(f):
                    nc_files.append(f)
                else:
                    nc_files.append(os.path.join(nc_folder, f))
            nc_files = sorted(nc_files)
            _safe_print(f"使用显式文件列表: {len(nc_files)} 个 NC 文件")
        else:
            search_path = os.path.join(nc_folder, dyn_file_pattern)
            nc_files = sorted(glob.glob(search_path))
            _safe_print(f"通过 glob 发现 {len(nc_files)} 个 NC 文件")
        if not nc_files:
            result["errors"].append(f"未找到匹配的 NC 文件: {search_path}")
            result["status"] = "error"
            return result

        _safe_print(f"找到 {len(nc_files)} 个 NC 文件")
        result["nc_file_count"] = len(nc_files)

        # 2. 创建输出目录
        os.makedirs(output_base, exist_ok=True)

        # ========== 2.5 区域裁剪计算 ==========
        raw_h_slice = None
        raw_w_slice = None
        hr_h_slice = h_slice
        hr_w_slice = w_slice
        region_crop_info = None

        if enable_region_crop and crop_lon_range and crop_lat_range:
            _safe_print(f"\n--- 区域裁剪: 模式={crop_mode} ---")
            _safe_print(f"    经度范围: {crop_lon_range}")
            _safe_print(f"    纬度范围: {crop_lat_range}")

            try:
                # 加载坐标数组
                lon_arr, lat_arr = load_coordinate_arrays(
                    static_file, nc_files, lon_var, lat_var
                )
                _safe_print(f"    坐标形状: lon={lon_arr.shape}, lat={lat_arr.shape}")

                # 计算区域裁剪索引
                h_start, h_end, w_start, w_end = compute_region_crop_indices(
                    lon_arr, lat_arr, crop_lon_range, crop_lat_range
                )
                _safe_print(f"    区域裁剪索引: H=[{h_start}:{h_end}], W=[{w_start}:{w_end}]")

                region_h = h_end - h_start
                region_w = w_end - w_start
                _safe_print(f"    区域尺寸: {region_h} x {region_w}")

                if crop_mode == "two_step":
                    # 两步裁剪模式：先保存 raw，再做尺寸调整保存 hr
                    raw_h_slice = slice(h_start, h_end)
                    raw_w_slice = slice(w_start, w_end)

                    # 如果用户提供了额外的 h_slice/w_slice，是相对于 raw 的再次裁剪
                    # 否则，根据 scale 自动调整
                    if scale and scale > 1:
                        # 计算能被 scale 整除的裁剪范围（相对于 raw）
                        final_h = (region_h // scale) * scale
                        final_w = (region_w // scale) * scale
                        trim_h = region_h - final_h
                        trim_w = region_w - final_w
                        hr_h_slice = slice(trim_h // 2, region_h - (trim_h - trim_h // 2))
                        hr_w_slice = slice(trim_w // 2, region_w - (trim_w - trim_w // 2))
                        _safe_print(f"    HR 裁剪（相对于 raw）: H=[{hr_h_slice.start}:{hr_h_slice.stop}], W=[{hr_w_slice.start}:{hr_w_slice.stop}]")
                        _safe_print(f"    HR 尺寸: {final_h} x {final_w}")

                    region_crop_info = {
                        "mode": "two_step",
                        "raw_slice": {"h": f"{h_start}:{h_end}", "w": f"{w_start}:{w_end}"},
                        "raw_size": {"h": region_h, "w": region_w},
                        "hr_slice_relative": {"h": f"{hr_h_slice.start if hr_h_slice else 0}:{hr_h_slice.stop if hr_h_slice else region_h}",
                                              "w": f"{hr_w_slice.start if hr_w_slice else 0}:{hr_w_slice.stop if hr_w_slice else region_w}"}
                    }

                else:  # one_step 模式
                    # 一步到位：直接计算能被 scale 整除的裁剪范围
                    if scale and scale > 1:
                        # 获取原始数据尺寸
                        with xr.open_dataset(nc_files[0], decode_times=False) as ds:
                            sample_var = dyn_vars[0] if dyn_vars else list(ds.data_vars)[0]
                            sample_shape = ds[sample_var].shape
                            max_h = sample_shape[-2]
                            max_w = sample_shape[-1]

                        h_start, h_end, w_start, w_end = adjust_crop_for_scale(
                            h_start, h_end, w_start, w_end, scale, max_h, max_w
                        )
                        _safe_print(f"    一步到位调整后: H=[{h_start}:{h_end}], W=[{w_start}:{w_end}]")

                    hr_h_slice = slice(h_start, h_end)
                    hr_w_slice = slice(w_start, w_end)
                    final_h = h_end - h_start
                    final_w = w_end - w_start
                    _safe_print(f"    HR 尺寸: {final_h} x {final_w}")

                    region_crop_info = {
                        "mode": "one_step",
                        "hr_slice": {"h": f"{h_start}:{h_end}", "w": f"{w_start}:{w_end}"},
                        "hr_size": {"h": final_h, "w": final_w}
                    }

                result["region_crop_info"] = region_crop_info

            except Exception as e:
                result["errors"].append(f"区域裁剪计算失败: {str(e)}")
                result["status"] = "error"
                import traceback
                _safe_print(f"区域裁剪错误: {traceback.format_exc()}")
                return result

        # ========== 3. 两步裁剪模式：先保存 raw ==========
        if enable_region_crop and crop_mode == "two_step" and raw_h_slice is not None:
            _safe_print(f"\n--- Step 3a: 保存区域裁剪后的 raw 数据 ---")

            # 保存动态变量到 raw/
            raw_dyn_saved = convert_dynamic_vars(
                nc_files, dyn_vars, output_base, result,
                allow_nan=allow_nan,
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
                test_ratio=test_ratio,
                h_slice=raw_h_slice,
                w_slice=raw_w_slice,
                scale=None,  # raw 不验证 scale
                workers=workers,
                output_subdir='raw',
                use_date_filename=use_date_filename,
                date_format=date_format,
                time_var=time_var
            )
            result["saved_files"]["raw_dynamic"] = raw_dyn_saved

            # 保存静态变量到 static_variables/raw/
            if stat_vars:
                raw_sta_saved = convert_static_vars(
                    static_file, stat_vars, mask_vars, output_base, result,
                    allow_nan=allow_nan,
                    lon_range=lon_range,
                    lat_range=lat_range,
                    h_slice=raw_h_slice,
                    w_slice=raw_w_slice,
                    fallback_nc_files=nc_files,
                    output_subdir='raw'  # 保存到 static_variables/raw/
                )
                result["saved_files"]["raw_static"] = raw_sta_saved

        # 3. 转换动态变量到 hr/（按时间顺序划分）
        # 对于两步裁剪，需要计算绝对的 hr slice
        final_hr_h_slice = hr_h_slice
        final_hr_w_slice = hr_w_slice

        if enable_region_crop and crop_mode == "two_step" and raw_h_slice is not None:
            # 两步裁剪：hr slice 是相对于 raw 的，需要转换为绝对位置
            if hr_h_slice is not None:
                abs_h_start = raw_h_slice.start + (hr_h_slice.start if hr_h_slice.start else 0)
                abs_h_end = raw_h_slice.start + (hr_h_slice.stop if hr_h_slice.stop else (raw_h_slice.stop - raw_h_slice.start))
                final_hr_h_slice = slice(abs_h_start, abs_h_end)
            else:
                final_hr_h_slice = raw_h_slice

            if hr_w_slice is not None:
                abs_w_start = raw_w_slice.start + (hr_w_slice.start if hr_w_slice.start else 0)
                abs_w_end = raw_w_slice.start + (hr_w_slice.stop if hr_w_slice.stop else (raw_w_slice.stop - raw_w_slice.start))
                final_hr_w_slice = slice(abs_w_start, abs_w_end)
            else:
                final_hr_w_slice = raw_w_slice

            _safe_print(f"    HR 绝对裁剪: H=[{final_hr_h_slice.start}:{final_hr_h_slice.stop}], W=[{final_hr_w_slice.start}:{final_hr_w_slice.stop}]")

        _safe_print(f"\n--- Step 3b: 保存 HR 数据 ---")
        dyn_saved = convert_dynamic_vars(
            nc_files, dyn_vars, output_base, result,
            allow_nan=allow_nan,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            h_slice=final_hr_h_slice,
            w_slice=final_hr_w_slice,
            scale=scale,
            workers=workers,
            output_subdir=output_subdir,
            use_date_filename=use_date_filename,
            date_format=date_format,
            time_var=time_var
        )
        result["saved_files"].update(dyn_saved)

        # 4. 转换静态变量到 hr/
        if stat_vars:
            _safe_print(f"\n--- Step 4: 保存静态变量到 HR ---")
            sta_saved = convert_static_vars(
                static_file, stat_vars, mask_vars, output_base, result,
                allow_nan=allow_nan,
                lon_range=lon_range,
                lat_range=lat_range,
                h_slice=final_hr_h_slice if enable_region_crop else h_slice,
                w_slice=final_hr_w_slice if enable_region_crop else w_slice,
                fallback_nc_files=nc_files,  # 如果没有静态文件，从动态文件中提取
                output_subdir='hr'  # 保存到 static_variables/hr/
            )
            result["saved_files"].update(sta_saved)

        # 5. 后置验证
        if run_validation:
            _safe_print("\n--- 执行后置验证 (Rule 1/2/3) ---")

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

        _safe_print(f"\n{result['message']}")

    except Exception as e:
        result["status"] = "error"
        result["errors"].append(str(e))
        import traceback
        result["traceback"] = traceback.format_exc()

    return result
