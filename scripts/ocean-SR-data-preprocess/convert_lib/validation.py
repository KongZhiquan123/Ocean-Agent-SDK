"""
validation.py - 后置验证规则

从 convert_npy.py 拆分
"""

import os
import sys
import json
import numpy as np
import xarray as xr
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _safe_print(msg: str) -> None:
    """向 stderr 输出日志，忽略 BrokenPipeError（父进程 pipe 关闭时）"""
    try:
        print(msg, file=sys.stderr, flush=True)
    except BrokenPipeError:
        pass
    except OSError:
        pass


from .constants import (
    LAND_THRESHOLD_ABS,
    HEURISTIC_SAMPLE_SIZE
)
from .check import get_spatial_shape, find_time_coord
from .mask import derive_mask, derive_staggered_mask, heuristic_mask_check


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
    """
    validation = {
        "rule": "Rule1",
        "description": "输出完整性与形状约定",
        "passed": True,
        "errors": [],
        "warnings": [],
        "details": {}
    }

    output_subdir = result.get('config', {}).get('output_subdir', 'hr')
    sta_out = os.path.join(output_dir, 'static_variables')

    # 1.1 检查目录结构存在
    splits = ['train', 'valid', 'test']
    missing_splits = []
    for split in splits:
        split_dir = os.path.join(output_dir, split, output_subdir)
        if not os.path.isdir(split_dir):
            missing_splits.append(f"{split}/{output_subdir}")

    if missing_splits:
        validation["errors"].append(f"目录不存在: {', '.join(missing_splits)}")
        validation["passed"] = False

    if not os.path.isdir(sta_out):
        validation["errors"].append(f"目录不存在: {sta_out}")
        validation["passed"] = False

    if not validation["passed"]:
        return validation

    # 1.2 检查预期文件完整
    for var in dyn_vars:
        for split in splits:
            var_dir = os.path.join(output_dir, split, output_subdir, var)
            if not os.path.isdir(var_dir):
                validation["errors"].append(f"缺少动态变量目录: {split}/{output_subdir}/{var}/")
                validation["passed"] = False
            else:
                npy_files = [f for f in os.listdir(var_dir) if f.endswith('.npy')]
                if not npy_files:
                    validation["errors"].append(f"动态变量目录为空: {split}/{output_subdir}/{var}/")
                    validation["passed"] = False

    sta_files = sorted(os.listdir(sta_out)) if os.path.isdir(sta_out) else []
    validation["details"]["static_files"] = sta_files

    for var in stat_vars:
        found = any(f.endswith(f"_{var}.npy") for f in sta_files)
        if not found:
            validation["warnings"].append(f"缺少静态变量文件: *_{var}.npy")

    # 1.3 检查网格形状
    lon_shape = None
    lat_shape = None

    if lon_var in saved_files:
        lon_shape = tuple(saved_files[lon_var]["shape"])
    if lat_var in saved_files:
        lat_shape = tuple(saved_files[lat_var]["shape"])

    validation["details"]["lon_shape"] = list(lon_shape) if lon_shape else None
    validation["details"]["lat_shape"] = list(lat_shape) if lat_shape else None

    is_1d_coords = (lon_shape and len(lon_shape) == 1) or (lat_shape and len(lat_shape) == 1)

    if is_1d_coords:
        grid_shape = None
        validation["details"]["coord_type"] = "1D"
    else:
        if lon_shape and lat_shape and lon_shape != lat_shape:
            validation["errors"].append(
                f"2D 网格坐标形状不一致: {lon_var} shape={lon_shape} != {lat_var} shape={lat_shape}"
            )
            validation["passed"] = False
        grid_shape = lon_shape or lat_shape
        validation["details"]["coord_type"] = "2D"
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
    if dyn_spatial_shapes:
        dyn_spatial = list(dyn_spatial_shapes.values())[0] if dyn_spatial_shapes else None

        if is_1d_coords and dyn_spatial:
            expected_h = dyn_spatial[0] if len(dyn_spatial) >= 1 else None
            expected_w = dyn_spatial[1] if len(dyn_spatial) >= 2 else None

            if lat_shape and expected_h and lat_shape[0] != expected_h:
                validation["errors"].append(
                    f"纬度坐标长度 {lat_shape[0]} 与动态变量 H 维度 {expected_h} 不匹配"
                )
                validation["passed"] = False

            if lon_shape and expected_w and lon_shape[0] != expected_w:
                validation["errors"].append(
                    f"经度坐标长度 {lon_shape[0]} 与动态变量 W 维度 {expected_w} 不匹配"
                )
                validation["passed"] = False

            if lat_shape and lon_shape and expected_h and expected_w:
                if lat_shape[0] == expected_h and lon_shape[0] == expected_w:
                    validation["details"]["coord_match"] = True
                    validation["details"]["expected_grid"] = f"H={expected_h}, W={expected_w}"
        elif grid_shape and dyn_spatial:
            if tuple(grid_shape) != tuple(dyn_spatial):
                for var, spatial in dyn_spatial_shapes.items():
                    if spatial != grid_shape:
                        validation["warnings"].append(
                            f"动态变量 '{var}' 空间形状 {spatial} 与网格形状 {grid_shape} 不匹配"
                        )

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

    mask_spatial_shapes = {}
    mask_arrays = {}

    for mask_var in mask_vars:
        if mask_var in saved_files:
            spatial = tuple(saved_files[mask_var].get("spatial_shape", saved_files[mask_var]["shape"]))
            mask_spatial_shapes[mask_var] = spatial
            validation["details"][f"{mask_var}_spatial_shape"] = list(spatial)
            try:
                mask_path = saved_files[mask_var]["path"]
                if os.path.exists(mask_path):
                    mask_arrays[mask_var] = np.load(mask_path)
                else:
                    validation["warnings"].append(f"掩码文件不存在: {mask_path}")
            except Exception as e:
                validation["warnings"].append(f"加载掩码 '{mask_var}' 失败: {str(e)}")

    # 自动派生 mask_u/mask_v
    derived_masks = {}
    if 'mask_rho' in mask_arrays:
        mask_rho = mask_arrays['mask_rho']
        rho_shape = mask_rho.shape

        if 'mask_u' not in mask_spatial_shapes:
            derived_u = derive_staggered_mask(mask_rho, 'u')
            derived_masks['mask_u'] = derived_u
            mask_spatial_shapes['mask_u'] = derived_u.shape
            mask_arrays['mask_u'] = derived_u
            validation["details"]["mask_u_derived"] = True
            validation["details"]["mask_u_spatial_shape"] = list(derived_u.shape)
            _safe_print(f"  从 mask_rho {rho_shape} 派生 mask_u {derived_u.shape}")

        if 'mask_v' not in mask_spatial_shapes:
            derived_v = derive_staggered_mask(mask_rho, 'v')
            derived_masks['mask_v'] = derived_v
            mask_spatial_shapes['mask_v'] = derived_v.shape
            mask_arrays['mask_v'] = derived_v
            validation["details"]["mask_v_derived"] = True
            validation["details"]["mask_v_spatial_shape"] = list(derived_v.shape)
            _safe_print(f"  从 mask_rho {rho_shape} 派生 mask_v {derived_v.shape}")

    def get_mask_for_var(var_name: str) -> Optional[str]:
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

        if "spatial_shape" in info:
            var_spatial = tuple(info["spatial_shape"])
        elif "shape" in info and len(info["shape"]) >= 2:
            var_spatial = tuple(info["shape"][-2:])
        else:
            continue

        expected_mask = get_mask_for_var(var)
        if expected_mask and expected_mask in mask_spatial_shapes:
            mask_spatial = mask_spatial_shapes[expected_mask]
            match = (var_spatial == mask_spatial)
            is_derived = expected_mask in derived_masks

            dyn_spatial_checks[var] = {
                "var_spatial": list(var_spatial),
                "expected_mask": expected_mask,
                "mask_spatial": list(mask_spatial),
                "mask_derived": is_derived,
                "match": match
            }

            if not match:
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

    # 2.2 精确对比
    if static_file and os.path.exists(static_file) and mask_src_var:
        try:
            with xr.open_dataset(static_file, decode_times=False) as ds:
                if mask_src_var in ds.variables:
                    src_mask = ds[mask_src_var].values
                    expected_mask = derive_mask(src_mask, mask_derive_op)

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

    # 2.5 启发式掩码验证
    if heuristic_check_var:
        _safe_print(f"  执行启发式掩码验证: {heuristic_check_var}")

        check_mask_name = get_mask_for_var(heuristic_check_var)

        if check_mask_name in mask_arrays:
            mask_arr = mask_arrays[check_mask_name]
            mask_source = "派生" if check_mask_name in derived_masks else "原始"
            _safe_print(f"    使用掩码: {check_mask_name} ({mask_source}), shape={mask_arr.shape}")

            if heuristic_check_var in saved_files:
                dyn_path = saved_files[heuristic_check_var]["path"]
                dyn_arr = np.load(dyn_path)

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

                _safe_print(f"    陆地零值比例: {heuristic_result.get('land_zero_ratio', 'N/A')}")
                _safe_print(f"    海洋零值比例: {heuristic_result.get('ocean_zero_ratio', 'N/A')}")
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
        _safe_print(f"已生成 manifest: {manifest_path}")
    except Exception as e:
        validation["warnings"].append(f"生成 manifest 失败: {str(e)}")

    # 3.3 检查时间单调性和间隔一致性
    if nc_files:
        try:
            time_values = []
            time_ranges = []
            time_coord = None

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

            if len(time_values) > 2:
                time_arr = np.array(time_values)
                diffs = np.diff(time_arr)

                median_diff = float(np.median(diffs))
                min_diff = float(np.min(diffs))
                max_diff = float(np.max(diffs))

                validation["details"]["time_interval"] = {
                    "median": median_diff,
                    "min": min_diff,
                    "max": max_diff
                }

                dup_count = int(np.sum(diffs <= 0))
                if dup_count > 0:
                    validation["warnings"].append(
                        f"检测到 {dup_count} 处时间重复或倒退（间隔 <= 0）"
                    )
                    validation["details"]["duplicate_count"] = dup_count

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
