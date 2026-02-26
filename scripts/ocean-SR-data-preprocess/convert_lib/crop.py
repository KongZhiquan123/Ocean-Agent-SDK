"""
crop.py - 裁剪相关函数

从 convert_npy.py 拆分

@changelog
  - 2026-02-07 Leizheng: 修复 BrokenPipeError — stderr 写入加保护
  - 2026-02-07 Leizheng: 新增流式处理函数 _stream_extract_save_worker / parallel_stream_extract_save
  - 2026-02-06 Leizheng: 添加并行提取进度日志与耗时统计
"""

import os
import sys
import time
import signal
import numpy as np
import xarray as xr
from typing import Any, Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial


def _safe_print(msg: str) -> None:
    """向 stderr 输出日志，忽略 BrokenPipeError（父进程 pipe 关闭时）"""
    try:
        print(msg, file=sys.stderr, flush=True)
    except BrokenPipeError:
        pass
    except OSError:
        pass


def _extract_var_from_file(nc_file: str, var_name: str, h_slice: Optional[slice] = None, w_slice: Optional[slice] = None) -> Dict[str, Any]:
    """
    从单个 NC 文件中提取变量数据（worker 函数）

    每个进程独立打开文件，避免 HDF5/netCDF4 并发问题
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
    """
    errors = []
    total = len(nc_files)

    # 限制 workers 数量
    actual_workers = min(workers, total, cpu_count())

    # 创建 partial 函数固定参数
    extract_func = partial(_extract_var_from_file, var_name=var_name, h_slice=h_slice, w_slice=w_slice)

    print(f"    并行提取 '{var_name}'，进程数: {actual_workers}，文件数: {total}", file=sys.stderr)

    t_start = time.time()

    if actual_workers > 1:
        # 多进程并行，使用 imap_unordered 显示进度
        results_ordered = [None] * total
        file_index_map = {f: i for i, f in enumerate(nc_files)}
        done_count = 0

        with Pool(processes=actual_workers) as pool:
            for r in pool.imap_unordered(extract_func, nc_files):
                done_count += 1
                idx = file_index_map[r["file"]]
                results_ordered[idx] = r
                if done_count % 100 == 0 or done_count == total:
                    elapsed = time.time() - t_start
                    rate = done_count / elapsed if elapsed > 0 else 0
                    eta = (total - done_count) / rate if rate > 0 else 0
                    print(f"      读取进度: {done_count}/{total} ({done_count*100//total}%) | {elapsed:.1f}s 已用 | {rate:.1f} files/s | ETA {eta:.0f}s", file=sys.stderr)

        results = results_ordered
    else:
        # 单进程顺序执行
        results = []
        for i, f in enumerate(nc_files):
            results.append(extract_func(f))
            if (i + 1) % 100 == 0 or (i + 1) == total:
                elapsed = time.time() - t_start
                print(f"      读取进度: {i+1}/{total} ({(i+1)*100//total}%) | {elapsed:.1f}s", file=sys.stderr)

    t_read = time.time()
    print(f"      读取完成: {t_read - t_start:.1f}s", file=sys.stderr)

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
        mem_est = sum(d.nbytes for d in data_list) / (1024**3)
        print(f"      合并 {len(data_list)} 个数组 (预计 {mem_est:.2f} GB)...", file=sys.stderr)
        t_merge = time.time()
        combined = np.concatenate(data_list, axis=0)
        t_done = time.time()
        print(f"      合并完成: {t_done - t_merge:.1f}s, shape={combined.shape}, {combined.nbytes/(1024**3):.2f} GB", file=sys.stderr)
        return combined, errors
    except Exception as e:
        errors.append(f"合并数据失败: {str(e)}")
        return None, errors


def parse_slice_str(slice_str: Optional[str]) -> Optional[slice]:
    """
    解析切片字符串为 slice 对象

    支持格式:
    - "0:680"   -> slice(0, 680)
    - ":680"    -> slice(None, 680)
    - "1:"      -> slice(1, None)
    - "1:-1"    -> slice(1, -1)
    - None      -> None (不裁剪)
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
    """
    if h_slice is None and w_slice is None:
        return arr

    h_sl = h_slice if h_slice else slice(None)
    w_sl = w_slice if w_slice else slice(None)

    ndim = arr.ndim

    if ndim == 1:
        # 1D 数组（如 latitude 或 longitude）
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


def compute_region_crop_indices(
    lon_arr: np.ndarray,
    lat_arr: np.ndarray,
    crop_lon_range: Tuple[float, float],
    crop_lat_range: Tuple[float, float]
) -> Tuple[int, int, int, int]:
    """
    根据经纬度范围计算裁剪索引
    """
    lon_min, lon_max = crop_lon_range
    lat_min, lat_max = crop_lat_range

    if lon_arr.ndim == 1 and lat_arr.ndim == 1:
        # 1D 坐标（规则网格）
        lon_mask = (lon_arr >= lon_min) & (lon_arr <= lon_max)
        lat_mask = (lat_arr >= lat_min) & (lat_arr <= lat_max)

        lon_indices = np.where(lon_mask)[0]
        lat_indices = np.where(lat_mask)[0]

        if len(lon_indices) == 0:
            raise ValueError(f"经度范围 [{lon_min}, {lon_max}] 内没有数据点。数据经度范围: [{lon_arr.min():.4f}, {lon_arr.max():.4f}]")
        if len(lat_indices) == 0:
            raise ValueError(f"纬度范围 [{lat_min}, {lat_max}] 内没有数据点。数据纬度范围: [{lat_arr.min():.4f}, {lat_arr.max():.4f}]")

        w_start, w_end = lon_indices[0], lon_indices[-1] + 1
        h_start, h_end = lat_indices[0], lat_indices[-1] + 1

    elif lon_arr.ndim == 2 and lat_arr.ndim == 2:
        # 2D 坐标（曲线网格，如 ROMS）
        combined_mask = (
            (lon_arr >= lon_min) & (lon_arr <= lon_max) &
            (lat_arr >= lat_min) & (lat_arr <= lat_max)
        )

        if not combined_mask.any():
            raise ValueError(
                f"指定的经纬度范围内没有数据点。\n"
                f"经度范围: [{lon_min}, {lon_max}]，数据范围: [{lon_arr.min():.4f}, {lon_arr.max():.4f}]\n"
                f"纬度范围: [{lat_min}, {lat_max}]，数据范围: [{lat_arr.min():.4f}, {lat_arr.max():.4f}]"
            )

        rows, cols = np.where(combined_mask)
        h_start, h_end = rows.min(), rows.max() + 1
        w_start, w_end = cols.min(), cols.max() + 1

    else:
        raise ValueError(f"不支持的坐标维度组合: lon={lon_arr.ndim}D, lat={lat_arr.ndim}D")

    return h_start, h_end, w_start, w_end


def adjust_crop_for_scale(
    h_start: int, h_end: int,
    w_start: int, w_end: int,
    scale: int,
    max_h: int, max_w: int
) -> Tuple[int, int, int, int]:
    """
    调整裁剪范围使其能被 scale 整除
    """
    crop_h = h_end - h_start
    crop_w = w_end - w_start

    # 计算能被 scale 整除的最大尺寸
    final_h = (crop_h // scale) * scale
    final_w = (crop_w // scale) * scale

    # 计算需要裁掉的像素数
    trim_h = crop_h - final_h
    trim_w = crop_w - final_w

    # 尽量从两端均匀裁剪
    trim_h_start = trim_h // 2
    trim_h_end = trim_h - trim_h_start
    trim_w_start = trim_w // 2
    trim_w_end = trim_w - trim_w_start

    new_h_start = h_start + trim_h_start
    new_h_end = h_end - trim_h_end
    new_w_start = w_start + trim_w_start
    new_w_end = w_end - trim_w_end

    # 确保不越界
    new_h_start = max(0, new_h_start)
    new_h_end = min(max_h, new_h_end)
    new_w_start = max(0, new_w_start)
    new_w_end = min(max_w, new_w_end)

    return new_h_start, new_h_end, new_w_start, new_w_end


def load_coordinate_arrays(
    static_file: Optional[str],
    nc_files: List[str],
    lon_var: str,
    lat_var: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载经纬度坐标数组
    """
    source_file = static_file if static_file and os.path.exists(static_file) else nc_files[0]

    with xr.open_dataset(source_file, decode_times=False) as ds:
        if lon_var not in ds.variables and lon_var not in ds.coords:
            raise ValueError(f"经度变量 '{lon_var}' 不存在于文件中")
        if lat_var not in ds.variables and lat_var not in ds.coords:
            raise ValueError(f"纬度变量 '{lat_var}' 不存在于文件中")

        lon_arr = ds[lon_var].values
        lat_arr = ds[lat_var].values

    return lon_arr, lat_arr


# ---------------------------------------------------------------------------
# 流式处理函数（读一个、处理一个、写一个）
# ---------------------------------------------------------------------------


def _stream_extract_save_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    流式 worker：读取单个 NC 文件的一个变量 → 裁剪 → 直接保存 NPY → 返回元数据

    task 字段:
        nc_file:    str                          - NC 文件路径
        var_name:   str                          - 变量名
        output_map: List[Tuple[int, str]]        - [(local_t, output_path), ...]
        h_slice:    Optional[Tuple[int|None, int|None]]  - 裁剪参数 (start, stop)
        w_slice:    Optional[Tuple[int|None, int|None]]  - 裁剪参数 (start, stop)

    返回:
        { status, file, var_name, per_file_T, shape_per_step, dtype,
          nan_count, inf_count, total_elements, saved_count }
        注意: 不返回 data，数据直接写入磁盘
    """
    nc_file = task["nc_file"]
    var_name = task["var_name"]
    output_map = task["output_map"]  # [(local_t, output_path), ...]
    h_slice_raw = task.get("h_slice")  # Tuple or None
    w_slice_raw = task.get("w_slice")  # Tuple or None

    # 重建 slice 对象（multiprocessing 不能 pickle slice）
    h_sl = slice(*h_slice_raw) if h_slice_raw is not None else None
    w_sl = slice(*w_slice_raw) if w_slice_raw is not None else None

    try:
        with xr.open_dataset(nc_file, decode_times=False) as ds:
            if var_name not in ds.data_vars and var_name not in ds.coords:
                return {
                    "status": "error",
                    "error": f"变量 '{var_name}' 不存在",
                    "file": nc_file,
                    "var_name": var_name,
                }

            data = ds[var_name].values

            # 裁剪空间维度
            if h_sl is not None or w_sl is not None:
                h_s = h_sl if h_sl else slice(None)
                w_s = w_sl if w_sl else slice(None)
                ndim = data.ndim
                if ndim == 2:
                    data = data[h_s, w_s]
                elif ndim == 3:
                    data = data[:, h_s, w_s]
                elif ndim == 4:
                    data = data[:, :, h_s, w_s]

        # NaN/Inf 全量检查（逐文件）
        nan_count = int(np.count_nonzero(np.isnan(data)))
        inf_count = int(np.count_nonzero(np.isinf(data)))
        total_elements = int(data.size)

        # 确定每文件时间步
        if data.ndim >= 3:
            per_file_T = data.shape[0]
        else:
            # 2D 静态形式，不应出现在动态变量流式处理中
            per_file_T = 1

        # 逐时间步保存
        saved_count = 0
        shape_per_step = None
        for local_t, out_path in output_map:
            # 确保目录存在
            out_dir = os.path.dirname(out_path)
            os.makedirs(out_dir, exist_ok=True)

            if per_file_T == 1 and data.ndim == 2:
                step_data = data
            else:
                step_data = data[local_t]

            np.save(out_path, step_data)
            saved_count += 1

            if shape_per_step is None:
                shape_per_step = list(step_data.shape)

        return {
            "status": "success",
            "file": nc_file,
            "var_name": var_name,
            "per_file_T": per_file_T,
            "shape_per_step": shape_per_step,
            "dtype": str(data.dtype),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "total_elements": total_elements,
            "saved_count": saved_count,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "file": nc_file,
            "var_name": var_name,
        }


def parallel_stream_extract_save(
    tasks: List[Dict[str, Any]],
    workers: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    并行流式处理：读取 → 裁剪 → 保存，一步到位

    Args:
        tasks:   任务列表，每个元素传给 _stream_extract_save_worker
        workers: 并行进程数

    Returns:
        (results_metadata, errors)
        - results_metadata: 成功结果的元数据列表
        - errors: 错误信息字符串列表
    """
    total = len(tasks)
    actual_workers = min(workers, total, cpu_count())
    results_metadata = []
    errors = []

    _safe_print(f"    流式处理，进程数: {actual_workers}，任务数: {total}")
    t_start = time.time()

    if actual_workers > 1:
        done_count = 0
        with Pool(processes=actual_workers) as pool:
            for r in pool.imap_unordered(_stream_extract_save_worker, tasks):
                done_count += 1
                if r["status"] == "success":
                    results_metadata.append(r)
                else:
                    errors.append(f"{r['file']}: {r.get('error', 'unknown error')}")

                if done_count % 100 == 0 or done_count == total:
                    elapsed = time.time() - t_start
                    rate = done_count / elapsed if elapsed > 0 else 0
                    eta = (total - done_count) / rate if rate > 0 else 0
                    _safe_print(
                        f"      流式进度: {done_count}/{total} ({done_count * 100 // total}%) "
                        f"| {elapsed:.1f}s 已用 | {rate:.1f} files/s | ETA {eta:.0f}s"
                    )
    else:
        for i, task in enumerate(tasks):
            r = _stream_extract_save_worker(task)
            if r["status"] == "success":
                results_metadata.append(r)
            else:
                errors.append(f"{r['file']}: {r.get('error', 'unknown error')}")

            if (i + 1) % 100 == 0 or (i + 1) == total:
                elapsed = time.time() - t_start
                _safe_print(
                    f"      流式进度: {i + 1}/{total} ({(i + 1) * 100 // total}%) | {elapsed:.1f}s"
                )

    elapsed = time.time() - t_start
    total_saved = sum(r.get("saved_count", 0) for r in results_metadata)
    _safe_print(
        f"    流式处理完成: {elapsed:.1f}s, 保存 {total_saved} 个 npy 文件, {len(errors)} 个错误"
    )

    return results_metadata, errors
