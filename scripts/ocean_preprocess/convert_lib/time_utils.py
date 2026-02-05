"""
time_utils.py - 时间处理工具函数

从 NC 文件提取时间戳，生成日期格式文件名

@author kongzhiquan
@date 2026-02-05
@version 1.0.0

功能:
- 从多个 NC 文件提取时间戳
- 自动检测合适的日期格式
- 生成无冲突的文件名
- 验证时间单调性
- 创建时间映射元数据

@changelog
  - 2026-02-05 kongzhiquan: v1.0.0 初始版本
"""

import os
import sys
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any
from collections import Counter

from .constants import TIME_COORD_CANDIDATES

__all__ = [
    'extract_timestamps_from_files',
    'detect_date_format',
    'generate_date_filenames',
    'validate_time_monotonic',
    'create_time_mapping',
    'DATE_FORMATS',
]

# 日期格式映射
DATE_FORMATS = {
    "YYYYMMDD": "%Y%m%d",
    "YYYYMMDDHH": "%Y%m%d%H",
    "YYYYMMDDHHmm": "%Y%m%d%H%M",
    "YYYY-MM-DD": "%Y-%m-%d",
    "YYYY-MM-DD_HH": "%Y-%m-%d_%H",
}


def _find_time_var(ds: xr.Dataset) -> Optional[str]:
    """
    在 Dataset 中查找时间变量

    Args:
        ds: xarray Dataset

    Returns:
        时间变量名，未找到返回 None
    """
    # 优先检查候选列表
    for c in TIME_COORD_CANDIDATES:
        if c in ds.coords or c in ds.variables:
            return c

    # 模糊匹配维度名
    for d in ds.dims:
        if "time" in d.lower():
            return d

    # 检查变量属性中是否有 axis='T'
    for var_name in ds.variables:
        attrs = ds[var_name].attrs
        if attrs.get('axis', '').upper() == 'T':
            return var_name

    return None


def _np_datetime64_to_datetime(dt64: np.datetime64) -> datetime:
    """
    numpy datetime64 转 Python datetime

    Args:
        dt64: numpy datetime64 值

    Returns:
        Python datetime 对象
    """
    # 处理 NaT
    if np.isnat(dt64):
        raise ValueError("遇到 NaT (Not a Time) 值")

    # 转换为纳秒时间戳再转换
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00', 'ns')) / np.timedelta64(1, 's')
    return datetime.utcfromtimestamp(float(ts))


def _extract_timestamps_with_cftime(
    nc_file: str,
    time_var: str
) -> Tuple[List[datetime], List[str]]:
    """
    使用 cftime 手动解析时间（当 xarray decode_times 失败时）

    Args:
        nc_file: NC 文件路径
        time_var: 时间变量名

    Returns:
        (timestamps, warnings)
    """
    warnings = []
    timestamps = []

    with xr.open_dataset(nc_file, decode_times=False) as ds:
        if time_var not in ds.variables and time_var not in ds.coords:
            return [], [f"时间变量 '{time_var}' 不存在"]

        time_values = ds[time_var].values
        units = ds[time_var].attrs.get('units', '')
        calendar = ds[time_var].attrs.get('calendar', 'standard')

        if not units:
            return [], [f"时间变量 '{time_var}' 缺少 units 属性"]

        # 尝试使用 cftime
        try:
            import cftime
            dates = cftime.num2date(time_values, units, calendar)
            for d in dates:
                # cftime 日期转 Python datetime
                try:
                    timestamps.append(datetime(
                        d.year, d.month, d.day,
                        d.hour, d.minute, d.second
                    ))
                except (ValueError, AttributeError) as e:
                    warnings.append(f"日期转换失败: {d} - {e}")
                    return [], warnings
        except ImportError:
            warnings.append("cftime 未安装，无法解析特殊日历格式")
            return [], warnings
        except Exception as e:
            warnings.append(f"cftime 解析失败: {e}")
            return [], warnings

    return timestamps, warnings


def extract_timestamps_from_files(
    nc_files: List[str],
    time_var: Optional[str] = None
) -> Tuple[Optional[List[datetime]], Optional[str], List[str]]:
    """
    从多个 NC 文件提取时间戳列表

    按文件顺序提取所有时间步的时间戳，合并为一个列表。

    Args:
        nc_files: NC 文件路径列表（应已排序）
        time_var: 指定的时间变量名（None 则自动检测）

    Returns:
        (timestamps, detected_time_var, warnings)
        - timestamps: datetime 列表，失败时返回 None
        - detected_time_var: 检测到的时间变量名
        - warnings: 警告信息列表
    """
    warnings_list = []
    all_timestamps = []
    detected_var = None

    if not nc_files:
        return None, None, ["NC 文件列表为空"]

    for i, nc_file in enumerate(nc_files):
        file_basename = os.path.basename(nc_file)

        try:
            # 先尝试 decode_times=True（xarray 自动解析）
            with xr.open_dataset(nc_file, decode_times=True) as ds:
                # 查找时间变量
                tv = time_var if time_var else _find_time_var(ds)

                if tv is None:
                    warnings_list.append(f"文件 '{file_basename}' 未找到时间变量")
                    return None, None, warnings_list

                if detected_var is None:
                    detected_var = tv
                elif detected_var != tv and time_var is None:
                    warnings_list.append(
                        f"文件 '{file_basename}' 时间变量不一致: "
                        f"预期 '{detected_var}', 实际 '{tv}'"
                    )

                # 获取时间值
                if tv not in ds.coords and tv not in ds.variables:
                    warnings_list.append(f"文件 '{file_basename}' 中不存在时间变量 '{tv}'")
                    return None, detected_var, warnings_list

                time_values = ds[tv].values

                # 转换为 datetime
                for t_idx, t in enumerate(time_values):
                    try:
                        if isinstance(t, np.datetime64):
                            dt = _np_datetime64_to_datetime(t)
                            all_timestamps.append(dt)
                        elif hasattr(t, 'year'):  # cftime 日期对象
                            dt = datetime(t.year, t.month, t.day,
                                         t.hour, t.minute, t.second)
                            all_timestamps.append(dt)
                        else:
                            warnings_list.append(
                                f"文件 '{file_basename}' 时间步 {t_idx}: "
                                f"无法解析时间值类型 {type(t)}"
                            )
                            return None, detected_var, warnings_list
                    except Exception as e:
                        warnings_list.append(
                            f"文件 '{file_basename}' 时间步 {t_idx}: 解析失败 - {e}"
                        )
                        return None, detected_var, warnings_list

        except Exception as e:
            # decode_times=True 失败，尝试手动解析
            print(f"  [Info] 文件 '{file_basename}' xarray 自动解析失败，尝试 cftime...",
                  file=sys.stderr)

            # 确定时间变量名
            tv = time_var if time_var else detected_var
            if tv is None:
                # 需要先检测时间变量名
                try:
                    with xr.open_dataset(nc_file, decode_times=False) as ds:
                        tv = _find_time_var(ds)
                        if tv is None:
                            warnings_list.append(f"文件 '{file_basename}' 未找到时间变量")
                            return None, None, warnings_list
                        detected_var = tv
                except Exception as e2:
                    warnings_list.append(f"文件 '{file_basename}' 打开失败: {e2}")
                    return None, None, warnings_list

            # 使用 cftime 解析
            ts, warns = _extract_timestamps_with_cftime(nc_file, tv)
            warnings_list.extend(warns)

            if not ts:
                warnings_list.append(f"文件 '{file_basename}' 时间解析完全失败")
                return None, detected_var, warnings_list

            all_timestamps.extend(ts)

    if not all_timestamps:
        warnings_list.append("未能提取任何时间戳")
        return None, detected_var, warnings_list

    return all_timestamps, detected_var, warnings_list


def detect_date_format(timestamps: List[datetime]) -> str:
    """
    根据时间间隔自动检测合适的日期格式

    规则：
    - 最小间隔 >= 1天 → "YYYYMMDD"
    - 最小间隔 >= 1小时 → "YYYYMMDDHH"
    - 其他 → "YYYYMMDDHHmm"

    Args:
        timestamps: datetime 列表

    Returns:
        推荐的日期格式字符串
    """
    if len(timestamps) < 2:
        return "YYYYMMDD"

    # 计算所有相邻时间间隔
    intervals = []
    for i in range(1, len(timestamps)):
        delta = timestamps[i] - timestamps[i - 1]
        intervals.append(delta)

    # 获取最小间隔
    min_interval = min(intervals)

    # 根据最小间隔选择格式
    if min_interval >= timedelta(days=1):
        return "YYYYMMDD"
    elif min_interval >= timedelta(hours=1):
        return "YYYYMMDDHH"
    else:
        return "YYYYMMDDHHmm"


def generate_date_filenames(
    timestamps: List[datetime],
    date_format: str = "auto"
) -> List[str]:
    """
    生成日期文件名，自动处理重复日期

    如果同一日期格式下有重复，会自动添加更精细的时间后缀。

    Args:
        timestamps: datetime 列表
        date_format: 日期格式，"auto" 则自动检测

    Returns:
        文件名列表（不含 .npy 扩展名）
    """
    if not timestamps:
        return []

    # 自动检测格式
    if date_format == "auto":
        date_format = detect_date_format(timestamps)

    fmt = DATE_FORMATS.get(date_format, "%Y%m%d")

    # 先生成基础日期字符串
    base_names = [ts.strftime(fmt) for ts in timestamps]

    # 检测重复
    counter = Counter(base_names)
    duplicates = {k for k, v in counter.items() if v > 1}

    if not duplicates:
        # 无重复，直接返回
        return base_names

    # 有重复，需要添加更精细的时间后缀
    filenames = []

    for i, (name, ts) in enumerate(zip(base_names, timestamps)):
        if name in duplicates:
            # 根据当前格式决定添加什么后缀
            if date_format == "YYYYMMDD":
                # 添加小时分钟
                suffix = ts.strftime("_%H%M")
            elif date_format == "YYYYMMDDHH":
                # 添加分钟秒
                suffix = ts.strftime("%M")
            else:
                # 已经很精细了，添加序号
                # 计算该名称之前出现的次数
                count = sum(1 for j in range(i) if base_names[j] == name)
                suffix = f"_{count:02d}"

            filenames.append(name + suffix)
        else:
            filenames.append(name)

    # 最终检查是否还有重复
    final_counter = Counter(filenames)
    final_duplicates = {k for k, v in final_counter.items() if v > 1}

    if final_duplicates:
        # 仍有重复，添加全局序号
        seen_count = {}
        final_filenames = []
        for fn in filenames:
            if fn in final_duplicates:
                idx = seen_count.get(fn, 0)
                final_filenames.append(f"{fn}_{idx:02d}")
                seen_count[fn] = idx + 1
            else:
                final_filenames.append(fn)
        return final_filenames

    return filenames


def validate_time_monotonic(
    timestamps: List[datetime]
) -> Tuple[bool, str]:
    """
    验证时间是否单调递增

    Args:
        timestamps: datetime 列表

    Returns:
        (is_valid, message)
        - is_valid: 是否单调递增
        - message: 描述信息
    """
    if len(timestamps) < 2:
        return True, "时间步数少于 2，无需验证"

    for i in range(1, len(timestamps)):
        if timestamps[i] <= timestamps[i - 1]:
            return False, (
                f"时间步 {i-1} ({timestamps[i-1]}) >= "
                f"时间步 {i} ({timestamps[i]})"
            )

    return True, "时间单调递增"


def create_time_mapping(
    timestamps: List[datetime],
    filenames: List[str],
    split_info: Dict[str, Dict[str, Any]],
    date_format: str
) -> Dict[str, Any]:
    """
    创建时间映射元数据

    用于保存到 manifest 文件，方便后续追溯文件名与时间的对应关系。

    Args:
        timestamps: datetime 列表
        filenames: 文件名列表
        split_info: 数据集划分信息
        date_format: 使用的日期格式

    Returns:
        时间映射字典
    """
    mapping = {
        "use_date_filename": True,
        "date_format": date_format,
        "total_timestamps": len(timestamps),
        "time_range": {
            "start": timestamps[0].isoformat() if timestamps else None,
            "end": timestamps[-1].isoformat() if timestamps else None,
        },
        "splits": {}
    }

    # 为每个 split 记录时间范围和文件名
    for var_name, var_split_info in split_info.items():
        mapping["splits"][var_name] = {}

        for split_name in ['train', 'valid', 'test']:
            info = var_split_info.get(split_name, {})
            start_idx = info.get('start', 0)
            end_idx = info.get('end', 0)

            if end_idx > start_idx and end_idx <= len(timestamps):
                mapping["splits"][var_name][split_name] = {
                    "start_time": timestamps[start_idx].isoformat(),
                    "end_time": timestamps[end_idx - 1].isoformat(),
                    "count": end_idx - start_idx,
                    "first_file": filenames[start_idx] if start_idx < len(filenames) else None,
                    "last_file": filenames[end_idx - 1] if end_idx - 1 < len(filenames) else None,
                }

    return mapping
