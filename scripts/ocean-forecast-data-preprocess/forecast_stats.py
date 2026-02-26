#!/usr/bin/env python3
"""
@file forecast_stats.py

@description 海洋预报数据统计工具
             读取已预处理的 NPY 文件，计算 per-variable 基础统计量
             包括 NaN 率、值域、均值、标准差、P5/P95 分位数
             新增：跨 split 时间边界验证、跨 split 分布一致性检查

@author Leizheng
@date 2026-02-26
@version 1.1.0

@changelog
  - 2026-02-26 Leizheng: v1.1.0 新增数据质量检查
    - _check_time_boundaries(): 验证 train_end < valid_start < test_start
    - _check_cross_split_distribution(): 比较 train vs valid/test 均值偏移和方差比
    - run_stats() 集成以上两项检查，结果写入 time_boundary_check / cross_split_check
  - 2026-02-26 Leizheng: v1.0.0 初始版本
    - 从 {dataset_root}/var_names.json 自动读取变量列表
    - 支持 train/valid/test 多 split 统计
    - 全 NaN 数组安全处理（min/max/mean/std 输出 null）
    - NaN 率 > 0.3 自动生成警告

用法:
    python forecast_stats.py --config /path/to/config.json --output /path/to/result.json
"""

import argparse
import glob
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np


# ========================================
# 统计计算
# ========================================

def _compute_var_stats(
    npy_files: List[str],
    max_files: int = 200
) -> Dict[str, Any]:
    """
    计算单个变量的统计量。

    Args:
        npy_files: NPY 文件路径列表（已排序）。
        max_files: 最大采样文件数。

    Returns:
        统计字典，包含 nan_rate, min, max, mean, std, p5, p95, sample_files。
        当全为 NaN 时 min/max/mean/std/p5/p95 均为 null。
    """
    files = npy_files[:max_files]
    sample_files = len(files)

    if sample_files == 0:
        return {
            "nan_rate": 1.0,
            "min": None, "max": None,
            "mean": None, "std": None,
            "p5": None, "p95": None,
            "sample_files": 0
        }

    all_values: List[np.ndarray] = []
    total_elements = 0
    nan_elements = 0

    for fpath in files:
        try:
            arr = np.load(fpath).astype(np.float32)
        except Exception:
            continue

        total_elements += arr.size
        nan_mask = ~np.isfinite(arr)
        nan_elements += int(np.sum(nan_mask))

        finite_vals = arr[~nan_mask].ravel()
        if finite_vals.size > 0:
            all_values.append(finite_vals)

    # NaN 率
    if total_elements == 0:
        nan_rate = 1.0
    else:
        nan_rate = nan_elements / total_elements

    if not all_values:
        # 全 NaN：min/max/mean/std/p5/p95 均为 null
        return {
            "nan_rate": round(float(nan_rate), 6),
            "min": None, "max": None,
            "mean": None, "std": None,
            "p5": None, "p95": None,
            "sample_files": sample_files
        }

    combined = np.concatenate(all_values)

    def _safe_float(val: float) -> Optional[float]:
        if math.isfinite(val):
            return round(float(val), 6)
        return None

    return {
        "nan_rate": round(float(nan_rate), 6),
        "min": _safe_float(float(np.min(combined))),
        "max": _safe_float(float(np.max(combined))),
        "mean": _safe_float(float(np.mean(combined))),
        "std": _safe_float(float(np.std(combined))),
        "p5": _safe_float(float(np.percentile(combined, 5))),
        "p95": _safe_float(float(np.percentile(combined, 95))),
        "sample_files": sample_files
    }


# ========================================
# 时间边界检查
# ========================================

def _check_time_boundaries(dataset_root: str) -> Dict[str, Any]:
    """
    读取 time_index.json，验证跨 split 时间边界无重叠。

    检查逻辑：
      train.end_ts < valid.start_ts  →  train/valid 无泄露
      valid.end_ts < test.start_ts   →  valid/test 无泄露

    Returns:
        {
          "passed": bool,
          "splits": {"train": {"start": ..., "end": ..., "count": ...}, ...},
          "boundaries": {
            "train_end_before_valid_start": {"passed": bool, "train_end": ..., "valid_start": ...},
            ...
          },
          "errors": [...]
        }
    """
    result: Dict[str, Any] = {
        "passed": True,
        "splits": {},
        "boundaries": {},
        "errors": []
    }

    time_index_path = os.path.join(dataset_root, "time_index.json")
    if not os.path.exists(time_index_path):
        result["errors"].append("未找到 time_index.json，跳过时间边界检查（预处理版本过旧）")
        return result

    try:
        with open(time_index_path, "r", encoding="utf-8") as f:
            time_index = json.load(f)
    except Exception as e:
        result["errors"].append(f"读取 time_index.json 失败: {e}")
        return result

    splits_data = time_index.get("splits", {})
    for split_name, split_info in splits_data.items():
        result["splits"][split_name] = {
            "start": split_info.get("start_ts"),
            "end":   split_info.get("end_ts"),
            "count": split_info.get("count", 0)
        }

    # 按标准顺序逐对检查边界
    split_order = ["train", "valid", "test"]
    present = [s for s in split_order if s in result["splits"]]

    for i in range(len(present) - 1):
        prev, curr = present[i], present[i + 1]
        prev_end   = result["splits"][prev].get("end")
        curr_start = result["splits"][curr].get("start")
        key = f"{prev}_end_before_{curr}_start"

        if prev_end and curr_start:
            ok = prev_end < curr_start
            result["boundaries"][key] = {
                "passed":         ok,
                f"{prev}_end":    prev_end,
                f"{curr}_start":  curr_start
            }
            if not ok:
                result["passed"] = False
                result["errors"].append(
                    f"时间泄露风险: {prev} 结束 {prev_end} ≥ {curr} 开始 {curr_start}"
                )
        else:
            result["boundaries"][key] = {
                "passed": None,
                "note":   "时间戳缺失，无法验证"
            }

    return result


# ========================================
# 跨 split 分布一致性检查
# ========================================

def _check_cross_split_distribution(
    stats: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    对比 valid/test 与 train 的均值偏移（z-score）和方差比，发现分布漂移。

    告警阈值：
      mean_z_score > 1.0  : 均值偏移超过 1 个训练标准差
      std_ratio > 2.0     : 方差超过训练集的 2 倍
      std_ratio < 0.5     : 方差不足训练集的 1/2

    Args:
        stats: {"train": {"sst": {mean, std, ...}}, "valid": {...}, "test": {...}}

    Returns:
        {
          "warnings": [...],
          "variables": {
            "sst": {
              "valid": {"mean_diff": ..., "mean_z_score": ..., "std_ratio": ..., "flag": "ok|warn_mean|warn_std|warn_both"},
              "test":  {...}
            }
          }
        }
    """
    result: Dict[str, Any] = {
        "warnings": [],
        "variables": {}
    }

    train_stats = stats.get("train", {})
    if not train_stats:
        result["warnings"].append("无 train split 统计量，跳过分布一致性对比")
        return result

    compare_splits = [s for s in ["valid", "test"] if s in stats]
    if not compare_splits:
        result["warnings"].append("无 valid/test split，跳过分布一致性对比")
        return result

    for var_name, tr in train_stats.items():
        tr_mean = tr.get("mean")
        tr_std  = tr.get("std")
        if tr_mean is None or tr_std is None:
            continue  # 全 NaN 变量跳过

        var_result: Dict[str, Any] = {}

        for split in compare_splits:
            sp = stats[split].get(var_name, {})
            sp_mean = sp.get("mean")
            sp_std  = sp.get("std")
            if sp_mean is None or sp_std is None:
                continue

            mean_diff = sp_mean - tr_mean

            # 均值偏移 z-score（以 train_std 为基准）
            if tr_std > 1e-9:
                mean_z = abs(mean_diff) / tr_std
            else:
                mean_z = float("inf") if abs(mean_diff) > 1e-9 else 0.0

            # 方差比（test_std / train_std）
            if tr_std > 1e-9 and sp_std is not None and sp_std >= 0:
                std_ratio: Optional[float] = sp_std / tr_std
            else:
                std_ratio = None

            # 判断告警等级
            warn_mean = mean_z > 1.0
            warn_std  = std_ratio is not None and (std_ratio > 2.0 or std_ratio < 0.5)

            if warn_mean and warn_std:
                flag = "warn_both"
            elif warn_mean:
                flag = "warn_mean"
            elif warn_std:
                flag = "warn_std"
            else:
                flag = "ok"

            if warn_mean:
                result["warnings"].append(
                    f"{var_name}/{split} 均值偏移 {mean_z:.2f}σ "
                    f"(train={tr_mean:.3g}, {split}={sp_mean:.3g})"
                )
            if warn_std and std_ratio is not None:
                result["warnings"].append(
                    f"{var_name}/{split} 方差比 {std_ratio:.2f}x "
                    f"(train_std={tr_std:.3g}, {split}_std={sp_std:.3g})"
                )

            var_result[split] = {
                "mean_diff":    round(float(mean_diff), 6),
                "mean_z_score": round(float(mean_z), 4) if math.isfinite(mean_z) else None,
                "std_ratio":    round(float(std_ratio), 4) if std_ratio is not None else None,
                "flag":         flag
            }

        if var_result:
            result["variables"][var_name] = var_result

    return result


# ========================================
# 主逻辑
# ========================================

def run_stats(
    dataset_root: str,
    splits: List[str],
    max_files: int = 200
) -> dict:
    """
    计算数据集的 per-variable 统计量。

    Args:
        dataset_root: 数据集根目录。
        splits: 要统计的 split 列表。
        max_files: 每变量最大采样文件数。

    Returns:
        结果字典。
    """
    result: Dict[str, Any] = {
        "status": "pass",
        "dataset_root": dataset_root,
        "stats": {},
        "warnings": [],
        "errors": []
    }

    # 读取变量列表
    var_names_path = os.path.join(dataset_root, "var_names.json")
    dyn_vars: Optional[List[str]] = None
    if os.path.exists(var_names_path):
        try:
            with open(var_names_path, "r", encoding="utf-8") as f:
                var_names_data = json.load(f)
            dyn_vars = var_names_data.get("dynamic", [])
        except Exception as e:
            result["warnings"].append(f"读取 var_names.json 失败: {e}")
    else:
        result["warnings"].append(f"未找到 var_names.json，将自动扫描目录: {var_names_path}")

    for split in splits:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.isdir(split_dir):
            result["warnings"].append(f"split 目录不存在: {split_dir}")
            continue

        # 确定变量目录列表
        if dyn_vars:
            var_list = [
                v for v in dyn_vars
                if os.path.isdir(os.path.join(split_dir, v))
            ]
        else:
            try:
                var_list = [
                    d for d in os.listdir(split_dir)
                    if os.path.isdir(os.path.join(split_dir, d))
                ]
            except Exception as e:
                result["errors"].append(f"扫描 {split_dir} 失败: {e}")
                continue

        if not var_list:
            result["warnings"].append(f"split {split!r} 下未找到任何变量目录")
            continue

        split_stats: Dict[str, Any] = {}
        for var_name in var_list:
            var_dir = os.path.join(split_dir, var_name)
            npy_files = sorted(glob.glob(os.path.join(var_dir, "*.npy")))

            if not npy_files:
                result["warnings"].append(f"{split}/{var_name}: 目录为空，跳过")
                continue

            var_stats = _compute_var_stats(npy_files, max_files=max_files)
            split_stats[var_name] = var_stats

            # NaN 率高于 0.3 生成警告
            if var_stats["nan_rate"] > 0.3:
                pct = round(var_stats["nan_rate"] * 100, 1)
                result["warnings"].append(
                    f"{var_name}/{split} NaN 率 {pct}% 偏高"
                )

        result["stats"][split] = split_stats

    # ---- 时间边界检查 ----
    time_boundary = _check_time_boundaries(dataset_root)
    result["time_boundary_check"] = time_boundary
    if not time_boundary["passed"]:
        for err in time_boundary["errors"]:
            result["warnings"].append(f"[时间边界] {err}")

    # ---- 跨 split 分布一致性检查 ----
    cross_split = _check_cross_split_distribution(result["stats"])
    result["cross_split_check"] = cross_split
    result["warnings"].extend(cross_split["warnings"])

    # 如有错误，将整体状态改为 error
    if result["errors"]:
        result["status"] = "error"

    return result


# ========================================
# CLI 入口
# ========================================

def main():
    parser = argparse.ArgumentParser(description="预报数据统计工具")
    parser.add_argument("--config", required=True, help="JSON 配置文件路径")
    parser.add_argument("--output", required=True, help="结果输出路径")
    args = parser.parse_args()

    # 读取配置
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        error_result = {
            "status": "error",
            "errors": [f"读取配置文件失败: {e}"],
            "warnings": [],
            "stats": {}
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(error_result, f, ensure_ascii=False, indent=2)
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)

    dataset_root: str = config.get("dataset_root", "")
    splits: List[str] = config.get("splits", ["train", "valid", "test"])
    max_files: int = int(config.get("max_files", 200))

    if not dataset_root:
        error_result = {
            "status": "error",
            "errors": ["配置文件缺少 dataset_root 字段"],
            "warnings": [],
            "stats": {}
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(error_result, f, ensure_ascii=False, indent=2)
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)

    result = run_stats(dataset_root, splits, max_files)

    # 先写入数据集目录（若失败，警告会被纳入 output 文件）
    dest = os.path.join(dataset_root, "data_stats.json")
    try:
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ 统计结果已写入: {dest}", file=sys.stderr)
    except Exception as e:
        result["warnings"].append(f"写入 data_stats.json 失败: {e}")

    # 再写入 output 文件（包含上面可能追加的警告）
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
