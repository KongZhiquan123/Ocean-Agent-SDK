"""
@file validate_dataset.py
@description Validate forecast-preprocessed data directory and detect dataset info.
             Checks var_names.json, time_index.json, NPY file shapes and time continuity.
@author Leizheng
@date 2026-02-26
@version 1.0.0

@changelog
  - 2026-02-26 Leizheng: v1.0.0 initial version for ocean forecast training

Usage:
    python validate_dataset.py --dataset_root /path/to/preprocessed_data

Output: JSON to stdout
"""

import argparse
import json
import os
import sys

import numpy as np


def validate_dataset(dataset_root):
    """Validate forecast-preprocessed data directory."""
    result = {
        "status": "ok",
        "dataset_root": os.path.abspath(dataset_root),
        "dyn_vars": [],
        "spatial_shape": None,
        "splits": {},
        "total_timesteps": 0,
        "time_range": None,
        "has_static": False,
        "static_vars": [],
        "warnings": [],
        "errors": [],
    }

    # 1. Check root directory exists
    if not os.path.isdir(dataset_root):
        result["status"] = "error"
        result["errors"].append(f"数据目录不存在: {dataset_root}")
        return result

    # 2. Check var_names.json
    var_names_path = os.path.join(dataset_root, "var_names.json")
    if not os.path.isfile(var_names_path):
        result["status"] = "error"
        result["errors"].append("缺少 var_names.json 文件")
        return result

    try:
        with open(var_names_path, "r") as f:
            var_names = json.load(f)
    except Exception as e:
        result["status"] = "error"
        result["errors"].append(f"var_names.json 解析失败: {e}")
        return result

    # Support both key formats: forecast-preprocess uses "dynamic"/"static"/"mask",
    # while some legacy configs may use "dyn_vars"/"stat_vars"/"mask_vars"
    dyn_vars = var_names.get("dynamic", var_names.get("dyn_vars", []))
    stat_vars = var_names.get("static", var_names.get("stat_vars", []))
    mask_vars = var_names.get("mask", var_names.get("mask_vars", []))
    spatial_shape = var_names.get("spatial_shape", None)

    if not dyn_vars:
        result["status"] = "error"
        result["errors"].append("var_names.json 中 dyn_vars 为空")
        return result

    result["dyn_vars"] = dyn_vars
    result["spatial_shape"] = spatial_shape

    # 3. Check time_index.json
    time_index_path = os.path.join(dataset_root, "time_index.json")
    time_index = None
    if os.path.isfile(time_index_path):
        try:
            with open(time_index_path, "r") as f:
                time_index = json.load(f)
        except Exception as e:
            result["warnings"].append(f"time_index.json 解析失败: {e}，将从文件名推断时间顺序")
    else:
        result["warnings"].append("缺少 time_index.json，将从文件名推断时间顺序")

    # 4. Check split directories
    expected_splits = ["train", "valid", "test"]
    total_timesteps = 0

    all_dates = []

    for split in expected_splits:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.isdir(split_dir):
            result["warnings"].append(f"缺少 {split}/ 目录")
            result["splits"][split] = 0
            continue

        # Get date list from time_index or from file names
        if time_index and split in time_index:
            dates = time_index[split]
        else:
            # Infer from first variable's directory
            first_var = dyn_vars[0]
            var_dir = os.path.join(split_dir, first_var)
            if os.path.isdir(var_dir):
                files = sorted([
                    os.path.splitext(f)[0]
                    for f in os.listdir(var_dir)
                    if f.endswith(".npy")
                ])
                dates = files
            else:
                dates = []

        result["splits"][split] = len(dates)
        total_timesteps += len(dates)
        all_dates.extend(dates)

        # Validate per-variable file counts and shapes
        if dates:
            for var in dyn_vars:
                var_dir = os.path.join(split_dir, var)
                if not os.path.isdir(var_dir):
                    result["errors"].append(f"{split}/{var}/ 目录不存在")
                    continue

                existing_files = set(
                    os.path.splitext(f)[0]
                    for f in os.listdir(var_dir)
                    if f.endswith(".npy")
                )

                missing = [d for d in dates if d not in existing_files]
                if missing:
                    n_missing = len(missing)
                    result["warnings"].append(
                        f"{split}/{var}/: 缺少 {n_missing} 个时间步文件 "
                        f"(如: {missing[0]})"
                    )

            # Check first file shape
            first_var = dyn_vars[0]
            first_file = os.path.join(split_dir, first_var, f"{dates[0]}.npy")
            if os.path.isfile(first_file):
                try:
                    arr = np.load(first_file)
                    detected_shape = list(arr.shape)
                    if spatial_shape is None:
                        result["spatial_shape"] = detected_shape
                    elif list(spatial_shape) != detected_shape:
                        result["warnings"].append(
                            f"{split}/{first_var}/{dates[0]}.npy 形状 {detected_shape} "
                            f"与 var_names.json 中的 spatial_shape {spatial_shape} 不匹配"
                        )
                except Exception as e:
                    result["warnings"].append(f"读取 {first_file} 失败: {e}")

    result["total_timesteps"] = total_timesteps

    # Time range
    if all_dates:
        sorted_dates = sorted(all_dates)
        result["time_range"] = {
            "start": sorted_dates[0],
            "end": sorted_dates[-1],
        }

    # Check time continuity per split
    if time_index:
        for split in expected_splits:
            if split in time_index:
                dates = time_index[split]
                if len(dates) > 1:
                    sorted_d = sorted(dates)
                    if sorted_d != dates:
                        result["warnings"].append(
                            f"{split} 的时间索引未按升序排列"
                        )

    # 5. Check static directory
    static_dir = os.path.join(dataset_root, "static")
    if os.path.isdir(static_dir):
        static_files = [f for f in os.listdir(static_dir) if f.endswith(".npy")]
        if static_files:
            result["has_static"] = True
            result["static_vars"] = [os.path.splitext(f)[0] for f in static_files]
    elif stat_vars:
        result["static_vars"] = stat_vars

    # 6. Warn on small datasets
    for split in expected_splits:
        n = result["splits"].get(split, 0)
        if n > 0 and n < 10:
            result["warnings"].append(
                f"{split} 仅有 {n} 个时间步，可能不足以训练"
            )

    # 7. Warn on UNet2d spatial dimension constraints
    if result["spatial_shape"]:
        h, w = result["spatial_shape"][:2]
        if h % 16 != 0 or w % 16 != 0:
            result["warnings"].append(
                f"空间尺寸 {h}×{w} 不能被 16 整除，UNet2d 等模型可能需要 padding 或裁剪"
            )

    # Final status
    if result["errors"]:
        result["status"] = "error"

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate forecast dataset")
    parser.add_argument("--dataset_root", required=True, help="Preprocessed data root directory")
    args = parser.parse_args()

    result = validate_dataset(args.dataset_root)
    print(json.dumps(result, ensure_ascii=False))

    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
