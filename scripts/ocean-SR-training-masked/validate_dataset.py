"""
validate_dataset.py - 验证预处理数据目录并自动检测训练参数

扫描 ocean-preprocess 输出目录结构，检测：
- 动态变量列表 (dyn_vars)
- 超分辨率倍数 (scale)
- HR/LR 数据形状
- 数据集各 split 的样本数量

用法:
    python validate_dataset.py --dataset_root /path/to/preprocessed_data

输出 JSON 格式结果。
"""

import argparse
import json
import os
import sys

import numpy as np


def validate_dataset(dataset_root):
    """验证预处理数据目录并自动检测参数"""
    result = {
        "status": "success",
        "dataset_root": os.path.abspath(dataset_root),
        "dyn_vars": [],
        "scale": None,
        "hr_shape": None,
        "lr_shape": None,
        "splits": {},
        "has_static": False,
        "static_vars": [],
        "warnings": [],
        "errors": [],
    }

    # 1. 检查根目录存在性
    if not os.path.isdir(dataset_root):
        result["status"] = "error"
        result["errors"].append(f"数据目录不存在: {dataset_root}")
        return result

    # 2. 检查 split 目录结构 (train/valid/test)
    expected_splits = ["train", "valid", "test"]
    found_splits = []
    for split in expected_splits:
        split_dir = os.path.join(dataset_root, split)
        if os.path.isdir(split_dir):
            found_splits.append(split)
        else:
            result["warnings"].append(f"缺少 {split} 目录")

    if not found_splits:
        result["status"] = "error"
        result["errors"].append(
            f"数据目录中未找到 train/valid/test 子目录: {dataset_root}"
        )
        return result

    # 3. 检测动态变量 (从 train/hr/ 目录)
    primary_split = "train" if "train" in found_splits else found_splits[0]
    hr_dir = os.path.join(dataset_root, primary_split, "hr")

    if not os.path.isdir(hr_dir):
        result["status"] = "error"
        result["errors"].append(f"缺少 HR 数据目录: {hr_dir}")
        return result

    # hr/ 下的子目录就是变量名
    dyn_vars = sorted(
        [
            d
            for d in os.listdir(hr_dir)
            if os.path.isdir(os.path.join(hr_dir, d)) and not d.startswith(".")
        ]
    )

    if not dyn_vars:
        # 可能是没有变量子目录的旧格式（直接 hr/*.npy）
        npy_files = [f for f in os.listdir(hr_dir) if f.endswith(".npy")]
        if npy_files:
            dyn_vars = ["_flat"]  # 标记为扁平结构
            result["warnings"].append(
                "HR 目录下没有变量子目录，数据可能是单变量扁平结构"
            )
        else:
            result["status"] = "error"
            result["errors"].append(f"HR 目录中未找到任何变量子目录或 .npy 文件: {hr_dir}")
            return result

    result["dyn_vars"] = dyn_vars

    # 4. 检测 HR shape
    if dyn_vars[0] == "_flat":
        sample_dir = hr_dir
    else:
        sample_dir = os.path.join(hr_dir, dyn_vars[0])

    npy_files = sorted([f for f in os.listdir(sample_dir) if f.endswith(".npy")])
    if npy_files:
        sample_path = os.path.join(sample_dir, npy_files[0])
        sample = np.load(sample_path)
        result["hr_shape"] = list(sample.shape)
    else:
        result["status"] = "error"
        result["errors"].append(f"变量目录中无 .npy 文件: {sample_dir}")
        return result

    # 5. 检测 LR shape 和推算 scale
    lr_dir = os.path.join(dataset_root, primary_split, "lr")
    if os.path.isdir(lr_dir):
        if dyn_vars[0] == "_flat":
            lr_sample_dir = lr_dir
        else:
            lr_sample_dir = os.path.join(lr_dir, dyn_vars[0])

        if os.path.isdir(lr_sample_dir):
            lr_npy_files = sorted(
                [f for f in os.listdir(lr_sample_dir) if f.endswith(".npy")]
            )
            if lr_npy_files:
                lr_sample = np.load(os.path.join(lr_sample_dir, lr_npy_files[0]))
                result["lr_shape"] = list(lr_sample.shape)

                # 推算 scale
                hr_h, hr_w = result["hr_shape"][-2], result["hr_shape"][-1]
                lr_h, lr_w = result["lr_shape"][-2], result["lr_shape"][-1]

                if lr_h > 0 and lr_w > 0:
                    scale_h = hr_h / lr_h
                    scale_w = hr_w / lr_w

                    if abs(scale_h - scale_w) < 0.01 and scale_h == int(scale_h):
                        result["scale"] = int(scale_h)
                    else:
                        result["warnings"].append(
                            f"HR/LR 尺寸比不一致或非整数: H={scale_h:.2f}, W={scale_w:.2f}"
                        )
                        # 取较大值向下取整
                        result["scale"] = int(max(scale_h, scale_w))
        else:
            result["warnings"].append(f"LR 目录中缺少变量子目录: {lr_sample_dir}")
    else:
        result["warnings"].append("未找到 LR 数据目录，scale 无法自动推算")

    # 6. 统计各 split 的样本数
    for split in found_splits:
        split_info = {"hr_count": 0, "lr_count": 0}

        # 统计 HR
        hr_split_dir = os.path.join(dataset_root, split, "hr")
        if os.path.isdir(hr_split_dir):
            if dyn_vars[0] == "_flat":
                var_dir = hr_split_dir
            else:
                var_dir = os.path.join(hr_split_dir, dyn_vars[0])

            if os.path.isdir(var_dir):
                split_info["hr_count"] = len(
                    [f for f in os.listdir(var_dir) if f.endswith(".npy")]
                )

        # 统计 LR
        lr_split_dir = os.path.join(dataset_root, split, "lr")
        if os.path.isdir(lr_split_dir):
            if dyn_vars[0] == "_flat":
                var_dir = lr_split_dir
            else:
                var_dir = os.path.join(lr_split_dir, dyn_vars[0])

            if os.path.isdir(var_dir):
                split_info["lr_count"] = len(
                    [f for f in os.listdir(var_dir) if f.endswith(".npy")]
                )

        # 检查 HR/LR 数量是否匹配
        if split_info["hr_count"] > 0 and split_info["lr_count"] > 0:
            if split_info["hr_count"] != split_info["lr_count"]:
                result["warnings"].append(
                    f"{split} 集 HR({split_info['hr_count']}) 和 LR({split_info['lr_count']}) 样本数量不匹配"
                )

        result["splits"][split] = split_info

    # 7. 检测静态变量
    static_dir = os.path.join(dataset_root, "static_variables")
    if os.path.isdir(static_dir):
        result["has_static"] = True
        result["static_vars"] = sorted(
            [
                os.path.splitext(f)[0]
                for f in os.listdir(static_dir)
                if f.endswith(".npy")
            ]
        )

    # 8. 总样本统计
    total_hr = sum(s["hr_count"] for s in result["splits"].values())
    total_lr = sum(s["lr_count"] for s in result["splits"].values())
    result["total_samples"] = {"hr": total_hr, "lr": total_lr}

    # 9. 最终检查
    if total_hr == 0:
        result["status"] = "error"
        result["errors"].append("未找到任何 HR 样本数据")
    elif total_lr == 0:
        result["warnings"].append("未找到 LR 数据，可能需要先进行下采样")

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate preprocessed dataset")
    parser.add_argument(
        "--dataset_root", type=str, required=True, help="Preprocessed dataset root directory"
    )
    args = parser.parse_args()

    info = validate_dataset(args.dataset_root)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
