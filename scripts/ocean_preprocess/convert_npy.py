#!/usr/bin/env python3
"""
convert_npy.py - Step C: NC 转 NPY 转换（CLI 入口）

@author leizheng
@contributors kongzhiquan
@date 2026-02-02
@version 3.1.1

功能:
- 将 NC 文件中的变量转换为 NPY 格式
- 按时间顺序划分数据集（train/valid/test）
- 每个时间步单独保存为一个文件（000000.npy, 000001.npy, ...）
- 输出目录结构: train/hr/uo/, train/hr/vo/, valid/hr/uo/, ..., static_variables/
- 执行后置验证 (Rule 1/2/3)

用法:
    python convert_npy.py --config config.json --output result.json

配置文件格式:
{
    "nc_folder": "/path/to/nc/files",
    "output_base": "/path/to/output",
    "dyn_vars": ["uo", "vo"],
    "static_file": "/path/to/static.nc",
    "stat_vars": ["lon_rho", "lat_rho", "h", "mask_rho"],
    "mask_vars": ["mask_rho", "mask_u", "mask_v"],
    "train_ratio": 0.7,
    "valid_ratio": 0.15,
    "test_ratio": 0.15,
    "h_slice": "0:680",
    "w_slice": "0:1440",
    "scale": 4,
    "workers": 32,
    "output_subdir": "hr",
    ...
}

Changelog:
    - 2026-02-05 kongzhiquan v3.1.1: 模块化重构
        - 将核心逻辑拆分到 convert_lib/ 子包
        - 本文件仅保留 CLI 入口
    - 2026-02-05 kongzhiquan v3.1.1: 修复 numpy int64 JSON 序列化错误
    - 2026-02-04 kongzhiquan v2.9.2: 修复 Rule1 对 1D 坐标数组的验证逻辑
    - 2026-02-04 kongzhiquan v2.9.1: 修复 Rule1 验证逻辑
    - 2026-02-04 leizheng v2.9.0: 逐时间步保存文件
    - 2026-02-04 leizheng v2.7.0: 修复 1D 坐标裁剪问题
    - 2026-02-04 leizheng v2.6.0: 文件级并行处理
    - 2026-02-04 leizheng v2.5.0: 支持从动态文件提取静态变量
    - 2026-02-04 leizheng v2.4.0: 支持粗网格模式
    - 2026-02-03 leizheng v2.3.0: 裁剪与多线程
    - 2026-02-03 leizheng v2.2.0: 数据集划分功能
    - 2026-02-02 leizheng v2.0.0: 初始版本
"""

import argparse
import json
import sys

# 导入核心函数和编码器
from convert_lib import convert_npy, NumpyEncoder


def main():
    parser = argparse.ArgumentParser(description="Step C: NC 转 NPY 转换")
    parser.add_argument("--config", required=True, help="JSON 配置文件路径")
    parser.add_argument("--output", required=True, help="结果输出 JSON 路径")
    args = parser.parse_args()

    # 读取配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 执行转换
    result = convert_npy(config)

    # 写入结果（使用 NumpyEncoder 处理 numpy 类型）
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    # 同时输出到 stdout
    print(json.dumps(result, ensure_ascii=False, cls=NumpyEncoder))


if __name__ == "__main__":
    main()
