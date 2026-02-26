#!/usr/bin/env python3
"""
validate_tensor.py - Step B: 张量约定验证

@author leizheng
@contributors kongzhiquan
@date 2026-02-02
@version 1.0.0

功能:
- 验证变量的张量形状是否符合约定
- 动态变量: [D, H, W] 或 [T, D, H, W]
- 静态变量: [H, W]
- 掩码变量: [H, W] 且必须是 2D

用法:
    python validate_tensor.py --config config.json --output result.json

配置文件格式:
{
    "inspect_result_path": "/path/to/inspect_result.json",
    "research_vars": ["uo", "vo"],
    "mask_vars": ["mask_rho", "mask_u", "mask_v"]
}
"""

import argparse
import json
import sys
from typing import Any, Dict, List

__all__ = [
    'validate_tensor_convention',
    'DEFAULT_MASK_VARS',
]

# ========================================
# 常量定义
# ========================================

DEFAULT_MASK_VARS = ['mask_u', 'mask_rho', 'mask_v', 'mask_psi']


# ========================================
# 验证函数
# ========================================

def validate_tensor_convention(
    variables_info: Dict[str, Any],
    research_vars: List[str],
    mask_vars: List[str]
) -> Dict[str, Any]:
    """
    验证张量约定

    Args:
        variables_info: 变量信息字典（来自 inspect 结果）
        research_vars: 研究变量列表
        mask_vars: 掩码变量列表

    Returns:
        验证结果字典
    """
    result = {
        "status": "pending",
        "research_vars": research_vars,
        "tensor_convention": {},
        "var_names_config": {
            "dynamic": [],
            "static": [],
            "research": research_vars,
            "mask": mask_vars
        },
        "warnings": [],
        "errors": [],
        "message": ""
    }

    # 1. 验证研究变量存在
    for var in research_vars:
        if var not in variables_info:
            result["errors"].append(f"研究变量 '{var}' 不存在于数据中")

    if result["errors"]:
        result["status"] = "error"
        result["message"] = "研究变量验证失败"
        return result

    # 2. 张量形状检查
    for var_name, var_info in variables_info.items():
        shape = tuple(var_info.get("shape", []))
        category = var_info.get("category", "unknown")
        ndim = len(shape)

        convention = {
            "name": var_name,
            "original_shape": list(shape),
            "category": category,
            "valid": False,
            "interpretation": "",
            "ndim": ndim
        }

        if category == "dynamic":
            result["var_names_config"]["dynamic"].append(var_name)

            if ndim == 3:
                convention["valid"] = True
                convention["interpretation"] = "[D, H, W]"
                convention["D"] = shape[0]
                convention["H"] = shape[1]
                convention["W"] = shape[2]
            elif ndim == 4:
                convention["valid"] = True
                convention["interpretation"] = "[T, D, H, W]"
                convention["T"] = shape[0]
                convention["D"] = shape[1]
                convention["H"] = shape[2]
                convention["W"] = shape[3]
            else:
                convention["interpretation"] = f"不符合约定: {ndim}D，预期 3D 或 4D"
                result["warnings"].append(
                    f"动态变量 '{var_name}' 维度不符合约定: shape={shape}"
                )

        elif category in ["static", "mask"]:
            result["var_names_config"]["static"].append(var_name)

            if ndim == 2:
                convention["valid"] = True
                convention["interpretation"] = "[H, W]"
                convention["H"] = shape[0]
                convention["W"] = shape[1]
            elif ndim == 1:
                # 标量或 1D 数据（如 depthmax）
                convention["valid"] = True
                convention["interpretation"] = "[N]"
                convention["N"] = shape[0]
            else:
                convention["interpretation"] = f"不符合约定: {ndim}D，预期 2D"

                # 掩码变量必须是 2D
                if var_name in mask_vars:
                    result["errors"].append(
                        f"掩码变量 '{var_name}' 维度错误: shape={shape}，必须是 2D"
                    )
                else:
                    result["warnings"].append(
                        f"静态变量 '{var_name}' 维度不符合约定: shape={shape}"
                    )

        result["tensor_convention"][var_name] = convention

    # 3. 设置最终状态
    if result["errors"]:
        result["status"] = "error"
        result["message"] = "张量约定验证失败"
    else:
        result["status"] = "pass"
        result["message"] = "张量约定验证通过"

    return result


# ========================================
# 主函数
# ========================================

def main():
    parser = argparse.ArgumentParser(description="Step B: 张量约定验证")
    parser.add_argument("--config", required=True, help="JSON 配置文件路径")
    parser.add_argument("--output", required=True, help="结果输出 JSON 路径")
    args = parser.parse_args()

    # 读取配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    inspect_result_path = config.get("inspect_result_path", "")
    research_vars = config.get("research_vars", [])
    mask_vars = config.get("mask_vars", DEFAULT_MASK_VARS)

    # 读取 inspect 结果
    try:
        with open(inspect_result_path, 'r', encoding='utf-8') as f:
            inspect_result = json.load(f)
    except Exception as e:
        result = {
            "status": "error",
            "errors": [f"读取 inspect 结果失败: {str(e)}"],
            "message": "验证失败"
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)

    variables_info = inspect_result.get("variables", {})

    # 执行验证
    result = validate_tensor_convention(variables_info, research_vars, mask_vars)

    # 写入结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 同时输出到 stdout
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
